"""
Simplified LLM review engine for demo.
Evaluates predictions and suggests corrections using Ollama.

Features:
- Prompt caching to avoid redundant LLM calls
- Parallel async execution with semaphore-based concurrency control
- Retry with exponential backoff for robustness
"""

import asyncio
import hashlib
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import aiohttp
import yaml  # type: ignore

from core.logging_config import get_logger

logger = get_logger(__name__)


class ReviewEngine:
    """LLM-powered review engine with caching, parallelism, and retry logic."""

    # Prompt version - increment when prompt format changes to invalidate cache
    PROMPT_VERSION = "v1.0"

    # Default configuration
    DEFAULT_MAX_CONCURRENT = 3  # Ollama handles limited concurrent requests well
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_BASE_DELAY = 1.0  # Base delay for exponential backoff (seconds)
    DEFAULT_TIMEOUT = 60  # Request timeout (seconds)

    def __init__(
        self,
        model_name: str = "mistral",
        ollama_url: str = "http://localhost:11434",
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        enable_cache: bool = True,
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.enable_cache = enable_cache
        self.labels = self._load_labels()

        # Prompt cache: hash -> response
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _load_labels(self) -> Dict[str, Any]:
        """Load label definitions from config."""
        config_path = Path(__file__).parent.parent / "configs" / "labels.yaml"
        if config_path.exists():
            with open(config_path) as f:
                data: Any = yaml.safe_load(f)
                return data if data else {"labels": []}
        return {"labels": []}

    def _build_prompt(self, text: str, pred_label: str, confidence: float) -> str:
        """Build the unified review prompt."""
        labels_desc = "\n".join([
            f"- {label['name']}: {label['definition']}"
            for label in self.labels.get("labels", [])
        ])

        return f"""You are a semantic auditor reviewing text classification predictions.

## Available Labels
{labels_desc}

## Task
Analyze whether the predicted label correctly captures the semantic intent of the text.

## Input
Text: "{text}"
Predicted Label: {pred_label}
Confidence: {confidence}

## Instructions
Respond in this exact format:
VERDICT: [Correct/Incorrect/Uncertain]
REASONING: [One sentence explaining why]
SUGGESTED_LABEL: [Only if Incorrect, otherwise "None"]
EXPLANATION: [Brief stakeholder-friendly explanation]

Be precise and concise."""

    def _get_prompt_hash(self, prompt: str) -> str:
        """Generate a cache key for a prompt, including version for invalidation."""
        content = f"{self.PROMPT_VERSION}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total = self._cache_hits + self._cache_misses
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": (self._cache_hits / total * 100) if total > 0 else 0,
            "cached_prompts": len(self._cache),
        }

    def clear_cache(self):
        """Clear the prompt cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    async def review_sample_async(
        self,
        text: str,
        pred_label: str,
        confidence: float,
        sample_id: str,
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> Dict[str, Any]:
        """Review a single sample asynchronously with caching and retry."""
        prompt = self._build_prompt(text, pred_label, confidence)
        prompt_hash = self._get_prompt_hash(prompt)

        # Check cache first
        if self.enable_cache and prompt_hash in self._cache:
            self._cache_hits += 1
            logger.debug(f"Cache hit for sample {sample_id}")
            cached = self._cache[prompt_hash].copy()
            cached.update({
                "sample_id": sample_id,
                "text": text,
                "pred_label": pred_label,
                "confidence": confidence,
                "cache_hit": True,
            })
            return cached

        self._cache_misses += 1
        start_time = time.time()

        # Use semaphore for concurrency control if provided
        async def _do_review():
            return await self._call_ollama_with_retry(prompt)

        try:
            if semaphore:
                async with semaphore:
                    response = await _do_review()
            else:
                response = await _do_review()

            latency_ms = int((time.time() - start_time) * 1000)
            logger.debug(f"Sample {sample_id} reviewed in {latency_ms}ms")

            # Parse response
            result = self._parse_response(response)
            result.update({
                "sample_id": sample_id,
                "text": text,
                "pred_label": pred_label,
                "confidence": confidence,
                "success": True,
                "prompt_hash": prompt_hash,
                "latency_ms": latency_ms,
                "cache_hit": False,
            })

            # Cache the parsed result (without sample-specific fields)
            if self.enable_cache:
                cache_entry = {
                    "verdict": result["verdict"],
                    "reasoning": result["reasoning"],
                    "suggested_label": result["suggested_label"],
                    "explanation": result["explanation"],
                    "success": True,
                    "prompt_hash": prompt_hash,
                    "latency_ms": latency_ms,
                }
                self._cache[prompt_hash] = cache_entry

            return result

        except Exception as e:
            logger.error(f"Failed to review sample {sample_id}: {e}")
            return {
                "sample_id": sample_id,
                "text": text,
                "pred_label": pred_label,
                "confidence": confidence,
                "verdict": "Uncertain",
                "reasoning": f"Error: {str(e)}",
                "suggested_label": None,
                "explanation": "Review failed",
                "success": False,
                "error": str(e),
                "cache_hit": False,
            }

    async def _call_ollama_with_retry(self, prompt: str) -> str:
        """Call Ollama API with exponential backoff retry."""
        last_exception: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                return await self._call_ollama(prompt)
            except asyncio.TimeoutError as e:
                last_exception = e
                logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries}")
            except aiohttp.ClientError as e:
                last_exception = e
                logger.warning(f"Network error on attempt {attempt + 1}/{self.max_retries}: {e}")
            except Exception as e:
                # For other errors, check if it's a server overload (5xx)
                if "5" in str(e)[:1]:  # 5xx errors
                    last_exception = e
                    logger.warning(f"Server error on attempt {attempt + 1}/{self.max_retries}: {e}")
                else:
                    raise  # Don't retry client errors (4xx)

            if attempt < self.max_retries - 1:
                # Exponential backoff with jitter
                delay = self.DEFAULT_BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
                logger.debug(f"Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)

        logger.error("Max retries exceeded for Ollama call")
        raise last_exception if last_exception else Exception("Max retries exceeded")

    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API."""
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 300}
        }

        timeout = aiohttp.ClientTimeout(total=self.DEFAULT_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    raise Exception(f"Ollama error: {resp.status}")
                result = await resp.json()
                response_text: str = result.get("response", "")
                return response_text

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format with regex fallback."""
        result = {
            "verdict": "Uncertain",
            "reasoning": "",
            "suggested_label": None,
            "explanation": "",
        }

        # Try regex-based parsing first (more robust)
        verdict_match = re.search(r'VERDICT:\s*(\w+)', response, re.IGNORECASE)
        if verdict_match:
            verdict = verdict_match.group(1).capitalize()
            if verdict in ["Correct", "Incorrect", "Uncertain"]:
                result["verdict"] = verdict
                logger.debug(f"Parsed verdict via regex: {verdict}")

        reasoning_match = re.search(r'REASONING:\s*(.+?)(?=\n[A-Z_]+:|$)', response, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip().split('\n')[0].strip()

        suggested_match = re.search(r'SUGGESTED_LABEL:\s*(.+?)(?=\n[A-Z_]+:|$)', response, re.IGNORECASE | re.DOTALL)
        if suggested_match:
            suggested = suggested_match.group(1).strip().split('\n')[0].strip()
            if suggested.lower() != "none":
                result["suggested_label"] = suggested

        explanation_match = re.search(r'EXPLANATION:\s*(.+?)(?=\n[A-Z_]+:|$)', response, re.IGNORECASE | re.DOTALL)
        if explanation_match:
            result["explanation"] = explanation_match.group(1).strip().split('\n')[0].strip()

        # Fallback to line-by-line parsing if regex didn't find verdict
        if result["verdict"] == "Uncertain" and "VERDICT:" in response.upper():
            lines = response.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line.upper().startswith("VERDICT:"):
                    verdict = line.split(":", 1)[1].strip()
                    if verdict in ["Correct", "Incorrect", "Uncertain"]:
                        result["verdict"] = verdict
                        logger.debug(f"Parsed verdict via line-by-line: {verdict}")
                elif line.upper().startswith("REASONING:") and not result["reasoning"]:
                    result["reasoning"] = line.split(":", 1)[1].strip()
                elif line.upper().startswith("SUGGESTED_LABEL:") and result["suggested_label"] is None:
                    suggested = line.split(":", 1)[1].strip()
                    if suggested.lower() != "none":
                        result["suggested_label"] = suggested
                elif line.upper().startswith("EXPLANATION:") and not result["explanation"]:
                    result["explanation"] = line.split(":", 1)[1].strip()

        return result

    async def review_batch_async(
        self,
        samples: List[Dict[str, Any]],
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """Review multiple samples with parallel execution.

        Uses asyncio.gather() with a semaphore to limit concurrent requests
        while maximizing throughput.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        completed = 0
        total = len(samples)
        results_lock = asyncio.Lock()

        async def review_with_progress(sample: Dict) -> Dict:
            nonlocal completed
            result = await self.review_sample_async(
                text=sample["text"],
                pred_label=sample["pred_label"],
                confidence=sample["confidence"],
                sample_id=sample["id"],
                semaphore=semaphore,
            )
            result["ground_truth"] = sample.get("ground_truth")

            # Thread-safe progress update
            async with results_lock:
                completed += 1
                if on_progress:
                    on_progress(completed, total)

            return result

        # Create all tasks and run them concurrently
        tasks = [review_with_progress(sample) for sample in samples]
        results = await asyncio.gather(*tasks)

        # Preserve original order
        return list(results)

    def generate_mock_results(self, samples: List[Dict]) -> List[Dict]:
        """Generate mock results without LLM (for --no-llm mode)."""
        results = []

        for sample in samples:
            is_correct = not sample.get("is_misclassified", False)

            if is_correct:
                verdict = "Correct"
                suggested = None
                reasoning = "The predicted label accurately captures the semantic intent."
            else:
                verdict = "Incorrect"
                suggested = sample.get("ground_truth", "Unknown")
                reasoning = f"The text indicates {suggested}, not {sample['pred_label']}."

            results.append({
                "sample_id": sample["id"],
                "text": sample["text"],
                "pred_label": sample["pred_label"],
                "confidence": sample["confidence"],
                "ground_truth": sample.get("ground_truth"),
                "verdict": verdict,
                "reasoning": reasoning,
                "suggested_label": suggested,
                "explanation": f"Classification as {suggested or sample['pred_label']}.",
                "success": True,
            })

        return results

