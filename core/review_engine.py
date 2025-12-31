"""
Simplified LLM review engine for demo.
Evaluates predictions and suggests corrections using Ollama.

Features:
- Prompt caching to avoid redundant LLM calls
- Parallel async execution with semaphore-based concurrency control
- Retry with exponential backoff for robustness
- Output integrity validation (schema + label guardrails)
"""

import asyncio
import hashlib
import json
import random
import re
import sqlite3
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import aiohttp
import yaml  # type: ignore

from core.config_loader import get_config
from core.logging_config import get_logger

logger = get_logger(__name__)


class Verdict(str, Enum):
    """Valid verdict values - enforces strict output schema."""
    CORRECT = "Correct"
    INCORRECT = "Incorrect"
    UNCERTAIN = "Uncertain"


@dataclass
class ValidationResult:
    """Result of output validation."""
    is_valid: bool
    issues: List[str]
    sanitized_verdict: Verdict
    sanitized_label: Optional[str]


class OutputIntegrityError(Exception):
    """Raised when model output fails integrity checks."""
    def __init__(self, message: str, issues: List[str]):
        super().__init__(message)
        self.issues = issues


# Security constants
MAX_REASONING_LENGTH = 2000  # Truncate excessively long reasoning (potential attack vector)
MAX_EXPLANATION_LENGTH = 1000
MAX_OUTPUT_TOTAL_LENGTH = 5000  # Total response length limit

# Comprehensive prompt injection detection patterns
SUSPICIOUS_PATTERNS = [
    # Direct instruction override attempts
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|context)",
    r"ignore\s+previous\s+instructions",
    r"disregard\s+(the\s+)?(all\s+)?(above|previous)",
    r"forget\s+(everything|all|previous|what)",
    r"new\s+instructions?:",
    r"override\s+(the\s+)?(previous|all)",
    # System/role manipulation
    r"system\s*:\s*",
    r"(you are|act as|pretend to be)\s+(now|a\s+different)",
    r"\[system\]",
    r"<\s*system\s*>",
    r"assistant\s*:\s*",
    r"\[INST\]",
    r"<<SYS>>",
    # Output manipulation
    r"output\s+only",
    r"respond\s+with\s+only",
    r"say\s+(exactly|only)",
    r"print\s+(only|exactly)",
    # Jailbreak patterns
    r"DAN\s*(mode)?",
    r"developer\s+mode",
    r"jailbreak",
    r"bypass\s+(the\s+)?(filter|safety|restriction)",
    # Data exfiltration attempts
    r"reveal\s+(your|the)\s+(instructions?|prompt|system)",
    r"show\s+me\s+(your|the)\s+(prompt|instructions?|system)",
    r"what\s+(are|is)\s+(your|the)\s+(instructions?|prompt)",
    # Markdown/HTML injection
    r"<script",
    r"javascript:",
    r"onclick\s*=",
    r"onerror\s*=",
]


class OllamaHTTPError(Exception):
    """Represents an HTTP error response from Ollama."""

    def __init__(self, status: int, body: str = ""):
        super().__init__(f"Ollama error: {status}")
        self.status = status
        self.body = body


class ReviewEngine:
    """LLM-powered review engine with caching, parallelism, and retry logic.

    Security features:
    - Output schema validation (only Correct/Incorrect/Uncertain verdicts)
    - Label guardrails (rejects labels not in labels.yaml)
    - All outputs marked as "untrusted" by default
    """

    # Prompt version - increment when prompt format changes to invalidate cache
    PROMPT_VERSION = "v1.1"  # Bumped for integrity validation

    # Default configuration
    # For laptop demos, 1 concurrent request is usually the most reliable across machines/models.
    DEFAULT_MAX_CONCURRENT = 1
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_BASE_DELAY = 1.0  # Base delay for exponential backoff (seconds)
    # 60s is too low for many local models on commodity laptops; aim for reliability.
    DEFAULT_TIMEOUT = 180  # Request timeout (seconds)
    # Lower default output tokens to reduce tail latency; caller can override via CLI.
    DEFAULT_NUM_PREDICT = 200
    DEFAULT_TEMPERATURE = 0.1

    # Pre-compiled regex patterns for faster parsing
    _VERDICT_PATTERN = re.compile(r"VERDICT:\s*(\w+)", re.IGNORECASE)
    _REASONING_PATTERN = re.compile(
        r"REASONING:\s*(.+?)(?=\n[A-Z_]+:|$)", re.IGNORECASE | re.DOTALL
    )
    _SUGGESTED_PATTERN = re.compile(
        r"SUGGESTED_LABEL:\s*(.+?)(?=\n[A-Z_]+:|$)", re.IGNORECASE | re.DOTALL
    )
    _EXPLANATION_PATTERN = re.compile(
        r"EXPLANATION:\s*(.+?)(?=\n[A-Z_]+:|$)", re.IGNORECASE | re.DOTALL
    )

    def __init__(
        self,
        model_name: Optional[str] = None,
        ollama_url: Optional[str] = None,
        max_concurrent: Optional[int] = None,
        max_retries: Optional[int] = None,
        timeout_s: Optional[int] = None,
        num_predict: Optional[int] = None,
        temperature: Optional[float] = None,
        enable_cache: Optional[bool] = None,
        strict_validation: Optional[bool] = None,
        persistent_cache: Optional[bool] = None,
        cache_dir: Optional[str] = None,
        cache_ttl_hours: Optional[int] = None,
        use_compact_prompt: Optional[bool] = None,
    ):
        """Initialize ReviewEngine with config defaults if parameters not provided."""
        # Load config for defaults
        config = get_config()
        perf_config = config.get_performance_config()
        cache_config = config.get_cache_config()
        demo_config = config.get_demo_config()
        
        # Use provided values or fall back to config, then hardcoded defaults
        self.model_name = model_name or config.get_model_default()
        self.ollama_url = ollama_url or config.get_ollama_url()
        self.max_concurrent = max_concurrent or perf_config.get("max_concurrent", DEFAULT_MAX_CONCURRENT)
        self.max_retries = max_retries or perf_config.get("max_retries", DEFAULT_MAX_RETRIES)
        self.timeout_s = timeout_s or perf_config.get("timeout", DEFAULT_TIMEOUT)
        self.num_predict = num_predict or perf_config.get("num_predict", DEFAULT_NUM_PREDICT)
        self.temperature = temperature if temperature is not None else perf_config.get("temperature", DEFAULT_TEMPERATURE)
        self.enable_cache = enable_cache if enable_cache is not None else cache_config.get("enable", True)
        self.strict_validation = strict_validation if strict_validation is not None else demo_config.get("strict_validation", True)
        self.persistent_cache = persistent_cache if persistent_cache is not None else cache_config.get("persistent", True)
        self.cache_dir = cache_dir or cache_config.get("cache_dir", ".cache")
        self.cache_ttl_hours = cache_ttl_hours if cache_ttl_hours is not None else cache_config.get("ttl_hours", 168)
        self.use_compact_prompt = use_compact_prompt if use_compact_prompt is not None else demo_config.get("use_compact_prompt", False)
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.num_predict = num_predict
        self.temperature = temperature
        self.enable_cache = enable_cache
        self.strict_validation = strict_validation
        self.use_compact_prompt = use_compact_prompt
        self.labels = self._load_labels()

        # Build set of valid label names for fast lookup (case-insensitive)
        self._valid_labels: Set[str] = set()
        self._valid_labels_lower: Dict[str, str] = {}  # lowercase -> canonical name
        for label in self.labels.get("labels", []):
            name = label.get("name", "")
            self._valid_labels.add(name)
            self._valid_labels_lower[name.lower()] = name

        # Compact label summary for compressed prompts
        self._labels_compact = "|".join(
            [f"{label['name']}:{label['definition'][:40]}" for label in self.labels.get("labels", [])]
        )

        # Prompt cache: hash -> response
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._persistent_hits = 0
        self._persistent_cache: Optional[PersistentCache] = None
        if self.enable_cache and self.persistent_cache:
            self._persistent_cache = PersistentCache(
                cache_dir=self.cache_dir,
                ttl_hours=self.cache_ttl_hours,
            )

        # Validation statistics
        self._validation_failures = 0
        self._invalid_labels_rejected = 0

        # HTTP session for connection pooling
        self._session: Optional[aiohttp.ClientSession] = None

    def _load_labels(self) -> Dict[str, Any]:
        """Load label definitions from config."""
        config_path = Path(__file__).parent.parent / "configs" / "labels.yaml"
        if config_path.exists():
            with open(config_path) as f:
                data: Any = yaml.safe_load(f)
                return data if data else {"labels": []}
        return {"labels": []}

    def _validate_verdict(self, verdict: str) -> Verdict:
        """Validate and sanitize verdict value.

        Returns:
            Validated Verdict enum value, defaults to UNCERTAIN if invalid.
        """
        verdict_upper = verdict.strip().upper()
        verdict_map = {
            "CORRECT": Verdict.CORRECT,
            "INCORRECT": Verdict.INCORRECT,
            "UNCERTAIN": Verdict.UNCERTAIN,
        }
        return verdict_map.get(verdict_upper, Verdict.UNCERTAIN)

    def _validate_label(self, label: Optional[str]) -> tuple[Optional[str], bool]:
        """Validate suggested label against known labels.

        Args:
            label: The label to validate

        Returns:
            Tuple of (canonical_label_or_none, was_valid)
        """
        if label is None or label.lower() in ("none", "n/a", ""):
            return None, True

        # Check exact match first
        if label in self._valid_labels:
            return label, True

        # Try case-insensitive match
        canonical = self._valid_labels_lower.get(label.lower())
        if canonical:
            return canonical, True

        # Label not found - this is a guardrail violation
        logger.warning(f"Model suggested invalid label '{label}' - rejecting")
        self._invalid_labels_rejected += 1
        return None, False

    def _validate_output(self, parsed: Dict[str, Any], raw_response: str = "") -> ValidationResult:
        """Validate parsed LLM output for integrity.

        Checks:
        1. Verdict is one of Correct/Incorrect/Uncertain
        2. Suggested label (if any) exists in labels.yaml
        3. No obvious prompt injection patterns
        4. Output length limits (defense against resource exhaustion)
        5. No embedded code/markup injection

        Args:
            parsed: The parsed response dictionary
            raw_response: Original raw response for length validation

        Returns:
            ValidationResult with sanitized values and any issues found.
        """
        issues: List[str] = []

        # Check total output length (potential attack vector)
        if raw_response and len(raw_response) > MAX_OUTPUT_TOTAL_LENGTH:
            issues.append(f"Response exceeds max length ({len(raw_response)} > {MAX_OUTPUT_TOTAL_LENGTH})")
            logger.warning(f"Truncating excessively long response: {len(raw_response)} chars")

        # Validate verdict
        raw_verdict = parsed.get("verdict", "Uncertain")
        sanitized_verdict = self._validate_verdict(raw_verdict)
        if raw_verdict not in ("Correct", "Incorrect", "Uncertain"):
            issues.append(f"Invalid verdict '{raw_verdict}' sanitized to {sanitized_verdict.value}")

        # Validate suggested label
        raw_label = parsed.get("suggested_label")
        sanitized_label, label_valid = self._validate_label(raw_label)
        if not label_valid:
            issues.append(f"Rejected unknown label '{raw_label}'")
            # If verdict was Incorrect but label is invalid, downgrade to Uncertain
            if sanitized_verdict == Verdict.INCORRECT:
                sanitized_verdict = Verdict.UNCERTAIN
                issues.append("Downgraded verdict to Uncertain due to invalid label")

        # Truncate excessively long reasoning/explanation
        reasoning = parsed.get("reasoning", "")
        explanation = parsed.get("explanation", "")

        if len(reasoning) > MAX_REASONING_LENGTH:
            issues.append(f"Reasoning truncated ({len(reasoning)} > {MAX_REASONING_LENGTH})")
            parsed["reasoning"] = reasoning[:MAX_REASONING_LENGTH] + "... [TRUNCATED]"

        if len(explanation) > MAX_EXPLANATION_LENGTH:
            issues.append(f"Explanation truncated ({len(explanation)} > {MAX_EXPLANATION_LENGTH})")
            parsed["explanation"] = explanation[:MAX_EXPLANATION_LENGTH] + "... [TRUNCATED]"

        # Check for potential prompt injection patterns in all text fields
        text_to_check = f"{reasoning} {explanation} {raw_response}"

        for pattern in SUSPICIOUS_PATTERNS:
            if re.search(pattern, text_to_check, re.IGNORECASE):
                issues.append("Suspicious pattern detected: potential prompt injection")
                self._validation_failures += 1
                logger.warning(f"Prompt injection pattern matched: {pattern[:30]}...")
                # Don't fully trust this output
                if self.strict_validation:
                    sanitized_verdict = Verdict.UNCERTAIN
                break

        # Log validation results for security auditing
        if issues:
            logger.debug(f"Validation issues ({len(issues)}): {issues}")

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            sanitized_verdict=sanitized_verdict,
            sanitized_label=sanitized_label,
        )

    def get_validation_stats(self) -> Dict[str, int]:
        """Return validation statistics."""
        return {
            "validation_failures": self._validation_failures,
            "invalid_labels_rejected": self._invalid_labels_rejected,
        }

    def get_valid_labels(self) -> List[str]:
        """Return list of valid label names."""
        return list(self._valid_labels)

    def _build_prompt(self, text: str, pred_label: str, confidence: float) -> str:
        """Build the unified review prompt."""
        if self.use_compact_prompt:
            return self._build_compact_prompt(text, pred_label, confidence)

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

    def _build_compact_prompt(self, text: str, pred_label: str, confidence: float) -> str:
        """Compressed prompt to reduce tokens and latency."""
        return (
            f'Audit: Is "{pred_label}" ({confidence}) correct for: "{text}"?\n'
            f"Labels: {self._labels_compact}\n"
            "Reply: VERDICT:[Correct/Incorrect/Uncertain] "
            "REASONING:[1 sentence] SUGGESTED:[label or None] EXPLANATION:[brief]"
        )

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
            "persistent_hits": self._persistent_hits,
            "persistent_enabled": self._persistent_cache is not None,
        }

    def clear_cache(self):
        """Clear the prompt cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._persistent_hits = 0
        if self._persistent_cache:
            self._persistent_cache.clear()

    async def review_sample_async(
        self,
        text: str,
        pred_label: str,
        confidence: float,
        sample_id: str,
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> Dict[str, Any]:
        """Review a single sample asynchronously with caching, retry, and validation.

        Security:
        - All outputs are validated against schema and label guardrails
        - Results are marked as 'untrusted' (sourced from LLM)
        - Invalid labels are rejected and verdict downgraded
        """
        prompt = self._build_prompt(text, pred_label, confidence)
        prompt_hash = self._get_prompt_hash(prompt)

        # Check cache first (in-memory, then persistent if enabled)
        cached = self._get_cached_response(prompt_hash)
        if cached is not None:
            logger.debug(f"Cache hit for sample {sample_id}")
            cached_copy = cached.copy()
            cached_copy.update({
                "sample_id": sample_id,
                "text": text,
                "pred_label": pred_label,
                "confidence": confidence,
                "cache_hit": True,
            })
            return cached_copy

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
            parsed = self._parse_response(response)

            # Validate output integrity (schema + label guardrails + injection detection)
            validation = self._validate_output(parsed, raw_response=response)
            if not validation.is_valid:
                logger.debug(f"Validation issues for {sample_id}: {validation.issues}")

            # Build result with sanitized/validated values
            result = {
                "verdict": validation.sanitized_verdict.value,
                "reasoning": parsed["reasoning"],
                "suggested_label": validation.sanitized_label,
                "explanation": parsed["explanation"],
                "sample_id": sample_id,
                "text": text,
                "pred_label": pred_label,
                "confidence": confidence,
                "success": True,
                "prompt_hash": prompt_hash,
                "latency_ms": latency_ms,
                "cache_hit": False,
                # Security metadata
                "_untrusted": True,  # Mark LLM output as untrusted
                "_validation_issues": validation.issues if validation.issues else None,
            }

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
                    "_untrusted": True,
                }
                self._store_cache(prompt_hash, cache_entry)

            return result

        except Exception as e:
            logger.error(f"Failed to review sample {sample_id}: {e}")
            return {
                "sample_id": sample_id,
                "text": text,
                "pred_label": pred_label,
                "confidence": confidence,
                "verdict": Verdict.UNCERTAIN.value,
                "reasoning": f"Error: {str(e)}",
                "suggested_label": None,
                "explanation": "Review failed",
                "success": False,
                "error": str(e),
                "cache_hit": False,
                "_untrusted": True,
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
            except OllamaHTTPError as e:
                last_exception = e
                # Retry on overload/transient failures; do not retry on client errors.
                if 500 <= e.status <= 599 or e.status in (408, 429):
                    logger.warning(
                        f"Server error on attempt {attempt + 1}/{self.max_retries}: {e} (status={e.status})"
                    )
                else:
                    raise
            except Exception:
                # Unknown errors: fail fast (we don't know if it's safe to retry)
                raise

            if attempt < self.max_retries - 1:
                # Exponential backoff with jitter
                delay = self.DEFAULT_BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
                logger.debug(f"Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)

        logger.error("Max retries exceeded for Ollama call")
        raise last_exception if last_exception else Exception("Max retries exceeded")

    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API using a pooled HTTP session."""
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.num_predict}
        }

        session = await self._get_session()
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise OllamaHTTPError(status=resp.status, body=body)
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
        verdict_match = self._VERDICT_PATTERN.search(response)
        if verdict_match:
            verdict = verdict_match.group(1).capitalize()
            if verdict in ["Correct", "Incorrect", "Uncertain"]:
                result["verdict"] = verdict
                logger.debug(f"Parsed verdict via regex: {verdict}")

        reasoning_match = self._REASONING_PATTERN.search(response)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip().split('\n')[0].strip()

        suggested_match = self._SUGGESTED_PATTERN.search(response)
        if suggested_match:
            suggested = suggested_match.group(1).strip().split('\n')[0].strip()
            if suggested.lower() != "none":
                result["suggested_label"] = suggested

        explanation_match = self._EXPLANATION_PATTERN.search(response)
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
        try:
            results = await asyncio.gather(*tasks)
            return list(results)
        finally:
            await self.aclose()

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

    async def warm_up_model(self):
        """Prime Ollama to warm KV cache and reduce cold-start latency."""
        warmup_prompt = "Respond with 'ready'"
        try:
            await self._call_ollama_with_retry(warmup_prompt)
            logger.debug("Model warm-up completed")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a pooled HTTP session."""
        if self._session and not self._session.closed:
            return self._session

        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent * 2,
            limit_per_host=self.max_concurrent * 2,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self._session

    async def aclose(self):
        """Close pooled HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    def _get_cached_response(self, prompt_hash: str) -> Optional[Dict[str, Any]]:
        """Return cached response from memory or persistent store."""
        if not self.enable_cache:
            return None

        if prompt_hash in self._cache:
            self._cache_hits += 1
            return self._cache[prompt_hash]

        if self._persistent_cache:
            cached = self._persistent_cache.get(
                prompt_hash=prompt_hash,
                version=self.PROMPT_VERSION,
                model=self.model_name,
            )
            if cached:
                self._persistent_hits += 1
                self._cache[prompt_hash] = cached
                self._cache_hits += 1
                return cached

        return None

    def _store_cache(self, prompt_hash: str, cache_entry: Dict[str, Any]):
        """Persist cache entry to memory and disk (if enabled)."""
        self._cache[prompt_hash] = cache_entry
        if self._persistent_cache:
            self._persistent_cache.set(
                prompt_hash=prompt_hash,
                version=self.PROMPT_VERSION,
                model=self.model_name,
                response=cache_entry,
            )


class PersistentCache:
    """SQLite-backed prompt cache with TTL and version-aware invalidation."""

    def __init__(self, cache_dir: str = ".cache", ttl_hours: int = 168):
        self.db_path = Path(cache_dir) / "prompt_cache.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    prompt_hash TEXT PRIMARY KEY,
                    version TEXT,
                    model TEXT,
                    response TEXT,
                    created_at REAL,
                    hit_count INTEGER DEFAULT 0
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_version ON cache(version)")

    def get(self, prompt_hash: str, version: str, model: str) -> Optional[Dict[str, Any]]:
        """Return cached response if present and not expired."""
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT response, created_at FROM cache WHERE prompt_hash=? AND version=? AND model=?",
                (prompt_hash, version, model),
            ).fetchone()

            if not row:
                return None

            response_text, created_at = row
            if self.ttl_seconds and (now - created_at) > self.ttl_seconds:
                conn.execute("DELETE FROM cache WHERE prompt_hash=?", (prompt_hash,))
                return None

            conn.execute(
                "UPDATE cache SET hit_count = hit_count + 1 WHERE prompt_hash=?",
                (prompt_hash,),
            )
            try:
                cached = json.loads(response_text)
                if isinstance(cached, dict):
                    return cached
                return None
            except json.JSONDecodeError:
                return None

    def set(self, prompt_hash: str, version: str, model: str, response: Dict[str, Any]):
        """Persist cache entry."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache (prompt_hash, version, model, response, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (prompt_hash, version, model, json.dumps(response), time.time()),
            )

    def clear(self):
        """Remove all cache entries."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")

