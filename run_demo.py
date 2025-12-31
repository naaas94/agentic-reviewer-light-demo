#!/usr/bin/env python3
"""
Agentic Reviewer Demo
=====================

Single-command demonstration of LLM-powered semantic auditing.

Usage:
    python run_demo.py                  # Full demo (requires Ollama)
    python run_demo.py --demo-fast      # Quick demo preset for sales engineers
    python run_demo.py --no-llm         # Dry run without Ollama
    python run_demo.py --samples 20     # Custom sample count
    python run_demo.py --seed 42        # Reproducible run
    python run_demo.py --redact         # Redact sensitive text in outputs
    python run_demo.py --verbose        # Enable debug logging
"""

import argparse
import asyncio
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from core.config_loader import get_config
from core.logging_config import get_logger, setup_logging
from core.report_generator import ReportGenerator
from core.review_engine import ReviewEngine
from core.synthetic_generator import SyntheticGenerator

# ============================================================================
# DEMO PRESETS
# ============================================================================

# DemoPresets removed - now loaded from config.yaml


# ============================================================================
# SYSTEM RESOURCE DETECTION
# ============================================================================

def detect_gpu_info() -> Dict[str, Any]:
    """Detect GPU information for Ollama acceleration.

    Returns dict with:
        - has_gpu: bool - whether a usable GPU was detected
        - gpu_name: str - GPU name if detected
        - gpu_vram_gb: float - VRAM in GB if detectable
        - gpu_type: str - "nvidia", "amd", "apple", or "none"
    """
    gpu_info = {
        "has_gpu": False,
        "gpu_name": None,
        "gpu_vram_gb": None,
        "gpu_type": "none",
    }

    # Try NVIDIA GPU detection (most common for Ollama acceleration)
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            gpu_info["has_gpu"] = True
            gpu_info["gpu_name"] = parts[0].strip() if parts else "NVIDIA GPU"
            gpu_info["gpu_type"] = "nvidia"
            if len(parts) > 1:
                try:
                    vram_mb = float(parts[1].strip())
                    gpu_info["gpu_vram_gb"] = round(vram_mb / 1024, 1)
                except (ValueError, IndexError):
                    pass
            return gpu_info
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    except Exception:
        pass

    # Check for Apple Silicon (Metal acceleration)
    if platform.system() == "Darwin":
        try:
            import subprocess
            result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                    capture_output=True, text=True, timeout=5)
            if "Apple" in result.stdout:
                gpu_info["has_gpu"] = True
                gpu_info["gpu_name"] = "Apple Silicon (Metal)"
                gpu_info["gpu_type"] = "apple"
                # Apple Silicon shares system RAM
                return gpu_info
        except Exception:
            pass

    return gpu_info


def detect_system_resources() -> Dict[str, Any]:
    """Detect available system resources for auto-tuning.

    Returns comprehensive system info including CPU, RAM, and GPU.
    """
    resources = {
        "cpu_count": os.cpu_count() or 1,
        "platform": platform.system(),
        "ram_gb": None,
    }

    # Try to get RAM info
    try:
        if platform.system() == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulong = ctypes.c_ulong

            class MEMORYSTATUS(ctypes.Structure):
                _fields_ = [
                    ('dwLength', c_ulong),
                    ('dwMemoryLoad', c_ulong),
                    ('dwTotalPhys', c_ulong),
                    ('dwAvailPhys', c_ulong),
                    ('dwTotalPageFile', c_ulong),
                    ('dwAvailPageFile', c_ulong),
                    ('dwTotalVirtual', c_ulong),
                    ('dwAvailVirtual', c_ulong),
                ]

            memstatus = MEMORYSTATUS()
            memstatus.dwLength = ctypes.sizeof(MEMORYSTATUS)
            kernel32.GlobalMemoryStatus(ctypes.byref(memstatus))
            resources["ram_gb"] = memstatus.dwTotalPhys / (1024**3)
        else:
            # Unix-like systems
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        kb = int(line.split()[1])
                        resources["ram_gb"] = kb / (1024**2)
                        break
    except Exception:
        pass  # RAM detection failed, leave as None

    # Detect GPU
    resources["gpu"] = detect_gpu_info()

    return resources


def suggest_concurrency(resources: Dict[str, Any], model_name: str) -> Tuple[int, str]:
    """Suggest optimal concurrency based on system resources and model size.

    Conservative defaults prioritize reliability over speed.

    Returns:
        Tuple of (suggested_concurrency, reason)
    """
    # Default: single request is safest for demos
    suggested = 1
    reason = "default (safest for demos)"

    ram_gb = resources.get("ram_gb")
    cpu_count = resources.get("cpu_count", 1)
    gpu_info = resources.get("gpu", {})
    model_info = get_model_info(model_name)

    model_size_gb = model_info["estimated_size_gb"]
    has_gpu = gpu_info.get("has_gpu", False)
    gpu_vram_gb = gpu_info.get("gpu_vram_gb")

    # GPU-accelerated inference can handle more concurrency
    if has_gpu:
        if gpu_vram_gb and gpu_vram_gb >= 16:
            # High-end GPU: can handle moderate concurrency
            if model_size_gb <= 8:
                suggested = 3
                reason = f"GPU ({gpu_vram_gb:.0f}GB VRAM) + small model"
            elif model_size_gb <= 16:
                suggested = 2
                reason = f"GPU ({gpu_vram_gb:.0f}GB VRAM) + medium model"
        elif gpu_vram_gb and gpu_vram_gb >= 8:
            # Mid-range GPU
            if model_size_gb <= 4:
                suggested = 2
                reason = f"GPU ({gpu_vram_gb:.0f}GB VRAM)"
        elif gpu_info.get("gpu_type") == "apple":
            # Apple Silicon shares RAM with GPU
            if ram_gb and ram_gb >= 16 and model_size_gb <= 8:
                suggested = 2
                reason = "Apple Silicon unified memory"

    # CPU-only: be more conservative
    elif ram_gb:
        # Need headroom: model in RAM + OS + working memory
        available_for_model = ram_gb - 4  # Reserve 4GB for OS/other

        if available_for_model >= model_size_gb * 2 and cpu_count >= 8:
            # Plenty of RAM headroom
            if model_size_gb <= 4:
                suggested = 2
                reason = f"sufficient RAM ({ram_gb:.0f}GB) + small model"
        elif ram_gb >= 32 and model_size_gb <= 8:
            suggested = 2
            reason = f"high RAM ({ram_gb:.0f}GB)"

    # Never exceed reasonable limits
    suggested = min(suggested, 4)

    return suggested, reason

# Python version check
MIN_PYTHON = (3, 9)
if sys.version_info < MIN_PYTHON:
    print(f"❌ Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required (found {sys.version_info.major}.{sys.version_info.minor})")
    sys.exit(1)

# ============================================================================
# DATA SENSITIVITY DETECTION
# ============================================================================

def detect_potential_pii(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect potential PII patterns in sample data.

    This is a lightweight heuristic check, NOT a comprehensive PII detector.
    Used to warn users when they might be processing real data without redaction.

    Returns:
        Dict with has_potential_pii, detected_patterns, sample_count
    """
    import re

    pii_patterns = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\b',
        "ssn": r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
        "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        "name_like": r'\b(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+\b',
    }

    detected = []
    samples_with_pii = 0

    for sample in samples:
        text = sample.get("text", "")
        sample_has_pii = False

        for pattern_name, pattern in pii_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                if pattern_name not in detected:
                    detected.append(pattern_name)
                sample_has_pii = True

        if sample_has_pii:
            samples_with_pii += 1

    # Also check if data looks "real" vs synthetic (synthetic IDs have known patterns)
    is_synthetic = all(
        sample.get("id", "").startswith("demo_") or sample.get("id", "").startswith("synth_")
        for sample in samples
    )

    return {
        "has_potential_pii": len(detected) > 0,
        "is_synthetic": is_synthetic,
        "detected_patterns": detected,
        "samples_with_pii": samples_with_pii,
        "total_samples": len(samples),
    }


# ============================================================================
# TERMINAL UI
# ============================================================================

class UI:
    """Minimal terminal UI."""

    W = 64

    @classmethod
    def header(cls, text: str):
        print(f"\n╔{'═' * (cls.W - 2)}╗")
        print(f"║{text.center(cls.W - 2)}║")

    @classmethod
    def sep(cls):
        print(f"╠{'═' * (cls.W - 2)}╣")

    @classmethod
    def row(cls, text: str):
        print(f"║ {text.ljust(cls.W - 4)} ║")

    @classmethod
    def footer(cls):
        print(f"╚{'═' * (cls.W - 2)}╝")

    @classmethod
    def phase(cls, num: int, name: str, result: str):
        cls.row(f"PHASE {num}: {name.ljust(28)} ✓ {result}")

    @classmethod
    def progress(cls, current: int, total: int):
        pct = current / total * 100 if total else 0
        bar = "█" * int(30 * current / total) + "░" * (30 - int(30 * current / total))
        print(f"\r║ Progress: [{bar}] {current}/{total} ({pct:.0f}%) ".ljust(cls.W - 1) + "║", end="", flush=True)


# ============================================================================
# HEALTH CHECKS
# ============================================================================

def get_ollama_config() -> Dict[str, str]:
    """Get Ollama configuration from environment variables."""
    import os
    return {
        "OLLAMA_MODELS": os.environ.get("OLLAMA_MODELS", ""),
        "OLLAMA_HOST": os.environ.get("OLLAMA_HOST", ""),
        "OLLAMA_PORT": os.environ.get("OLLAMA_PORT", ""),
    }


def list_ollama_models(url: str = "http://localhost:11434") -> List[str]:
    """List all available Ollama models."""
    try:
        import urllib.request
        req = urllib.request.urlopen(f"{url}/api/tags", timeout=5)
        data = json.loads(req.read())
        return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        return []


# MODEL_PREFERENCES removed - now loaded from config.yaml
def get_model_preferences():
    """Get model preferences from config."""
    config = get_config()
    return config.get_model_preferences()


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get estimated model info based on name patterns.

    Returns:
        Dict with estimated_size_gb, is_quantized, context_size
    """
    info = {
        "estimated_size_gb": 4.0,  # Default assumption
        "is_quantized": False,
        "context_size": 4096,
    }

    model_lower = model_name.lower()

    # Detect quantization (various naming patterns: :q4, -q4, q4_, _q4)
    if any(q in model_lower for q in [":q4", ":q5", ":q8", "q4_", "q5_", "q8_", "-q4", "-q5", "-q8", "_q4", "_q5", "_q8"]):
        info["is_quantized"] = True

    # Estimate size based on parameter count in name
    if "70b" in model_lower:
        info["estimated_size_gb"] = 40.0 if info["is_quantized"] else 140.0
    elif "34b" in model_lower or "33b" in model_lower:
        info["estimated_size_gb"] = 20.0 if info["is_quantized"] else 70.0
    elif "13b" in model_lower:
        info["estimated_size_gb"] = 8.0 if info["is_quantized"] else 26.0
    elif "7b" in model_lower or "8b" in model_lower:
        info["estimated_size_gb"] = 4.0 if info["is_quantized"] else 16.0
    elif "3b" in model_lower:
        info["estimated_size_gb"] = 2.0 if info["is_quantized"] else 6.0
    elif "1b" in model_lower or "2b" in model_lower:
        info["estimated_size_gb"] = 1.0 if info["is_quantized"] else 3.0
    elif any(small in model_lower for small in ["phi", "tinyllama", "gemma:2b"]):
        info["estimated_size_gb"] = 2.0

    # Detect extended context
    if any(ctx in model_lower for ctx in ["32k", "128k", "context"]):
        info["context_size"] = 32768

    return info


def check_ollama(
    url: str = "http://localhost:11434",
    preferred_model: Optional[str] = None
) -> Tuple[bool, Optional[str], str, List[str]]:
    """Check Ollama availability and select best model.

    Model selection priority:
    1. Exact match to preferred_model (if specified)
    2. Prefix match to preferred_model (e.g., "mistral" matches "mistral:latest")
    3. Best model from preference list based on priority score
    4. Any available model as fallback

    Returns:
        Tuple of (is_running, selected_model, status, all_models)
    """
    logger = get_logger("main")

    try:
        import urllib.error
        import urllib.request
        req = urllib.request.urlopen(f"{url}/api/tags", timeout=5)
        data = json.loads(req.read())
        models = [m.get("name", "") for m in data.get("models", [])]

        if not models:
            return True, None, "no_model", []

        # Priority 1: Exact match to preferred model
        if preferred_model:
            if preferred_model in models:
                logger.debug(f"Exact model match: {preferred_model}")
                return True, preferred_model, "ready", models

            # Priority 2: Prefix match (e.g., "mistral" matches "mistral:latest")
            for model in models:
                if model.startswith(preferred_model) or model.split(":")[0] == preferred_model:
                    logger.debug(f"Prefix model match: {model} (requested: {preferred_model})")
                    return True, model, "ready", models

            logger.debug(f"Preferred model '{preferred_model}' not found, selecting best available")

        # Priority 3: Score all available models and pick best
        best_model = None
        best_score = float('inf')  # Lower is better
        model_preferences = get_model_preferences()

        for model in models:
            model_lower = model.lower()
            for keyword, priority, _ in model_preferences:
                if keyword in model_lower:
                    # Prefer non-code models for classification tasks
                    if "code" in model_lower:
                        priority += 3
                    # Prefer instruction-tuned variants
                    if "instruct" in model_lower or "chat" in model_lower:
                        priority -= 1

                    if priority < best_score:
                        best_score = priority
                        best_model = model
                    break

        if best_model:
            model_info = get_model_info(best_model)
            logger.debug(f"Selected model: {best_model} (score: {best_score}, ~{model_info['estimated_size_gb']:.1f}GB)")
            return True, best_model, "ready", models

        # Priority 4: Any available model as fallback
        logger.debug(f"No preferred model found, using first available: {models[0]}")
        return True, models[0], "ready", models

    except urllib.error.URLError as e:
        logger.debug(f"Ollama connection failed: {e}")
        return False, None, "not_running", []
    except Exception as e:
        logger.debug(f"Ollama check error: {e}")
        return False, None, "not_running", []


def pull_model(model: Optional[str] = None, url: Optional[str] = None) -> bool:
    """Pull a model from Ollama (with progress indication)."""
    import json
    import urllib.request
    
    # Use config defaults if not provided
    if model is None:
        config = get_config()
        model = config.get_model_default()
    if url is None:
        config = get_config()
        url = config.get_ollama_url()

    print(f"\n📥 Pulling {model} model (this may take a few minutes)...")
    print("   This is a one-time download.\n")

    try:
        req = urllib.request.Request(
            f"{url}/api/pull",
            data=json.dumps({"name": model}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=600) as response:
            # Stream the response to show progress
            while True:
                line = response.readline()
                if not line:
                    break
                try:
                    data = json.loads(line)
                    status = data.get("status", "")
                    if "pulling" in status:
                        completed = data.get("completed", 0)
                        total = data.get("total", 1)
                        pct = (completed / total * 100) if total else 0
                        print(f"\r   Progress: {pct:.0f}%", end="", flush=True)
                    elif status == "success":
                        print("\r   Progress: 100%")
                        print(f"✓ Model {model} ready!")
                        return True
                except json.JSONDecodeError:
                    pass

        return True
    except Exception as e:
        print(f"\n❌ Failed to pull model: {e}")
        return False


# ============================================================================
# DEMO ORCHESTRATOR
# ============================================================================

class Demo:
    """Main demo orchestrator."""

    # Warning banner for untrusted outputs
    UNTRUSTED_WARNING = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║ ⚠️  UNTRUSTED OUTPUT WARNING                                                   ║
║                                                                               ║
║ This output contains LLM-generated content that has not been verified by      ║
║ humans. Labels, reasoning, and explanations may be incorrect or misleading.   ║
║                                                                               ║
║ DO NOT use this output for production decisions without human review.         ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

    def __init__(
        self,
        n_samples: int,
        seed: int,
        use_llm: bool,
        model: str,
        max_concurrent: int,
        max_retries: int,
        timeout_s: int,
        num_predict: int,
        temperature: float,
        ollama_url: str,
        enable_cache: bool,
        persistent_cache: bool,
        cache_dir: str,
        use_compact_prompt: bool,
        warmup: bool,
        redact: bool = False,
        strict_validation: bool = True,
    ):
        self.n_samples = n_samples
        self.seed = seed
        self.use_llm = use_llm
        self.model = model
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.num_predict = num_predict
        self.temperature = temperature
        self.ollama_url = ollama_url
        self.enable_cache = enable_cache
        self.persistent_cache = persistent_cache
        self.cache_dir = cache_dir
        self.use_compact_prompt = use_compact_prompt
        self.warmup = warmup
        self.redact = redact
        self.strict_validation = strict_validation
        self.run_id = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        self.run_dir = f"outputs/{self.run_id}"

    def run(self) -> Dict[str, Any]:
        """Execute complete demo."""
        start_time = time.time()

        # Header
        UI.header("AGENTIC REVIEWER DEMO")
        UI.sep()
        UI.row(f"Run ID: {self.run_id}")
        UI.row(f"Samples: {self.n_samples} | Seed: {self.seed} | LLM: {'On' if self.use_llm else 'Off'}")
        UI.sep()

        # Create output directory
        os.makedirs(self.run_dir, exist_ok=True)

        # Phase 1: Generate data
        generator = SyntheticGenerator(seed=self.seed)
        samples = generator.generate_samples(self.n_samples)
        UI.phase(1, "Generate Synthetic Data", f"{len(samples)} samples")

        # Security check: warn if data looks like real PII and redaction is off
        pii_check = detect_potential_pii(samples)
        if pii_check["has_potential_pii"] and not self.redact:
            print()
            print("⚠️  " + "=" * 58)
            print("⚠️  POTENTIAL PII DETECTED")
            print("⚠️  " + "=" * 58)
            print(f"⚠️  Patterns found: {', '.join(pii_check['detected_patterns'])}")
            print(f"⚠️  Affected samples: {pii_check['samples_with_pii']}/{pii_check['total_samples']}")
            print("⚠️  Consider running with --redact to mask sensitive data.")
            print("⚠️  " + "=" * 58)
            print()
        elif not pii_check["is_synthetic"]:
            # Data doesn't have demo_ prefix - might be real data
            UI.row("NOTE: Data may not be synthetic (missing demo_ prefix)")
            UI.row("      Consider --redact if using real client data")

        # Phase 2: Review
        engine = ReviewEngine(
            model_name=self.model,
            ollama_url=self.ollama_url,
            max_concurrent=self.max_concurrent,
            max_retries=self.max_retries,
            timeout_s=self.timeout_s,
            num_predict=self.num_predict,
            temperature=self.temperature,
            enable_cache=self.enable_cache,
            strict_validation=self.strict_validation,
            persistent_cache=self.persistent_cache,
            cache_dir=self.cache_dir,
            use_compact_prompt=self.use_compact_prompt,
        )
        if self.use_llm:
            # Run warm-up and review in the same event loop to avoid session issues
            results = asyncio.run(self._review_with_warmup(engine, samples))
        else:
            results = engine.generate_mock_results(samples)

        success = sum(1 for r in results if r.get("success"))
        UI.phase(2, "LLM Review", f"{success}/{len(results)} reviews")

        # Phase 3: Generate report
        report_gen = ReportGenerator()
        report = report_gen.generate_report(results, self.run_id, {"seed": self.seed})
        UI.phase(3, "Generate Report", f"{len(report.split())} words")

        # Phase 4: Save artifacts
        self._save_artifacts(samples, results, report, generator.get_metadata())
        UI.phase(4, "Save Artifacts", "5 files")

        # Results summary
        duration = time.time() - start_time
        stats = self._calculate_stats(results)

        UI.sep()
        UI.row("RESULTS")
        UI.row(f"├─ Correct:   {stats['correct']:3} ({stats['correct_pct']:.1f}%)")
        UI.row(f"├─ Incorrect: {stats['incorrect']:3} ({stats['incorrect_pct']:.1f}%) → corrections suggested")
        UI.row(f"└─ Uncertain: {stats['uncertain']:3} ({stats['uncertain_pct']:.1f}%)")
        UI.row("")
        UI.row(f"Duration: {duration:.1f}s | Output: {self.run_dir}/")
        UI.footer()

        print(f"\n✓ Demo complete. Explore: {self.run_dir}/")

        return {"run_id": self.run_id, "stats": stats, "duration": duration}

    async def _review_with_warmup(self, engine: ReviewEngine, samples: List[Dict]) -> List[Dict]:
        """Review with optional warm-up, all in the same event loop."""
        # Warm-up if enabled (best-effort, continue even if it fails)
        if self.warmup:
            try:
                await engine.warm_up_model()
            except Exception as warm_err:
                # Warm-up is best-effort; continue even if it fails
                logger = get_logger("main")
                logger.debug(f"Warm-up skipped due to error: {warm_err}")

        # Now run the actual review
        return await self._review_with_progress(engine, samples)

    async def _review_with_progress(self, engine: ReviewEngine, samples: List[Dict]) -> List[Dict]:
        """Review with progress indicator."""
        def on_progress(current, total):
            UI.progress(current, total)

        results = await engine.review_batch_async(samples, on_progress)
        print()  # Clear progress line
        return results

    def _redact_text(self, text: str) -> str:
        """Redact potentially sensitive text content.

        For demo purposes, replaces the middle portion of text with [REDACTED].
        In production, this should use more sophisticated PII detection.
        """
        if not self.redact or not text:
            return text

        if len(text) <= 20:
            return "[REDACTED]"

        # Keep first and last 10 chars, redact middle
        return f"{text[:10]}[...REDACTED...]{text[-10:]}"

    def _save_artifacts(self, samples, results, report, metadata):
        """Save all output artifacts with optional redaction and warnings."""
        # Config with metadata
        config_data = {
            "run_id": self.run_id,
            "seed": self.seed,
            "n_samples": self.n_samples,
            "use_llm": self.use_llm,
            "model": self.model,
            "generated_at": datetime.now().isoformat(),
            "redacted": self.redact,
            "strict_validation": self.strict_validation,
            "_warning": "UNTRUSTED: LLM-generated content requires human verification",
        }

        with open(f"{self.run_dir}/00_config.json", "w") as f:
            json.dump(config_data, f, indent=2)

        # Synthetic data (optionally redacted)
        samples_df = pd.DataFrame(samples)
        if self.redact:
            samples_df["text"] = samples_df["text"].apply(self._redact_text)
        samples_df.to_csv(f"{self.run_dir}/01_synthetic_data.csv", index=False)

        # Results (optionally redacted, with untrusted markers)
        results_to_save = []
        for r in results:
            r_copy = r.copy()
            if self.redact:
                r_copy["text"] = self._redact_text(r_copy.get("text", ""))
                r_copy["reasoning"] = self._redact_text(r_copy.get("reasoning", ""))
            # Ensure untrusted marker is present
            r_copy["_untrusted"] = True
            results_to_save.append(r_copy)

        with open(f"{self.run_dir}/02_review_results.json", "w") as f:
            json.dump(results_to_save, f, indent=2)

        # Labeled dataset (optionally redacted)
        labeled = [{
            "id": r["sample_id"],
            "text": self._redact_text(r["text"]) if self.redact else r["text"],
            "original_label": r["pred_label"],
            "verdict": r["verdict"],
            "corrected_label": r.get("suggested_label") or r["pred_label"],
            "reasoning": self._redact_text(r["reasoning"]) if self.redact else r["reasoning"],
            "ground_truth": r.get("ground_truth"),
            "_untrusted": True,
        } for r in results]
        pd.DataFrame(labeled).to_csv(f"{self.run_dir}/03_labeled_dataset.csv", index=False)

        # Report with warning banner
        report_with_warning = self.UNTRUSTED_WARNING + "\n" + report
        with open(f"{self.run_dir}/04_report.md", "w", encoding="utf-8") as f:
            f.write(report_with_warning)

        # Metrics with validation stats
        stats = self._calculate_stats(results)
        stats["_warning"] = "UNTRUSTED: Statistics derived from LLM outputs"
        stats["redacted"] = self.redact

        with open(f"{self.run_dir}/05_metrics.json", "w") as f:
            json.dump(stats, f, indent=2)

    def _calculate_stats(self, results: List[Dict]) -> Dict:
        total = len(results)
        verdicts = {"Correct": 0, "Incorrect": 0, "Uncertain": 0}
        for r in results:
            v = r.get("verdict", "Uncertain")
            verdicts[v] = verdicts.get(v, 0) + 1

        return {
            "total": total,
            "correct": verdicts["Correct"],
            "incorrect": verdicts["Incorrect"],
            "uncertain": verdicts["Uncertain"],
            "correct_pct": verdicts["Correct"] / total * 100 if total else 0,
            "incorrect_pct": verdicts["Incorrect"] / total * 100 if total else 0,
            "uncertain_pct": verdicts["Uncertain"] / total * 100 if total else 0,
        }


# ============================================================================
# CLI
# ============================================================================

def print_ollama_setup():
    """Print Ollama setup instructions."""
    print("""
╭─────────────────────────────────────────────────────────────╮
│  OLLAMA SETUP (2 minutes, one-time)                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Install Ollama (free, local, no API keys):              │
│     https://ollama.ai/download                              │
│                                                             │
│  2. Start Ollama server:                                    │
│     - Windows: Open Ollama from Start menu, or              │
│     - Command line: ollama serve                            │
│                                                             │
│  3. Verify it's running:                                    │
│     - Check system tray for Ollama icon                     │
│     - Visit http://localhost:11434 in browser               │
│                                                             │
│  4. Run this demo again - it will auto-download the model   │
│                                                             │
╰─────────────────────────────────────────────────────────────╯
""")


def main():
    parser = argparse.ArgumentParser(
        description="Agentic Reviewer Demo - LLM-powered semantic auditing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_demo.py                    # Full demo with LLM (recommended)
  python run_demo.py --demo-fast        # Quick demo preset for sales engineers
  python run_demo.py --samples 20       # More samples
  python run_demo.py --seed 42          # Reproducible run
  python run_demo.py --redact           # Redact sensitive text in outputs
  python run_demo.py --mock             # Quick preview without LLM
  python run_demo.py --verbose          # Enable debug logging

Presets:
  --demo-fast     Quick, reliable demo (8 samples, 1 concurrent, short responses)
  --benchmark     Reproducible performance testing (deterministic, no variance)
        """
    )

    # Preset options (mutually exclusive convenience flags)
    preset_group = parser.add_argument_group("presets", "Pre-configured settings for common use cases")
    preset_group.add_argument("--demo-fast", action="store_true",
                              help="Quick demo preset: fewer samples, faster responses, max reliability")
    preset_group.add_argument("--benchmark", action="store_true",
                              help="Benchmark preset: deterministic, reproducible results")

    # Load config for defaults
    config = get_config()
    
    # Core options
    parser.add_argument("--samples", "-n", type=int, default=None, 
                       help=f"Number of samples (default: {config.get('demo.default_samples', 12)})")
    parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--mock", action="store_true", help="Use mock results (skip LLM)")
    parser.add_argument("--model", type=str, default=None, 
                       help=f"Ollama model (default: {config.get_model_default()})")
    parser.add_argument("--ollama-url", type=str, default=None, 
                       help=f"Ollama base URL (default: {config.get_ollama_url()})")

    # Performance tuning
    perf_config = config.get_performance_config()
    cache_config = config.get_cache_config()
    perf_group = parser.add_argument_group("performance", "Performance tuning options")
    perf_group.add_argument("--max-concurrent", type=int, default=None,
                            help=f"Max concurrent Ollama requests (default: {perf_config.get('max_concurrent', 1)})")
    perf_group.add_argument("--max-retries", type=int, default=None,
                            help=f"Max retries per request (default: {perf_config.get('max_retries', 3)})")
    perf_group.add_argument("--timeout", type=int, default=None,
                            help=f"Per-request timeout seconds (default: {perf_config.get('timeout', 180)})")
    perf_group.add_argument("--num-predict", type=int, default=None,
                            help=f"Ollama num_predict tokens (default: {perf_config.get('num_predict', 200)})")
    perf_group.add_argument("--temperature", type=float, default=None,
                            help=f"Ollama temperature (default: {perf_config.get('temperature', 0.1)})")
    perf_group.add_argument("--auto-tune", action="store_true",
                            help="Auto-tune concurrency based on system resources")
    perf_group.add_argument("--compact-prompts", action="store_true",
                            help="Use compact prompt template to reduce latency")
    perf_group.add_argument("--cache-dir", type=str, default=None,
                            help=f"Directory for persistent prompt cache (default: {cache_config.get('cache_dir', '.cache')})")

    # Security options
    security_group = parser.add_argument_group("security", "Security and privacy options")
    security_group.add_argument("--redact", action="store_true",
                                help="Redact potentially sensitive text in output artifacts")
    security_group.add_argument("--no-strict-validation", action="store_true",
                                help="Disable strict output validation (not recommended)")

    # Other options
    parser.add_argument("--no-cache", action="store_true", help="Disable prompt cache")
    parser.add_argument("--no-persistent-cache", action="store_true",
                        help="Disable disk-backed prompt cache (memory cache still used)")
    parser.add_argument("--yes", action="store_true",
                        help="Non-interactive: auto-download model if needed (no prompts)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose/debug logging")
    parser.add_argument("--no-warmup", action="store_true", help="Skip model warm-up request")

    # Hidden aliases for backwards compatibility
    parser.add_argument("--no-llm", action="store_true", dest="mock", help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Load config (reload in case it changed)
    config = get_config()
    perf_config = config.get_performance_config()
    cache_config = config.get_cache_config()
    demo_config = config.get_demo_config()

    # Apply presets (presets set defaults, explicit args override)
    if args.demo_fast:
        preset = config.get_preset("demo_fast")
    elif args.benchmark:
        preset = config.get_preset("benchmark")
    else:
        preset = {}

    # Resolve final values: explicit arg > preset > config > hardcoded default
    n_samples = (args.samples if args.samples is not None 
                 else preset.get("samples", demo_config.get("default_samples", 12)))
    max_concurrent = (args.max_concurrent if args.max_concurrent is not None 
                     else preset.get("max_concurrent", perf_config.get("max_concurrent", 1)))
    max_retries = (args.max_retries if args.max_retries is not None 
                   else perf_config.get("max_retries", 3))
    timeout_s = (args.timeout if args.timeout is not None 
                 else preset.get("timeout", perf_config.get("timeout", 180)))
    num_predict = (args.num_predict if args.num_predict is not None 
                   else preset.get("num_predict", perf_config.get("num_predict", 200)))
    temperature = (args.temperature if args.temperature is not None 
                  else preset.get("temperature", perf_config.get("temperature", 0.1)))

    # Model and URL from config if not provided
    model_default = args.model if args.model is not None else config.get_model_default()
    ollama_url_default = args.ollama_url if args.ollama_url is not None else config.get_ollama_url()

    seed = args.seed or int(datetime.now().timestamp())
    warmup = args.no_warmup == False and demo_config.get("warmup", True)
    persistent_cache = args.no_persistent_cache == False and cache_config.get("persistent", True)
    cache_dir = args.cache_dir if args.cache_dir is not None else cache_config.get("cache_dir", ".cache")
    use_compact_prompt = args.compact_prompts or demo_config.get("use_compact_prompt", False)

    # Setup logging based on verbosity
    setup_logging(verbose=args.verbose)
    module_logger = get_logger("main")

    print("\n🔍 Checking environment...")
    module_logger.debug(f"Python version: {sys.version}")
    module_logger.debug(f"Arguments: samples={n_samples}, seed={seed}, mock={args.mock}, model={args.model}")

    # Show preset info
    if args.demo_fast:
        print("📦 Using --demo-fast preset (optimized for quick, reliable demos)")
    elif args.benchmark:
        print("📦 Using --benchmark preset (deterministic, reproducible)")

    # Auto-tune concurrency if requested
    if args.auto_tune:
        resources = detect_system_resources()
        suggested, reason = suggest_concurrency(resources, args.model)
        gpu_info = resources.get("gpu", {})

        if gpu_info.get("has_gpu"):
            gpu_name = gpu_info.get("gpu_name", "Unknown GPU")
            vram = gpu_info.get("gpu_vram_gb")
            vram_str = f" ({vram:.0f}GB)" if vram else ""
            module_logger.debug(f"Detected GPU: {gpu_name}{vram_str}")

        if suggested != max_concurrent:
            module_logger.debug(f"Auto-tune: system resources = {resources}")
            print(f"🔧 Auto-tuned concurrency: {max_concurrent} → {suggested} ({reason})")
            max_concurrent = suggested
        else:
            module_logger.debug(f"Auto-tune: keeping concurrency at {max_concurrent} ({reason})")

    use_llm = True
    model_to_use = model_default

    if args.mock:
        print("✓ Mock mode (skipping LLM)")
        use_llm = False
    else:
        # Check Ollama status with improved model selection
        ollama_running, available_model, status, all_models = check_ollama(ollama_url_default, preferred_model=model_default)
        module_logger.debug(f"Ollama status: running={ollama_running}, model={available_model}, status={status}, available={all_models}")

        if status == "not_running":
            print("⚠️  Ollama server is not accessible")
            print("\nNote: If Ollama is running in the system tray, you may need to:")
            print("  1. Right-click the Ollama icon in the system tray")
            print("  2. Ensure the server is started (it may be running but not listening)")
            print("  3. Or restart Ollama from the Start menu")
            print_ollama_setup()

            # Offer to continue in mock mode
            print("Options:")
            print("  1. Start/restart Ollama server (recommended) - see instructions above")
            print("  2. Run with --mock flag for a quick preview without LLM")
            print()
            sys.exit(1)

        elif status == "no_model":
            print("✓ Ollama is running")

            # Check Ollama configuration
            config = get_ollama_config()
            ollama_env_config = get_ollama_config()
            if ollama_env_config["OLLAMA_MODELS"]:
                module_logger.debug(f"OLLAMA_MODELS set to: {ollama_env_config['OLLAMA_MODELS']}")

            # Check what models are actually available
            available_models = list_ollama_models()
            if available_models:
                print(f"⚠️  Found {len(available_models)} model(s), but none are compatible:")
                for model in available_models:
                    print(f"   - {model}")
                print(f"\nWould you like to download '{model_default}' instead? (free, ~4GB)")
            else:
                print("⚠️  No language models installed")

                # Check if models might be on a different drive
                if not ollama_env_config["OLLAMA_MODELS"]:
                    print("\nNote: If Ollama models are stored on a different drive (e.g., D:),")
                    print("you may need to set the OLLAMA_MODELS environment variable:")
                    print("  PowerShell: $env:OLLAMA_MODELS = 'D:\\Apps\\Ollama\\models'")
                    print("  Then restart Ollama for the change to take effect.")
                    print()

                print(f"Would you like to download '{model_default}'? (free, ~4GB)")

            print("This is a one-time download.")
            print("\nAlternatively, you can download manually:")
            print(f"  ollama pull {model_default}")
            print("\n")

            try:
                # Non-interactive safe behavior: don't hang CI/headless runs on input().
                if args.yes:
                    response = "y"
                else:
                    if not sys.stdin.isatty():
                        print("❌ No usable Ollama model found and this session is non-interactive.")
                        print(f"Run: ollama pull {model_default}")
                        print("Or rerun this demo with --yes to auto-download.")
                        sys.exit(2)

                    response = input("Download now? [Y/n]: ").strip().lower()

                if response in ["", "y", "yes"]:
                    if pull_model(model_default, ollama_url_default):
                        model_to_use = model_default
                    else:
                        print("\nFailed to download. You can try manually:")
                        print(f"  ollama pull {model_default}")
                        print("\nOr run with --mock for preview.")
                        sys.exit(1)
                else:
                    print(f"\nTo download manually, run: ollama pull {model_default}")
                    print("Or run with --mock for a quick preview without LLM.")
                    sys.exit(0)
            except KeyboardInterrupt:
                print(f"\n\nTo download manually, run: ollama pull {model_default}")
                print("Or run with --mock for a quick preview.")
                sys.exit(0)

        else:  # status == "ready"
            model_to_use = available_model
            # Show if we fell back to a different model
            if model_default != model_to_use and not model_to_use.startswith(model_default):
                print(f"✓ Ollama ready (requested '{model_default}', using '{model_to_use}')")
            else:
                print(f"✓ Ollama ready with model: {model_to_use}")

    # Show redaction warning if enabled
    if args.redact:
        print("🔒 Redaction enabled: sensitive text will be masked in outputs")

    # Run demo
    demo = Demo(
        n_samples=n_samples,
        seed=seed,
        use_llm=use_llm,
        model=model_to_use,
        max_concurrent=max_concurrent,
        max_retries=max_retries,
        timeout_s=timeout_s,
        num_predict=num_predict,
        temperature=temperature,
        ollama_url=ollama_url_default,
        enable_cache=(not args.no_cache) and cache_config.get("enable", True),
        persistent_cache=persistent_cache,
        cache_dir=cache_dir,
        use_compact_prompt=use_compact_prompt,
        warmup=warmup,
        redact=args.redact,
        strict_validation=(not args.no_strict_validation) and demo_config.get("strict_validation", True),
    )

    try:
        demo.run()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted")
        sys.exit(130)
    except Exception as e:
        module_logger.error(f"Demo failed: {e}")
        print(f"\n❌ Error: {e}")
        if args.verbose:
            print("\n--- Traceback ---")
            traceback.print_exc()
            print("-----------------")
        else:
            print("Try running with --verbose for detailed error information.")
        print("Try running with --mock for a quick preview.")
        sys.exit(1)


if __name__ == "__main__":
    main()
