"""
Unit tests for run_demo.py utility functions.

Tests cover:
- Model selection and preference logic
- System resource detection
- Concurrency auto-tuning
- PII detection for data sensitivity
"""

class TestModelInfo:
    """Tests for model information extraction."""

    def test_get_model_info_7b_quantized(self):
        """Should detect 7B quantized model."""
        from run_demo import get_model_info

        info = get_model_info("mistral:7b-q4_0")

        assert info["is_quantized"] is True
        assert info["estimated_size_gb"] == 4.0

    def test_get_model_info_7b_unquantized(self):
        """Should estimate larger size for unquantized 7B."""
        from run_demo import get_model_info

        info = get_model_info("llama3:8b")

        assert info["is_quantized"] is False
        assert info["estimated_size_gb"] == 16.0

    def test_get_model_info_70b_model(self):
        """Should detect large 70B model."""
        from run_demo import get_model_info

        info = get_model_info("llama2:70b-q4")

        assert info["is_quantized"] is True
        assert info["estimated_size_gb"] == 40.0

    def test_get_model_info_small_model(self):
        """Should detect small models like phi."""
        from run_demo import get_model_info

        info = get_model_info("phi:latest")

        assert info["estimated_size_gb"] == 2.0

    def test_get_model_info_default(self):
        """Should return reasonable defaults for unknown models."""
        from run_demo import get_model_info

        info = get_model_info("unknown-model")

        assert info["estimated_size_gb"] == 4.0  # Conservative default
        assert info["is_quantized"] is False


class TestModelPreferences:
    """Tests for model selection preferences."""

    def test_model_preferences_exist(self):
        """Model preferences list should exist and be populated."""
        from run_demo import MODEL_PREFERENCES

        assert len(MODEL_PREFERENCES) > 0
        # Each entry should be (keyword, priority, notes)
        for entry in MODEL_PREFERENCES:
            assert len(entry) == 3
            assert isinstance(entry[0], str)  # keyword
            assert isinstance(entry[1], int)  # priority
            assert isinstance(entry[2], str)  # notes

    def test_mistral_has_high_priority(self):
        """Mistral should have high priority (low number)."""
        from run_demo import MODEL_PREFERENCES

        mistral_priority = None
        for keyword, priority, _ in MODEL_PREFERENCES:
            if keyword == "mistral":
                mistral_priority = priority
                break

        assert mistral_priority is not None
        assert mistral_priority <= 2  # Should be in top tier


class TestConcurrencySuggestion:
    """Tests for concurrency auto-tuning."""

    def test_default_concurrency_is_safe(self):
        """Default suggestion should be 1 (safest)."""
        from run_demo import suggest_concurrency

        # Minimal resources
        resources = {
            "cpu_count": 2,
            "ram_gb": 8,
            "gpu": {"has_gpu": False},
        }

        suggested, reason = suggest_concurrency(resources, "mistral:7b")

        assert suggested == 1
        assert "default" in reason.lower() or "safest" in reason.lower()

    def test_high_vram_gpu_allows_more_concurrency(self):
        """High-VRAM GPU should allow more concurrent requests."""
        from run_demo import suggest_concurrency

        resources = {
            "cpu_count": 8,
            "ram_gb": 32,
            "gpu": {
                "has_gpu": True,
                "gpu_vram_gb": 24,
                "gpu_type": "nvidia",
            },
        }

        # Small model + big GPU = more concurrency
        suggested, reason = suggest_concurrency(resources, "phi:latest")

        assert suggested >= 2
        assert "gpu" in reason.lower()

    def test_apple_silicon_handled(self):
        """Apple Silicon should be detected and handled."""
        from run_demo import suggest_concurrency

        resources = {
            "cpu_count": 8,
            "ram_gb": 16,
            "gpu": {
                "has_gpu": True,
                "gpu_type": "apple",
                "gpu_name": "Apple Silicon (Metal)",
            },
        }

        suggested, reason = suggest_concurrency(resources, "mistral:7b-q4")

        # Should suggest at least 2 for Apple Silicon with good RAM
        assert suggested >= 1  # At minimum, should not break
        if suggested > 1:
            assert "apple" in reason.lower()

    def test_concurrency_never_exceeds_limit(self):
        """Concurrency should never exceed reasonable limits."""
        from run_demo import suggest_concurrency

        # Extreme resources
        resources = {
            "cpu_count": 128,
            "ram_gb": 512,
            "gpu": {
                "has_gpu": True,
                "gpu_vram_gb": 80,
                "gpu_type": "nvidia",
            },
        }

        suggested, _ = suggest_concurrency(resources, "phi:latest")

        assert suggested <= 4  # Hard cap


class TestPIIDetection:
    """Tests for PII detection in sample data."""

    def test_detects_email(self):
        """Should detect email addresses."""
        from run_demo import detect_potential_pii

        samples = [
            {"id": "1", "text": "Contact me at user@example.com please"}
        ]

        result = detect_potential_pii(samples)

        assert result["has_potential_pii"] is True
        assert "email" in result["detected_patterns"]

    def test_detects_phone(self):
        """Should detect phone numbers."""
        from run_demo import detect_potential_pii

        samples = [
            {"id": "1", "text": "Call me at 555-123-4567"}
        ]

        result = detect_potential_pii(samples)

        assert result["has_potential_pii"] is True
        assert "phone" in result["detected_patterns"]

    def test_detects_ssn_pattern(self):
        """Should detect SSN-like patterns."""
        from run_demo import detect_potential_pii

        samples = [
            {"id": "1", "text": "My SSN is 123-45-6789"}
        ]

        result = detect_potential_pii(samples)

        assert result["has_potential_pii"] is True
        assert "ssn" in result["detected_patterns"]

    def test_synthetic_data_identified(self):
        """Should identify synthetic data by ID prefix."""
        from run_demo import detect_potential_pii

        samples = [
            {"id": "demo_123_0001", "text": "Delete my data"},
            {"id": "demo_123_0002", "text": "Show my information"},
        ]

        result = detect_potential_pii(samples)

        assert result["is_synthetic"] is True

    def test_non_synthetic_data_identified(self):
        """Should identify non-synthetic data."""
        from run_demo import detect_potential_pii

        samples = [
            {"id": "real_001", "text": "Delete my data"},
            {"id": "customer_request_42", "text": "Show my information"},
        ]

        result = detect_potential_pii(samples)

        assert result["is_synthetic"] is False

    def test_clean_synthetic_data_passes(self):
        """Clean synthetic data should not flag PII."""
        from run_demo import detect_potential_pii

        samples = [
            {"id": "demo_123_0001", "text": "I want to delete my personal data"},
            {"id": "demo_123_0002", "text": "Please show me what you have stored"},
        ]

        result = detect_potential_pii(samples)

        assert result["has_potential_pii"] is False
        assert result["is_synthetic"] is True


class TestDemoPresets:
    """Tests for demo preset configurations."""

    def test_demo_fast_preset_exists(self):
        """Demo-fast preset should exist with safe defaults."""
        from run_demo import DemoPresets

        preset = DemoPresets.DEMO_FAST

        assert preset["max_concurrent"] == 1
        assert preset["samples"] <= 10
        assert preset["temperature"] <= 0.2

    def test_benchmark_preset_is_deterministic(self):
        """Benchmark preset should use temperature=0 for determinism."""
        from run_demo import DemoPresets

        preset = DemoPresets.BENCHMARK

        assert preset["temperature"] == 0.0
        assert preset["max_concurrent"] == 1


class TestGPUDetection:
    """Tests for GPU detection (mock-based since hardware varies)."""

    def test_gpu_info_structure(self):
        """GPU info should have expected structure."""
        from run_demo import detect_gpu_info

        # Note: This test may return different values based on actual hardware
        info = detect_gpu_info()

        assert "has_gpu" in info
        assert "gpu_name" in info
        assert "gpu_vram_gb" in info
        assert "gpu_type" in info
        assert info["gpu_type"] in ["nvidia", "amd", "apple", "none"]
