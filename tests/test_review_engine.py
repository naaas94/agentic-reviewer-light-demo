"""
Unit tests for the ReviewEngine.

Tests cover:
- Response parsing (various formats)
- Cache behavior
- Mock result generation
- Configuration handling
"""

from core.review_engine import ReviewEngine


class TestResponseParsing:
    """Tests for LLM response parsing."""

    def test_parse_correct_response(self):
        """Should parse a correctly formatted response."""
        engine = ReviewEngine()
        response = """VERDICT: Correct
REASONING: The text clearly requests data access.
SUGGESTED_LABEL: None
EXPLANATION: User wants to see their personal data."""

        result = engine._parse_response(response)

        assert result["verdict"] == "Correct"
        assert "data access" in result["reasoning"]
        assert result["suggested_label"] is None
        assert "personal data" in result["explanation"]

    def test_parse_incorrect_response(self):
        """Should parse response with suggested correction."""
        engine = ReviewEngine()
        response = """VERDICT: Incorrect
REASONING: The text requests deletion, not access.
SUGGESTED_LABEL: Erasure
EXPLANATION: User wants their data deleted."""

        result = engine._parse_response(response)

        assert result["verdict"] == "Incorrect"
        assert result["suggested_label"] == "Erasure"

    def test_parse_uncertain_response(self):
        """Should parse uncertain verdict."""
        engine = ReviewEngine()
        response = """VERDICT: Uncertain
REASONING: The text is ambiguous.
SUGGESTED_LABEL: None
EXPLANATION: Could be multiple categories."""

        result = engine._parse_response(response)

        assert result["verdict"] == "Uncertain"
        assert result["suggested_label"] is None

    def test_parse_malformed_response_returns_defaults(self):
        """Should return defaults for malformed response."""
        engine = ReviewEngine()
        response = "This is not formatted correctly at all."

        result = engine._parse_response(response)

        assert result["verdict"] == "Uncertain"
        assert result["reasoning"] == ""
        assert result["suggested_label"] is None

    def test_parse_partial_response(self):
        """Should handle partial responses gracefully."""
        engine = ReviewEngine()
        response = """VERDICT: Correct
REASONING: Makes sense."""

        result = engine._parse_response(response)

        assert result["verdict"] == "Correct"
        assert result["reasoning"] == "Makes sense."
        assert result["explanation"] == ""

    def test_parse_response_with_extra_whitespace(self):
        """Should handle responses with extra whitespace."""
        engine = ReviewEngine()
        response = """  VERDICT:   Correct   
  REASONING:   The label is accurate.  
  SUGGESTED_LABEL:   None   
  EXPLANATION:   Classification is correct.  """  # noqa: W291

        result = engine._parse_response(response)

        assert result["verdict"] == "Correct"
        assert result["reasoning"] == "The label is accurate."

    def test_parse_response_with_multiline_reasoning(self):
        """Should capture first line of reasoning."""
        engine = ReviewEngine()
        response = """VERDICT: Incorrect
REASONING: The text talks about deletion.
This is extra reasoning that should be ignored.
SUGGESTED_LABEL: Erasure
EXPLANATION: User wants data removed."""

        result = engine._parse_response(response)

        assert result["verdict"] == "Incorrect"
        assert "deletion" in result["reasoning"]

    def test_parse_invalid_verdict_defaults_to_uncertain(self):
        """Invalid verdict should default to Uncertain."""
        engine = ReviewEngine()
        response = """VERDICT: Maybe
REASONING: Not sure."""

        result = engine._parse_response(response)

        assert result["verdict"] == "Uncertain"

    def test_parse_none_suggested_label_variations(self):
        """Various 'none' representations should result in None."""
        engine = ReviewEngine()

        none_variations = ["None", "none", "NONE", "N/A"]

        for none_val in none_variations:
            response = f"""VERDICT: Correct
REASONING: Test.
SUGGESTED_LABEL: {none_val}
EXPLANATION: Test."""

            result = engine._parse_response(response)
            # "None" and "none" should be treated as None
            if none_val.lower() == "none":
                assert result["suggested_label"] is None
            else:
                # Others like "N/A" are kept as-is
                assert result["suggested_label"] == none_val


class TestCaching:
    """Tests for prompt caching functionality."""

    def test_cache_enabled_by_default(self):
        """Cache should be enabled by default."""
        engine = ReviewEngine()
        assert engine.enable_cache is True

    def test_cache_can_be_disabled(self):
        """Cache should be disableable."""
        engine = ReviewEngine(enable_cache=False)
        assert engine.enable_cache is False

    def test_cache_stats_initial(self):
        """Initial cache stats should be zero."""
        engine = ReviewEngine()
        stats = engine.get_cache_stats()

        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["cached_prompts"] == 0

    def test_clear_cache(self):
        """Clear cache should reset all stats."""
        engine = ReviewEngine()
        engine._cache["test"] = {"verdict": "Correct"}
        engine._cache_hits = 5
        engine._cache_misses = 3

        engine.clear_cache()

        stats = engine.get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["cached_prompts"] == 0

    def test_prompt_hash_consistency(self):
        """Same prompt should produce same hash."""
        engine = ReviewEngine()
        prompt = "Test prompt content"

        hash1 = engine._get_prompt_hash(prompt)
        hash2 = engine._get_prompt_hash(prompt)

        assert hash1 == hash2
        assert len(hash1) == 16  # MD5 truncated to 16 chars

    def test_different_prompts_different_hashes(self):
        """Different prompts should produce different hashes."""
        engine = ReviewEngine()

        hash1 = engine._get_prompt_hash("Prompt A")
        hash2 = engine._get_prompt_hash("Prompt B")

        assert hash1 != hash2


class TestMockResults:
    """Tests for mock result generation."""

    def test_mock_results_correct_count(self):
        """Should generate results for all samples."""
        engine = ReviewEngine()
        samples = [
            {"id": "1", "text": "Test", "pred_label": "Access Request", "confidence": 0.9},
            {"id": "2", "text": "Test2", "pred_label": "Erasure", "confidence": 0.8},
        ]

        results = engine.generate_mock_results(samples)

        assert len(results) == 2

    def test_mock_results_correct_classification(self):
        """Correctly classified samples should have Correct verdict."""
        engine = ReviewEngine()
        samples = [{
            "id": "1",
            "text": "Test text",
            "pred_label": "Access Request",
            "confidence": 0.9,
            "ground_truth": "Access Request",
            "is_misclassified": False,
        }]

        results = engine.generate_mock_results(samples)

        assert results[0]["verdict"] == "Correct"
        assert results[0]["suggested_label"] is None

    def test_mock_results_incorrect_classification(self):
        """Misclassified samples should have Incorrect verdict."""
        engine = ReviewEngine()
        samples = [{
            "id": "1",
            "text": "Delete my data",
            "pred_label": "Access Request",
            "confidence": 0.7,
            "ground_truth": "Erasure",
            "is_misclassified": True,
        }]

        results = engine.generate_mock_results(samples)

        assert results[0]["verdict"] == "Incorrect"
        assert results[0]["suggested_label"] == "Erasure"

    def test_mock_results_contain_required_fields(self):
        """Mock results should contain all required fields."""
        engine = ReviewEngine()
        samples = [{
            "id": "1",
            "text": "Test",
            "pred_label": "Access Request",
            "confidence": 0.9,
            "ground_truth": "Access Request",
            "is_misclassified": False,
        }]

        results = engine.generate_mock_results(samples)

        required_fields = {
            "sample_id", "text", "pred_label", "confidence",
            "ground_truth", "verdict", "reasoning",
            "suggested_label", "explanation", "success"
        }

        assert required_fields.issubset(results[0].keys())
        assert results[0]["success"] is True


class TestConfiguration:
    """Tests for engine configuration."""

    def test_default_configuration(self):
        """Should have sensible defaults."""
        engine = ReviewEngine()

        assert engine.model_name == "mistral"
        assert engine.ollama_url == "http://localhost:11434"
        assert engine.max_concurrent == 3
        assert engine.max_retries == 3

    def test_custom_configuration(self):
        """Should accept custom configuration."""
        engine = ReviewEngine(
            model_name="llama2",
            ollama_url="http://custom:8080",
            max_concurrent=5,
            max_retries=5,
        )

        assert engine.model_name == "llama2"
        assert engine.ollama_url == "http://custom:8080"
        assert engine.max_concurrent == 5
        assert engine.max_retries == 5

    def test_labels_loaded(self):
        """Should load labels from config."""
        engine = ReviewEngine()

        # Should have loaded labels
        assert "labels" in engine.labels
        assert len(engine.labels["labels"]) > 0


class TestPromptBuilding:
    """Tests for prompt construction."""

    def test_prompt_contains_input(self):
        """Prompt should contain the input text and label."""
        engine = ReviewEngine()
        prompt = engine._build_prompt(
            text="Delete my data",
            pred_label="Access Request",
            confidence=0.85
        )

        assert "Delete my data" in prompt
        assert "Access Request" in prompt
        assert "0.85" in prompt

    def test_prompt_contains_instructions(self):
        """Prompt should contain response format instructions."""
        engine = ReviewEngine()
        prompt = engine._build_prompt(
            text="Test",
            pred_label="Test",
            confidence=0.5
        )

        assert "VERDICT:" in prompt
        assert "REASONING:" in prompt
        assert "SUGGESTED_LABEL:" in prompt
        assert "EXPLANATION:" in prompt

    def test_prompt_contains_labels(self):
        """Prompt should contain available labels."""
        engine = ReviewEngine()
        prompt = engine._build_prompt(
            text="Test",
            pred_label="Test",
            confidence=0.5
        )

        # Should contain at least some of the GDPR labels
        assert "Access Request" in prompt or "Erasure" in prompt
