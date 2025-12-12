"""
Unit tests for the SyntheticGenerator.

Tests cover:
- Sample generation with correct structure
- Reproducibility via seed
- Misclassification rate adherence
- Confusion pattern behavior (static and dynamic)
- Custom templates and configuration
"""

import pytest
from core.synthetic_generator import SyntheticGenerator


class TestSyntheticGenerator:
    """Tests for SyntheticGenerator class."""
    
    def test_generate_samples_returns_correct_count(self):
        """Should generate the requested number of samples."""
        gen = SyntheticGenerator(seed=42)
        samples = gen.generate_samples(n_samples=10)
        assert len(samples) == 10
    
    def test_generate_samples_structure(self):
        """Each sample should have all required fields."""
        gen = SyntheticGenerator(seed=42)
        samples = gen.generate_samples(n_samples=5)
        
        required_fields = {"id", "text", "pred_label", "confidence", "ground_truth", "is_misclassified"}
        
        for sample in samples:
            assert required_fields.issubset(sample.keys()), f"Missing fields in sample: {sample}"
            assert isinstance(sample["id"], str)
            assert isinstance(sample["text"], str)
            assert len(sample["text"]) > 0
            assert isinstance(sample["confidence"], float)
            assert 0 <= sample["confidence"] <= 1
    
    def test_seed_reproducibility(self):
        """Same seed should produce identical samples."""
        # Create first generator, generate samples
        gen1 = SyntheticGenerator(seed=12345)
        samples1 = gen1.generate_samples(n_samples=10)
        
        # Create second generator with same seed (this reseeds random)
        gen2 = SyntheticGenerator(seed=12345)
        samples2 = gen2.generate_samples(n_samples=10)
        
        assert samples1 == samples2
    
    def test_different_seeds_produce_different_samples(self):
        """Different seeds should produce different samples."""
        gen1 = SyntheticGenerator(seed=111)
        gen2 = SyntheticGenerator(seed=222)
        
        samples1 = gen1.generate_samples(n_samples=10)
        samples2 = gen2.generate_samples(n_samples=10)
        
        # At least some samples should differ
        texts1 = [s["text"] for s in samples1]
        texts2 = [s["text"] for s in samples2]
        assert texts1 != texts2
    
    def test_misclassification_rate_zero(self):
        """With 0% misclassification rate, all predictions should be correct."""
        gen = SyntheticGenerator(seed=42)
        samples = gen.generate_samples(n_samples=20, misclassification_rate=0.0)
        
        for sample in samples:
            assert sample["pred_label"] == sample["ground_truth"]
            assert sample["is_misclassified"] is False
    
    def test_misclassification_rate_one(self):
        """With 100% misclassification rate, all predictions should be wrong."""
        gen = SyntheticGenerator(seed=42)
        samples = gen.generate_samples(n_samples=20, misclassification_rate=1.0)
        
        for sample in samples:
            assert sample["pred_label"] != sample["ground_truth"]
            assert sample["is_misclassified"] is True
    
    def test_confidence_ranges(self):
        """Confidence should be in expected ranges based on classification correctness."""
        gen = SyntheticGenerator(seed=42)
        samples = gen.generate_samples(n_samples=50, misclassification_rate=0.5)
        
        for sample in samples:
            if sample["is_misclassified"]:
                # Misclassified samples should have lower confidence (0.55-0.82)
                assert 0.5 <= sample["confidence"] <= 0.85
            else:
                # Correct samples should have higher confidence (0.78-0.98)
                assert 0.75 <= sample["confidence"] <= 1.0
    
    def test_all_labels_are_valid(self):
        """All labels should be from the defined set."""
        gen = SyntheticGenerator(seed=42)
        samples = gen.generate_samples(n_samples=50)
        
        valid_labels = set(gen.labels)
        
        for sample in samples:
            assert sample["pred_label"] in valid_labels
            assert sample["ground_truth"] in valid_labels
    
    def test_sample_ids_are_unique(self):
        """All sample IDs should be unique."""
        gen = SyntheticGenerator(seed=42)
        samples = gen.generate_samples(n_samples=100)
        
        ids = [s["id"] for s in samples]
        assert len(ids) == len(set(ids)), "Duplicate IDs found"
    
    def test_get_metadata(self):
        """Metadata should contain required fields."""
        gen = SyntheticGenerator(seed=42)
        metadata = gen.get_metadata()
        
        assert "seed" in metadata
        assert "labels" in metadata
        assert metadata["seed"] == 42
        assert len(metadata["labels"]) > 0


class TestConfusionPatterns:
    """Tests for confusion pattern behavior."""
    
    def test_static_confusion_patterns(self):
        """Static confusion patterns should be followed."""
        gen = SyntheticGenerator(seed=42, use_dynamic_confusion=False)
        
        # Generate many misclassified samples
        samples = gen.generate_samples(n_samples=100, misclassification_rate=1.0)
        
        # Check that confusion follows defined patterns
        confusion_map = {}
        for a, c in gen.confusion_patterns:
            confusion_map[a] = c
        
        for sample in samples:
            actual = sample["ground_truth"]
            predicted = sample["pred_label"]
            
            if actual in confusion_map:
                # Should follow the defined pattern
                assert predicted == confusion_map[actual], \
                    f"Expected {actual} to confuse with {confusion_map[actual]}, got {predicted}"
    
    def test_dynamic_confusion_produces_variety(self):
        """Dynamic confusion should produce varied misclassifications."""
        gen = SyntheticGenerator(seed=42, use_dynamic_confusion=True)
        
        # Generate many samples for a specific label
        samples = gen.generate_samples(n_samples=200, misclassification_rate=1.0)
        
        # Group by ground truth
        by_ground_truth = {}
        for s in samples:
            gt = s["ground_truth"]
            if gt not in by_ground_truth:
                by_ground_truth[gt] = []
            by_ground_truth[gt].append(s["pred_label"])
        
        # At least some ground truths should have multiple different predictions
        has_variety = False
        for gt, predictions in by_ground_truth.items():
            if len(set(predictions)) > 1:
                has_variety = True
                break
        
        assert has_variety, "Dynamic confusion should produce variety in predictions"
    
    def test_custom_confusion_patterns(self):
        """Custom confusion patterns should override defaults."""
        custom_patterns = [
            ("Access Request", "Complaint"),
            ("Erasure", "Portability"),
        ]
        gen = SyntheticGenerator(seed=42, confusion_patterns=custom_patterns)
        
        # Verify patterns are set
        assert gen.confusion_patterns == custom_patterns
    
    def test_confusion_matrix_preview(self):
        """Preview should return confusion info for all labels."""
        gen = SyntheticGenerator(seed=42)
        preview = gen.get_confusion_matrix_preview()
        
        assert len(preview) == len(gen.labels)
        for label in gen.labels:
            assert label in preview
            assert isinstance(preview[label], list)


class TestCustomTemplates:
    """Tests for custom template functionality."""
    
    def test_custom_templates_merge(self):
        """Custom templates should be merged with defaults."""
        custom = {
            "Access Request": ["Custom access request text"],
            "New Category": ["Text for new category"],
        }
        gen = SyntheticGenerator(seed=42, custom_templates=custom)
        
        # Original Access Request templates should still exist + custom one
        assert "Custom access request text" in gen.templates["Access Request"]
        assert len(gen.templates["Access Request"]) > 1
        
        # New category should be added
        assert "New Category" in gen.templates
        assert "New Category" in gen.labels
    
    def test_text_variation(self):
        """Text variations should be applied."""
        gen = SyntheticGenerator(seed=42)
        
        # Generate many samples and check for variations
        samples = gen.generate_samples(n_samples=100)
        texts = [s["text"] for s in samples]
        
        # Should have some variety (lowercase, prefixes, etc.)
        has_lowercase = any(t[0].islower() for t in texts if t)
        has_hi = any(t.startswith("Hi,") for t in texts)
        has_thanks = any(t.endswith("Thanks.") for t in texts)
        
        # At least some variations should appear
        assert has_lowercase or has_hi or has_thanks


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_zero_samples(self):
        """Should handle zero samples gracefully."""
        gen = SyntheticGenerator(seed=42)
        samples = gen.generate_samples(n_samples=0)
        assert samples == []
    
    def test_single_sample(self):
        """Should handle single sample correctly."""
        gen = SyntheticGenerator(seed=42)
        samples = gen.generate_samples(n_samples=1)
        assert len(samples) == 1
    
    def test_large_sample_count(self):
        """Should handle large sample counts."""
        gen = SyntheticGenerator(seed=42)
        samples = gen.generate_samples(n_samples=1000)
        assert len(samples) == 1000
        
        # IDs should still be unique
        ids = [s["id"] for s in samples]
        assert len(ids) == len(set(ids))
