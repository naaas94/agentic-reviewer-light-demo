"""
Synthetic data generator for GDPR/CCPA classification demo.
Generates realistic data subject requests with intentional misclassifications.

Features:
- Configurable confusion patterns (static or dynamic)
- Extensible templates via configuration
- Semantic similarity-based confusion generation
"""

import random
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple


class SyntheticGenerator:
    """Generates synthetic classification data for demonstration.

    Supports both static confusion patterns (for reproducible demos) and
    dynamic semantic similarity-based confusion (for realistic edge cases).
    """

    TEMPLATES = {
        "Access Request": [
            "What information do you have about me?",
            "Send me a copy of my data.",
            "I want to see what data you have stored about me.",
            "Please provide a copy of all data you hold on me.",
            "Under GDPR, I request access to my data.",
            "Can I get a report of all personal data you've collected on me?",
            "I'd like to review the information in my profile.",
        ],
        "Erasure": [
            "Delete my data permanently.",
            "I want to be removed from your system.",
            "Please erase all my personal information.",
            "I invoke my right to be forgotten.",
            "Wipe my data from your systems completely.",
            "Remove my account and all associated data.",
            "I no longer want you to have my information - delete everything.",
        ],
        "Rectification": [
            "My email address has changed, please update it.",
            "The information you have about me is incorrect.",
            "Please correct my personal details.",
            "My name is spelled wrong in your system.",
            "I moved, please update my address.",
            "The phone number on file is outdated.",
        ],
        "Portability": [
            "I want to export my data to another service.",
            "Give me my data in a machine-readable format.",
            "I need a portable copy of my information.",
            "Transfer my data to a competitor.",
            "Provide my data as a downloadable file.",
        ],
        "Objection": [
            "I don't want you to use my data for marketing.",
            "Stop processing my data for profiling.",
            "I object to automated decision making about me.",
            "Don't use my data for personalized ads.",
            "I opt out of having my data analyzed.",
        ],
        "Complaint": [
            "I want to file a complaint about how you handle my data.",
            "This is a formal complaint about data protection.",
            "I'm reporting a data protection violation.",
            "Your data practices violate my rights.",
            "I need to escalate a privacy concern.",
        ],
        "General Inquiry": [
            "How do you protect my data?",
            "What is your privacy policy?",
            "How long do you keep my information?",
            "Who has access to my data?",
            "Are you GDPR compliant?",
        ],
    }

    # Default confusion patterns: (actual_label, confused_with_label)
    # These represent common real-world classification errors
    DEFAULT_CONFUSION_PATTERNS: List[Tuple[str, str]] = [
        ("Erasure", "Access Request"),       # Both about "my data"
        ("Portability", "Access Request"),   # Both request data copies
        ("Objection", "Erasure"),            # Both about stopping data use
        ("Rectification", "General Inquiry"), # Both about data accuracy
        ("Complaint", "Objection"),          # Both express dissatisfaction
    ]

    # Semantic similarity scores for dynamic confusion (0-1, higher = more confusable)
    # Based on linguistic/semantic overlap between categories
    SEMANTIC_SIMILARITY: Dict[str, Dict[str, float]] = {
        "Access Request": {
            "Portability": 0.7,     # Both involve getting data
            "General Inquiry": 0.4,
            "Erasure": 0.3,
        },
        "Erasure": {
            "Objection": 0.6,       # Both involve stopping data use
            "Portability": 0.4,     # Both involve "removing" data
            "Access Request": 0.3,
        },
        "Rectification": {
            "General Inquiry": 0.5,  # Both can ask about data details
            "Access Request": 0.3,
        },
        "Portability": {
            "Access Request": 0.7,   # Both involve data copies
            "Erasure": 0.4,
        },
        "Objection": {
            "Erasure": 0.6,         # Both about limiting data use
            "Complaint": 0.5,        # Both express dissatisfaction
        },
        "Complaint": {
            "Objection": 0.5,
            "General Inquiry": 0.3,
        },
        "General Inquiry": {
            "Access Request": 0.4,
            "Rectification": 0.3,
        },
    }

    def __init__(
        self,
        seed: Optional[int] = None,
        confusion_patterns: Optional[List[Tuple[str, str]]] = None,
        use_dynamic_confusion: bool = False,
        custom_templates: Optional[Dict[str, List[str]]] = None,
    ):
        """Initialize the synthetic generator.

        Args:
            seed: Random seed for reproducibility
            confusion_patterns: Custom confusion patterns (overrides defaults)
            use_dynamic_confusion: Use semantic similarity for dynamic confusion
            custom_templates: Additional templates to merge with defaults
        """
        self.seed = seed or int(datetime.now().timestamp())
        random.seed(self.seed)

        # Merge custom templates if provided
        self.templates = self.TEMPLATES.copy()
        if custom_templates:
            for label, texts in custom_templates.items():
                if label in self.templates:
                    self.templates[label] = self.templates[label] + texts
                else:
                    self.templates[label] = texts

        self.labels = list(self.templates.keys())
        self.confusion_patterns = confusion_patterns or self.DEFAULT_CONFUSION_PATTERNS
        self.use_dynamic_confusion = use_dynamic_confusion

    def generate_samples(
        self,
        n_samples: int = 15,
        misclassification_rate: float = 0.35
    ) -> List[Dict[str, Any]]:
        """Generate synthetic samples with realistic distribution."""
        samples = []

        for i in range(n_samples):
            is_misclassified = random.random() < misclassification_rate
            actual_label = random.choice(self.labels)
            text = random.choice(self.templates[actual_label])

            # Add variation
            text = self._add_variation(text)

            # Determine predicted label
            if is_misclassified:
                pred_label = self._get_confused_label(actual_label)
                confidence = round(random.uniform(0.55, 0.82), 2)
            else:
                pred_label = actual_label
                confidence = round(random.uniform(0.78, 0.98), 2)

            samples.append({
                "id": f"demo_{self.seed}_{i:04d}",
                "text": text,
                "pred_label": pred_label,
                "confidence": confidence,
                "ground_truth": actual_label,
                "is_misclassified": is_misclassified,
            })

        return samples

    def _get_confused_label(self, actual: str) -> str:
        """Get a confused label for misclassification.

        Uses either static patterns or dynamic semantic similarity.
        """
        if self.use_dynamic_confusion:
            return self._get_dynamic_confused_label(actual)
        else:
            return self._get_static_confused_label(actual)

    def _get_static_confused_label(self, actual: str) -> str:
        """Get confused label from static patterns."""
        for a, confused in self.confusion_patterns:
            if a == actual:
                return confused
        return random.choice([label for label in self.labels if label != actual])

    def _get_dynamic_confused_label(self, actual: str) -> str:
        """Get confused label based on semantic similarity scores.

        Labels with higher similarity scores are more likely to be chosen.
        """
        similarities = self.SEMANTIC_SIMILARITY.get(actual, {})

        if not similarities:
            # Fallback to random if no similarity data
            return random.choice([label for label in self.labels if label != actual])

        # Weight labels by similarity score
        candidates = []
        weights = []

        for label in self.labels:
            if label != actual:
                # Use similarity score as weight, default to small value
                weight = similarities.get(label, 0.1)
                candidates.append(label)
                weights.append(weight)

        # Normalize weights and select
        total_weight = sum(weights)
        normalized = [w / total_weight for w in weights]

        return random.choices(candidates, weights=normalized, k=1)[0]

    def _add_variation(self, text: str) -> str:
        """Add realistic variation to text."""
        variations: List[Callable[[str], str]] = [
            lambda t: t,
            lambda t: t.lower(),
            lambda t: "Hi, " + t[0].lower() + t[1:],
            lambda t: t + " Thanks.",
            lambda t: "Hello, " + t[0].lower() + t[1:] + " Best regards.",
            lambda t: "URGENT: " + t,
            lambda t: t.replace(".", "!"),
        ]
        selected = random.choice(variations)
        return selected(text)

    def get_confusion_matrix_preview(self) -> Dict[str, List[str]]:
        """Preview which labels can be confused with which.

        Useful for understanding the confusion patterns in effect.
        """
        matrix = {}
        for label in self.labels:
            confused_with = []
            if self.use_dynamic_confusion:
                similarities = self.SEMANTIC_SIMILARITY.get(label, {})
                # Sort by similarity score descending
                sorted_similar = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
                confused_with = [label for label, _ in sorted_similar[:3]]
            else:
                for a, c in self.confusion_patterns:
                    if a == label:
                        confused_with.append(c)
            matrix[label] = confused_with if confused_with else ["(random)"]
        return matrix

    def get_metadata(self) -> Dict[str, Any]:
        """Return generator metadata for reproducibility."""
        return {
            "seed": self.seed,
            "labels": self.labels,
            "use_dynamic_confusion": self.use_dynamic_confusion,
            "confusion_patterns": self.confusion_patterns if not self.use_dynamic_confusion else "dynamic",
        }

