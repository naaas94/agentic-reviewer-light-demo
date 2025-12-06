"""
Pytest configuration and shared fixtures.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_results():
    """Sample review results for testing."""
    return [
        {
            "sample_id": "demo_42_0001",
            "text": "What information do you have about me?",
            "pred_label": "Access Request",
            "confidence": 0.92,
            "ground_truth": "Access Request",
            "verdict": "Correct",
            "reasoning": "The text clearly requests data access.",
            "suggested_label": None,
            "explanation": "User wants to see their personal data.",
            "success": True,
        },
        {
            "sample_id": "demo_42_0002",
            "text": "Delete my data permanently.",
            "pred_label": "Access Request",
            "confidence": 0.65,
            "ground_truth": "Erasure",
            "verdict": "Incorrect",
            "reasoning": "The text requests deletion, not access.",
            "suggested_label": "Erasure",
            "explanation": "User wants their data removed.",
            "success": True,
        },
        {
            "sample_id": "demo_42_0003",
            "text": "Can I transfer my data?",
            "pred_label": "Portability",
            "confidence": 0.70,
            "ground_truth": "Portability",
            "verdict": "Uncertain",
            "reasoning": "Could be portability or access.",
            "suggested_label": None,
            "explanation": "Ambiguous request.",
            "success": True,
        },
    ]


@pytest.fixture
def sample_synthetic_samples():
    """Sample synthetic data for testing."""
    return [
        {
            "id": "demo_42_0001",
            "text": "What information do you have about me?",
            "pred_label": "Access Request",
            "confidence": 0.92,
            "ground_truth": "Access Request",
            "is_misclassified": False,
        },
        {
            "id": "demo_42_0002",
            "text": "Delete my data permanently.",
            "pred_label": "Access Request",
            "confidence": 0.65,
            "ground_truth": "Erasure",
            "is_misclassified": True,
        },
    ]
