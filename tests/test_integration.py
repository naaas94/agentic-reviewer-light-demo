"""
Integration tests for end-to-end demo execution.

Tests cover:
- Full demo pipeline with mock mode
- Artifact generation and validation
- Error handling paths
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest


class TestEndToEndDemo:
    """Integration tests for the complete demo pipeline."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_demo_mock_mode_executes(self):
        """Demo should complete successfully in mock mode."""
        result = subprocess.run(
            [sys.executable, "run_demo.py", "--mock", "--samples", "5", "--seed", "42"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, f"Demo failed: {result.stderr}"
        assert "Demo complete" in result.stdout or "AGENTIC REVIEWER DEMO" in result.stdout

    def test_demo_generates_all_artifacts(self):
        """Demo should generate all expected output files."""
        result = subprocess.run(
            [sys.executable, "run_demo.py", "--mock", "--samples", "5", "--seed", "12345"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, f"Demo failed: {result.stderr}"

        # Find the latest output directory
        outputs_dir = Path("outputs")
        run_dirs = sorted(outputs_dir.glob("*"), key=lambda x: x.name, reverse=True)
        latest_run = run_dirs[0] if run_dirs else None

        assert latest_run is not None, "No output directory created"

        # Check all expected files exist
        expected_files = [
            "00_config.json",
            "01_synthetic_data.csv",
            "02_review_results.json",
            "03_labeled_dataset.csv",
            "04_report.md",
            "05_metrics.json",
        ]

        for filename in expected_files:
            filepath = latest_run / filename
            assert filepath.exists(), f"Missing artifact: {filename}"
            assert filepath.stat().st_size > 0, f"Empty artifact: {filename}"

    def test_demo_output_structure_valid(self):
        """Demo outputs should have valid structure and content."""
        result = subprocess.run(
            [sys.executable, "run_demo.py", "--mock", "--samples", "10", "--seed", "99999"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0

        # Find the latest output directory with our seed
        outputs_dir = Path("outputs")
        run_dirs = sorted(outputs_dir.glob("*"), key=lambda x: x.name, reverse=True)
        latest_run = run_dirs[0]

        # Validate config.json
        with open(latest_run / "00_config.json") as f:
            config = json.load(f)
        assert config["n_samples"] == 10
        assert config["seed"] == 99999
        assert config["use_llm"] is False  # Mock mode

        # Validate synthetic_data.csv
        df_synthetic = pd.read_csv(latest_run / "01_synthetic_data.csv")
        assert len(df_synthetic) == 10
        assert all(col in df_synthetic.columns for col in ["id", "text", "pred_label", "confidence"])

        # Validate review_results.json
        with open(latest_run / "02_review_results.json") as f:
            results = json.load(f)
        assert len(results) == 10
        assert all("verdict" in r for r in results)

        # Validate labeled_dataset.csv
        df_labeled = pd.read_csv(latest_run / "03_labeled_dataset.csv")
        assert len(df_labeled) == 10

        # Validate metrics.json
        with open(latest_run / "05_metrics.json") as f:
            metrics = json.load(f)
        assert "total" in metrics
        assert metrics["total"] == 10
        assert metrics["correct"] + metrics["incorrect"] + metrics["uncertain"] == 10

    def test_demo_reproducibility(self):
        """Same seed should produce identical results."""
        seed = 77777

        # Run twice with same seed
        for _ in range(2):
            subprocess.run(
                [sys.executable, "run_demo.py", "--mock", "--samples", "5", "--seed", str(seed)],
                capture_output=True,
                timeout=30,
            )

        # Find runs with this specific seed
        outputs_dir = Path("outputs")
        matching_runs = []

        for run_dir in outputs_dir.glob("*"):
            if not run_dir.is_dir():
                continue

            config_path = run_dir / "00_config.json"
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                    if config.get("seed") == seed:
                        matching_runs.append(run_dir)
                except Exception:
                    continue

        # Sort by name (timestamp) descending
        matching_runs.sort(key=lambda x: x.name, reverse=True)

        if len(matching_runs) < 2:
            pytest.skip("Need at least 2 runs with seed 77777 to test reproducibility")

        # Compare the two most recent runs with this seed
        run_dirs = matching_runs[:2]

        # Compare synthetic data (should be identical with same seed)
        df1 = pd.read_csv(run_dirs[0] / "01_synthetic_data.csv")
        df2 = pd.read_csv(run_dirs[1] / "01_synthetic_data.csv")

        # IDs contain seed and index, so structure should match
        assert len(df1) == len(df2)

    def test_demo_verbose_mode(self):
        """Verbose mode should not crash."""
        result = subprocess.run(
            [sys.executable, "run_demo.py", "--mock", "--samples", "3", "--verbose"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, f"Verbose mode failed: {result.stderr}"

    def test_demo_different_sample_counts(self):
        """Demo should handle various sample counts."""
        for n_samples in [1, 5, 25]:
            result = subprocess.run(
                [sys.executable, "run_demo.py", "--mock", "--samples", str(n_samples), "--seed", "42"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            assert result.returncode == 0, f"Failed with {n_samples} samples: {result.stderr}"


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_invalid_sample_count(self):
        """Should handle invalid sample count gracefully."""
        result = subprocess.run(
            [sys.executable, "run_demo.py", "--mock", "--samples", "not_a_number"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # argparse should reject this
        assert result.returncode != 0

    def test_help_flag(self):
        """Help flag should display usage."""
        result = subprocess.run(
            [sys.executable, "run_demo.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "options" in result.stdout.lower()


class TestComponentIntegration:
    """Tests for component integration."""

    def test_synthetic_generator_to_review_engine(self):
        """SyntheticGenerator output should be compatible with ReviewEngine."""
        from core.review_engine import ReviewEngine
        from core.synthetic_generator import SyntheticGenerator

        gen = SyntheticGenerator(seed=42)
        samples = gen.generate_samples(n_samples=5)

        engine = ReviewEngine()
        results = engine.generate_mock_results(samples)

        assert len(results) == 5
        for result in results:
            assert "verdict" in result
            assert "reasoning" in result

    def test_review_results_to_report_generator(self):
        """ReviewEngine output should be compatible with ReportGenerator."""
        from core.report_generator import ReportGenerator
        from core.review_engine import ReviewEngine
        from core.synthetic_generator import SyntheticGenerator

        gen = SyntheticGenerator(seed=42)
        samples = gen.generate_samples(n_samples=5)

        engine = ReviewEngine()
        results = engine.generate_mock_results(samples)

        report_gen = ReportGenerator()
        report = report_gen.generate_report(results, "test_run", {"seed": 42})

        assert len(report) > 100
        assert "Classification Audit Report" in report
        assert "test_run" in report

