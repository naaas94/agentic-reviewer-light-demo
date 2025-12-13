"""
Unit tests for the ReportGenerator.

Tests cover:
- Report generation with various inputs
- Statistics calculation
- Formatting correctness
"""

from core.report_generator import ReportGenerator


class TestReportGeneration:
    """Tests for report generation."""

    def test_generate_report_returns_string(self):
        """Should return a non-empty string."""
        gen = ReportGenerator()
        results = [
            {"verdict": "Correct", "pred_label": "Access Request", "text": "Test"},
        ]

        report = gen.generate_report(results, "test_run", {"seed": 42})

        assert isinstance(report, str)
        assert len(report) > 0

    def test_report_contains_run_id(self):
        """Report should contain the run ID."""
        gen = ReportGenerator()
        results = [{"verdict": "Correct", "pred_label": "Test", "text": "Test"}]

        report = gen.generate_report(results, "my_unique_run_123", {})

        assert "my_unique_run_123" in report

    def test_report_contains_statistics(self):
        """Report should contain correct/incorrect counts."""
        gen = ReportGenerator()
        results = [
            {"verdict": "Correct", "pred_label": "A", "text": "Test"},
            {"verdict": "Correct", "pred_label": "B", "text": "Test"},
            {"verdict": "Incorrect", "pred_label": "C", "text": "Test", "suggested_label": "D", "reasoning": "Wrong"},
        ]

        report = gen.generate_report(results, "test", {})

        assert "2" in report  # 2 correct
        assert "1" in report  # 1 incorrect

    def test_report_contains_corrections(self):
        """Report should contain correction details."""
        gen = ReportGenerator()
        results = [
            {
                "verdict": "Incorrect",
                "pred_label": "Access Request",
                "suggested_label": "Erasure",
                "text": "Delete my data please",
                "reasoning": "Text requests deletion",
            },
        ]

        report = gen.generate_report(results, "test", {})

        assert "Erasure" in report
        assert "Access Request" in report

    def test_report_is_valid_markdown(self):
        """Report should contain valid markdown structure."""
        gen = ReportGenerator()
        results = [{"verdict": "Correct", "pred_label": "Test", "text": "Test"}]

        report = gen.generate_report(results, "test", {})

        # Should have headers
        assert "# " in report
        assert "## " in report

        # Should have table
        assert "|" in report


class TestStatisticsCalculation:
    """Tests for statistics calculation."""

    def test_calculate_stats_empty(self):
        """Should handle empty results."""
        gen = ReportGenerator()
        stats = gen._calculate_stats([])

        assert stats["total"] == 0
        assert stats["correct"] == 0
        assert stats["incorrect"] == 0
        assert stats["uncertain"] == 0

    def test_calculate_stats_all_correct(self):
        """Should calculate 100% correct."""
        gen = ReportGenerator()
        results = [
            {"verdict": "Correct", "pred_label": "A"},
            {"verdict": "Correct", "pred_label": "B"},
        ]

        stats = gen._calculate_stats(results)

        assert stats["total"] == 2
        assert stats["correct"] == 2
        assert stats["correct_pct"] == 100.0

    def test_calculate_stats_mixed(self):
        """Should calculate mixed results correctly."""
        gen = ReportGenerator()
        results = [
            {"verdict": "Correct", "pred_label": "A"},
            {"verdict": "Incorrect", "pred_label": "B", "suggested_label": "C", "reasoning": "Wrong", "text": "Test"},
            {"verdict": "Uncertain", "pred_label": "D"},
        ]

        stats = gen._calculate_stats(results)

        assert stats["total"] == 3
        assert stats["correct"] == 1
        assert stats["incorrect"] == 1
        assert stats["uncertain"] == 1
        assert abs(stats["correct_pct"] - 33.33) < 1

    def test_calculate_stats_label_distribution(self):
        """Should calculate label distribution."""
        gen = ReportGenerator()
        results = [
            {"verdict": "Correct", "pred_label": "Access Request"},
            {"verdict": "Correct", "pred_label": "Access Request"},
            {"verdict": "Incorrect", "pred_label": "Erasure", "suggested_label": "X", "reasoning": "Y", "text": "Z"},
        ]

        stats = gen._calculate_stats(results)

        assert stats["label_distribution"]["Access Request"] == 2
        assert stats["label_distribution"]["Erasure"] == 1

    def test_calculate_stats_limits_corrections(self):
        """Should limit corrections to 5."""
        gen = ReportGenerator()
        results = [
            {"verdict": "Incorrect", "pred_label": f"Label{i}", "suggested_label": "X", "reasoning": "Y", "text": f"Text{i}"}
            for i in range(10)
        ]

        stats = gen._calculate_stats(results)

        assert len(stats["corrections"]) == 5


class TestFormatting:
    """Tests for report formatting helpers."""

    def test_format_label_dist(self):
        """Should format label distribution as table rows."""
        gen = ReportGenerator()
        dist = {"Access Request": 5, "Erasure": 3}

        formatted = gen._format_label_dist(dist)

        assert "Access Request" in formatted
        assert "5" in formatted
        assert "Erasure" in formatted
        assert "|" in formatted

    def test_format_corrections_empty(self):
        """Should handle empty corrections."""
        gen = ReportGenerator()
        formatted = gen._format_corrections([])

        assert "No corrections needed" in formatted

    def test_format_corrections_with_data(self):
        """Should format corrections with details."""
        gen = ReportGenerator()
        corrections = [
            {
                "text": "Delete my data",
                "predicted": "Access Request",
                "suggested": "Erasure",
                "reasoning": "Text requests deletion",
            }
        ]

        formatted = gen._format_corrections(corrections)

        assert "Delete my data" in formatted
        assert "Access Request" in formatted
        assert "Erasure" in formatted
