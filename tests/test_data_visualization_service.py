"""Tests for the DataVisualizationService."""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from services.data_visualization_service import DataVisualizationService


class TestDataVisualizationService(unittest.TestCase):
    """Test cases for the DataVisualizationService."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_ads_client = Mock()
        self.service = DataVisualizationService(
            ads_client=self.mock_ads_client, config={"test_config": True}
        )

        # Create test data
        self.test_data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", "2024-01-31"),
                "clicks": np.random.randint(100, 1000, 31),
                "impressions": np.random.randint(1000, 10000, 31),
                "cost": np.random.uniform(100, 1000, 31),
                "conversions": np.random.randint(1, 100, 31),
            }
        )

    def tearDown(self):
        """Clean up after tests."""
        plt.close("all")

    def test_create_time_series_plot(self):
        """Test creating a time series plot."""
        # Test with single metric
        fig = self.service.create_time_series_plot(
            data=self.test_data, metrics=["clicks"], title="Click Performance"
        )
        self.assertIsNotNone(fig)

        # Test with multiple metrics
        fig = self.service.create_time_series_plot(
            data=self.test_data, metrics=["clicks", "impressions"], title="Performance Metrics"
        )
        self.assertIsNotNone(fig)

    def test_create_scatter_plot(self):
        """Test creating a scatter plot."""
        fig = self.service.create_scatter_plot(
            data=self.test_data,
            x_metric="impressions",
            y_metric="clicks",
            title="Clicks vs Impressions",
        )
        self.assertIsNotNone(fig)

    def test_create_bar_chart(self):
        """Test creating a bar chart."""
        # Group data by week
        weekly_data = self.test_data.set_index("date").resample("W").sum()

        fig = self.service.create_bar_chart(
            data=weekly_data, metric="clicks", title="Weekly Clicks"
        )
        self.assertIsNotNone(fig)

    def test_create_heatmap(self):
        """Test creating a heatmap."""
        # Create correlation matrix
        corr_matrix = self.test_data.drop("date", axis=1).corr()

        fig = self.service.create_heatmap(data=corr_matrix, title="Metric Correlations")
        self.assertIsNotNone(fig)

    def test_create_pie_chart(self):
        """Test creating a pie chart."""
        # Create campaign cost distribution
        cost_data = {"Campaign A": 1000, "Campaign B": 2000, "Campaign C": 1500}

        fig = self.service.create_pie_chart(data=cost_data, title="Campaign Cost Distribution")
        self.assertIsNotNone(fig)

    def test_create_box_plot(self):
        """Test creating a box plot."""
        fig = self.service.create_box_plot(
            data=self.test_data, metrics=["clicks", "conversions"], title="Metric Distribution"
        )
        self.assertIsNotNone(fig)

    def test_create_histogram(self):
        """Test creating a histogram."""
        fig = self.service.create_histogram(
            data=self.test_data["clicks"], title="Click Distribution", bins=20
        )
        self.assertIsNotNone(fig)

    def test_create_funnel_chart(self):
        """Test creating a funnel chart."""
        funnel_data = {
            "Impressions": 10000,
            "Clicks": 1000,
            "Conversions": 100,
            "Repeat Customers": 20,
        }

        fig = self.service.create_funnel_chart(data=funnel_data, title="Conversion Funnel")
        self.assertIsNotNone(fig)

    def test_create_performance_dashboard(self):
        """Test creating a performance dashboard."""
        dashboard = self.service.create_performance_dashboard(
            data=self.test_data, title="Campaign Performance Dashboard"
        )
        self.assertIsInstance(dashboard, dict)
        self.assertIn("time_series", dashboard)
        self.assertIn("metrics", dashboard)
        self.assertIn("correlations", dashboard)

    def test_error_handling(self):
        """Test error handling in visualization creation."""
        # Test with invalid metric
        with self.assertRaises(ValueError):
            self.service.create_time_series_plot(
                data=self.test_data, metrics=["invalid_metric"], title="Invalid Plot"
            )

        # Test with invalid data
        with self.assertRaises(ValueError):
            self.service.create_scatter_plot(
                data=None, x_metric="clicks", y_metric="impressions", title="Invalid Plot"
            )

    def test_style_customization(self):
        """Test customizing plot styles."""
        # Test with custom style parameters
        fig = self.service.create_time_series_plot(
            data=self.test_data,
            metrics=["clicks"],
            title="Custom Style Plot",
            style={"figsize": (12, 6), "color": "red", "grid": True, "fontsize": 12},
        )
        self.assertIsNotNone(fig)

    def test_save_plot(self):
        """Test saving plots to file."""
        with patch("matplotlib.pyplot.savefig") as mock_savefig:
            fig = self.service.create_time_series_plot(
                data=self.test_data, metrics=["clicks"], title="Test Plot"
            )
            self.service.save_plot(fig, "test_plot.png")
            mock_savefig.assert_called_once()


if __name__ == "__main__":
    unittest.main()
