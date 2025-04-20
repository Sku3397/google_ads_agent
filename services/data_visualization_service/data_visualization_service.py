"""
Data Visualization Service for Google Ads Management System

This module provides data visualization capabilities to create charts, graphs,
and interactive dashboards for analyzing Google Ads performance data.
"""

from services.base_service import BaseService
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import pandas as pd
import os
import json
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend


class DataVisualizationService(BaseService):
    """
    Service for creating data visualizations of Google Ads performance data.
    """

    def __init__(
        self,
        ads_api=None,
        optimizer=None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the DataVisualizationService.

        Args:
            ads_api: Google Ads API client instance
            optimizer: AI optimizer instance for generating suggestions
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(ads_api, optimizer, config, logger)
        self.logger.info("DataVisualizationService initialized.")

        # Create directories for outputs
        self.charts_dir = os.path.join("reports", "charts")
        os.makedirs(self.charts_dir, exist_ok=True)

        # Set default chart style
        plt.style.use("seaborn-v0_8-whitegrid")

        # Default colors
        self.colors = ["#4285F4", "#34A853", "#FBBC05", "#EA4335", "#8F00FF", "#00ACC1"]

        # Default sizes
        self.default_figsize = (10, 6)
        self.default_dpi = 100

    def create_campaign_performance_chart(
        self,
        campaign_data: List[Dict[str, Any]],
        metric: str = "clicks",
        days: int = 30,
        campaign_ids: Optional[List[str]] = None,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a line chart showing campaign performance over time.

        Args:
            campaign_data: List of campaign performance data dictionaries
            metric: The metric to visualize (e.g., 'clicks', 'impressions', 'cost')
            days: Number of days to include in the chart
            campaign_ids: Optional list of campaign IDs to include (if None, include all)
            filename: Optional custom filename for the saved chart

        Returns:
            Dictionary with the path to the saved chart and metadata
        """
        start_time = datetime.now()
        metric_title = metric.replace("_", " ").title()
        self.logger.info(f"Creating {metric_title} performance chart for the last {days} days")

        if not campaign_data:
            self.logger.error("No campaign data provided for visualization")
            return {"status": "error", "message": "No data provided"}

        try:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(campaign_data)

            # Filter by campaign IDs if specified
            if campaign_ids:
                df = df[df["campaign_id"].isin(campaign_ids)]

            if df.empty:
                self.logger.warning("No campaigns found after filtering")
                return {"status": "error", "message": "No campaigns found for visualization"}

            # Ensure 'date' column exists and is datetime type
            if "date" not in df.columns:
                self.logger.error("Campaign data missing 'date' column")
                return {"status": "error", "message": "Campaign data missing 'date' column"}

            df["date"] = pd.to_datetime(df["date"])

            # Filter to specified date range
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            df = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)]

            # Group by date and campaign, then sum the metric
            if "campaign_name" not in df.columns:
                df["campaign_name"] = df["campaign_id"].astype(str)

            pivot_df = df.pivot_table(
                index="date", columns="campaign_name", values=metric, aggfunc="sum"
            )

            # Create the figure
            fig, ax = plt.subplots(figsize=self.default_figsize, dpi=self.default_dpi)

            # Plot each campaign
            for i, campaign in enumerate(pivot_df.columns):
                color = self.colors[i % len(self.colors)]
                pivot_df[campaign].plot(
                    ax=ax, label=campaign, linewidth=2, color=color, marker="o", markersize=4
                )

            # Set chart title and labels
            ax.set_title(f"{metric_title} by Campaign (Last {days} Days)", fontsize=16, pad=20)
            ax.set_xlabel("Date", fontsize=12, labelpad=10)
            ax.set_ylabel(metric_title, fontsize=12, labelpad=10)

            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

            # Format y-axis based on metric type
            if metric == "cost":
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x:,.2f}"))
            elif metric in ["clicks", "impressions"]:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.0f}"))
            elif metric in ["ctr", "conv_rate"]:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.1%}"))

            # Add legend
            ax.legend(title="Campaigns", loc="upper left", bbox_to_anchor=(1, 1))

            # Add grid
            ax.grid(True, linestyle="--", alpha=0.7)

            # Set tight layout
            plt.tight_layout()

            # Save the chart
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"campaign_{metric}_{timestamp}.png"

            chart_path = os.path.join(self.charts_dir, filename)
            plt.savefig(chart_path, bbox_inches="tight")
            plt.close(fig)

            # Return result
            result = {
                "status": "success",
                "chart_path": chart_path,
                "metric": metric,
                "days": days,
                "campaign_count": len(pivot_df.columns),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            }

            self.logger.info(f"Created {metric_title} chart at {chart_path}")
            self._track_execution(start_time, success=True)
            return result

        except Exception as e:
            self.logger.error(f"Error creating campaign performance chart: {str(e)}")
            self._track_execution(start_time, success=False)
            return {"status": "error", "message": str(e)}

    def create_keyword_performance_chart(
        self,
        keyword_data: List[Dict[str, Any]],
        metric: str = "clicks",
        top_n: int = 10,
        campaign_id: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a bar chart showing top keyword performance.

        Args:
            keyword_data: List of keyword performance data dictionaries
            metric: The metric to visualize (e.g., 'clicks', 'impressions', 'cost')
            top_n: Number of top keywords to include
            campaign_id: Optional campaign ID to filter data
            filename: Optional custom filename for the saved chart

        Returns:
            Dictionary with the path to the saved chart and metadata
        """
        start_time = datetime.now()
        metric_title = metric.replace("_", " ").title()
        self.logger.info(f"Creating top {top_n} keywords by {metric_title} chart")

        if not keyword_data:
            self.logger.error("No keyword data provided for visualization")
            return {"status": "error", "message": "No data provided"}

        try:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(keyword_data)

            # Filter by campaign ID if specified
            if campaign_id:
                df = df[df["campaign_id"] == campaign_id]
                campaign_name = df["campaign_name"].iloc[0] if not df.empty else "Unknown"

            if df.empty:
                self.logger.warning("No keywords found after filtering")
                return {"status": "error", "message": "No keywords found for visualization"}

            # Get top N keywords by metric
            if metric not in df.columns:
                self.logger.error(f"Keyword data missing '{metric}' column")
                return {"status": "error", "message": f"Keyword data missing '{metric}' column"}

            # Sort by metric and take top N
            df = df.sort_values(by=metric, ascending=False).head(top_n)

            # Combine keyword text and match type
            if "keyword_text" in df.columns and "match_type" in df.columns:
                df["keyword"] = df["keyword_text"] + " [" + df["match_type"].str.lower() + "]"
            elif "keyword_text" in df.columns:
                df["keyword"] = df["keyword_text"]
            else:
                self.logger.error("Keyword data missing 'keyword_text' column")
                return {"status": "error", "message": "Keyword data missing 'keyword_text' column"}

            # Create the figure
            fig, ax = plt.subplots(figsize=self.default_figsize, dpi=self.default_dpi)

            # Plot horizontal bar chart
            colors = [self.colors[i % len(self.colors)] for i in range(len(df))]
            bars = ax.barh(df["keyword"], df[metric], color=colors)

            # Add value labels to the bars
            for i, bar in enumerate(bars):
                value = df[metric].iloc[i]
                label_x = bar.get_width() * 1.01

                # Format the value based on metric type
                if metric == "cost":
                    value_text = f"${value:,.2f}"
                elif metric in ["clicks", "impressions"]:
                    value_text = f"{value:,.0f}"
                elif metric in ["ctr", "conv_rate"]:
                    value_text = f"{value:.1%}"
                else:
                    value_text = f"{value:g}"

                ax.text(
                    label_x,
                    bar.get_y() + bar.get_height() / 2,
                    value_text,
                    va="center",
                    fontsize=10,
                )

            # Set chart title and labels
            title = f"Top {top_n} Keywords by {metric_title}"
            if campaign_id:
                title += f" - {campaign_name}"

            ax.set_title(title, fontsize=16, pad=20)
            ax.set_xlabel(metric_title, fontsize=12, labelpad=10)
            ax.set_ylabel("Keyword", fontsize=12, labelpad=10)

            # Format x-axis based on metric type
            if metric == "cost":
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x:,.0f}"))
            elif metric in ["clicks", "impressions"]:
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.0f}"))
            elif metric in ["ctr", "conv_rate"]:
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.0%}"))

            # Add grid
            ax.grid(True, axis="x", linestyle="--", alpha=0.7)

            # Invert the y-axis so the top keyword is at the top
            ax.invert_yaxis()

            # Set tight layout
            plt.tight_layout()

            # Save the chart
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"top_keywords_{metric}_{timestamp}.png"

            chart_path = os.path.join(self.charts_dir, filename)
            plt.savefig(chart_path, bbox_inches="tight")
            plt.close(fig)

            # Return result
            result = {
                "status": "success",
                "chart_path": chart_path,
                "metric": metric,
                "top_n": top_n,
                "campaign_id": campaign_id,
                "keyword_count": len(df),
            }

            self.logger.info(f"Created top keywords by {metric_title} chart at {chart_path}")
            self._track_execution(start_time, success=True)
            return result

        except Exception as e:
            self.logger.error(f"Error creating keyword performance chart: {str(e)}")
            self._track_execution(start_time, success=False)
            return {"status": "error", "message": str(e)}

    def create_performance_summary_dashboard(
        self, campaign_data: List[Dict[str, Any]], days: int = 30, filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a comprehensive performance dashboard with multiple metrics.

        Args:
            campaign_data: List of campaign performance data dictionaries
            days: Number of days to include in the dashboard
            filename: Optional custom filename for the saved dashboard

        Returns:
            Dictionary with the path to the saved dashboard and metadata
        """
        start_time = datetime.now()
        self.logger.info(f"Creating performance summary dashboard for the last {days} days")

        if not campaign_data:
            self.logger.error("No campaign data provided for dashboard")
            return {"status": "error", "message": "No data provided"}

        try:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(campaign_data)

            if df.empty:
                self.logger.warning("No campaign data available for dashboard")
                return {"status": "error", "message": "No campaign data available"}

            # Ensure date column is datetime
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])

                # Filter to specified date range
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=days)
                df = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)]

            # Create a 2x2 dashboard
            fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=self.default_dpi)

            # 1. Daily Clicks and Impressions (dual y-axis)
            ax1 = axes[0, 0]
            if "date" in df.columns and "clicks" in df.columns and "impressions" in df.columns:
                daily_metrics = (
                    df.groupby("date").agg({"clicks": "sum", "impressions": "sum"}).reset_index()
                )

                # Plot clicks
                color1 = self.colors[0]
                lns1 = ax1.plot(
                    daily_metrics["date"],
                    daily_metrics["clicks"],
                    color=color1,
                    marker="o",
                    label="Clicks",
                )
                ax1.set_ylabel("Clicks", color=color1, fontsize=12)
                ax1.tick_params(axis="y", labelcolor=color1)

                # Create second y-axis for impressions
                ax1b = ax1.twinx()
                color2 = self.colors[1]
                lns2 = ax1b.plot(
                    daily_metrics["date"],
                    daily_metrics["impressions"],
                    color=color2,
                    marker="s",
                    label="Impressions",
                )
                ax1b.set_ylabel("Impressions", color=color2, fontsize=12)
                ax1b.tick_params(axis="y", labelcolor=color2)

                # Combine legends
                lns = lns1 + lns2
                labs = [l.get_label() for l in lns]
                ax1.legend(lns, labs, loc="upper left")

                # Format x-axis dates
                ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
                ax1.set_title("Daily Clicks and Impressions", fontsize=14)
                ax1.grid(True, linestyle="--", alpha=0.7)
            else:
                ax1.text(0.5, 0.5, "Data not available", ha="center", va="center", fontsize=14)
                ax1.set_title("Daily Clicks and Impressions", fontsize=14)

            # 2. Campaign Performance Comparison (bar chart)
            ax2 = axes[0, 1]
            if "campaign_name" in df.columns and "clicks" in df.columns:
                campaign_perf = (
                    df.groupby("campaign_name")
                    .agg({"clicks": "sum", "cost": "sum", "conversions": "sum"})
                    .reset_index()
                )

                # Calculate CPC and CVR
                campaign_perf["cpc"] = campaign_perf["cost"] / campaign_perf["clicks"].replace(0, 1)
                campaign_perf["cvr"] = campaign_perf["conversions"] / campaign_perf[
                    "clicks"
                ].replace(0, 1)

                # Sort by clicks
                campaign_perf = campaign_perf.sort_values("clicks", ascending=False)

                # Plot top 5 campaigns by clicks
                top_campaigns = campaign_perf.head(5)
                colors = [self.colors[i % len(self.colors)] for i in range(len(top_campaigns))]
                bars = ax2.bar(
                    top_campaigns["campaign_name"], top_campaigns["clicks"], color=colors
                )

                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 5,
                        f"{height:,.0f}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )

                ax2.set_title("Top 5 Campaigns by Clicks", fontsize=14)
                ax2.set_ylabel("Clicks", fontsize=12)
                ax2.set_xlabel("Campaign", fontsize=12)
                ax2.set_xticklabels(top_campaigns["campaign_name"], rotation=45, ha="right")
                ax2.grid(True, axis="y", linestyle="--", alpha=0.7)
            else:
                ax2.text(0.5, 0.5, "Data not available", ha="center", va="center", fontsize=14)
                ax2.set_title("Top 5 Campaigns by Clicks", fontsize=14)

            # 3. Cost and Conversions Over Time
            ax3 = axes[1, 0]
            if "date" in df.columns and "cost" in df.columns and "conversions" in df.columns:
                daily_metrics = (
                    df.groupby("date").agg({"cost": "sum", "conversions": "sum"}).reset_index()
                )

                # Plot cost
                color1 = self.colors[2]
                lns1 = ax3.plot(
                    daily_metrics["date"],
                    daily_metrics["cost"],
                    color=color1,
                    marker="o",
                    label="Cost",
                )
                ax3.set_ylabel("Cost ($)", color=color1, fontsize=12)
                ax3.tick_params(axis="y", labelcolor=color1)
                ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x:,.0f}"))

                # Create second y-axis for conversions
                ax3b = ax3.twinx()
                color2 = self.colors[3]
                lns2 = ax3b.plot(
                    daily_metrics["date"],
                    daily_metrics["conversions"],
                    color=color2,
                    marker="s",
                    label="Conversions",
                )
                ax3b.set_ylabel("Conversions", color=color2, fontsize=12)
                ax3b.tick_params(axis="y", labelcolor=color2)

                # Combine legends
                lns = lns1 + lns2
                labs = [l.get_label() for l in lns]
                ax3.legend(lns, labs, loc="upper left")

                # Format x-axis dates
                ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
                ax3.set_title("Daily Cost and Conversions", fontsize=14)
                ax3.grid(True, linestyle="--", alpha=0.7)
            else:
                ax3.text(0.5, 0.5, "Data not available", ha="center", va="center", fontsize=14)
                ax3.set_title("Daily Cost and Conversions", fontsize=14)

            # 4. Performance Metrics Summary (horizontal bar chart)
            ax4 = axes[1, 1]
            metrics_summary = {
                "Total Clicks": df["clicks"].sum() if "clicks" in df.columns else 0,
                "Total Impressions": df["impressions"].sum() if "impressions" in df.columns else 0,
                "Total Cost": df["cost"].sum() if "cost" in df.columns else 0,
                "Total Conversions": df["conversions"].sum() if "conversions" in df.columns else 0,
                "Avg. CTR": (
                    df["clicks"].sum() / df["impressions"].sum()
                    if "clicks" in df.columns
                    and "impressions" in df.columns
                    and df["impressions"].sum() > 0
                    else 0
                ),
                "Avg. CPC": (
                    df["cost"].sum() / df["clicks"].sum()
                    if "cost" in df.columns and "clicks" in df.columns and df["clicks"].sum() > 0
                    else 0
                ),
                "Conv. Rate": (
                    df["conversions"].sum() / df["clicks"].sum()
                    if "conversions" in df.columns
                    and "clicks" in df.columns
                    and df["clicks"].sum() > 0
                    else 0
                ),
                "Cost/Conv.": (
                    df["cost"].sum() / df["conversions"].sum()
                    if "cost" in df.columns
                    and "conversions" in df.columns
                    and df["conversions"].sum() > 0
                    else 0
                ),
            }

            # Create a text-based summary
            summary_text = "\n".join(
                [
                    f"Performance Summary (Last {days} Days)",
                    f"",
                    f"Total Clicks: {metrics_summary['Total Clicks']:,.0f}",
                    f"Total Impressions: {metrics_summary['Total Impressions']:,.0f}",
                    f"Total Cost: ${metrics_summary['Total Cost']:,.2f}",
                    f"Total Conversions: {metrics_summary['Total Conversions']:,.1f}",
                    f"",
                    f"Avg. CTR: {metrics_summary['Avg. CTR']:.2%}",
                    f"Avg. CPC: ${metrics_summary['Avg. CPC']:.2f}",
                    f"Conv. Rate: {metrics_summary['Conv. Rate']:.2%}",
                    f"Cost/Conv.: ${metrics_summary['Cost/Conv.']:.2f}",
                ]
            )

            ax4.text(
                0.5,
                0.5,
                summary_text,
                ha="center",
                va="center",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=1", facecolor="#f9f9f9", alpha=0.5),
            )
            ax4.set_title("Performance Metrics Summary", fontsize=14)
            ax4.axis("off")  # Hide the axes

            # Main dashboard title
            plt.suptitle(
                f"Google Ads Performance Dashboard - Last {days} Days", fontsize=18, y=0.98
            )

            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            # Save the dashboard
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"performance_dashboard_{timestamp}.png"

            dashboard_path = os.path.join(self.charts_dir, filename)
            plt.savefig(dashboard_path, bbox_inches="tight")
            plt.close(fig)

            # Return result
            result = {
                "status": "success",
                "dashboard_path": dashboard_path,
                "days": days,
                "metrics_summary": metrics_summary,
            }

            self.logger.info(f"Created performance dashboard at {dashboard_path}")
            self._track_execution(start_time, success=True)
            return result

        except Exception as e:
            self.logger.error(f"Error creating performance dashboard: {str(e)}")
            self._track_execution(start_time, success=False)
            return {"status": "error", "message": str(e)}

    def export_visualization_data(
        self,
        data: Union[List[Dict[str, Any]], pd.DataFrame],
        format: str = "csv",
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Export data used for visualizations in various formats.

        Args:
            data: Data to export (list of dictionaries or DataFrame)
            format: Format to export (csv, json, excel)
            filename: Optional custom filename

        Returns:
            Dictionary with the path to the exported file
        """
        start_time = datetime.now()
        self.logger.info(f"Exporting data in {format} format")

        # Convert to DataFrame if not already
        if not isinstance(data, pd.DataFrame):
            df = pd.DataFrame(data)
        else:
            df = data

        if df.empty:
            self.logger.error("No data provided for export")
            return {"status": "error", "message": "No data provided"}

        try:
            # Create the exports directory if it doesn't exist
            exports_dir = os.path.join("reports", "exports")
            os.makedirs(exports_dir, exist_ok=True)

            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"data_export_{timestamp}"

            # Export in the specified format
            if format.lower() == "csv":
                export_path = os.path.join(exports_dir, f"{filename}.csv")
                df.to_csv(export_path, index=False)
            elif format.lower() == "json":
                export_path = os.path.join(exports_dir, f"{filename}.json")
                df.to_json(export_path, orient="records", date_format="iso")
            elif format.lower() == "excel":
                export_path = os.path.join(exports_dir, f"{filename}.xlsx")
                df.to_excel(export_path, index=False)
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return {"status": "error", "message": f"Unsupported export format: {format}"}

            # Return result
            result = {
                "status": "success",
                "export_path": export_path,
                "format": format.lower(),
                "rows": len(df),
                "columns": list(df.columns),
            }

            self.logger.info(f"Exported data to {export_path}")
            self._track_execution(start_time, success=True)
            return result

        except Exception as e:
            self.logger.error(f"Error exporting data: {str(e)}")
            self._track_execution(start_time, success=False)
            return {"status": "error", "message": str(e)}

    def run(self, **kwargs):
        """
        Run the data visualization service with provided parameters.

        Args:
            **kwargs: Keyword arguments including:
                - action: Action to perform (e.g., "create_campaign_chart", "create_keyword_chart", etc.)
                - data: Data to visualize
                - metric: Metric to visualize
                - days: Number of days to include
                - format: Export format
                - filename: Custom filename

        Returns:
            Results of the requested action
        """
        action = kwargs.get("action", "")
        self.logger.info(f"DataVisualizationService run called with action: {action}")

        if action == "create_campaign_chart":
            return self.create_campaign_performance_chart(
                kwargs.get("data", []),
                kwargs.get("metric", "clicks"),
                kwargs.get("days", 30),
                kwargs.get("campaign_ids", None),
                kwargs.get("filename", None),
            )
        elif action == "create_keyword_chart":
            return self.create_keyword_performance_chart(
                kwargs.get("data", []),
                kwargs.get("metric", "clicks"),
                kwargs.get("top_n", 10),
                kwargs.get("campaign_id", None),
                kwargs.get("filename", None),
            )
        elif action == "create_dashboard":
            return self.create_performance_summary_dashboard(
                kwargs.get("data", []), kwargs.get("days", 30), kwargs.get("filename", None)
            )
        elif action == "export_data":
            return self.export_visualization_data(
                kwargs.get("data", []), kwargs.get("format", "csv"), kwargs.get("filename", None)
            )
        else:
            self.logger.warning(f"Unknown action: {action}")
            return {"status": "error", "message": f"Unknown action: {action}"}
