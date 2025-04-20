"""
Scheduler Service for Google Ads Management System

This package provides scheduling capabilities for running
periodic tasks, optimizations, and maintenance operations
for Google Ads campaigns.
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
import json

from .scheduler_service import SchedulerService
from ..base_service import BaseService

logger = logging.getLogger(__name__)


class SchedulerService(BaseService):
    """Service for managing scheduled tasks and coordinating analysis runs."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the scheduler service."""
        super().__init__(config)
        self.scheduler = None  # Will be initialized in start()
        self.tasks = {}
        self.task_history = []

        # Default schedules for different analysis types
        self.default_schedules = {
            "causal_impact": {"hour": 6, "minute": 0, "frequency": "daily"},
            "control_group_update": {
                "hour": 5,
                "minute": 0,
                "frequency": "weekly",
                "day": "monday",
            },
        }

    def schedule_causal_analysis(
        self,
        campaign_ids: List[str],
        metrics: List[str],
        schedule_type: str = "daily",
        hour: Optional[int] = None,
        minute: Optional[int] = None,
        day_of_week: Optional[str] = None,
    ) -> List[str]:
        """
        Schedule causal impact analysis tasks.

        Args:
            campaign_ids: List of campaign IDs to analyze
            metrics: List of metrics to analyze
            schedule_type: Type of schedule ('daily', 'weekly')
            hour: Hour to run (0-23)
            minute: Minute to run (0-59)
            day_of_week: Day for weekly schedule

        Returns:
            List of scheduled task IDs
        """
        if hour is None:
            hour = self.default_schedules["causal_impact"]["hour"]
        if minute is None:
            minute = self.default_schedules["causal_impact"]["minute"]

        task_ids = []

        for campaign_id in campaign_ids:
            for metric in metrics:
                task_name = f"causal_impact_analysis_{campaign_id}_{metric}"

                if schedule_type == "daily":
                    task_id = self.schedule_daily(
                        name=task_name,
                        function=lambda: self._run_causal_analysis(campaign_id, metric),
                        hour=hour,
                        minute=minute,
                    )
                elif schedule_type == "weekly":
                    if not day_of_week:
                        day_of_week = "monday"
                    task_id = self.schedule_weekly(
                        name=task_name,
                        function=lambda: self._run_causal_analysis(campaign_id, metric),
                        day_of_week=day_of_week,
                        hour=hour,
                        minute=minute,
                    )
                else:
                    raise ValueError(f"Invalid schedule type: {schedule_type}")

                task_ids.append(task_id)

                logger.info(
                    f"Scheduled causal analysis for campaign {campaign_id}, "
                    f"metric {metric} at {hour:02d}:{minute:02d} {schedule_type}"
                )

        return task_ids

    def schedule_control_group_updates(
        self,
        campaign_ids: List[str],
        metrics: List[str],
        schedule_type: str = "weekly",
        hour: Optional[int] = None,
        minute: Optional[int] = None,
        day_of_week: str = "monday",
    ) -> List[str]:
        """
        Schedule control group update tasks.

        Args:
            campaign_ids: List of campaign IDs
            metrics: List of metrics to optimize for
            schedule_type: Type of schedule ('weekly', 'monthly')
            hour: Hour to run (0-23)
            minute: Minute to run (0-59)
            day_of_week: Day for weekly schedule

        Returns:
            List of scheduled task IDs
        """
        if hour is None:
            hour = self.default_schedules["control_group_update"]["hour"]
        if minute is None:
            minute = self.default_schedules["control_group_update"]["minute"]

        task_ids = []

        for campaign_id in campaign_ids:
            for metric in metrics:
                task_name = f"control_group_update_{campaign_id}_{metric}"

                if schedule_type == "weekly":
                    task_id = self.schedule_weekly(
                        name=task_name,
                        function=lambda: self._update_control_group(campaign_id, metric),
                        day_of_week=day_of_week,
                        hour=hour,
                        minute=minute,
                    )
                else:
                    raise ValueError(f"Invalid schedule type: {schedule_type}")

                task_ids.append(task_id)

                logger.info(
                    f"Scheduled control group update for campaign {campaign_id}, "
                    f"metric {metric} at {hour:02d}:{minute:02d} {schedule_type}"
                )

        return task_ids

    def _run_causal_analysis(self, campaign_id: str, metric: str) -> Dict[str, Any]:
        """Run causal impact analysis for a campaign and metric."""
        try:
            return self.ads_agent.analyze_bid_impact(campaign_id=campaign_id, metric=metric)
        except Exception as e:
            logger.error(
                f"Error running causal analysis for campaign {campaign_id}, "
                f"metric {metric}: {str(e)}"
            )
            raise

    def _update_control_group(self, campaign_id: str, metric: str) -> Dict[str, Any]:
        """Update control group for a campaign and metric."""
        try:
            return self.ads_agent.build_campaign_control_group(
                target_campaign_id=campaign_id, metric=metric
            )
        except Exception as e:
            logger.error(
                f"Error updating control group for campaign {campaign_id}, "
                f"metric {metric}: {str(e)}"
            )
            raise

    # ... existing methods ...


__all__ = ["SchedulerService"]
