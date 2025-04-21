"""
Scheduler Service for Google Ads Management System

This module provides scheduling capabilities for running periodic tasks,
optimizations, and maintenance operations for Google Ads campaigns.
"""

import logging
import json
import os
import time
import threading
import schedule
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path

from services.base_service import BaseService


class SchedulerService(BaseService):
    """
    Service for scheduling and executing periodic tasks for Google Ads management.

    This service manages the scheduling and execution of various optimization tasks,
    including bid adjustments, keyword analysis, performance reporting, and other
    maintenance operations. It supports:

    - One-time scheduled tasks
    - Recurring tasks (daily, weekly, monthly)
    - Task dependencies and priorities
    - Execution history tracking
    - Failure handling and retry logic
    """

    def __init__(
        self,
        ads_api=None,
        optimizer=None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the SchedulerService.

        Args:
            ads_api: Google Ads API client instance
            optimizer: AI optimizer instance for generating suggestions
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(ads_api, optimizer, config, logger)

        # Initialize scheduling library
        self.scheduler = schedule

        # Tasks configuration
        self.tasks_file = os.path.join("data", "scheduled_tasks.json")
        self.tasks = self._load_tasks()

        # Thread for running the scheduler
        self.scheduler_thread = None
        self.is_running = False

        # Task execution history
        self.execution_history = []

        self.logger.info("SchedulerService initialized.")

    def _load_tasks(self) -> Dict[str, Any]:
        """
        Load scheduled tasks from the tasks configuration file.

        Returns:
            Dictionary containing scheduled tasks configuration
        """
        if os.path.exists(self.tasks_file):
            try:
                with open(self.tasks_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading tasks from {self.tasks_file}: {str(e)}")

        # Return default task configuration if file doesn't exist or has errors
        return {"tasks": [], "last_updated": datetime.now().isoformat()}

    def _save_tasks(self) -> bool:
        """
        Save scheduled tasks to the tasks configuration file.

        Returns:
            Boolean indicating success
        """
        try:
            # Update last_updated timestamp
            self.tasks["last_updated"] = datetime.now().isoformat()

            # Ensure directory exists
            os.makedirs(os.path.dirname(self.tasks_file), exist_ok=True)

            # Save tasks to file
            with open(self.tasks_file, "w") as f:
                json.dump(self.tasks, f, indent=2, default=str)

            self.logger.info(f"Tasks saved to {self.tasks_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving tasks to {self.tasks_file}: {str(e)}")
            return False

    def add_task(
        self,
        task_name: str,
        task_type: str,
        schedule_type: str,
        schedule_time: str,
        parameters: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ) -> Dict[str, Any]:
        """
        Add a new scheduled task.

        Args:
            task_name: Name of the task
            task_type: Type of task (e.g., 'optimize_bids', 'keyword_analysis')
            schedule_type: Type of schedule ('one_time', 'daily', 'weekly', 'monthly')
            schedule_time: Time to run the task (format depends on schedule_type)
            parameters: Optional parameters for the task
            enabled: Whether the task is initially enabled

        Returns:
            Dictionary with task details
        """
        # Generate unique task ID
        task_id = f"task_{len(self.tasks['tasks']) + 1}_{int(time.time())}"

        # Create new task
        new_task = {
            "id": task_id,
            "name": task_name,
            "type": task_type,
            "schedule_type": schedule_type,
            "schedule_time": schedule_time,
            "parameters": parameters or {},
            "enabled": enabled,
            "created_at": datetime.now().isoformat(),
            "last_run": None,
            "next_run": None,
            "status": "pending",
            "execution_count": 0,
        }

        # Add task to tasks list
        self.tasks["tasks"].append(new_task)

        # Save tasks
        self._save_tasks()

        # If scheduler is running, schedule the new task
        if self.is_running and enabled:
            self._schedule_task(new_task)

        self.logger.info(f"Added new task: {task_name} ({task_id})")
        return new_task

    def update_task(self, task_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing scheduled task.

        Args:
            task_id: ID of the task to update
            updates: Dictionary with fields to update

        Returns:
            Updated task dictionary
        """
        # Find task by ID
        for i, task in enumerate(self.tasks["tasks"]):
            if task["id"] == task_id:
                # Update task fields
                for key, value in updates.items():
                    if key in task:
                        task[key] = value

                # Update modified timestamp
                task["updated_at"] = datetime.now().isoformat()

                # Save tasks
                self._save_tasks()

                # If scheduler is running, reschedule the task
                if self.is_running:
                    self._reschedule_task(task)

                self.logger.info(f"Updated task: {task['name']} ({task_id})")
                return task

        raise ValueError(f"Task with ID {task_id} not found")

    def delete_task(self, task_id: str) -> bool:
        """
        Delete a scheduled task.

        Args:
            task_id: ID of the task to delete

        Returns:
            Boolean indicating success
        """
        # Find task by ID
        for i, task in enumerate(self.tasks["tasks"]):
            if task["id"] == task_id:
                # Remove task from list
                deleted_task = self.tasks["tasks"].pop(i)

                # Save tasks
                self._save_tasks()

                # If scheduler is running, unschedule the task
                if self.is_running:
                    self._unschedule_task(deleted_task)

                self.logger.info(f"Deleted task: {deleted_task['name']} ({task_id})")
                return True

        self.logger.warning(f"Task with ID {task_id} not found for deletion")
        return False

    def get_tasks(self, include_disabled: bool = False) -> List[Dict[str, Any]]:
        """
        Get all scheduled tasks.

        Args:
            include_disabled: Whether to include disabled tasks

        Returns:
            List of task dictionaries
        """
        if include_disabled:
            return self.tasks["tasks"]
        else:
            return [task for task in self.tasks["tasks"] if task["enabled"]]

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific task by ID.

        Args:
            task_id: ID of the task to get

        Returns:
            Task dictionary or None if not found
        """
        for task in self.tasks["tasks"]:
            if task["id"] == task_id:
                return task
        return None

    def start(self):
        """
        Start the scheduler in a background thread.
        """
        if self.is_running:
            self.logger.warning("Scheduler is already running")
            return

        # Schedule all enabled tasks
        for task in self.tasks["tasks"]:
            if task["enabled"]:
                self._schedule_task(task)

        # Start scheduler in a background thread
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()

        self.logger.info("Scheduler started")

    def stop(self):
        """
        Stop the scheduler.
        """
        if not self.is_running:
            self.logger.warning("Scheduler is not running")
            return

        # Set flag to stop the scheduler
        self.is_running = False

        # Clear all scheduled jobs
        self.scheduler.clear()

        # Wait for scheduler thread to stop
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)

        self.logger.info("Scheduler stopped")

    def _run_scheduler(self):
        """
        Run the scheduler loop.
        """
        self.logger.info("Scheduler thread started")

        while self.is_running:
            # Run pending tasks
            self.scheduler.run_pending()

            # Sleep for a short time to prevent high CPU usage
            time.sleep(1)

        self.logger.info("Scheduler thread stopped")

    def _execute_task(self, task: Dict[str, Any]):
        """
        Execute a scheduled task.

        Args:
            task: Task dictionary
        """
        task_id = task["id"]
        task_name = task["name"]
        task_type = task["type"]
        parameters = task["parameters"]

        self.logger.info(f"Executing task: {task_name} ({task_id})")

        start_time = datetime.now()

        # Update task status
        task["status"] = "running"
        task["last_run"] = start_time.isoformat()
        task["execution_count"] += 1

        try:
            # Execute task based on type
            result = self._handle_task_execution(task_type, parameters)

            # Update task status
            task["status"] = "completed"

            # Record execution history
            execution_record = {
                "task_id": task_id,
                "task_name": task_name,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
                "status": "success",
                "result": result,
            }

            self.logger.info(f"Task execution completed: {task_name} ({task_id})")

        except Exception as e:
            error_message = f"Error executing task {task_name} ({task_id}): {str(e)}"
            self.logger.error(error_message)

            # Update task status
            task["status"] = "failed"

            # Record execution history
            execution_record = {
                "task_id": task_id,
                "task_name": task_name,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
                "status": "error",
                "error": str(e),
            }

        # Add execution record to history
        self.execution_history.append(execution_record)

        # Trim history if too long
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]

        # Save tasks
        self._save_tasks()

    def _handle_task_execution(self, task_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle the execution of a task based on its type.

        Args:
            task_type: Type of task to execute
            parameters: Task parameters

        Returns:
            Result dictionary
        """
        # This method would dispatch to the appropriate service method
        # based on the task type. For now, we'll just return a placeholder.

        # Example tasks that could be implemented:
        if task_type == "optimize_bids":
            # Call bid service to optimize bids
            pass
        elif task_type == "keyword_analysis":
            # Call keyword service to analyze keywords
            pass
        elif task_type == "performance_report":
            # Generate performance report
            pass
        elif task_type == "personalization_update":
            # Run personalization updates
            try:
                # Check if we have access to the AdsAgent services
                if (
                    hasattr(self, "ads_agent")
                    and hasattr(self.ads_agent, "services")
                    and "personalization" in self.ads_agent.services
                ):
                    personalization_service = self.ads_agent.services["personalization"]

                    # Apply configuration from parameters
                    if parameters.get("segment_count"):
                        personalization_service.segment_count = parameters["segment_count"]
                    if parameters.get("min_observations"):
                        personalization_service.min_observations = parameters["min_observations"]
                    if parameters.get("data_lookback_days"):
                        personalization_service.data_lookback_days = parameters[
                            "data_lookback_days"
                        ]

                    # Run the update
                    result = personalization_service.run_personalization_update()

                    return {
                        "status": "success" if result else "error",
                        "message": (
                            "Personalization update completed"
                            if result
                            else "Personalization update failed"
                        ),
                        "timestamp": datetime.now().isoformat(),
                    }
                else:
                    self.logger.error("Cannot access personalization service or ads_agent not set")
                    return {
                        "status": "error",
                        "message": "Cannot access personalization service",
                        "timestamp": datetime.now().isoformat(),
                    }
            except Exception as e:
                self.logger.error(f"Error executing personalization update: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Error: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                }

        elif task_type == 'update_ltv_predictions':
            if "ltv_bidding" in self.services:
                self.logger.info(f"Executing LTV prediction update task with params: {parameters}")
                result = self.services["ltv_bidding"].update_ltv_predictions(**parameters)
            else:
                result = {"status": "error", "message": "LTVBiddingService not available."}
        elif task_type == 'apply_ltv_bids':
            if "ltv_bidding" in self.services:
                self.logger.info(f"Executing LTV bid application task with params: {parameters}")
                result = self.services["ltv_bidding"].apply_ltv_bidding_strategy(**parameters)
            else:
                result = {"status": "error", "message": "LTVBiddingService not available."}
        # Handle potential simulation task
        elif task_type == 'run_simulation':
            if "simulation" in self.services:
                self.logger.info(f"Executing simulation task with params: {parameters}")
                # Example: Run a specific simulation type based on params
                sim_type = parameters.get('simulation_type', 'forecast')
                entity_id = parameters.get('entity_id')
                if sim_type == 'forecast' and entity_id:
                    result = self.services["simulation"].get_performance_forecast(
                        campaign_id=entity_id, # Assuming entity_id is campaign_id for forecast
                        days_to_forecast=parameters.get('days_to_forecast', 30),
                        lookback_days=parameters.get('lookback_days', 90)
                    )
                elif sim_type == 'bid_simulation' and entity_id:
                    # Placeholder for calling bid simulation if needed via scheduler
                    # result = self.services["simulation"].simulate_bid_changes(...)
                    result = {"status": "placeholder", "message": "Bid simulation via scheduler TBD"}
                else:
                    result = {"status": "error", "message": f"Invalid simulation parameters: {parameters}"}
            else:
                result = {"status": "error", "message": "SimulationService not available."}
        else: # Default case if task_type is not recognized
            self.logger.warning(f"Task type '{task_type}' is not recognized.")
            result = {"status": "error", "message": f"Unknown task type: {task_type}"}

        # Default error if task type is known but service is missing or logic fails
        # This condition might need refinement based on how result is handled above
        if result is None:
             # Logged inside specific handlers usually, maybe remove this generic one?
             pass # Or set a generic error if not handled above
             # result = {
             #      "status": "error",
             #      "message": f"Error executing task {task_name} ({task_id}): Unknown error",
             # }

        return result

    def _schedule_task(self, task: Dict[str, Any]):
        """
        Schedule a task based on its schedule type.

        Args:
            task: Task dictionary
        """
        task_id = task["id"]
        schedule_type = task["schedule_type"]
        schedule_time = task["schedule_time"]

        # Schedule based on schedule type
        if schedule_type == "one_time":
            # One-time schedule (datetime string)
            try:
                schedule_datetime = datetime.fromisoformat(schedule_time)
                # Calculate seconds until scheduled time
                seconds_until = (schedule_datetime - datetime.now()).total_seconds()
                if seconds_until > 0:
                    # Schedule one-time task
                    self.scheduler.every(seconds_until).seconds.do(self._execute_task, task).tag(
                        task_id
                    )
                    task["next_run"] = schedule_datetime.isoformat()
                else:
                    self.logger.warning(
                        f"Task {task_id} scheduled in the past, will not be executed"
                    )
                    task["status"] = "expired"
            except ValueError:
                self.logger.error(f"Invalid datetime format for one-time task: {schedule_time}")
                task["status"] = "error"

        elif schedule_type == "daily":
            # Daily schedule (HH:MM format)
            try:
                hour, minute = map(int, schedule_time.split(":"))
                job = (
                    self.scheduler.every()
                    .day.at(schedule_time)
                    .do(self._execute_task, task)
                    .tag(task_id)
                )
                # Calculate next run time
                now = datetime.now()
                next_run = datetime(now.year, now.month, now.day, hour, minute)
                if next_run < now:
                    next_run = next_run + timedelta(days=1)
                task["next_run"] = next_run.isoformat()
            except Exception as e:
                self.logger.error(f"Error scheduling daily task: {str(e)}")
                task["status"] = "error"

        elif schedule_type == "weekly":
            # Weekly schedule (DAY:HH:MM format)
            try:
                day, time = schedule_time.split(":")
                hour, minute = map(int, time.split(":"))

                # Map day string to method
                day_methods = {
                    "monday": self.scheduler.every().monday,
                    "tuesday": self.scheduler.every().tuesday,
                    "wednesday": self.scheduler.every().wednesday,
                    "thursday": self.scheduler.every().thursday,
                    "friday": self.scheduler.every().friday,
                    "saturday": self.scheduler.every().saturday,
                    "sunday": self.scheduler.every().sunday,
                }

                if day.lower() in day_methods:
                    job = (
                        day_methods[day.lower()]
                        .at(f"{hour:02d}:{minute:02d}")
                        .do(self._execute_task, task)
                        .tag(task_id)
                    )
                    # Calculate next run (simplified)
                    task["next_run"] = "Next " + day
                else:
                    raise ValueError(f"Invalid day of week: {day}")
            except Exception as e:
                self.logger.error(f"Error scheduling weekly task: {str(e)}")
                task["status"] = "error"

        elif schedule_type == "monthly":
            # Monthly schedule (DAY:HH:MM format, DAY is day of month)
            try:
                day_of_month = int(schedule_time.split(":")[0])
                time_part = ":".join(schedule_time.split(":")[1:])

                # For now, we'll use a daily check that only executes on the right day
                # This is a limitation of the schedule library
                def monthly_job_wrapper(task):
                    if datetime.now().day == day_of_month:
                        self._execute_task(task)

                job = (
                    self.scheduler.every()
                    .day.at(time_part)
                    .do(monthly_job_wrapper, task)
                    .tag(task_id)
                )
                task["next_run"] = f"Next month on day {day_of_month}"
            except Exception as e:
                self.logger.error(f"Error scheduling monthly task: {str(e)}")
                task["status"] = "error"

        else:
            self.logger.error(f"Unknown schedule type: {schedule_type}")
            task["status"] = "error"

    def _reschedule_task(self, task: Dict[str, Any]):
        """
        Reschedule a task after updates.

        Args:
            task: Task dictionary
        """
        # Unschedule existing task
        self._unschedule_task(task)

        # Schedule task if enabled
        if task["enabled"]:
            self._schedule_task(task)

    def _unschedule_task(self, task: Dict[str, Any]):
        """
        Unschedule a task.

        Args:
            task: Task dictionary
        """
        # Clear all jobs with the task ID tag
        self.scheduler.clear(tag=task["id"])

    def execute_task_now(self, task_id: str) -> Dict[str, Any]:
        """
        Execute a task immediately regardless of its schedule.

        Args:
            task_id: ID of the task to execute

        Returns:
            Dictionary with execution results
        """
        # Find task by ID
        task = self.get_task(task_id)
        if not task:
            error_message = f"Task with ID {task_id} not found"
            self.logger.error(error_message)
            return {"status": "error", "message": error_message}

        # Execute task
        try:
            self._execute_task(task)
            return {"status": "success", "message": f"Task {task['name']} executed"}
        except Exception as e:
            error_message = f"Error executing task {task['name']}: {str(e)}"
            self.logger.error(error_message)
            return {"status": "error", "message": error_message}

    def get_execution_history(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the execution history for all tasks or a specific task.

        Args:
            task_id: Optional task ID to filter history

        Returns:
            List of execution records
        """
        if task_id:
            return [record for record in self.execution_history if record["task_id"] == task_id]
        else:
            return self.execution_history

    def run(self, **kwargs):
        """
        Run method required by the BaseService interface.
        For SchedulerService, this can perform maintenance or check for overdue tasks.
        """
        self.logger.info("Running SchedulerService maintenance")

        # Check for overdue tasks
        for task in self.tasks["tasks"]:
            if (
                task["enabled"]
                and task["status"] == "pending"
                and task["schedule_type"] == "one_time"
            ):
                try:
                    next_run = (
                        datetime.fromisoformat(task["next_run"]) if task.get("next_run") else None
                    )
                    if next_run and next_run < datetime.now():
                        self.logger.warning(f"Task {task['name']} ({task['id']}) is overdue")
                        # Could automatically execute or reschedule here
                except (ValueError, TypeError):
                    pass

        # Return success
        return {"status": "success", "message": "SchedulerService maintenance completed"}

    def register_personalization_updates(
        self,
        hour: int = 2,
        minute: int = 0,
        day_of_week: str = "monday",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Register weekly personalization updates to run at a specified time.

        This schedules the personalization service to update user segments and
        performance metrics on a weekly basis.

        Args:
            hour: Hour of day to run (0-23, default 2 AM)
            minute: Minute of hour to run (0-59, default 0)
            day_of_week: Day of week to run (default: monday)
            parameters: Additional parameters for personalization updates

        Returns:
            Dictionary with task details
        """
        task_name = "Weekly Personalization Updates"
        task_type = "personalization_update"
        schedule_type = "weekly"
        schedule_time = f"{day_of_week}:{hour:02d}:{minute:02d}"

        # Default parameters if none provided
        if parameters is None:
            parameters = {"segment_count": 5, "min_observations": 100, "data_lookback_days": 90}

        # Add the task
        task = self.add_task(
            task_name=task_name,
            task_type=task_type,
            schedule_type=schedule_type,
            schedule_time=schedule_time,
            parameters=parameters,
            enabled=True,
        )

        self.logger.info(
            f"Registered personalization updates to run weekly on {day_of_week} at {hour:02d}:{minute:02d}"
        )

        return task
