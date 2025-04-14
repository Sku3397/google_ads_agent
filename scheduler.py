import schedule
import time
import threading
import json
import os
import uuid
from datetime import datetime, timedelta

class ScheduledTask:
    """
    Class representing a single scheduled task.
    """
    def __init__(self, task_id, name, function, schedule_type, 
                 args=None, kwargs=None, hour=9, minute=0, day_of_week=None):
        """
        Initialize a scheduled task.
        
        Args:
            task_id (str): Unique identifier for the task
            name (str): Human-readable name for the task
            function (callable): Function to call when scheduled
            schedule_type (str): Type of schedule ('daily', 'weekly', 'once')
            args (list, optional): Positional arguments for the function
            kwargs (dict, optional): Keyword arguments for the function
            hour (int): Hour of day to run (0-23)
            minute (int): Minute of hour to run (0-59)
            day_of_week (str, optional): Day of week for weekly tasks 
                                        ('monday', 'tuesday', etc.)
        """
        self.task_id = task_id
        self.name = name
        self.function = function
        self.schedule_type = schedule_type
        self.args = args or []
        self.kwargs = kwargs or {}
        self.hour = hour
        self.minute = minute
        self.day_of_week = day_of_week
        self.job = None
        self.last_run = None
        self.next_run = None
        self.created_at = datetime.now()
        self.status = "scheduled"  # scheduled, running, completed, failed
        self.result = None
        self.error = None
    
    def to_dict(self):
        """Convert task to a dictionary for serialization."""
        return {
            'task_id': self.task_id,
            'name': self.name,
            'schedule_type': self.schedule_type,
            'args': self.args,
            'kwargs': self.kwargs,
            'hour': self.hour,
            'minute': self.minute,
            'day_of_week': self.day_of_week,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'created_at': self.created_at.isoformat(),
            'status': self.status,
            'error': self.error
        }

class AdsScheduler:
    """
    Enhanced scheduler for Google Ads Optimization Agent that supports multiple tasks.
    """
    def __init__(self, logger=None):
        """
        Initialize the scheduler.
        
        Args:
            logger (object, optional): Logger object for recording events
        """
        self.scheduler = schedule
        self.tasks = {}
        self.running = False
        self.thread = None
        self.logger = logger
        self.task_history = []
        self.tasks_file = "scheduled_tasks.json"
        
        # Try to load existing tasks
        self._load_tasks()
    
    def add_task(self, name, function, schedule_type, args=None, kwargs=None, 
                hour=9, minute=0, day_of_week=None):
        """
        Add a new task to the scheduler.
        
        Args:
            name (str): Human-readable name for the task
            function (callable): Function to call when scheduled
            schedule_type (str): Type of schedule ('daily', 'weekly', 'once')
            args (list, optional): Positional arguments for the function
            kwargs (dict, optional): Keyword arguments for the function
            hour (int): Hour of day to run (0-23)
            minute (int): Minute of hour to run (0-59)
            day_of_week (str, optional): Day of week for weekly tasks
            
        Returns:
            str: The task ID
        """
        task_id = str(uuid.uuid4())
        
        # Create the task
        task = ScheduledTask(
            task_id=task_id,
            name=name,
            function=function,
            schedule_type=schedule_type,
            args=args,
            kwargs=kwargs,
            hour=hour,
            minute=minute,
            day_of_week=day_of_week
        )
        
        # Schedule the task based on type
        self._schedule_task(task)
        
        # Add to tasks dictionary
        self.tasks[task_id] = task
        
        # Log the addition
        if self.logger:
            self.logger.info(f"Task {name} (ID: {task_id}) scheduled for {schedule_type} at {hour:02d}:{minute:02d}"
                            + (f" on {day_of_week}" if day_of_week else ""))
        
        # Save tasks to file
        self._save_tasks()
        
        return task_id
    
    def _schedule_task(self, task):
        """
        Schedule a task with the scheduler.
        
        Args:
            task (ScheduledTask): Task to schedule
        """
        # Create a wrapper function that updates task status
        def task_wrapper():
            try:
                task.status = "running"
                task.last_run = datetime.now()
                
                if self.logger:
                    self.logger.info(f"Running task: {task.name} (ID: {task.task_id})")
                
                result = task.function(*task.args, **task.kwargs)
                
                task.status = "completed"
                task.result = result
                
                # Add to history
                self.task_history.append({
                    'task_id': task.task_id,
                    'name': task.name,
                    'status': 'completed',
                    'start_time': task.last_run.isoformat(),
                    'end_time': datetime.now().isoformat()
                })
                
                if self.logger:
                    self.logger.info(f"Task completed: {task.name} (ID: {task.task_id})")
                
            except Exception as e:
                task.status = "failed"
                task.error = str(e)
                
                # Add to history
                self.task_history.append({
                    'task_id': task.task_id,
                    'name': task.name,
                    'status': 'failed',
                    'start_time': task.last_run.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'error': str(e)
                })
                
                if self.logger:
                    self.logger.error(f"Task failed: {task.name} (ID: {task.task_id}). Error: {str(e)}")
        
        # Schedule based on type
        time_str = f"{task.hour:02d}:{task.minute:02d}"
        
        if task.schedule_type == 'daily':
            job = self.scheduler.every().day.at(time_str).do(task_wrapper)
            # Calculate next run time
            now = datetime.now()
            run_time = now.replace(hour=task.hour, minute=task.minute, second=0, microsecond=0)
            if run_time < now:
                run_time = run_time + timedelta(days=1)
            task.next_run = run_time
            
        elif task.schedule_type == 'weekly' and task.day_of_week:
            day_attr = getattr(self.scheduler.every(), task.day_of_week)
            job = day_attr.at(time_str).do(task_wrapper)
            # Calculate next run time 
            now = datetime.now()
            today = now.weekday()
            days = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 
                    'friday': 4, 'saturday': 5, 'sunday': 6}
            target_day = days.get(task.day_of_week.lower(), 0)
            days_ahead = target_day - today
            if days_ahead < 0 or (days_ahead == 0 and now.replace(hour=task.hour, minute=task.minute) < now):
                days_ahead += 7
            run_time = now.replace(hour=task.hour, minute=task.minute, second=0, microsecond=0) + timedelta(days=days_ahead)
            task.next_run = run_time
            
        elif task.schedule_type == 'once':
            # For one-time tasks, we use a separate timer thread
            run_time = datetime.now().replace(hour=task.hour, minute=task.minute, second=0, microsecond=0)
            if run_time < datetime.now():
                run_time = run_time + timedelta(days=1)  # Schedule for tomorrow
            
            # Calculate seconds until the task should run
            delay = (run_time - datetime.now()).total_seconds()
            
            # Use a timer to run the task after the delay
            timer = threading.Timer(delay, task_wrapper)
            timer.daemon = True
            timer.start()
            
            task.next_run = run_time
            job = timer  # Store the timer as the job
        
        task.job = job
    
    def remove_task(self, task_id):
        """
        Remove a task from the scheduler.
        
        Args:
            task_id (str): ID of the task to remove
            
        Returns:
            bool: True if the task was removed, False otherwise
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        # Cancel the task
        if task.schedule_type == 'once':
            # For one-time tasks, cancel the timer
            if hasattr(task.job, 'cancel'):
                task.job.cancel()
        else:
            # For regular tasks, cancel the scheduled job
            self.scheduler.cancel_job(task.job)
        
        # Remove from tasks dictionary
        del self.tasks[task_id]
        
        # Log the removal
        if self.logger:
            self.logger.info(f"Task {task.name} (ID: {task_id}) removed from scheduler")
        
        # Save tasks to file
        self._save_tasks()
        
        return True
    
    def get_tasks(self):
        """
        Get all scheduled tasks.
        
        Returns:
            dict: Dictionary of all tasks
        """
        return self.tasks
    
    def get_task(self, task_id):
        """
        Get a specific task.
        
        Args:
            task_id (str): ID of the task to get
            
        Returns:
            ScheduledTask: The task or None if not found
        """
        return self.tasks.get(task_id)
    
    def get_task_history(self, limit=10):
        """
        Get the task execution history.
        
        Args:
            limit (int): Maximum number of history items to return
            
        Returns:
            list: List of task history items
        """
        return self.task_history[-limit:] if limit else self.task_history
    
    def schedule_daily(self, function, hour=9, minute=0, name=None, args=None, kwargs=None):
        """
        Schedule a function to run daily at specified time.
        
        Args:
            function (callable): Function to call when scheduled
            hour (int): Hour of day (0-23)
            minute (int): Minute of hour (0-59)
            name (str, optional): Human-readable name for the task
            args (list, optional): Positional arguments for the function
            kwargs (dict, optional): Keyword arguments for the function
            
        Returns:
            str: The task ID
        """
        task_name = name or f"Daily task at {hour:02d}:{minute:02d}"
        return self.add_task(
            name=task_name,
            function=function,
            schedule_type='daily',
            args=args,
            kwargs=kwargs,
            hour=hour,
            minute=minute
        )
    
    def schedule_weekly(self, function, day_of_week, hour=9, minute=0, name=None, args=None, kwargs=None):
        """
        Schedule a function to run weekly on specified day and time.
        
        Args:
            function (callable): Function to call when scheduled
            day_of_week (str): Day of week (monday, tuesday, etc.)
            hour (int): Hour of day (0-23)
            minute (int): Minute of hour (0-59)
            name (str, optional): Human-readable name for the task
            args (list, optional): Positional arguments for the function
            kwargs (dict, optional): Keyword arguments for the function
            
        Returns:
            str: The task ID
        """
        task_name = name or f"Weekly task on {day_of_week} at {hour:02d}:{minute:02d}"
        return self.add_task(
            name=task_name,
            function=function,
            schedule_type='weekly',
            args=args,
            kwargs=kwargs,
            hour=hour,
            minute=minute,
            day_of_week=day_of_week
        )
    
    def schedule_once(self, function, hour, minute, name=None, args=None, kwargs=None):
        """
        Schedule a function to run once at the specified time.
        
        Args:
            function (callable): Function to call when scheduled
            hour (int): Hour of day (0-23)
            minute (int): Minute of hour (0-59)
            name (str, optional): Human-readable name for the task
            args (list, optional): Positional arguments for the function
            kwargs (dict, optional): Keyword arguments for the function
            
        Returns:
            str: The task ID
        """
        task_name = name or f"One-time task at {hour:02d}:{minute:02d}"
        return self.add_task(
            name=task_name,
            function=function,
            schedule_type='once',
            args=args,
            kwargs=kwargs,
            hour=hour,
            minute=minute
        )
    
    def start(self):
        """Start the scheduler and run pending tasks."""
        if self.running:
            if self.logger:
                self.logger.warning("Scheduler is already running")
            return
        
        self.running = True
        if self.logger:
            self.logger.info("Scheduler started")
        
        # Start in a separate thread
        self.thread = threading.Thread(target=self._run_scheduler)
        self.thread.daemon = True
        self.thread.start()
    
    def _run_scheduler(self):
        """Run the scheduler loop."""
        try:
            while self.running:
                self.scheduler.run_pending()
                time.sleep(1)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Scheduler error: {str(e)}")
            self.running = False
    
    def stop(self):
        """Stop the scheduler."""
        if not self.running:
            if self.logger:
                self.logger.warning("Scheduler is not running")
            return
        
        self.running = False
        if self.logger:
            self.logger.info("Scheduler stopped")
    
    def _save_tasks(self):
        """Save scheduled tasks to file."""
        try:
            # Convert tasks to serializable format
            serializable_tasks = {}
            for task_id, task in self.tasks.items():
                serializable_tasks[task_id] = task.to_dict()
            
            with open(self.tasks_file, 'w') as f:
                json.dump(serializable_tasks, f, indent=2)
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error saving tasks to file: {str(e)}")
    
    def _load_tasks(self):
        """Load scheduled tasks from file."""
        try:
            if not os.path.exists(self.tasks_file):
                return
            
            with open(self.tasks_file, 'r') as f:
                serialized_tasks = json.load(f)
            
            # We can't fully deserialize tasks because they contain function references
            # But we can log the information for debugging purposes
            if self.logger:
                self.logger.info(f"Found {len(serialized_tasks)} previously scheduled tasks in {self.tasks_file}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading tasks from file: {str(e)}")
    
    def is_running(self):
        """Check if the scheduler is running."""
        return self.running 