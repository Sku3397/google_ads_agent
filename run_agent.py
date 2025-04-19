#!/usr/bin/env python
"""
Google Ads Autonomous Agent Runner

This script provides a command-line interface to run the autonomous Google Ads agent
with options for immediate execution or scheduling.
"""

import argparse
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/agent_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AgentRunner")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Google Ads Autonomous Agent Runner")
    
    # Mode selection
    parser.add_argument('--mode', choices=['immediate', 'schedule', 'service'], default='immediate',
                      help='Run mode: immediate (run once), schedule (run on schedule), service (run as daemon)')
    
    # Schedule options
    parser.add_argument('--frequency', choices=['daily', 'weekly'], default='daily',
                      help='Frequency for scheduled runs (daily or weekly)')
    parser.add_argument('--day', choices=['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
                      default='monday', help='Day of the week for weekly schedule')
    parser.add_argument('--hour', type=int, default=4, help='Hour to run (0-23)')
    parser.add_argument('--minute', type=int, default=0, help='Minute to run (0-59)')
    
    # Analysis options
    parser.add_argument('--days', type=int, default=30, help='Number of days of data to analyze (1-90)')
    
    # Safety thresholds
    parser.add_argument('--max-budget-change', type=float, default=20.0,
                      help='Maximum budget change percent allowed (0-100)')
    parser.add_argument('--max-bid-change', type=float, default=30.0,
                      help='Maximum bid change percent allowed (0-100)')
    
    return parser.parse_args()

def run_agent(args):
    """Run the agent with the specified settings"""
    try:
        # Import here to avoid import errors before checking requirements
        from autonomous_agent import GoogleAdsAutonomousAgent
        
        # Create the agent
        logger.info("Initializing Google Ads Autonomous Agent")
        agent = GoogleAdsAutonomousAgent()
        
        # Set analysis parameters
        agent.days_to_analyze = args.days
        agent.max_daily_budget_change_pct = args.max_budget_change
        agent.max_bid_change_pct = args.max_bid_change
        
        # Run based on selected mode
        if args.mode == 'immediate':
            logger.info("Running immediate analysis")
            agent.run_immediate_analysis()
            logger.info("Immediate analysis completed")
            
        elif args.mode == 'schedule':
            # Schedule based on frequency
            if args.frequency == 'daily':
                logger.info(f"Scheduling daily analysis at {args.hour:02d}:{args.minute:02d}")
                agent.schedule_daily_analysis(hour=args.hour, minute=args.minute)
            
            elif args.frequency == 'weekly':
                logger.info(f"Scheduling weekly analysis on {args.day} at {args.hour:02d}:{args.minute:02d}")
                agent.schedule_weekly_analysis(day_of_week=args.day, hour=args.hour, minute=args.minute)
            
            # Run the scheduler
            stop_event = agent.run_scheduler()
            
            logger.info("Agent running on schedule. Press Ctrl+C to stop.")
            try:
                # Keep the main thread alive until interrupted
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Stopping agent...")
                stop_event.set()
        
        elif args.mode == 'service':
            # Run as a background service
            logger.info("Starting agent as a background service")
            
            # Schedule based on frequency
            if args.frequency == 'daily':
                agent.schedule_daily_analysis(hour=args.hour, minute=args.minute)
            elif args.frequency == 'weekly':
                agent.schedule_weekly_analysis(day_of_week=args.day, hour=args.hour, minute=args.minute)
            
            # Run the scheduler in detached mode
            agent.run_scheduler()
            logger.info("Agent service started successfully")
            
    except ImportError as e:
        logger.error(f"Failed to import required module: {str(e)}")
        logger.error("Please install required packages: pip install schedule pandas")
        return 1
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")
        return 1
    
    return 0

def check_requirements():
    """Check if all required packages are installed"""
    try:
        import schedule
        import pandas
        from pathlib import Path
        return True
    except ImportError as e:
        logger.error(f"Missing required package: {str(e)}")
        logger.error("Please install required packages: pip install schedule pandas")
        return False

def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Run the agent
    return run_agent(args)

if __name__ == "__main__":
    sys.exit(main()) 