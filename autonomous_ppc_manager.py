#!/usr/bin/env python

import argparse
import logging
import os
import sys
import json
import time
from datetime import datetime, timedelta
import schedule

# Local imports
from ads_api import GoogleAdsAPI
from optimizer import AdsOptimizer
from autonomous_ppc_agent import AutonomousPPCAgent
from config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/autonomous_manager_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("AutonomousPPCManager")

def create_agent():
    """Create and initialize the autonomous PPC agent"""
    try:
        # Load configuration
        logger.info("Loading configuration")
        config = load_config()
        
        # Initialize Google Ads API
        logger.info("Initializing Google Ads API")
        ads_api = GoogleAdsAPI(config['google_ads'])
        
        # Initialize optimizer
        logger.info("Initializing Gemini optimizer")
        optimizer = AdsOptimizer(config['google_ai'])
        
        # Initialize agent with settings
        agent_config = {
            'auto_implement_threshold': config.get('agent', {}).get('auto_implement_threshold', 80),
            'max_daily_budget_change_pct': config.get('agent', {}).get('max_daily_budget_change_pct', 20),
            'min_data_points_required': config.get('agent', {}).get('min_data_points_required', 30),
            'max_keyword_bid_adjustment': config.get('agent', {}).get('max_keyword_bid_adjustment', 50)
        }
        
        logger.info("Initializing Autonomous PPC Agent")
        agent = AutonomousPPCAgent(ads_api, optimizer, agent_config)
        
        return agent
    except Exception as e:
        logger.error(f"Failed to create agent: {str(e)}")
        raise

def run_daily_optimization():
    """Run daily optimization cycle"""
    logger.info("===== Starting Daily Optimization Cycle =====")
    
    try:
        # Create agent
        agent = create_agent()
        
        # Run the daily optimization
        result = agent.run_daily_optimization()
        
        # Save report to file
        if result['status'] == 'success':
            os.makedirs('reports', exist_ok=True)
            report_file = f"reports/daily_report_{datetime.now().strftime('%Y%m%d')}.json"
            with open(report_file, 'w') as f:
                json.dump(result['report'], f, indent=2, default=str)
            
            logger.info(f"Daily optimization completed successfully. Report saved to {report_file}")
            
            # Log summary statistics
            recs = result['recommendations']
            logger.info(f"Generated {recs['total']} recommendations, {recs['implemented']} implemented automatically")
            
            return True
        else:
            logger.error(f"Daily optimization failed: {result['error']}")
            return False
    except Exception as e:
        logger.error(f"Error in daily optimization: {str(e)}")
        return False

def run_weekly_analysis():
    """Run weekly comprehensive analysis"""
    logger.info("===== Starting Weekly Analysis =====")
    
    try:
        # Create agent
        agent = create_agent()
        
        # Refresh data with longer lookback period
        agent.refresh_campaign_data(days=90)
        agent.refresh_keyword_data(days=90)
        
        # Generate in-depth report
        report = agent.generate_performance_report()
        
        # Save report to file
        os.makedirs('reports', exist_ok=True)
        report_file = f"reports/weekly_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Weekly analysis completed successfully. Report saved to {report_file}")
        return True
    except Exception as e:
        logger.error(f"Error in weekly analysis: {str(e)}")
        return False

def run_one_time_analysis(days=30, entity_type='all', auto_implement=False):
    """
    Run one-time analysis and optimization
    
    Args:
        days: Number of days to analyze
        entity_type: Type of entities to analyze ('all', 'campaigns', 'keywords')
        auto_implement: Whether to automatically implement recommendations
    """
    logger.info(f"===== Starting One-Time Analysis ({entity_type}, {days} days) =====")
    
    try:
        # Create agent
        agent = create_agent()
        
        # Refresh data
        if entity_type in ['all', 'campaigns']:
            agent.refresh_campaign_data(days=days)
            
        if entity_type in ['all', 'keywords']:
            agent.refresh_keyword_data(days=days)
        
        # Generate recommendations
        recommendations = agent.analyze_and_recommend(entity_type=entity_type, days=days)
        
        # Implement recommendations if requested
        if auto_implement:
            implemented, pending = agent.execute_recommendations(recommendations, auto_implement=True)
            logger.info(f"Auto-implemented {len(implemented)} recommendations, {len(pending)} pending")
        else:
            # Just summarize recommendations without implementing
            high_confidence = len([r for r in recommendations if r.confidence_score >= agent.auto_implement_threshold])
            logger.info(f"Generated {len(recommendations)} recommendations, {high_confidence} with high confidence")
            
            # Group recommendations by type
            rec_by_type = {}
            for rec in recommendations:
                if rec.action_type not in rec_by_type:
                    rec_by_type[rec.action_type] = []
                rec_by_type[rec.action_type].append(rec)
            
            # Log recommendation summary by type
            for action_type, recs in rec_by_type.items():
                logger.info(f"- {action_type}: {len(recs)} recommendations")
        
        # Generate report
        report = agent.generate_performance_report(entity_type=entity_type)
        
        # Save recommendations and report to file
        os.makedirs('reports', exist_ok=True)
        
        # Save recommendations
        rec_file = f"reports/recommendations_{entity_type}_{datetime.now().strftime('%Y%m%d')}.json"
        with open(rec_file, 'w') as f:
            json.dump([agent._recommendation_to_dict(r) for r in recommendations], f, indent=2, default=str)
            
        # Save report
        report_file = f"reports/analysis_report_{entity_type}_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"One-time analysis completed. Recommendations saved to {rec_file}")
        logger.info(f"Analysis report saved to {report_file}")
        
        return True
    except Exception as e:
        logger.error(f"Error in one-time analysis: {str(e)}")
        return False

def schedule_jobs():
    """Schedule recurring jobs"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Schedule daily optimization (every day at 8 AM)
    schedule.every().day.at("08:00").do(run_daily_optimization)
    logger.info("Scheduled daily optimization to run at 08:00")
    
    # Schedule weekly analysis (every Monday at 9 AM)
    schedule.every().monday.at("09:00").do(run_weekly_analysis)
    logger.info("Scheduled weekly analysis to run every Monday at 09:00")
    
    logger.info("Scheduler started. Press Ctrl+C to exit.")
    
    # Run jobs continuously
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Scheduler error: {str(e)}")

def main():
    """Main function to parse command-line arguments and run appropriate task"""
    parser = argparse.ArgumentParser(description='Autonomous Google Ads PPC Manager')
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Command: run-daily
    daily_parser = subparsers.add_parser('run-daily', help='Run daily optimization')
    
    # Command: run-weekly
    weekly_parser = subparsers.add_parser('run-weekly', help='Run weekly analysis')
    
    # Command: analyze
    analyze_parser = subparsers.add_parser('analyze', help='Run one-time analysis')
    analyze_parser.add_argument('--days', type=int, default=30, help='Number of days to analyze')
    analyze_parser.add_argument('--entity-type', choices=['all', 'campaigns', 'keywords'], default='all',
                              help='Type of entities to analyze')
    analyze_parser.add_argument('--auto-implement', action='store_true', 
                              help='Automatically implement high-confidence recommendations')
    
    # Command: schedule
    schedule_parser = subparsers.add_parser('schedule', help='Start scheduler for recurring tasks')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Execute appropriate function based on command
    if args.command == 'run-daily':
        run_daily_optimization()
    elif args.command == 'run-weekly':
        run_weekly_analysis()
    elif args.command == 'analyze':
        run_one_time_analysis(
            days=args.days,
            entity_type=args.entity_type,
            auto_implement=args.auto_implement
        )
    elif args.command == 'schedule':
        schedule_jobs()
    else:
        parser.print_help()
        
if __name__ == "__main__":
    main() 