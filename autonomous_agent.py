import logging
import time
import json
import os
from datetime import datetime, timedelta
import schedule
import threading
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Import project modules
from config import load_config
from ads_api import GoogleAdsAPI
from optimizer import AdsOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/autonomous_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AutonomousAgent")

class GoogleAdsAutonomousAgent:
    """
    Fully autonomous Google Ads optimization agent that:
    1. Runs on schedule without requiring manual intervention
    2. Analyzes campaign, ad group, keyword, and ad performance
    3. Uses LLM to generate actionable optimization suggestions
    4. Implements optimizations automatically
    5. Tracks performance changes from optimizations
    """
    
    def __init__(self, config_path: str = ".env"):
        """
        Initialize the autonomous Google Ads agent.
        
        Args:
            config_path: Path to configuration file
        """
        self.initialized = False
        self.config = None
        self.ads_api = None
        self.optimizer = None
        
        # Performance tracking
        self.optimization_history = []
        self.performance_tracking = {}
        
        # Analysis settings
        self.days_to_analyze = 30
        self.analysis_frequency = "daily"  # daily, weekly
        self.execution_hour = 4  # 4 AM by default
        
        # Safety thresholds
        self.max_daily_budget_change_pct = 20  # Max 20% budget change per day
        self.max_bid_change_pct = 30  # Max 30% bid change per optimization
        self.max_keywords_to_pause_pct = 10  # Max 10% of keywords paused per day
        
        # Initialize
        self._initialize(config_path)
        
        # Create history directory if it doesn't exist
        Path("history").mkdir(exist_ok=True)
        
    def _initialize(self, config_path: str):
        """Initialize APIs and components"""
        try:
            # Load configuration
            logger.info("Loading configuration...")
            self.config = load_config(config_path)
            
            # Initialize Google Ads API
            logger.info("Initializing Google Ads API...")
            self.ads_api = GoogleAdsAPI(self.config['google_ads'])
            
            # Initialize Optimizer with AI capabilities
            logger.info("Initializing AI Optimizer...")
            self.optimizer = AdsOptimizer(self.config['google_ai']['api_key'])
            
            # Load optimization history if exists
            self._load_optimization_history()
            
            self.initialized = True
            logger.info("Autonomous agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize autonomous agent: {str(e)}")
            raise
    
    def _load_optimization_history(self):
        """Load optimization history from file"""
        history_file = Path("history/optimization_history.json")
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    self.optimization_history = json.load(f)
                logger.info(f"Loaded {len(self.optimization_history)} historical optimization records")
            except Exception as e:
                logger.error(f"Failed to load optimization history: {str(e)}")
                self.optimization_history = []
    
    def _save_optimization_history(self):
        """Save optimization history to file"""
        try:
            with open("history/optimization_history.json", "w") as f:
                json.dump(self.optimization_history, f, indent=2)
            logger.info(f"Saved {len(self.optimization_history)} optimization records to history")
        except Exception as e:
            logger.error(f"Failed to save optimization history: {str(e)}")
    
    def run_scheduled_analysis(self):
        """
        Run a complete analysis cycle on schedule.
        This is the main entry point for the scheduled task.
        """
        logger.info("Starting scheduled analysis cycle")
        
        try:
            # 1. Fetch all necessary data
            campaign_data = self._fetch_campaign_data()
            
            if not campaign_data:
                logger.warning("No campaign data available. Aborting analysis cycle.")
                return
            
            # 2. Analyze each campaign
            for campaign in campaign_data:
                self._analyze_campaign(campaign)
            
            # 3. Run account-level analysis
            self._analyze_account(campaign_data)
            
            # 4. Save results
            self._save_optimization_history()
            
            logger.info("Completed scheduled analysis cycle successfully")
            
        except Exception as e:
            logger.error(f"Error during scheduled analysis cycle: {str(e)}")
    
    def _fetch_campaign_data(self) -> List[Dict[str, Any]]:
        """Fetch campaign data from Google Ads API"""
        try:
            logger.info(f"Fetching campaign data for the last {self.days_to_analyze} days...")
            campaigns = self.ads_api.get_campaign_performance(days_ago=self.days_to_analyze)
            logger.info(f"Fetched data for {len(campaigns)} campaigns")
            
            # Store current campaign data
            timestamp = datetime.now().isoformat()
            with open(f"history/campaigns_{timestamp.replace(':', '-')}.json", "w") as f:
                json.dump(campaigns, f, indent=2)
                
            return campaigns
            
        except Exception as e:
            logger.error(f"Error fetching campaign data: {str(e)}")
            return []
    
    def _fetch_keyword_data(self, campaign_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch keyword data for a specific campaign or all enabled campaigns"""
        try:
            if campaign_id:
                logger.info(f"Fetching keyword data for campaign {campaign_id}...")
            else:
                logger.info(f"Fetching keyword data for all enabled campaigns...")
                
            keywords = self.ads_api.get_keyword_performance(days_ago=self.days_to_analyze, campaign_id=campaign_id)
            logger.info(f"Fetched data for {len(keywords)} keywords")
            
            return keywords
            
        except Exception as e:
            logger.error(f"Error fetching keyword data: {str(e)}")
            return []
    
    def _analyze_campaign(self, campaign: Dict[str, Any]):
        """Analyze a single campaign and apply optimizations"""
        campaign_id = campaign.get('id')
        campaign_name = campaign.get('name', 'Unknown Campaign')
        
        logger.info(f"Analyzing campaign: {campaign_name} (ID: {campaign_id})")
        
        # Skip paused or removed campaigns
        if campaign.get('status') != 'ENABLED':
            logger.info(f"Skipping non-enabled campaign: {campaign_name}")
            return
        
        # Fetch keywords for this campaign
        keywords = self._fetch_keyword_data(campaign_id)
        
        if not keywords:
            logger.info(f"No keywords found for campaign {campaign_name}")
            return
            
        # Get optimization suggestions using the AI optimizer
        try:
            logger.info(f"Generating optimization suggestions for campaign {campaign_name}...")
            suggestions = self.optimizer.get_optimization_suggestions([campaign], keywords)
            
            # Apply optimizations automatically
            if suggestions:
                self._apply_optimizations(suggestions, campaign)
                
        except Exception as e:
            logger.error(f"Error analyzing campaign {campaign_name}: {str(e)}")
    
    def _analyze_account(self, campaigns: List[Dict[str, Any]]):
        """Analyze the entire account and implement account-level optimizations"""
        logger.info("Analyzing account-level performance...")
        
        try:
            # Fetch all keywords for a comprehensive analysis
            all_keywords = self._fetch_keyword_data()
            
            if not all_keywords:
                logger.warning("No keywords found for account-level analysis")
                return
                
            # Generate account-level optimization suggestions
            logger.info("Generating account-level optimization suggestions...")
            suggestions = self.optimizer.get_optimization_suggestions(campaigns, all_keywords)
            
            # Apply optimizations automatically
            if suggestions:
                self._apply_optimizations(suggestions)
                
            logger.info("Account-level analysis complete")
            
        except Exception as e:
            logger.error(f"Error during account-level analysis: {str(e)}")
    
    def _apply_optimizations(self, suggestions: Dict[str, Any], campaign: Optional[Dict[str, Any]] = None):
        """Apply optimization suggestions with safety checks"""
        try:
            # Track which optimizations were applied and their results
            applied_optimizations = []
            
            # Process campaign recommendations
            if 'campaign_recommendations' in suggestions:
                for rec in suggestions.get('campaign_recommendations', []):
                    # Apply safety checks for campaign changes
                    if self._is_safe_campaign_change(rec, campaign):
                        # Apply the campaign recommendation
                        result = self._apply_campaign_recommendation(rec)
                        if result:
                            applied_optimizations.append({
                                "type": "campaign",
                                "timestamp": datetime.now().isoformat(),
                                "recommendation": rec,
                                "result": result
                            })
            
            # Process keyword recommendations
            if 'keyword_recommendations' in suggestions:
                for rec in suggestions.get('keyword_recommendations', []):
                    # Apply safety checks for keyword changes
                    if self._is_safe_keyword_change(rec):
                        # Apply the keyword recommendation
                        result = self._apply_keyword_recommendation(rec)
                        if result:
                            applied_optimizations.append({
                                "type": "keyword",
                                "timestamp": datetime.now().isoformat(),
                                "recommendation": rec,
                                "result": result
                            })
            
            # Add to optimization history
            if applied_optimizations:
                self.optimization_history.extend(applied_optimizations)
                logger.info(f"Applied {len(applied_optimizations)} optimizations successfully")
            else:
                logger.info("No optimizations were applied (either none recommended or failed safety checks)")
                
        except Exception as e:
            logger.error(f"Error applying optimizations: {str(e)}")
    
    def _is_safe_campaign_change(self, recommendation: Dict[str, Any], campaign: Optional[Dict[str, Any]]) -> bool:
        """Apply safety checks for campaign-level changes"""
        # Budget change safety checks
        if "budget" in recommendation.get("recommendation", "").lower():
            try:
                # Get current budget
                current_budget = 0
                if campaign:
                    current_budget = campaign.get('budget_amount', 0)
                
                # Extract recommended budget
                if "recommended_budget" in recommendation:
                    new_budget = float(recommendation["recommended_budget"])
                    
                    # Calculate percentage change
                    if current_budget > 0:
                        change_pct = abs((new_budget - current_budget) / current_budget * 100)
                        
                        # Check if change exceeds threshold
                        if change_pct > self.max_daily_budget_change_pct:
                            logger.warning(f"Budget change of {change_pct:.1f}% exceeds safety threshold of {self.max_daily_budget_change_pct}%")
                            return False
            except Exception as e:
                logger.error(f"Error in budget safety check: {str(e)}")
                return False
        
        return True
    
    def _is_safe_keyword_change(self, recommendation: Dict[str, Any]) -> bool:
        """Apply safety checks for keyword-level changes"""
        # Bid change safety checks
        if recommendation.get("type") == "ADJUST_BID":
            try:
                current_bid = float(recommendation.get("current_bid", 0))
                recommended_bid = float(recommendation.get("recommended_bid", 0))
                
                if current_bid > 0:
                    change_pct = abs((recommended_bid - current_bid) / current_bid * 100)
                    
                    # Check if change exceeds threshold
                    if change_pct > self.max_bid_change_pct:
                        logger.warning(f"Bid change of {change_pct:.1f}% exceeds safety threshold of {self.max_bid_change_pct}%")
                        return False
            except Exception as e:
                logger.error(f"Error in bid safety check: {str(e)}")
                return False
        
        return True
    
    def _apply_campaign_recommendation(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a campaign-level recommendation"""
        try:
            campaign_name = recommendation.get('campaign_name', 'Unknown')
            action = recommendation.get('recommendation', '')
            
            logger.info(f"Applying campaign recommendation to {campaign_name}: {action}")
            
            # Handle budget adjustments
            if "budget" in action.lower() and "campaign_id" in recommendation:
                campaign_id = recommendation.get("campaign_id")
                
                if "recommended_budget" in recommendation:
                    new_budget = float(recommendation["recommended_budget"])
                    budget_micros = int(new_budget * 1000000)  # Convert to micros
                    
                    # Apply the budget change
                    changes = {"budget_micros": budget_micros}
                    success, message = self.ads_api.apply_optimization(
                        optimization_type="budget_adjustment",
                        entity_type="campaign",
                        entity_id=campaign_id,
                        changes=changes
                    )
                    
                    return {
                        "success": success,
                        "message": message,
                        "action": "budget_adjustment",
                        "campaign_id": campaign_id,
                        "new_budget": new_budget
                    }
            
            # Handle campaign status changes
            if any(x in action.lower() for x in ["pause", "enable", "resume"]):
                campaign_id = recommendation.get("campaign_id")
                
                if "pause" in action.lower():
                    status = "PAUSED"
                elif "enable" in action.lower() or "resume" in action.lower():
                    status = "ENABLED"
                else:
                    return {"success": False, "message": "Unclear status change in recommendation"}
                
                # Apply the status change
                changes = {"status": status}
                success, message = self.ads_api.apply_optimization(
                    optimization_type="status_change",
                    entity_type="campaign",
                    entity_id=campaign_id,
                    changes=changes
                )
                
                return {
                    "success": success,
                    "message": message,
                    "action": "status_change",
                    "campaign_id": campaign_id,
                    "new_status": status
                }
                
            return {"success": False, "message": "No actionable recommendation found"}
            
        except Exception as e:
            logger.error(f"Error applying campaign recommendation: {str(e)}")
            return {"success": False, "message": str(e)}
    
    def _apply_keyword_recommendation(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a keyword-level recommendation"""
        try:
            keyword = recommendation.get('keyword', 'Unknown')
            action_type = recommendation.get('type', '')
            
            logger.info(f"Applying {action_type} to keyword '{keyword}'")
            
            # Handle bid adjustments
            if action_type == "ADJUST_BID" and "keyword_id" in recommendation:
                keyword_id = recommendation.get("keyword_id")
                
                if "recommended_bid" in recommendation:
                    new_bid = float(recommendation["recommended_bid"])
                    bid_micros = int(new_bid * 1000000)  # Convert to micros
                    
                    # Apply the bid change
                    changes = {"bid_micros": bid_micros}
                    success, message = self.ads_api.apply_optimization(
                        optimization_type="bid_adjustment",
                        entity_type="keyword",
                        entity_id=keyword_id,
                        changes=changes
                    )
                    
                    return {
                        "success": success,
                        "message": message,
                        "action": "bid_adjustment",
                        "keyword_id": keyword_id,
                        "new_bid": new_bid
                    }
            
            # Handle keyword status changes
            if action_type == "PAUSE" and "keyword_id" in recommendation:
                keyword_id = recommendation.get("keyword_id")
                
                # Apply the status change to pause the keyword
                changes = {"status": "PAUSED"}
                success, message = self.ads_api.apply_optimization(
                    optimization_type="status_change",
                    entity_type="keyword",
                    entity_id=keyword_id,
                    changes=changes
                )
                
                return {
                    "success": success,
                    "message": message,
                    "action": "status_change",
                    "keyword_id": keyword_id,
                    "new_status": "PAUSED"
                }
                
            return {"success": False, "message": "No actionable recommendation found"}
            
        except Exception as e:
            logger.error(f"Error applying keyword recommendation: {str(e)}")
            return {"success": False, "message": str(e)}
    
    def schedule_daily_analysis(self, hour: int = 4, minute: int = 0):
        """Schedule the agent to run daily analysis at specified time"""
        self.execution_hour = hour
        self.analysis_frequency = "daily"
        
        schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(self.run_scheduled_analysis)
        logger.info(f"Scheduled daily analysis at {hour:02d}:{minute:02d}")
    
    def schedule_weekly_analysis(self, day_of_week: str = "monday", hour: int = 4, minute: int = 0):
        """Schedule the agent to run weekly analysis at specified time"""
        self.execution_hour = hour
        self.analysis_frequency = "weekly"
        
        getattr(schedule.every(), day_of_week.lower()).at(f"{hour:02d}:{minute:02d}").do(self.run_scheduled_analysis)
        logger.info(f"Scheduled weekly analysis on {day_of_week} at {hour:02d}:{minute:02d}")
    
    def run_scheduler(self):
        """Run the scheduler in the background"""
        stop_event = threading.Event()
        
        def run_scheduling():
            while not stop_event.is_set():
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        # Start scheduler in a separate thread
        scheduler_thread = threading.Thread(target=run_scheduling)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        
        logger.info("Scheduler started in background thread")
        
        return stop_event  # Return the event so it can be set to stop the scheduler
    
    def run_immediate_analysis(self):
        """Run a complete analysis cycle immediately"""
        logger.info("Starting immediate analysis cycle")
        self.run_scheduled_analysis()

# Main function to run the autonomous agent
def main():
    try:
        # Initialize the autonomous agent
        agent = GoogleAdsAutonomousAgent()
        
        # Schedule daily analysis at 4 AM
        agent.schedule_daily_analysis(hour=4, minute=0)
        
        # Also run immediately upon startup
        agent.run_immediate_analysis()
        
        # Run the scheduler in the background
        stop_event = agent.run_scheduler()
        
        logger.info("Autonomous agent running. Press Ctrl+C to stop.")
        
        try:
            # Keep the main thread alive until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping autonomous agent...")
            stop_event.set()
            
    except Exception as e:
        logger.error(f"Error in autonomous agent: {str(e)}")

if __name__ == "__main__":
    main() 