import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import threading
import os
import json
import traceback
import re
import uuid
import copy
import math

from config import load_config
from ads_api import GoogleAdsAPI
from optimizer import AdsOptimizer
from scheduler import AdsScheduler
from logger import AdsAgentLogger
from chat_interface import ChatInterface

# Define task types for the scheduler
TASK_TYPES = {
    "fetch_campaign_data": {
        "name": "Fetch Campaign Data",
        "description": "Retrieve campaign performance data from Google Ads",
        "icon": "📊",
        "params": ["days"]
    },
    "campaign_analysis": {
        "name": "Campaign Analysis",
        "description": "Analyze campaigns and generate campaign-level optimization suggestions",
        "icon": "🔍",
        "params": ["days"]
    },
    "fetch_keyword_data": {
        "name": "Fetch Keyword Data",
        "description": "Retrieve keyword performance data from Google Ads",
        "icon": "🔤",
        "params": ["days", "campaign_id"]
    },
    "keyword_analysis": {
        "name": "Keyword Analysis",
        "description": "Analyze keywords and generate keyword-level optimization suggestions",
        "icon": "🔎",
        "params": ["days", "campaign_id"]
    },
    "comprehensive_analysis": {
        "name": "Comprehensive Analysis",
        "description": "Analyze both campaigns and keywords for complete optimization",
        "icon": "🧠",
        "params": ["days", "campaign_id"]
    },
    "apply_optimizations": {
        "name": "Apply Pending Optimizations",
        "description": "Apply pending optimization suggestions automatically",
        "icon": "✅",
        "params": []
    }
}

MAX_KEYWORDS_PER_CALL = 1000 # Max keywords to send in one API call (Gemini can handle up to 1M tokens)

# Set page configuration
st.set_page_config(
    page_title="Google Ads Optimization Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    .status-container {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .log-entry-info {
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-bottom: 0.25rem;
        background-color: #f0f2f6;
    }
    .log-entry-warning {
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-bottom: 0.25rem;
        background-color: #fff3cd;
    }
    .log-entry-error {
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-bottom: 0.25rem;
        background-color: #f8d7da;
    }
    .chat-message-user {
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        background-color: #e9ecef;
        max-width: 80%;
        margin-left: auto;
    }
    .chat-message-assistant {
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        background-color: #d1e7dd;
        max-width: 80%;
    }
    .suggestion-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
        background-color: #f8f9fa;
    }
    .suggestion-card-applied {
        border-left: 5px solid #198754;
    }
    .suggestion-card-pending {
        border-left: 5px solid #0d6efd;
    }
    .suggestion-card-failed {
        border-left: 5px solid #dc3545;
    }
    .task-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
        background-color: #f8f9fa;
    }
    .task-status-scheduled {
        color: #0d6efd;
    }
    .task-status-running {
        color: #fd7e14;
    }
    .task-status-completed {
        color: #198754;
    }
    .task-status-failed {
        color: #dc3545;
    }
    .keyword-metrics {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .edit-field {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to initialize session state
def init_session_state():
    # Default session state variables
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.config = None
        st.session_state.ads_api = None
        st.session_state.optimizer = None
        st.session_state.scheduler = None
        st.session_state.chat_interface = None
        st.session_state.ppc_agent = None  # Autonomous PPC agent
        st.session_state.logger = None
        st.session_state.campaigns = []
        st.session_state.keywords = []
        st.session_state.suggestions = []
        st.session_state.scheduler_running = False
        st.session_state.scheduler_thread = None
        st.session_state.chat_messages = []
        st.session_state.edit_suggestions = {}
        st.session_state.edit_suggestion_id = None
        st.session_state.active_tasks = {}
        st.session_state.auto_accept_edits = True  # Add auto-accept setting, default to True

# Initialize components
def initialize_components():
    if not st.session_state.initialized:
        try:
            # Initialize logger first and store in session state
            if st.session_state.logger is None:
                st.session_state.logger = AdsAgentLogger()

            # Load configuration
            st.session_state.logger.info("Loading configuration from .env file...")
            st.session_state.config = load_config()
            
            # Initialize APIs
            st.session_state.logger.info("Initializing Google Ads API client...")
            st.session_state.ads_api = GoogleAdsAPI(st.session_state.config['google_ads'])
            
            # Use the correct config section for the optimizer (Google AI)
            st.session_state.logger.info("Initializing AdsOptimizer with Google AI...")
            st.session_state.optimizer = AdsOptimizer(st.session_state.config['google_ai'])
            
            # Initialize autonomous PPC agent
            st.session_state.logger.info("Initializing Autonomous PPC Agent...")
            
            # Set agent configuration from config or use defaults
            agent_config = st.session_state.config.get('agent', {})
            if not agent_config:
                # Use default settings if not specified in config
                agent_config = {
                    'auto_implement_threshold': 80,
                    'max_daily_budget_change_pct': 20,
                    'min_data_points_required': 30,
                    'max_keyword_bid_adjustment': 50
                }
                
            # Import the AutonomousPPCAgent
            from autonomous_ppc_agent import AutonomousPPCAgent
            
            # Create the agent
            st.session_state.ppc_agent = AutonomousPPCAgent(
                st.session_state.ads_api,
                st.session_state.optimizer,
                agent_config
            )
            
            # Initialize scheduler with logger
            st.session_state.logger.info("Initializing Scheduler...")
            st.session_state.scheduler = AdsScheduler(logger=st.session_state.logger)
            
            # Initialize chat interface
            st.session_state.logger.info("Initializing Chat Interface...")
            st.session_state.chat_interface = ChatInterface(
                st.session_state.ads_api,
                st.session_state.optimizer,
                st.session_state.logger
            )
            
            # Start the scheduler
            st.session_state.scheduler.start()
            st.session_state.scheduler_running = True
            
            # IMPORTANT: No data should be fetched on startup
            # Campaign and keyword data should only be loaded when explicitly requested by the user
            # or when needed by the AI for a specific analysis
            
            st.session_state.initialized = True
            st.session_state.logger.info("Application initialized successfully")
            
        except Exception as e:
            # Ensure logger exists before logging exception
            if st.session_state.logger:
                 st.session_state.logger.exception(f"Error initializing application: {str(e)}")
            else:
                 # Fallback if logger failed to initialize
                 print(f"CRITICAL ERROR during logger initialization: {str(e)}")
                 traceback.print_exc()
            st.error(f"Failed to initialize application: {str(e)}")

# Function to run scheduler in a separate thread
def run_scheduler_thread(days=30, hour=9, minute=0, frequency='daily', day_of_week=None, task_type="comprehensive_analysis", campaign_id=None):
    """
    Run the scheduler thread that will execute specified tasks at scheduled times.
    
    Args:
        days (int): Number of days to look back for data (1-365)
        hour (int): Hour to run the task (0-23)
        minute (int): Minute to run the task (0-59)
        frequency (str): Frequency of execution ('daily', 'weekly', 'once')
        day_of_week (str, optional): Day of week for weekly schedules
        task_type (str): Type of task to execute (e.g., "fetch_campaign_data", "comprehensive_analysis")
        campaign_id (str, optional): Campaign ID to analyze (if task requires it)
    """
    def run_task():
        try:
            st.session_state.logger.info(f"Running scheduled {task_type} for the last {days} days...")
            
            # Different logic based on task type
            if task_type == "fetch_campaign_data":
                campaigns = st.session_state.ads_api.get_campaign_performance(days_ago=days)
                st.session_state.campaigns = campaigns
                st.session_state.logger.info(f"Scheduled campaign data fetch completed: {len(campaigns)} campaigns retrieved")
                return f"Retrieved {len(campaigns)} campaigns from the last {days} days"
                
            elif task_type == "fetch_keyword_data":
                keywords = st.session_state.ads_api.get_keyword_performance(days_ago=days, campaign_id=campaign_id)
                st.session_state.keywords = keywords
                campaign_info = f"for campaign {campaign_id}" if campaign_id else "across all campaigns"
                st.session_state.logger.info(f"Scheduled keyword data fetch completed: {len(keywords)} keywords retrieved {campaign_info}")
                return f"Retrieved {len(keywords)} keywords from the last {days} days {campaign_info}"
                
            elif task_type == "campaign_analysis":
                # Fetch campaign data and generate suggestions
                campaigns = st.session_state.ads_api.get_campaign_performance(days_ago=days)
                st.session_state.campaigns = campaigns
                
                suggestions = st.session_state.optimizer.get_optimization_suggestions(campaigns)
                st.session_state.suggestions = suggestions
                
                suggestion_count = len(suggestions) if isinstance(suggestions, list) else 0
                st.session_state.logger.info(f"Scheduled campaign analysis completed: {len(campaigns)} campaigns analyzed, {suggestion_count} suggestions generated")
                
                # Add to chat history
                if st.session_state.chat_interface:
                    st.session_state.chat_interface.add_message(
                        'system', 
                        f"Scheduled campaign analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
                        f"{len(campaigns)} campaigns analyzed, {suggestion_count} suggestions generated."
                    )
                return f"Analyzed {len(campaigns)} campaigns, generated {suggestion_count} suggestions"
                
            elif task_type == "keyword_analysis":
                # Fetch both campaign and keyword data
                campaigns = st.session_state.ads_api.get_campaign_performance(days_ago=days)
                st.session_state.campaigns = campaigns
                
                keywords = st.session_state.ads_api.get_keyword_performance(days_ago=days, campaign_id=campaign_id)
                st.session_state.keywords = keywords
                
                suggestions = st.session_state.optimizer.get_optimization_suggestions(campaigns, keywords)
                st.session_state.suggestions = suggestions
                
                suggestion_count = len(suggestions) if isinstance(suggestions, list) else 0
                campaign_info = f"for campaign {campaign_id}" if campaign_id else "across all campaigns"
                st.session_state.logger.info(f"Scheduled keyword analysis completed: {len(keywords)} keywords analyzed {campaign_info}, {suggestion_count} suggestions generated")
                
                # Add to chat history
                if st.session_state.chat_interface:
                    st.session_state.chat_interface.add_message(
                        'system', 
                        f"Scheduled keyword analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
                        f"{len(keywords)} keywords analyzed {campaign_info}, {suggestion_count} suggestions generated."
                    )
                return f"Analyzed {len(keywords)} keywords {campaign_info}, generated {suggestion_count} suggestions"
                
            elif task_type == "comprehensive_analysis" or task_type == "optimize":
                # Fetch both campaign and keyword data for comprehensive analysis
                campaigns = st.session_state.ads_api.get_campaign_performance(days_ago=days)
                st.session_state.campaigns = campaigns
                
                keywords = st.session_state.ads_api.get_keyword_performance(days_ago=days, campaign_id=campaign_id)
                st.session_state.keywords = keywords
                
                suggestions = st.session_state.optimizer.get_optimization_suggestions(campaigns, keywords)
                st.session_state.suggestions = suggestions
                
                suggestion_count = len(suggestions) if isinstance(suggestions, list) else 0
                campaign_info = f"for campaign {campaign_id}" if campaign_id else "across all campaigns"
                st.session_state.logger.info(f"Scheduled comprehensive analysis completed: {len(campaigns)} campaigns and {len(keywords)} keywords analyzed, {suggestion_count} suggestions generated")
                
                # Add to chat history
                if st.session_state.chat_interface:
                    st.session_state.chat_interface.add_message(
                        'system', 
                        f"Scheduled comprehensive analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
                        f"{len(campaigns)} campaigns and {len(keywords)} keywords analyzed, {suggestion_count} suggestions generated."
                    )
                return f"Analyzed {len(campaigns)} campaigns and {len(keywords)} keywords, generated {suggestion_count} suggestions"
                
            elif task_type == "apply_optimizations":
                if st.session_state.suggestions and isinstance(st.session_state.suggestions, list):
                    success_count, failure_count = apply_all_optimizations()
                    st.session_state.logger.info(f"Scheduled optimization application completed: {success_count} applied successfully, {failure_count} failed")
                    return f"Applied {success_count} optimizations successfully, {failure_count} failed"
                else:
                    st.session_state.logger.warning("No optimization suggestions available to apply")
                    return "No optimization suggestions available to apply"
                    
            else:
                st.session_state.logger.error(f"Unknown task type: {task_type}")
                return f"Error: Unknown task type {task_type}"
                
        except Exception as e:
            error_message = f"Error in scheduled task: {str(e)}"
            st.session_state.logger.exception(error_message)
            if st.session_state.chat_interface:
                st.session_state.chat_interface.add_message(
                    'system', 
                    f"Scheduled task failed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {str(e)}"
                )
            return error_message
    
    # Create scheduler with the task
    scheduler = AdsScheduler(logger=st.session_state.logger)
    
    # Schedule based on frequency
    if frequency == 'daily':
        task_id = scheduler.schedule_daily(
            function=run_task,
            hour=hour,
            minute=minute,
            name=f"{task_type} (last {days} days)"
        )
        st.session_state.logger.info(f"Scheduled daily task (ID: {task_id}) at {hour:02d}:{minute:02d}")
    elif frequency == 'weekly' and day_of_week:
        task_id = scheduler.schedule_weekly(
            function=run_task,
            day_of_week=day_of_week,
            hour=hour,
            minute=minute,
            name=f"{task_type} (last {days} days)"
        )
        st.session_state.logger.info(f"Scheduled weekly task (ID: {task_id}) on {day_of_week} at {hour:02d}:{minute:02d}")
    elif frequency == 'once':
        task_id = scheduler.schedule_once(
            function=run_task,
            hour=hour,
            minute=minute,
            name=f"{task_type} (last {days} days)"
        )
        st.session_state.logger.info(f"Scheduled one-time task (ID: {task_id}) at {hour:02d}:{minute:02d}")
    
    st.session_state.scheduler_running = True
    
    try:
        scheduler.start()
    except Exception as e:
        st.session_state.logger.exception(f"Scheduler error: {str(e)}")
    finally:
        st.session_state.scheduler_running = False

# Function to fetch campaign data
def fetch_campaign_data(days=30):
    try:
        st.session_state.logger.info(f"Fetching campaign data for the last {days} days...")
        campaigns = st.session_state.ads_api.get_campaign_performance(days_ago=days)
        
        # Ensure all campaigns have metric fields populated, even if zero
        for campaign in campaigns:
            # Set default values for key metrics if not present
            for metric in ['clicks', 'impressions', 'conversions', 'cost', 'average_cpc']:
                if metric not in campaign or campaign[metric] is None:
                    campaign[metric] = 0
            
            # Calculate derived metrics for each campaign
            impressions = campaign.get('impressions', 0)
            clicks = campaign.get('clicks', 0)
            conversions = campaign.get('conversions', 0)
            cost = campaign.get('cost', 0)
            
            # CTR (Click-Through Rate)
            if impressions > 0:
                campaign['ctr'] = (clicks / impressions) * 100
            else:
                campaign['ctr'] = 0
                
            # Conversion Rate
            if clicks > 0:
                campaign['conversion_rate'] = (conversions / clicks) * 100
            else:
                campaign['conversion_rate'] = 0
                
            # CPA (Cost Per Acquisition/Conversion)
            if conversions > 0:
                campaign['cpa'] = cost / conversions
            else:
                campaign['cpa'] = 0
        
        st.session_state.campaigns = campaigns
        st.session_state.logger.info(f"Fetched and processed data for {len(campaigns)} campaigns")
        return st.session_state.campaigns
    except Exception as e:
        st.session_state.logger.exception(f"Error fetching campaign data: {str(e)}")
        st.error(f"Failed to fetch campaign data: {str(e)}")
        return []

# Function to fetch keyword data
def fetch_keyword_data(days=30, campaign_id=None):
    try:
        # Check if we have a specific campaign_id
        if campaign_id:
            campaign_info = f"for campaign ID {campaign_id}"
        else:
            # No specific campaign - check if we have campaigns in session state first
            if 'campaigns' in st.session_state and st.session_state.campaigns:
                enabled_campaigns = st.session_state.campaigns
                campaign_info = "from all ENABLED campaigns"
            else:
                # Only fetch campaigns if not already in session state
                st.session_state.logger.info("No campaigns in session state, fetching enabled campaigns...")
                enabled_campaigns = st.session_state.ads_api.get_campaign_performance(days_ago=days)
                if not enabled_campaigns:
                    st.session_state.logger.warning("No ENABLED campaigns found. Cannot fetch keywords.")
                    st.warning("No enabled campaigns found. Please check your Google Ads account.")
                    return []
                campaign_info = "from all ENABLED campaigns" 
            
        st.session_state.logger.info(f"Fetching keyword data {campaign_info} for the last {days} days...")
        keywords = st.session_state.ads_api.get_keyword_performance(days_ago=days, campaign_id=campaign_id)
        
        # Process and enhance keywords with derived metrics
        for keyword in keywords:
            # Set default values for key metrics if not present
            for metric in ['clicks', 'impressions', 'conversions', 'cost', 'average_cpc']:
                if metric not in keyword or keyword[metric] is None:
                    keyword[metric] = 0
            
            # Calculate derived metrics for each keyword
            impressions = keyword.get('impressions', 0)
            clicks = keyword.get('clicks', 0)
            conversions = keyword.get('conversions', 0)
            cost = keyword.get('cost', 0)
            
            # CTR (Click-Through Rate)
            if impressions > 0:
                keyword['ctr'] = (clicks / impressions) * 100
            else:
                keyword['ctr'] = 0
                
            # Conversion Rate
            if clicks > 0:
                keyword['conversion_rate'] = (conversions / clicks) * 100
            else:
                keyword['conversion_rate'] = 0
                
            # CPA (Cost Per Acquisition/Conversion)
            if conversions > 0:
                keyword['cpa'] = cost / conversions
            else:
                keyword['cpa'] = 0
        
        # Count keywords by status for logging
        if keywords:
            keyword_count = len(keywords)
            status_counts = {}
            for kw in keywords:
                status = kw.get('status', 'UNKNOWN')
                status_counts[status] = status_counts.get(status, 0) + 1
            status_info = ", ".join([f"{status}: {count}" for status, count in status_counts.items()])
            st.session_state.logger.info(f"Keyword status distribution: {status_info}")
        else:
            keyword_count = 0
                
        st.session_state.keywords = keywords
        st.session_state.logger.info(f"Fetched and processed data for {keyword_count} enabled keywords")
        return st.session_state.keywords
    except Exception as e:
        st.session_state.logger.exception(f"Error fetching keyword data: {str(e)}")
        st.error(f"Failed to fetch keyword data: {str(e)}")
        return []

# Function to get optimization suggestions
def get_optimization_suggestions(campaigns=None, keywords=None):
    if campaigns is None:
        campaigns = st.session_state.campaigns
    
    if keywords is None:
        keywords = st.session_state.keywords
        
    if not campaigns:
        st.session_state.logger.warning("No campaign data available for optimization")
        st.warning("No campaign data available for optimization. Please fetch campaign data first.")
        return None
        
    try:
        all_suggestions = []
        spinner_message = "Getting optimization suggestions from Gemini..."
        with st.spinner(spinner_message):
            # Determine if keywords are provided and if batching is needed
            if keywords and len(keywords) > MAX_KEYWORDS_PER_CALL:
                st.session_state.logger.info(f"Keyword count ({len(keywords)}) exceeds limit ({MAX_KEYWORDS_PER_CALL}). Processing in batches...")
                num_batches = math.ceil(len(keywords) / MAX_KEYWORDS_PER_CALL)

                for j in range(num_batches):
                    start_index = j * MAX_KEYWORDS_PER_CALL
                    end_index = start_index + MAX_KEYWORDS_PER_CALL
                    keyword_batch = keywords[start_index:end_index]

                    # Update spinner message for progress
                    st.spinner(f"{spinner_message} (Batch {j + 1}/{num_batches})")
                    st.session_state.logger.info(f"Sending batch {j + 1}/{num_batches} ({len(keyword_batch)} keywords) to Gemini...")

                    try:
                        batch_suggestions = st.session_state.optimizer.get_optimization_suggestions(campaigns, keyword_batch)

                        # Check if the response is in the new format (with campaign_recommendations and keyword_recommendations)
                        if isinstance(batch_suggestions, dict):
                            formatted_suggestions = []
                            
                            # Process campaign recommendations
                            if 'campaign_recommendations' in batch_suggestions:
                                for i, rec in enumerate(batch_suggestions.get('campaign_recommendations', [])):
                                    formatted_suggestions.append({
                                        "index": len(all_suggestions) + len(formatted_suggestions) + 1,
                                        "title": f"Campaign: {rec.get('campaign_name', 'All Campaigns')}",
                                        "action_type": "CAMPAIGN_OPTIMIZATION",
                                        "entity_type": "campaign",
                                        "entity_id": rec.get('campaign_id', 'general'),
                                        "change": rec.get('recommendation', 'No specific action'),
                                        "rationale": rec.get('issue', '') + ". " + rec.get('expected_impact', ''),
                                        "priority": rec.get('priority', 'MEDIUM'),
                                        "status": "pending"
                                    })
                            
                            # Process keyword recommendations
                            if 'keyword_recommendations' in batch_suggestions:
                                for i, rec in enumerate(batch_suggestions.get('keyword_recommendations', [])):
                                    # Skip error types
                                    if rec.get('type') == 'ERROR':
                                        continue
                                        
                                    action_type = rec.get('type', '')
                                    if action_type == 'ADJUST_BID':
                                        action = 'BID_ADJUSTMENT'
                                    elif action_type == 'PAUSE':
                                        action = 'STATUS_CHANGE'
                                    else:
                                        action = action_type
                                        
                                    formatted_suggestions.append({
                                        "index": len(all_suggestions) + len(formatted_suggestions) + 1,
                                        "title": f"Keyword: {rec.get('keyword', 'Unknown')}",
                                        "action_type": action,
                                        "entity_type": "keyword",
                                        "entity_id": rec.get('keyword_id', rec.get('keyword', 'unknown')),
                                        "change": rec.get('details', f"{action_type} recommendation"),
                                        "rationale": rec.get('rationale', 'No rationale provided'),
                                        "current_value": rec.get('current_bid', 0.0),
                                        "edited_value": rec.get('recommended_bid', None),
                                        "priority": rec.get('priority', 'MEDIUM'),
                                        "status": "pending"
                                    })
                            
                            if formatted_suggestions:
                                all_suggestions.extend(formatted_suggestions)
                                st.session_state.logger.info(f"Formatted and added {len(formatted_suggestions)} suggestions from batch {j + 1}")
                            else:
                                st.session_state.logger.warning(f"Batch {j + 1} had no usable recommendations in new format")
                        # Handle the legacy list format
                        elif isinstance(batch_suggestions, list):
                            valid_batch_suggestions = [s for s in batch_suggestions if s.get('action_type') != 'ERROR']
                            if valid_batch_suggestions:
                                all_suggestions.extend(valid_batch_suggestions)
                                st.session_state.logger.info(f"Received {len(valid_batch_suggestions)} valid suggestions for batch {j + 1}.")
                            elif batch_suggestions: # Log if only errors were returned
                                st.session_state.logger.warning(f"Gemini returned only errors for batch {j + 1}: {batch_suggestions}")
                        else:
                            st.session_state.logger.warning(f"Received unexpected suggestion format for batch {j + 1}: {batch_suggestions}")

                    except Exception as batch_e:
                        st.session_state.logger.error(f"Error processing keyword batch {j + 1}: {str(batch_e)}")
                        # Add an error suggestion for this batch
                        all_suggestions.append({
                            "index": len(all_suggestions) + 1,
                            "title": f"Error Processing Keyword Batch {j + 1}",
                            "action_type": "ERROR", "entity_type": "system", "entity_id": f"batch_{j+1}",
                            "change": f"Investigate batch processing error: {str(batch_e)}",
                            "rationale": f"Processing failed for keyword batch {j+1}.",
                            "status": "error"
                        })
                st.session_state.logger.info(f"Finished processing {num_batches} keyword batches. Total suggestions: {len(all_suggestions)}.")

            # Process all keywords at once if count is within limit (or no keywords provided)
            else:
                keyword_info = f"({len(keywords)} keywords)" if keywords else "(campaign level only)"
                st.session_state.logger.info(f"Sending campaign data {keyword_info} to Gemini for analysis...")
                campaign_suggestions = st.session_state.optimizer.get_optimization_suggestions(campaigns, keywords)

                # Check if the response is in the new format (with campaign_recommendations and keyword_recommendations)
                if isinstance(campaign_suggestions, dict):
                    formatted_suggestions = []
                    
                    # Process campaign recommendations
                    if 'campaign_recommendations' in campaign_suggestions:
                        for i, rec in enumerate(campaign_suggestions.get('campaign_recommendations', [])):
                            formatted_suggestions.append({
                                "index": len(formatted_suggestions) + 1,
                                "title": f"Campaign: {rec.get('campaign_name', 'All Campaigns')}",
                                "action_type": "CAMPAIGN_OPTIMIZATION",
                                "entity_type": "campaign",
                                "entity_id": rec.get('campaign_id', 'general'),
                                "change": rec.get('recommendation', 'No specific action'),
                                "rationale": rec.get('issue', '') + ". " + rec.get('expected_impact', ''),
                                "priority": rec.get('priority', 'MEDIUM'),
                                "status": "pending"
                            })
                    
                    # Process keyword recommendations
                    if 'keyword_recommendations' in campaign_suggestions:
                        for i, rec in enumerate(campaign_suggestions.get('keyword_recommendations', [])):
                            # Skip error types
                            if rec.get('type') == 'ERROR':
                                continue
                                
                            action_type = rec.get('type', '')
                            if action_type == 'ADJUST_BID':
                                action = 'BID_ADJUSTMENT'
                            elif action_type == 'PAUSE':
                                action = 'STATUS_CHANGE'
                            else:
                                action = action_type
                                
                            formatted_suggestions.append({
                                "index": len(formatted_suggestions) + 1,
                                "title": f"Keyword: {rec.get('keyword', 'Unknown')}",
                                "action_type": action,
                                "entity_type": "keyword",
                                "entity_id": rec.get('keyword_id', rec.get('keyword', 'unknown')),
                                "change": rec.get('details', f"{action_type} recommendation"),
                                "rationale": rec.get('rationale', 'No rationale provided'),
                                "current_value": rec.get('current_bid', 0.0),
                                "edited_value": rec.get('recommended_bid', None),
                                "priority": rec.get('priority', 'MEDIUM'),
                                "status": "pending"
                            })
                    
                    if formatted_suggestions:
                        all_suggestions.extend(formatted_suggestions)
                        st.session_state.logger.info(f"Formatted and added {len(formatted_suggestions)} suggestions from new response format")
                    else:
                        st.session_state.logger.warning(f"New response format had no usable recommendations: {campaign_suggestions}")
                # Handle the legacy list format
                elif isinstance(campaign_suggestions, list):
                    valid_suggestions = [s for s in campaign_suggestions if s.get('action_type') != 'ERROR']
                    if valid_suggestions:
                        all_suggestions.extend(valid_suggestions)
                        st.session_state.logger.info(f"Received {len(valid_suggestions)} valid suggestions.")
                    elif campaign_suggestions: # Log if only errors were returned
                        st.session_state.logger.warning(f"Gemini returned only errors: {campaign_suggestions}")
                else:
                    st.session_state.logger.warning(f"Received unexpected suggestion format: {campaign_suggestions}")

            # Store combined suggestions
            st.session_state.suggestions = all_suggestions
            suggestion_count = len(st.session_state.suggestions)
            st.session_state.logger.info(f"Total optimization suggestions received: {suggestion_count}")

            # Auto-apply suggestions if enabled
            auto_accept = st.session_state.get('auto_accept_edits', True)
            if auto_accept and suggestion_count > 0:
                st.session_state.logger.info("Auto-accept enabled. Automatically applying all suggestions...")
                success_count, failure_count = apply_all_optimizations()
                st.session_state.logger.info(f"Auto-applied {success_count} suggestions successfully. {failure_count} failed.")

            return st.session_state.suggestions

    except Exception as e:
        st.session_state.logger.exception(f"Error getting optimization suggestions: {str(e)}")
        st.error(f"Failed to get optimization suggestions: {str(e)}")
        return None

# Function to get proper keyword criterion ID
def get_keyword_criterion_id(keyword_text, campaign_id=None, ad_group_id=None):
    """
    Get the proper criterion ID for a keyword given its text.
    
    Args:
        keyword_text (str): The text of the keyword
        campaign_id (str, optional): Campaign ID to narrow the search
        ad_group_id (str, optional): Ad group ID to narrow the search
        
    Returns:
        str: The criterion ID in the proper format 
        bool: True if successful, False otherwise
    """
    try:
        # Clean up keyword text to handle special formatting
        clean_keyword = keyword_text
        
        # Remove any "[EXACT]", "[BROAD]", "[PHRASE]" match type indicators
        match_type_patterns = [r'\s*\[(EXACT|BROAD|PHRASE)\]\s*$', r'\s*\[(Exact|Broad|Phrase)\]\s*$']
        for pattern in match_type_patterns:
            clean_keyword = re.sub(pattern, '', clean_keyword)
        
        # Remove extra quotes that might be in the keyword
        clean_keyword = clean_keyword.replace("''", "'").replace('""', '"')
        
        # Remove any leading/trailing quotes and whitespace
        clean_keyword = clean_keyword.strip("'\" \t\n")
        
        st.session_state.logger.info(f"Attempting to find criterion ID for keyword: '{keyword_text}' (cleaned to: '{clean_keyword}')")
        
        # Ensure we have access to the customer ID
        if not hasattr(st.session_state, 'ads_api') or not st.session_state.ads_api:
            st.session_state.logger.error("Ads API not initialized, cannot construct resource name")
            return None, False
            
        customer_id = st.session_state.ads_api.customer_id
        
        # Fetch recent keyword data if needed
        keywords = []
        if hasattr(st.session_state, 'keywords') and st.session_state.keywords:
            keywords = st.session_state.keywords
        elif hasattr(st.session_state, 'chat_interface') and st.session_state.chat_interface and hasattr(st.session_state.chat_interface, 'latest_keywords') and st.session_state.chat_interface.latest_keywords:
            keywords = st.session_state.chat_interface.latest_keywords
        else:
            # Fetch fresh keyword data
            st.session_state.logger.info("No existing keyword data found, fetching now")
            keywords = fetch_keyword_data(days=30, campaign_id=campaign_id)
            
        if not keywords:
            st.session_state.logger.error("No keyword data available to search for criterion ID")
            return None, False
        
        # Filter to find exact keyword - try both original and cleaned versions
        for kw in keywords:
            kw_text = kw.get('keyword_text', '').lower()
            
            # Skip negative keywords - can't adjust bids on them
            if kw.get('is_negative', False):
                continue
                
            # Try to match by both original and cleaned keyword
            if kw_text == keyword_text.lower() or kw_text == clean_keyword.lower():
                # First check if we already have the resource_name
                if 'resource_name' in kw and kw['resource_name']:
                    resource_name = kw['resource_name']
                    # Make sure it has the proper format
                    if not resource_name.startswith('customers/'):
                        resource_name = f"customers/{customer_id}/{resource_name.lstrip('/')}"
                    st.session_state.logger.info(f"Found resource name: {resource_name}")
                    return resource_name, True
                    
                # Next try to construct from criterion_id and ad_group_id
                elif 'criterion_id' in kw and kw['criterion_id'] and 'ad_group_id' in kw and kw['ad_group_id']:
                    resource_name = f"customers/{customer_id}/adGroups/{kw['ad_group_id']}/criteria/{kw['criterion_id']}"
                    st.session_state.logger.info(f"Constructed resource name: {resource_name}")
                    return resource_name, True
                    
                # Log what fields we do have for debugging
                else:
                    fields = ', '.join([f"{key}={value}" for key, value in kw.items() if key in ['ad_group_id', 'criterion_id', 'resource_name']])
                    st.session_state.logger.warning(f"Found keyword '{keyword_text}' but missing required fields for resource name. Available fields: {fields}")
        
        st.session_state.logger.error(f"Could not find keyword '{keyword_text}' (or '{clean_keyword}') in available data")
        return None, False
        
    except Exception as e:
        st.session_state.logger.exception(f"Error getting criterion ID for keyword '{keyword_text}': {str(e)}")
        return None, False

# Function to apply a single optimization
def apply_optimization(suggestion):
    """
    Apply a single optimization suggestion to Google Ads.
    
    Args:
        suggestion (dict): Optimization suggestion dictionary
        
    Returns:
        bool: Success status
        str: Status message
    """
    if not suggestion or not isinstance(suggestion, dict):
        return False, "Invalid suggestion format"
    
    try:
        entity_type = suggestion.get('entity_type')
        entity_id = suggestion.get('entity_id')
        action_type = suggestion.get('action_type')
        
        if not entity_type or not entity_id or not action_type:
            return False, "Missing required suggestion fields"
        
        # Handle CAMPAIGN_OPTIMIZATION action type by converting to budget_adjustment
        if action_type == 'CAMPAIGN_OPTIMIZATION':
            # Extract campaign ID
            campaign_id = entity_id
            if campaign_id == 'general':
                # Try to find the campaign ID from the context
                campaign_name = suggestion.get('title', '').replace('Campaign: ', '')
                for campaign in st.session_state.campaigns:
                    if campaign_name == campaign.get('name'):
                        campaign_id = campaign.get('id')
                        break
                
                if campaign_id == 'general':
                    return False, f"Could not find campaign ID for '{campaign_name}'"
            
            # Determine budget amount from suggestion
            # Default to a small budget (e.g., $10 per day)
            budget_amount = 10.0
            
            # Try to extract budget from the change field
            change_text = suggestion.get('change', '')
            budget_matches = re.findall(r'\$(\d+(?:\.\d+)?)-?\$?(\d+(?:\.\d+)?)', change_text)
            if budget_matches:
                # Use the average if a range is specified
                if len(budget_matches[0]) > 1 and budget_matches[0][1]:
                    min_budget = float(budget_matches[0][0])
                    max_budget = float(budget_matches[0][1])
                    budget_amount = (min_budget + max_budget) / 2
                else:
                    budget_amount = float(budget_matches[0][0])
            
            # Convert to budget adjustment
            action_type = 'BUDGET_ADJUSTMENT'
            changes = {
                'budget_micros': int(budget_amount * 1000000)
            }
            
            st.session_state.logger.info(f"Converted CAMPAIGN_OPTIMIZATION to BUDGET_ADJUSTMENT with amount ${budget_amount}")
            
            # Apply the optimization as a budget adjustment
            success, message = st.session_state.ads_api.apply_optimization(
                optimization_type=action_type.lower(),
                entity_type=entity_type,
                entity_id=campaign_id,
                changes=changes
            )
            
            return success, message
        
        # Check for negative keywords before proceeding
        if entity_type.lower() == 'keyword' and action_type == 'BID_ADJUSTMENT':
            if suggestion.get('is_negative', False):
                return False, "Cannot adjust bid for negative keyword"
                
            # Check if the keyword is marked as negative in our data
            keyword_data = None
            if hasattr(st.session_state, 'keywords') and st.session_state.keywords:
                for kw in st.session_state.keywords:
                    if (kw.get('resource_name') == entity_id or 
                        kw.get('keyword_text', '').lower() == entity_id.lower()):
                        if kw.get('is_negative', False):
                            return False, f"Cannot adjust bid for negative keyword: {kw.get('keyword_text', entity_id)}"
        
        # Handle the ADD action type for keywords
        if entity_type.lower() == 'keyword' and action_type == 'ADD':
            # Extract necessary data
            keyword_text = entity_id
            
            # Try to determine match type from text or change description
            match_type = 'EXACT'  # Default to exact match
            match_type_patterns = {
                'EXACT': r'\[EXACT\]|\[exact\]',
                'PHRASE': r'\[PHRASE\]|\[phrase\]',
                'BROAD': r'\[BROAD\]|\[broad\]'
            }
            
            # Extract match type from keyword text if included
            for mt, pattern in match_type_patterns.items():
                if re.search(pattern, keyword_text):
                    match_type = mt
                    # Remove the match type from the keyword text
                    keyword_text = re.sub(pattern, '', keyword_text).strip()
                    break
            
            # Also check change and rationale text for match type
            change_text = suggestion.get('change', '')
            rationale_text = suggestion.get('rationale', '')
            combined_text = change_text + " " + rationale_text
            
            # Check for match type in description
            if 'exact match' in combined_text.lower():
                match_type = 'EXACT'
            elif 'phrase match' in combined_text.lower():
                match_type = 'PHRASE'
            elif 'broad match' in combined_text.lower():
                match_type = 'BROAD'
            
            # Try to determine campaign from rationale or change text
            campaign_id = None
            
            # Look for campaign name in rationale or change text
            if 'campaign_id' in suggestion:
                campaign_id = suggestion.get('campaign_id')
            else:
                for campaign in st.session_state.campaigns:
                    campaign_name = campaign.get('name', '')
                    if campaign_name in rationale_text or campaign_name in change_text:
                        campaign_id = campaign.get('id')
                        st.session_state.logger.info(f"Found campaign ID {campaign_id} for keyword addition")
                        break
            
            if not campaign_id:
                # As a fallback, use the first campaign
                if st.session_state.campaigns:
                    campaign_id = st.session_state.campaigns[0].get('id')
                    st.session_state.logger.info(f"Using first campaign (ID: {campaign_id}) for keyword addition")
                else:
                    return False, "Could not determine which campaign to add the keyword to"
            
            # Try to determine a bid based on similar keywords
            bid = 1.0  # Default bid $1.00
            if hasattr(st.session_state, 'keywords') and st.session_state.keywords:
                # Find similar keywords to use as a reference
                similar_keywords = [k for k in st.session_state.keywords if k.get('keyword_text') and keyword_text.lower() in k.get('keyword_text', '').lower()]
                if similar_keywords:
                    # Use the average bid of similar keywords
                    bids = [k.get('current_bid', 0) for k in similar_keywords if k.get('current_bid', 0) > 0]
                    if bids:
                        bid = sum(bids) / len(bids)
                        st.session_state.logger.info(f"Using average bid ${bid:.2f} from similar keywords")
            
            # Convert bid to micros
            bid_micros = int(bid * 1000000)
            
            # Apply the optimization (add keyword)
            changes = {
                'campaign_id': str(campaign_id),
                'ad_group_id': None,  # Let the API find an appropriate ad group
                'keyword_text': keyword_text,
                'match_type': match_type,
                'bid_micros': bid_micros
            }
            
            st.session_state.logger.info(f"Adding keyword '{keyword_text}' with {match_type} match type and ${bid:.2f} bid to campaign {campaign_id}")
            
            # Apply the optimization
            return st.session_state.ads_api.apply_optimization(
                optimization_type='add',
                entity_type='keyword',
                entity_id=str(campaign_id),  # Use campaign ID as entity ID
                changes=changes
            )
        
        # Handle the case where entity_id might be a keyword text instead of a proper criterion ID
        if entity_type.lower() == 'keyword':
            # Check if entity_id appears to be a keyword text rather than a proper criterion ID
            if '/' not in entity_id and not entity_id.startswith('customers/'):
                st.session_state.logger.info(f"Entity ID '{entity_id}' appears to be keyword text, attempting to find criterion ID")
                # Try to get the proper criterion ID
                criterion_id, success = get_keyword_criterion_id(entity_id)
                if success:
                    # Use the proper criterion ID
                    st.session_state.logger.info(f"Found criterion ID '{criterion_id}' for keyword '{entity_id}'")
                    entity_id = criterion_id
                else:
                    return False, f"Could not find criterion ID for keyword: {entity_id}"
        
        # Prepare changes based on action type
        changes = {}
        
        if action_type == 'BID_ADJUSTMENT':
            # Get the edited or original change_value
            if 'edited_value' in suggestion and suggestion['edited_value'] is not None:
                try:
                    change_value = float(suggestion['edited_value'])
                    # Convert to micros (multiply by 1,000,000)
                    bid_micros = int(change_value * 1000000)
                    changes['bid_micros'] = bid_micros
                except (ValueError, TypeError) as e:
                    return False, f"Invalid bid value '{suggestion['edited_value']}': {str(e)}"
            elif 'change_value' in suggestion:
                change_type = suggestion['change_value'].get('type')
                value = suggestion['change_value'].get('value')
                
                # If we have a current value to work with
                if 'current_value' in suggestion and suggestion['current_value'] is not None:
                    try:
                        current_value = float(suggestion['current_value'])
                        
                        if change_type == 'percentage_increase':
                            new_value = current_value * (1 + value/100)
                        elif change_type == 'percentage_decrease':
                            new_value = current_value * (1 - value/100)
                        elif change_type == 'absolute':
                            new_value = value
                        else:
                            return False, f"Unsupported change type: {change_type}"
                        
                        # Convert to micros (multiply by 1,000,000)
                        bid_micros = int(new_value * 1000000)
                        changes['bid_micros'] = bid_micros
                    except (ValueError, TypeError) as e:
                        return False, f"Invalid current bid value '{suggestion['current_value']}': {str(e)}"
                else:
                    return False, "Cannot apply bid adjustment without current value"
            else:
                return False, "Missing change value for bid adjustment"
                
        elif action_type == 'STATUS_CHANGE':
            # Get the edited or original status
            if 'edited_value' in suggestion and suggestion['edited_value'] is not None:
                status = suggestion['edited_value']
            elif 'change_value' in suggestion and 'type' in suggestion['change_value'] and suggestion['change_value']['type'] == 'status':
                status = suggestion['change_value']['value']
            else:
                # Try to extract status from the change description
                change = suggestion.get('change', '').lower()
                if 'pause' in change:
                    status = 'PAUSED'
                elif 'enable' in change or 'resume' in change:
                    status = 'ENABLED'
                elif 'remove' in change:
                    status = 'REMOVED'
                else:
                    return False, "Cannot determine status change from suggestion"
            
            changes['status'] = status
        
        elif action_type == 'BUDGET_ADJUSTMENT':
            # Similar to bid adjustment but for campaign budgets
            if 'edited_value' in suggestion and suggestion['edited_value'] is not None:
                try:
                    budget_micros = int(float(suggestion['edited_value']) * 1000000)
                    changes['budget_micros'] = budget_micros
                except (ValueError, TypeError) as e:
                    return False, f"Invalid budget value '{suggestion['edited_value']}': {str(e)}"
            else:
                return False, "Missing edited value for budget adjustment"
        
        else:
            return False, f"Unsupported action type: {action_type}"
        
        # Apply the optimization
        st.session_state.logger.info(f"Applying {action_type} to {entity_type} {entity_id} with changes: {changes}")
        success, message = st.session_state.ads_api.apply_optimization(
            optimization_type=action_type.lower(),
            entity_type=entity_type,
            entity_id=entity_id,
            changes=changes
        )
        
        if success:
            st.session_state.logger.info(f"Successfully applied optimization: {message}")
        else:
            st.session_state.logger.error(f"Failed to apply optimization: {message}")
        
        return success, message
        
    except Exception as e:
        error_message = f"Error applying optimization: {str(e)}"
        st.session_state.logger.exception(error_message)
        return False, error_message

# Function to apply all pending optimizations
def apply_all_optimizations():
    """
    Apply all pending optimization suggestions.
    
    Returns:
        int: Number of successfully applied optimizations
        int: Number of failed optimizations
    """
    if not st.session_state.suggestions or not isinstance(st.session_state.suggestions, list):
        st.warning("No optimization suggestions to apply")
        return 0, 0
    
    success_count = 0
    failure_count = 0
    
    with st.spinner("Applying optimization suggestions..."):
        for i, suggestion in enumerate(st.session_state.suggestions):
            # Skip already applied suggestions
            if suggestion.get('applied', False):
                continue
                
            success, message = apply_optimization(suggestion)
            
            # Update suggestion status
            suggestion['applied'] = success
            suggestion['status'] = 'applied' if success else 'failed'
            suggestion['result_message'] = message
            
            if success:
                success_count += 1
            else:
                failure_count += 1
    
    return success_count, failure_count

# Function to start scheduler
def start_scheduler(days=30, hour=9, minute=0, frequency='daily', day_of_week=None, task_type="comprehensive_analysis", campaign_id=None):
    """
    Start a scheduler thread to run the specified task at scheduled times.
    
    Args:
        days (int): Number of days to analyze
        hour (int): Hour to run the task (0-23)
        minute (int): Minute to run the task (0-59)
        frequency (str): Frequency of execution ('daily', 'weekly', 'once')
        day_of_week (str, optional): Day of week for weekly schedules
        task_type (str): Type of task to execute
        campaign_id (str, optional): Campaign ID if required by the task
    """
    if st.session_state.scheduler_running:
        st.session_state.logger.warning("Scheduler is already running")
        st.warning("Scheduler is already running")
        return
        
    try:
        st.session_state.logger.info(f"Starting scheduler for {task_type} with parameters: days={days}, hour={hour}, minute={minute}, frequency={frequency}, day_of_week={day_of_week}")
        
        # Create and start thread
        scheduler_thread = threading.Thread(
            target=run_scheduler_thread,
            args=(days, hour, minute, frequency, day_of_week, task_type, campaign_id),
            daemon=True
        )
        
        scheduler_thread.start()
        st.session_state.scheduler_thread = scheduler_thread
        st.session_state.scheduler_running = True
        
        st.session_state.logger.info("Scheduler started successfully")
        st.success(f"Scheduler for {task_type} started successfully. Will run {frequency} at {hour:02d}:{minute:02d}")
        
    except Exception as e:
        st.session_state.logger.exception(f"Error starting scheduler: {str(e)}")
        st.error(f"Failed to start scheduler: {str(e)}")

# Function to stop scheduler
def stop_scheduler():
    if not st.session_state.scheduler_running:
        st.session_state.logger.warning("No scheduler is currently running")
        st.warning("No scheduler is currently running")
        return
        
    try:
        # Can't directly stop the thread, so we set the flag to False
        # and the scheduler will check this flag and exit gracefully
        st.session_state.scheduler_running = False
        
        st.session_state.logger.info("Scheduler will stop after the current iteration")
        st.info("Scheduler will stop after the current iteration")
        
    except Exception as e:
        st.session_state.logger.exception(f"Error stopping scheduler: {str(e)}")
        st.error(f"Failed to stop scheduler: {str(e)}")

# Function to render campaigns data as a table and charts
def render_campaign_data(campaigns):
    """Render campaign performance data in a table."""
    if not campaigns:
        st.info("No campaign data available. Use the sidebar to fetch data.")
        return

    st.subheader("Campaign Performance Overview")

    # Convert list of dicts to DataFrame
    df_campaigns = pd.DataFrame(campaigns)

    # Define columns to display (including pre-calculated metrics)
    display_columns = ['id', 'name', 'status']
    
    # Define metric columns that *might* exist
    metric_columns = ['clicks', 'impressions', 'conversions', 'cost', 'average_cpc', 'ctr', 'conversion_rate', 'cpa']
    
    # Add metric columns to display if they exist
    for col in metric_columns:
        if col in df_campaigns.columns:
            display_columns.append(col)
    
    # Filter DataFrame to display columns and calculate available metrics
    try:
        display_df = df_campaigns[display_columns].copy()
        
        # Format metrics for display
        if 'ctr' in display_df.columns:
            display_df['ctr'] = display_df['ctr'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "0.00%")
            
        if 'conversion_rate' in display_df.columns:
            display_df['conversion_rate'] = display_df['conversion_rate'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "0.00%")
            
        if 'cost' in display_df.columns:
            display_df['cost'] = display_df['cost'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "$0.00")
            
        if 'average_cpc' in display_df.columns:
            display_df['average_cpc'] = display_df['average_cpc'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "$0.00")
            
        if 'cpa' in display_df.columns:
            display_df['cpa'] = display_df['cpa'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "$0.00")
            
        # Rename columns for better display
        column_renames = {
            'id': 'ID',
            'name': 'Campaign Name',
            'status': 'Status',
            'clicks': 'Clicks',
            'impressions': 'Impressions',
            'conversions': 'Conversions',
            'cost': 'Cost',
            'average_cpc': 'Avg CPC',
            'ctr': 'CTR',
            'conversion_rate': 'Conv. Rate',
            'cpa': 'Cost per Conv.'
        }
        display_df = display_df.rename(columns={col: column_renames.get(col, col) for col in display_df.columns})
        
    except KeyError:
        # Fallback if columns are missing
        available_cols = [col for col in display_columns if col in df_campaigns.columns]
        display_df = df_campaigns[available_cols].copy()
    
    # Calculate totals safely
    totals = {}
    for col in ['clicks', 'impressions', 'conversions', 'cost']:
        if col in df_campaigns.columns:
            # Convert to numeric with errors='coerce' to handle non-numeric values
            numeric_values = pd.to_numeric(df_campaigns[col], errors='coerce')
            # Replace NaN with 0 before summing
            numeric_values = numeric_values.fillna(0)
            total = numeric_values.sum()
            # Format differently based on the metric
            if col == 'cost':
                totals[f'Total {col.capitalize()}'] = f"${total:.2f}"
            else:
                totals[f'Total {col.capitalize()}'] = int(total)
        else:
            totals[f'Total {col.capitalize()}'] = 'N/A' # Indicate if metric wasn't fetched
            
    # Calculate averages/rates safely
    avg_metrics = {}
    
    # Convert metric values to numbers for calculation
    clicks = 0
    impressions = 0 
    cost = 0
    conversions = 0
    
    if 'clicks' in df_campaigns.columns:
        clicks = pd.to_numeric(df_campaigns['clicks'], errors='coerce').fillna(0).sum()
    if 'impressions' in df_campaigns.columns:
        impressions = pd.to_numeric(df_campaigns['impressions'], errors='coerce').fillna(0).sum()
    if 'cost' in df_campaigns.columns:
        cost = pd.to_numeric(df_campaigns['cost'], errors='coerce').fillna(0).sum()
    if 'conversions' in df_campaigns.columns:
        conversions = pd.to_numeric(df_campaigns['conversions'], errors='coerce').fillna(0).sum()

    # Calculate CTR (Click-Through Rate)
    if impressions > 0:
        ctr = (clicks / impressions) * 100
        avg_metrics['Avg CTR'] = f"{ctr:.2f}%"
    else:
        avg_metrics['Avg CTR'] = '0.00%'

    # Calculate CPC (Cost Per Click)
    if clicks > 0:
        cpc = cost / clicks
        avg_metrics['Avg CPC'] = f"${cpc:.2f}"
    else:
        avg_metrics['Avg CPC'] = '$0.00'

    # Calculate Conversion Rate
    if clicks > 0:
        conv_rate = (conversions / clicks) * 100
        avg_metrics['Avg Conv Rate'] = f"{conv_rate:.2f}%"
    else:
        avg_metrics['Avg Conv Rate'] = '0.00%'
        
    # Calculate CPA (Cost Per Acquisition/Conversion)
    if conversions > 0:
        cpa = cost / conversions
        avg_metrics['Avg CPA'] = f"${cpa:.2f}"
    else:
        avg_metrics['Avg CPA'] = '$0.00'

    # Display Totals and Averages
    st.metric("Total Campaigns", len(df_campaigns))
    cols = st.columns(len(totals) + len(avg_metrics))
    i = 0
    for key, value in totals.items():
        cols[i].metric(key, value)
        i += 1
    for key, value in avg_metrics.items():
        cols[i].metric(key, value)
        i += 1
        
    # Display the table with available columns
    st.dataframe(display_df, use_container_width=True)

# Function to render chat interface
def render_chat_interface():
    """Render the chat interface."""
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Get chat history from chat interface if available
    if st.session_state.chat_interface:
        # Initialize chat with informative message about data context
        if not st.session_state.chat_messages:
            # Ensure we have data context
            campaigns, keywords = st.session_state.chat_interface.ensure_data_context(fetch_keywords=False)
            if campaigns or keywords:
                data_summary = f"📊 **Data available**: {len(campaigns)} campaigns and {len(keywords) if keywords else 0} keywords from the last 30 days."
            else:
                data_summary = "No campaign data loaded yet. Use the command 'fetch campaign data' to get started."
            
            assistant_message = f"""
Hi! I'm your Google Ads PPC Expert assistant. I can help you analyze your campaigns and provide optimization suggestions.

{data_summary}

You can ask me questions about your account performance, or use commands like:
- "Analyze my campaigns"
- "Fetch keyword data for the last 14 days"
- "What are my best performing keywords?"
- "Give me optimization suggestions for my campaigns"

How can I help optimize your Google Ads campaigns today?
"""
            st.session_state.chat_messages.append({"role": "assistant", "content": assistant_message})
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_messages):
            if message["role"] == "user":
                st.markdown(f"<div class='chat-message-user'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-message-assistant'><strong>PPC Expert:</strong> {message['content']}</div>", unsafe_allow_html=True)
    
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_area("Your message:", key="chat_input", height=100)
        with col2:
            st.write("")
            st.write("")
            submit_button = st.form_submit_button("Send")
        
        if submit_button and user_input.strip():
            # Add user message to chat
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            
            # Process the message
            if st.session_state.chat_interface:
                with st.spinner("Thinking..."):
                    response, result = st.session_state.chat_interface.process_user_message(user_input)
                    
                    # Add response to chat
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    
                    # Handle command results if any
                    if result and 'command' in result:
                        if result['command'] == 'fetch_data' and 'result' in result:
                            st.session_state.campaigns = result['result']
                        elif result['command'] == 'analyze' and 'result' in result:
                            st.session_state.suggestions = result['result']
                
                # Force refresh to show new messages
                st.rerun()
            else:
                st.error("Chat interface not initialized")
    
    # Data refresh indicator
    if st.session_state.chat_interface and st.session_state.chat_interface.last_data_refresh:
        last_refresh = st.session_state.chat_interface.last_data_refresh
        time_diff = datetime.now() - last_refresh
        if time_diff.total_seconds() < 60:
            refresh_time = "less than a minute ago"
        elif time_diff.total_seconds() < 3600:
            refresh_time = f"{int(time_diff.total_seconds() / 60)} minutes ago"
        else:
            refresh_time = f"{int(time_diff.total_seconds() / 3600)} hours ago"
        
        # Display last data refresh time
        st.caption(f"Last data refresh: {refresh_time} | Campaign count: {len(st.session_state.chat_interface.latest_campaigns) if st.session_state.chat_interface.latest_campaigns else 0} | Keyword count: {len(st.session_state.chat_interface.latest_keywords) if st.session_state.chat_interface.latest_keywords else 0}")
    
    # Refresh data button
    if st.button("Refresh Campaign & Keyword Data"):
        with st.spinner("Refreshing data..."):
            campaigns, keywords = st.session_state.chat_interface.ensure_data_context(days=30, force_refresh=True, fetch_keywords=True)
            st.success(f"Data refreshed successfully! Loaded {len(campaigns)} campaigns and {len(keywords)} keywords.")
            time.sleep(1)
            st.rerun()

# Function to display logs in the UI
def render_logs():
    st.subheader("📋 Application Logs")
    log_level = st.selectbox("Filter logs by level", ["ALL", "INFO", "WARNING", "ERROR", "DEBUG"], index=0)

    # Use session state logger
    if st.session_state.logger:
        if log_level == "ALL":
            logs_to_display = st.session_state.logger.get_recent_logs()
        else:
            logs_to_display = st.session_state.logger.get_recent_logs(level=log_level)

        if logs_to_display:
            log_container = st.container()
            with log_container:
                # Display logs in reverse chronological order (newest first)
                for timestamp, level, message in reversed(logs_to_display):
                    log_class = f"log-entry-{level.lower()}"
                    st.markdown(f'<div class="{log_class}"><b>{timestamp} [{level}]</b>: {message}</div>', unsafe_allow_html=True)
        else:
            st.info("No logs to display for the selected level.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Recent Logs Display"):
                st.session_state.logger.clear_recent_logs()
                st.rerun()
        with col2:
            latest_log_file = st.session_state.logger.get_latest_log_file()
            if latest_log_file and os.path.exists(latest_log_file):
                with open(latest_log_file, "r", encoding="utf-8") as f: # Ensure correct encoding read
                    st.download_button(
                        label="Download Full Log File",
                        data=f.read(),
                        file_name=os.path.basename(latest_log_file),
                        mime="text/plain"
                    )
            else:
                 st.info("No log file found.")
    else:
        st.warning("Logger not initialized yet.")

# Function to show an editable suggestion
def render_editable_suggestion(suggestion, index):
    """
    Render a single suggestion with editing capabilities.
    
    Args:
        suggestion (dict): Suggestion to render
        index (int): Index of the suggestion
        
    Returns:
        dict: The edited suggestion
    """
    # Initialize suggestions dict if not exists
    if 'edit_suggestions' not in st.session_state:
        st.session_state.edit_suggestions = {}
    
    if str(index) not in st.session_state.edit_suggestions:
        st.session_state.edit_suggestions[str(index)] = copy.deepcopy(suggestion)
    
    edited_suggestion = st.session_state.edit_suggestions[str(index)]
    
    # Check for negative keywords - can't adjust bids on these
    is_negative_keyword = False
    if edited_suggestion.get('entity_type', '').lower() == 'keyword':
        entity_id = edited_suggestion.get('entity_id', '')
        action_type = edited_suggestion.get('action_type', '')
        
        # If this is a bid adjustment on a keyword, check if it's negative
        if action_type == 'BID_ADJUSTMENT':
            # Look up keyword data
            keyword_data = None
            if hasattr(st.session_state, 'keywords') and st.session_state.keywords:
                # Try to find keyword in current keyword data
                for kw in st.session_state.keywords:
                    if (kw.get('resource_name') == entity_id or 
                        kw.get('keyword_text', '').lower() == entity_id.lower()):
                        keyword_data = kw
                        break
            
            # If we found the keyword and it's negative, mark it
            if keyword_data and keyword_data.get('is_negative', False):
                is_negative_keyword = True
                edited_suggestion['is_negative'] = True
                if not edited_suggestion.get('status') == 'failed':
                    edited_suggestion['status'] = 'failed'
                    edited_suggestion['result_message'] = "Cannot adjust bid for negative keyword"
    
    # Check if we should auto-apply this suggestion
    auto_accept = st.session_state.get('auto_accept_edits', True)
    if (auto_accept and 
        not edited_suggestion.get('applied', False) and 
        edited_suggestion.get('status') != 'failed' and
        not is_negative_keyword):
        # Auto-apply the suggestion
        success, message = apply_optimization(edited_suggestion)
        
        # Update suggestion status
        edited_suggestion['applied'] = success
        edited_suggestion['status'] = 'applied' if success else 'failed'
        edited_suggestion['result_message'] = message
    
    # Determine card style based on status
    card_class = "suggestion-card"
    if edited_suggestion.get('applied', False):
        card_class += " suggestion-card-applied"
    elif edited_suggestion.get('status') == 'failed':
        card_class += " suggestion-card-failed"
    else:
        card_class += " suggestion-card-pending"
    
    # Render suggestion card
    st.markdown(f"""<div class="{card_class}">""", unsafe_allow_html=True)
    
    # Add selection checkbox
    is_selected = edited_suggestion.get('selected', False)
    selected = st.checkbox(f"Select", value=is_selected, key=f"select_{index}")
    edited_suggestion['selected'] = selected
    
    # Display suggestion header
    action_type = edited_suggestion.get('action_type', 'UNKNOWN')
    title = edited_suggestion.get('title', 'Untitled Suggestion')
    
    # Create columns for the header
    header_col1, header_col2 = st.columns([4, 1])
    
    with header_col1:
        st.markdown(f"### {index+1}. {title}")
        st.markdown(f"*Action Type: {action_type}*")
        
        # Show negative keyword warning if applicable
        if is_negative_keyword:
            st.warning("⚠️ This is a negative keyword - cannot adjust bid")
    
    with header_col2:
        # Display status or apply button
        if edited_suggestion.get('applied', False):
            st.success("Applied ✓")
        elif edited_suggestion.get('status') == 'failed':
            st.error("Failed ✗")
        elif is_negative_keyword and action_type == 'BID_ADJUSTMENT':
            st.error("Cannot Apply")
        else:
            if st.button(f"Apply #{index+1}", key=f"apply_button_{index}"):
                apply_result = apply_optimization(edited_suggestion)
                st.rerun()
    
    # Display suggestion details
    entity_type = edited_suggestion.get('entity_type', 'unknown')
    entity_id = edited_suggestion.get('entity_id', 'unknown')
    
    st.markdown(f"**Entity**: {entity_type.upper()} - {entity_id}")
    
    # Create columns for editable fields
    col1, col2 = st.columns(2)
    
    with col1:
        # Render appropriate editor based on action type
        if action_type == 'BID_ADJUSTMENT':
            st.markdown("#### Bid Adjustment")
            
            # Get current bid value with fallback to 0
            current_bid = edited_suggestion.get('current_value', 0)
            if current_bid is None:
                current_bid = 0
                
            # Ensure current_bid is a float before formatting
            try:
                current_bid = float(current_bid)
            except (ValueError, TypeError):
                current_bid = 0.0
                
            # Show current bid (read-only)
            st.text_input("Current Bid", value=f"${current_bid:.2f}", disabled=True, key=f"current_bid_{index}")
            
            # Create a min_value that is less than the current bid to avoid errors
            min_value = 0.01  # Default minimum
            
            # Calculate new bid based on percentage or absolute change
            change_value = edited_suggestion.get('change_value', {})
            change_type = change_value.get('type') if change_value else None
            
            if change_type == 'percentage_increase':
                percentage = change_value.get('value', 10)
                new_bid = current_bid * (1 + percentage/100)
                bid_note = f"Increase by {percentage}%"
            elif change_type == 'percentage_decrease':
                percentage = change_value.get('value', 10)
                new_bid = current_bid * (1 - percentage/100)
                bid_note = f"Decrease by {percentage}%"
            elif change_type == 'absolute':
                new_bid = change_value.get('value', current_bid)
                bid_note = f"Set to specific value"
            else:
                new_bid = current_bid * 1.1  # Default 10% increase
                bid_note = "Default 10% increase"
            
            # Ensure new_bid is at least the minimum value
            new_bid = max(float(new_bid), min_value)
            
            # Allow editing the new bid with safe min_value
            edited_bid = st.number_input(
                "New Bid",
                min_value=min_value,
                value=new_bid,
                step=0.01,
                format="%.2f",
                help="Enter the new bid amount",
                key=f"new_bid_{index}"
            )
            
            # Store the edited value
            if 'change_value' not in edited_suggestion:
                edited_suggestion['change_value'] = {}
            
            edited_suggestion['change_value']['value'] = edited_bid
            if change_type:
                edited_suggestion['change_value']['type'] = 'absolute'  # Change to absolute after editing
            
            # Show percentage change from original
            if current_bid > 0:
                pct_change = ((edited_bid / current_bid) - 1) * 100
                st.markdown(f"*{pct_change:+.1f}% from original bid*")
            
        elif action_type == 'BUDGET_ADJUSTMENT':
            st.markdown("#### Budget Adjustment")
            
            # Get current budget value with fallback to 0
            current_budget = edited_suggestion.get('current_value', 0)
            if current_budget is None:
                current_budget = 0
            
            # Ensure current_budget is a float
            try:
                current_budget = float(current_budget)
            except (ValueError, TypeError):
                current_budget = 0.0
                
            # Show current budget (read-only)
            st.text_input("Current Budget", value=f"${current_budget:.2f}", disabled=True, key=f"current_budget_{index}")
            
            # Create a min_value that is less than the current budget to avoid errors
            min_value = 0.01  # Default minimum
            
            # Calculate new budget based on percentage or absolute change
            change_value = edited_suggestion.get('change_value', {})
            change_type = change_value.get('type') if change_value else None
            
            if change_type == 'percentage_increase':
                percentage = change_value.get('value', 20)
                new_budget = current_budget * (1 + percentage/100)
                budget_note = f"Increase by {percentage}%"
            elif change_type == 'percentage_decrease':
                percentage = change_value.get('value', 20)
                new_budget = current_budget * (1 - percentage/100)
                budget_note = f"Decrease by {percentage}%"
            elif change_type == 'absolute':
                new_budget = change_value.get('value', current_budget)
                budget_note = f"Set to specific value"
            else:
                new_budget = current_budget * 1.2  # Default 20% increase
                budget_note = "Default 20% increase"
            
            # Ensure new_budget is at least the minimum value
            new_budget = max(float(new_budget), min_value)
            
            # Allow editing the new budget with safe min_value
            edited_budget = st.number_input(
                "New Budget",
                min_value=min_value,
                value=new_budget,
                step=1.0,
                format="%.2f",
                help="Enter the new daily budget amount",
                key=f"new_budget_{index}"
            )
            
            # Store the edited value
            if 'change_value' not in edited_suggestion:
                edited_suggestion['change_value'] = {}
            
            edited_suggestion['change_value']['value'] = edited_budget
            if change_type:
                edited_suggestion['change_value']['type'] = 'absolute'  # Change to absolute after editing
            
            # Show percentage change from original
            if current_budget > 0:
                pct_change = ((edited_budget / current_budget) - 1) * 100
                st.markdown(f"*{pct_change:+.1f}% from original budget*")
            
        elif action_type == 'STATUS_CHANGE':
            st.markdown("#### Status Change")
            
            # Get current status
            current_status = edited_suggestion.get('current_value', 'UNKNOWN')
            if current_status is None:
                current_status = 'UNKNOWN'
                
            # Show current status (read-only)
            st.text_input("Current Status", value=current_status, disabled=True, key=f"current_status_{index}")
            
            # Get the new status from the suggestion
            change_value = edited_suggestion.get('change_value', {})
            new_status = change_value.get('value', 'PAUSED') if change_value else 'PAUSED'
            
            # Allow selecting the new status
            edited_status = st.selectbox(
                "New Status",
                options=["ENABLED", "PAUSED", "REMOVED"],
                index=["ENABLED", "PAUSED", "REMOVED"].index(new_status) if new_status in ["ENABLED", "PAUSED", "REMOVED"] else 1,
                key=f"new_status_{index}"
            )
            
            # Store the edited value
            if 'change_value' not in edited_suggestion:
                edited_suggestion['change_value'] = {}
            
            edited_suggestion['change_value']['type'] = 'status'
            edited_suggestion['change_value']['value'] = edited_status
        
        # For other action types, show a simple text editor
        else:
            change = edited_suggestion.get('change', 'No specific change details')
            edited_change = st.text_area("Change Details", value=change, key=f"change_{index}")
            edited_suggestion['change'] = edited_change
    
    with col2:
        # Display rationale
        rationale = edited_suggestion.get('rationale', 'No rationale provided')
        st.markdown("#### Rationale")
        edited_rationale = st.text_area("Why this change is recommended", value=rationale, key=f"rationale_{index}")
        edited_suggestion['rationale'] = edited_rationale
        
        # Display result message if available
        if edited_suggestion.get('result_message'):
            st.markdown("#### Result")
            st.info(edited_suggestion['result_message'])
    
    # Close the card div
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Return the edited suggestion
    return edited_suggestion

# Function to apply selected optimizations
def apply_selected_optimizations():
    """
    Apply only the selected optimization suggestions.
    
    Returns:
        int: Number of successfully applied optimizations
        int: Number of failed optimizations
    """
    if not st.session_state.suggestions or not isinstance(st.session_state.suggestions, list):
        st.warning("No optimization suggestions to apply")
        return 0, 0
    
    success_count = 0
    failure_count = 0
    
    with st.spinner("Applying selected optimization suggestions..."):
        for i, suggestion in enumerate(st.session_state.suggestions):
            # Skip suggestions that are not selected or already applied
            if not suggestion.get('selected', False) or suggestion.get('applied', False):
                continue
                
            success, message = apply_optimization(suggestion)
            
            # Update suggestion status
            suggestion['applied'] = success
            suggestion['status'] = 'applied' if success else 'failed'
            suggestion['result_message'] = message
            
            if success:
                success_count += 1
            else:
                failure_count += 1
    
    return success_count, failure_count

# Function to render suggestions list
def render_suggestions():
    """Render the list of optimization suggestions with edit and apply functionality."""
    if not st.session_state.suggestions or not isinstance(st.session_state.suggestions, list):
        st.info("No optimization suggestions available. Use the sidebar to analyze your campaigns.")
        return
    
    st.subheader("Optimization Suggestions")
    
    # Add selection controls
    col1, col2, col3 = st.columns([2, 2, 2])
    
    # Track whether selection buttons were clicked to prevent auto-apply
    selection_clicked = False
    
    with col1:
        if st.button("Select All"):
            selection_clicked = True
            for suggestion in st.session_state.suggestions:
                if not suggestion.get('applied', False):
                    suggestion['selected'] = True
            st.rerun()
    
    with col2:
        if st.button("Deselect All"):
            selection_clicked = True
            for suggestion in st.session_state.suggestions:
                suggestion['selected'] = False
            st.rerun()
    
    with col3:
        # Count selected but not applied suggestions
        selected_count = sum(1 for s in st.session_state.suggestions 
                           if s.get('selected', False) and not s.get('applied', False))
        
        if selected_count > 0:
            if st.button(f"Apply {selected_count} Selected Suggestions"):
                success_count, failure_count = apply_selected_optimizations()
                st.success(f"Applied {success_count} selected suggestions successfully. {failure_count} failed.")
                st.rerun()
    
    # If auto-accept is enabled and there are pending suggestions, apply them automatically
    # BUT ONLY if we haven't just clicked a selection button
    auto_accept = st.session_state.get('auto_accept_edits', True)
    pending_count = sum(1 for s in st.session_state.suggestions if not s.get('applied', False))
    
    if auto_accept and pending_count > 0 and not selection_clicked:
        if not hasattr(st.session_state, 'suggestions_last_auto_applied') or st.session_state.suggestions_last_auto_applied != id(st.session_state.suggestions):
            success_count, failure_count = apply_all_optimizations()
            st.success(f"Auto-applied {success_count} suggestions successfully. {failure_count} failed.")
            # Mark that we've auto-applied this set of suggestions
            st.session_state.suggestions_last_auto_applied = id(st.session_state.suggestions)
    # Otherwise show apply all button if auto-accept is disabled
    elif not auto_accept and pending_count > 0:
        if st.button(f"Apply All Pending Suggestions ({pending_count})"):
            success_count, failure_count = apply_all_optimizations()
            st.success(f"Applied {success_count} suggestions successfully. {failure_count} failed.")
    
    # Render each suggestion with edit capability
    for i, suggestion in enumerate(st.session_state.suggestions):
        render_editable_suggestion(suggestion, i)

# Function to create a task function for a specific task type
def create_task_function(task_type, task_params):
    """
    Create a function that executes the specified task type with given parameters.
    
    Args:
        task_type (str): Type of task to execute
        task_params (dict): Parameters for the task
        
    Returns:
        callable: Function that will execute the task
    """
    def task_function():
        try:
            if task_type == 'fetch_campaign_data':
                days = task_params.get('days', 30)
                campaigns = fetch_campaign_data(days)
                return f"Fetched data for {len(campaigns)} campaigns from the last {days} days"
                
            elif task_type == 'campaign_analysis':
                days = task_params.get('days', 30)
                # Ensure we have fresh campaign data
                campaigns = fetch_campaign_data(days)
                if campaigns:
                    suggestions = get_optimization_suggestions(campaigns=campaigns, keywords=None)
                    return f"Analyzed {len(campaigns)} campaigns and generated {len(suggestions) if isinstance(suggestions, list) else 0} suggestions"
                return "No campaign data available for optimization"
                
            elif task_type == 'fetch_keyword_data':
                days = task_params.get('days', 30)
                campaign_id = task_params.get('campaign_id')
                keywords = fetch_keyword_data(days, campaign_id)
                return f"Fetched data for {len(keywords)} keywords from the last {days} days"
                
            elif task_type == 'keyword_analysis':
                days = task_params.get('days', 30)
                campaign_id = task_params.get('campaign_id')
                # Ensure we have fresh data
                campaigns = fetch_campaign_data(days)
                keywords = fetch_keyword_data(days, campaign_id)
                if campaigns and keywords:
                    suggestions = get_optimization_suggestions(campaigns=campaigns, keywords=keywords)
                    return f"Analyzed {len(keywords)} keywords across {len(campaigns)} campaigns and generated {len(suggestions) if isinstance(suggestions, list) else 0} suggestions"
                return "Failed to fetch required data for analysis"
                
            elif task_type == 'comprehensive_analysis':
                days = task_params.get('days', 30)
                campaign_id = task_params.get('campaign_id')
                # Fetch all required data
                campaigns = fetch_campaign_data(days)
                keywords = fetch_keyword_data(days, campaign_id)
                # Perform comprehensive analysis
                if campaigns and keywords:
                    st.session_state.logger.info(f"Running comprehensive analysis on {len(campaigns)} campaigns and {len(keywords)} keywords")
                    suggestions = get_optimization_suggestions(campaigns=campaigns, keywords=keywords)
                    return f"Completed comprehensive analysis of {len(campaigns)} campaigns and {len(keywords)} keywords. Generated {len(suggestions) if isinstance(suggestions, list) else 0} optimization suggestions."
                return "Failed to fetch required data for comprehensive analysis"
                
            elif task_type == 'apply_optimizations':
                success_count, failure_count = apply_all_optimizations()
                return f"Applied {success_count} optimizations successfully. {failure_count} failed."
                
            else:
                return f"Unknown task type: {task_type}"
                
        except Exception as e:
            error_message = f"Error executing task {task_type}: {str(e)}"
            st.session_state.logger.exception(error_message)
            return error_message
    
    return task_function

# Function to schedule a task
def schedule_task(task_type, schedule_type, hour, minute, day_of_week=None, **task_params):
    """
    Schedule a task to run at the specified time.
    
    Args:
        task_type (str): Type of task to schedule
        schedule_type (str): Type of schedule (daily, weekly, once)
        hour (int): Hour to run the task (0-23)
        minute (int): Minute to run the task (0-59)
        day_of_week (str, optional): Day of week for weekly schedules
        **task_params: Additional parameters for the task
        
    Returns:
        str: Task ID if successful, None otherwise
    """
    try:
        if not st.session_state.initialized or not st.session_state.scheduler:
            st.session_state.logger.error("Cannot schedule task: Scheduler not initialized")
            return None
        
        # Get task information
        task_info = TASK_TYPES.get(task_type)
        if not task_info:
            st.session_state.logger.error(f"Unknown task type: {task_type}")
            return None
        
        # Create function to execute the task
        task_function = create_task_function(task_type, task_params)
        
        # Create task name with emojis and parameters
        task_name = f"{task_info['icon']} {task_info['name']}"
        
        # Add important parameters to the task name
        param_strings = []
        if 'days' in task_params:
            param_strings.append(f"days={task_params['days']}")
        if 'campaign_id' in task_params and task_params['campaign_id']:
            param_strings.append(f"campaign_id={task_params['campaign_id']}")
        
        if param_strings:
            task_name += f" ({', '.join(param_strings)})"
        
        # Schedule the task based on schedule type
        if schedule_type == "daily":
            st.session_state.logger.info(f"Scheduling daily task {task_name} at {hour:02d}:{minute:02d}")
            task_id = st.session_state.scheduler.schedule_daily(
                function=task_function,
                hour=hour,
                minute=minute,
                name=task_name,
                args=[],
                kwargs={}
            )
        elif schedule_type == "weekly" and day_of_week:
            st.session_state.logger.info(f"Scheduling weekly task {task_name} on {day_of_week} at {hour:02d}:{minute:02d}")
            task_id = st.session_state.scheduler.schedule_weekly(
                function=task_function,
                day_of_week=day_of_week,
                hour=hour,
                minute=minute,
                name=task_name,
                args=[],
                kwargs={}
            )
        elif schedule_type == "once":
            st.session_state.logger.info(f"Scheduling one-time task {task_name} at {hour:02d}:{minute:02d}")
            task_id = st.session_state.scheduler.schedule_once(
                function=task_function,
                hour=hour,
                minute=minute,
                name=task_name,
                args=[],
                kwargs={}
            )
        else:
            st.session_state.logger.error(f"Invalid schedule type: {schedule_type}")
            return None
            
        st.session_state.logger.info(f"Scheduled task {task_name} (ID: {task_id})")
        return task_id
        
    except Exception as e:
        st.session_state.logger.exception(f"Error scheduling task: {str(e)}")
        return None

# Function to render tasks in scheduler
def render_scheduler_tasks():
    """Render the list of scheduled tasks with management options."""
    # Get all tasks from the scheduler
    tasks = st.session_state.scheduler.get_tasks()
    
    if not tasks:
        st.info("No tasks scheduled. Use the form below to create a new scheduled task.")
        return
    
    st.subheader("Scheduled Tasks")
    
    for task_id, task in tasks.items():
        # Create card for each task
        with st.container():
            st.markdown("<div class='task-card'>", unsafe_allow_html=True)
            
            # Task header with name and next run time
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(task.name)
            with col2:
                status_class = f"task-status-{task.status}"
                st.markdown(f"<span class='{status_class}'>{task.status.title()}</span>", unsafe_allow_html=True)
            
            # Task details
            schedule_type = task.schedule_type.title()
            if task.schedule_type == 'weekly' and task.day_of_week:
                schedule_info = f"{schedule_type} on {task.day_of_week.title()} at {task.hour:02d}:{task.minute:02d}"
            else:
                schedule_info = f"{schedule_type} at {task.hour:02d}:{task.minute:02d}"
            
            st.write(f"Schedule: {schedule_info}")
            
            if task.next_run:
                next_run = task.next_run
                if isinstance(next_run, str):
                    # If it's stored as ISO string, parse it
                    next_run = datetime.fromisoformat(next_run)
                st.write(f"Next run: {next_run.strftime('%Y-%m-%d %H:%M')}")
            
            if task.last_run:
                last_run = task.last_run
                if isinstance(last_run, str):
                    # If it's stored as ISO string, parse it
                    last_run = datetime.fromisoformat(last_run)
                st.write(f"Last run: {last_run.strftime('%Y-%m-%d %H:%M')}")
            
            # Task actions
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Run Now", key=f"run_{task_id}"):
                    st.info(f"Running task {task.name}...")
                    try:
                        # Run the task function directly
                        if hasattr(task, 'function') and callable(task.function):
                            result = task.function()
                            st.success(f"Task completed: {result}")
                    except Exception as e:
                        st.error(f"Error executing task: {str(e)}")
                        st.session_state.logger.exception(f"Error executing task {task.name} (ID: {task_id}): {str(e)}")
            
            with col2:
                if st.button("Remove", key=f"remove_{task_id}"):
                    if st.session_state.scheduler.remove_task(task_id):
                        # Also remove from active tasks
                        if task_id in st.session_state.active_tasks:
                            del st.session_state.active_tasks[task_id]
                        st.success(f"Task {task.name} removed from scheduler")
                        st.rerun()
                    else:
                        st.error(f"Failed to remove task {task.name}")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Show task history
    task_history = st.session_state.scheduler.get_task_history()
    if task_history:
        with st.expander("Task Execution History", expanded=False):
            for entry in reversed(task_history):
                status = entry.get('status', 'unknown')
                name = entry.get('name', 'Unknown task')
                start_time = entry.get('start_time')
                end_time = entry.get('end_time')
                
                # Format times if they are strings
                if isinstance(start_time, str):
                    start_time = datetime.fromisoformat(start_time).strftime('%Y-%m-%d %H:%M:%S')
                
                if isinstance(end_time, str):
                    end_time = datetime.fromisoformat(end_time).strftime('%Y-%m-%d %H:%M:%S')
                
                if status == 'completed':
                    st.success(f"{name} - Started: {start_time}, Completed: {end_time}")
                elif status == 'failed':
                    error = entry.get('error', 'Unknown error')
                    st.error(f"{name} - Started: {start_time}, Failed: {end_time} - Error: {error}")
                else:
                    st.info(f"{name} - Started: {start_time}, Status: {status}")

# Function to render scheduler configuration form
def render_scheduler_form():
    """Render the form for scheduling tasks."""
    st.subheader("Schedule a New Task")
    
    with st.form("schedule_task_form"):
        # Task type selection
        task_type = st.selectbox(
            "Task Type",
            options=list(TASK_TYPES.keys()),
            format_func=lambda x: TASK_TYPES[x]["name"],
            help="Select the type of task to schedule"
        )
        
        # Display task description
        st.markdown(f"**{TASK_TYPES[task_type]['description']}**")
        
        # Create columns for schedule parameters
        col1, col2 = st.columns(2)
        
        with col1:
            # Schedule type
            schedule_type = st.radio(
                "Schedule Type",
                options=["once", "daily", "weekly"],
                format_func=lambda x: x.capitalize(),
                help="Select how often to run this task"
            )
            
            # Day of week for weekly schedules
            day_of_week = None
            if schedule_type == "weekly":
                days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                day_of_week = st.selectbox(
                    "Day of Week",
                    options=days,
                    index=0,
                    help="Select which day of the week to run this task"
                ).lower()
            
            # Date for one-time tasks or start date for recurring tasks
            if schedule_type == "once":
                date = st.date_input(
                    "Date",
                    value=datetime.now().date() + timedelta(days=1),
                    min_value=datetime.now().date(),
                    help="Select the date to run this task"
                )
            else:
                date = st.date_input(
                    "Start Date",
                    value=datetime.now().date(),
                    min_value=datetime.now().date(),
                    help="Select when to start this recurring task"
                )
            
            # Time of day
            time_val = st.time_input(
                "Time",
                value=datetime.now().replace(hour=9, minute=0, second=0).time(),
                help="Select the time to run this task"
            )
        
        with col2:
            # Task parameters
            st.subheader("Task Parameters")
            
            # Days of data to analyze - with flexible range up to 365 days
            if "days" in TASK_TYPES[task_type]["params"]:
                days_ago = st.number_input(
                    "Days of Data",
                    min_value=1,
                    max_value=365,
                    value=30,
                    help="Number of days of data to analyze (1-365)"
                )
            
            # Campaign ID filter (optional)
            campaign_id = None
            if "campaign_id" in TASK_TYPES[task_type]["params"]:
                campaign_id_input = st.text_input(
                    "Campaign ID (optional)",
                    value="",
                    help="Enter a specific campaign ID to analyze, or leave blank for all campaigns"
                )
                campaign_id = campaign_id_input if campaign_id_input else None
        
        # Submit button
        submitted = st.form_submit_button("Schedule Task")
        
        if submitted:
            # Calculate hour and minute from time_input
            hour = time_val.hour
            minute = time_val.minute
            
            # Create task parameters dictionary
            task_params = {}
            if "days" in TASK_TYPES[task_type]["params"]:
                task_params["days"] = int(days_ago)
            if "campaign_id" in TASK_TYPES[task_type]["params"] and campaign_id:
                task_params["campaign_id"] = campaign_id
            
            # Schedule the task
            try:
                task_id = schedule_task(
                    task_type=task_type,
                    schedule_type=schedule_type,
                    hour=hour,
                    minute=minute,
                    day_of_week=day_of_week,
                    **task_params
                )
                
                if task_id:
                    st.success(f"Task scheduled successfully! Task ID: {task_id}")
                    # Rerun the app to refresh the UI
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to schedule task. Please check the logs for details.")
            
            except Exception as e:
                st.error(f"Error scheduling task: {str(e)}")
                st.session_state.logger.exception(f"Error scheduling task: {str(e)}")

# Function to render autonomous agent features
def render_autonomous_agent():
    """Render the autonomous agent tab with controls and insights."""
    
    st.subheader("🤖 Autonomous PPC Manager")
    
    # Agent status and configuration
    st.write("This tab controls the autonomous PPC management features, which allow the system to act as a professional Google Ads manager. The autonomous agent automatically analyzes your account and makes data-driven optimization decisions.")
    
    # Display agent configuration
    if 'ppc_agent' in st.session_state:
        agent = st.session_state.ppc_agent
        
        # Create columns for key agent settings
        settings_cols = st.columns(2)
        
        with settings_cols[0]:
            st.markdown("### Agent Settings")
            
            # Allow editing of auto-implement threshold
            new_threshold = st.slider(
                "Auto-implement confidence threshold",
                min_value=60,
                max_value=95,
                value=agent.auto_implement_threshold,
                step=5,
                help="Recommendations with confidence scores above this threshold will be automatically implemented"
            )
            
            # Update agent setting if changed
            if new_threshold != agent.auto_implement_threshold:
                agent.auto_implement_threshold = new_threshold
                st.success(f"Updated auto-implement threshold to {new_threshold}")
            
            # Allow editing of max bid adjustment
            new_bid_pct = st.slider(
                "Maximum keyword bid adjustment (%)",
                min_value=10,
                max_value=100,
                value=agent.max_keyword_bid_adjustment,
                step=5,
                help="Maximum percentage change for automatic keyword bid adjustments"
            )
            
            # Update agent setting if changed
            if new_bid_pct != agent.max_keyword_bid_adjustment:
                agent.max_keyword_bid_adjustment = new_bid_pct
                st.success(f"Updated maximum bid adjustment to {new_bid_pct}%")
            
            # Add data time range setting
            time_periods = st.multiselect(
                "Analysis Time Periods (days)",
                options=[7, 14, 30, 90],
                default=[30],
                help="Time periods to analyze for performance data comparison"
            )
            if time_periods:
                agent.time_periods = time_periods
                st.success(f"Updated analysis time periods to {time_periods} days")
                
        with settings_cols[1]:
            st.markdown("### Current Status")
            
            # Show data freshness
            if agent.last_data_refresh:
                time_diff = datetime.now() - agent.last_data_refresh
                if time_diff.total_seconds() < 3600:
                    freshness = f"Data refreshed {int(time_diff.total_seconds() / 60)} minutes ago"
                    freshness_color = "green"
                elif time_diff.total_seconds() < 86400:
                    freshness = f"Data refreshed {int(time_diff.total_seconds() / 3600)} hours ago"
                    freshness_color = "orange"
                else:
                    freshness = f"Data refreshed {int(time_diff.total_seconds() / 86400)} days ago"
                    freshness_color = "red"
                    
                st.markdown(f"**Data Freshness**: <span style='color:{freshness_color}'>{freshness}</span>", unsafe_allow_html=True)
            else:
                st.markdown("**Data Freshness**: <span style='color:red'>No data loaded yet</span>", unsafe_allow_html=True)
            
            # Show recommendation stats
            st.markdown(f"**Total Campaigns**: {len(agent.campaigns)}")
            st.markdown(f"**Total Keywords**: {len(agent.keywords)}")
            st.markdown(f"**Historical Recommendations**: {len(agent.recommendation_history)}")
            
            # Count pending recommendations
            pending_recs = sum(1 for r in agent.recommendations if r.status == 'pending')
            if pending_recs > 0:
                st.markdown(f"**Pending Recommendations**: <span style='color:orange'>{pending_recs}</span>", unsafe_allow_html=True)
    
    # Create autonomous agent action buttons
    st.markdown("### Actions")
    
    action_cols = st.columns(3)
    
    with action_cols[0]:
        if st.button("🔄 Refresh Account Data", help="Fetch fresh campaign and keyword data"):
            with st.spinner("Refreshing campaign and keyword data..."):
                try:
                    # Retrieve data for multiple time periods
                    days = 90  # Fetch 90 days by default for comprehensive analysis
                    campaigns = st.session_state.ppc_agent.refresh_campaign_data(days=days)
                    keywords = st.session_state.ppc_agent.refresh_keyword_data(days=days)
                    
                    st.session_state.campaigns = campaigns  # Update main app state
                    st.session_state.keywords = keywords    # Update main app state
                    
                    st.success(f"Successfully refreshed {len(campaigns)} campaigns and {len(keywords)} keywords for the past {days} days")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error refreshing data: {str(e)}")
    
    with action_cols[1]:
        if st.button("🔍 Analyze & Generate Recommendations", help="Analyze account and generate optimization recommendations"):
            with st.spinner("Analyzing account and generating recommendations..."):
                try:
                    # Make sure we have data first
                    if not st.session_state.ppc_agent.campaigns:
                        st.session_state.ppc_agent.refresh_campaign_data(days=90)  # Use 90 days for better analysis
                    if not st.session_state.ppc_agent.keywords:
                        st.session_state.ppc_agent.refresh_keyword_data(days=90)  # Use 90 days for better analysis
                    
                    # Generate recommendations
                    recommendations = st.session_state.ppc_agent.analyze_and_recommend(entity_type='all', days=90)
                    
                    # Store recommendations in session state for the suggestions tab too
                    # Convert to the format expected by the suggestions tab
                    tab_suggestions = []
                    for i, rec in enumerate(recommendations):
                        tab_suggestions.append({
                            "index": i,
                            "title": f"{rec.entity_type.capitalize()}: {rec.entity_id}",
                            "action_type": rec.action_type.upper(),
                            "entity_type": rec.entity_type,
                            "entity_id": rec.entity_id,
                            "change": f"Change from {rec.current_value} to {rec.recommended_value}" if rec.current_value and rec.recommended_value else rec.rationale,
                            "rationale": rec.rationale,
                            "current_value": rec.current_value,
                            "edited_value": rec.recommended_value,
                            "priority": rec.priority,
                            "status": rec.status,
                            "confidence_score": rec.confidence_score
                        })
                    
                    st.session_state.suggestions = tab_suggestions
                    
                    high_confidence = len([r for r in recommendations if r.confidence_score >= st.session_state.ppc_agent.auto_implement_threshold])
                    st.success(f"Generated {len(recommendations)} recommendations ({high_confidence} high confidence)")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error analyzing account: {str(e)}")
    
    with action_cols[2]:
        if st.button("✅ Run Daily Optimization", help="Run full autonomous optimization cycle"):
            with st.spinner("Running autonomous optimization cycle..."):
                try:
                    # Use 90 days for comprehensive analysis
                    result = st.session_state.ppc_agent.run_daily_optimization(days=90)
                    
                    if result['status'] == 'success':
                        recs = result['recommendations']
                        st.success(f"Optimization complete! Generated {recs['total']} recommendations, auto-implemented {recs['implemented']}.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Optimization failed: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error running optimization: {str(e)}")
    
    # Display pending manual review recommendations
    if 'ppc_agent' in st.session_state and st.session_state.ppc_agent.recommendations:
        pending_recs = [r for r in st.session_state.ppc_agent.recommendations if r.status == 'pending']
        if pending_recs:
            st.markdown("### Recommendations Pending Manual Review")
            
            # Create a section for all pending recommendations
            with st.expander(f"Pending Manual Review ({len(pending_recs)})", expanded=True):
                # Add buttons to apply all or selected recommendations
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Select All Pending"):
                        # Mark all pending recommendations as selected
                        for rec in pending_recs:
                            rec.selected = True
                        st.rerun()
                
                with col2:
                    # Count selected recommendations
                    selected_count = sum(1 for r in pending_recs if getattr(r, 'selected', False))
                    if selected_count > 0:
                        if st.button(f"Apply {selected_count} Selected"):
                            # Apply all selected recommendations
                            applied_count = 0
                            for rec in pending_recs:
                                if getattr(rec, 'selected', False):
                                    # Convert to format expected by apply_optimization
                                    suggestion = {
                                        "entity_type": rec.entity_type,
                                        "entity_id": rec.entity_id,
                                        "action_type": rec.action_type.upper(),
                                        "current_value": rec.current_value,
                                        "edited_value": rec.recommended_value,
                                        "rationale": rec.rationale
                                    }
                                    success, message = apply_optimization(suggestion)
                                    if success:
                                        rec.status = 'implemented'
                                        rec.implementation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        applied_count += 1
                                    else:
                                        rec.status = 'failed'
                                        rec.failure_reason = message
                            
                            st.success(f"Applied {applied_count} recommendations successfully.")
                            # Update the agent's recommendations
                            st.session_state.ppc_agent.save_recommendations()
                            st.rerun()
                
                # Display each pending recommendation with approve/reject options
                for i, rec in enumerate(pending_recs):
                    # Create card for recommendation
                    with st.container():
                        cols = st.columns([0.5, 3.5, 1])
                        with cols[0]:
                            # Add checkbox for selection
                            selected = getattr(rec, 'selected', False)
                            rec.selected = st.checkbox(f"Select", value=selected, key=f"agent_rec_{i}")
                        
                        with cols[1]:
                            # Show recommendation details
                            st.markdown(f"**{rec.entity_type.capitalize()}: {rec.entity_id}**")
                            st.markdown(f"*Action: {rec.action_type}* | Confidence: {rec.confidence_score:.1f}%")
                            if rec.current_value is not None and rec.recommended_value is not None:
                                st.markdown(f"Change from **{rec.current_value}** to **{rec.recommended_value}**")
                            st.markdown(f"Rationale: {rec.rationale}")
                        
                        with cols[2]:
                            # Add apply button for this recommendation
                            if st.button("Apply", key=f"apply_rec_{i}"):
                                # Convert to format expected by apply_optimization
                                suggestion = {
                                    "entity_type": rec.entity_type,
                                    "entity_id": rec.entity_id,
                                    "action_type": rec.action_type.upper(),
                                    "current_value": rec.current_value,
                                    "edited_value": rec.recommended_value,
                                    "rationale": rec.rationale
                                }
                                success, message = apply_optimization(suggestion)
                                if success:
                                    rec.status = 'implemented'
                                    rec.implementation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    st.success(f"Applied successfully: {message}")
                                else:
                                    rec.status = 'failed'
                                    rec.failure_reason = message
                                    st.error(f"Failed to apply: {message}")
                                
                                # Update the agent's recommendations
                                st.session_state.ppc_agent.save_recommendations()
                                st.rerun()
                        
                        st.markdown("---")
    
    # Display latest recommendations
    st.markdown("### Latest Analysis & Recommendations")
    
    if 'ppc_agent' in st.session_state and st.session_state.ppc_agent.recommendation_history:
        # Get the most recent recommendations
        recent_recs = st.session_state.ppc_agent.recommendation_history[-10:]  # Last 10 recommendations
        
        # Create an expander for recent recommendations
        with st.expander("Recent Recommendations", expanded=True):
            for rec in reversed(recent_recs):  # Show newest first
                # Determine status color
                if rec['status'] == 'implemented':
                    status_color = 'green'
                elif rec['status'] == 'failed':
                    status_color = 'red'
                else:
                    status_color = 'orange'
                
                # Format recommendation display
                st.markdown(f"""
                <div style='border-left: 4px solid {status_color}; padding-left: 10px; margin-bottom: 10px;'>
                    <strong>{rec['entity_type'].capitalize()}: {rec['entity_id']}</strong> - 
                    <span style='color:{status_color};'>{rec['status'].upper()}</span><br/>
                    <strong>Action:</strong> {rec['action_type']} | 
                    <strong>Confidence:</strong> {rec['confidence_score']:.1f}% | 
                    <strong>Impact:</strong> {rec['impact_score']:.1f}%<br/>
                    <strong>Rationale:</strong> {rec['rationale']}<br/>
                    {f"<strong>Change:</strong> {rec['current_value']} → {rec['recommended_value']}<br/>" if rec['current_value'] and rec['recommended_value'] else ""}
                    {f"<strong>Implemented:</strong> {rec['implementation_time']}" if rec['implementation_time'] else ""}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No recommendations have been generated yet. Click 'Analyze & Generate Recommendations' to get started.")
    
    # Display insights and performance report
    st.markdown("### Performance Insights")
    
    if 'ppc_agent' in st.session_state and st.session_state.ppc_agent.campaigns:
        # Generate a quick report if needed
        with st.spinner("Generating insights..."):
            report = st.session_state.ppc_agent.generate_performance_report()
            
            # Display insights
            if report['insights']:
                for insight in report['insights']:
                    st.markdown(f"• {insight}")
            else:
                st.info("No insights available yet. More data is needed for meaningful insights.")
                
            # Show campaign summary stats if available
            if 'campaigns' in report['summary']:
                campaign_stats = report['summary']['campaigns']
                st.markdown("#### Campaign Performance Summary")
                
                # Create metrics in columns
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Impressions", f"{campaign_stats['total_impressions']:,}")
                with metric_cols[1]:
                    st.metric("Clicks", f"{campaign_stats['total_clicks']:,}")
                with metric_cols[2]:
                    st.metric("Conversions", f"{campaign_stats['total_conversions']:.1f}")
                with metric_cols[3]:
                    st.metric("Cost", f"${campaign_stats['total_cost']:.2f}")
                    
            # Show top performers if available
            if 'top_performers' in report and 'keywords' in report['top_performers']:
                st.markdown("#### Top Performing Keywords")
                top_keywords = report['top_performers']['keywords']
                
                if top_keywords:
                    # Create a DataFrame for display
                    top_kw_data = []
                    for kw in top_keywords:
                        top_kw_data.append({
                            'Keyword': kw.get('keyword_text', 'Unknown'),
                            'Clicks': kw.get('clicks', 0),
                            'Conversions': kw.get('conversions', 0),
                            'CTR': f"{kw.get('ctr', 0) * 100:.2f}%",
                            'Cost': f"${kw.get('cost', 0):.2f}",
                            'Conv. Rate': f"{kw.get('conversion_rate', 0) * 100:.2f}%"
                        })
                    
                    top_kw_df = pd.DataFrame(top_kw_data)
                    st.dataframe(top_kw_df)
    else:
        st.info("No campaign data available yet. Click 'Refresh Account Data' to get started.")

# Main app logic
def main():
    # Initialize session state first
    init_session_state()

    # Initialize components (this now includes the logger via session state)
    initialize_components()

    # Check if initialization was successful
    if not st.session_state.initialized or st.session_state.logger is None:
        st.error("Application failed to initialize. Check logs for details.")
        # Optionally render logs even if init failed partially
        if st.session_state.logger:
             render_logs()
        return

    # Main app layout
    st.title("📊 Google Ads Optimization Agent")

    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Controls & Settings")

        # Auto-accept toggle
        st.session_state.auto_accept_edits = st.toggle(
            "Auto-Accept Gemini Edits",
            value=st.session_state.get('auto_accept_edits', True),
            help="Automatically accept edits suggested by Gemini for optimizations. Disable to review edits manually."
        )

        st.subheader("Data Fetching")
        days = st.slider("Days to Analyze", 1, 90, 30)

        if st.button("Fetch Campaign Data"):
            fetch_campaign_data(days)

        campaign_options = {c['name']: c['id'] for c in st.session_state.campaigns}
        selected_campaign_name = st.selectbox("Select Campaign for Keywords", options=list(campaign_options.keys()))

        if selected_campaign_name:
            selected_campaign_id = campaign_options[selected_campaign_name]
            if st.button(f"Fetch Keyword Data for '{selected_campaign_name}'"):
                fetch_keyword_data(days, selected_campaign_id)
        else:
             selected_campaign_id = None
             st.info("Fetch campaigns first to select one for keyword analysis.")
             
        # Add comprehensive analysis button
        st.subheader("Analysis")
        if st.button("🧠 Run Comprehensive Analysis", help="Fetch data, analyze with Gemini, and generate keyword bid suggestions"):
            with st.spinner("Running comprehensive analysis..."):
                # First fetch the campaign data
                campaigns = fetch_campaign_data(days)
                if campaigns:
                    # Then fetch keyword data
                    if selected_campaign_id:
                        st.info(f"Analyzing keywords for campaign '{selected_campaign_name}'")
                        keywords = fetch_keyword_data(days, selected_campaign_id)
                    else:
                        st.info("Analyzing keywords across all campaigns")
                        keywords = fetch_keyword_data(days)
                        
                    # Perform analysis if we have data
                    if keywords:
                        st.info(f"Sending {len(keywords)} keywords to Gemini for analysis...")
                        suggestions = get_optimization_suggestions(campaigns=campaigns, keywords=keywords)
                        if suggestions:
                            st.success(f"Analysis complete! Generated {len(suggestions)} optimization suggestions.")
                            # Switch to the Suggestions tab
                            st.rerun()
                    else:
                        st.error("No keyword data available. Please check your query filters.")
                else:
                    st.error("No campaign data available. Please check your account access.")

        # Scheduler Form
        render_scheduler_form()

    # Main area layout using tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["💬 Chat Interface", "📊 Campaign Data", "💡 Suggestions", "🤖 Autonomous Agent", "🕒 Scheduler", "📋 Logs"])

    with tab1:
        render_chat_interface()

    with tab2:
        if st.session_state.campaigns:
            render_campaign_data(st.session_state.campaigns)
        else:
            st.info("No campaign data loaded. Fetch data from the sidebar.")

    with tab3:
        render_suggestions()
        
    with tab4:
        render_autonomous_agent()

    with tab5:
        st.subheader("🕒 Scheduled Tasks")
        render_scheduler_tasks()

    with tab6:
        render_logs()

# Run the app
if __name__ == "__main__":
    main() 