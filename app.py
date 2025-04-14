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

from config import load_config
from ads_api import GoogleAdsAPI
from optimizer import AdsOptimizer
from scheduler import AdsScheduler
from logger import AdsAgentLogger
from chat_interface import ChatInterface

# Initialize logger
logger = AdsAgentLogger()

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
        st.session_state.campaigns = []
        st.session_state.keywords = []
        st.session_state.suggestions = []
        st.session_state.scheduler_running = False
        st.session_state.scheduler_thread = None
        st.session_state.chat_messages = []
        st.session_state.edit_suggestions = {}
        st.session_state.edit_suggestion_id = None
        st.session_state.active_tasks = {}

# Initialize components
def initialize_components():
    if not st.session_state.initialized:
        try:
            # Load configuration
            logger.info("Loading configuration from .env file...")
            st.session_state.config = load_config()
            
            # Initialize APIs
            logger.info("Initializing Google Ads API client...")
            st.session_state.ads_api = GoogleAdsAPI(st.session_state.config['google_ads'])
            
            logger.info("Initializing AdsOptimizer with OpenAI...")
            st.session_state.optimizer = AdsOptimizer(st.session_state.config['openai'])
            
            # Initialize scheduler with logger
            logger.info("Initializing Scheduler...")
            st.session_state.scheduler = AdsScheduler(logger=logger)
            
            # Initialize chat interface
            logger.info("Initializing Chat Interface...")
            st.session_state.chat_interface = ChatInterface(
                st.session_state.ads_api,
                st.session_state.optimizer,
                st.session_state.config,
                logger
            )
            
            # Start the scheduler
            st.session_state.scheduler.start()
            st.session_state.scheduler_running = True
            
            st.session_state.initialized = True
            logger.info("Application initialized successfully")
            
        except Exception as e:
            logger.exception(f"Error initializing application: {str(e)}")
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
            logger.info(f"Running scheduled {task_type} for the last {days} days...")
            
            # Different logic based on task type
            if task_type == "fetch_campaign_data":
                campaigns = st.session_state.ads_api.get_campaign_performance(days_ago=days)
                st.session_state.campaigns = campaigns
                logger.info(f"Scheduled campaign data fetch completed: {len(campaigns)} campaigns retrieved")
                return f"Retrieved {len(campaigns)} campaigns from the last {days} days"
                
            elif task_type == "fetch_keyword_data":
                keywords = st.session_state.ads_api.get_keyword_performance(days_ago=days, campaign_id=campaign_id)
                st.session_state.keywords = keywords
                campaign_info = f"for campaign {campaign_id}" if campaign_id else "across all campaigns"
                logger.info(f"Scheduled keyword data fetch completed: {len(keywords)} keywords retrieved {campaign_info}")
                return f"Retrieved {len(keywords)} keywords from the last {days} days {campaign_info}"
                
            elif task_type == "campaign_analysis":
                # Fetch campaign data and generate suggestions
                campaigns = st.session_state.ads_api.get_campaign_performance(days_ago=days)
                st.session_state.campaigns = campaigns
                
                suggestions = st.session_state.optimizer.get_optimization_suggestions(campaigns)
                st.session_state.suggestions = suggestions
                
                suggestion_count = len(suggestions) if isinstance(suggestions, list) else 0
                logger.info(f"Scheduled campaign analysis completed: {len(campaigns)} campaigns analyzed, {suggestion_count} suggestions generated")
                
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
                logger.info(f"Scheduled keyword analysis completed: {len(keywords)} keywords analyzed {campaign_info}, {suggestion_count} suggestions generated")
                
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
                logger.info(f"Scheduled comprehensive analysis completed: {len(campaigns)} campaigns and {len(keywords)} keywords analyzed, {suggestion_count} suggestions generated")
                
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
                    logger.info(f"Scheduled optimization application completed: {success_count} applied successfully, {failure_count} failed")
                    return f"Applied {success_count} optimizations successfully, {failure_count} failed"
                else:
                    logger.warning("No optimization suggestions available to apply")
                    return "No optimization suggestions available to apply"
                    
            else:
                logger.error(f"Unknown task type: {task_type}")
                return f"Error: Unknown task type {task_type}"
                
        except Exception as e:
            error_message = f"Error in scheduled task: {str(e)}"
            logger.exception(error_message)
            if st.session_state.chat_interface:
                st.session_state.chat_interface.add_message(
                    'system', 
                    f"Scheduled task failed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {str(e)}"
                )
            return error_message
    
    # Create scheduler with the task
    scheduler = AdsScheduler(logger=logger)
    
    # Schedule based on frequency
    if frequency == 'daily':
        task_id = scheduler.schedule_daily(
            function=run_task,
            hour=hour,
            minute=minute,
            name=f"{task_type} (last {days} days)"
        )
        logger.info(f"Scheduled daily task (ID: {task_id}) at {hour:02d}:{minute:02d}")
    elif frequency == 'weekly' and day_of_week:
        task_id = scheduler.schedule_weekly(
            function=run_task,
            day_of_week=day_of_week,
            hour=hour,
            minute=minute,
            name=f"{task_type} (last {days} days)"
        )
        logger.info(f"Scheduled weekly task (ID: {task_id}) on {day_of_week} at {hour:02d}:{minute:02d}")
    elif frequency == 'once':
        task_id = scheduler.schedule_once(
            function=run_task,
            hour=hour,
            minute=minute,
            name=f"{task_type} (last {days} days)"
        )
        logger.info(f"Scheduled one-time task (ID: {task_id}) at {hour:02d}:{minute:02d}")
    
    st.session_state.scheduler_running = True
    
    try:
        scheduler.start()
    except Exception as e:
        logger.exception(f"Scheduler error: {str(e)}")
    finally:
        st.session_state.scheduler_running = False

# Function to fetch campaign data
def fetch_campaign_data(days=30):
    try:
        with st.spinner(f"Fetching campaign data for the last {days} days..."):
            logger.info(f"Fetching campaign data for the last {days} days...")
            st.session_state.campaigns = st.session_state.ads_api.get_campaign_performance(days_ago=days)
            logger.info(f"Fetched data for {len(st.session_state.campaigns)} campaigns")
            return st.session_state.campaigns
    except Exception as e:
        logger.exception(f"Error fetching campaign data: {str(e)}")
        st.error(f"Failed to fetch campaign data: {str(e)}")
        return []

# Function to fetch keyword data
def fetch_keyword_data(days=30, campaign_id=None):
    try:
        campaign_info = f"for campaign ID {campaign_id}" if campaign_id else "for all campaigns"
        with st.spinner(f"Fetching keyword data {campaign_info} for the last {days} days..."):
            logger.info(f"Fetching keyword data {campaign_info} for the last {days} days...")
            st.session_state.keywords = st.session_state.ads_api.get_keyword_performance(days_ago=days, campaign_id=campaign_id)
            logger.info(f"Fetched data for {len(st.session_state.keywords)} keywords")
            return st.session_state.keywords
    except Exception as e:
        logger.exception(f"Error fetching keyword data: {str(e)}")
        st.error(f"Failed to fetch keyword data: {str(e)}")
        return []

# Function to get optimization suggestions
def get_optimization_suggestions(campaigns=None, keywords=None):
    if campaigns is None:
        campaigns = st.session_state.campaigns
    
    if keywords is None:
        keywords = st.session_state.keywords
        
    if not campaigns:
        logger.warning("No campaign data available for optimization")
        st.warning("No campaign data available for optimization. Please fetch campaign data first.")
        return None
        
    try:
        with st.spinner("Getting optimization suggestions from GPT-4..."):
            logger.info("Sending campaign and keyword data to GPT-4 for analysis...")
            st.session_state.suggestions = st.session_state.optimizer.get_optimization_suggestions(campaigns, keywords)
            logger.info(f"Received {len(st.session_state.suggestions) if isinstance(st.session_state.suggestions, list) else 0} optimization suggestions from GPT-4")
            return st.session_state.suggestions
    except Exception as e:
        logger.exception(f"Error getting optimization suggestions: {str(e)}")
        st.error(f"Failed to get optimization suggestions: {str(e)}")
        return None

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
        
        # Prepare changes based on action type
        changes = {}
        
        if action_type == 'BID_ADJUSTMENT':
            # Get the edited or original change_value
            if 'edited_value' in suggestion:
                change_value = suggestion['edited_value']
                
                # Convert to micros (multiply by 1,000,000)
                bid_micros = int(float(change_value) * 1000000)
                changes['bid_micros'] = bid_micros
            elif 'change_value' in suggestion:
                change_type = suggestion['change_value'].get('type')
                value = suggestion['change_value'].get('value')
                
                # If we have a current value to work with
                if 'current_value' in suggestion:
                    current_value = suggestion['current_value']
                    
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
                else:
                    return False, "Cannot apply bid adjustment without current value"
            else:
                return False, "Missing change value for bid adjustment"
                
        elif action_type == 'STATUS_CHANGE':
            # Get the edited or original status
            if 'edited_value' in suggestion:
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
            if 'edited_value' in suggestion:
                budget_micros = int(float(suggestion['edited_value']) * 1000000)
                changes['budget_micros'] = budget_micros
            else:
                return False, "Missing edited value for budget adjustment"
        
        else:
            return False, f"Unsupported action type: {action_type}"
        
        # Apply the optimization
        logger.info(f"Applying {action_type} to {entity_type} {entity_id} with changes: {changes}")
        success, message = st.session_state.ads_api.apply_optimization(
            optimization_type=action_type.lower(),
            entity_type=entity_type,
            entity_id=entity_id,
            changes=changes
        )
        
        if success:
            logger.info(f"Successfully applied optimization: {message}")
        else:
            logger.error(f"Failed to apply optimization: {message}")
        
        return success, message
        
    except Exception as e:
        error_message = f"Error applying optimization: {str(e)}"
        logger.exception(error_message)
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
        logger.warning("Scheduler is already running")
        st.warning("Scheduler is already running")
        return
        
    try:
        logger.info(f"Starting scheduler for {task_type} with parameters: days={days}, hour={hour}, minute={minute}, frequency={frequency}, day_of_week={day_of_week}")
        
        # Create and start thread
        scheduler_thread = threading.Thread(
            target=run_scheduler_thread,
            args=(days, hour, minute, frequency, day_of_week, task_type, campaign_id),
            daemon=True
        )
        
        scheduler_thread.start()
        st.session_state.scheduler_thread = scheduler_thread
        st.session_state.scheduler_running = True
        
        logger.info("Scheduler started successfully")
        st.success(f"Scheduler for {task_type} started successfully. Will run {frequency} at {hour:02d}:{minute:02d}")
        
    except Exception as e:
        logger.exception(f"Error starting scheduler: {str(e)}")
        st.error(f"Failed to start scheduler: {str(e)}")

# Function to stop scheduler
def stop_scheduler():
    if not st.session_state.scheduler_running:
        logger.warning("No scheduler is currently running")
        st.warning("No scheduler is currently running")
        return
        
    try:
        # Can't directly stop the thread, so we set the flag to False
        # and the scheduler will check this flag and exit gracefully
        st.session_state.scheduler_running = False
        
        logger.info("Scheduler will stop after the current iteration")
        st.info("Scheduler will stop after the current iteration")
        
    except Exception as e:
        logger.exception(f"Error stopping scheduler: {str(e)}")
        st.error(f"Failed to stop scheduler: {str(e)}")

# Function to render campaigns data as a table and charts
def render_campaign_data(campaigns, keywords=None):
    if not campaigns:
        st.info("No campaign data available. Use the sidebar to fetch campaign data.")
        return
        
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(campaigns)
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Campaigns", len(df))
    with col2:
        st.metric("Total Clicks", int(df['clicks'].sum()))
    with col3:
        st.metric("Total Conversions", round(df['conversions'].sum(), 2))
    with col4:
        st.metric("Total Cost", f"${round(df['cost'].sum(), 2)}")
    
    # Display data table
    st.subheader("Campaign Performance Data")
    
    # Format the dataframe for display
    display_df = df.copy()
    display_df['ctr'] = display_df['ctr'].apply(lambda x: f"{x:.2f}%")
    display_df['conversion_rate'] = display_df['conversion_rate'].apply(lambda x: f"{x:.2f}%")
    display_df['average_cpc'] = display_df['average_cpc'].apply(lambda x: f"${x:.2f}")
    display_df['cost'] = display_df['cost'].apply(lambda x: f"${x:.2f}")
    display_df['cost_per_conversion'] = display_df['cost_per_conversion'].apply(lambda x: f"${x:.2f}")
    
    st.dataframe(display_df)
    
    # Create charts
    st.subheader("Campaign Performance Visualization")
    
    tab1, tab2, tab3 = st.tabs(["Clicks & Conversions", "CTR & Conversion Rate", "Cost Metrics"])
    
    with tab1:
        # Create a bar chart for clicks and conversions
        fig = px.bar(
            df.sort_values('clicks', ascending=False).head(10),
            x='name',
            y=['clicks', 'conversions'],
            barmode='group',
            title='Top 10 Campaigns by Clicks',
            labels={'name': 'Campaign', 'value': 'Count', 'variable': 'Metric'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        # Create a scatter plot for CTR vs. Conversion Rate
        fig = px.scatter(
            df,
            x='ctr',
            y='conversion_rate',
            size='impressions',
            color='cost',
            hover_name='name',
            title='CTR vs. Conversion Rate',
            labels={
                'ctr': 'Click-Through Rate (%)', 
                'conversion_rate': 'Conversion Rate (%)',
                'impressions': 'Impressions',
                'cost': 'Cost ($)'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        # Create a bar chart for cost metrics
        fig = px.bar(
            df.sort_values('cost', ascending=False).head(10),
            x='name',
            y=['cost', 'cost_per_conversion'],
            barmode='group',
            title='Top 10 Campaigns by Cost',
            labels={'name': 'Campaign', 'value': 'Amount ($)', 'variable': 'Metric'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Display keyword data if available
    if keywords and len(keywords) > 0:
        st.subheader("Keyword Performance Data")
        
        # Convert to DataFrame
        kw_df = pd.DataFrame(keywords)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Keywords", len(kw_df))
        with col2:
            st.metric("Keywords With Clicks", len(kw_df[kw_df['clicks'] > 0]))
        with col3:
            st.metric("Keywords With Conversions", len(kw_df[kw_df['conversions'] > 0]))
        with col4:
            avg_quality = kw_df['quality_score'].mean() if 'quality_score' in kw_df and not kw_df['quality_score'].isna().all() else 0
            st.metric("Average Quality Score", f"{avg_quality:.1f}")
        
        # Format the dataframe for display
        display_kw_df = kw_df.copy()
        if 'ctr' in display_kw_df:
            display_kw_df['ctr'] = display_kw_df['ctr'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "0.00%")
        if 'average_cpc' in display_kw_df:
            display_kw_df['average_cpc'] = display_kw_df['average_cpc'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "$0.00")
        if 'cost' in display_kw_df:
            display_kw_df['cost'] = display_kw_df['cost'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "$0.00")
        if 'cost_per_conversion' in display_kw_df:
            display_kw_df['cost_per_conversion'] = display_kw_df['cost_per_conversion'].apply(
                lambda x: f"${x:.2f}" if pd.notnull(x) and x > 0 else "-"
            )
        if 'current_bid' in display_kw_df:
            display_kw_df['current_bid'] = display_kw_df['current_bid'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "-")
        
        # Display keyword table
        st.dataframe(display_kw_df)
        
        # Create keyword charts
        st.subheader("Keyword Performance Visualization")
        
        tab1, tab2 = st.tabs(["Top Keywords by Clicks", "Keyword Quality Analysis"])
        
        with tab1:
            # Top keywords by clicks
            if len(kw_df) > 0 and 'clicks' in kw_df:
                top_keywords = kw_df.sort_values('clicks', ascending=False).head(20)
                fig = px.bar(
                    top_keywords,
                    x='keyword_text',
                    y='clicks',
                    color='conversions',
                    title='Top 20 Keywords by Clicks',
                    labels={
                        'keyword_text': 'Keyword',
                        'clicks': 'Clicks',
                        'conversions': 'Conversions'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Quality score analysis
            if len(kw_df) > 0 and 'quality_score' in kw_df and not kw_df['quality_score'].isna().all():
                # Count keywords by quality score
                quality_counts = kw_df['quality_score'].value_counts().sort_index()
                
                fig = px.bar(
                    x=quality_counts.index,
                    y=quality_counts.values,
                    title='Keyword Distribution by Quality Score',
                    labels={
                        'x': 'Quality Score',
                        'y': 'Number of Keywords'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

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
            campaigns, keywords = st.session_state.chat_interface.ensure_data_context()
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
            campaigns, keywords = st.session_state.chat_interface.ensure_data_context(days=30)
            st.success(f"Data refreshed successfully! Loaded {len(campaigns)} campaigns and {len(keywords)} keywords.")
            time.sleep(1)
            st.rerun()

# Function to display logs in the UI
def render_logs():
    st.subheader("System Logs")
    
    # Create tabs for different log levels
    tab1, tab2, tab3 = st.tabs(["All Logs", "Warnings", "Errors"])
    
    with tab1:
        logs = logger.get_recent_logs(limit=50)
        for timestamp, level, message in reversed(logs):
            log_class = f"log-entry-{level.lower()}" if level.lower() in ['warning', 'error'] else "log-entry-info"
            st.markdown(f"<div class='{log_class}'><strong>{timestamp} [{level}]</strong> {message}</div>", unsafe_allow_html=True)
    
    with tab2:
        logs = logger.get_recent_logs(level="WARNING", limit=50)
        if logs:
            for timestamp, level, message in reversed(logs):
                st.markdown(f"<div class='log-entry-warning'><strong>{timestamp} [{level}]</strong> {message}</div>", unsafe_allow_html=True)
        else:
            st.info("No warnings logged")
    
    with tab3:
        logs = logger.get_recent_logs(level="ERROR", limit=50)
        if logs:
            for timestamp, level, message in reversed(logs):
                st.markdown(f"<div class='log-entry-error'><strong>{timestamp} [{level}]</strong> {message}</div>", unsafe_allow_html=True)
        else:
            st.info("No errors logged")

# Function to show an editable suggestion
def render_editable_suggestion(suggestion, index):
    """
    Render a single optimization suggestion with editing capabilities.
    
    Args:
        suggestion (dict): The suggestion to render
        index (int): The index of the suggestion
    """
    # Determine card style based on status
    card_class = "suggestion-card"
    if suggestion.get('status') == 'applied':
        card_class += " suggestion-card-applied"
    elif suggestion.get('status') == 'failed':
        card_class += " suggestion-card-failed"
    else:
        card_class += " suggestion-card-pending"
    
    # Start the card
    st.markdown(f"<div class='{card_class}'>", unsafe_allow_html=True)
    
    # Header with action badge
    action_type = suggestion.get('action_type', 'OTHER')
    col1, col2 = st.columns([10, 2])
    with col1:
        st.subheader(f"{index+1}. {suggestion.get('title', 'Suggestion')}")
    with col2:
        st.info(action_type)
    
    # Entity information
    entity_type = suggestion.get('entity_type', '').title()
    entity_id = suggestion.get('entity_id', 'Unknown')
    st.write(f"**Target**: {entity_type} - {entity_id}")
    
    # Original change and rationale
    with st.expander("Details", expanded=False):
        st.write("**Suggested Change**:")
        st.write(suggestion.get('change', 'No specific change provided'))
        st.write("**Rationale**:")
        st.write(suggestion.get('rationale', 'No rationale provided'))
    
    # Editable fields based on action type
    st.write("**Edit Suggestion**:")
    
    suggestion_id = str(suggestion.get('index', index))
    
    # Initialize the edit state for this suggestion if not exists
    if suggestion_id not in st.session_state.edit_suggestions:
        st.session_state.edit_suggestions[suggestion_id] = {}
    
    # Create editable fields based on action type
    if action_type == 'BID_ADJUSTMENT':
        # Display current bid if available
        if 'current_value' in suggestion:
            st.write(f"Current Bid: ${suggestion['current_value']:.2f}")
        
        # Calculate appropriate initial value and min value to prevent StreamlitValueBelowMinError
        initial_value = 0.01  # Default minimum
        
        if 'edited_value' in suggestion and suggestion['edited_value'] > 0:
            initial_value = suggestion['edited_value']
        elif 'change_value' in suggestion and suggestion['change_value'].get('type') == 'absolute':
            value = suggestion['change_value'].get('value', 0.0)
            initial_value = max(0.01, value)
        elif 'current_value' in suggestion:
            # Apply percentage change if specified
            if 'change_value' in suggestion:
                change_type = suggestion['change_value'].get('type')
                change_value = suggestion['change_value'].get('value', 0.0)
                
                current = suggestion['current_value']
                
                if change_type == 'percentage_increase':
                    initial_value = current * (1 + change_value/100)
                elif change_type == 'percentage_decrease':
                    initial_value = current * (1 - change_value/100)
                    # Ensure minimum value
                    initial_value = max(0.01, initial_value)
                else:
                    initial_value = current
            else:
                initial_value = suggestion['current_value']
        
        # Round to 2 decimal places for display
        initial_value = round(initial_value, 2)
        
        # Ensure minimum value is respected
        initial_value = max(0.01, initial_value)
        
        new_bid = st.number_input(
            "New Bid ($):",
            min_value=0.01,
            max_value=1000.0,
            value=float(initial_value),
            step=0.01,
            key=f"bid_{suggestion_id}",
            help="Enter the new bid amount in dollars"
        )
        
        st.session_state.edit_suggestions[suggestion_id]['edited_value'] = new_bid
        
    elif action_type == 'STATUS_CHANGE':
        # Display current status if available
        if 'current_value' in suggestion:
            st.write(f"Current Status: {suggestion['current_value']}")
        
        # Determine initial status value
        initial_status = "ENABLED"
        if 'edited_value' in suggestion:
            initial_status = suggestion['edited_value']
        elif 'change_value' in suggestion and suggestion['change_value'].get('type') == 'status':
            initial_status = suggestion['change_value'].get('value', 'ENABLED')
        elif 'change' in suggestion:
            # Try to extract from change text
            change_text = suggestion['change'].lower()
            if 'pause' in change_text:
                initial_status = "PAUSED"
            elif 'remove' in change_text:
                initial_status = "REMOVED"
        
        new_status = st.selectbox(
            "New Status:",
            options=["ENABLED", "PAUSED", "REMOVED"],
            index=["ENABLED", "PAUSED", "REMOVED"].index(initial_status),
            key=f"status_{suggestion_id}",
            help="Select the new status for this entity"
        )
        
        st.session_state.edit_suggestions[suggestion_id]['edited_value'] = new_status
        
    elif action_type == 'BUDGET_ADJUSTMENT':
        # Display current budget if available
        if 'current_value' in suggestion:
            st.write(f"Current Budget: ${suggestion['current_value']:.2f}")
        
        # Calculate appropriate initial value and min value to prevent StreamlitValueBelowMinError
        initial_value = 1.0  # Default minimum for budget
        
        if 'edited_value' in suggestion and suggestion['edited_value'] > 0:
            initial_value = suggestion['edited_value']
        elif 'change_value' in suggestion and suggestion['change_value'].get('type') == 'absolute':
            value = suggestion['change_value'].get('value', 0.0)
            initial_value = max(1.0, value)
        elif 'current_value' in suggestion:
            # Apply percentage change if specified
            if 'change_value' in suggestion:
                change_type = suggestion['change_value'].get('type')
                change_value = suggestion['change_value'].get('value', 0.0)
                
                current = suggestion['current_value']
                
                if change_type == 'percentage_increase':
                    initial_value = current * (1 + change_value/100)
                elif change_type == 'percentage_decrease':
                    initial_value = current * (1 - change_value/100)
                    # Ensure minimum value
                    initial_value = max(1.0, initial_value)
                else:
                    initial_value = current
            else:
                initial_value = suggestion['current_value']
        
        # Round to 2 decimal places for display
        initial_value = round(initial_value, 2)
        
        # Ensure minimum value is respected
        initial_value = max(1.0, initial_value)
        
        new_budget = st.number_input(
            "New Budget ($):",
            min_value=1.0,  # Minimum budget is $1
            max_value=10000.0,
            value=float(initial_value),
            step=1.0,
            key=f"budget_{suggestion_id}",
            help="Enter the new budget amount in dollars"
        )
        
        st.session_state.edit_suggestions[suggestion_id]['edited_value'] = new_budget
    
    elif action_type in ['MATCH_TYPE_CHANGE', 'QUALITY_IMPROVEMENT', 'NEGATIVE_KEYWORD', 'CAMPAIGN_SETTINGS']:
        # For these action types, we don't need numeric inputs but might need other inputs
        if action_type == 'MATCH_TYPE_CHANGE':
            new_match_type = st.selectbox(
                "New Match Type:",
                options=["EXACT", "PHRASE", "BROAD"],
                key=f"match_type_{suggestion_id}",
                help="Select the new match type for this keyword"
            )
            st.session_state.edit_suggestions[suggestion_id]['edited_value'] = new_match_type
            
        elif action_type == 'NEGATIVE_KEYWORD':
            # Show the suggested negative keyword
            if 'entity_id' in suggestion:
                st.write(f"Suggested Negative Keyword: {suggestion['entity_id']}")
            # Allow editing if needed
            edited_keyword = st.text_input(
                "Edit Negative Keyword:",
                value=suggestion.get('entity_id', ''),
                key=f"neg_kw_{suggestion_id}",
                help="Edit the negative keyword if needed"
            )
            st.session_state.edit_suggestions[suggestion_id]['edited_value'] = edited_keyword
    
    # Apply button and status
    col1, col2 = st.columns([3, 1])
    
    # Show apply button if not already applied
    if not suggestion.get('applied', False):
        with col1:
            if st.button("Apply This Change", key=f"apply_{suggestion_id}"):
                # Update suggestion with edited values
                for key, value in st.session_state.edit_suggestions.get(suggestion_id, {}).items():
                    suggestion[key] = value
                
                # Apply the optimization
                success, message = apply_optimization(suggestion)
                
                # Update suggestion status
                suggestion['applied'] = success
                suggestion['status'] = 'applied' if success else 'failed'
                suggestion['result_message'] = message
                
                if success:
                    st.success(f"Successfully applied: {message}")
                else:
                    st.error(f"Failed to apply: {message}")
    
    # Show status in the right column
    with col2:
        if suggestion.get('applied', False):
            if suggestion.get('status') == 'applied':
                st.success("Applied ✓")
            else:
                st.error("Failed ✗")
    
    # Show result message if present
    if suggestion.get('result_message'):
        st.info(suggestion['result_message'])
    
    # End card
    st.markdown("</div>", unsafe_allow_html=True)

# Function to render suggestion list
def render_suggestions():
    """Render the list of optimization suggestions with edit and apply functionality."""
    if not st.session_state.suggestions or not isinstance(st.session_state.suggestions, list):
        st.info("No optimization suggestions available. Use the sidebar to analyze your campaigns.")
        return
    
    st.subheader("Optimization Suggestions")
    
    # Show apply all button at the top
    pending_count = sum(1 for s in st.session_state.suggestions if not s.get('applied', False))
    if pending_count > 0:
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
                    logger.info(f"Running comprehensive analysis on {len(campaigns)} campaigns and {len(keywords)} keywords")
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
            logger.exception(error_message)
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
            logger.error("Cannot schedule task: Scheduler not initialized")
            return None
        
        # Get task information
        task_info = TASK_TYPES.get(task_type)
        if not task_info:
            logger.error(f"Unknown task type: {task_type}")
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
            logger.info(f"Scheduling daily task {task_name} at {hour:02d}:{minute:02d}")
            task_id = st.session_state.scheduler.schedule_daily(
                function=task_function,
                hour=hour,
                minute=minute,
                name=task_name,
                args=[],
                kwargs={}
            )
        elif schedule_type == "weekly" and day_of_week:
            logger.info(f"Scheduling weekly task {task_name} on {day_of_week} at {hour:02d}:{minute:02d}")
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
            logger.info(f"Scheduling one-time task {task_name} at {hour:02d}:{minute:02d}")
            task_id = st.session_state.scheduler.schedule_once(
                function=task_function,
                hour=hour,
                minute=minute,
                name=task_name,
                args=[],
                kwargs={}
            )
        else:
            logger.error(f"Invalid schedule type: {schedule_type}")
            return None
            
        logger.info(f"Scheduled task {task_name} (ID: {task_id})")
        return task_id
        
    except Exception as e:
        logger.exception(f"Error scheduling task: {str(e)}")
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
                        logger.exception(f"Error executing task {task.name} (ID: {task_id}): {str(e)}")
            
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
    """Render a form to create new scheduled tasks."""
    st.subheader("Schedule New Task")
    
    with st.form(key="schedule_task_form"):
        # Task selection
        st.markdown("#### Task Selection")
        task_type = st.selectbox(
            "What would you like to schedule?",
            options=list(TASK_TYPES.keys()),
            format_func=lambda x: f"{TASK_TYPES[x]['icon']} {TASK_TYPES[x]['name']}",
            help="Select the type of task to schedule"
        )
        
        # Display description of selected task
        st.info(TASK_TYPES[task_type]['description'])
        
        # Task parameters
        st.markdown("#### Data Parameters")
        task_params = {}
        
        # Dynamic parameters based on task type
        required_params = TASK_TYPES[task_type]['params']
        
        if 'days' in required_params:
            # More flexible date range options
            date_range_col1, date_range_col2 = st.columns(2)
            
            with date_range_col1:
                date_option = st.radio(
                    "Data Time Range",
                    options=["Last X Days", "Custom Date Range"],
                    index=0,
                    help="Select how to specify the time range"
                )
            
            with date_range_col2:
                if date_option == "Last X Days":
                    task_params['days'] = st.number_input(
                        "Number of Days",
                        min_value=1,
                        max_value=365,  # Extended from 90 days to 365
                        value=30,
                        help="Number of days of historical data to analyze (1-365)"
                    )
                else:
                    # Custom date range using date picker
                    today = datetime.now().date()
                    start_date = st.date_input(
                        "Start Date", 
                        value=today - timedelta(days=30),
                        max_value=today,
                        help="Select start date for analysis period"
                    )
                    end_date = st.date_input(
                        "End Date", 
                        value=today, 
                        min_value=start_date,
                        max_value=today,
                        help="Select end date for analysis period"
                    )
                    # Calculate days between dates
                    delta = end_date - start_date
                    task_params['days'] = delta.days + 1  # +1 to include both start and end dates
        
        if 'campaign_id' in required_params:
            st.markdown("#### Campaign Selection")
            # If we have campaigns loaded, show a dropdown
            campaign_selection = st.radio(
                "Campaign Scope",
                options=["All Campaigns", "Specific Campaign"],
                index=0,
                help="Choose whether to process all campaigns or a specific one"
            )
            
            if campaign_selection == "Specific Campaign":
                if st.session_state.campaigns:
                    campaign_options = [(c['id'], f"{c['name']} (ID: {c['id']})") for c in st.session_state.campaigns]
                    campaign_ids = [c[0] for c in campaign_options]
                    campaign_labels = [c[1] for c in campaign_options]
                    
                    selected_index = st.selectbox(
                        "Select Campaign",
                        options=range(len(campaign_options)),
                        format_func=lambda i: campaign_labels[i],
                        help="Select a specific campaign to analyze"
                    )
                    
                    task_params['campaign_id'] = campaign_ids[selected_index]
                else:
                    st.warning("No campaigns loaded. Please fetch campaign data first or enter a campaign ID manually.")
                    # Allow manual entry
                    task_params['campaign_id'] = st.text_input(
                        "Campaign ID",
                        value="",
                        help="Enter a specific campaign ID"
                    )
            else:
                # All campaigns selected
                task_params['campaign_id'] = None
        
        # Schedule settings
        st.markdown("#### Schedule Settings")
        schedule_type = st.selectbox(
            "How often should this task run?",
            options=["daily", "weekly", "once"],
            format_func=lambda x: x.capitalize(),
            help="Frequency of task execution"
        )
        
        # Schedule details
        time_col1, time_col2 = st.columns(2)
        
        with time_col1:
            hour = st.number_input(
                "Hour (24-hour format)",
                min_value=0,
                max_value=23,
                value=9,
                help="Hour to run the task (0-23)"
            )
        
        with time_col2:
            minute = st.number_input(
                "Minute",
                min_value=0,
                max_value=59,
                value=0,
                help="Minute to run the task (0-59)"
            )
        
        # Day of week for weekly schedules
        day_of_week = None
        if schedule_type == "weekly":
            day_of_week = st.selectbox(
                "Day of Week",
                options=["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
                format_func=lambda x: x.capitalize(),
                help="Day of the week to run the task"
            )
        
        # Submit button
        submit_col1, submit_col2 = st.columns([3, 1])
        with submit_col2:
            submit_button = st.form_submit_button("Schedule Task")
        
        if submit_button:
            # Schedule the task
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
                # Force refresh to show the new task
                st.rerun()
            else:
                st.error("Failed to schedule task. Check the logs for details.")

# Main app logic
def main():
    # Initialize session state
    init_session_state()
    
    # Sidebar navigation
    st.sidebar.title("Google Ads Optimizer")
    
    # Initialize components button
    if not st.session_state.initialized:
        if st.sidebar.button("Initialize App"):
            initialize_components()
    else:
        st.sidebar.success("App initialized ✅")
    
    # Main navigation
    app_mode = st.sidebar.selectbox(
        "Navigation",
        ["Dashboard", "Campaign Analysis", "Keyword Analysis", "Optimization", "Chat Assistant", "Scheduler", "System Logs"]
    )
    
    # Main content area
    st.markdown("<h1 class='main-header'>Google Ads Optimization Agent</h1>", unsafe_allow_html=True)
    
    # Check if app is initialized before showing content
    if not st.session_state.initialized:
        st.warning("Please initialize the app using the button in the sidebar.")
        return
        
    if app_mode == "Dashboard":
        st.markdown("<h2 class='sub-header'>Campaign Performance Dashboard</h2>", unsafe_allow_html=True)
        
        # Dashboard actions
        with st.expander("Dashboard Options", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                days = st.number_input("Days of data to fetch", min_value=1, max_value=90, value=30, step=1)
            with col2:
                if st.button("Refresh Campaign Data"):
                    fetch_campaign_data(days)
        
        # Render campaign data if available
        render_campaign_data(st.session_state.campaigns)
    
    elif app_mode == "Campaign Analysis":
        st.markdown("<h2 class='sub-header'>Campaign Analysis</h2>", unsafe_allow_html=True)
        
        # Analysis actions
        with st.expander("Analysis Options", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                days = st.number_input("Days of data to analyze", min_value=1, max_value=90, value=30, step=1)
            with col2:
                if st.button("Analyze Campaigns"):
                    campaigns = fetch_campaign_data(days)
                    if campaigns:
                        get_optimization_suggestions(campaigns)
        
        # Display campaign data
        render_campaign_data(st.session_state.campaigns)
        
        # Display optimization suggestions
        render_suggestions()
    
    elif app_mode == "Keyword Analysis":
        st.markdown("<h2 class='sub-header'>Keyword Analysis</h2>", unsafe_allow_html=True)
        
        # Keyword analysis options
        with st.expander("Keyword Options", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                days = st.number_input("Days of data", min_value=1, max_value=90, value=30, step=1)
            
            with col2:
                campaign_id = None
                if st.session_state.campaigns:
                    # Create a dropdown of available campaigns plus "All Campaigns" option
                    campaign_options = [("all", "All Campaigns")] + [(c['id'], f"{c['name']} (ID: {c['id']})") for c in st.session_state.campaigns]
                    campaign_ids = [c[0] for c in campaign_options]
                    campaign_labels = [c[1] for c in campaign_options]
                    
                    selected_index = st.selectbox(
                        "Campaign Filter",
                        options=range(len(campaign_options)),
                        format_func=lambda i: campaign_labels[i]
                    )
                    
                    selected_campaign = campaign_ids[selected_index]
                    if selected_campaign != "all":
                        campaign_id = selected_campaign
            
            with col3:
                if st.button("Fetch Keyword Data"):
                    fetch_keyword_data(days, campaign_id)
                    
                    # If we have keywords and campaigns, try to get optimization suggestions
                    if st.session_state.keywords and st.session_state.campaigns:
                        if st.checkbox("Generate optimization suggestions", value=True):
                            get_optimization_suggestions(st.session_state.campaigns, st.session_state.keywords)
        
        # Display campaign and keyword data
        if st.session_state.keywords:
            render_campaign_data(st.session_state.campaigns, st.session_state.keywords)
            
            # Display optimization suggestions if available
            if st.session_state.suggestions and isinstance(st.session_state.suggestions, list):
                render_suggestions()
        else:
            st.info("No keyword data available. Use the options above to fetch keyword data.")
        
    elif app_mode == "Optimization":
        st.markdown("<h2 class='sub-header'>Optimization Suggestions</h2>", unsafe_allow_html=True)
        
        # Optimization actions
        with st.expander("Optimization Options", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                days = st.number_input("Days of data", min_value=1, max_value=90, value=30, step=1)
            with col2:
                include_keywords = st.checkbox("Include keyword data", value=True)
            with col3:
                if st.button("Get Optimization Suggestions"):
                    campaigns = fetch_campaign_data(days)
                    keywords = fetch_keyword_data(days) if include_keywords else None
                    if campaigns:
                        get_optimization_suggestions(campaigns, keywords)
        
        # Display optimization suggestions
        render_suggestions()
        
    elif app_mode == "Chat Assistant":
        st.markdown("<h2 class='sub-header'>PPC Expert Chat Assistant</h2>", unsafe_allow_html=True)
        
        # Render chat interface
        render_chat_interface()
        
    elif app_mode == "Scheduler":
        st.markdown("<h2 class='sub-header'>Task Scheduler</h2>", unsafe_allow_html=True)
        
        # Scheduler status
        scheduler_status = st.session_state.scheduler.is_running()
        if scheduler_status:
            st.success("Scheduler is running")
        else:
            st.warning("Scheduler is not running")
            
            # Quick start scheduler with default settings
            quick_start_col1, quick_start_col2 = st.columns([2, 1])
            with quick_start_col1:
                quick_task = st.selectbox(
                    "Quick start task",
                    options=["comprehensive_analysis", "campaign_analysis", "keyword_analysis", "fetch_campaign_data", "fetch_keyword_data"],
                    format_func=lambda x: TASK_TYPES[x]['name'] if x in TASK_TYPES else x.replace('_', ' ').title(),
                    help="Select a task to quickly schedule"
                )
            with quick_start_col2:
                if st.button("Start Scheduler"):
                    # Start with default settings: daily at 9:00am with 30 days of data
                    start_scheduler(
                        days=30,
                        hour=9,
                        minute=0,
                        frequency='daily',
                        task_type=quick_task
                    )
                    st.rerun()
        
        # Display tasks
        render_scheduler_tasks()
        
        # Add new task form
        render_scheduler_form()
    
    elif app_mode == "System Logs":
        st.markdown("<h2 class='sub-header'>System Logs</h2>", unsafe_allow_html=True)
        
        # Display logs
        render_logs()

# Run the app
if __name__ == "__main__":
    main() 