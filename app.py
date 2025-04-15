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
    """Render campaign performance data with charts and metrics."""
    if not campaigns:
        st.warning("No campaign data available. Please fetch data first.")
        return
    
    # Create a DataFrame for easier manipulation
    df_campaigns = pd.DataFrame(campaigns)
    
    # Key metrics section
    st.subheader("Key Campaign Metrics")
    
    # Calculate overall metrics
    total_cost = df_campaigns['cost'].sum()
    total_conversions = df_campaigns['conversions'].sum()
    total_clicks = df_campaigns['clicks'].sum()
    overall_ctr = (total_clicks / df_campaigns['impressions'].sum() * 100) if df_campaigns['impressions'].sum() > 0 else 0
    overall_conversion_rate = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
    overall_cpc = total_cost / total_clicks if total_clicks > 0 else 0
    overall_cpa = total_cost / total_conversions if total_conversions > 0 else 0
    
    # Display metrics in columns
    cols = st.columns(4)
    cols[0].metric("Total Spend", f"${total_cost:.2f}")
    cols[1].metric("Total Conversions", f"{total_conversions:.1f}")
    cols[2].metric("Conversion Rate", f"{overall_conversion_rate:.2f}%")
    cols[3].metric("Cost Per Conversion", f"${overall_cpa:.2f}" if total_conversions > 0 else "N/A")
    
    cols = st.columns(4)
    cols[0].metric("Total Clicks", f"{total_clicks:,}")
    cols[1].metric("CTR", f"{overall_ctr:.2f}%")
    cols[2].metric("Average CPC", f"${overall_cpc:.2f}" if total_clicks > 0 else "N/A")
    cols[3].metric("Campaigns", f"{len(campaigns)}")
    
    # Charts section
    st.subheader("Campaign Performance")
    
    # Create tab layout for different visualizations
    chart_tabs = st.tabs(["Cost & Conversions", "Click Performance", "Campaign Table"])
    
    with chart_tabs[0]:
        # Prepare data for chart - top 10 campaigns by spend
        top_campaigns = df_campaigns.sort_values('cost', ascending=False).head(10)
        
        # Create a combo chart with cost and conversions
        fig = go.Figure()
        
        # Add cost bars
        fig.add_trace(
            go.Bar(
                x=top_campaigns['name'],
                y=top_campaigns['cost'],
                name='Cost ($)',
                marker_color='#4285F4'
            )
        )
        
        # Add conversion line
        fig.add_trace(
            go.Scatter(
                x=top_campaigns['name'],
                y=top_campaigns['conversions'],
                name='Conversions',
                marker_color='#34A853',
                mode='lines+markers',
                yaxis='y2'
            )
        )
        
        # Set up the layout with two y-axes
        fig.update_layout(
            title='Top 10 Campaigns by Spend',
            xaxis_title='Campaign',
            yaxis_title='Cost ($)',
            yaxis2=dict(
                title='Conversions',
                overlaying='y',
                side='right'
            ),
            barmode='group',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Display the figure
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_tabs[1]:
        # Create a scatter plot of CTR vs CPC with bubble size as clicks
        fig = px.scatter(
            df_campaigns,
            x='ctr',
            y='average_cpc',
            size='clicks',
            color='conversions',
            hover_name='name',
            size_max=60,
            labels={
                'ctr': 'CTR (%)',
                'average_cpc': 'Average CPC ($)',
                'clicks': 'Clicks',
                'conversions': 'Conversions'
            },
            title='CTR vs CPC (bubble size = clicks, color = conversions)'
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_tabs[2]:
        # Display the campaign data as a sortable table
        st.dataframe(
            df_campaigns[[
                'name', 'status', 'clicks', 'impressions', 'ctr', 'average_cpc',
                'cost', 'conversions', 'conversion_rate', 'cost_per_conversion'
            ]].sort_values('cost', ascending=False),
            use_container_width=True,
            hide_index=True
        )
    
    # Show keyword data if available
    if keywords:
        st.subheader("Keyword Performance")
        
        # Create a DataFrame for keywords
        df_keywords = pd.DataFrame(keywords)
        
        # Display key metrics for keywords
        kw_cost = df_keywords['cost'].sum()
        kw_conversions = df_keywords['conversions'].sum()
        kw_clicks = df_keywords['clicks'].sum()
        kw_ctr = (kw_clicks / df_keywords['impressions'].sum() * 100) if df_keywords['impressions'].sum() > 0 else 0
        kw_conversion_rate = (kw_conversions / kw_clicks * 100) if kw_clicks > 0 else 0
        
        cols = st.columns(4)
        cols[0].metric("Keyword Spend", f"${kw_cost:.2f}")
        cols[1].metric("Keyword Conversions", f"{kw_conversions:.1f}")
        cols[2].metric("Keyword Conversion Rate", f"{kw_conversion_rate:.2f}%")
        cols[3].metric("Total Keywords", f"{len(keywords):,}")
        
        # Create tabs for keyword visualizations
        kw_tabs = st.tabs(["Top Keywords", "Keyword Table", "Quality Score Distribution"])
        
        with kw_tabs[0]:
            # Top 10 keywords by conversions
            top_keywords = df_keywords.sort_values('conversions', ascending=False).head(10)
            
            if not top_keywords.empty:
                fig = px.bar(
                    top_keywords,
                    x='keyword_text',
                    y=['cost', 'conversions'],
                    barmode='group',
                    labels={
                        'keyword_text': 'Keyword',
                        'cost': 'Cost ($)',
                        'conversions': 'Conversions',
                        'variable': 'Metric'
                    },
                    title='Top 10 Keywords by Conversions',
                    color_discrete_sequence=['#4285F4', '#34A853']
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No conversion data available for keywords")
        
        with kw_tabs[1]:
            # Display the keyword data as a sortable table
            if not df_keywords.empty:
                # Ensure all required columns exist
                required_cols = ['keyword_text', 'match_type', 'clicks', 'impressions', 'ctr', 
                                'average_cpc', 'cost', 'conversions', 'conversion_rate', 
                                'cost_per_conversion', 'quality_score']
                
                # Filter to columns that exist in the dataframe
                display_cols = [col for col in required_cols if col in df_keywords.columns]
                
                st.dataframe(
                    df_keywords[display_cols].sort_values('cost', ascending=False),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.write("No keyword data available")
        
        with kw_tabs[2]:
            # Quality score distribution
            if 'quality_score' in df_keywords.columns:
                quality_counts = df_keywords['quality_score'].value_counts().sort_index()
                
                fig = px.bar(
                    x=quality_counts.index,
                    y=quality_counts.values,
                    labels={'x': 'Quality Score', 'y': 'Number of Keywords'},
                    title='Keyword Quality Score Distribution',
                    color=quality_counts.index,
                    color_continuous_scale='RdYlGn'  # Red to Yellow to Green
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Average quality score
                avg_qs = df_keywords['quality_score'].mean()
                st.metric("Average Quality Score", f"{avg_qs:.1f} / 10")
            else:
                st.write("Quality score data not available")
    else:
        st.info("Keyword data not available. Fetch keyword data to see detailed keyword performance.")

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
    Render an editable UI for a single optimization suggestion.
    
    Args:
        suggestion (dict): The suggestion to render
        index (int): Index of the suggestion in the list
    """
    # Make a copy of the suggestion to track changes
    if str(index) not in st.session_state.edit_suggestions:
        st.session_state.edit_suggestions[str(index)] = copy.deepcopy(suggestion)
    
    edited_suggestion = st.session_state.edit_suggestions[str(index)]
    
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
    
    # Display suggestion header
    action_type = edited_suggestion.get('action_type', 'UNKNOWN')
    title = edited_suggestion.get('title', 'Untitled Suggestion')
    
    # Create columns for the header
    header_col1, header_col2 = st.columns([4, 1])
    
    with header_col1:
        st.markdown(f"### {index+1}. {title}")
        st.markdown(f"*Action Type: {action_type}*")
    
    with header_col2:
        # Display status or apply button
        if edited_suggestion.get('applied', False):
            st.success("Applied ✓")
        elif edited_suggestion.get('status') == 'failed':
            st.error("Failed ✗")
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
            
            # Allow editing the new bid with safe min_value
            edited_bid = st.number_input(
                "New Bid",
                min_value=min_value,
                value=float(new_bid),
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
            
            # Allow editing the new budget with safe min_value
            edited_budget = st.number_input(
                "New Budget",
                min_value=min_value,
                value=float(new_budget),
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
                logger.exception(f"Error scheduling task: {str(e)}")

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