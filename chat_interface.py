from openai import OpenAI
import re
import json
from datetime import datetime, timedelta

class ChatInterface:
    """
    Chat interface for the Google Ads Optimization Agent that allows for 
    natural language interaction with the agent.
    """
    def __init__(self, ads_api, optimizer, config, logger):
        """
        Initialize the chat interface with required components.
        
        Args:
            ads_api (GoogleAdsAPI): API instance for Google Ads
            optimizer (AdsOptimizer): Optimizer instance with GPT-4
            config (dict): Configuration dictionary
            logger (AdsAgentLogger): Logger instance
        """
        self.ads_api = ads_api
        self.optimizer = optimizer
        self.logger = logger
        self.openai_client = OpenAI(api_key=config['openai']['api_key'])
        self.chat_history = []
        self.last_data_refresh = None
        self.latest_campaigns = None
        self.latest_keywords = None
        self.data_freshness_threshold = timedelta(hours=4)  # Data considered fresh if < 4 hours old
        
        # Define command patterns with more specific triggers for different analyses
        self.command_patterns = {
            'fetch_data': r'(fetch|get|retrieve|pull|download).*(data|campaigns|performance)',
            'fetch_keywords': r'(fetch|get|retrieve|pull|download).*(keyword|keywords)',
            'analyze_campaigns': r'(analyze|evaluate|assess|optimization|optimize|suggestions).*(campaign|campaigns)',
            'analyze_keywords': r'(analyze|evaluate|assess|optimization|optimize|suggestions).*(keyword|keywords)',
            'comprehensive_analysis': r'(analyze|evaluate|assess|optimization|optimize|suggestions).*(account|full|complete|comprehensive)',
            'help': r'help|assist|guide|instructions|commands',
            'custom_query': r'query|search|find|filter',
            'schedule': r'schedule|automate|recurring|daily|weekly'
        }
    
    def add_message(self, role, content):
        """
        Add a message to the chat history.
        
        Args:
            role (str): Message role ('user', 'assistant', or 'system')
            content (str): Message content
        """
        self.chat_history.append({"role": role, "content": content})
        # Keep history to a reasonable size to avoid token limitations
        if len(self.chat_history) > 20:
            # Keep the first system message if present
            if self.chat_history[0]['role'] == 'system':
                self.chat_history = [self.chat_history[0]] + self.chat_history[-19:]
            else:
                self.chat_history = self.chat_history[-20:]
    
    def get_chat_history(self):
        """Get the current chat history"""
        return self.chat_history
    
    def detect_command(self, message):
        """
        Detect which command the user is trying to execute
        
        Args:
            message (str): User's message
            
        Returns:
            str: Detected command or None
        """
        for command, pattern in self.command_patterns.items():
            if re.search(pattern, message, re.IGNORECASE):
                return command
        return None
    
    def parse_parameters(self, message, command):
        """
        Extract parameters from user message based on command
        
        Args:
            message (str): User's message
            command (str): Detected command
            
        Returns:
            dict: Extracted parameters
        """
        params = {}
        
        if command == 'fetch_data':
            # Extract days parameter
            days_match = re.search(r'(\d+)\s*days?', message, re.IGNORECASE)
            if days_match:
                params['days'] = int(days_match.group(1))
            else:
                params['days'] = 30  # Default
        
        elif command == 'schedule':
            # Extract time parameters
            time_match = re.search(r'at\s*(\d{1,2})(?::(\d{2}))?(?:\s*(am|pm))?', message, re.IGNORECASE)
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2) or 0)
                period = time_match.group(3)
                
                if period and period.lower() == 'pm' and hour < 12:
                    hour += 12
                elif period and period.lower() == 'am' and hour == 12:
                    hour = 0
                    
                params['hour'] = hour
                params['minute'] = minute
            else:
                params['hour'] = 9
                params['minute'] = 0
                
            # Extract frequency parameter
            if 'daily' in message.lower():
                params['frequency'] = 'daily'
            elif 'weekly' in message.lower():
                params['frequency'] = 'weekly'
                # Extract day of week
                days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
                for day in days:
                    if day in message.lower():
                        params['day_of_week'] = day
                        break
                else:
                    params['day_of_week'] = 'monday'  # Default
            else:
                params['frequency'] = 'daily'  # Default
                
        return params
    
    def get_data_summary(self, campaigns, keywords):
        """
        Create a concise summary of campaign and keyword data for context.
        
        Args:
            campaigns (list): List of campaign data
            keywords (list): List of keyword data
            
        Returns:
            str: Formatted summary of the data
        """
        if not campaigns:
            return "No campaign data available."
        
        # Calculate key account metrics
        total_spend = sum(c['cost'] for c in campaigns)
        total_conversions = sum(c['conversions'] for c in campaigns)
        total_clicks = sum(c['clicks'] for c in campaigns)
        total_impressions = sum(c['impressions'] for c in campaigns)
        
        # Calculate account-level metrics
        overall_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        overall_conversion_rate = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
        overall_cost_per_conversion = (total_spend / total_conversions) if total_conversions > 0 else 0
        
        # Data date range
        days_of_data = campaigns[0].get('days', 30) if campaigns else 30
        
        # Create performance tiers for campaigns
        high_performing = []
        low_performing = []
        
        for campaign in campaigns:
            if campaign['conversion_rate'] > 2.0 and campaign['conversions'] > 5:
                high_performing.append({
                    'name': campaign['name'],
                    'stats': f"Conv Rate: {campaign['conversion_rate']:.2f}%, Conversions: {campaign['conversions']:.1f}, Cost: ${campaign['cost']:.2f}"
                })
            elif campaign['cost'] > 100 and campaign['conversions'] < 1:
                low_performing.append({
                    'name': campaign['name'],
                    'stats': f"Spent ${campaign['cost']:.2f} with {campaign['conversions']:.1f} conversions"
                })
        
        # Keyword insights
        keyword_insights = []
        if keywords and len(keywords) > 0:
            # Top converting keywords
            top_keywords = sorted(keywords, key=lambda k: k.get('conversions', 0), reverse=True)[:5]
            for kw in top_keywords:
                if kw.get('conversions', 0) > 0:
                    keyword_insights.append({
                        'text': kw['keyword_text'],
                        'stats': f"Conversions: {kw['conversions']:.1f}, Cost: ${kw['cost']:.2f}, Conv Rate: {kw['conversion_rate']:.2f}%, CPC: ${kw['average_cpc']:.2f}"
                    })
            
            # Find high-spend keywords with no conversions
            wasted_spend = []
            for kw in keywords:
                if kw.get('cost', 0) > 50 and kw.get('conversions', 0) < 1:
                    wasted_spend.append({
                        'text': kw['keyword_text'],
                        'stats': f"Cost: ${kw['cost']:.2f}, Clicks: {kw['clicks']}, Impressions: {kw['impressions']}, CTR: {kw['ctr']:.2f}%"
                    })
        
        # Format the summary
        summary = f"""
ACCOUNT PERFORMANCE SUMMARY (Last {days_of_data} Days):

Key Metrics:
- Total Campaigns: {len(campaigns)}
- Total Spend: ${total_spend:.2f}
- Total Conversions: {total_conversions:.1f}
- Total Clicks: {total_clicks}
- Total Impressions: {total_impressions:,}
- Overall CTR: {overall_ctr:.2f}%
- Overall Conversion Rate: {overall_conversion_rate:.2f}%
- Overall Cost Per Conversion: ${overall_cost_per_conversion:.2f}

Top Performing Campaigns:
{self._format_campaign_list(high_performing, "None identified")}

Underperforming Campaigns:
{self._format_campaign_list(low_performing, "None identified")}

Keyword Data:
- Total Keywords: {len(keywords) if keywords else 'N/A'}

Top Converting Keywords:
{self._format_keyword_list(keyword_insights, "None found")}

Keywords With High Spend & No Conversions:
{self._format_keyword_list(wasted_spend, "None found")}

Data Freshness: {self._get_data_freshness_string()}
"""
        return summary
    
    def _format_campaign_list(self, campaigns, default_message):
        """Format a list of campaigns for the data summary"""
        if not campaigns:
            return default_message
            
        result = ""
        for i, campaign in enumerate(campaigns[:3], 1):  # Show top 3
            result += f"{i}. {campaign['name']} - {campaign['stats']}\n"
        
        if len(campaigns) > 3:
            result += f"... and {len(campaigns) - 3} more"
            
        return result
        
    def _format_keyword_list(self, keywords, default_message):
        """Format a list of keywords for the data summary"""
        if not keywords:
            return default_message
            
        result = ""
        for i, keyword in enumerate(keywords[:3], 1):  # Show top 3
            result += f"{i}. \"{keyword['text']}\" - {keyword['stats']}\n"
        
        if len(keywords) > 3:
            result += f"... and {len(keywords) - 3} more"
            
        return result
    
    def _get_data_freshness_string(self):
        """Get a string indicating how fresh the data is"""
        if not self.last_data_refresh:
            return "Unknown (data has not been refreshed)"
            
        time_diff = datetime.now() - self.last_data_refresh
        
        if time_diff < timedelta(minutes=5):
            return "Very fresh (refreshed just now)"
        elif time_diff < timedelta(hours=1):
            return f"Fresh (refreshed {int(time_diff.total_seconds() / 60)} minutes ago)"
        elif time_diff < timedelta(hours=6):
            return f"Relatively fresh (refreshed {int(time_diff.total_seconds() / 3600)} hours ago)"
        else:
            return f"Stale (refreshed {int(time_diff.total_seconds() / 3600)} hours ago)"

    def ensure_data_context(self, days=30, force_refresh=False):
        """
        Ensure we have recent campaign and keyword data for context.
        If data is too old or missing, refresh it.
        
        Args:
            days (int): Number of days to fetch data for
            force_refresh (bool): If True, force a refresh even if data is recent
            
        Returns:
            tuple: (campaigns, keywords) with the latest data
        """
        # Check if we need to refresh the data
        current_time = datetime.now()
        needs_refresh = (
            force_refresh or
            self.last_data_refresh is None or 
            current_time - self.last_data_refresh > self.data_freshness_threshold or
            self.latest_campaigns is None
        )
        
        if needs_refresh:
            try:
                self.logger.info(f"Refreshing campaign and keyword data for chat context (last {days} days)")
                self.latest_campaigns = self.ads_api.get_campaign_performance(days_ago=days)
                self.latest_keywords = self.ads_api.get_keyword_performance(days_ago=days)
                self.last_data_refresh = current_time
                
                self.logger.info(f"Data refresh successful: {len(self.latest_campaigns)} campaigns, {len(self.latest_keywords) if self.latest_keywords else 0} keywords")
                return self.latest_campaigns, self.latest_keywords
            except Exception as e:
                error_message = f"Error refreshing data for chat context: {str(e)}"
                self.logger.error(error_message)
                # Return whatever we have, even if it's old
                return self.latest_campaigns, self.latest_keywords
        else:
            self.logger.info("Using cached data for chat context (still fresh)")
            return self.latest_campaigns, self.latest_keywords
    
    def process_user_message(self, message):
        """
        Process a user message and generate a response.
        
        Args:
            message (str): User's message
            
        Returns:
            tuple: (response message, command result)
        """
        self.add_message('user', message)
        self.logger.info(f"User message received: {message}")
        
        # Check for direct commands
        command = self.detect_command(message)
        
        if command:
            self.logger.info(f"Detected command: {command}")
            params = self.parse_parameters(message, command)
            
            try:
                if command == 'fetch_data':
                    days = params.get('days', 30)
                    self.logger.info(f"Fetching campaign data for last {days} days")
                    campaigns = self.ads_api.get_campaign_performance(days_ago=days)
                    self.latest_campaigns = campaigns  # Update stored data
                    self.last_data_refresh = datetime.now()
                    response = f"I've fetched data for {len(campaigns)} campaigns from the last {days} days. Here's a summary of your account performance:\n\n"
                    
                    # Add a brief summary of the campaigns
                    total_spend = sum(c['cost'] for c in campaigns)
                    total_conversions = sum(c['conversions'] for c in campaigns)
                    avg_ctr = sum(c['ctr'] for c in campaigns) / len(campaigns) if campaigns else 0
                    
                    response += f"• Total Spend: ${total_spend:.2f}\n"
                    response += f"• Total Conversions: {total_conversions:.1f}\n"
                    response += f"• Average CTR: {avg_ctr:.2f}%\n\n"
                    
                    response += "Would you like me to analyze this data and provide optimization recommendations?"
                    
                    return response, {'command': command, 'result': campaigns}
                
                elif command == 'fetch_keywords':
                    days = params.get('days', 30)
                    self.logger.info(f"Fetching keyword data for last {days} days")
                    # Ensure we also have campaign data
                    campaigns = self.latest_campaigns
                    if not campaigns:
                        campaigns = self.ads_api.get_campaign_performance(days_ago=days)
                        self.latest_campaigns = campaigns
                        
                    keywords = self.ads_api.get_keyword_performance(days_ago=days)
                    self.latest_keywords = keywords
                    self.last_data_refresh = datetime.now()
                    
                    response = f"I've fetched data for {len(keywords)} keywords across {len(campaigns)} campaigns from the last {days} days."
                    
                    # Add a brief summary of top keywords
                    if keywords:
                        # Find keywords with conversions
                        converting_keywords = [k for k in keywords if k.get('conversions', 0) > 0]
                        converting_keywords.sort(key=lambda k: k.get('conversions', 0), reverse=True)
                        
                        if converting_keywords:
                            response += "\n\nTop converting keywords:\n"
                            for i, keyword in enumerate(converting_keywords[:3], 1):
                                response += f"{i}. \"{keyword['keyword_text']}\" - {keyword['conversions']:.1f} conversions, ${keyword['cost']:.2f} spent\n"
                        
                        # Find keywords with high spend and no conversions
                        wasted_keywords = [k for k in keywords if k.get('cost', 0) > 50 and k.get('conversions', 0) < 1]
                        wasted_keywords.sort(key=lambda k: k.get('cost', 0), reverse=True)
                        
                        if wasted_keywords:
                            response += "\n\nKeywords with high spend and no conversions:\n"
                            for i, keyword in enumerate(wasted_keywords[:3], 1):
                                response += f"{i}. \"{keyword['keyword_text']}\" - ${keyword['cost']:.2f} spent, {keyword['clicks']} clicks\n"
                    
                    response += "\n\nWould you like me to analyze these keywords and provide optimization suggestions?"
                    
                    return response, {'command': command, 'result': keywords}
                
                elif command == 'analyze_campaigns':
                    days = params.get('days', 30)
                    self.logger.info(f"Analyzing campaign data for last {days} days")
                    # Ensure we have fresh campaign data
                    campaigns, _ = self.ensure_data_context(days)
                    
                    if not campaigns:
                        return "I don't have any campaign data to analyze. Let me fetch that for you first.", None
                    
                    suggestions = self.optimizer.get_optimization_suggestions(campaigns)
                    
                    if isinstance(suggestions, list):
                        suggestion_count = len(suggestions)
                        response = f"I've completed the campaign analysis for the last {days} days and have {suggestion_count} optimization suggestions for you.\n\n"
                        
                        # Add a brief summary of top 3 suggestions
                        if suggestion_count > 0:
                            response += "Here are the top 3 recommendations:\n\n"
                            for i, sugg in enumerate(suggestions[:3], 1):
                                response += f"{i}. {sugg.get('action_type', 'ACTION')}: {sugg.get('title', 'Untitled')}\n"
                                if 'entity_type' in sugg and 'entity_id' in sugg:
                                    response += f"   • For {sugg['entity_type']} '{sugg['entity_id']}'\n"
                                if 'change' in sugg:
                                    response += f"   • {sugg['change']}\n"
                                response += "\n"
                        
                        response += "You can view all the detailed suggestions in the Optimization tab of the application."
                    else:
                        response = f"I've analyzed your campaigns but couldn't generate specific suggestions: {suggestions}"
                    
                    return response, {'command': command, 'result': suggestions}
                
                elif command == 'analyze_keywords':
                    days = params.get('days', 30)
                    self.logger.info(f"Analyzing keyword data for last {days} days")
                    # Ensure we have fresh data
                    campaigns, keywords = self.ensure_data_context(days)
                    
                    if not campaigns:
                        return "I don't have any campaign data to analyze. Let me fetch that for you first.", None
                    
                    if not keywords:
                        # Try to fetch keywords if not available
                        try:
                            keywords = self.ads_api.get_keyword_performance(days_ago=days)
                            self.latest_keywords = keywords
                            self.last_data_refresh = datetime.now()
                        except Exception as e:
                            self.logger.error(f"Error fetching keyword data: {str(e)}")
                            return f"I couldn't fetch keyword data: {str(e)}. Please check your API connection.", None
                    
                    suggestions = self.optimizer.get_optimization_suggestions(campaigns, keywords)
                    
                    if isinstance(suggestions, list):
                        keyword_suggestions = [s for s in suggestions if s.get('entity_type') == 'keyword']
                        suggestion_count = len(keyword_suggestions)
                        
                        response = f"I've completed the keyword analysis for the last {days} days and have {suggestion_count} keyword-specific optimization suggestions for you.\n\n"
                        
                        # Add a brief summary of top 3 suggestions
                        if suggestion_count > 0:
                            response += "Here are the top 3 keyword recommendations:\n\n"
                            for i, sugg in enumerate(keyword_suggestions[:3], 1):
                                response += f"{i}. {sugg.get('action_type', 'ACTION')}: {sugg.get('title', 'Untitled')}\n"
                                if 'entity_id' in sugg:
                                    response += f"   • For keyword '{sugg['entity_id']}'\n"
                                if 'change' in sugg:
                                    response += f"   • {sugg['change']}\n"
                                response += "\n"
                        
                        response += "You can view all the detailed suggestions in the Optimization tab of the application."
                    else:
                        response = f"I've analyzed your keywords but couldn't generate specific suggestions: {suggestions}"
                    
                    return response, {'command': command, 'result': suggestions}
                
                elif command == 'comprehensive_analysis':
                    days = params.get('days', 30)
                    self.logger.info(f"Running comprehensive analysis for last {days} days")
                    # Force a refresh of both campaigns and keywords
                    campaigns, keywords = self.ensure_data_context(days, force_refresh=True)
                    
                    if not campaigns:
                        return "I couldn't fetch campaign data for analysis. Please check your API connection.", None
                    
                    if not keywords:
                        return "I fetched campaign data but couldn't get keyword data. Running campaign-only analysis.", None
                    
                    suggestions = self.optimizer.get_optimization_suggestions(campaigns, keywords)
                    
                    if isinstance(suggestions, list):
                        suggestion_count = len(suggestions)
                        campaign_suggestions = len([s for s in suggestions if s.get('entity_type') == 'campaign'])
                        keyword_suggestions = len([s for s in suggestions if s.get('entity_type') == 'keyword'])
                        
                        response = f"I've completed a comprehensive account analysis for the last {days} days and found {suggestion_count} total optimization opportunities:\n\n"
                        response += f"• {campaign_suggestions} campaign-level suggestions\n"
                        response += f"• {keyword_suggestions} keyword-level suggestions\n\n"
                        
                        # Add a brief summary of top suggestions by type
                        if suggestion_count > 0:
                            response += "Here are my top recommendations:\n\n"
                            
                            # Get top 2 of each type
                            campaign_suggs = [s for s in suggestions if s.get('entity_type') == 'campaign'][:2]
                            keyword_suggs = [s for s in suggestions if s.get('entity_type') == 'keyword'][:2]
                            
                            for i, sugg in enumerate(campaign_suggs + keyword_suggs, 1):
                                response += f"{i}. {sugg.get('action_type', 'ACTION')} for {sugg.get('entity_type', 'entity')}: {sugg.get('title', 'Untitled')}\n"
                                if 'entity_id' in sugg:
                                    response += f"   • Target: '{sugg['entity_id']}'\n"
                                if 'change' in sugg:
                                    response += f"   • Change: {sugg['change']}\n"
                                response += "\n"
                        
                        response += "You can view all the detailed suggestions in the Optimization tab of the application."
                    else:
                        response = f"I attempted a comprehensive analysis but couldn't generate specific suggestions: {suggestions}"
                    
                    return response, {'command': command, 'result': suggestions}
                
                elif command == 'help':
                    help_text = """
I can help you optimize your Google Ads campaigns. Here are commands you can use:

• **Campaign Data**: "Fetch campaign data for the last 14 days"
• **Keyword Data**: "Fetch keyword data for the last 30 days"
• **Campaign Analysis**: "Analyze my campaigns and give optimization suggestions"
• **Keyword Analysis**: "Analyze my keywords and provide bid suggestions"
• **Full Account Analysis**: "Run a comprehensive account analysis"
• **Custom Query**: "Find campaigns with low CTR" or "Which keywords need bid adjustments?"
• **Scheduler**: "Schedule daily campaign analysis at 9am"

You can also ask me questions about PPC strategy, ad optimization, or anything related to your Google Ads performance.
"""
                    return help_text, {'command': command, 'result': 'help_displayed'}
                
                elif command == 'schedule':
                    response = f"Setting up scheduled task with parameters: {json.dumps(params, indent=2)}"
                    return response, {'command': command, 'result': params}
                
            except Exception as e:
                self.logger.exception(f"Error processing command '{command}': {str(e)}")
                error_message = f"I encountered an error processing your request: {str(e)}"
                return error_message, {'command': command, 'error': str(e)}
        
        # For non-command messages, use GPT to generate a response
        try:
            # Ensure we have data context - auto-refresh if older than threshold
            campaigns, keywords = self.ensure_data_context()
            data_summary = self.get_data_summary(campaigns, keywords)
            
            # Create messages including chat history for context
            messages = self.chat_history[-10:]  # Use last 10 messages for context
            
            # Add a system message at the beginning with data context
            system_message = f"""
You are an expert Google Ads PPC specialist assistant helping to optimize the user's advertising campaigns.
You've worked in digital advertising for over 10 years and have managed millions in ad spend.

CURRENT ACCOUNT DATA:
{data_summary}

RESPONSE GUIDELINES:
1. Base all responses on the specific performance data shown above
2. Provide data-driven, specific advice rather than generic statements
3. When appropriate, mention specific campaigns, keywords or metrics from the data
4. If the user asks about something not covered in the available data, suggest how they could get that data
5. For optimization questions, focus on actionable steps based on the most significant opportunities in the data
6. If you're not sure about something, admit it and suggest how to find out

The user can perform actions like "fetch campaign data", "analyze keywords", or "run comprehensive analysis" 
to get more data or analytical insights.
"""
            
            messages = [
                {"role": "system", "content": system_message}
            ] + messages
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=1200,
                temperature=0.3  # Lower temperature for more factual, data-driven responses
            )
            
            response_text = response.choices[0].message.content
            self.add_message('assistant', response_text)
            return response_text, None
            
        except Exception as e:
            self.logger.exception(f"Error generating chat response: {str(e)}")
            error_message = f"I encountered an error: {str(e)}"
            self.add_message('assistant', error_message)
            return error_message, None 