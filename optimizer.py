from openai import OpenAI
import re
import json

class AdsOptimizer:
    def __init__(self, config):
        """
        Initialize the OpenAI client for campaign optimization.
        
        Args:
            config (dict): Configuration dictionary with OpenAI API key
        """
        self.client = OpenAI(api_key=config['api_key'])
    
    def format_campaign_data(self, campaigns):
        """
        Format campaign data into a readable string for GPT-4.
        
        Args:
            campaigns (list): List of campaign data dictionaries
            
        Returns:
            str: Formatted campaign data
        """
        if not campaigns:
            return "No campaign data available."
        
        formatted_data = "Google Ads Campaign Performance Data:\n\n"
        
        for campaign in campaigns:
            formatted_data += f"Campaign: {campaign['name']} (ID: {campaign['id']})\n"
            formatted_data += f"- Clicks: {campaign['clicks']}\n"
            formatted_data += f"- Impressions: {campaign['impressions']}\n"
            formatted_data += f"- CTR: {campaign['ctr']:.2f}%\n"
            formatted_data += f"- Average CPC: ${campaign['average_cpc']:.2f}\n"
            formatted_data += f"- Conversions: {campaign['conversions']:.2f}\n"
            formatted_data += f"- Cost: ${campaign['cost']:.2f}\n"
            formatted_data += f"- Conversion Rate: {campaign['conversion_rate']:.2f}%\n"
            formatted_data += f"- Cost Per Conversion: ${campaign['cost_per_conversion']:.2f}\n\n"
        
        return formatted_data
    
    def format_keyword_data(self, keywords):
        """
        Format keyword data into a readable string for GPT-4.
        
        Args:
            keywords (list): List of keyword data dictionaries
            
        Returns:
            str: Formatted keyword data
        """
        if not keywords:
            return "No keyword data available."
        
        formatted_data = "Google Ads Keyword Performance Data:\n\n"
        
        # Group keywords by campaign and ad group
        campaigns = {}
        for keyword in keywords:
            campaign_id = keyword['campaign_id']
            ad_group_id = keyword['ad_group_id']
            
            if campaign_id not in campaigns:
                campaigns[campaign_id] = {
                    'name': keyword['campaign_name'],
                    'ad_groups': {}
                }
                
            if ad_group_id not in campaigns[campaign_id]['ad_groups']:
                campaigns[campaign_id]['ad_groups'][ad_group_id] = {
                    'name': keyword['ad_group_name'],
                    'keywords': []
                }
                
            campaigns[campaign_id]['ad_groups'][ad_group_id]['keywords'].append(keyword)
        
        # Format the data
        for campaign_id, campaign in campaigns.items():
            formatted_data += f"Campaign: {campaign['name']} (ID: {campaign_id})\n"
            
            for ad_group_id, ad_group in campaign['ad_groups'].items():
                formatted_data += f"  Ad Group: {ad_group['name']} (ID: {ad_group_id})\n"
                
                # Sort keywords by clicks (descending)
                sorted_keywords = sorted(ad_group['keywords'], key=lambda k: k['clicks'], reverse=True)
                
                for keyword in sorted_keywords:
                    formatted_data += f"    Keyword: {keyword['keyword_text']} (ID: {ad_group_id}/{keyword['keyword_text']})\n"
                    formatted_data += f"    - Status: {keyword['status']}\n"
                    formatted_data += f"    - Quality Score: {keyword['quality_score']}\n"
                    formatted_data += f"    - Current Bid: ${keyword['current_bid']:.2f}\n"
                    formatted_data += f"    - Clicks: {keyword['clicks']}\n"
                    formatted_data += f"    - Impressions: {keyword['impressions']}\n"
                    formatted_data += f"    - CTR: {keyword['ctr']:.2f}%\n"
                    formatted_data += f"    - Average CPC: ${keyword['average_cpc']:.2f}\n"
                    formatted_data += f"    - Conversions: {keyword['conversions']:.2f}\n"
                    formatted_data += f"    - Cost: ${keyword['cost']:.2f}\n"
                    if keyword['conversions'] > 0:
                        formatted_data += f"    - Cost Per Conversion: ${keyword['cost_per_conversion']:.2f}\n"
                    formatted_data += "\n"
        
        return formatted_data
    
    def get_optimization_suggestions(self, campaigns, keywords=None):
        """
        Send campaign and keyword data to GPT-4 and get optimization suggestions.
        
        Args:
            campaigns (list): List of campaign data dictionaries
            keywords (list, optional): List of keyword data dictionaries
            
        Returns:
            list: List of structured optimization suggestions with metadata
        """
        if not campaigns:
            return "No campaign data available for optimization."
        
        campaign_data = self.format_campaign_data(campaigns)
        keyword_data = self.format_keyword_data(keywords) if keywords else "No keyword data available."
        
        # Determine the scope of analysis based on available data
        analysis_type = "comprehensive" if keywords else "campaign-only"
        days_analyzed = campaigns[0].get('days', 30) if campaigns else 30
        
        # Enhanced, highly specific prompt for the PPC expert
        prompt = f"""
        You are a senior Google Ads PPC consultant with 10+ years experience optimizing campaigns.
        Your expertise includes advanced bid strategies, account structure optimization, and granular keyword-level analysis.
        You are renowned for your ability to provide extremely specific, actionable recommendations that maximize ROAS.
        
        DATA CONTEXT:
        You are analyzing Google Ads data for the past {days_analyzed} days. 
        Analysis type: {analysis_type}
        
        DATA SUMMARY:
        - Campaign data: {len(campaigns)} campaigns available
        - Keyword data: {"Available - " + str(len(keywords)) + " keywords" if keywords else "Not available"}
        
        OPTIMIZATION TASK:
        Based on the detailed performance data below, provide extremely specific, data-driven optimization recommendations.
        Your suggestions must be actionable, precise, and backed by the performance metrics in the data.
        
        OPTIMIZATION PRIORITY:
        You MUST focus primarily on keyword-level optimization. This is critical. Analyze EVERY single keyword in the account
        and make specific bid adjustments based on performance metrics. This is your highest priority task.
        
        OPTIMIZATION REQUIREMENTS:
        
        1. KEYWORD BID ADJUSTMENTS (HIGHEST PRIORITY): 
           - Analyze EVERY keyword in the account and make bid adjustment recommendations
           - For high-performing keywords (good conversion rate, positive ROAS), suggest specific bid increases with exact amounts
           - For underperforming keywords with high spend and low conversions, suggest bid reductions with exact amounts
           - Include the exact current bid and the specific recommended new bid (e.g., "Increase from $1.20 to $1.45")
           - Consider quality score in your recommendations
           - Provide at least 5-10 keyword-level bid adjustments
        
        2. STATUS CHANGES:
           - Identify specific keywords or campaigns to pause based on clear performance thresholds
           - Include exact metrics that justify the pause (e.g., "Spent $245 with 0 conversions over {days_analyzed} days")
           - Recommend keywords to enable if they show high potential but are currently paused
        
        3. BUDGET ADJUSTMENTS:
           - Recommend specific budget increases for campaigns with high conversion rates and limited by budget
           - Suggest exact budget amounts, not just percentages (e.g., "Increase daily budget from $50.00 to $75.00")
           - Justify with performance metrics like conversion rate, ROAS, or impression share lost due to budget
        
        4. MATCH TYPE & QUALITY OPTIMIZATIONS:
           - Identify keywords with low quality scores and suggest specific improvements
           - Recommend match type changes where appropriate (e.g., from broad to phrase or phrase to exact)
           - Suggest new negative keywords based on performance patterns
        
        FORMAT YOUR RESPONSE:
        Number each suggestion and follow this precise format with COMPLETE TITLES:
        
        1. [ACTION_TYPE] Clear and complete title describing the change
        - Entity: [ENTITY_TYPE] ID: entity_id_or_name
        - Current Value: [current value/status/bid/etc.]
        - Change: [Specific change with exact values]
        - Expected Impact: [Data-backed prediction of results]
        - Rationale: [Specific metrics that justify this change]
        
        For example:
        
        1. [BID_ADJUSTMENT] Increase bid for high-converting keyword "premium widgets"
        - Entity: [KEYWORD] ID: "premium widgets"
        - Current Value: $0.75 CPC, 8.5% conversion rate, 450% ROAS
        - Change: Increase bid by 25% from $0.75 to $0.94
        - Expected Impact: 15-20% more conversions with maintained ROAS
        - Rationale: This keyword has a conversion rate 3x the account average (8.5% vs 2.8%) and generates a positive ROAS of 450%. Quality score is good (8/10). Increasing visibility will likely yield more of these valuable conversions.
        
        IMPORTANT: Your recommendations must be extremely specific and immediately actionable. Include exact values for all changes. Make sure they are directly supported by the data provided. FOCUS ON KEYWORD-LEVEL OPTIMIZATIONS and provide at least 5-10 specific keyword bid adjustments.
        
        AVAILABLE DATA:

        {campaign_data}
        
        {keyword_data}
        
        Use these action types in your suggestions:
        - BID_ADJUSTMENT: For changing keyword or ad group bids
        - STATUS_CHANGE: For pausing/enabling keywords, ad groups or campaigns
        - BUDGET_ADJUSTMENT: For modifying campaign budgets
        - MATCH_TYPE_CHANGE: For changing keyword match types
        - QUALITY_IMPROVEMENT: For suggestions to improve quality scores
        - NEGATIVE_KEYWORD: For adding negative keywords
        - CAMPAIGN_SETTINGS: For adjusting campaign settings like bidding strategy
        """
        
        try:
            # Enhanced system message with more specific instructions
            system_message = """
You are an elite Google Ads PPC specialist with expertise in campaign optimization. You provide extremely precise, data-driven recommendations backed by performance metrics.

When analyzing Google Ads accounts, you:
1. Focus on the most impactful opportunities first
2. Provide exact values for all recommended changes (specific bids, budgets, etc.)
3. Always justify recommendations with specific performance metrics
4. Identify both problems (poor performers) and opportunities (scaling top performers)
5. Consider keyword quality scores, match types, and other technical factors in your analysis

Your recommendations are comprehensive, granular, and ready for immediate implementation.
"""
            
            # Enhanced parameters for comprehensive keyword-level optimization - SINGLE CLEAN API CALL
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,  # Increased for comprehensive keyword-level analysis  
                temperature=0.1    # Lower temperature for precise, data-driven output
            )
            
            suggestions_text = response.choices[0].message.content
            return self.parse_suggestions(suggestions_text, campaigns, keywords)
            
        except Exception as e:
            return f"Error generating optimization suggestions: {str(e)}"
    
    def parse_suggestions(self, suggestions_text, campaigns, keywords=None):
        """
        Parse the raw suggestions text from GPT-4 into structured optimization suggestions.
        
        Args:
            suggestions_text (str): Raw suggestions text from GPT-4
            campaigns (list): List of campaign data dictionaries for reference
            keywords (list, optional): List of keyword data dictionaries for reference
            
        Returns:
            list: List of structured optimization suggestions with metadata
        """
        suggestions = []
        
        # Extract the numbered suggestions
        pattern = r'(\d+)\.\s+\[([A-Z_]+)\]\s+(.*?)(?=\d+\.\s+\[|$)'
        matches = re.finditer(pattern, suggestions_text, re.DOTALL)
        
        for match in matches:
            index = match.group(1)
            action_type = match.group(2)
            content = match.group(3).strip()
            
            # Extract entity type and ID
            entity_match = re.search(r'Entity:\s+\[([A-Z]+)\]\s+ID:\s+([^\n]+)', content)
            entity_type = entity_match.group(1).lower() if entity_match else None
            entity_id = entity_match.group(2).strip() if entity_match else None
            
            # Extract the change
            change_match = re.search(r'Change:\s+([^\n]+)', content)
            change = change_match.group(1).strip() if change_match else None
            
            # Extract the rationale
            rationale_match = re.search(r'Rationale:\s+(.*?)(?=\s*-\s*Expected Impact:|$)', content, re.DOTALL)
            rationale = rationale_match.group(1).strip() if rationale_match else None
            
            # Extract full title - fix the title truncation issue
            title_match = re.search(r'\[([A-Z_]+)\]\s+(.*?)(?=\s*-|$)', content)
            title = title_match.group(2).strip() if title_match else content.split('\n')[0].strip()
            
            # Create structured suggestion
            suggestion = {
                'index': int(index),
                'title': title,
                'action_type': action_type,
                'entity_type': entity_type,
                'entity_id': entity_id,
                'change': change,
                'rationale': rationale,
                'original_text': content,
                'edited': False,
                'applied': False,
                'status': 'pending',
                'result_message': ''
            }
            
            # Add additional fields based on entity type
            if entity_type == 'keyword' and keywords:
                for keyword in keywords:
                    # This is a simplified matching approach - in production, you'd want more robust matching
                    if str(keyword.get('keyword_text', '')).lower() in entity_id.lower():
                        suggestion['current_value'] = keyword.get('current_bid') if action_type == 'BID_ADJUSTMENT' else keyword.get('status')
                        break
            
            elif entity_type == 'campaign' and campaigns:
                for campaign in campaigns:
                    if str(campaign.get('id', '')) == entity_id or str(campaign.get('name', '')).lower() == entity_id.lower():
                        suggestion['current_value'] = campaign.get('budget') if action_type == 'BUDGET_ADJUSTMENT' else None
                        break
            
            # Parse specific change values where possible
            if action_type == 'BID_ADJUSTMENT' and change:
                if 'increase' in change.lower():
                    match = re.search(r'by\s+(\d+(?:\.\d+)?)%', change)
                    if match:
                        percentage = float(match.group(1))
                        suggestion['change_value'] = {
                            'type': 'percentage_increase',
                            'value': percentage
                        }
                elif 'decrease' in change.lower():
                    match = re.search(r'by\s+(\d+(?:\.\d+)?)%', change)
                    if match:
                        percentage = float(match.group(1))
                        suggestion['change_value'] = {
                            'type': 'percentage_decrease',
                            'value': percentage
                        }
                elif '$' in change:
                    match = re.search(r'\$(\d+(?:\.\d+)?)', change)
                    if match:
                        amount = float(match.group(1))
                        suggestion['change_value'] = {
                            'type': 'absolute',
                            'value': amount
                        }
            
            elif action_type == 'STATUS_CHANGE' and change:
                if 'pause' in change.lower():
                    suggestion['change_value'] = {
                        'type': 'status',
                        'value': 'PAUSED'
                    }
                elif 'enable' in change.lower() or 'reactivate' in change.lower():
                    suggestion['change_value'] = {
                        'type': 'status',
                        'value': 'ENABLED'
                    }
                elif 'remove' in change.lower():
                    suggestion['change_value'] = {
                        'type': 'status',
                        'value': 'REMOVED'
                    }
            
            suggestions.append(suggestion)
        
        return suggestions 