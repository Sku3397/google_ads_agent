from openai import OpenAI
import re
import json
from datetime import datetime

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
        
        # Format the data with comprehensive metrics for better analysis
        for campaign_id, campaign in campaigns.items():
            formatted_data += f"Campaign: {campaign['name']} (ID: {campaign_id})\n"
            
            for ad_group_id, ad_group in campaign['ad_groups'].items():
                formatted_data += f"  Ad Group: {ad_group['name']} (ID: {ad_group_id})\n"
                
                # Sort keywords by clicks (descending)
                sorted_keywords = sorted(ad_group['keywords'], key=lambda k: k['clicks'], reverse=True)
                
                # Calculate ad group level metrics for comparison
                if sorted_keywords:
                    total_clicks = sum(k['clicks'] for k in sorted_keywords)
                    total_impressions = sum(k['impressions'] for k in sorted_keywords)
                    total_conversions = sum(k['conversions'] for k in sorted_keywords)
                    total_cost = sum(k['cost'] for k in sorted_keywords)
                    
                    # Only calculate rates if denominators are non-zero
                    avg_ctr = (sum(k['clicks'] for k in sorted_keywords) / sum(k['impressions'] for k in sorted_keywords) * 100) if total_impressions > 0 else 0
                    avg_conv_rate = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
                    avg_cost_per_conv = (total_cost / total_conversions) if total_conversions > 0 else 0
                    
                    formatted_data += f"  Ad Group Totals: {total_clicks} clicks, {total_impressions} impr, {total_conversions:.1f} conv, ${total_cost:.2f} cost\n"
                    formatted_data += f"  Ad Group Averages: {avg_ctr:.2f}% CTR, {avg_conv_rate:.2f}% conv rate, ${avg_cost_per_conv:.2f} cost/conv\n\n"
                
                for keyword in sorted_keywords:
                    formatted_data += f"    Keyword: {keyword['keyword_text']} [{keyword['match_type']}] (ID: {keyword['id']})\n"
                    formatted_data += f"    - Status: {keyword['status']}\n"
                    formatted_data += f"    - Quality Score: {keyword['quality_score']}/10\n"
                    formatted_data += f"    - Current Bid: ${keyword['current_bid']:.2f}\n"
                    formatted_data += f"    - Clicks: {keyword['clicks']} | Impressions: {keyword['impressions']}\n"
                    formatted_data += f"    - CTR: {keyword['ctr']:.2f}% | Avg CPC: ${keyword['average_cpc']:.2f}\n"
                    formatted_data += f"    - Conversions: {keyword['conversions']:.1f} | Conv. Rate: {keyword['conversion_rate']:.2f}%\n"
                    formatted_data += f"    - Cost: ${keyword['cost']:.2f}"
                    
                    # Include cost per conversion only when there are conversions
                    if keyword['conversions'] > 0:
                        formatted_data += f" | Cost/Conv: ${keyword['cost_per_conversion']:.2f}\n"
                    else:
                        formatted_data += " | Cost/Conv: No conversions\n"
                        
                    # Include ROAS if available
                    if 'roas' in keyword and keyword['roas'] > 0:
                        formatted_data += f"    - ROAS: {keyword['roas']:.0f}%\n"
                        
                    # Include impression share metrics if available
                    if 'search_impression_share' in keyword and keyword['search_impression_share'] is not None:
                        formatted_data += f"    - Search Impression Share: {keyword['search_impression_share'] * 100:.1f}%\n"
                        
                    if 'top_impression_pct' in keyword and keyword['top_impression_pct'] is not None:
                        formatted_data += f"    - Top of Page Rate: {keyword['top_impression_pct'] * 100:.1f}%\n"
                        
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
        
        try:
            # Format data for GPT-4
            campaign_data = self.format_campaign_data(campaigns)
            keyword_data = self.format_keyword_data(keywords) if keywords else "No keyword data available."
            
            # Determine the scope of analysis based on available data
            analysis_type = "comprehensive" if keywords and len(keywords) > 0 else "campaign-only"
            days_analyzed = campaigns[0].get('days', 30) if campaigns else 30
            
            # Calculate summary metrics to inform the AI about account performance context
            if campaigns:
                total_campaign_clicks = sum(c.get('clicks', 0) for c in campaigns)
                total_campaign_impressions = sum(c.get('impressions', 0) for c in campaigns)
                total_campaign_cost = sum(c.get('cost', 0) for c in campaigns)
                total_campaign_conversions = sum(c.get('conversions', 0) for c in campaigns)
                avg_campaign_ctr = (total_campaign_clicks / total_campaign_impressions * 100) if total_campaign_impressions > 0 else 0
                avg_campaign_conv_rate = (total_campaign_conversions / total_campaign_clicks * 100) if total_campaign_clicks > 0 else 0
                avg_campaign_cpc = total_campaign_cost / total_campaign_clicks if total_campaign_clicks > 0 else 0
                avg_campaign_cpa = total_campaign_cost / total_campaign_conversions if total_campaign_conversions > 0 else 0
            
            # Enhanced, highly specific prompt for the PPC expert
            prompt = f"""
            You are a senior Google Ads PPC consultant with 10+ years experience optimizing campaigns.
            Your expertise includes advanced bid strategies, account structure optimization, and granular keyword-level analysis.
            You are renowned for your ability to provide extremely specific, actionable recommendations that maximize ROAS.
            
            DATA CONTEXT:
            You are analyzing Google Ads data for the past {days_analyzed} days. 
            Analysis type: {analysis_type}
            
            ACCOUNT PERFORMANCE SUMMARY:
            - Total clicks: {total_campaign_clicks:,}
            - Total impressions: {total_campaign_impressions:,}
            - Total cost: ${total_campaign_cost:,.2f}
            - Total conversions: {total_campaign_conversions:.1f}
            - Average CTR: {avg_campaign_ctr:.2f}%
            - Average conversion rate: {avg_campaign_conv_rate:.2f}%
            - Average CPC: ${avg_campaign_cpc:.2f}
            - Average CPA: ${avg_campaign_cpa:.2f if total_campaign_conversions > 0 else 'No conversions'}
            
            DATA SUMMARY:
            - Campaign data: {len(campaigns)} campaigns available
            - Keyword data: {"Available - " + str(len(keywords)) + " keywords" if keywords and len(keywords) > 0 else "Not available"}
            
            OPTIMIZATION TASK:
            Based on the detailed performance data below, provide extremely specific, data-driven optimization recommendations.
            Your suggestions must be actionable, precise, and backed by the performance metrics in the data.
            
            OPTIMIZATION PRIORITY:
            {"You MUST focus primarily on keyword-level optimization. This is critical. Analyze EVERY single keyword in the account and make specific bid adjustments based on performance metrics. This is your highest priority task." if keywords and len(keywords) > 0 else "Since keyword data is not available, focus on campaign-level optimizations such as budget adjustments and status changes."}
            
            OPTIMIZATION GUIDELINES:
            
            1. {"KEYWORD BID ADJUSTMENTS (HIGHEST PRIORITY): \n   - Analyze EVERY keyword in the account and make bid adjustment recommendations\n   - For high-performing keywords (good conversion rate, positive ROAS), suggest specific bid increases with exact amounts\n   - For underperforming keywords with high spend and low conversions, suggest bid reductions with exact amounts\n   - Include the exact current bid and the specific recommended new bid (e.g., \"Increase from $1.20 to $1.45\")\n   - Consider quality score in your recommendations\n   - Provide at least 5-10 keyword-level bid adjustments" if keywords and len(keywords) > 0 else "CAMPAIGN BUDGET ADJUSTMENTS (HIGHEST PRIORITY):\n   - Recommend specific budget adjustments for each campaign based on performance\n   - Include the exact current budget and recommended new budget\n   - Justify with conversion rate and cost per conversion metrics"}
            
            2. STATUS CHANGES:
               - Identify specific {"keywords or" if keywords and len(keywords) > 0 else ""} campaigns to pause based on clear performance thresholds
               - Include exact metrics that justify the pause (e.g., "Spent $245 with 0 conversions over {days_analyzed} days")
               - Recommend {"keywords or" if keywords and len(keywords) > 0 else ""} campaigns to enable if they show high potential but are currently paused
            
            3. {"BUDGET ADJUSTMENTS:" if keywords and len(keywords) > 0 else "ADDITIONAL CAMPAIGN OPTIMIZATIONS:"}
               - Recommend specific budget increases for campaigns with high conversion rates and limited by budget
               - Suggest exact budget amounts, not just percentages (e.g., "Increase daily budget from $50.00 to $75.00")
               - Justify with performance metrics like conversion rate, ROAS, or impression share lost due to budget
            
            4. {"MATCH TYPE & QUALITY OPTIMIZATIONS:\n   - Identify keywords with low quality scores and suggest specific improvements\n   - Recommend match type changes where appropriate (e.g., from broad to phrase or phrase to exact)\n   - Suggest new negative keywords based on performance patterns" if keywords and len(keywords) > 0 else "BID STRATEGY OPTIMIZATION:\n   - Analyze current bidding strategies and recommend changes if appropriate\n   - Suggest specific bidding strategies for each campaign based on goals and performance"}
            
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
            
            IMPORTANT: Your recommendations must be extremely specific and immediately actionable. Include exact values for all changes. Make sure they are directly supported by the data provided. {"FOCUS ON KEYWORD-LEVEL OPTIMIZATIONS and provide at least 5-10 specific keyword bid adjustments." if keywords and len(keywords) > 0 else "FOCUS ON CAMPAIGN-LEVEL OPTIMIZATIONS for each campaign in the account."}
            
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
            
            # Parse suggestions into structured format
            structured_suggestions = self.parse_suggestions(suggestions_text, campaigns, keywords)
            
            # Validate we got actionable suggestions
            if isinstance(structured_suggestions, list) and len(structured_suggestions) > 0:
                return structured_suggestions
            else:
                # If no suggestions were parsed, return the raw text
                return [{"title": "Optimization Suggestions", "content": suggestions_text, "type": "raw"}]
                
        except Exception as e:
            error_message = f"Error generating optimization suggestions: {str(e)}"
            return [{"title": "Error", "content": error_message, "type": "error"}]
    
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
        
        # Improved regex pattern to capture all suggestions even with complex formatting
        pattern = r'(\d+)\.\s+\[([A-Z_]+)\]\s+(.*?)(?=\d+\.\s+\[|$)'
        matches = re.finditer(pattern, suggestions_text, re.DOTALL)
        
        suggestion_count = 0
        for match in matches:
            suggestion_count += 1
            index = match.group(1)
            action_type = match.group(2)
            content = match.group(3).strip()
            
            # Extract entity type and ID with improved pattern matching
            entity_match = re.search(r'Entity:\s+\[([A-Z]+)\]\s+ID:\s+(.*?)(?=\n|$)', content, re.DOTALL)
            entity_type = entity_match.group(1).lower() if entity_match else None
            entity_id = entity_match.group(2).strip() if entity_match else None
            
            # Extract current value with better handling for multiline values
            current_value_match = re.search(r'Current Value:\s+(.*?)(?=\n- Change:|$)', content, re.DOTALL)
            current_value_text = current_value_match.group(1).strip() if current_value_match else None
            
            # Extract the change with improved pattern
            change_match = re.search(r'Change:\s+(.*?)(?=\n- Expected Impact:|$)', content, re.DOTALL)
            change = change_match.group(1).strip() if change_match else None
            
            # Extract expected impact
            impact_match = re.search(r'Expected Impact:\s+(.*?)(?=\n- Rationale:|$)', content, re.DOTALL)
            expected_impact = impact_match.group(1).strip() if impact_match else None
            
            # Extract the rationale with better multiline handling
            rationale_match = re.search(r'Rationale:\s+(.*?)(?=$)', content, re.DOTALL)
            rationale = rationale_match.group(1).strip() if rationale_match else None
            
            # Extract full title with improved pattern
            title_match = re.search(r'\[([A-Z_]+)\]\s+(.*?)(?=\n|$)', content)
            title = title_match.group(2).strip() if title_match else content.split('\n')[0].strip()
            
            # Create structured suggestion with enhanced fields
            suggestion = {
                'index': int(index),
                'title': title,
                'action_type': action_type,
                'entity_type': entity_type,
                'entity_id': entity_id,
                'current_value_text': current_value_text,
                'change': change,
                'expected_impact': expected_impact,
                'rationale': rationale,
                'original_text': content,
                'edited': False,
                'applied': False,
                'status': 'pending',
                'result_message': '',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Add additional fields based on entity type with improved matching
            if entity_type == 'keyword' and keywords:
                # More robust keyword matching
                matching_keywords = []
                for keyword in keywords:
                    # Check several potential matches
                    keyword_text_match = keyword.get('keyword_text', '').lower() == entity_id.lower() if entity_id else False
                    keyword_id_match = keyword.get('id', '') == entity_id if entity_id else False
                    contains_match = entity_id and keyword.get('keyword_text', '').lower() in entity_id.lower()
                    
                    if keyword_text_match or keyword_id_match or contains_match:
                        matching_keywords.append(keyword)
                
                if matching_keywords:
                    # Use the first match if multiple are found
                    matched_keyword = matching_keywords[0]
                    suggestion['current_value'] = matched_keyword.get('current_bid') if action_type == 'BID_ADJUSTMENT' else matched_keyword.get('status')
                    suggestion['keyword_data'] = {
                        'text': matched_keyword.get('keyword_text'),
                        'match_type': matched_keyword.get('match_type'),
                        'clicks': matched_keyword.get('clicks'),
                        'impressions': matched_keyword.get('impressions'),
                        'ctr': matched_keyword.get('ctr'),
                        'conversions': matched_keyword.get('conversions'),
                        'conversion_rate': matched_keyword.get('conversion_rate'),
                        'cost': matched_keyword.get('cost'),
                        'quality_score': matched_keyword.get('quality_score')
                    }
            
            elif entity_type == 'campaign' and campaigns:
                # More robust campaign matching
                for campaign in campaigns:
                    # Check for campaign id or name match
                    campaign_id_match = str(campaign.get('id', '')) == entity_id if entity_id else False
                    campaign_name_match = campaign.get('name', '').lower() == entity_id.lower() if entity_id else False
                    contains_name_match = entity_id and campaign.get('name', '').lower() in entity_id.lower()
                    
                    if campaign_id_match or campaign_name_match or contains_name_match:
                        suggestion['current_value'] = campaign.get('budget') if action_type == 'BUDGET_ADJUSTMENT' else campaign.get('status')
                        suggestion['campaign_data'] = {
                            'name': campaign.get('name'),
                            'id': campaign.get('id'),
                            'clicks': campaign.get('clicks'),
                            'impressions': campaign.get('impressions'),
                            'ctr': campaign.get('ctr'),
                            'conversions': campaign.get('conversions'),
                            'conversion_rate': campaign.get('conversion_rate'),
                            'cost': campaign.get('cost')
                        }
                        break
            
            # Parse specific change values with improved extraction
            if action_type == 'BID_ADJUSTMENT' and change:
                # Extract numeric values for bids using regex
                if 'increase' in change.lower():
                    # Check for percentage increase
                    pct_match = re.search(r'(\d+(?:\.\d+)?)%', change)
                    if pct_match:
                        percentage = float(pct_match.group(1))
                        suggestion['change_value'] = {
                            'type': 'percentage_increase',
                            'value': percentage
                        }
                    
                    # Check for absolute values (from X to Y)
                    abs_match = re.search(r'from\s+\$(\d+(?:\.\d+)?)\s+to\s+\$(\d+(?:\.\d+)?)', change)
                    if abs_match:
                        from_value = float(abs_match.group(1))
                        to_value = float(abs_match.group(2))
                        suggestion['change_value'] = {
                            'type': 'absolute',
                            'from': from_value,
                            'to': to_value,
                            'value': to_value
                        }
                
                elif 'decrease' in change.lower():
                    # Check for percentage decrease
                    pct_match = re.search(r'(\d+(?:\.\d+)?)%', change)
                    if pct_match:
                        percentage = float(pct_match.group(1))
                        suggestion['change_value'] = {
                            'type': 'percentage_decrease',
                            'value': percentage
                        }
                    
                    # Check for absolute values (from X to Y)
                    abs_match = re.search(r'from\s+\$(\d+(?:\.\d+)?)\s+to\s+\$(\d+(?:\.\d+)?)', change)
                    if abs_match:
                        from_value = float(abs_match.group(1))
                        to_value = float(abs_match.group(2))
                        suggestion['change_value'] = {
                            'type': 'absolute',
                            'from': from_value,
                            'to': to_value,
                            'value': to_value
                        }
                
                elif '$' in change:
                    # Extract dollar amounts
                    amounts = re.findall(r'\$(\d+(?:\.\d+)?)', change)
                    if len(amounts) >= 2:
                        # If we have at least 2 dollar amounts, assume from/to format
                        from_value = float(amounts[0])
                        to_value = float(amounts[1])
                        suggestion['change_value'] = {
                            'type': 'absolute',
                            'from': from_value,
                            'to': to_value,
                            'value': to_value
                        }
                    elif len(amounts) == 1:
                        # If only one amount, use as target value
                        amount = float(amounts[0])
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
            
            elif action_type == 'BUDGET_ADJUSTMENT' and change:
                # Extract dollar amounts for budget changes
                amounts = re.findall(r'\$(\d+(?:\.\d+)?)', change)
                if len(amounts) >= 2:
                    # If we have at least 2 dollar amounts, assume from/to format
                    from_value = float(amounts[0])
                    to_value = float(amounts[1])
                    suggestion['change_value'] = {
                        'type': 'absolute',
                        'from': from_value,
                        'to': to_value,
                        'value': to_value
                    }
                elif len(amounts) == 1:
                    # If only one amount, use as target value
                    amount = float(amounts[0])
                    suggestion['change_value'] = {
                        'type': 'absolute',
                        'value': amount
                    }
                
                # Also check for percentage changes
                pct_match = re.search(r'(\d+(?:\.\d+)?)%', change)
                if pct_match and 'increase' in change.lower():
                    percentage = float(pct_match.group(1))
                    suggestion['change_value'] = {
                        'type': 'percentage_increase',
                        'value': percentage
                    }
                elif pct_match and 'decrease' in change.lower():
                    percentage = float(pct_match.group(1))
                    suggestion['change_value'] = {
                        'type': 'percentage_decrease',
                        'value': percentage
                    }
            
            suggestions.append(suggestion)
        
        if suggestion_count == 0 and suggestions_text:
            # If we couldn't parse any suggestions but have text, create a single raw suggestion
            suggestions.append({
                'index': 1,
                'title': 'Optimization Suggestions',
                'action_type': 'INFO',
                'entity_type': 'account',
                'entity_id': 'all',
                'change': 'Review optimization recommendations',
                'rationale': 'Based on account performance',
                'original_text': suggestions_text,
                'edited': False,
                'applied': False,
                'status': 'pending',
                'result_message': '',
                'type': 'raw'
            })
        
        return suggestions 