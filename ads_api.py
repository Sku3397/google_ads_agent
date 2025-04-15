from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
import pandas as pd
from datetime import datetime, timedelta
import logging

class GoogleAdsAPI:
    def __init__(self, config):
        """
        Initialize Google Ads API client.
        
        Args:
            config (dict): Configuration dictionary with Google Ads credentials
        """
        self.client = GoogleAdsClient.load_from_dict({
            'client_id': config['client_id'],
            'client_secret': config['client_secret'],
            'developer_token': config['developer_token'],
            'refresh_token': config['refresh_token'],
            'login_customer_id': config['login_customer_id'],
            'use_proto_plus': True,
        })
        self.customer_id = config['customer_id']
    
    def _get_date_range_clause(self, days_ago):
        """
        Generate a date range clause for Google Ads API queries.
        
        Args:
            days_ago (int): Number of days to look back (1-365)
            
        Returns:
            str: Date range clause for Google Ads API
        """
        # Enhanced validation with clear error message
        if not isinstance(days_ago, int):
            raise ValueError(f"days_ago must be an integer, got {type(days_ago).__name__}")
        
        if days_ago < 1:
            raise ValueError(f"days_ago must be at least 1, got {days_ago}")
            
        if days_ago > 365:
            raise ValueError(f"days_ago must be at most 365, got {days_ago}")
        
        # Calculate date range with proper handling
        end_date = datetime.now().date() - timedelta(days=1)  # Yesterday to account for data lag
        start_date = end_date - timedelta(days=days_ago-1)  # -1 to include end_date in range
        
        # Format dates as YYYY-MM-DD
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Log the date range being used
        logging.info(f"Using date range from {start_str} to {end_str} ({days_ago} days)")
        
        # Return the date range clause in the format Google Ads API expects
        return f"segments.date BETWEEN '{start_str}' AND '{end_str}'"
    
    def get_campaign_performance(self, days_ago=30):
        """
        Fetch campaign performance data from Google Ads.
        
        Args:
            days_ago (int): Number of days to look back for data (1-365)
            
        Returns:
            list: List of campaign data dictionaries
        """
        # Validate input
        if not isinstance(days_ago, int) or days_ago < 1 or days_ago > 365:
            raise ValueError("days_ago must be an integer between 1 and 365")
        
        ga_service = self.client.get_service("GoogleAdsService")
        
        # Use date range clause instead of LAST_X_DAYS
        date_range_clause = self._get_date_range_clause(days_ago)
        
        query = f"""
            SELECT
                campaign.id,
                campaign.name,
                campaign.status,
                campaign.advertising_channel_type,
                campaign.bidding_strategy_type,
                campaign_budget.amount_micros,
                metrics.clicks,
                metrics.impressions,
                metrics.ctr,
                metrics.average_cpc,
                metrics.conversions,
                metrics.cost_micros,
                metrics.cost_per_conversion
            FROM campaign
            WHERE {date_range_clause}
            ORDER BY metrics.clicks DESC
        """
        
        try:
            # Issue the search request
            response = ga_service.search(customer_id=self.customer_id, query=query)
            
            # Process and return the results
            campaigns = []
            for row in response:
                campaign = {
                    'id': row.campaign.id,
                    'name': row.campaign.name,
                    'status': row.campaign.status.name,
                    'channel_type': row.campaign.advertising_channel_type.name if hasattr(row.campaign, 'advertising_channel_type') else None,
                    'bidding_strategy': row.campaign.bidding_strategy_type.name if hasattr(row.campaign, 'bidding_strategy_type') else None,
                    'budget': row.campaign_budget.amount_micros / 1000000 if hasattr(row, 'campaign_budget') and row.campaign_budget.amount_micros else None,
                    'clicks': row.metrics.clicks,
                    'impressions': row.metrics.impressions,
                    'ctr': row.metrics.ctr * 100 if row.metrics.ctr else 0,  # Convert to percentage
                    'average_cpc': row.metrics.average_cpc / 1000000 if row.metrics.average_cpc else 0,  # Convert micros to actual currency
                    'conversions': row.metrics.conversions,
                    'cost': row.metrics.cost_micros / 1000000 if row.metrics.cost_micros else 0,  # Convert micros to actual currency
                    'conversion_rate': (row.metrics.conversions / row.metrics.clicks * 100) if row.metrics.clicks else 0,  # Calculate manually and convert to percentage
                    'cost_per_conversion': row.metrics.cost_per_conversion / 1000000 if row.metrics.cost_per_conversion else 0,  # Convert micros to actual currency
                    'days': days_ago  # Store the days parameter for reference
                }
                campaigns.append(campaign)
            
            return campaigns
            
        except GoogleAdsException as ex:
            error_message = f"Google Ads API error: Request with ID '{ex.request_id}' failed with status '{ex.error.code().name}'"
            if ex.failure:
                error_message += f": {ex.failure.errors[0].message}"
            logging.error(error_message)
            raise Exception(error_message)
        except Exception as e:
            error_message = f"Error fetching campaign data: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)
    
    def get_keyword_performance(self, days_ago=30, campaign_id=None):
        """
        Fetch keyword performance data from Google Ads.
        
        Args:
            days_ago (int): Number of days to look back for data (1-365)
            campaign_id (str, optional): Filter by specific campaign ID
            
        Returns:
            list: List of keyword data dictionaries
        """
        # Validate input with improved error messages
        if not isinstance(days_ago, int):
            raise ValueError(f"days_ago must be an integer, got {type(days_ago).__name__}")
        
        if days_ago < 1:
            raise ValueError(f"days_ago must be at least 1, got {days_ago}")
            
        if days_ago > 365:
            raise ValueError(f"days_ago must be at most 365, got {days_ago}")
        
        ga_service = self.client.get_service("GoogleAdsService")
        
        # Use date range clause instead of LAST_X_DAYS
        date_range_clause = self._get_date_range_clause(days_ago)
        
        # Construct WHERE clause with improved campaign filtering
        where_clause = date_range_clause
        if campaign_id:
            # Clean and validate campaign_id
            campaign_id = str(campaign_id).strip()
            if not campaign_id.isdigit():
                raise ValueError(f"campaign_id must contain only digits, got '{campaign_id}'")
            where_clause += f" AND campaign.id = {campaign_id}"
        
        # Add serving status check to ensure we only get active keywords
        where_clause += " AND ad_group_criterion.status != 'REMOVED'"
        
        # Enhanced query with more relevant metrics for optimization
        query = f"""
            SELECT
                campaign.id,
                campaign.name,
                ad_group.id,
                ad_group.name,
                ad_group_criterion.keyword.text,
                ad_group_criterion.keyword.match_type,
                ad_group_criterion.status,
                ad_group_criterion.system_serving_status,
                ad_group_criterion.quality_info.quality_score,
                ad_group_criterion.effective_cpc_bid_micros,
                metrics.clicks,
                metrics.impressions,
                metrics.ctr,
                metrics.average_cpc,
                metrics.conversions,
                metrics.cost_micros,
                metrics.cost_per_conversion,
                metrics.top_impression_percentage,
                metrics.search_impression_share,
                metrics.search_top_impression_share
            FROM keyword_view
            WHERE {where_clause}
            ORDER BY metrics.clicks DESC
            LIMIT 10000  # Increased limit to ensure we get all relevant keywords
        """
        
        try:
            # Issue the search request with better error handling
            logging.info(f"Fetching keyword data for the last {days_ago} days{' for campaign ' + campaign_id if campaign_id else ''}")
            response = ga_service.search(customer_id=self.customer_id, query=query)
            
            # Process and return the results with enhanced error handling
            keywords = []
            for row in response:
                try:
                    # Skip if no keyword text (shouldn't happen but good to check)
                    if not hasattr(row.ad_group_criterion.keyword, "text"):
                        continue
                    
                    keyword = {
                        'campaign_id': row.campaign.id,
                        'campaign_name': row.campaign.name,
                        'ad_group_id': row.ad_group.id,
                        'ad_group_name': row.ad_group.name,
                        'keyword_text': row.ad_group_criterion.keyword.text,
                        'match_type': row.ad_group_criterion.keyword.match_type.name if hasattr(row.ad_group_criterion.keyword, 'match_type') else None,
                        'status': row.ad_group_criterion.status.name if hasattr(row.ad_group_criterion, 'status') else None,
                        'system_serving_status': row.ad_group_criterion.system_serving_status.name if hasattr(row.ad_group_criterion, 'system_serving_status') else None,
                        'quality_score': row.ad_group_criterion.quality_info.quality_score if hasattr(row.ad_group_criterion, 'quality_info') and hasattr(row.ad_group_criterion.quality_info, 'quality_score') else 0,
                        'current_bid': row.ad_group_criterion.effective_cpc_bid_micros / 1000000 if hasattr(row.ad_group_criterion, 'effective_cpc_bid_micros') and row.ad_group_criterion.effective_cpc_bid_micros else 0,
                        'clicks': row.metrics.clicks if hasattr(row.metrics, 'clicks') else 0,
                        'impressions': row.metrics.impressions if hasattr(row.metrics, 'impressions') else 0,
                        'ctr': row.metrics.ctr * 100 if hasattr(row.metrics, 'ctr') and row.metrics.ctr else 0,  # Convert to percentage
                        'average_cpc': row.metrics.average_cpc / 1000000 if hasattr(row.metrics, 'average_cpc') and row.metrics.average_cpc else 0,  # Convert micros to actual currency
                        'conversions': row.metrics.conversions,
                        'cost': row.metrics.cost_micros / 1000000 if row.metrics.cost_micros else 0,  # Convert micros to actual currency
                        'conversion_rate': (row.metrics.conversions / row.metrics.clicks * 100) if row.metrics.clicks else 0,  # Calculate manually and convert to percentage
                        'cost_per_conversion': row.metrics.cost_per_conversion / 1000000 if row.metrics.cost_per_conversion else 0,  # Convert micros to actual currency
                        'top_impression_pct': row.metrics.top_impression_percentage if hasattr(row.metrics, 'top_impression_percentage') else None,
                        'search_impression_share': row.metrics.search_impression_share if hasattr(row.metrics, 'search_impression_share') else None,
                        'search_top_impression_share': row.metrics.search_top_impression_share if hasattr(row.metrics, 'search_top_impression_share') else None,
                        'days': days_ago  # Store the days parameter for reference
                    }
                    
                    # Calculate ROAS if possible
                    if hasattr(row.metrics, 'conversions') and row.metrics.conversions > 0 and hasattr(row.metrics, 'cost_micros') and row.metrics.cost_micros > 0:
                        # Assuming each conversion is worth $50 (this should be configurable in a real implementation)
                        conversion_value = row.metrics.conversions * 50
                        roas = (conversion_value / (row.metrics.cost_micros / 1000000)) * 100  # As percentage
                        keyword['roas'] = roas
                    else:
                        keyword['roas'] = 0
                    
                    # Create a unique ID for the keyword
                    keyword['id'] = f"adgroups/{row.ad_group.id}/criteria/{row.ad_group_criterion.keyword.text}"
                    
                    keywords.append(keyword)
                except Exception as row_error:
                    # Log error but continue processing other rows
                    logging.error(f"Error processing keyword row: {str(row_error)}")
            
            logging.info(f"Fetched {len(keywords)} keywords{' for campaign ' + campaign_id if campaign_id else ''}")
            return keywords
            
        except GoogleAdsException as ex:
            error_message = f"Google Ads API error: Request with ID '{ex.request_id}' failed with status '{ex.error.code().name}'"
            if ex.failure:
                error_message += f": {ex.failure.errors[0].message}"
            logging.error(error_message)
            raise Exception(error_message)
        except Exception as e:
            error_message = f"Error fetching keyword data: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)
    
    def apply_optimization(self, optimization_type, entity_type, entity_id, changes):
        """
        Apply an optimization to Google Ads.
        
        Args:
            optimization_type (str): Type of optimization (e.g., 'bid_adjustment', 'status_change')
            entity_type (str): Type of entity to optimize (e.g., 'keyword', 'campaign', 'ad_group')
            entity_id (str): ID of the entity to optimize
            changes (dict): Dictionary of changes to apply
        
        Returns:
            bool: Success status
            str: Status message
        """
        try:
            if entity_type == 'keyword' and optimization_type == 'bid_adjustment':
                return self._adjust_keyword_bid(entity_id, changes.get('bid_micros', 0))
            elif entity_type == 'keyword' and optimization_type == 'status_change':
                return self._update_keyword_status(entity_id, changes.get('status', 'ENABLED'))
            elif entity_type == 'campaign' and optimization_type == 'status_change':
                return self._update_campaign_status(entity_id, changes.get('status', 'ENABLED'))
            elif entity_type == 'campaign' and optimization_type == 'budget_adjustment':
                return self._update_campaign_budget(entity_id, changes.get('budget_micros', 0))
            else:
                return False, f"Unsupported optimization: {optimization_type} for {entity_type}"
        
        except Exception as e:
            error_message = f"Error applying optimization: {str(e)}"
            logging.error(error_message)
            return False, error_message
    
    def _adjust_keyword_bid(self, keyword_criterion_id, bid_micros):
        """
        Adjust the bid for a keyword.
        
        Args:
            keyword_criterion_id (str): ID of the ad group criterion (keyword)
            bid_micros (int): New bid in micros
            
        Returns:
            bool: Success status
            str: Status message
        """
        # Get the services
        ad_group_criterion_service = self.client.get_service("AdGroupCriterionService")
        
        # Extract ad group ID from the criterion ID (format: adGroups/123/criteria/456)
        parts = keyword_criterion_id.split('/')
        if len(parts) != 4:
            return False, f"Invalid criterion ID format: {keyword_criterion_id}"
        
        ad_group_id = parts[1]
        criterion_id = parts[3]
        
        # Create the operation
        operation = self.client.get_type("AdGroupCriterionOperation")
        criterion = operation.update
        criterion.resource_name = keyword_criterion_id
        criterion.cpc_bid_micros = bid_micros
        
        # Set the update mask
        field_mask = self.client.get_type("FieldMask")
        field_mask.paths.append("cpc_bid_micros")
        operation.update_mask = field_mask
        
        try:
            # Submit the operation
            response = ad_group_criterion_service.mutate_ad_group_criteria(
                customer_id=self.customer_id, operations=[operation]
            )
            
            return True, f"Successfully updated bid for keyword {criterion_id} to ${bid_micros/1000000:.2f}"
            
        except GoogleAdsException as ex:
            error_message = f"Failed to update keyword bid: {ex.failure.errors[0].message}"
            logging.error(error_message)
            return False, error_message
    
    def _update_keyword_status(self, keyword_criterion_id, status):
        """
        Update the status of a keyword.
        
        Args:
            keyword_criterion_id (str): ID of the ad group criterion (keyword)
            status (str): New status (ENABLED, PAUSED, REMOVED)
            
        Returns:
            bool: Success status
            str: Status message
        """
        # Get the services
        ad_group_criterion_service = self.client.get_service("AdGroupCriterionService")
        
        # Create the operation
        operation = self.client.get_type("AdGroupCriterionOperation")
        criterion = operation.update
        criterion.resource_name = keyword_criterion_id
        
        # Set the status
        status_enum = self.client.enums.AdGroupCriterionStatusEnum
        if status == "ENABLED":
            criterion.status = status_enum.ENABLED
        elif status == "PAUSED":
            criterion.status = status_enum.PAUSED
        elif status == "REMOVED":
            criterion.status = status_enum.REMOVED
        else:
            return False, f"Invalid status: {status}. Must be ENABLED, PAUSED, or REMOVED."
        
        # Set the update mask
        field_mask = self.client.get_type("FieldMask")
        field_mask.paths.append("status")
        operation.update_mask = field_mask
        
        try:
            # Submit the operation
            response = ad_group_criterion_service.mutate_ad_group_criteria(
                customer_id=self.customer_id, operations=[operation]
            )
            
            return True, f"Successfully updated keyword status to {status}"
            
        except GoogleAdsException as ex:
            error_message = f"Failed to update keyword status: {ex.failure.errors[0].message}"
            logging.error(error_message)
            return False, error_message
    
    def _update_campaign_status(self, campaign_id, status):
        """
        Update the status of a campaign.
        
        Args:
            campaign_id (str): ID of the campaign
            status (str): New status (ENABLED, PAUSED, REMOVED)
            
        Returns:
            bool: Success status
            str: Status message
        """
        # Get the services
        campaign_service = self.client.get_service("CampaignService")
        
        # Create the operation
        operation = self.client.get_type("CampaignOperation")
        campaign = operation.update
        campaign.resource_name = f"customers/{self.customer_id}/campaigns/{campaign_id}"
        
        # Set the status
        status_enum = self.client.enums.CampaignStatusEnum
        if status == "ENABLED":
            campaign.status = status_enum.ENABLED
        elif status == "PAUSED":
            campaign.status = status_enum.PAUSED
        elif status == "REMOVED":
            campaign.status = status_enum.REMOVED
        else:
            return False, f"Invalid status: {status}. Must be ENABLED, PAUSED, or REMOVED."
        
        # Set the update mask
        field_mask = self.client.get_type("FieldMask")
        field_mask.paths.append("status")
        operation.update_mask = field_mask
        
        try:
            # Submit the operation
            response = campaign_service.mutate_campaigns(
                customer_id=self.customer_id, operations=[operation]
            )
            
            return True, f"Successfully updated campaign status to {status}"
            
        except GoogleAdsException as ex:
            error_message = f"Failed to update campaign status: {ex.failure.errors[0].message}"
            logging.error(error_message)
            return False, error_message
            
    def _update_campaign_budget(self, campaign_id, budget_micros):
        """
        Update the budget of a campaign.
        
        Args:
            campaign_id (str): ID of the campaign
            budget_micros (int): New budget in micros
            
        Returns:
            bool: Success status
            str: Status message
        """
        # Get the services
        campaign_service = self.client.get_service("CampaignService")
        
        try:
            # First, get the campaign to find its budget resource name
            query = f"""
                SELECT
                    campaign.id,
                    campaign.name,
                    campaign.campaign_budget
                FROM campaign
                WHERE campaign.id = {campaign_id}
            """
            
            response = self.client.get_service("GoogleAdsService").search(
                customer_id=self.customer_id, query=query
            )
            
            # Get the first (and should be only) result
            campaign_data = next(iter(response), None)
            if not campaign_data or not hasattr(campaign_data.campaign, 'campaign_budget'):
                return False, f"Could not find budget for campaign ID: {campaign_id}"
            
            budget_resource_name = campaign_data.campaign.campaign_budget
            
            # Create the budget operation
            budget_operation = self.client.get_type("CampaignBudgetOperation")
            budget = budget_operation.update
            budget.resource_name = budget_resource_name
            budget.amount_micros = budget_micros
            
            # Set the update mask
            field_mask = self.client.get_type("FieldMask")
            field_mask.paths.append("amount_micros")
            budget_operation.update_mask = field_mask
            
            # Submit the operation
            budget_service = self.client.get_service("CampaignBudgetService")
            response = budget_service.mutate_campaign_budgets(
                customer_id=self.customer_id, operations=[budget_operation]
            )
            
            return True, f"Successfully updated campaign budget to ${budget_micros/1000000:.2f}"
            
        except GoogleAdsException as ex:
            error_message = f"Failed to update campaign budget: {ex.failure.errors[0].message}"
            logging.error(error_message)
            return False, error_message
        except StopIteration:
            return False, f"Campaign ID {campaign_id} not found"
        except Exception as e:
            error_message = f"Error updating campaign budget: {str(e)}"
            logging.error(error_message)
            return False, error_message 