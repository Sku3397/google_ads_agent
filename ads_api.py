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
        Fetch minimal campaign performance data from Google Ads.
        
        Args:
            days_ago (int): Number of days to look back for data (1-365)
            
        Returns:
            list: List of campaign data dictionaries (minimal fields)
        """
        # Validate input
        if not isinstance(days_ago, int) or days_ago < 1 or days_ago > 365:
            raise ValueError("days_ago must be an integer between 1 and 365")
        
        ga_service = self.client.get_service("GoogleAdsService")
        
        # Use date range clause instead of LAST_X_DAYS
        date_range_clause = self._get_date_range_clause(days_ago)
        
        # Enhanced query with metrics - ONLY ENABLED campaigns
        query = f"""
            SELECT
                campaign.id,
                campaign.name,
                campaign.status,
                metrics.impressions,
                metrics.clicks,
                metrics.conversions,
                metrics.cost_micros,
                metrics.average_cpc
            FROM campaign
            WHERE {date_range_clause} AND campaign.status = 'ENABLED'
            ORDER BY campaign.name
        """
        
        try:
            # Issue the search request
            logging.info(f"Fetching campaign data (ENABLED only) for the last {days_ago} days")
            response = ga_service.search(customer_id=self.customer_id, query=query)
            
            # Process and return the results
            campaigns = []
            for row in response:
                campaign = {
                    'id': row.campaign.id,
                    'name': row.campaign.name,
                    'status': row.campaign.status.name,
                    'impressions': row.metrics.impressions if hasattr(row.metrics, 'impressions') else 0,
                    'clicks': row.metrics.clicks if hasattr(row.metrics, 'clicks') else 0,
                    'conversions': row.metrics.conversions if hasattr(row.metrics, 'conversions') else 0,
                    'cost': row.metrics.cost_micros / 1000000 if hasattr(row.metrics, 'cost_micros') else 0,
                    'average_cpc': row.metrics.average_cpc / 1000000 if hasattr(row.metrics, 'average_cpc') and row.metrics.average_cpc else 0,
                    # Store the days parameter for reference if needed elsewhere
                    'days': days_ago
                }
                campaigns.append(campaign)
            
            logging.info(f"Found {len(campaigns)} ENABLED campaigns")
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
        Fetch keyword performance data from Google Ads (reduced fields).
        
        Args:
            days_ago (int): Number of days to look back for data (1-365)
            campaign_id (str, optional): Filter by specific campaign ID
            
        Returns:
            list: List of keyword data dictionaries (reduced fields)
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
        
        # Ensure we only get ENABLED keywords from ENABLED ad groups and ENABLED campaigns
        # Using exact string comparison with proper quotes
        where_clause += " AND campaign.status = 'ENABLED'"
        where_clause += " AND ad_group.status = 'ENABLED'"
        where_clause += " AND ad_group_criterion.status = 'ENABLED'"
        
        logging.info(f"Using keyword filter: {where_clause}")
        
        # Query with reduced, specified fields
        query = f"""
            SELECT
                campaign.id,
                campaign.name,
                campaign.status,
                ad_group.id,
                ad_group.name,
                ad_group.status,
                ad_group_criterion.keyword.text,
                ad_group_criterion.keyword.match_type,
                ad_group_criterion.status,
                ad_group_criterion.system_serving_status,
                ad_group_criterion.quality_info.quality_score,
                ad_group_criterion.effective_cpc_bid_micros,
                ad_group_criterion.criterion_id,
                ad_group_criterion.resource_name,
                ad_group_criterion.negative,
                metrics.clicks,
                metrics.impressions,
                metrics.average_cpc,
                metrics.conversions,
                metrics.cost_micros,
                metrics.top_impression_percentage,
                metrics.search_impression_share,
                metrics.search_top_impression_share
            FROM keyword_view
            WHERE {where_clause}
            ORDER BY metrics.clicks DESC
            LIMIT 10000
        """
        
        # SAFETY CHECK: Ensure removed metrics are not in the query
        removed_metrics = ["metrics.ctr", "metrics.cost_per_conversion", "metrics.average_position"]
        for metric in removed_metrics:
             if metric in query:
                 logging.warning(f"Found removed metric {metric} in query. Please update the query definition.")
                 # Attempt to remove defensively, though the query string should be correct
                 query = query.replace(f"{metric},", "")
                 query = query.replace(metric, "")
        
        try:
            # Issue the search request with better error handling
            logging.info(f"Fetching reduced keyword data for the last {days_ago} days{' for campaign ' + campaign_id if campaign_id else ''}")
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
                        'campaign_status': row.campaign.status.name if hasattr(row.campaign, 'status') else 'UNKNOWN',
                        'ad_group_id': row.ad_group.id,
                        'ad_group_name': row.ad_group.name,
                        'ad_group_status': row.ad_group.status.name if hasattr(row.ad_group, 'status') else 'UNKNOWN',
                        'keyword_text': row.ad_group_criterion.keyword.text,
                        'match_type': row.ad_group_criterion.keyword.match_type.name if hasattr(row.ad_group_criterion.keyword, 'match_type') else None,
                        'status': row.ad_group_criterion.status.name if hasattr(row.ad_group_criterion, 'status') else None,
                        'system_serving_status': row.ad_group_criterion.system_serving_status.name if hasattr(row.ad_group_criterion, 'system_serving_status') else None,
                        'quality_score': row.ad_group_criterion.quality_info.quality_score if hasattr(row.ad_group_criterion, 'quality_info') and hasattr(row.ad_group_criterion.quality_info, 'quality_score') else 0,
                        'current_bid': row.ad_group_criterion.effective_cpc_bid_micros / 1000000 if hasattr(row.ad_group_criterion, 'effective_cpc_bid_micros') and row.ad_group_criterion.effective_cpc_bid_micros else 0,
                        'criterion_id': row.ad_group_criterion.criterion_id if hasattr(row.ad_group_criterion, 'criterion_id') else None,
                        'resource_name': row.ad_group_criterion.resource_name if hasattr(row.ad_group_criterion, 'resource_name') else None,
                        'is_negative': row.ad_group_criterion.negative if hasattr(row.ad_group_criterion, 'negative') else False,
                        'clicks': row.metrics.clicks if hasattr(row.metrics, 'clicks') else 0,
                        'impressions': row.metrics.impressions if hasattr(row.metrics, 'impressions') else 0,
                        # Note: average_cpc is fetched directly as requested
                        'average_cpc': row.metrics.average_cpc / 1000000 if hasattr(row.metrics, 'average_cpc') and row.metrics.average_cpc else 0,
                        'conversions': row.metrics.conversions,
                        'cost': row.metrics.cost_micros / 1000000 if row.metrics.cost_micros else 0,
                        'top_impression_pct': row.metrics.top_impression_percentage if hasattr(row.metrics, 'top_impression_percentage') else None,
                        'search_impression_share': row.metrics.search_impression_share if hasattr(row.metrics, 'search_impression_share') else None,
                        'search_top_impression_share': row.metrics.search_top_impression_share if hasattr(row.metrics, 'search_top_impression_share') else None,
                        'days': days_ago
                    }
                    keywords.append(keyword)

                except AttributeError as ae:
                    logging.warning(f"Attribute error processing keyword row: {ae}. Row data: {row}")
                    continue # Skip this keyword row if essential attributes are missing
            
            logging.info(f"Found {len(keywords)} keywords with data in the specified time period.")
            return keywords
            
        except GoogleAdsException as ex:
            error_message = f"Google Ads API error fetching keywords: Request ID '{ex.request_id}', Status '{ex.error.code().name}'"
            if ex.failure:
                 error_message += f": {ex.failure.errors[0].message}"
            logging.error(error_message)
            raise Exception(error_message)
        except Exception as e:
            error_message = f"Error fetching keyword data: {str(e)}"
            logging.exception(error_message) # Log traceback for unexpected errors
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
            elif entity_type == 'keyword' and optimization_type == 'add':
                return self._add_keyword(
                    changes.get('campaign_id'),
                    changes.get('ad_group_id'),
                    changes.get('keyword_text', ''),
                    changes.get('match_type', 'EXACT'),
                    changes.get('bid_micros', 1000000)  # Default to $1.00
                )
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
        
        # First, check if this is a negative keyword
        # We need to fetch the criterion to check
        try:
            ga_service = self.client.get_service("GoogleAdsService")
            query = f"""
                SELECT 
                    ad_group_criterion.resource_name,
                    ad_group_criterion.negative,
                    ad_group_criterion.keyword.text
                FROM ad_group_criterion
                WHERE ad_group_criterion.resource_name = '{keyword_criterion_id}'
            """
            
            response = ga_service.search(customer_id=self.customer_id, query=query)
            
            # Check if the keyword is negative
            for row in response:
                if hasattr(row.ad_group_criterion, 'negative') and row.ad_group_criterion.negative:
                    keyword_text = row.ad_group_criterion.keyword.text if hasattr(row.ad_group_criterion.keyword, 'text') else "Unknown"
                    return False, f"Cannot adjust bid for negative keyword '{keyword_text}'"
        
        except GoogleAdsException as ex:
            # If we can't determine if it's negative, proceed with the update and let it fail if needed
            logging.warning(f"Unable to check if keyword is negative: {str(ex)}")
            
        except Exception as e:
            logging.warning(f"Error checking if keyword is negative: {str(e)}")
            
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
        
        # Set the update mask for Google Ads API v19 (using strings instead of FieldMask type)
        operation.update_mask.paths.append("cpc_bid_micros")
        
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
        
        # Set the update mask for Google Ads API v19
        operation.update_mask.paths.append("status")
        
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
        
        # Set the update mask for Google Ads API v19
        operation.update_mask.paths.append("status")
        
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
            
            # Set the update mask for Google Ads API v19
            budget_operation.update_mask.paths.append("amount_micros")
            
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
    
    def _add_keyword(self, campaign_id, ad_group_id, keyword_text, match_type="EXACT", bid_micros=1000000):
        """
        Add a new keyword to a specified ad group.
        
        Args:
            campaign_id (str): Campaign ID
            ad_group_id (str): Ad group ID to add the keyword to
            keyword_text (str): Text of the keyword to add
            match_type (str): Match type (EXACT, PHRASE, BROAD)
            bid_micros (int): Bid amount in micros
            
        Returns:
            bool: Success status
            str: Status message
        """
        # Input validation
        if not campaign_id or not isinstance(campaign_id, str):
            return False, "Invalid campaign ID"
            
        if not keyword_text:
            return False, "Keyword text is required"
        
        try:
            # If ad_group_id is not provided, try to find an appropriate ad group in the campaign
            if not ad_group_id:
                ga_service = self.client.get_service("GoogleAdsService")
                query = f"""
                    SELECT
                        ad_group.id,
                        ad_group.name
                    FROM ad_group
                    WHERE campaign.id = {campaign_id} AND ad_group.status = 'ENABLED'
                    LIMIT 1
                """
                
                response = ga_service.search(customer_id=self.customer_id, query=query)
                
                # Get the first ad group from the response
                ad_group = next(iter(response), None)
                if not ad_group:
                    return False, f"No enabled ad groups found in campaign {campaign_id}"
                    
                ad_group_id = ad_group.ad_group.id
                logging.info(f"Using ad group '{ad_group.ad_group.name}' (ID: {ad_group_id}) for new keyword")
            
            # Create the ad group criterion with the keyword
            ad_group_criterion_service = self.client.get_service("AdGroupCriterionService")
            ad_group_criterion_operation = self.client.get_type("AdGroupCriterionOperation")
            
            # Create the ad group criterion (keyword)
            ad_group_criterion = ad_group_criterion_operation.create
            ad_group_criterion.ad_group = f"customers/{self.customer_id}/adGroups/{ad_group_id}"
            
            # Set the keyword text and match type
            ad_group_criterion.keyword.text = keyword_text
            
            # Set match type
            match_type_enum = self.client.enums.KeywordMatchTypeEnum
            if match_type == "EXACT":
                ad_group_criterion.keyword.match_type = match_type_enum.EXACT
            elif match_type == "PHRASE":
                ad_group_criterion.keyword.match_type = match_type_enum.PHRASE
            elif match_type == "BROAD":
                ad_group_criterion.keyword.match_type = match_type_enum.BROAD
            else:
                return False, f"Invalid match type: {match_type}. Must be EXACT, PHRASE, or BROAD."
            
            # Set the bid
            ad_group_criterion.cpc_bid_micros = bid_micros
            
            # Add the operation to a list of operations
            operations = [ad_group_criterion_operation]
            
            # Submit the operations
            response = ad_group_criterion_service.mutate_ad_group_criteria(
                customer_id=self.customer_id, operations=operations
            )
            
            # Extract the resource name from the response
            new_criterion_resource_name = response.results[0].resource_name
            
            return True, f"Successfully added keyword '{keyword_text}' with {match_type} match type and ${bid_micros/1000000:.2f} bid to ad group {ad_group_id}"
            
        except GoogleAdsException as ex:
            error_message = f"Failed to add keyword: {ex.failure.errors[0].message}"
            logging.error(error_message)
            return False, error_message
        except Exception as e:
            error_message = f"Error adding keyword: {str(e)}"
            logging.error(error_message)
            return False, error_message 