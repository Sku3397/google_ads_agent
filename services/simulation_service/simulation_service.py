import logging
from typing import Dict, List, Any, Optional, Tuple
from ..base_service import BaseService
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException

logger = logging.getLogger(__name__)

class SimulationService(BaseService):
    """Service for simulating changes to Google Ads campaigns and predicting their impact."""
    
    def __init__(self, client: GoogleAdsClient, customer_id: str):
        """
        Initialize the simulation service.
        
        Args:
            client: The Google Ads API client
            customer_id: The Google Ads customer ID
        """
        super().__init__(client, customer_id)
        self.simulation_service = self.client.get_service("AdGroupCriterionSimulationService")
        
    def simulate_bid_changes(self, keyword_ids: List[str], bid_modifiers: List[float]) -> Dict[str, Any]:
        """
        Simulate the impact of bid changes on specified keywords.
        
        Args:
            keyword_ids: List of keyword criterion IDs to simulate changes for
            bid_modifiers: List of bid modifiers to simulate (e.g. [1.1, 1.2] for 10% and 20% increases)
            
        Returns:
            Dictionary containing simulation results for each keyword and bid modifier
        """
        if len(keyword_ids) != len(bid_modifiers):
            raise ValueError("Number of keywords must match number of bid modifiers")
            
        results = {}
        
        try:
            for keyword_id, bid_modifier in zip(keyword_ids, bid_modifiers):
                # Get current keyword data
                keyword_data = self._get_keyword_data(keyword_id)
                if not keyword_data:
                    logger.warning(f"Could not find data for keyword {keyword_id}")
                    continue
                    
                current_bid = keyword_data.get("current_bid", 0)
                new_bid = current_bid * bid_modifier
                
                # Get simulation data
                simulation = self._get_bid_simulation(keyword_id, new_bid)
                
                if simulation:
                    results[keyword_id] = {
                        "current_bid": current_bid,
                        "simulated_bid": new_bid,
                        "estimated_metrics": simulation
                    }
                
            return results
            
        except GoogleAdsException as ex:
            error_message = f"Google Ads API error: Request with ID '{ex.request_id}' failed with status '{ex.error.code().name}'"
            if ex.failure:
                error_message += f": {ex.failure.errors[0].message}"
            logger.error(error_message)
            raise Exception(error_message)
            
        except Exception as e:
            error_message = f"Error simulating bid changes: {str(e)}"
            logger.error(error_message)
            raise Exception(error_message)
    
    def _get_keyword_data(self, criterion_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current data for a keyword criterion.
        
        Args:
            criterion_id: The criterion ID of the keyword
            
        Returns:
            Dictionary with keyword data or None if not found
        """
        try:
            ga_service = self.client.get_service("GoogleAdsService")
            
            query = f"""
                SELECT
                    ad_group_criterion.criterion_id,
                    ad_group_criterion.keyword.text,
                    ad_group_criterion.effective_cpc_bid_micros,
                    metrics.clicks,
                    metrics.impressions,
                    metrics.cost_micros
                FROM keyword_view
                WHERE ad_group_criterion.criterion_id = {criterion_id}
                LIMIT 1
            """
            
            response = ga_service.search(customer_id=self.customer_id, query=query)
            
            for row in response:
                return {
                    "criterion_id": row.ad_group_criterion.criterion_id,
                    "keyword_text": row.ad_group_criterion.keyword.text,
                    "current_bid": row.ad_group_criterion.effective_cpc_bid_micros / 1000000,
                    "clicks": row.metrics.clicks,
                    "impressions": row.metrics.impressions,
                    "cost": row.metrics.cost_micros / 1000000
                }
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting keyword data: {str(e)}")
            return None
    
    def _get_bid_simulation(self, criterion_id: str, target_bid: float) -> Optional[Dict[str, Any]]:
        """
        Get bid simulation data for a keyword.
        
        Args:
            criterion_id: The criterion ID of the keyword
            target_bid: The target bid to simulate
            
        Returns:
            Dictionary with simulation metrics or None if simulation not available
        """
        try:
            # Convert target bid to micros
            target_bid_micros = int(target_bid * 1000000)
            
            # Get simulation data
            query = f"""
                SELECT
                    ad_group_criterion_simulation.criterion_id,
                    ad_group_criterion_simulation.start_date,
                    ad_group_criterion_simulation.end_date,
                    ad_group_criterion_simulation.cpc_bid_point_list.points
                FROM ad_group_criterion_simulation
                WHERE ad_group_criterion_simulation.criterion_id = {criterion_id}
                AND ad_group_criterion_simulation.type = CPC_BID
                ORDER BY ad_group_criterion_simulation.start_date DESC
                LIMIT 1
            """
            
            ga_service = self.client.get_service("GoogleAdsService")
            response = ga_service.search(customer_id=self.customer_id, query=query)
            
            for row in response:
                simulation_points = row.ad_group_criterion_simulation.cpc_bid_point_list.points
                
                # Find closest bid point to our target
                closest_point = min(simulation_points, 
                                  key=lambda p: abs(p.cpc_bid_micros - target_bid_micros))
                
                return {
                    "estimated_clicks": closest_point.clicks,
                    "estimated_impressions": closest_point.impressions,
                    "estimated_cost": closest_point.cost_micros / 1000000,
                    "estimated_conversions": closest_point.conversions,
                    "simulation_date_range": {
                        "start_date": row.ad_group_criterion_simulation.start_date,
                        "end_date": row.ad_group_criterion_simulation.end_date
                    }
                }
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting bid simulation: {str(e)}")
            return None
            
    def get_performance_forecast(self, campaign_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get a performance forecast for a campaign.
        
        Args:
            campaign_id: The campaign ID to forecast
            days: Number of days to forecast (default 30)
            
        Returns:
            Dictionary with forecast metrics
        """
        try:
            # Get historical performance
            query = f"""
                SELECT
                    campaign.id,
                    metrics.clicks,
                    metrics.impressions,
                    metrics.cost_micros,
                    metrics.conversions,
                    segments.date
                FROM campaign
                WHERE campaign.id = {campaign_id}
                AND segments.date DURING LAST_30_DAYS
                ORDER BY segments.date DESC
            """
            
            ga_service = self.client.get_service("GoogleAdsService")
            response = ga_service.search(customer_id=self.customer_id, query=query)
            
            # Process historical data
            historical_data = []
            for row in response:
                historical_data.append({
                    "date": row.segments.date,
                    "clicks": row.metrics.clicks,
                    "impressions": row.metrics.impressions,
                    "cost": row.metrics.cost_micros / 1000000,
                    "conversions": row.metrics.conversions
                })
            
            # Calculate averages for forecast
            if historical_data:
                avg_daily_clicks = sum(d["clicks"] for d in historical_data) / len(historical_data)
                avg_daily_impressions = sum(d["impressions"] for d in historical_data) / len(historical_data)
                avg_daily_cost = sum(d["cost"] for d in historical_data) / len(historical_data)
                avg_daily_conversions = sum(d["conversions"] for d in historical_data) / len(historical_data)
                
                # Project forward
                forecast = {
                    "forecast_days": days,
                    "estimated_metrics": {
                        "clicks": int(avg_daily_clicks * days),
                        "impressions": int(avg_daily_impressions * days),
                        "cost": round(avg_daily_cost * days, 2),
                        "conversions": round(avg_daily_conversions * days, 2)
                    },
                    "daily_averages": {
                        "clicks": round(avg_daily_clicks, 2),
                        "impressions": int(avg_daily_impressions),
                        "cost": round(avg_daily_cost, 2),
                        "conversions": round(avg_daily_conversions, 2)
                    }
                }
                
                return forecast
            else:
                return {
                    "error": "No historical data available for forecasting"
                }
                
        except GoogleAdsException as ex:
            error_message = f"Google Ads API error: Request with ID '{ex.request_id}' failed with status '{ex.error.code().name}'"
            if ex.failure:
                error_message += f": {ex.failure.errors[0].message}"
            logger.error(error_message)
            raise Exception(error_message)
            
        except Exception as e:
            error_message = f"Error generating forecast: {str(e)}"
            logger.error(error_message)
            raise Exception(error_message) 