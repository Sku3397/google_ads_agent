"""
Bid Service for Google Ads Management System

This module provides bid management and optimization services.
It implements various bidding strategies such as target CPA, target ROAS,
and manages bid adjustments based on performance metrics.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import os
import json

from services.base_service import BaseService
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException

logger = logging.getLogger(__name__)

class BidService(BaseService):
    """
    Bid Service for managing and optimizing keyword and campaign bids.
    """
    
    def __init__(self, client: GoogleAdsClient, customer_id: str):
        """
        Initialize the bid service.
        
        Args:
            client: The Google Ads API client
            customer_id: The Google Ads customer ID
        """
        super().__init__(client, customer_id)
        self.ad_group_criterion_service = self.client.get_service("AdGroupCriterionService")
        
        # Default thresholds and settings
        self.min_data_points = 50             # Minimum data points required for confidence
        self.max_bid_increase_pct = 50        # Maximum bid increase allowed (percent)
        self.max_bid_decrease_pct = 30        # Maximum bid decrease allowed (percent)
        self.min_conversions = 5              # Minimum conversions needed for ROAS/CPA strategies
        self.update_frequency_days = 7        # How often to update bids
        self.target_cpa = None                # Target CPA (cost per acquisition)
        self.target_roas = None               # Target ROAS (return on ad spend)
        
        # Override with config values if available
        if self.config.get('bid', None):
            bid_config = self.config['bid']
            self.min_data_points = bid_config.get('min_data_points', self.min_data_points)
            self.max_bid_increase_pct = bid_config.get('max_bid_increase_pct', self.max_bid_increase_pct)
            self.max_bid_decrease_pct = bid_config.get('max_bid_decrease_pct', self.max_bid_decrease_pct)
            self.min_conversions = bid_config.get('min_conversions', self.min_conversions)
            self.update_frequency_days = bid_config.get('update_frequency_days', self.update_frequency_days)
            self.target_cpa = bid_config.get('target_cpa', self.target_cpa)
            self.target_roas = bid_config.get('target_roas', self.target_roas)
        
        self.logger.info(f"BidService initialized with settings: "
                        f"max_bid_increase={self.max_bid_increase_pct}%, "
                        f"min_data_points={self.min_data_points}")
    
    def optimize_keyword_bids(self, 
                             days: int = 30, 
                             campaign_id: Optional[str] = None,
                             strategy: str = "performance_based") -> Dict[str, Any]:
        """
        Optimize keyword bids based on the specified strategy.
        
        Args:
            days: Number of days of data to analyze
            campaign_id: Optional campaign ID to filter keywords
            strategy: Bidding strategy to use
                - 'performance_based': General performance metrics
                - 'target_cpa': Target cost per acquisition
                - 'target_roas': Target return on ad spend
                - 'position_based': Position or impression share based
            
        Returns:
            Dictionary with bid optimization results
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting keyword bid optimization using '{strategy}' strategy")
            
            # Fetch keyword performance data
            keywords = self._get_keyword_performance_data(days, campaign_id)
            
            if not keywords:
                self.logger.warning("No keyword data available for bid optimization")
                self._track_execution(start_time, False)
                return {"status": "failed", "message": "No keyword data available"}
            
            # Select optimization strategy based on input
            if strategy == "target_cpa":
                bid_recommendations = self._optimize_bids_target_cpa(keywords)
            elif strategy == "target_roas":
                bid_recommendations = self._optimize_bids_target_roas(keywords)
            elif strategy == "position_based":
                bid_recommendations = self._optimize_bids_position_based(keywords)
            else:  # Default to performance-based
                bid_recommendations = self._optimize_bids_performance_based(keywords)
            
            # Apply safety checks and filters
            final_recommendations = self._apply_bid_safety_checks(bid_recommendations)
            
            # Compile results
            result = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "days_analyzed": days,
                "campaign_id": campaign_id,
                "strategy": strategy,
                "total_keywords_analyzed": len(keywords),
                "total_recommendations": len(final_recommendations),
                "bid_recommendations": final_recommendations
            }
            
            # Save results
            self.save_data(
                result,
                f"bid_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "reports/bids"
            )
            
            self.logger.info(f"Keyword bid optimization completed with {len(final_recommendations)} recommendations")
            self._track_execution(start_time, True)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during keyword bid optimization: {str(e)}")
            self._track_execution(start_time, False)
            return {"status": "failed", "message": str(e)}
    
    def _get_keyword_performance_data(self, 
                                     days: int, 
                                     campaign_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get keyword performance data for bid optimization.
        
        Args:
            days: Number of days of data to analyze
            campaign_id: Optional campaign ID to filter keywords
            
        Returns:
            List of keyword dictionaries with performance data
        """
        try:
            self.logger.info(f"Fetching keyword performance data for the last {days} days")
            
            # Use the Google Ads API to fetch keyword performance data
            keywords = self.ads_api.get_keyword_performance(days_ago=days, campaign_id=campaign_id)
            
            # Filter out keywords with insufficient data
            filtered_keywords = [kw for kw in keywords if kw.get("impressions", 0) >= self.min_data_points]
            
            self.logger.info(f"Fetched {len(keywords)} keywords, {len(filtered_keywords)} with sufficient data")
            return filtered_keywords
            
        except Exception as e:
            self.logger.error(f"Error fetching keyword performance data: {str(e)}")
            return []
    
    def _optimize_bids_performance_based(self, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize bids based on overall keyword performance.
        
        Args:
            keywords: List of keyword dictionaries with performance data
            
        Returns:
            List of bid recommendation dictionaries
        """
        self.logger.info("Optimizing bids using performance-based strategy")
        
        recommendations = []
        
        for keyword in keywords:
            # Get performance metrics
            current_bid = keyword.get("current_bid", 0)
            clicks = keyword.get("clicks", 0)
            impressions = keyword.get("impressions", 0)
            conversions = keyword.get("conversions", 0)
            cost = keyword.get("cost", 0)
            quality_score = keyword.get("quality_score", 0)
            
            # Skip if no current bid or impressions
            if current_bid <= 0 or impressions <= 0:
                continue
            
            # Calculate metrics
            ctr = clicks / impressions if impressions > 0 else 0
            conversion_rate = conversions / clicks if clicks > 0 else 0
            cpc = cost / clicks if clicks > 0 else 0
            
            # Calculate performance score (higher is better)
            # This is a simplified score - in a real implementation, this would be more sophisticated
            performance_score = 0
            
            if conversion_rate > 0:
                # If converting, prioritize conversion rate and quality score
                performance_score = (conversion_rate * 5) + (ctr * 2) + (quality_score / 10)
            else:
                # If not converting, prioritize CTR and quality score
                performance_score = ctr * 3 + (quality_score / 10)
            
            # Determine bid adjustment based on performance score
            if performance_score > 1.5:
                # High performing - increase bid
                adjustment_pct = min(20, self.max_bid_increase_pct)
                new_bid = current_bid * (1 + (adjustment_pct / 100))
                confidence = 0.8
                rationale = "High-performing keyword with excellent metrics"
            elif performance_score > 0.8:
                # Good performing - slight increase
                adjustment_pct = min(10, self.max_bid_increase_pct)
                new_bid = current_bid * (1 + (adjustment_pct / 100))
                confidence = 0.7
                rationale = "Good-performing keyword with solid metrics"
            elif performance_score < 0.3 and conversions == 0:
                # Poor performing with no conversions - decrease bid
                adjustment_pct = min(20, self.max_bid_decrease_pct)
                new_bid = current_bid * (1 - (adjustment_pct / 100))
                confidence = 0.6
                rationale = "Underperforming keyword with no conversions"
            else:
                # Average performing - no change
                continue
            
            # Add recommendation
            recommendations.append({
                "keyword_id": keyword.get("ad_group_criterion_id", ""),
                "keyword_text": keyword.get("keyword_text", ""),
                "match_type": keyword.get("match_type", ""),
                "ad_group_id": keyword.get("ad_group_id", ""),
                "ad_group_name": keyword.get("ad_group_name", ""),
                "campaign_id": keyword.get("campaign_id", ""),
                "campaign_name": keyword.get("campaign_name", ""),
                "current_bid": current_bid,
                "recommended_bid": new_bid,
                "adjustment_pct": adjustment_pct if new_bid > current_bid else -adjustment_pct,
                "confidence": confidence,
                "rationale": rationale,
                "metrics": {
                    "impressions": impressions,
                    "clicks": clicks,
                    "ctr": ctr,
                    "conversions": conversions,
                    "conversion_rate": conversion_rate,
                    "cost": cost,
                    "quality_score": quality_score,
                    "performance_score": performance_score
                }
            })
        
        self.logger.info(f"Generated {len(recommendations)} performance-based bid recommendations")
        return recommendations
    
    def _optimize_bids_target_cpa(self, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize bids using target CPA (Cost Per Acquisition) strategy.
        
        Args:
            keywords: List of keyword dictionaries with performance data
            
        Returns:
            List of bid recommendation dictionaries
        """
        self.logger.info("Optimizing bids using target CPA strategy")
        
        # If no target CPA is set, use the average CPA from converting keywords
        if not self.target_cpa:
            converting_keywords = [kw for kw in keywords if kw.get("conversions", 0) >= self.min_conversions]
            if converting_keywords:
                total_cost = sum(kw.get("cost", 0) for kw in converting_keywords)
                total_conversions = sum(kw.get("conversions", 0) for kw in converting_keywords)
                self.target_cpa = total_cost / total_conversions if total_conversions > 0 else None
            
            if not self.target_cpa:
                self.logger.warning("Could not calculate target CPA, insufficient conversion data")
                return []
                
            self.logger.info(f"Using calculated target CPA: ${self.target_cpa:.2f}")
        
        recommendations = []
        
        for keyword in keywords:
            # Get performance metrics
            current_bid = keyword.get("current_bid", 0)
            clicks = keyword.get("clicks", 0)
            conversions = keyword.get("conversions", 0)
            cost = keyword.get("cost", 0)
            
            # Skip if no current bid or clicks
            if current_bid <= 0 or clicks <= 0:
                continue
            
            # Calculate metrics
            conversion_rate = conversions / clicks if clicks > 0 else 0
            avg_cpc = cost / clicks if clicks > 0 else 0
            current_cpa = cost / conversions if conversions > 0 else float('inf')
            
            # Skip if no conversions or conversion rate is zero
            if conversion_rate <= 0:
                continue
            
            # Calculate recommended bid based on target CPA and conversion rate
            # Formula: bid = target_cpa * conversion_rate
            recommended_bid = self.target_cpa * conversion_rate
            
            # Apply adjustment if there's already a CPA to compare against
            if conversions >= self.min_conversions:
                # If current CPA is higher than target, decrease bid
                if current_cpa > self.target_cpa:
                    adjustment_factor = self.target_cpa / current_cpa
                    recommended_bid = current_bid * adjustment_factor
                    rationale = f"Current CPA (${current_cpa:.2f}) is higher than target (${self.target_cpa:.2f})"
                    confidence = 0.8
                # If current CPA is lower than target, can increase bid
                elif current_cpa < self.target_cpa * 0.8:  # At least 20% better than target
                    adjustment_factor = min(1.2, self.target_cpa / current_cpa)  # Cap at 20% increase
                    recommended_bid = current_bid * adjustment_factor
                    rationale = f"Current CPA (${current_cpa:.2f}) is well below target (${self.target_cpa:.2f})"
                    confidence = 0.7
                else:
                    # CPA is close to target, no change needed
                    continue
            else:
                # Not enough conversion data for high confidence
                # Use conversion rate-based estimate with lower confidence
                adjustment_pct = 0
                if recommended_bid > current_bid * 1.3:  # Cap increase at 30%
                    recommended_bid = current_bid * 1.3
                    adjustment_pct = 30
                elif recommended_bid < current_bid * 0.8:  # Cap decrease at 20%
                    recommended_bid = current_bid * 0.8
                    adjustment_pct = -20
                
                rationale = f"Estimated bid for target CPA of ${self.target_cpa:.2f} with limited conversion data"
                confidence = 0.5
            
            # Calculate adjustment percentage
            adjustment_pct = ((recommended_bid / current_bid) - 1) * 100
            
            # Add recommendation
            recommendations.append({
                "keyword_id": keyword.get("ad_group_criterion_id", ""),
                "keyword_text": keyword.get("keyword_text", ""),
                "match_type": keyword.get("match_type", ""),
                "ad_group_id": keyword.get("ad_group_id", ""),
                "ad_group_name": keyword.get("ad_group_name", ""),
                "campaign_id": keyword.get("campaign_id", ""),
                "campaign_name": keyword.get("campaign_name", ""),
                "current_bid": current_bid,
                "recommended_bid": recommended_bid,
                "adjustment_pct": adjustment_pct,
                "confidence": confidence,
                "rationale": rationale,
                "metrics": {
                    "clicks": clicks,
                    "conversions": conversions,
                    "conversion_rate": conversion_rate,
                    "cost": cost,
                    "current_cpa": current_cpa,
                    "target_cpa": self.target_cpa
                }
            })
        
        self.logger.info(f"Generated {len(recommendations)} target CPA bid recommendations")
        return recommendations
    
    def _optimize_bids_target_roas(self, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize bids using target ROAS (Return On Ad Spend) strategy.
        
        Args:
            keywords: List of keyword dictionaries with performance data
            
        Returns:
            List of bid recommendation dictionaries
        """
        self.logger.info("Optimizing bids using target ROAS strategy")
        
        # If no target ROAS is set, use the average ROAS from converting keywords
        if not self.target_roas:
            converting_keywords = [kw for kw in keywords if kw.get("conversions", 0) >= self.min_conversions]
            if converting_keywords:
                total_value = sum(kw.get("conversion_value", 0) for kw in converting_keywords)
                total_cost = sum(kw.get("cost", 0) for kw in converting_keywords)
                self.target_roas = total_value / total_cost if total_cost > 0 else None
            
            if not self.target_roas:
                self.logger.warning("Could not calculate target ROAS, insufficient conversion value data")
                return []
                
            self.logger.info(f"Using calculated target ROAS: {self.target_roas:.2f}")
        
        # For this simplified implementation, we'll assume conversion value is tracked
        # In a real implementation, you'd need to ensure conversion value tracking is set up
        recommendations = []
        
        return recommendations  # Placeholder: implement similar to target CPA
    
    def _optimize_bids_position_based(self, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize bids based on position metrics like impression share.
        
        Args:
            keywords: List of keyword dictionaries with performance data
            
        Returns:
            List of bid recommendation dictionaries
        """
        self.logger.info("Optimizing bids using position-based strategy")
        
        # For this simplified implementation, we'll focus on search top impression share
        # In a real implementation, you'd use more position metrics
        recommendations = []
        
        return recommendations  # Placeholder: implement based on impression share metrics
    
    def _apply_bid_safety_checks(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply safety checks to ensure bid recommendations are reasonable.
        
        Args:
            recommendations: List of bid recommendation dictionaries
            
        Returns:
            Filtered list of bid recommendation dictionaries
        """
        self.logger.info(f"Applying safety checks to {len(recommendations)} bid recommendations")
        
        safe_recommendations = []
        
        for rec in recommendations:
            current_bid = rec.get("current_bid", 0)
            recommended_bid = rec.get("recommended_bid", 0)
            adjustment_pct = rec.get("adjustment_pct", 0)
            
            # Skip if no current bid or recommended bid
            if current_bid <= 0 or recommended_bid <= 0:
                continue
            
            # Ensure max increase isn't exceeded
            if adjustment_pct > self.max_bid_increase_pct:
                # Cap the increase
                recommended_bid = current_bid * (1 + (self.max_bid_increase_pct / 100))
                adjustment_pct = self.max_bid_increase_pct
                
            # Ensure max decrease isn't exceeded
            elif adjustment_pct < -self.max_bid_decrease_pct:
                # Cap the decrease
                recommended_bid = current_bid * (1 - (self.max_bid_decrease_pct / 100))
                adjustment_pct = -self.max_bid_decrease_pct
            
            # Ensure minimum bid
            if recommended_bid < 0.01:  # $0.01 minimum bid
                recommended_bid = 0.01
                adjustment_pct = ((recommended_bid / current_bid) - 1) * 100
            
            # Update the recommendation
            rec["recommended_bid"] = recommended_bid
            rec["adjustment_pct"] = adjustment_pct
            
            # Always add a safety note
            rec["safety_note"] = f"Bid adjustment capped at {adjustment_pct:.1f}% for safety"
            
            safe_recommendations.append(rec)
        
        self.logger.info(f"{len(safe_recommendations)} recommendations passed safety checks")
        return safe_recommendations
    
    def apply_bid_recommendations(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply bid recommendations to keywords.
        
        Args:
            recommendations: List of bid recommendation dictionaries
            
        Returns:
            Dictionary with application results
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting to apply {len(recommendations)} bid recommendations")
            
            success_count = 0
            failure_count = 0
            results = []
            
            for recommendation in recommendations:
                # Get recommendation details
                keyword_id = recommendation.get("keyword_id", "")
                current_bid = recommendation.get("current_bid", 0)
                recommended_bid = recommendation.get("recommended_bid", 0)
                
                # Skip if no keyword ID or invalid bids
                if not keyword_id or current_bid <= 0 or recommended_bid <= 0:
                    failure_count += 1
                    results.append({
                        "success": False,
                        "keyword_id": keyword_id,
                        "error": "Invalid keyword ID or bid values"
                    })
                    continue
                
                # Convert to micros for the API
                bid_micros = int(recommended_bid * 1000000)
                
                # Apply the bid change
                success, message = self.ads_api.apply_optimization(
                    optimization_type="bid_adjustment",
                    entity_type="keyword",
                    entity_id=keyword_id,
                    changes={"bid_micros": bid_micros}
                )
                
                # Record the result
                result = {
                    "success": success,
                    "message": message,
                    "keyword_id": keyword_id,
                    "keyword_text": recommendation.get("keyword_text", ""),
                    "current_bid": current_bid,
                    "new_bid": recommended_bid,
                    "applied_time": datetime.now().isoformat()
                }
                
                results.append(result)
                
                if success:
                    success_count += 1
                else:
                    failure_count += 1
            
            # Compile results
            application_results = {
                "status": "success" if failure_count == 0 else "partial_success",
                "timestamp": datetime.now().isoformat(),
                "total_recommendations": len(recommendations),
                "success_count": success_count,
                "failure_count": failure_count,
                "results": results
            }
            
            self.logger.info(f"Applied {success_count} bid recommendations, {failure_count} failures")
            self._track_execution(start_time, failure_count == 0)
            
            return application_results
            
        except Exception as e:
            self.logger.error(f"Error applying bid recommendations: {str(e)}")
            self._track_execution(start_time, False)
            return {"status": "failed", "message": str(e)}
    
    def optimize_campaign_budgets(self, days: int = 30) -> Dict[str, Any]:
        """
        Optimize campaign budgets based on performance.
        
        Args:
            days: Number of days of data to analyze
            
        Returns:
            Dictionary with budget optimization results
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting campaign budget optimization for the last {days} days")
            
            # Fetch campaign performance data
            campaigns = self._get_campaign_performance_data(days)
            
            if not campaigns:
                self.logger.warning("No campaign data available for budget optimization")
                self._track_execution(start_time, False)
                return {"status": "failed", "message": "No campaign data available"}
            
            # Generate budget recommendations
            budget_recommendations = self._generate_budget_recommendations(campaigns)
            
            # Apply safety checks
            final_recommendations = self._apply_budget_safety_checks(budget_recommendations)
            
            # Compile results
            result = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "days_analyzed": days,
                "total_campaigns_analyzed": len(campaigns),
                "total_recommendations": len(final_recommendations),
                "budget_recommendations": final_recommendations
            }
            
            # Save results
            self.save_data(
                result,
                f"budget_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "reports/budgets"
            )
            
            self.logger.info(f"Campaign budget optimization completed with {len(final_recommendations)} recommendations")
            self._track_execution(start_time, True)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during campaign budget optimization: {str(e)}")
            self._track_execution(start_time, False)
            return {"status": "failed", "message": str(e)}
    
    def _get_campaign_performance_data(self, days: int) -> List[Dict[str, Any]]:
        """
        Get campaign performance data for budget optimization.
        
        Args:
            days: Number of days of data to analyze
            
        Returns:
            List of campaign dictionaries with performance data
        """
        try:
            self.logger.info(f"Fetching campaign performance data for the last {days} days")
            
            # Use the Google Ads API to fetch campaign performance data
            campaigns = self.ads_api.get_campaign_performance(days_ago=days)
            
            # Filter for active campaigns
            active_campaigns = [c for c in campaigns if c.get("status") == "ENABLED"]
            
            self.logger.info(f"Fetched {len(campaigns)} campaigns, {len(active_campaigns)} are active")
            return active_campaigns
            
        except Exception as e:
            self.logger.error(f"Error fetching campaign performance data: {str(e)}")
            return []
    
    def _generate_budget_recommendations(self, campaigns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate budget recommendations for campaigns.
        
        Args:
            campaigns: List of campaign dictionaries with performance data
            
        Returns:
            List of budget recommendation dictionaries
        """
        self.logger.info("Generating campaign budget recommendations")
        
        # Placeholder for budget recommendations
        # In a real implementation, you'd analyze performance metrics
        # to determine optimal budget allocation
        recommendations = []
        
        return recommendations
    
    def _apply_budget_safety_checks(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply safety checks to ensure budget recommendations are reasonable.
        
        Args:
            recommendations: List of budget recommendation dictionaries
            
        Returns:
            Filtered list of budget recommendation dictionaries
        """
        self.logger.info(f"Applying safety checks to {len(recommendations)} budget recommendations")
        
        # Placeholder for safety checks
        # In a real implementation, you'd apply caps and minimums
        safe_recommendations = recommendations
        
        return safe_recommendations
    
    def adjust_keyword_bids(self, adjustments: Dict[str, float]) -> Dict[str, Any]:
        """
        Adjust bids for specified keywords based on simulation results or optimization suggestions.
        
        Args:
            adjustments: Dictionary mapping keyword criterion IDs to new bid amounts (in currency units, not micros)
            
        Returns:
            Dictionary containing results of bid adjustments for each keyword
        """
        results = {}
        operations = []
        
        try:
            for criterion_id, new_bid in adjustments.items():
                # Convert bid to micros
                bid_micros = int(new_bid * 1000000)
                
                # Create operation for bid adjustment
                operation = self.client.get_type("AdGroupCriterionOperation")
                criterion = operation.update
                criterion.resource_name = f"customers/{self.customer_id}/adGroupCriteria/{criterion_id}"
                criterion.cpc_bid_micros = bid_micros
                operation.update_mask.paths.append("cpc_bid_micros")
                
                operations.append(operation)
                results[criterion_id] = {
                    "new_bid": new_bid,
                    "status": "pending"
                }
            
            if operations:
                response = self.ad_group_criterion_service.mutate_ad_group_criteria(
                    customer_id=self.customer_id, operations=operations
                )
                
                # Process response
                for i, result in enumerate(response.results):
                    criterion_id = operations[i].update.resource_name.split("/")[-1]
                    results[criterion_id]["status"] = "success"
                    results[criterion_id]["message"] = f"Bid updated to {results[criterion_id]['new_bid']}"
            
            return results
            
        except GoogleAdsException as ex:
            error_message = f"Google Ads API error: Request with ID '{ex.request_id}' failed with status '{ex.error.code().name}'"
            if ex.failure:
                error_message += f": {ex.failure.errors[0].message}"
            logger.error(error_message)
            
            # Mark all pending adjustments as failed
            for criterion_id in results:
                if results[criterion_id]["status"] == "pending":
                    results[criterion_id]["status"] = "failed"
                    results[criterion_id]["message"] = error_message
            
            return results
            
        except Exception as e:
            error_message = f"Error adjusting keyword bids: {str(e)}"
            logger.error(error_message)
            
            # Mark all pending adjustments as failed
            for criterion_id in results:
                if results[criterion_id]["status"] == "pending":
                    results[criterion_id]["status"] = "failed"
                    results[criterion_id]["message"] = error_message
            
            return results
    
    def get_current_bids(self, criterion_ids: List[str]) -> Dict[str, Optional[float]]:
        """
        Get current bid amounts for specified keywords.
        
        Args:
            criterion_ids: List of keyword criterion IDs to fetch bids for
            
        Returns:
            Dictionary mapping criterion IDs to current bid amounts (in currency units, not micros)
        """
        results = {cid: None for cid in criterion_ids}
        
        try:
            ga_service = self.client.get_service("GoogleAdsService")
            
            # Create a query for all IDs at once
            ids_str = ",".join(criterion_ids)
            query = f"""
                SELECT
                    ad_group_criterion.criterion_id,
                    ad_group_criterion.effective_cpc_bid_micros
                FROM keyword_view
                WHERE ad_group_criterion.criterion_id IN ({ids_str})
            """
            
            response = ga_service.search(customer_id=self.customer_id, query=query)
            
            for row in response:
                cid = str(row.ad_group_criterion.criterion_id)
                bid_micros = row.ad_group_criterion.effective_cpc_bid_micros
                results[cid] = bid_micros / 1000000 if bid_micros else 0.0
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching current bids: {str(e)}")
            return results 