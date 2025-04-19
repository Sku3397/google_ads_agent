import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import json
from dataclasses import dataclass
import os

# Local imports
from ads_api import GoogleAdsAPI
from optimizer import AdsOptimizer

logger = logging.getLogger(__name__)

@dataclass
class Recommendation:
    """Data class for storing optimization recommendations with confidence scores"""
    id: str
    entity_type: str  # 'campaign', 'ad_group', 'keyword', 'ad'
    entity_id: str
    action_type: str  # 'bid_adjustment', 'status_change', 'budget_adjustment', etc.
    current_value: Union[float, str, None]
    recommended_value: Union[float, str, None]
    impact_score: float  # 0-100 score of expected impact
    confidence_score: float  # 0-100 confidence in the recommendation
    rationale: str  # Human-readable explanation
    priority: str  # 'HIGH', 'MEDIUM', 'LOW'
    status: str  # 'pending', 'approved', 'rejected', 'implemented', 'failed'
    implementation_time: Optional[datetime] = None
    result_metrics: Optional[Dict] = None
    time_period: int = 30  # Default time period
    
    def to_dict(self):
        return {
            'id': self.id,
            'entity_type': self.entity_type,
            'entity_id': self.entity_id,
            'action_type': self.action_type,
            'current_value': self.current_value,
            'recommended_value': self.recommended_value,
            'impact_score': self.impact_score,
            'confidence_score': self.confidence_score,
            'rationale': self.rationale,
            'priority': self.priority,
            'status': self.status,
            'implementation_time': self.implementation_time.isoformat() if self.implementation_time else None,
            'result_metrics': self.result_metrics,
            'time_period': self.time_period
        }

class AutonomousPPCAgent:
    """
    Autonomous PPC Agent that acts like a professional Google Ads manager
    by analyzing campaigns and making data-driven optimization decisions.
    """
    
    def __init__(self, ads_api: GoogleAdsAPI, optimizer: AdsOptimizer, config: Dict):
        """
        Initialize the Autonomous PPC Agent
        
        Args:
            ads_api: Google Ads API instance
            optimizer: Optimizer instance for generating optimization suggestions
            config: Configuration dictionary with agent settings
        """
        self.ads_api = ads_api
        self.optimizer = optimizer
        self.config = config
        
        # Configure autonomous agent settings
        self.auto_implement_threshold = config.get('auto_implement_threshold', 80)  # Confidence threshold for auto-implementation
        self.max_daily_budget_change_pct = config.get('max_daily_budget_change_pct', 20)  # Max % to change budget in a day
        self.min_data_points_required = config.get('min_data_points_required', 30)  # Min data points needed for high confidence
        self.max_keyword_bid_adjustment = config.get('max_keyword_bid_adjustment', 50)  # Max % to adjust bids by
        self.time_periods = config.get('time_periods', [7, 14, 30, 90])  # Default time periods to analyze
        
        # Initialize data storage
        self.campaigns = []
        self.keywords = []
        self.ad_groups = []
        self.ads = []
        
        # Performance tracking
        self.performance_history = {}
        self.recommendation_history = []
        self.recommendations = []  # Add this attribute to store current recommendations
        self.last_data_refresh = None
        self.last_analysis = None
        
        # Decision history for learning
        self.decision_outcomes = []
        
        # Load historical data if available
        self._load_history()
        
        logger.info("Autonomous PPC Agent initialized")
        
    def _load_history(self):
        """Load historical performance and recommendation data from storage"""
        try:
            # Check if history file exists
            history_file = os.path.join("data", "performance_history.json")
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    self.performance_history = json.load(f)
                    
            recommendation_file = os.path.join("data", "recommendation_history.json")
            if os.path.exists(recommendation_file):
                with open(recommendation_file, 'r') as f:
                    self.recommendation_history = json.load(f)
                    
            # Create data directory if it doesn't exist
            if not os.path.exists("data"):
                os.makedirs("data")
                
            logger.info("Loaded historical performance and recommendation data")
        except Exception as e:
            logger.warning(f"Could not load historical data: {str(e)}")
            
    def _save_history(self):
        """Save historical performance and recommendation data to storage"""
        try:
            # Ensure data directory exists
            if not os.path.exists("data"):
                os.makedirs("data")
                
            # Save performance history
            with open(os.path.join("data", "performance_history.json"), 'w') as f:
                json.dump(self.performance_history, f, default=str)
                
            # Save recommendation history
            with open(os.path.join("data", "recommendation_history.json"), 'w') as f:
                json.dump(self.recommendation_history, f, default=str)
                
            logger.info("Saved historical performance and recommendation data")
        except Exception as e:
            logger.error(f"Could not save historical data: {str(e)}")
    
    def refresh_campaign_data(self, days: int = 30) -> List[Dict]:
        """
        Fetch fresh campaign performance data
        
        Args:
            days: Number of days of historical data to fetch
            
        Returns:
            List of campaign data dictionaries
        """
        try:
            logger.info(f"Fetching campaign data for last {days} days")
            campaigns = self.ads_api.get_campaign_performance(days_ago=days)
            
            # Store and track campaign performance
            self.campaigns = campaigns
            self._update_performance_history('campaigns', campaigns)
            self.last_data_refresh = datetime.now()
            
            logger.info(f"Fetched {len(campaigns)} campaigns")
            return campaigns
        except Exception as e:
            logger.error(f"Failed to fetch campaign data: {str(e)}")
            raise
    
    def refresh_keyword_data(self, days: int = 30, campaign_id: Optional[str] = None) -> List[Dict]:
        """
        Fetch fresh keyword performance data
        
        Args:
            days: Number of days of historical data to fetch
            campaign_id: Optional campaign ID to filter keywords
            
        Returns:
            List of keyword data dictionaries
        """
        try:
            logger.info(f"Fetching keyword data for last {days} days")
            keywords = self.ads_api.get_keyword_performance(days_ago=days, campaign_id=campaign_id)
            
            # Store and track keyword performance
            self.keywords = keywords
            self._update_performance_history('keywords', keywords)
            self.last_data_refresh = datetime.now()
            
            logger.info(f"Fetched {len(keywords)} keywords")
            return keywords
        except Exception as e:
            logger.error(f"Failed to fetch keyword data: {str(e)}")
            raise
    
    def _update_performance_history(self, entity_type: str, entities: List[Dict]):
        """
        Update performance history with latest data
        
        Args:
            entity_type: Type of entities ('campaigns', 'keywords', etc.)
            entities: List of entity data dictionaries
        """
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Initialize performance history for entity type if needed
        if entity_type not in self.performance_history:
            self.performance_history[entity_type] = {}
        
        # Update performance history with today's data
        self.performance_history[entity_type][today] = entities
        
        # Keep only the last 90 days of history to avoid excessive storage
        dates = sorted(self.performance_history[entity_type].keys())
        if len(dates) > 90:
            for old_date in dates[:-90]:
                del self.performance_history[entity_type][old_date]
        
        # Save updated history
        self._save_history()
    
    def analyze_and_recommend(self, entity_type: str = 'all', days: int = 30) -> List[Recommendation]:
        """
        Analyze performance data and generate recommendations
        
        Args:
            entity_type: Type of entities to analyze ('campaigns', 'keywords', 'all')
            days: Number of days of historical data to analyze
            
        Returns:
            List of recommendations with confidence scores
        """
        logger.info(f"Starting autonomous analysis for {entity_type}")
        
        # Ensure we have fresh data
        if entity_type in ['campaigns', 'all'] or not self.campaigns:
            self.refresh_campaign_data(days)
            
        if entity_type in ['keywords', 'all'] or not self.keywords:
            self.refresh_keyword_data(days)
        
        # Get raw suggestions from optimizer
        raw_suggestions = self.optimizer.get_optimization_suggestions(self.campaigns, self.keywords)
        
        # Process the suggestions into recommendations with confidence scores
        recommendations = self._process_recommendations(raw_suggestions)
        
        # Sort recommendations by impact score and confidence
        recommendations.sort(key=lambda x: (x.impact_score * x.confidence_score), reverse=True)
        
        # Store recommendations in the agent instance
        self.recommendations = recommendations
        
        # Log recommendations summary
        high_confidence = len([r for r in recommendations if r.confidence_score >= self.auto_implement_threshold])
        logger.info(f"Generated {len(recommendations)} recommendations, {high_confidence} with high confidence")
        
        return recommendations
    
    def _process_recommendations(self, raw_suggestions: Dict) -> List[Recommendation]:
        """
        Process raw optimizer suggestions into actionable recommendations
        with confidence scoring and impact assessment
        
        Args:
            raw_suggestions: Raw suggestions from optimizer
            
        Returns:
            List of processed recommendations
        """
        recommendations = []
        
        # Process campaign recommendations
        if 'campaign_recommendations' in raw_suggestions:
            for i, rec in enumerate(raw_suggestions.get('campaign_recommendations', [])):
                # Calculate confidence based on available data and recommendation type
                confidence = self._calculate_confidence(
                    entity_type='campaign',
                    action_type=self._map_action_type(rec.get('recommendation', '')),
                    data_points=self._get_campaign_data_points(rec.get('campaign_name', '')),
                    priority=rec.get('priority', 'MEDIUM')
                )
                
                # Calculate potential impact score
                impact = self._estimate_impact(
                    entity_type='campaign',
                    action_type=self._map_action_type(rec.get('recommendation', '')),
                    priority=rec.get('priority', 'MEDIUM')
                )
                
                recommendations.append(Recommendation(
                    id=f"campaign_{i}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    entity_type='campaign',
                    entity_id=rec.get('campaign_name', 'unknown'),
                    action_type=self._map_action_type(rec.get('recommendation', '')),
                    current_value=None,  # Need to extract from data
                    recommended_value=None,  # Need to extract from recommendation
                    impact_score=impact,
                    confidence_score=confidence,
                    rationale=f"{rec.get('issue', '')}. {rec.get('expected_impact', '')}",
                    priority=rec.get('priority', 'MEDIUM'),
                    status='pending'
                ))
        
        # Process keyword recommendations
        if 'keyword_recommendations' in raw_suggestions:
            for i, rec in enumerate(raw_suggestions.get('keyword_recommendations', [])):
                # Skip error types
                if rec.get('type') == 'ERROR':
                    continue
                    
                action_type = rec.get('type', '')
                if action_type == 'ADJUST_BID':
                    mapped_action = 'bid_adjustment'
                    current_value = rec.get('current_bid', 0)
                    recommended_value = rec.get('recommended_bid', 0)
                elif action_type == 'PAUSE':
                    mapped_action = 'status_change'
                    current_value = 'ENABLED'
                    recommended_value = 'PAUSED'
                else:
                    mapped_action = action_type.lower()
                    current_value = None
                    recommended_value = None
                
                # Calculate confidence based on available data
                confidence = self._calculate_confidence(
                    entity_type='keyword',
                    action_type=mapped_action,
                    data_points=self._get_keyword_data_points(rec.get('keyword', '')),
                    priority=rec.get('priority', 'MEDIUM')
                )
                
                # Calculate potential impact score
                impact = self._estimate_impact(
                    entity_type='keyword',
                    action_type=mapped_action,
                    priority=rec.get('priority', 'MEDIUM')
                )
                
                recommendations.append(Recommendation(
                    id=f"keyword_{i}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    entity_type='keyword',
                    entity_id=rec.get('keyword', 'unknown'),
                    action_type=mapped_action,
                    current_value=current_value,
                    recommended_value=recommended_value,
                    impact_score=impact,
                    confidence_score=confidence,
                    rationale=rec.get('rationale', 'Based on performance data'),
                    priority=rec.get('priority', 'MEDIUM'),
                    status='pending'
                ))
        
        return recommendations
    
    def _map_action_type(self, recommendation_text: str) -> str:
        """Map recommendation text to action type"""
        text_lower = recommendation_text.lower()
        
        if 'budget' in text_lower or 'spending' in text_lower:
            return 'budget_adjustment'
        elif 'bid' in text_lower:
            return 'bid_adjustment'
        elif 'pause' in text_lower:
            return 'status_change'
        elif 'target' in text_lower and ('audience' in text_lower or 'demographic' in text_lower):
            return 'audience_targeting'
        elif 'keyword' in text_lower and ('add' in text_lower or 'new' in text_lower):
            return 'keyword_addition'
        elif 'ad' in text_lower and ('text' in text_lower or 'copy' in text_lower):
            return 'ad_copy_update'
        else:
            return 'general_optimization'
    
    def _calculate_confidence(self, entity_type: str, action_type: str, data_points: int, priority: str) -> float:
        """
        Calculate confidence score for a recommendation (0-100)
        
        Args:
            entity_type: Type of entity
            action_type: Type of action
            data_points: Number of data points available
            priority: Priority level from the optimizer
            
        Returns:
            Confidence score (0-100)
        """
        # Base confidence from data points
        data_confidence = min(100, (data_points / self.min_data_points_required) * 100)
        
        # Adjust based on action type
        action_confidence = {
            'bid_adjustment': 90,  # High confidence in bid adjustments
            'status_change': 70,   # Medium confidence in status changes
            'budget_adjustment': 60,  # More cautious with budget changes
            'keyword_addition': 80,  # Good confidence in keyword additions
            'ad_copy_update': 75,  # Good confidence in ad copy updates
            'general_optimization': 50  # Lower confidence in general recommendations
        }.get(action_type, 50)
        
        # Adjust based on priority
        priority_factor = {
            'HIGH': 1.1,
            'MEDIUM': 1.0,
            'LOW': 0.8
        }.get(priority, 1.0)
        
        # Calculate final confidence score
        confidence = (data_confidence * 0.6 + action_confidence * 0.4) * priority_factor
        
        # Cap confidence at 100
        return min(100, max(0, confidence))
    
    def _estimate_impact(self, entity_type: str, action_type: str, priority: str) -> float:
        """
        Estimate potential impact of a recommendation (0-100)
        
        Args:
            entity_type: Type of entity
            action_type: Type of action
            priority: Priority level from the optimizer
            
        Returns:
            Impact score (0-100)
        """
        # Base impact by action type
        base_impact = {
            'bid_adjustment': 70,
            'status_change': 85,
            'budget_adjustment': 90,
            'keyword_addition': 60,
            'ad_copy_update': 65,
            'general_optimization': 40
        }.get(action_type, 50)
        
        # Adjust based on priority
        priority_factor = {
            'HIGH': 1.3,
            'MEDIUM': 1.0,
            'LOW': 0.7
        }.get(priority, 1.0)
        
        # Entity type impact adjustment
        entity_factor = {
            'campaign': 1.2,  # Campaign changes have broader impact
            'ad_group': 1.0,
            'keyword': 0.8,   # Keyword changes have more focused impact
            'ad': 0.9
        }.get(entity_type, 1.0)
        
        # Calculate final impact score
        impact = base_impact * priority_factor * entity_factor
        
        # Cap impact at 100
        return min(100, max(0, impact))
    
    def _get_campaign_data_points(self, campaign_name: str) -> int:
        """Get number of data points available for a campaign"""
        # Find campaign in current data
        campaign = next((c for c in self.campaigns if c.get('name') == campaign_name), None)
        
        if campaign:
            # For simplicity, use impressions as a proxy for data points
            return campaign.get('impressions', 0)
        
        return 0
    
    def _get_keyword_data_points(self, keyword_text: str) -> int:
        """Get number of data points available for a keyword"""
        # Find keyword in current data
        keyword = next((k for k in self.keywords if k.get('keyword_text') == keyword_text), None)
        
        if keyword:
            # For simplicity, use impressions as a proxy for data points
            return keyword.get('impressions', 0)
        
        return 0
    
    def execute_recommendations(self, recommendations: List[Recommendation], 
                               auto_implement: bool = True) -> Tuple[List[Recommendation], List[Recommendation]]:
        """
        Execute recommendations, automatically applying those with high confidence
        if auto_implement is True
        
        Args:
            recommendations: List of recommendations to consider
            auto_implement: Whether to automatically implement high-confidence recommendations
            
        Returns:
            Tuple of (implemented_recommendations, pending_recommendations)
        """
        implemented = []
        pending = []
        
        for rec in recommendations:
            # Check if we should auto-implement
            should_implement = (
                auto_implement and 
                rec.confidence_score >= self.auto_implement_threshold and
                self._is_safe_to_implement(rec)
            )
            
            if should_implement:
                success = self._implement_recommendation(rec)
                if success:
                    rec.status = 'implemented'
                    rec.implementation_time = datetime.now()
                    implemented.append(rec)
                else:
                    rec.status = 'failed'
                    pending.append(rec)
            else:
                pending.append(rec)
        
        # Update recommendation history
        self.recommendation_history.extend([self._recommendation_to_dict(r) for r in implemented])
        self._save_history()
        
        # Log results
        logger.info(f"Automatically implemented {len(implemented)} high-confidence recommendations")
        logger.info(f"{len(pending)} recommendations pending manual review")
        
        return implemented, pending
    
    def _recommendation_to_dict(self, rec: Recommendation) -> Dict:
        """Convert Recommendation object to dictionary for storage"""
        return {
            'id': rec.id,
            'entity_type': rec.entity_type,
            'entity_id': rec.entity_id,
            'action_type': rec.action_type,
            'current_value': rec.current_value,
            'recommended_value': rec.recommended_value,
            'impact_score': rec.impact_score,
            'confidence_score': rec.confidence_score,
            'rationale': rec.rationale,
            'priority': rec.priority,
            'status': rec.status,
            'implementation_time': rec.implementation_time.isoformat() if rec.implementation_time else None,
            'result_metrics': rec.result_metrics
        }
    
    def _is_safe_to_implement(self, rec: Recommendation) -> bool:
        """
        Check if a recommendation is safe to implement automatically
        
        Args:
            rec: Recommendation to check
            
        Returns:
            Boolean indicating if implementation is safe
        """
        # Validate based on action type
        if rec.action_type == 'bid_adjustment':
            # Check that current and recommended values exist
            if rec.current_value is None or rec.recommended_value is None:
                return False
                
            # Calculate change percentage
            current = float(rec.current_value)
            if current == 0:  # Avoid division by zero
                return False
                
            recommended = float(rec.recommended_value)
            change_pct = abs((recommended - current) / current * 100)
            
            # Check if change is within safe limits
            return change_pct <= self.max_keyword_bid_adjustment
            
        elif rec.action_type == 'budget_adjustment':
            # Check that current and recommended values exist
            if rec.current_value is None or rec.recommended_value is None:
                return False
                
            # Calculate change percentage
            current = float(rec.current_value)
            if current == 0:  # Avoid division by zero
                return False
                
            recommended = float(rec.recommended_value)
            change_pct = abs((recommended - current) / current * 100)
            
            # Check if change is within safe limits
            return change_pct <= self.max_daily_budget_change_pct
            
        elif rec.action_type == 'status_change':
            # Status changes are usually safe if confidence is high
            return True
            
        # Default to requiring manual review for other action types
        return False
    
    def _implement_recommendation(self, rec: Recommendation) -> bool:
        """
        Implement a recommendation through the Google Ads API
        
        Args:
            rec: Recommendation to implement
            
        Returns:
            Boolean indicating success
        """
        try:
            logger.info(f"Implementing {rec.action_type} for {rec.entity_type} {rec.entity_id}")
            
            # Prepare changes based on recommendation type
            changes = {}
            
            if rec.action_type == 'bid_adjustment':
                # Convert to micros (multiply by 1,000,000)
                bid_micros = int(float(rec.recommended_value) * 1000000)
                changes['bid_micros'] = bid_micros
                
            elif rec.action_type == 'status_change':
                changes['status'] = rec.recommended_value
                
            elif rec.action_type == 'budget_adjustment':
                # Convert to micros (multiply by 1,000,000)
                budget_micros = int(float(rec.recommended_value) * 1000000)
                changes['budget_micros'] = budget_micros
            
            # Apply the optimization through the API
            success, message = self.ads_api.apply_optimization(
                optimization_type=rec.action_type,
                entity_type=rec.entity_type,
                entity_id=rec.entity_id,
                changes=changes
            )
            
            if success:
                logger.info(f"Successfully implemented recommendation: {message}")
            else:
                logger.error(f"Failed to implement recommendation: {message}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error implementing recommendation: {str(e)}")
            return False
    
    def track_recommendation_outcomes(self, days_to_analyze: int = 7):
        """
        Track outcomes of implemented recommendations to improve future decisions
        
        Args:
            days_to_analyze: Number of days to analyze after implementation
        """
        # Get recommendations that have been implemented but not yet evaluated
        implemented_recs = [r for r in self.recommendation_history 
                          if r['status'] == 'implemented' and not r.get('result_metrics')]
        
        # Skip if no recommendations to evaluate
        if not implemented_recs:
            logger.info("No implemented recommendations to evaluate")
            return
        
        logger.info(f"Evaluating outcomes for {len(implemented_recs)} implemented recommendations")
        
        # For each recommendation, compare before/after metrics
        for rec in implemented_recs:
            # Skip if implementation was too recent
            implementation_time = datetime.fromisoformat(rec['implementation_time']) if rec['implementation_time'] else None
            if not implementation_time or (datetime.now() - implementation_time).days < days_to_analyze:
                continue
            
            # Get before/after metrics
            before_metrics = self._get_metrics_before_implementation(rec, days_to_analyze)
            after_metrics = self._get_metrics_after_implementation(rec, days_to_analyze)
            
            # Calculate impact
            impact = self._calculate_impact(before_metrics, after_metrics)
            
            # Update recommendation with results
            rec['result_metrics'] = {
                'before': before_metrics,
                'after': after_metrics,
                'impact': impact
            }
            
            # Add to decision outcomes for learning
            self.decision_outcomes.append({
                'recommendation_id': rec['id'],
                'entity_type': rec['entity_type'],
                'action_type': rec['action_type'],
                'confidence_score': rec['confidence_score'],
                'impact_score_predicted': rec['impact_score'],
                'impact_score_actual': impact.get('overall_impact', 0),
                'implementation_time': rec['implementation_time'],
                'metrics': impact
            })
        
        # Save updated recommendation history
        self._save_history()
        
        logger.info("Completed evaluation of recommendation outcomes")
    
    def _get_metrics_before_implementation(self, rec: Dict, days: int) -> Dict:
        """Get performance metrics before recommendation implementation"""
        # Simplistic implementation - ideally would use historical data properly
        return {
            'impressions': 0,
            'clicks': 0,
            'conversions': 0,
            'cost': 0,
            'ctr': 0,
            'conversion_rate': 0
        }
    
    def _get_metrics_after_implementation(self, rec: Dict, days: int) -> Dict:
        """Get performance metrics after recommendation implementation"""
        # Simplistic implementation - ideally would use current data properly
        return {
            'impressions': 0,
            'clicks': 0,
            'conversions': 0,
            'cost': 0,
            'ctr': 0,
            'conversion_rate': 0
        }
    
    def _calculate_impact(self, before: Dict, after: Dict) -> Dict:
        """Calculate impact of a recommendation by comparing before/after metrics"""
        # Simplistic implementation - would calculate real impact in production
        return {
            'impression_change_pct': 0,
            'click_change_pct': 0,
            'conversion_change_pct': 0,
            'cost_change_pct': 0,
            'ctr_change_pct': 0,
            'conversion_rate_change_pct': 0,
            'overall_impact': 0  # Score from 0-100
        }
    
    def improve_decision_making(self):
        """Learn from previous decisions to improve future recommendations"""
        # This would be a more sophisticated system in production
        # For now, we'll just update confidence thresholds based on outcomes
        if len(self.decision_outcomes) < 5:
            logger.info("Not enough decision outcomes to improve decision making")
            return
        
        logger.info("Improving decision making based on past outcomes")
        
        # Calculate average accuracy of impact predictions
        actual_impacts = [d['impact_score_actual'] for d in self.decision_outcomes]
        predicted_impacts = [d['impact_score_predicted'] for d in self.decision_outcomes]
        
        if len(actual_impacts) > 0:
            accuracy = sum(1 for i, p in zip(actual_impacts, predicted_impacts) 
                          if abs(i - p) <= 20) / len(actual_impacts)
            
            logger.info(f"Current impact prediction accuracy: {accuracy:.2%}")
            
            # Adjust auto-implement threshold based on accuracy
            if accuracy > 0.8:
                # If we're very accurate, we can be more aggressive
                self.auto_implement_threshold = max(75, self.auto_implement_threshold - 5)
            elif accuracy < 0.5:
                # If we're not accurate, be more conservative
                self.auto_implement_threshold = min(90, self.auto_implement_threshold + 5)
                
            logger.info(f"Adjusted auto-implement threshold to {self.auto_implement_threshold}")
    
    def generate_performance_report(self, entity_type: str = 'all') -> Dict:
        """
        Generate a performance report with insights and trends
        
        Args:
            entity_type: Type of entities to include in report
            
        Returns:
            Dictionary with performance data and insights
        """
        logger.info(f"Generating performance report for {entity_type}")
        
        report = {
            'generated_time': datetime.now().isoformat(),
            'summary': {},
            'trends': {},
            'top_performers': {},
            'underperformers': {},
            'recommendations': {},
            'insights': []
        }
        
        # Generate campaign summary if available
        if (entity_type in ['campaigns', 'all']) and self.campaigns:
            # Basic metrics
            total_impressions = sum(c.get('impressions', 0) for c in self.campaigns)
            total_clicks = sum(c.get('clicks', 0) for c in self.campaigns)
            total_conversions = sum(c.get('conversions', 0) for c in self.campaigns)
            total_cost = sum(c.get('cost', 0) for c in self.campaigns)
            
            # Derived metrics
            avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
            avg_conv_rate = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
            avg_cpa = (total_cost / total_conversions) if total_conversions > 0 else 0
            
            # Add to report
            report['summary']['campaigns'] = {
                'count': len(self.campaigns),
                'total_impressions': total_impressions,
                'total_clicks': total_clicks,
                'total_conversions': total_conversions,
                'total_cost': total_cost,
                'avg_ctr': avg_ctr,
                'avg_conversion_rate': avg_conv_rate,
                'avg_cpa': avg_cpa
            }
            
            # Find top and bottom performers
            if self.campaigns:
                # Sort by conversions
                sorted_by_conv = sorted(self.campaigns, key=lambda c: c.get('conversions', 0), reverse=True)
                report['top_performers']['campaigns'] = sorted_by_conv[:3]
                
                # Sort by conversion rate (with minimum clicks)
                campaigns_with_clicks = [c for c in self.campaigns if c.get('clicks', 0) >= 10]
                if campaigns_with_clicks:
                    for c in campaigns_with_clicks:
                        c['conv_rate'] = (c.get('conversions', 0) / c.get('clicks', 1)) * 100
                    sorted_by_conv_rate = sorted(campaigns_with_clicks, key=lambda c: c.get('conv_rate', 0), reverse=True)
                    report['top_performers']['campaigns_by_conv_rate'] = sorted_by_conv_rate[:3]
        
        # Generate keyword summary if available
        if (entity_type in ['keywords', 'all']) and self.keywords:
            # Basic metrics
            total_impressions = sum(k.get('impressions', 0) for k in self.keywords)
            total_clicks = sum(k.get('clicks', 0) for k in self.keywords)
            total_conversions = sum(k.get('conversions', 0) for k in self.keywords)
            total_cost = sum(k.get('cost', 0) for k in self.keywords)
            
            # Derived metrics
            avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
            avg_conv_rate = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
            avg_cpa = (total_cost / total_conversions) if total_conversions > 0 else 0
            
            # Add to report
            report['summary']['keywords'] = {
                'count': len(self.keywords),
                'total_impressions': total_impressions,
                'total_clicks': total_clicks,
                'total_conversions': total_conversions,
                'total_cost': total_cost,
                'avg_ctr': avg_ctr,
                'avg_conversion_rate': avg_conv_rate,
                'avg_cpa': avg_cpa
            }
            
            # Find top and bottom performers
            if self.keywords:
                # Sort by conversions
                sorted_by_conv = sorted(self.keywords, key=lambda k: k.get('conversions', 0), reverse=True)
                report['top_performers']['keywords'] = sorted_by_conv[:5]
                
                # Sort by conversion rate (with minimum clicks)
                keywords_with_clicks = [k for k in self.keywords if k.get('clicks', 0) >= 10]
                if keywords_with_clicks:
                    for k in keywords_with_clicks:
                        k['conv_rate'] = (k.get('conversions', 0) / k.get('clicks', 1)) * 100
                    sorted_by_conv_rate = sorted(keywords_with_clicks, key=lambda k: k.get('conv_rate', 0), reverse=True)
                    report['top_performers']['keywords_by_conv_rate'] = sorted_by_conv_rate[:5]
                
                # Find underperforming keywords (high cost, no conversions)
                high_cost_no_conv = [k for k in self.keywords 
                                   if k.get('cost', 0) > 50 and k.get('conversions', 0) < 1]
                report['underperformers']['keywords_high_cost_no_conv'] = sorted(
                    high_cost_no_conv, key=lambda k: k.get('cost', 0), reverse=True
                )[:5]
        
        # Add general insights
        report['insights'] = self._generate_insights()
        
        return report
    
    def _generate_insights(self) -> List[str]:
        """Generate general insights about account performance"""
        insights = []
        
        # Add some sample insights (would be more sophisticated in production)
        if self.campaigns:
            # Evaluate overall performance
            total_cost = sum(c.get('cost', 0) for c in self.campaigns)
            total_conversions = sum(c.get('conversions', 0) for c in self.campaigns)
            if total_cost > 0 and total_conversions > 0:
                cpa = total_cost / total_conversions
                insights.append(f"Overall account CPA is ${cpa:.2f}")
            
            # Evaluate campaign distribution
            if len(self.campaigns) > 1:
                # Check if one campaign is dominating spend
                costs = [c.get('cost', 0) for c in self.campaigns]
                max_cost = max(costs)
                total_cost = sum(costs)
                if total_cost > 0:
                    max_cost_pct = (max_cost / total_cost) * 100
                    if max_cost_pct > 80:
                        insights.append(f"One campaign accounts for {max_cost_pct:.1f}% of total spend")
        
        if self.keywords:
            # Check keyword performance distribution
            converting_keywords = [k for k in self.keywords if k.get('conversions', 0) > 0]
            if self.keywords and converting_keywords:
                conv_pct = (len(converting_keywords) / len(self.keywords)) * 100
                insights.append(f"Only {conv_pct:.1f}% of keywords have generated conversions")
            
            # Check for high quality score keywords
            high_qs_keywords = [k for k in self.keywords if k.get('quality_score', 0) >= 8]
            if high_qs_keywords:
                high_qs_pct = (len(high_qs_keywords) / len(self.keywords)) * 100
                insights.append(f"{high_qs_pct:.1f}% of keywords have quality scores of 8 or higher")
        
        return insights
    
    def run_daily_optimization(self, days=None):
        """
        Run a complete daily optimization cycle including data refresh,
        analysis, and auto-implementation of high-confidence recommendations.
        
        Args:
            days: Optional days parameter to override default time period
            
        Returns:
            Dict with optimization results and statistics
        """
        try:
            logger.info("Starting daily optimization cycle")
            result = {
                'status': 'success',
                'recommendations': {
                    'total': 0,
                    'high_confidence': 0,
                    'implemented': 0,
                    'pending': 0,
                    'failed': 0
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Use provided days or the agent's longest time period
            if days is None:
                days = max(self.time_periods) if self.time_periods else 30
            
            # Refresh data
            logger.info(f"Refreshing data for the past {days} days")
            self.refresh_campaign_data(days)
            self.refresh_keyword_data(days)
            
            # Generate recommendations
            logger.info("Analyzing data and generating recommendations")
            recommendations = self.analyze_and_recommend(entity_type='all', days=days)
            
            # Store recommendations in the agent
            self.recommendations = recommendations
            
            # Update result statistics
            result['recommendations']['total'] = len(recommendations)
            result['recommendations']['high_confidence'] = len([
                r for r in recommendations 
                if r.confidence_score >= self.auto_implement_threshold
            ])
            
            # Auto-implement high-confidence recommendations
            logger.info(f"Auto-implementing recommendations with confidence >= {self.auto_implement_threshold}")
            implemented, pending = self.execute_recommendations(
                recommendations, auto_implement=True
            )
            
            # Update result statistics
            result['recommendations']['implemented'] = len(implemented)
            result['recommendations']['pending'] = len(pending)
            
            # Count failed recommendations
            result['recommendations']['failed'] = len([
                r for r in recommendations 
                if r.status == 'failed'
            ])
            
            # Save recommendations to history
            self.save_recommendations()
            
            # Log summary
            logger.info(f"Daily optimization cycle completed: "
                        f"{len(implemented)} recommendations implemented, "
                        f"{len(pending)} pending manual review")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in daily optimization cycle: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def _apply_recommendation(self, rec: Recommendation) -> bool:
        """
        Apply a recommendation to the Google Ads account
        
        Args:
            rec: Recommendation to apply
            
        Returns:
            Boolean indicating success
        """
        try:
            logger.info(f"Applying recommendation: {rec.action_type} for {rec.entity_type} {rec.entity_id}")
            
            # Prepare changes based on recommendation type
            changes = {}
            
            if rec.action_type == 'bid_adjustment':
                # Convert to micros (multiply by 1,000,000)
                bid_micros = int(float(rec.recommended_value) * 1000000)
                changes['bid_micros'] = bid_micros
                
            elif rec.action_type == 'status_change':
                changes['status'] = rec.recommended_value
                
            elif rec.action_type == 'budget_adjustment':
                # Convert to micros (multiply by 1,000,000)
                budget_micros = int(float(rec.recommended_value) * 1000000)
                changes['budget_micros'] = budget_micros
            
            # Apply the optimization through the API
            success, message = self.ads_api.apply_optimization(
                optimization_type=rec.action_type,
                entity_type=rec.entity_type,
                entity_id=rec.entity_id,
                changes=changes
            )
            
            if success:
                logger.info(f"Successfully applied recommendation: {message}")
            else:
                logger.error(f"Failed to apply recommendation: {message}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error applying recommendation: {str(e)}")
            return False
    
    def _evaluate_implementations(self):
        """
        Evaluate outcomes of recently implemented recommendations
        """
        # Get recommendations that have been implemented but not yet evaluated
        implemented_recs = [r for r in self.recommendation_history 
                          if r['status'] == 'implemented' and not r.get('result_metrics')]
        
        # Skip if no recommendations to evaluate
        if not implemented_recs:
            logger.info("No implemented recommendations to evaluate")
            return
        
        logger.info(f"Evaluating outcomes for {len(implemented_recs)} implemented recommendations")
        
        # For each recommendation, compare before/after metrics
        for rec in implemented_recs:
            # Skip if implementation was too recent
            implementation_time = datetime.fromisoformat(rec['implementation_time']) if rec['implementation_time'] else None
            if not implementation_time or (datetime.now() - implementation_time).days < 7:
                continue
            
            # Get before/after metrics
            before_metrics = self._get_metrics_before_implementation(rec, 7)
            after_metrics = self._get_metrics_after_implementation(rec, 7)
            
            # Calculate impact
            impact = self._calculate_impact(before_metrics, after_metrics)
            
            # Update recommendation with results
            rec['result_metrics'] = {
                'before': before_metrics,
                'after': after_metrics,
                'impact': impact
            }
            
            # Add to decision outcomes for learning
            self.decision_outcomes.append({
                'recommendation_id': rec['id'],
                'entity_type': rec['entity_type'],
                'action_type': rec['action_type'],
                'confidence_score': rec['confidence_score'],
                'impact_score_predicted': rec['impact_score'],
                'impact_score_actual': impact.get('overall_impact', 0),
                'implementation_time': rec['implementation_time'],
                'metrics': impact
            })
        
        # Save updated recommendation history
        self._save_history()
        
        logger.info("Completed evaluation of recommendation outcomes")
    
    def _save_historical_data(self):
        """
        Save historical performance and recommendation data to storage
        """
        try:
            # Save performance history
            with open('data/performance_history.json', 'w') as f:
                json.dump(self.performance_history, f, default=str)
            
            # Save recommendation history
            with open('data/recommendation_history.json', 'w') as f:
                json.dump(self.recommendation_history, f, default=str)
            
            logger.info("Saved historical performance and recommendation data")
        except Exception as e:
            logger.error(f"Error saving historical data: {str(e)}")
    
    def _get_metrics_before_implementation(self, rec: Dict, days: int) -> Dict:
        """Get performance metrics before recommendation implementation"""
        # Simplistic implementation - ideally would use historical data properly
        return {
            'impressions': 0,
            'clicks': 0,
            'conversions': 0,
            'cost': 0,
            'ctr': 0,
            'conversion_rate': 0
        }
    
    def _get_metrics_after_implementation(self, rec: Dict, days: int) -> Dict:
        """Get performance metrics after recommendation implementation"""
        # Simplistic implementation - ideally would use current data properly
        return {
            'impressions': 0,
            'clicks': 0,
            'conversions': 0,
            'cost': 0,
            'ctr': 0,
            'conversion_rate': 0
        }
    
    def _calculate_impact(self, before: Dict, after: Dict) -> Dict:
        """Calculate impact of a recommendation by comparing before/after metrics"""
        # Simplistic implementation - would calculate real impact in production
        return {
            'impression_change_pct': 0,
            'click_change_pct': 0,
            'conversion_change_pct': 0,
            'cost_change_pct': 0,
            'ctr_change_pct': 0,
            'conversion_rate_change_pct': 0,
            'overall_impact': 0  # Score from 0-100
        }
    
    def _calculate_confidence_score(self, suggestion: Dict) -> float:
        """Calculate confidence score for a recommendation"""
        # Implementation of confidence score calculation based on the suggestion
        # This is a placeholder and should be replaced with the actual logic
        return 50  # Placeholder return, actual implementation needed
    
    def _calculate_impact_score(self, suggestion: Dict) -> float:
        """Calculate impact score for a recommendation"""
        # Implementation of impact score calculation based on the suggestion
        # This is a placeholder and should be replaced with the actual logic
        return 50  # Placeholder return, actual implementation needed
    
    def _normalize_action_type(self, recommendation_text: str) -> str:
        """Normalize recommendation text to action type"""
        # Implementation of normalization logic based on the recommendation text
        # This is a placeholder and should be replaced with the actual logic
        return recommendation_text.lower()
    
    def save_recommendations(self):
        """
        Save current recommendations to the history and to storage.
        """
        try:
            # Create recommendations directory if it doesn't exist
            os.makedirs("data/recommendations", exist_ok=True)
            
            # Convert recommendations to dictionaries if we have recommendations
            if hasattr(self, 'recommendations') and self.recommendations:
                rec_dicts = [rec.to_dict() for rec in self.recommendations]
                
                # Add to history
                self.recommendation_history.extend(rec_dicts)
                
                # Save to file with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"data/recommendations/recommendations_{timestamp}.json"
                
                with open(filename, 'w') as f:
                    json.dump(rec_dicts, f, default=str)
                
                logger.info(f"Saved {len(rec_dicts)} recommendations to {filename}")
            
            # Save overall history too
            with open("data/recommendation_history.json", 'w') as f:
                json.dump(self.recommendation_history, f, default=str)
                
            logger.info(f"Updated recommendation history with {len(self.recommendation_history)} total recommendations")
            
        except Exception as e:
            logger.error(f"Error saving recommendations: {str(e)}") 