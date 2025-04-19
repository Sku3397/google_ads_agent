"""
Bandit Service for Google Ads Management System

This module implements a Bayesian multi-armed bandit for dynamic budget and traffic allocation.
It uses Thompson Sampling to balance exploration and exploitation.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import pymc3 as pm
import numpy as np
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException

from services.base_service import BaseService

logger = logging.getLogger(__name__)

class BanditService(BaseService):
    """
    Service for managing budget/traffic allocation using Bayesian multi-armed bandits.
    """
    
    def __init__(self, client: GoogleAdsClient, customer_id: str):
        """
        Initialize the BanditService.
        
        Args:
            client: The Google Ads API client
            customer_id: The Google Ads customer ID
        """
        super().__init__(client, customer_id)
        self.bandits = {}
        self.alpha = 1.0  # Prior for success
        self.beta = 1.0   # Prior for failure
        self.exploration_rate = 0.1  # Exploration rate for Thompson Sampling
        self.logger.info("BanditService initialized.")

    def initialize_bandit(self, campaign_ids: List[str]) -> Dict[str, Any]:
        """
        Initialize a bandit for a set of campaigns.
        
        Args:
            campaign_ids: List of campaign IDs to include in the bandit
            
        Returns:
            Dictionary with initialization status
        """
        try:
            bandit_id = f"bandit_{len(self.bandits) + 1}"
            self.bandits[bandit_id] = {
                'campaigns': campaign_ids,
                'arms': {cid: {'alpha': self.alpha, 'beta': self.beta, 'rewards': 0.0, 'trials': 0} for cid in campaign_ids},
                'total_rewards': 0.0,
                'total_trials': 0
            }
            self.logger.info(f"Initialized bandit {bandit_id} with {len(campaign_ids)} campaigns")
            return {
                "status": "success",
                "bandit_id": bandit_id,
                "message": f"Bandit {bandit_id} initialized with {len(campaign_ids)} campaigns"
            }
        except Exception as e:
            error_message = f"Error initializing bandit: {str(e)}"
            self.logger.error(error_message)
            return {
                "status": "failed",
                "message": error_message
            }
    
    def update_bandit(self, bandit_id: str, campaign_id: str, reward: float, trial: bool = True) -> Dict[str, Any]:
        """
        Update bandit statistics after observing a reward.
        
        Args:
            bandit_id: The ID of the bandit to update
            campaign_id: The campaign ID (arm) to update
            reward: The observed reward (e.g., conversion value)
            trial: Whether this counts as a trial (default True)
            
        Returns:
            Dictionary with update status
        """
        try:
            if bandit_id not in self.bandits:
                raise ValueError(f"Bandit {bandit_id} not found")
            if campaign_id not in self.bandits[bandit_id]['arms']:
                raise ValueError(f"Campaign {campaign_id} not found in bandit {bandit_id}")
            
            arm = self.bandits[bandit_id]['arms'][campaign_id]
            arm['rewards'] += reward
            arm['trials'] += 1 if trial else 0
            arm['alpha'] += reward
            arm['beta'] += (1 if trial else 0) - reward
            
            self.bandits[bandit_id]['total_rewards'] += reward
            self.bandits[bandit_id]['total_trials'] += 1 if trial else 0
            
            self.logger.info(f"Updated bandit {bandit_id} for campaign {campaign_id} with reward {reward}")
            return {
                "status": "success",
                "message": f"Updated bandit {bandit_id} for campaign {campaign_id}"
            }
        except Exception as e:
            error_message = f"Error updating bandit: {str(e)}"
            self.logger.error(error_message)
            return {
                "status": "failed",
                "message": error_message
            }
    
    def select_arm(self, bandit_id: str) -> Dict[str, Any]:
        """
        Select a campaign (arm) to allocate budget/traffic to using Thompson Sampling.
        
        Args:
            bandit_id: The ID of the bandit to select from
            
        Returns:
            Dictionary with selected campaign and rationale
        """
        try:
            if bandit_id not in self.bandits:
                raise ValueError(f"Bandit {bandit_id} not found")
            
            bandit = self.bandits[bandit_id]
            if np.random.random() < self.exploration_rate:
                # Explore: randomly select an arm
                selected_campaign = np.random.choice(bandit['campaigns'])
                rationale = "Random selection for exploration"
            else:
                # Exploit: use Thompson Sampling to select based on sampled success rates
                samples = {cid: np.random.beta(arm['alpha'], arm['beta']) for cid, arm in bandit['arms'].items()}
                selected_campaign = max(samples, key=samples.get)
                rationale = f"Thompson Sampling selected campaign {selected_campaign} with sampled rate {samples[selected_campaign]:.3f}"
            
            self.logger.info(f"Selected campaign {selected_campaign} for bandit {bandit_id}: {rationale}")
            return {
                "status": "success",
                "selected_campaign": selected_campaign,
                "rationale": rationale
            }
        except Exception as e:
            error_message = f"Error selecting arm: {str(e)}"
            self.logger.error(error_message)
            return {
                "status": "failed",
                "message": error_message
            }
    
    def allocate_budget(self, bandit_id: str, total_budget: float) -> Dict[str, Any]:
        """
        Allocate budget dynamically across campaigns based on bandit selection.
        
        Args:
            bandit_id: The ID of the bandit to allocate budget for
            total_budget: Total budget to allocate
            
        Returns:
            Dictionary with budget allocation for each campaign
        """
        try:
            if bandit_id not in self.bandits:
                raise ValueError(f"Bandit {bandit_id} not found")
            
            selection = self.select_arm(bandit_id)
            if selection['status'] != 'success':
                raise ValueError(selection['message'])
            
            selected_campaign = selection['selected_campaign']
            bandit = self.bandits[bandit_id]
            num_campaigns = len(bandit['campaigns'])
            
            # Allocate a larger share to the selected campaign, with some budget to others for exploration
            base_allocation = total_budget * 0.1 / num_campaigns  # 10% spread across all for exploration
            selected_allocation = total_budget * 0.9  # 90% to selected campaign
            
            allocations = {cid: base_allocation for cid in bandit['campaigns']}
            allocations[selected_campaign] += selected_allocation
            
            self.logger.info(f"Allocated budget for bandit {bandit_id}: {allocations}")
            return {
                "status": "success",
                "allocations": allocations,
                "message": f"Budget allocated with {selected_allocation:.2f} to campaign {selected_campaign}"
            }
        except Exception as e:
            error_message = f"Error allocating budget: {str(e)}"
            self.logger.error(error_message)
            return {
                "status": "failed",
                "message": error_message
            }
    
    def get_bandit_stats(self, bandit_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific bandit.
        
        Args:
            bandit_id: The ID of the bandit to get stats for
            
        Returns:
            Dictionary with bandit statistics
        """
        try:
            if bandit_id not in self.bandits:
                raise ValueError(f"Bandit {bandit_id} not found")
            
            bandit = self.bandits[bandit_id]
            stats = {
                "bandit_id": bandit_id,
                "num_campaigns": len(bandit['campaigns']),
                "total_rewards": bandit['total_rewards'],
                "total_trials": bandit['total_trials'],
                "arms": {
                    cid: {
                        "alpha": arm['alpha'],
                        "beta": arm['beta'],
                        "rewards": arm['rewards'],
                        "trials": arm['trials'],
                        "estimated_success_rate": arm['rewards'] / max(1, arm['trials'])
                    } for cid, arm in bandit['arms'].items()
                }
            }
            return {
                "status": "success",
                "stats": stats
            }
        except Exception as e:
            error_message = f"Error getting bandit stats: {str(e)}"
            self.logger.error(error_message)
            return {
                "status": "failed",
                "message": error_message
            }

    def update_bandits(self, performance_data: List[Dict[str, Any]], metric: str = 'conversions', trials_metric: str = 'clicks'):
        """
        Update bandit models based on recent performance data.
        
        Args:
            performance_data: List of dictionaries, each containing performance metrics 
                              for an entity (e.g., campaign performance). 
                              Must include entity 'id', the specified 'metric' (e.g., conversions),
                              and the 'trials_metric' (e.g., clicks or impressions).
            metric: The metric representing reward/success (e.g., 'conversions').
            trials_metric: The metric representing the number of trials (e.g., 'clicks', 'impressions').
        """
        start_time = datetime.now()
        self.logger.info(f"Updating bandits based on recent performance data using metric '{metric}' over '{trials_metric}'.")
        
        updates_applied = 0
        for data in performance_data:
            entity_id = data.get('id')
            
            if entity_id not in self.bandits:
                self.logger.warning(f"Received performance data for unknown entity {entity_id}. Skipping update.")
                continue
                
            bandit = self.bandits[entity_id]
            
            # Get rewards (e.g., conversions) and trials (e.g., clicks)
            rewards = data.get(metric, 0)
            trials = data.get(trials_metric, 0)
            
            if trials < 0 or rewards < 0:
                self.logger.warning(f"Invalid data for entity {entity_id}: trials={trials}, rewards={rewards}. Skipping update.")
                continue

            # Ensure rewards are not greater than trials
            if rewards > trials:
                 self.logger.warning(f"Rewards ({rewards}) exceed trials ({trials}) for entity {entity_id}. Capping rewards at trials.")
                 rewards = trials
                 
            # Failures = trials - rewards
            failures = trials - rewards

            # Update Beta distribution parameters: alpha += rewards, beta += failures
            bandit['alpha'] += rewards
            bandit['beta'] += failures
            bandit['pulls'] += trials
            bandit['rewards'] += rewards
            bandit['last_updated'] = datetime.now().isoformat()
            
            self.logger.debug(f"Updated bandit for {bandit['entity_type']} {entity_id}: +{rewards} rewards, +{failures} failures (total trials: {trials}). New alpha={bandit['alpha']}, beta={bandit['beta']}")
            updates_applied += 1
            
        self.logger.info(f"Finished updating {updates_applied} bandits.")
        
        # Persist bandit state
        self.save_bandit_state()
        
        self._track_execution(start_time, success=True)

    def get_bandit_state(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the current state of all bandits.
        
        Returns:
            Dictionary mapping entity_id to bandit state.
        """
        return self.bandits.copy()

    def save_bandit_state(self, filename: str = "bandit_state.json"):
        """
        Save the current state of the bandits to a file.
        
        Args:
            filename: The name of the file to save the state to.
        """
        state_to_save = self.get_bandit_state()
        self.save_data(state_to_save, filename, directory="history")
        self.logger.info(f"Bandit state saved to history/{filename}")

    def load_bandit_state(self, filename: str = "bandit_state.json"):
        """
        Load bandit state from a file.
        
        Args:
            filename: The name of the file to load the state from.
        """
        loaded_state = self.load_data(filename, directory="history")
        if loaded_state:
            # Basic validation before loading
            valid_state = {}
            for entity_id, state in loaded_state.items():
                 if isinstance(state, dict) and 'alpha' in state and 'beta' in state:
                     valid_state[entity_id] = state
                 else:
                    self.logger.warning(f"Invalid state format for entity {entity_id} in {filename}. Skipping.")
            
            self.bandits = valid_state
            self.logger.info(f"Bandit state successfully loaded from history/{filename} for {len(self.bandits)} entities.")
        else:
            self.logger.warning(f"Could not load bandit state from history/{filename}. Starting fresh or using defaults.")
            
    def run(self, **kwargs):
        """
        Placeholder run method for the service. 
        Actual logic might be triggered by the scheduler based on specific needs.
        """
        self.logger.info("BanditService run method called (currently a placeholder).")
        # Example: Load state, update based on new data, allocate budget
        # self.load_bandit_state()
        # performance_data = fetch_performance_data() # Needs implementation
        # self.update_bandits(performance_data)
        # allocations = self.allocate_budget(total_budget=1000) 
        # apply_allocations(allocations) # Needs implementation
        pass 