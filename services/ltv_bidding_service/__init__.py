"""
LTV Bidding Service for Google Ads Management System

This package provides lifetime value bidding capabilities for optimizing
bidding strategies based on predicted customer lifetime value rather than
immediate conversion value.
"""

# This file makes the ltv_bidding_service directory a Python package.

from .ltv_bidding_service import LTVBiddingService

__all__ = ["LTVBiddingService"]
