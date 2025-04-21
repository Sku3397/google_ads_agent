"""
Bid Service Package for Google Ads Management System

This package provides bid management and optimization services.
"""

# This file makes the bid_service directory a Python package.

# Define the public API for this package
from .bid_service import BidService

__all__ = ["BidService"]
