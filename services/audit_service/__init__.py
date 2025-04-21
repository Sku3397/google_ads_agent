"""
Audit Service Package for Google Ads Management System

This package provides audit and analysis services for Google Ads accounts.
"""

# This file makes the audit_service directory a Python package.

# Define the public API for this package
from .audit_service import AuditService

__all__ = ["AuditService"]
