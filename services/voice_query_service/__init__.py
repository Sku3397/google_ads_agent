"""
Voice Query Service for optimizing Google Ads campaigns for voice search.

This service helps analyze and optimize ads for voice search queries,
which tend to be longer, more conversational, and question-oriented.
"""

# This file makes the voice_query_service directory a Python package.

from .voice_query_service import VoiceQueryService

__all__ = ["VoiceQueryService"]

# Service metadata
SERVICE_NAME = "VoiceQueryService"
SERVICE_DESCRIPTION = "Voice search optimization for Google Ads"
SERVICE_VERSION = "1.0.0"
SUPPORTED_METHODS = [
    "voice_pattern_detection",
    "question_word_analysis",
    "conversational_keyword_generation",
    "voice_search_simulation",
]
