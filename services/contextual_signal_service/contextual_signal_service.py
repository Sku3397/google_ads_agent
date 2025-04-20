"""
Contextual Signal Service for Google Ads Agent

This service integrates external contextual data sources to enhance ad targeting and optimization.
"""

import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import os
import json
import time

from services.base_service import BaseService


@dataclass
class ContextualSignal:
    """Data class representing a contextual signal"""

    signal_type: str
    source: str
    timestamp: datetime
    value: Any
    relevance_score: float
    metadata: Dict[str, Any]


class ContextualSignalService(BaseService):
    """
    Service for gathering and analyzing external contextual signals
    to improve Google Ads campaign performance.

    This service integrates with various external APIs and data sources to
    gather contextual information such as:
    - Weather data
    - Industry trends
    - Competitive landscape
    - Seasonality factors
    - Economic indicators
    - Social media trends
    """

    def __init__(
        self,
        ads_api=None,
        optimizer=None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the Contextual Signal Service.

        Args:
            ads_api: Google Ads API client instance
            optimizer: AI optimizer instance for generating suggestions
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(ads_api, optimizer, config, logger)

        self.api_keys = {
            "weather": self.config.get("weather_api_key", os.environ.get("WEATHER_API_KEY", "")),
            "news": self.config.get("news_api_key", os.environ.get("NEWS_API_KEY", "")),
            "trends": self.config.get("trends_api_key", os.environ.get("TRENDS_API_KEY", "")),
            "economic": self.config.get("economic_api_key", os.environ.get("ECONOMIC_API_KEY", "")),
            "social": self.config.get("social_api_key", os.environ.get("SOCIAL_API_KEY", "")),
        }

        self.signal_cache = {}
        self.cache_expiry = {}
        self.cache_validity = {
            "weather": timedelta(hours=6),
            "news": timedelta(hours=12),
            "trends": timedelta(days=1),
            "economic": timedelta(days=1),
            "social": timedelta(hours=4),
            "seasonal": timedelta(days=7),
        }

        self.logger.info("Contextual Signal Service initialized")

    def get_all_signals(
        self,
        location: Optional[str] = None,
        industry: Optional[str] = None,
        keywords: Optional[List[str]] = None,
    ) -> Dict[str, List[ContextualSignal]]:
        """
        Get all available contextual signals.

        Args:
            location: Geographic location to get signals for
            industry: Industry segment to get signals for
            keywords: List of keywords to get relevant signals for

        Returns:
            Dictionary of signal type to list of signals
        """
        signals = {}
        start_time = datetime.now()

        try:
            # Get weather signals
            signals["weather"] = self.get_weather_signals(location)

            # Get news and events signals
            signals["news"] = self.get_news_signals(industry, keywords)

            # Get industry trend signals
            signals["trends"] = self.get_trend_signals(industry, keywords)

            # Get economic signals
            signals["economic"] = self.get_economic_signals(location)

            # Get social media signals
            signals["social"] = self.get_social_signals(keywords)

            # Get seasonal signals
            signals["seasonal"] = self.get_seasonal_signals(industry, location)

            self._track_execution(start_time, True)
            return signals

        except Exception as e:
            self.logger.error(f"Error getting contextual signals: {str(e)}")
            self._track_execution(start_time, False)
            return {}

    def get_weather_signals(self, location: Optional[str] = None) -> List[ContextualSignal]:
        """
        Get weather-related signals for a location.

        Args:
            location: Geographic location to get weather for

        Returns:
            List of weather signals
        """
        if not location:
            return []

        # Check cache
        cache_key = f"weather_{location}"
        if cache_key in self.signal_cache and datetime.now() < self.cache_expiry.get(
            cache_key, datetime.min
        ):
            self.logger.info(f"Using cached weather data for {location}")
            return self.signal_cache[cache_key]

        signals = []

        try:
            if not self.api_keys["weather"]:
                self.logger.warning("Weather API key not configured")
                return []

            # Example: OpenWeatherMap API
            url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={self.api_keys['weather']}&units=metric"

            # Use backoff retry pattern for API calls
            max_retries = 3
            retries = 0

            while retries < max_retries:
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    data = response.json()

                    # Extract weather data
                    if "main" in data and "weather" in data:
                        # Temperature signal
                        signals.append(
                            ContextualSignal(
                                signal_type="temperature",
                                source="openweathermap",
                                timestamp=datetime.now(),
                                value=data["main"]["temp"],
                                relevance_score=0.7,  # Higher for extreme temperatures
                                metadata={
                                    "unit": "celsius",
                                    "location": location,
                                    "condition": data["weather"][0]["main"],
                                    "description": data["weather"][0]["description"],
                                },
                            )
                        )

                        # Weather condition signal
                        signals.append(
                            ContextualSignal(
                                signal_type="weather_condition",
                                source="openweathermap",
                                timestamp=datetime.now(),
                                value=data["weather"][0]["main"],
                                relevance_score=0.8,  # Higher for extreme conditions
                                metadata={
                                    "location": location,
                                    "description": data["weather"][0]["description"],
                                    "icon": data["weather"][0]["icon"],
                                },
                            )
                        )

                    # Cache the result
                    self.signal_cache[cache_key] = signals
                    self.cache_expiry[cache_key] = datetime.now() + self.cache_validity["weather"]

                    break  # Success, exit retry loop

                except requests.exceptions.RequestException as e:
                    retries += 1
                    if retries >= max_retries:
                        self.logger.error(
                            f"Failed to get weather data after {max_retries} retries: {str(e)}"
                        )
                        return []

                    # Exponential backoff
                    wait_time = 2**retries
                    self.logger.warning(
                        f"Weather API request failed, retrying in {wait_time}s... ({retries}/{max_retries})"
                    )
                    time.sleep(wait_time)

            return signals

        except Exception as e:
            self.logger.error(f"Error getting weather signals: {str(e)}")
            return []

    def get_news_signals(
        self, industry: Optional[str] = None, keywords: Optional[List[str]] = None
    ) -> List[ContextualSignal]:
        """
        Get news and events signals related to the industry or keywords.

        Args:
            industry: Industry to get news for
            keywords: List of keywords to get news for

        Returns:
            List of news signals
        """
        if not industry and not keywords:
            return []

        # Build search query from industry and keywords
        search_query = industry or ""
        if keywords:
            if search_query:
                search_query += " OR "
            search_query += " OR ".join(keywords)

        # Check cache
        cache_key = f"news_{search_query}"
        if cache_key in self.signal_cache and datetime.now() < self.cache_expiry.get(
            cache_key, datetime.min
        ):
            self.logger.info(f"Using cached news data for query: {search_query}")
            return self.signal_cache[cache_key]

        signals = []

        try:
            if not self.api_keys["news"]:
                self.logger.warning("News API key not configured")
                return []

            # Example: NewsAPI
            url = f"https://newsapi.org/v2/everything?q={search_query}&apiKey={self.api_keys['news']}&sortBy=publishedAt&language=en"

            max_retries = 3
            retries = 0

            while retries < max_retries:
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    data = response.json()

                    # Extract news articles
                    if "articles" in data:
                        for article in data["articles"][:10]:  # Limit to top 10 articles
                            pub_date = datetime.fromisoformat(
                                article["publishedAt"].replace("Z", "+00:00")
                            )

                            # Calculate relevance score based on freshness and title relevance
                            days_old = (datetime.now() - pub_date).days
                            freshness_score = max(0, 1 - (days_old / 7))  # Decay over a week

                            # Check keyword relevance in title
                            title_relevance = 0
                            if keywords:
                                for keyword in keywords:
                                    if keyword.lower() in article["title"].lower():
                                        title_relevance += 0.2  # Add 0.2 for each matching keyword

                            relevance_score = min(0.9, freshness_score * 0.6 + title_relevance)

                            signals.append(
                                ContextualSignal(
                                    signal_type="news_article",
                                    source=article["source"]["name"],
                                    timestamp=pub_date,
                                    value=article["title"],
                                    relevance_score=relevance_score,
                                    metadata={
                                        "url": article["url"],
                                        "description": article["description"],
                                        "source": article["source"]["name"],
                                        "published_at": article["publishedAt"],
                                    },
                                )
                            )

                    # Cache the result
                    self.signal_cache[cache_key] = signals
                    self.cache_expiry[cache_key] = datetime.now() + self.cache_validity["news"]

                    break  # Success, exit retry loop

                except requests.exceptions.RequestException as e:
                    retries += 1
                    if retries >= max_retries:
                        self.logger.error(
                            f"Failed to get news data after {max_retries} retries: {str(e)}"
                        )
                        return []

                    # Exponential backoff
                    wait_time = 2**retries
                    self.logger.warning(
                        f"News API request failed, retrying in {wait_time}s... ({retries}/{max_retries})"
                    )
                    time.sleep(wait_time)

            return signals

        except Exception as e:
            self.logger.error(f"Error getting news signals: {str(e)}")
            return []

    def get_trend_signals(
        self, industry: Optional[str] = None, keywords: Optional[List[str]] = None
    ) -> List[ContextualSignal]:
        """
        Get industry trend signals.

        Args:
            industry: Industry to get trends for
            keywords: List of keywords to get trends for

        Returns:
            List of trend signals
        """
        if not industry and not keywords:
            return []

        # Combine industry and keywords for search
        search_terms = [industry] if industry else []
        if keywords:
            search_terms.extend(keywords)

        # Check cache
        cache_key = f"trends_{'_'.join(search_terms)}"
        if cache_key in self.signal_cache and datetime.now() < self.cache_expiry.get(
            cache_key, datetime.min
        ):
            self.logger.info(f"Using cached trend data for {search_terms}")
            return self.signal_cache[cache_key]

        signals = []

        try:
            # Note: In a production environment, you would use Google Trends API or similar
            # For this implementation, we'll simulate trend data

            # Simulate trend data for each search term
            for term in search_terms:
                # Generate random trend data (in a real implementation, this would be API data)
                trend_value = np.random.normal(100, 15)  # Mean 100, std 15

                # Calculate trend direction (last 7 days)
                # In real implementation, would compare with historical data
                last_week_value = trend_value * (1 + np.random.uniform(-0.2, 0.2))
                trend_direction = "up" if trend_value > last_week_value else "down"

                signals.append(
                    ContextualSignal(
                        signal_type="search_trend",
                        source="google_trends_simulation",  # Would be actual API in production
                        timestamp=datetime.now(),
                        value=trend_value,
                        relevance_score=0.85,  # Trends usually highly relevant
                        metadata={
                            "term": term,
                            "trend_direction": trend_direction,
                            "percent_change": abs(
                                (trend_value - last_week_value) / last_week_value * 100
                            ),
                            "period": "last_7_days",
                        },
                    )
                )

            # Cache the result
            self.signal_cache[cache_key] = signals
            self.cache_expiry[cache_key] = datetime.now() + self.cache_validity["trends"]

            return signals

        except Exception as e:
            self.logger.error(f"Error getting trend signals: {str(e)}")
            return []

    def get_economic_signals(self, location: Optional[str] = None) -> List[ContextualSignal]:
        """
        Get economic indicators relevant to the location.

        Args:
            location: Geographic location to get economic indicators for

        Returns:
            List of economic signals
        """
        if not location:
            return []

        # Check cache
        cache_key = f"economic_{location}"
        if cache_key in self.signal_cache and datetime.now() < self.cache_expiry.get(
            cache_key, datetime.min
        ):
            self.logger.info(f"Using cached economic data for {location}")
            return self.signal_cache[cache_key]

        signals = []

        try:
            # Note: Would use economic API in production (e.g., Alpha Vantage, FRED)
            # For this implementation, simulate economic data

            # Map location to country/region for economic data lookup
            # In real implementation, would have proper geocoding
            country_mapping = {
                "new york": "us",
                "los angeles": "us",
                "london": "uk",
                "paris": "france",
                "tokyo": "japan",
                "sydney": "australia",
                # Add more mappings as needed
            }

            # Try to get country from location (simple approach)
            country = None
            location_lower = location.lower()
            for loc, cntry in country_mapping.items():
                if loc in location_lower:
                    country = cntry
                    break

            if not country:
                # Default to global if can't determine country
                country = "global"

            # Sample economic indicators (would come from API in production)
            indicators = {
                "us": {"gdp_growth": 2.1, "inflation": 3.2, "unemployment": 3.8},
                "uk": {"gdp_growth": 1.7, "inflation": 2.8, "unemployment": 4.2},
                "france": {"gdp_growth": 1.5, "inflation": 2.5, "unemployment": 7.5},
                "japan": {"gdp_growth": 1.2, "inflation": 1.0, "unemployment": 2.8},
                "australia": {"gdp_growth": 2.0, "inflation": 2.2, "unemployment": 3.5},
                "global": {"gdp_growth": 3.1, "inflation": 3.5, "unemployment": 5.0},
            }

            # Get relevant economic data
            if country in indicators:
                for indicator, value in indicators[country].items():
                    signals.append(
                        ContextualSignal(
                            signal_type=f"economic_{indicator}",
                            source="economic_data_simulation",  # Would be actual API in production
                            timestamp=datetime.now(),
                            value=value,
                            relevance_score=0.7,  # Economic data moderately relevant
                            metadata={
                                "location": location,
                                "country": country,
                                "indicator_name": indicator,
                                "period": "latest_quarter",
                            },
                        )
                    )

            # Cache the result
            self.signal_cache[cache_key] = signals
            self.cache_expiry[cache_key] = datetime.now() + self.cache_validity["economic"]

            return signals

        except Exception as e:
            self.logger.error(f"Error getting economic signals: {str(e)}")
            return []

    def get_social_signals(self, keywords: Optional[List[str]] = None) -> List[ContextualSignal]:
        """
        Get social media trend signals related to keywords.

        Args:
            keywords: List of keywords to get social media trends for

        Returns:
            List of social media signals
        """
        if not keywords:
            return []

        # Check cache
        cache_key = f"social_{'_'.join(keywords)}"
        if cache_key in self.signal_cache and datetime.now() < self.cache_expiry.get(
            cache_key, datetime.min
        ):
            self.logger.info(f"Using cached social media data for {keywords}")
            return self.signal_cache[cache_key]

        signals = []

        try:
            # Note: Would use social media APIs in production (Twitter, Facebook, etc.)
            # For this implementation, simulate social media data

            # Platform sentiment simulation for each keyword
            platforms = ["twitter", "facebook", "instagram", "tiktok"]

            for keyword in keywords:
                for platform in platforms:
                    # Simulate sentiment and volume
                    sentiment = np.random.uniform(-1, 1)  # -1 to 1 scale
                    volume = np.random.lognormal(5, 1)  # Log-normal distribution for volumes

                    signals.append(
                        ContextualSignal(
                            signal_type="social_sentiment",
                            source=f"{platform}_simulation",  # Would be actual API in production
                            timestamp=datetime.now(),
                            value=sentiment,
                            relevance_score=0.75,  # Social signals fairly relevant
                            metadata={
                                "keyword": keyword,
                                "platform": platform,
                                "volume": volume,
                                "period": "last_24h",
                            },
                        )
                    )

            # Cache the result
            self.signal_cache[cache_key] = signals
            self.cache_expiry[cache_key] = datetime.now() + self.cache_validity["social"]

            return signals

        except Exception as e:
            self.logger.error(f"Error getting social signals: {str(e)}")
            return []

    def get_seasonal_signals(
        self, industry: Optional[str] = None, location: Optional[str] = None
    ) -> List[ContextualSignal]:
        """
        Get seasonality signals relevant to the industry and location.

        Args:
            industry: Industry to get seasonality data for
            location: Geographic location to get seasonality data for

        Returns:
            List of seasonality signals
        """
        if not industry and not location:
            return []

        # Check cache
        cache_key = f"seasonal_{industry}_{location}"
        if cache_key in self.signal_cache and datetime.now() < self.cache_expiry.get(
            cache_key, datetime.min
        ):
            self.logger.info(f"Using cached seasonality data for {industry} in {location}")
            return self.signal_cache[cache_key]

        signals = []

        try:
            # Current date information
            now = datetime.now()
            month = now.month
            day = now.day

            # Upcoming holidays (simple holiday detection)
            holidays = self._get_upcoming_holidays(location)

            # Generate seasonality signals

            # 1. Current season
            seasons = {
                # Northern hemisphere
                "north": {
                    1: "winter",
                    2: "winter",
                    3: "spring",
                    4: "spring",
                    5: "spring",
                    6: "summer",
                    7: "summer",
                    8: "summer",
                    9: "fall",
                    10: "fall",
                    11: "fall",
                    12: "winter",
                },
                # Southern hemisphere
                "south": {
                    1: "summer",
                    2: "summer",
                    3: "fall",
                    4: "fall",
                    5: "fall",
                    6: "winter",
                    7: "winter",
                    8: "winter",
                    9: "spring",
                    10: "spring",
                    11: "spring",
                    12: "summer",
                },
            }

            # Simple hemisphere detection
            hemisphere = "north"  # Default to northern hemisphere
            southern_locations = [
                "australia",
                "new zealand",
                "argentina",
                "brazil",
                "chile",
                "south africa",
            ]
            if location and any(loc in location.lower() for loc in southern_locations):
                hemisphere = "south"

            current_season = seasons[hemisphere][month]

            signals.append(
                ContextualSignal(
                    signal_type="season",
                    source="calendar_analysis",
                    timestamp=now,
                    value=current_season,
                    relevance_score=0.6,  # Seasonal factors moderately relevant
                    metadata={"hemisphere": hemisphere, "month": month, "location": location},
                )
            )

            # 2. Holiday proximity signals
            for holiday in holidays:
                days_until = (holiday["date"] - now.date()).days
                if 0 <= days_until <= 30:  # Only include holidays in next 30 days
                    proximity_score = 1 - (days_until / 30)  # Higher score as holiday approaches
                    signals.append(
                        ContextualSignal(
                            signal_type="holiday_proximity",
                            source="calendar_analysis",
                            timestamp=now,
                            value=holiday["name"],
                            relevance_score=0.8
                            * proximity_score,  # More relevant as holiday approaches
                            metadata={
                                "holiday_name": holiday["name"],
                                "days_until": days_until,
                                "date": holiday["date"].isoformat(),
                                "location": location,
                            },
                        )
                    )

            # 3. Industry seasonality (if industry provided)
            if industry:
                # Industry seasonality mapping (simplified)
                industry_seasonality = {
                    "retail": {
                        "high": [11, 12],  # November, December (holiday shopping)
                        "medium": [
                            1,
                            5,
                            6,
                            8,
                            # January (post-holiday), May (summer prep), June (graduation), August (back to school)
                        ],
                        "low": [2, 3, 4, 7, 9, 10],  # Other months
                    },
                    "travel": {
                        "high": [6, 7, 8, 12],  # Summer vacation, Christmas
                        "medium": [3, 4, 5, 11],  # Spring break, Thanksgiving
                        "low": [1, 2, 9, 10],  # Other months
                    },
                    "home improvement": {
                        "high": [4, 5, 6],  # Spring/early summer
                        "medium": [7, 8, 9],  # Late summer/early fall
                        "low": [1, 2, 3, 10, 11, 12],  # Winter, late fall
                    },
                    # Add more industries as needed
                }

                # Find nearest matching industry
                matched_industry = None
                for ind in industry_seasonality.keys():
                    if ind in industry.lower():
                        matched_industry = ind
                        break

                if matched_industry:
                    seasonal_demand = "medium"  # Default
                    if month in industry_seasonality[matched_industry]["high"]:
                        seasonal_demand = "high"
                    elif month in industry_seasonality[matched_industry]["low"]:
                        seasonal_demand = "low"

                    signals.append(
                        ContextualSignal(
                            signal_type="industry_seasonality",
                            source="industry_analysis",
                            timestamp=now,
                            value=seasonal_demand,
                            relevance_score=0.85,  # Industry-specific seasonality highly relevant
                            metadata={
                                "industry": matched_industry,
                                "month": month,
                                "demand_level": seasonal_demand,
                            },
                        )
                    )

            # Cache the result
            self.signal_cache[cache_key] = signals
            self.cache_expiry[cache_key] = datetime.now() + self.cache_validity["seasonal"]

            return signals

        except Exception as e:
            self.logger.error(f"Error getting seasonal signals: {str(e)}")
            return []

    def _get_upcoming_holidays(self, location: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get upcoming holidays for a location.

        Args:
            location: Geographic location to get holidays for

        Returns:
            List of holiday dictionaries with name and date
        """
        # Simplified holiday list
        # In production, would use a proper holiday API or database
        now = datetime.now()
        year = now.year

        # Basic US holidays
        us_holidays = [
            {"name": "New Year's Day", "date": datetime(year, 1, 1).date()},
            {"name": "Valentine's Day", "date": datetime(year, 2, 14).date()},
            {"name": "Memorial Day", "date": datetime(year, 5, 31).date()},  # Simplified
            {"name": "Independence Day", "date": datetime(year, 7, 4).date()},
            {"name": "Labor Day", "date": datetime(year, 9, 6).date()},  # Simplified
            {"name": "Halloween", "date": datetime(year, 10, 31).date()},
            {"name": "Thanksgiving", "date": datetime(year, 11, 25).date()},  # Simplified
            {"name": "Black Friday", "date": datetime(year, 11, 26).date()},  # Simplified
            {"name": "Cyber Monday", "date": datetime(year, 11, 29).date()},  # Simplified
            {"name": "Christmas", "date": datetime(year, 12, 25).date()},
        ]

        # Basic UK holidays
        uk_holidays = [
            {"name": "New Year's Day", "date": datetime(year, 1, 1).date()},
            {"name": "Valentine's Day", "date": datetime(year, 2, 14).date()},
            {"name": "Mother's Day", "date": datetime(year, 3, 14).date()},  # Simplified
            {"name": "Easter", "date": datetime(year, 4, 4).date()},  # Simplified
            {"name": "May Bank Holiday", "date": datetime(year, 5, 3).date()},
            {"name": "Summer Bank Holiday", "date": datetime(year, 8, 30).date()},
            {"name": "Halloween", "date": datetime(year, 10, 31).date()},
            {"name": "Guy Fawkes Night", "date": datetime(year, 11, 5).date()},
            {"name": "Black Friday", "date": datetime(year, 11, 26).date()},  # Simplified
            {"name": "Christmas", "date": datetime(year, 12, 25).date()},
            {"name": "Boxing Day", "date": datetime(year, 12, 26).date()},
        ]

        # Default to US holidays
        holidays = us_holidays

        # Simple location-based holiday selection
        if location:
            location_lower = location.lower()
            if any(
                country in location_lower
                for country in ["uk", "england", "britain", "scotland", "wales"]
            ):
                holidays = uk_holidays

        # Filter to only upcoming holidays or in the last 7 days
        today = now.date()
        filtered_holidays = [
            h
            for h in holidays
            if h["date"] >= today - timedelta(days=7)  # Include recent holidays too
        ]

        return filtered_holidays

    def analyze_signals_for_keywords(
        self, signals: Dict[str, List[ContextualSignal]], keywords: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze signals to extract keyword-specific relevance scores and insights.

        Args:
            signals: Dictionary of signals by type
            keywords: List of keywords to analyze signals for

        Returns:
            Dictionary of keyword to signal type to relevance score
        """
        results = {}

        for keyword in keywords:
            keyword_lower = keyword.lower()
            results[keyword] = {}

            # Process each signal type
            for signal_type, signal_list in signals.items():
                # Initial score for this signal type
                type_score = 0
                signal_count = 0

                for signal in signal_list:
                    # Calculate relevance to this keyword
                    keyword_relevance = 0

                    # Check for keyword in signal value
                    if isinstance(signal.value, str) and keyword_lower in signal.value.lower():
                        keyword_relevance += 0.8

                    # Check metadata for keyword relevance
                    for key, value in signal.metadata.items():
                        if isinstance(value, str) and keyword_lower in value.lower():
                            keyword_relevance += 0.5

                    # For news signals, check title and description
                    if signal.signal_type == "news_article":
                        if (
                            "description" in signal.metadata
                            and keyword_lower in signal.metadata["description"].lower()
                        ):
                            keyword_relevance += 0.6

                    # For social signals, check if exact keyword match
                    if (
                        signal.signal_type == "social_sentiment"
                        and signal.metadata.get("keyword", "").lower() == keyword_lower
                    ):
                        keyword_relevance = 1.0  # Direct match

                    # If we found relevance, include in score
                    if keyword_relevance > 0:
                        type_score += signal.relevance_score * keyword_relevance
                        signal_count += 1

                # Calculate average score if we have signals
                if signal_count > 0:
                    results[keyword][signal_type] = type_score / signal_count
                else:
                    results[keyword][signal_type] = 0

        return results

    def get_recommendations_from_signals(
        self, signals: Dict[str, List[ContextualSignal]], keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate specific recommendations based on contextual signals.

        Args:
            signals: Dictionary of signals by type
            keywords: List of keywords for campaigns

        Returns:
            List of recommendation dictionaries
        """
        recommendations = []

        # Get keyword-specific signal analysis
        keyword_analysis = self.analyze_signals_for_keywords(signals, keywords)

        # Check for weather-based recommendations
        if "weather" in signals and signals["weather"]:
            weather_conditions = [
                s for s in signals["weather"] if s.signal_type == "weather_condition"
            ]
            if weather_conditions:
                for condition in weather_conditions:
                    if condition.value.lower() in ["rain", "snow", "thunderstorm"]:
                        # Bad weather - recommend indoor product keywords
                        recommendations.append(
                            {
                                "type": "bid_adjustment",
                                "trigger": f"Bad weather ({condition.value}) detected in {condition.metadata.get('location', 'target area')}",
                                "action": "Increase bids for indoor-related keywords",
                                "adjustment_factor": 1.15,
                                "confidence": 0.75,
                                "keywords_affected": [
                                    k
                                    for k in keywords
                                    if any(
                                        indoor_term in k.lower()
                                        for indoor_term in ["indoor", "inside", "home", "delivery"]
                                    )
                                ],
                            }
                        )
                    elif condition.value.lower() in ["clear", "sunny"]:
                        # Good weather - recommend outdoor product keywords
                        recommendations.append(
                            {
                                "type": "bid_adjustment",
                                "trigger": f"Good weather ({condition.value}) detected in {condition.metadata.get('location', 'target area')}",
                                "action": "Increase bids for outdoor-related keywords",
                                "adjustment_factor": 1.15,
                                "confidence": 0.75,
                                "keywords_affected": [
                                    k
                                    for k in keywords
                                    if any(
                                        outdoor_term in k.lower()
                                        for outdoor_term in [
                                            "outdoor",
                                            "outside",
                                            "garden",
                                            "patio",
                                        ]
                                    )
                                ],
                            }
                        )

        # Check for holiday-based recommendations
        if "seasonal" in signals:
            holiday_signals = [
                s for s in signals["seasonal"] if s.signal_type == "holiday_proximity"
            ]
            for holiday in holiday_signals:
                days_until = holiday.metadata.get("days_until", 0)
                if days_until < 14:  # Within 2 weeks of holiday
                    holiday_name = holiday.value

                    # Create holiday bidding recommendation
                    keywords_affected = []
                    for keyword in keywords:
                        # Simple relevance check - see if holiday terms appear in keyword
                        if (
                            holiday_name.lower().split()[0] in keyword.lower()
                        ):  # E.g. "Christmas" in "Christmas gifts"
                            keywords_affected.append(keyword)

                    if keywords_affected:
                        # Calculate adjustment factor - higher as holiday approaches
                        adjustment_factor = 1.1 + (0.2 * (1 - days_until / 14))  # 1.1 to 1.3 range

                        recommendations.append(
                            {
                                "type": "bid_adjustment",
                                "trigger": f"Upcoming holiday: {holiday_name} in {days_until} days",
                                "action": f"Increase bids for {holiday_name}-related keywords",
                                "adjustment_factor": round(adjustment_factor, 2),
                                "confidence": 0.8,
                                "keywords_affected": keywords_affected,
                            }
                        )

        # Check for industry seasonality recommendations
        if "seasonal" in signals:
            industry_signals = [
                s for s in signals["seasonal"] if s.signal_type == "industry_seasonality"
            ]
            for industry_signal in industry_signals:
                demand_level = industry_signal.value
                industry = industry_signal.metadata.get("industry", "")

                if demand_level == "high":
                    recommendations.append(
                        {
                            "type": "budget_adjustment",
                            "trigger": f"High season for {industry}",
                            "action": "Increase campaign budgets during peak season",
                            "adjustment_factor": 1.25,
                            "confidence": 0.85,
                            "industry": industry,
                        }
                    )
                elif demand_level == "low":
                    recommendations.append(
                        {
                            "type": "budget_adjustment",
                            "trigger": f"Low season for {industry}",
                            "action": "Decrease campaign budgets during off-season",
                            "adjustment_factor": 0.8,
                            "confidence": 0.75,
                            "industry": industry,
                        }
                    )

        # Generate keyword suggestions based on trending news topics
        if "news" in signals and "trends" in signals:
            # Extract trending topics from news and trends
            trending_topics = set()

            # From news
            for news_signal in signals["news"]:
                if news_signal.relevance_score > 0.7:  # Only highly relevant news
                    # Extract keywords from title
                    title_words = news_signal.value.lower().split()
                    # Filter out common words (would use a proper NLP library in production)
                    common_words = {
                        "the",
                        "a",
                        "an",
                        "to",
                        "in",
                        "of",
                        "for",
                        "and",
                        "on",
                        "is",
                        "are",
                    }
                    for word in title_words:
                        if word not in common_words and len(word) > 4:  # Simple filtering
                            trending_topics.add(word)

            # From trends
            for trend_signal in signals["trends"]:
                if trend_signal.metadata.get("trend_direction") == "up":
                    trending_topics.add(trend_signal.metadata.get("term", "").lower())

            # Generate keyword suggestions
            if trending_topics:
                recommendations.append(
                    {
                        "type": "new_keywords",
                        "trigger": "Detected trending topics in news and search trends",
                        "action": "Consider adding these trending keywords to campaigns",
                        "confidence": 0.7,
                        "suggested_keywords": list(trending_topics),
                    }
                )

        return recommendations

    def apply_signal_based_optimizations(self, campaign_id: str) -> Tuple[bool, str]:
        """
        Apply optimizations to a campaign based on contextual signals.

        Args:
            campaign_id: ID of the campaign to optimize

        Returns:
            Success status and message
        """
        start_time = datetime.now()

        try:
            if not self.ads_api:
                return False, "Google Ads API client not initialized"

            # 1. Get campaign data
            campaign_data = self.ads_api.get_campaign_performance(days_ago=30)
            campaign = next((c for c in campaign_data if c["id"] == campaign_id), None)

            if not campaign:
                return False, f"Campaign with ID {campaign_id} not found"

            # 2. Get keywords for this campaign
            keywords_data = self.ads_api.get_keyword_performance(
                days_ago=30, campaign_id=campaign_id
            )
            keywords = [k["keyword_text"] for k in keywords_data]

            if not keywords:
                return False, f"No keywords found for campaign {campaign_id}"

            # 3. Get location and industry info (would normally get from campaign settings)
            # For demo, using placeholders
            location = "New York"  # Placeholder
            industry = "Retail"  # Placeholder

            # 4. Get contextual signals
            signals = self.get_all_signals(location=location, industry=industry, keywords=keywords)

            if not signals:
                return False, "No contextual signals available"

            # 5. Generate recommendations
            recommendations = self.get_recommendations_from_signals(signals, keywords)

            if not recommendations:
                self._track_execution(start_time, True)
                return True, "No contextual signal-based optimizations recommended at this time"

            # 6. Apply recommendations (only bid adjustments for now)
            applied_count = 0
            for rec in recommendations:
                if rec["type"] == "bid_adjustment" and rec["keywords_affected"]:
                    # Find the relevant keyword criteria
                    for keyword in rec["keywords_affected"]:
                        keyword_data = next(
                            (k for k in keywords_data if k["keyword_text"] == keyword), None
                        )

                        if keyword_data and "resource_name" in keyword_data:
                            # Calculate new bid
                            current_bid_micros = int(keyword_data["current_bid"] * 1000000)
                            new_bid_micros = int(current_bid_micros * rec["adjustment_factor"])

                            # Apply the bid change
                            success, message = self.ads_api.apply_optimization(
                                "bid_adjustment",
                                "keyword",
                                keyword_data["resource_name"],
                                {"bid_micros": new_bid_micros},
                            )

                            if success:
                                applied_count += 1
                                self.logger.info(
                                    f"Applied signal-based bid adjustment to keyword '{keyword}': "
                                    f"${current_bid_micros / 1000000:.2f} -> ${new_bid_micros / 1000000:.2f}"
                                )

            # 7. Track and return results
            self._track_execution(start_time, True)

            if applied_count > 0:
                return (
                    True,
                    f"Successfully applied {applied_count} contextual signal-based optimizations",
                )
            else:
                return True, "No optimizations applied - no actionable recommendations found"

        except Exception as e:
            self.logger.error(f"Error applying signal-based optimizations: {str(e)}")
            self._track_execution(start_time, False)
            return False, f"Error applying signal-based optimizations: {str(e)}"
