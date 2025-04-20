"""
Generative Content Service for Google Ads Management System

This module provides AI-powered generation of ad content, headlines, descriptions,
and creative suggestions to improve ad performance.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
import re

from services.base_service import BaseService


class GenerativeContentService(BaseService):
    """
    Service for generating ad content and creative suggestions using AI.
    """

    def __init__(
        self,
        ads_api=None,
        optimizer=None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the GenerativeContentService.

        Args:
            ads_api: Google Ads API client instance
            optimizer: AI optimizer instance for generating suggestions
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(ads_api, optimizer, config, logger)
        self.logger.info("GenerativeContentService initialized.")

        # Content generation settings
        self.max_headline_length = 30
        self.max_description_length = 90
        self.tones = ["professional", "conversational", "enthusiastic", "informative"]

    def generate_headlines(
        self,
        keywords: List[str],
        product_info: Dict[str, Any],
        num_headlines: int = 5,
        tone: str = "professional",
    ) -> List[Dict[str, Any]]:
        """
        Generate ad headlines based on keywords and product information.

        Args:
            keywords: List of target keywords
            product_info: Dictionary with product/service information
            num_headlines: Number of headlines to generate (default 5)
            tone: Tone for the headlines (default "professional")

        Returns:
            List of headline dictionaries
        """
        start_time = datetime.now()
        self.logger.info(f"Generating {num_headlines} headlines with '{tone}' tone")

        if not keywords or not product_info:
            self.logger.error("Missing required input for headline generation")
            return []

        try:
            # Validate tone
            if tone not in self.tones:
                self.logger.warning(f"Invalid tone '{tone}'. Using 'professional' instead.")
                tone = "professional"

            # Format product info for the prompt
            product_name = product_info.get("name", "")
            product_features = ", ".join(product_info.get("features", []))
            product_benefits = ", ".join(product_info.get("benefits", []))

            # Create prompt for the AI optimizer
            prompt = f"""
            Generate {num_headlines} compelling Google Ads headlines for the following product/service:
            
            Product/Service: {product_name}
            Features: {product_features}
            Benefits: {product_benefits}
            Target Keywords: {', '.join(keywords[:5])}
            
            Tone: {tone}
            
            Requirements:
            - Each headline must be under {self.max_headline_length} characters
            - Include strong call-to-actions
            - Each headline should be unique
            - Some headlines should include target keywords
            - Focus on benefits rather than features
            - Be specific and quantify value where possible
            
            Format the response as a JSON array of objects with 'headline' and 'rationale' properties.
            """

            # If optimizer is available, use it to generate headlines
            if self.optimizer:
                response = self.optimizer.generate_content(prompt, temperature=0.7)
                headlines = self._extract_headlines_from_response(response)

                # Ensure we have at least one headline
                if not headlines:
                    # Fallback to template-based headlines
                    headlines = self._generate_fallback_headlines(
                        keywords, product_name, num_headlines
                    )
            else:
                # If no optimizer, use fallback method
                headlines = self._generate_fallback_headlines(keywords, product_name, num_headlines)

            # Validate headlines (length, etc.)
            valid_headlines = []
            for headline in headlines:
                text = headline.get("headline", "")
                if text and len(text) <= self.max_headline_length:
                    valid_headlines.append(headline)
                else:
                    self.logger.warning(f"Skipping invalid headline: '{text}'")

            self.logger.info(f"Generated {len(valid_headlines)} valid headlines")
            self._track_execution(start_time, success=True)
            return valid_headlines

        except Exception as e:
            self.logger.error(f"Error generating headlines: {str(e)}")
            self._track_execution(start_time, success=False)
            return []

    def generate_descriptions(
        self,
        keywords: List[str],
        product_info: Dict[str, Any],
        num_descriptions: int = 3,
        tone: str = "professional",
    ) -> List[Dict[str, Any]]:
        """
        Generate ad descriptions based on keywords and product information.

        Args:
            keywords: List of target keywords
            product_info: Dictionary with product/service information
            num_descriptions: Number of descriptions to generate (default 3)
            tone: Tone for the descriptions (default "professional")

        Returns:
            List of description dictionaries
        """
        start_time = datetime.now()
        self.logger.info(f"Generating {num_descriptions} descriptions with '{tone}' tone")

        if not keywords or not product_info:
            self.logger.error("Missing required input for description generation")
            return []

        try:
            # Validate tone
            if tone not in self.tones:
                self.logger.warning(f"Invalid tone '{tone}'. Using 'professional' instead.")
                tone = "professional"

            # Format product info for the prompt
            product_name = product_info.get("name", "")
            product_features = ", ".join(product_info.get("features", []))
            product_benefits = ", ".join(product_info.get("benefits", []))

            # Create prompt for the AI optimizer
            prompt = f"""
            Generate {num_descriptions} compelling Google Ads descriptions for the following product/service:
            
            Product/Service: {product_name}
            Features: {product_features}
            Benefits: {product_benefits}
            Target Keywords: {', '.join(keywords[:5])}
            
            Tone: {tone}
            
            Requirements:
            - Each description must be under {self.max_description_length} characters
            - Include clear call-to-actions
            - Each description should be unique
            - Some descriptions should include target keywords
            - Focus on benefits and value proposition
            - Include social proof or urgency when appropriate
            
            Format the response as a JSON array of objects with 'description' and 'rationale' properties.
            """

            # If optimizer is available, use it to generate descriptions
            if self.optimizer:
                response = self.optimizer.generate_content(prompt, temperature=0.7)
                descriptions = self._extract_descriptions_from_response(response)

                # Ensure we have at least one description
                if not descriptions:
                    # Fallback to template-based descriptions
                    descriptions = self._generate_fallback_descriptions(
                        keywords, product_info, num_descriptions
                    )
            else:
                # If no optimizer, use fallback method
                descriptions = self._generate_fallback_descriptions(
                    keywords, product_info, num_descriptions
                )

            # Validate descriptions (length, etc.)
            valid_descriptions = []
            for description in descriptions:
                text = description.get("description", "")
                if text and len(text) <= self.max_description_length:
                    valid_descriptions.append(description)
                else:
                    self.logger.warning(f"Skipping invalid description: '{text}'")

            self.logger.info(f"Generated {len(valid_descriptions)} valid descriptions")
            self._track_execution(start_time, success=True)
            return valid_descriptions

        except Exception as e:
            self.logger.error(f"Error generating descriptions: {str(e)}")
            self._track_execution(start_time, success=False)
            return []

    def analyze_ad_performance(
        self, ad_performance_data: List[Dict[str, Any]], campaign_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze ad performance and generate insights.

        Args:
            ad_performance_data: List of ad performance data dictionaries
            campaign_id: Optional campaign ID to filter data

        Returns:
            Dictionary with analysis and insights
        """
        start_time = datetime.now()
        self.logger.info(
            f"Analyzing ad performance {f'for campaign {campaign_id}' if campaign_id else ''}"
        )

        if not ad_performance_data:
            self.logger.error("No ad performance data provided for analysis")
            return {"status": "error", "message": "No data provided"}

        try:
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(ad_performance_data)

            # Filter by campaign if specified
            if campaign_id:
                df = df[df["campaign_id"] == campaign_id]

            if df.empty:
                self.logger.warning("No ads found for analysis after filtering")
                return {"status": "error", "message": "No ads found for analysis"}

            # Perform analysis
            analysis = {
                "status": "success",
                "total_ads": len(df),
                "top_performing_headlines": self._analyze_top_elements(df, "headline"),
                "top_performing_descriptions": self._analyze_top_elements(df, "description"),
                "performance_by_length": self._analyze_performance_by_length(df),
                "word_pattern_analysis": self._analyze_word_patterns(df),
                "improvement_suggestions": [],
            }

            # Generate improvement suggestions if optimizer is available
            if self.optimizer and not df.empty:
                suggestions = self._generate_improvement_suggestions(df)
                analysis["improvement_suggestions"] = suggestions

            self.logger.info(f"Completed analysis of {len(df)} ads")
            self._track_execution(start_time, success=True)
            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing ad performance: {str(e)}")
            self._track_execution(start_time, success=False)
            return {"status": "error", "message": str(e)}

    def suggest_responsive_search_ad(
        self,
        keywords: List[str],
        product_info: Dict[str, Any],
        competitor_ads: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete responsive search ad suggestion.

        Args:
            keywords: List of target keywords
            product_info: Dictionary with product/service information
            competitor_ads: Optional list of competitor ads for reference

        Returns:
            Dictionary with complete responsive search ad suggestion
        """
        start_time = datetime.now()
        self.logger.info("Generating responsive search ad suggestion")

        try:
            # Generate headlines
            headlines = self.generate_headlines(keywords, product_info, num_headlines=15)

            # Generate descriptions
            descriptions = self.generate_descriptions(keywords, product_info, num_descriptions=4)

            # Create the full ad suggestion
            ad_suggestion = {
                "headlines": [h.get("headline", "") for h in headlines],
                "descriptions": [d.get("description", "") for d in descriptions],
                "path1": self._generate_display_path(keywords, product_info),
                "path2": self._generate_display_path(keywords, product_info, is_second=True),
                "final_url": product_info.get("url", ""),
                "callout_extensions": self._generate_callout_extensions(product_info),
                "sitelink_extensions": self._generate_sitelink_extensions(product_info),
            }

            self.logger.info(
                f"Generated RSA suggestion with {len(headlines)} headlines and {len(descriptions)} descriptions"
            )
            self._track_execution(start_time, success=True)
            return ad_suggestion

        except Exception as e:
            self.logger.error(f"Error generating responsive search ad: {str(e)}")
            self._track_execution(start_time, success=False)
            return {"status": "error", "message": str(e)}

    # Helper methods

    def _extract_headlines_from_response(self, response: str) -> List[Dict[str, Any]]:
        """Extract headline objects from AI response"""
        try:
            # If the response is JSON formatted, parse it directly
            if response.strip().startswith("[") or response.strip().startswith("{"):
                import json

                data = json.loads(response)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict) and "headlines" in data:
                    return data["headlines"]

            # Otherwise, try to extract using regex
            import re

            headlines = []
            # Match patterns like {"headline": "text", "rationale": "text"}
            pattern = r'{\s*"headline"\s*:\s*"([^"]*)"\s*,\s*"rationale"\s*:\s*"([^"]*)"\s*}'
            matches = re.findall(pattern, response)

            for headline, rationale in matches:
                headlines.append({"headline": headline, "rationale": rationale})

            return headlines

        except Exception as e:
            self.logger.error(f"Error extracting headlines from response: {str(e)}")
            return []

    def _extract_descriptions_from_response(self, response: str) -> List[Dict[str, Any]]:
        """Extract description objects from AI response"""
        try:
            # If the response is JSON formatted, parse it directly
            if response.strip().startswith("[") or response.strip().startswith("{"):
                import json

                data = json.loads(response)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict) and "descriptions" in data:
                    return data["descriptions"]

            # Otherwise, try to extract using regex
            import re

            descriptions = []
            # Match patterns like {"description": "text", "rationale": "text"}
            pattern = r'{\s*"description"\s*:\s*"([^"]*)"\s*,\s*"rationale"\s*:\s*"([^"]*)"\s*}'
            matches = re.findall(pattern, response)

            for description, rationale in matches:
                descriptions.append({"description": description, "rationale": rationale})

            return descriptions

        except Exception as e:
            self.logger.error(f"Error extracting descriptions from response: {str(e)}")
            return []

    def _generate_fallback_headlines(
        self, keywords: List[str], product_name: str, count: int
    ) -> List[Dict[str, Any]]:
        """Generate fallback headlines when AI generation fails"""
        self.logger.info("Using fallback headline generation")

        templates = [
            "Get {product} Today | {benefit}",
            "{product} - {benefit} | Shop Now",
            "Top-Rated {product} | {benefit}",
            "Save on {product} | {benefit}",
            "Professional {product} | {benefit}",
            "{keyword} | {product} Experts",
            "Best {product} for {keyword}",
            "{benefit} with Our {product}",
            "Quality {product} | Free Shipping",
            "{product} Starting at ${price}",
        ]

        headlines = []
        benefits = [
            "Great Value",
            "High Quality",
            "Fast Service",
            "Professional Results",
            "24/7 Support",
        ]
        prices = ["49", "99", "199", "29.99", "59.99"]

        # Generate headlines using templates
        for i in range(min(count, len(templates))):
            template = templates[i % len(templates)]
            keyword = keywords[i % len(keywords)] if keywords else ""
            benefit = benefits[i % len(benefits)]
            price = prices[i % len(prices)]

            headline = (
                template.replace("{product}", product_name[:15])
                .replace("{keyword}", keyword[:15])
                .replace("{benefit}", benefit)
                .replace("{price}", price)
            )

            # Ensure headline is within character limit
            if len(headline) > self.max_headline_length:
                headline = headline[: self.max_headline_length - 3] + "..."

            headlines.append(
                {"headline": headline, "rationale": "Fallback template-based headline"}
            )

        return headlines

    def _generate_fallback_descriptions(
        self, keywords: List[str], product_info: Dict[str, Any], count: int
    ) -> List[Dict[str, Any]]:
        """Generate fallback descriptions when AI generation fails"""
        self.logger.info("Using fallback description generation")

        templates = [
            "{product} provides {benefit}. {feature} with expert support. Call today for a free quote!",
            "Looking for {keyword}? Our {product} offers {benefit}. Visit our website for exclusive deals!",
            "Save time and money with our {product}. {benefit} guaranteed. Order now and get {offer}!",
            "Professional {product} with {feature}. {benefit} for all customers. Limited time offer!",
            "Trusted by thousands, our {product} delivers {benefit}. {feature} included. Shop now!",
        ]

        descriptions = []
        product_name = product_info.get("name", "our product")
        features = product_info.get(
            "features", ["high quality", "expert design", "premium service"]
        )
        benefits = product_info.get(
            "benefits", ["great results", "significant savings", "peace of mind"]
        )
        offers = [
            "free shipping",
            "10% off",
            "a free consultation",
            "next-day delivery",
            "24/7 support",
        ]

        # Generate descriptions using templates
        for i in range(min(count, len(templates))):
            template = templates[i % len(templates)]
            keyword = keywords[i % len(keywords)] if keywords else "quality service"
            feature = features[i % len(features)] if features else "Premium quality"
            benefit = benefits[i % len(benefits)] if benefits else "Great value"
            offer = offers[i % len(offers)]

            description = (
                template.replace("{product}", product_name)
                .replace("{keyword}", keyword)
                .replace("{feature}", feature)
                .replace("{benefit}", benefit)
                .replace("{offer}", offer)
            )

            # Ensure description is within character limit
            if len(description) > self.max_description_length:
                description = description[: self.max_description_length - 3] + "..."

            descriptions.append(
                {"description": description, "rationale": "Fallback template-based description"}
            )

        return descriptions

    def _analyze_top_elements(self, df: pd.DataFrame, element_type: str) -> List[Dict[str, Any]]:
        """Analyze top performing headlines or descriptions"""
        # Placeholder implementation
        return []

    def _analyze_performance_by_length(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how length affects performance"""
        # Placeholder implementation
        return {}

    def _analyze_word_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze word patterns that correlate with performance"""
        # Placeholder implementation
        return {}

    def _generate_improvement_suggestions(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate suggestions to improve ad content"""
        # Placeholder implementation
        return []

    def _generate_display_path(
        self, keywords: List[str], product_info: Dict[str, Any], is_second: bool = False
    ) -> str:
        """Generate a display path segment"""
        # Simple implementation that selects a keyword or product category
        if is_second:
            if "category" in product_info:
                return product_info["category"].lower().replace(" ", "-")[:15]
            else:
                return "products"
        else:
            if keywords and len(keywords) > 0:
                return keywords[0].lower().replace(" ", "-")[:15]
            elif "name" in product_info:
                return product_info["name"].lower().replace(" ", "-")[:15]
            else:
                return "services"

    def _generate_callout_extensions(self, product_info: Dict[str, Any]) -> List[str]:
        """Generate callout extensions based on product info"""
        callouts = ["Free Shipping", "24/7 Customer Service", "No Hidden Fees"]

        if "benefits" in product_info and product_info["benefits"]:
            for benefit in product_info["benefits"][:3]:
                if len(benefit) <= 25:  # Character limit for callouts
                    callouts.append(benefit)

        return callouts[:10]  # Google allows up to 20 callouts, but we'll limit to 10

    def _generate_sitelink_extensions(self, product_info: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate sitelink extensions based on product info"""
        # Basic sitelinks that apply to most businesses
        sitelinks = [
            {"text": "About Us", "url": "/about"},
            {"text": "Contact", "url": "/contact"},
            {"text": "Services", "url": "/services"},
        ]

        # Add product-specific sitelinks if available
        if "products" in product_info and product_info["products"]:
            for product in product_info["products"][:2]:
                if "name" in product and "url" in product:
                    sitelinks.append(
                        {
                            "text": product["name"][:25],  # Character limit for sitelink text
                            "url": product["url"],
                        }
                    )

        return sitelinks[:6]  # Google allows up to 8 sitelinks per ad, but we'll limit to 6

    def run(self, **kwargs):
        """
        Run the generative content service with provided parameters.

        Args:
            **kwargs: Keyword arguments including:
                - keywords: List of target keywords
                - product_info: Product/service information
                - action: Action to perform (e.g., "generate_headlines", "generate_descriptions", etc.)
                - num_items: Number of items to generate
                - tone: Tone for the generated content

        Returns:
            Results of the requested action
        """
        action = kwargs.get("action", "")
        self.logger.info(f"GenerativeContentService run called with action: {action}")

        if action == "generate_headlines":
            return self.generate_headlines(
                kwargs.get("keywords", []),
                kwargs.get("product_info", {}),
                kwargs.get("num_items", 5),
                kwargs.get("tone", "professional"),
            )
        elif action == "generate_descriptions":
            return self.generate_descriptions(
                kwargs.get("keywords", []),
                kwargs.get("product_info", {}),
                kwargs.get("num_items", 3),
                kwargs.get("tone", "professional"),
            )
        elif action == "suggest_responsive_search_ad":
            return self.suggest_responsive_search_ad(
                kwargs.get("keywords", []),
                kwargs.get("product_info", {}),
                kwargs.get("competitor_ads", []),
            )
        elif action == "analyze_ad_performance":
            return self.analyze_ad_performance(
                kwargs.get("ad_performance_data", []), kwargs.get("campaign_id", None)
            )
        else:
            self.logger.warning(f"Unknown action: {action}")
            return {"status": "error", "message": f"Unknown action: {action}"}
