import google.generativeai as genai
import re
import json
import logging
from datetime import datetime
import requests
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)


class AdsOptimizer:
    """Google Ads Account Optimizer that uses Gemini to generate optimization suggestions."""

    def __init__(self, api_key: str):
        """
        Initialize the AdsOptimizer with the Google AI API key.

        Args:
            api_key: The Google AI API key for Gemini
        """
        # Ensure api_key is a string, not a dictionary
        if isinstance(api_key, dict) and "api_key" in api_key:
            self.api_key = api_key["api_key"]
        else:
            self.api_key = api_key

        # Check that the API key is valid
        if not self.api_key or len(str(self.api_key).strip()) < 10:
            error_msg = "Invalid API key format. Please provide a valid Google AI API key."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Initializing Google AI Optimizer with Gemini models")
        self.use_direct_api = True  # Default to direct API which is more reliable

        try:
            # Test direct API with a minimal request to validate the API key
            test_prompt = "Hello, this is a test."
            response = self._generate_with_direct_api(test_prompt, temperature=0.1)
            if response:
                logger.info("Successfully connected to Gemini API")

            # Initialize SDK as well but keep using direct API as primary
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
                self.gemini_model = self.model  # For compatibility with chat interface
                logger.info("Also initialized Gemini SDK as fallback")
            except Exception as sdk_error:
                logger.warning(f"Failed to initialize Gemini SDK: {str(sdk_error)}")
                logger.info("Will use direct API only")

        except Exception as e:
            logger.error(f"Failed to connect to Gemini API: {str(e)}")
            logger.warning("Will attempt to use the API when needed, but may fail")
            # Don't raise here - let the system try to continue and fail at generation time if needed

    def _generate_with_sdk(self, prompt: str, temperature: float = 0.2) -> str:
        """
        Generate content using the Google GenerativeAI SDK.

        Args:
            prompt: The prompt to send to the model
            temperature: The temperature for generation (0.0 to 1.0)

        Returns:
            The generated text as a string
        """
        try:
            # Already configured in __init__, but configure again to be safe
            genai.configure(api_key=self.api_key)

            # Updated model names to match the latest available versions
            model_names = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro-vision"]

            last_error = None
            for model_name in model_names:
                try:
                    logger.info(f"Attempting to use Gemini model: {model_name}")
                    model = genai.GenerativeModel(model_name)

                    # Set generation config with improved settings
                    generation_config = {
                        "temperature": temperature,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": 2048,
                    }

                    # Set safety settings
                    safety_settings = [
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_ONLY_HIGH",
                        }
                    ]

                    # Generate content with improved error handling
                    response = model.generate_content(
                        prompt, generation_config=generation_config, safety_settings=safety_settings
                    )

                    # Extract the text from the response with better error handling
                    if hasattr(response, "text"):
                        logger.info(f"Successfully generated content using {model_name}")
                        return response.text
                    elif hasattr(response, "parts"):
                        logger.info(f"Successfully generated content using {model_name}")
                        return " ".join([part.text for part in response.parts])
                    elif hasattr(response, "candidates") and response.candidates:
                        # Access potential candidates structure
                        text_parts = []
                        for candidate in response.candidates:
                            if hasattr(candidate, "content") and hasattr(
                                candidate.content, "parts"
                            ):
                                for part in candidate.content.parts:
                                    if hasattr(part, "text"):
                                        text_parts.append(part.text)
                        if text_parts:
                            return " ".join(text_parts)

                    logger.error(f"Unexpected response format: {response}")
                    raise Exception("Unexpected response format from Gemini API")

                except Exception as e:
                    last_error = e
                    logger.warning(f"Failed to use model {model_name}: {str(e)}")
                    continue  # Try the next model

            # If we get here, all models failed
            raise last_error or Exception("All Gemini models failed")

        except Exception as e:
            error_message = str(e)

            # Check for specific error types and provide clearer error messages
            if "404" in error_message:
                if "models/" in error_message:
                    model_name = (
                        error_message.split("models/")[1].split(" ")[0]
                        if "models/" in error_message
                        else "unknown"
                    )
                    logger.error(f"Model not found: {model_name}")
                    raise Exception(
                        f"Model '{model_name}' not supported by the current API version. Try updating to the latest SDK version."
                    )
                else:
                    logger.error(f"404 error: {error_message}")
                    raise Exception(f"Resource not found error: {error_message}")
            elif "403" in error_message:
                logger.error(f"Authorization error: {error_message}")
                raise Exception(
                    f"Authorization error. Please check your API key and permissions: {error_message}"
                )
            else:
                logger.error(f"SDK generation error: {error_message}")
                raise Exception(f"Error generating content with SDK: {error_message}")

    def _generate_with_direct_api(self, prompt: str, temperature: float = 0.2) -> str:
        """
        Generate content using the direct Gemini API.

        Args:
            prompt: The prompt to send to the model
            temperature: The temperature for generation (0.0 to 1.0)

        Returns:
            The generated text as a string
        """
        try:
            # Updated model names to use the latest available Gemini models
            model_names = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro-vision"]

            last_error = None
            for model_name in model_names:
                try:
                    logger.info(f"Attempting to use direct API with model: {model_name}")

                    # Updated API endpoint using latest API version, using generativelanguage API
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"

                    # Request payload
                    payload = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": temperature,
                            "topK": 40,
                            "topP": 0.95,
                            "maxOutputTokens": 2048,
                        },
                        "safetySettings": [
                            {
                                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                                "threshold": "BLOCK_ONLY_HIGH",
                            }
                        ],
                    }

                    # Log request details for debugging (excluding prompt for privacy)
                    logger.debug(f"Making request to {url} for model {model_name}")

                    # Make the request
                    response = requests.post(url, json=payload)

                    # Add better error handling
                    if response.status_code != 200:
                        error_data = (
                            response.json()
                            if response.text
                            else {"error": {"message": "Unknown error"}}
                        )
                        error_message = error_data.get("error", {}).get(
                            "message", "Unknown API error"
                        )
                        logger.error(
                            f"Gemini API error with {model_name}: {response.status_code} - {error_message}"
                        )
                        raise Exception(
                            f"Gemini API error: {response.status_code} - {error_message}"
                        )

                    # Parse the response
                    response_json = response.json()
                    logger.debug(f"Received response from {model_name}: {response.status_code}")

                    # Extract the generated text with more robust parsing
                    generated_text = ""
                    if "candidates" in response_json and len(response_json["candidates"]) > 0:
                        candidate = response_json["candidates"][0]

                        # Check if content exists
                        if "content" in candidate:
                            content = candidate["content"]

                            # Extract text from parts
                            if "parts" in content:
                                for part in content["parts"]:
                                    if "text" in part:
                                        generated_text += part["text"]

                    if not generated_text:
                        raise Exception(f"No text generated from {model_name} model")

                    logger.info(
                        f"Successfully generated content using direct API with {model_name}"
                    )
                    return generated_text

                except Exception as e:
                    last_error = e
                    logger.warning(f"Direct API failed with model {model_name}: {str(e)}")
                    continue  # Try the next model

            # If we get here, all models failed
            raise last_error or Exception("All Gemini models failed with direct API")

        except Exception as e:
            logger.error(f"Error generating content with direct API: {str(e)}")
            raise

    def generate_content(self, prompt: str, temperature: float = 0.2) -> str:
        """
        Generate content using either the SDK or direct API.

        Args:
            prompt: The prompt to send to the model
            temperature: The temperature for generation (0.0 to 1.0)

        Returns:
            The generated text as a string
        """
        if self.use_direct_api:
            return self._generate_with_direct_api(prompt, temperature)

        try:
            # First try with SDK
            return self._generate_with_sdk(prompt, temperature)
        except Exception as sdk_error:
            logger.warning(f"SDK generation failed: {str(sdk_error)}")
            logger.info("Falling back to direct API")

            try:
                # Fall back to direct API
                return self._generate_with_direct_api(prompt, temperature)
            except Exception as api_error:
                # If both methods fail, log error and raise exception
                logger.error(f"Both SDK and direct API failed: {str(api_error)}")
                error_msg = str(api_error)

                if "API key not valid" in error_msg or "Invalid API key" in error_msg:
                    raise Exception("Invalid API key. Please check your Google AI API key.")
                elif "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                    raise Exception("API quota exceeded. Please try again later.")
                else:
                    raise Exception(f"Failed to generate content: {str(api_error)}")

    def _parse_json_from_response(self, response_text: str) -> Dict:
        """
        Parse JSON from the model's response, handling various edge cases.

        Args:
            response_text: The raw text response from the model

        Returns:
            Parsed JSON as a dictionary
        """
        try:
            # Extract JSON from response (sometimes the model adds extra text)
            json_str = response_text

            # Try to find JSON-like content within fenced code blocks
            import re

            json_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
            if json_blocks:
                json_str = json_blocks[0]

            # Clean up any non-JSON text that might be present
            if not json_str.strip().startswith("{") and not json_str.strip().startswith("["):
                # Find the first { or [ character
                start_idx = json_str.find("{")
                if start_idx == -1 or json_str.find("[") != -1 and json_str.find("[") < start_idx:
                    start_idx = json_str.find("[")

                if start_idx != -1:
                    json_str = json_str[start_idx:]

            # Find the last } or ] character
            if json_str.strip().startswith("{"):
                end_idx = json_str.rfind("}")
                if end_idx != -1:
                    json_str = json_str[: end_idx + 1]
            elif json_str.strip().startswith("["):
                end_idx = json_str.rfind("]")
                if end_idx != -1:
                    json_str = json_str[: end_idx + 1]

            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {str(e)}")
            logger.error(f"Response text: {response_text}")
            raise ValueError(f"Failed to parse JSON from model response: {str(e)}")

    def format_campaign_data(self, campaigns: List[Dict[str, Any]]) -> str:
        """
        Format campaign data into a structured string for the AI model.

        Args:
            campaigns: List of campaign data dictionaries

        Returns:
            Formatted string with campaign information
        """
        result = "CAMPAIGN PERFORMANCE DATA:\n\n"

        for idx, campaign in enumerate(campaigns, 1):
            result += f"Campaign {idx}: {campaign.get('name', 'Unnamed')}\n"
            result += f"Status: {campaign.get('status', 'UNKNOWN')}\n"
            result += f"Budget: {campaign.get('budget_amount', 0)} {campaign.get('budget_currency', 'USD')}/day\n"

            # Performance metrics
            result += "Performance (last 30 days):\n"
            result += f"- Impressions: {campaign.get('impressions', 0)}\n"
            result += f"- Clicks: {campaign.get('clicks', 0)}\n"
            result += f"- CTR: {campaign.get('ctr', 0):.2%}\n"
            result += f"- Conversions: {campaign.get('conversions', 0)}\n"
            result += f"- Conversion Rate: {campaign.get('conversion_rate', 0):.2%}\n"
            result += f"- Cost: {campaign.get('cost', 0)} {campaign.get('currency', 'USD')}\n"
            result += f"- Average CPC: {campaign.get('average_cpc', 0)} {campaign.get('currency', 'USD')}\n"
            result += f"- ROAS: {campaign.get('roas', 0):.2f}\n\n"

        return result

    def format_keyword_data(
        self,
        keywords: List[Dict[str, Any]],
        limit: int = 100,
        include_best: bool = True,
        include_worst: bool = True,
    ) -> str:
        """
        Format keyword data into a structured string for the AI model.

        Args:
            keywords: List of keyword data dictionaries
            limit: Maximum number of keywords to include
            include_best: Whether to include the best performing keywords
            include_worst: Whether to include the worst performing keywords

        Returns:
            Formatted string with keyword information
        """
        # Check if we have any keywords to process
        if not keywords:
            return "No keyword data available for analysis."

        # Double check that we only analyze active keywords
        # These should already be filtered by the API query, but we do it again for safety
        active_keywords = [
            kw
            for kw in keywords
            if (
                kw.get("status") == "ENABLED"
                and kw.get("ad_group_status") == "ENABLED"
                and kw.get("campaign_status") == "ENABLED"
            )
        ]

        logger.info(
            f"Formatting keyword data: {len(keywords)} total keywords, {len(active_keywords)} active keywords"
        )

        # If no active keywords, return early with information
        if not active_keywords:
            return (
                f"No active keywords found for analysis. Total keywords received: {len(keywords)}."
            )

        # Filter keywords with at least some impressions
        keywords_with_data = [kw for kw in active_keywords if kw.get("impressions", 0) > 10]

        if not keywords_with_data:
            # Fall back to any keywords with impressions
            keywords_with_data = [kw for kw in active_keywords if kw.get("impressions", 0) > 0]
            logger.info(
                f"Few keywords with significant data: falling back to {len(keywords_with_data)} keywords with any impressions"
            )

            # If still no data, just use active keywords
            if not keywords_with_data:
                keywords_with_data = active_keywords
                logger.info(
                    f"No keywords with impressions. Using all {len(active_keywords)} active keywords."
                )

        # Calculate CTR for sorting since we might not have it directly
        for kw in keywords_with_data:
            impressions = kw.get("impressions", 0)
            clicks = kw.get("clicks", 0)
            if impressions > 0:
                kw["ctr"] = clicks / impressions
            else:
                kw["ctr"] = 0

        # Sort by CTR for now, but could be made more sophisticated
        best_keywords = sorted(keywords_with_data, key=lambda x: x.get("ctr", 0), reverse=True)[
            : limit // 2
        ]

        worst_keywords = sorted(keywords_with_data, key=lambda x: x.get("ctr", 0))[: limit // 2]

        # Start building the output
        result = "KEYWORD PERFORMANCE DATA:\n\n"

        # Add overall statistics
        result += f"Total Keywords Analyzed: {len(keywords)}\n"
        result += f"Active Keywords: {len(active_keywords)}\n"
        result += f"Keywords With Significant Data: {len(keywords_with_data)}\n\n"

        # Best performing keywords
        if include_best and best_keywords:
            result += "TOP PERFORMING KEYWORDS (by CTR):\n"
            for idx, kw in enumerate(best_keywords, 1):
                result += (
                    f"{idx}. '{kw.get('keyword_text', '')}' [{kw.get('match_type', 'Unknown')}]\n"
                )
                result += f"   Campaign: {kw.get('campaign_name', 'Unknown')}\n"
                result += f"   Ad Group: {kw.get('ad_group_name', 'Unknown')}\n"
                result += f"   Impressions: {kw.get('impressions', 0)}\n"
                result += f"   Clicks: {kw.get('clicks', 0)}\n"
                result += f"   CTR: {kw.get('ctr', 0):.2%}\n"
                result += f"   Conversions: {kw.get('conversions', 0)}\n"
                result += f"   Cost: {kw.get('cost', 0)} {kw.get('currency', 'USD')}\n"
                result += f"   Current Bid: ${kw.get('current_bid', 0):.2f}\n\n"

        # Worst performing keywords
        if include_worst and worst_keywords:
            result += "LOWEST PERFORMING KEYWORDS (by CTR):\n"
            for idx, kw in enumerate(worst_keywords, 1):
                result += (
                    f"{idx}. '{kw.get('keyword_text', '')}' [{kw.get('match_type', 'Unknown')}]\n"
                )
                result += f"   Campaign: {kw.get('campaign_name', 'Unknown')}\n"
                result += f"   Ad Group: {kw.get('ad_group_name', 'Unknown')}\n"
                result += f"   Impressions: {kw.get('impressions', 0)}\n"
                result += f"   Clicks: {kw.get('clicks', 0)}\n"
                result += f"   CTR: {kw.get('ctr', 0):.2%}\n"
                result += f"   Conversions: {kw.get('conversions', 0)}\n"
                result += f"   Cost: {kw.get('cost', 0)} {kw.get('currency', 'USD')}\n"
                result += f"   Current Bid: ${kw.get('current_bid', 0):.2f}\n\n"

        return result

    def get_keyword_suggestions(
        self, business_info: Dict[str, Any], existing_keywords: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate keyword suggestions for the business.

        Args:
            business_info: Dictionary with business information
            existing_keywords: List of existing keywords to avoid duplicates

        Returns:
            Dictionary with suggested keywords and rationale
        """
        # Format business information
        business_prompt = f"""
        Business Name: {business_info.get('name', 'Not provided')}
        Industry: {business_info.get('industry', 'Not provided')}
        Products/Services: {', '.join(business_info.get('products_services', ['Not provided']))}
        Target Audience: {business_info.get('target_audience', 'Not provided')}
        Geographic Focus: {business_info.get('geographic_focus', 'Not provided')}
        Key Value Propositions: {', '.join(business_info.get('value_props', ['Not provided']))}
        """

        # Format existing keywords to avoid duplicates
        existing_kw_text = [kw.get("keyword_text", "").lower() for kw in existing_keywords]
        existing_kw_sample = ", ".join(existing_kw_text[:50]) if existing_kw_text else "None"

        prompt = f"""
        You are an expert Google Ads keyword researcher. Based on the following business information,
        suggest new keywords that would help this business reach their target audience effectively.
        
        {business_prompt}
        
        The business already uses these keywords (sample): {existing_kw_sample}
        
        Please provide 20 new recommended keywords in the following JSON format:
        {{
            "keyword_suggestions": [
                {{
                    "keyword": "keyword text",
                    "match_type": "BROAD|PHRASE|EXACT",
                    "rationale": "Brief explanation why this keyword is relevant",
                    "estimated_search_volume": "LOW|MEDIUM|HIGH",
                    "suggested_bid_range": "Approximate bid range in USD"
                }}
            ],
            "ad_group_recommendations": [
                {{
                    "name": "Suggested Ad Group Name",
                    "theme": "Theme of this ad group",
                    "keywords": ["keyword1", "keyword2", "keyword3"]
                }}
            ],
            "overall_strategy": "Brief paragraph on overall keyword strategy"
        }}
        
        Provide thoughtful recommendations that:
        1. Target different stages of the buying funnel
        2. Include a mix of commercial and informational keywords
        3. Incorporate the business's unique selling propositions
        4. Consider local intent if the business has a geographic focus
        
        Generate your response as parseable JSON without any additional text.
        """

        # Get suggestions from the model
        response_text = self.generate_content(prompt, temperature=0.2)

        # Parse the JSON response
        suggestions = self._parse_json_from_response(response_text)

        return suggestions

    def get_optimization_suggestions(
        self,
        campaign_data: List[Dict[str, Any]],
        keyword_data: List[Dict[str, Any]],
        business_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get optimization suggestions for the Google Ads account.

        Args:
            campaign_data: List of campaign performance data
            keyword_data: List of keyword performance data
            business_info: Optional dictionary with business information

        Returns:
            Dictionary with optimization suggestions
        """
        # Format data for the prompt
        campaign_text = self.format_campaign_data(campaign_data)
        keyword_text = self.format_keyword_data(keyword_data)

        # Format business info if provided
        business_text = ""
        if business_info:
            business_text = f"""
            BUSINESS INFORMATION:
            Business Name: {business_info.get('name', 'Not provided')}
            Industry: {business_info.get('industry', 'Not provided')}
            Products/Services: {', '.join(business_info.get('products_services', ['Not provided']))}
            Target Audience: {business_info.get('target_audience', 'Not provided')}
            Geographic Focus: {business_info.get('geographic_focus', 'Not provided')}
            """

        prompt = f"""
        You are an expert Google Ads consultant analyzing the following account data.
        Provide detailed optimization recommendations to improve performance.
        
        {business_text}
        
        {campaign_text}
        
        {keyword_text}
        
        Based on this data, provide comprehensive optimization recommendations in the following JSON format:
        {{
            "campaign_recommendations": [
                {{
                    "campaign_name": "Affected campaign name or 'All Campaigns'",
                    "issue": "Clear description of the issue",
                    "recommendation": "Specific action to take",
                    "expected_impact": "Expected result of implementing this change",
                    "priority": "HIGH|MEDIUM|LOW"
                }}
            ],
            "keyword_recommendations": [
                {{
                    "type": "ADD|PAUSE|ADJUST_BID|CHANGE_MATCH_TYPE",
                    "keyword": "Affected keyword text or new keyword to add",
                    "current_bid": "Current bid amount in USD (for ADJUST_BID type)",
                    "recommended_bid": "Suggested new bid amount in USD (for ADJUST_BID type)",
                    "rationale": "Why this change is recommended",
                    "details": "Specific details about the recommendation",
                    "priority": "HIGH|MEDIUM|LOW" 
                }}
            ],
            "ad_copy_recommendations": [
                {{
                    "target": "Specific ad group or 'General'",
                    "recommendation": "Description of the recommendation",
                    "suggested_elements": {{
                        "headlines": ["Headline 1", "Headline 2", "Headline 3"],
                        "descriptions": ["Description 1", "Description 2"],
                        "call_to_action": "Suggested CTA"
                    }},
                    "rationale": "Why these changes would improve performance"
                }}
            ],
            "bid_strategy_recommendations": {{
                "current_issues": "Issues with the current bidding approach",
                "recommended_strategy": "Recommended bid strategy",
                "implementation_steps": ["Step 1", "Step 2", "Step 3"],
                "expected_outcome": "Expected result of these changes"
            }},
            "budget_recommendations": {{
                "campaigns_needing_adjustment": ["Campaign 1", "Campaign 2"],
                "recommended_changes": "Specific budget recommendations",
                "rationale": "Why these changes would improve performance" 
            }},
            "overall_assessment": "Brief paragraph summarizing the account's performance and top priorities"
        }}
        
        IMPORTANT REQUIREMENTS:
        1. Focus heavily on providing detailed bid adjustment recommendations for keywords with data
        2. For each ADJUST_BID recommendation, ALWAYS include both current_bid and recommended_bid fields with specific dollar values
        3. Include at least 5 specific bid adjustment recommendations if any keywords have performance data
        4. Base bid recommendations on performance metrics like CTR, conversion rate, and cost per conversion
        5. Provide actionable, specific recommendations based solely on the data provided
        
        Generate your response as parseable JSON without any additional text.
        """

        logger.info(f"Sending optimization request to Gemini with {len(keyword_data)} keywords")

        # Get suggestions from the model
        response_text = self.generate_content(prompt, temperature=0.2)

        # Parse the JSON response
        try:
            suggestions = self._parse_json_from_response(response_text)
            logger.info(
                f"Successfully parsed optimization suggestions with {len(suggestions.get('keyword_recommendations', []))} keyword recommendations"
            )
            return suggestions
        except Exception as e:
            logger.error(f"Error parsing optimization suggestions: {str(e)}")
            error_message = str(e)
            # Return an error suggestion
            return {
                "keyword_recommendations": [
                    {
                        "type": "ERROR",
                        "keyword": "Error Generating Suggestions",
                        "rationale": f"Failed to generate optimization suggestions: {error_message[:100]}...",
                        "details": "Please try again or contact support if the problem persists.",
                        "priority": "HIGH",
                    }
                ],
                "overall_assessment": f"Error occurred while generating suggestions: {error_message[:200]}",
            }
