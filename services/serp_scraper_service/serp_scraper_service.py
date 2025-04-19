"""
SERP Scraper Service

This service provides functionality for scraping and analyzing search engine results pages
to gather data on ad positions, organic rankings, competitor ads, and more.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from services.base_service import BaseService


@dataclass
class SERPResult:
    """Data class for storing SERP results"""
    query: str
    timestamp: str
    ads_top: List[Dict[str, Any]]
    ads_bottom: List[Dict[str, Any]]
    organic_results: List[Dict[str, Any]]
    related_searches: List[str]
    knowledge_panel: Optional[Dict[str, Any]] = None
    local_pack: Optional[List[Dict[str, Any]]] = None
    shopping_results: Optional[List[Dict[str, Any]]] = None


class SERPScraperService(BaseService):
    """
    Service for scraping and analyzing search engine results pages.
    
    This service uses Selenium WebDriver to scrape Google search results
    and extract data about ads, organic results, and other SERP features.
    """
    
    def __init__(
        self,
        ads_api=None,
        optimizer=None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
        webdriver_path: Optional[str] = None,
    ):
        """
        Initialize the SERP Scraper Service.
        
        Args:
            ads_api: Google Ads API client instance
            optimizer: AI optimizer instance
            config: Configuration dictionary
            logger: Logger instance
            webdriver_path: Path to the Chrome WebDriver executable
        """
        super().__init__(ads_api, optimizer, config, logger)
        
        self.webdriver_path = webdriver_path or self.config.get("webdriver_path")
        self.driver = None
        self.use_proxy = self.config.get("use_proxy", False)
        self.proxy_config = self.config.get("proxy_config", {})
        self.user_agent = self.config.get("user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        self.results_dir = os.path.join("data", "serp_results")
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.logger.info("SERP Scraper Service initialized")
    
    def _initialize_driver(self) -> None:
        """Initialize the Selenium WebDriver with appropriate options"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument(f"user-agent={self.user_agent}")
            
            if self.use_proxy and self.proxy_config:
                proxy = f"{self.proxy_config['host']}:{self.proxy_config['port']}"
                chrome_options.add_argument(f"--proxy-server={proxy}")
            
            service = Service(executable_path=self.webdriver_path)
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.set_window_size(1920, 1080)
            
            self.logger.info("WebDriver initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing WebDriver: {str(e)}")
            raise
    
    def _close_driver(self) -> None:
        """Close the WebDriver if it's open"""
        if self.driver:
            self.driver.quit()
            self.driver = None
            self.logger.info("WebDriver closed")
    
    def scrape_serp(
        self, query: str, location: Optional[str] = None, 
        language: str = "en", device: str = "desktop"
    ) -> SERPResult:
        """
        Scrape a Google search results page for the given query.
        
        Args:
            query: Search query to scrape
            location: Optional location to use for the search
            language: Language code for the search
            device: Device type to emulate ('desktop' or 'mobile')
            
        Returns:
            SERPResult object containing the scraped data
        """
        start_time = datetime.now()
        
        try:
            if not self.driver:
                self._initialize_driver()
            
            # Set device emulation if mobile
            if device.lower() == "mobile":
                mobile_emulation = {
                    "deviceMetrics": {"width": 360, "height": 740, "pixelRatio": 3.0},
                    "userAgent": "Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36"
                }
                self.driver.execute_cdp_cmd("Emulation.setDeviceMetricsOverride", mobile_emulation)
            
            # Construct search URL
            url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            if location:
                url += f"&near={location.replace(' ', '+')}"
            
            # Navigate to the URL
            self.logger.info(f"Navigating to URL: {url}")
            self.driver.get(url)
            
            # Wait for results to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "search"))
            )
            
            # Add a short delay to ensure everything loads
            time.sleep(2)
            
            # Extract data from the page
            timestamp = datetime.now().isoformat()
            
            # Extract ad results at the top
            ads_top = self._extract_ads("top")
            
            # Extract organic results
            organic_results = self._extract_organic_results()
            
            # Extract ad results at the bottom
            ads_bottom = self._extract_ads("bottom")
            
            # Extract related searches
            related_searches = self._extract_related_searches()
            
            # Extract knowledge panel if present
            knowledge_panel = self._extract_knowledge_panel()
            
            # Extract local pack if present
            local_pack = self._extract_local_pack()
            
            # Extract shopping results if present
            shopping_results = self._extract_shopping_results()
            
            # Create the result object
            result = SERPResult(
                query=query,
                timestamp=timestamp,
                ads_top=ads_top,
                ads_bottom=ads_bottom,
                organic_results=organic_results,
                related_searches=related_searches,
                knowledge_panel=knowledge_panel,
                local_pack=local_pack,
                shopping_results=shopping_results
            )
            
            # Save the result to a file
            self._save_serp_result(result)
            
            self._track_execution(start_time, True)
            return result
            
        except Exception as e:
            self.logger.error(f"Error scraping SERP for query '{query}': {str(e)}")
            self._track_execution(start_time, False)
            raise
    
    def _extract_ads(self, position: str) -> List[Dict[str, Any]]:
        """
        Extract ad results from the SERP.
        
        Args:
            position: Position of ads to extract ('top' or 'bottom')
            
        Returns:
            List of ad dictionaries
        """
        ads = []
        
        try:
            # Adjust selector based on position
            if position == "top":
                ad_container = self.driver.find_element(By.CSS_SELECTOR, "div[id='tads']")
            else:  # bottom
                ad_container = self.driver.find_element(By.CSS_SELECTOR, "div[id='bottomads']")
            
            ad_elements = ad_container.find_elements(By.CSS_SELECTOR, "div.uEierd")
            
            for ad_element in ad_elements:
                try:
                    ad_data = {
                        "title": ad_element.find_element(By.CSS_SELECTOR, "div.CCgQ5").text,
                        "display_url": ad_element.find_element(By.CSS_SELECTOR, "span.x2VHCd").text,
                        "description": ad_element.find_element(By.CSS_SELECTOR, "div.MUxGbd").text,
                        "position": position,
                        "extensions": []
                    }
                    
                    # Extract ad extensions
                    try:
                        extension_elements = ad_element.find_elements(By.CSS_SELECTOR, "div.MUxGbd.lyLwlc")
                        for ext in extension_elements:
                            ad_data["extensions"].append(ext.text)
                    except:
                        pass
                    
                    ads.append(ad_data)
                except Exception as e:
                    self.logger.warning(f"Error extracting ad data: {str(e)}")
            
        except Exception as e:
            self.logger.warning(f"No ads found at {position}: {str(e)}")
        
        return ads
    
    def _extract_organic_results(self) -> List[Dict[str, Any]]:
        """
        Extract organic search results from the SERP.
        
        Returns:
            List of organic result dictionaries
        """
        organic_results = []
        
        try:
            organic_elements = self.driver.find_elements(By.CSS_SELECTOR, "div.g")
            
            for i, org_element in enumerate(organic_elements):
                try:
                    result_data = {
                        "position": i + 1,
                        "title": org_element.find_element(By.CSS_SELECTOR, "h3").text,
                        "url": org_element.find_element(By.CSS_SELECTOR, "a").get_attribute("href"),
                        "description": org_element.find_element(By.CSS_SELECTOR, "div.VwiC3b").text
                    }
                    
                    # Extract rich snippets if present
                    try:
                        snippet_elements = org_element.find_elements(By.CSS_SELECTOR, "div.UDZeY")
                        if snippet_elements:
                            result_data["rich_snippet"] = snippet_elements[0].text
                    except:
                        pass
                    
                    organic_results.append(result_data)
                except Exception as e:
                    self.logger.warning(f"Error extracting organic result data: {str(e)}")
            
        except Exception as e:
            self.logger.warning(f"Error extracting organic results: {str(e)}")
        
        return organic_results
    
    def _extract_related_searches(self) -> List[str]:
        """
        Extract related searches from the SERP.
        
        Returns:
            List of related search queries
        """
        related_searches = []
        
        try:
            related_elements = self.driver.find_elements(By.CSS_SELECTOR, "div.card-section a")
            
            for element in related_elements:
                related_searches.append(element.text)
            
        except Exception as e:
            self.logger.warning(f"Error extracting related searches: {str(e)}")
        
        return related_searches
    
    def _extract_knowledge_panel(self) -> Optional[Dict[str, Any]]:
        """
        Extract knowledge panel data if present.
        
        Returns:
            Dictionary with knowledge panel data or None if not present
        """
        try:
            panel = self.driver.find_element(By.CSS_SELECTOR, "div.kp-wholepage")
            
            title = panel.find_element(By.CSS_SELECTOR, "h2").text
            
            panel_data = {
                "title": title,
                "attributes": {}
            }
            
            # Extract attributes
            attr_elements = panel.find_elements(By.CSS_SELECTOR, "div.rVusze")
            for attr in attr_elements:
                try:
                    key = attr.find_element(By.CSS_SELECTOR, "span.w8qArf").text
                    value = attr.find_element(By.CSS_SELECTOR, "span.kno-fv").text
                    panel_data["attributes"][key] = value
                except:
                    pass
            
            return panel_data
            
        except Exception as e:
            self.logger.debug(f"No knowledge panel found: {str(e)}")
            return None
    
    def _extract_local_pack(self) -> Optional[List[Dict[str, Any]]]:
        """
        Extract local pack results if present.
        
        Returns:
            List of local pack results or None if not present
        """
        try:
            local_pack = self.driver.find_element(By.CSS_SELECTOR, "div.BNeawe")
            local_results = []
            
            local_elements = local_pack.find_elements(By.CSS_SELECTOR, "div.rllt__details")
            
            for element in local_elements:
                try:
                    result = {
                        "name": element.find_element(By.CSS_SELECTOR, "div.dbg0pd").text,
                        "rating": element.find_element(By.CSS_SELECTOR, "span.yi40Hd").text,
                        "type": element.find_element(By.CSS_SELECTOR, "div.rllt__details div:nth-child(2)").text,
                        "address": element.find_element(By.CSS_SELECTOR, "div.rllt__details div:nth-child(3)").text,
                    }
                    local_results.append(result)
                except:
                    pass
            
            return local_results
            
        except Exception as e:
            self.logger.debug(f"No local pack found: {str(e)}")
            return None
    
    def _extract_shopping_results(self) -> Optional[List[Dict[str, Any]]]:
        """
        Extract shopping results if present.
        
        Returns:
            List of shopping results or None if not present
        """
        try:
            shopping_container = self.driver.find_element(By.CSS_SELECTOR, "div.commercial-unit-desktop-top")
            shopping_results = []
            
            product_elements = shopping_container.find_elements(By.CSS_SELECTOR, "div.pla-unit")
            
            for element in product_elements:
                try:
                    result = {
                        "title": element.find_element(By.CSS_SELECTOR, "div.pla-unit-title").text,
                        "price": element.find_element(By.CSS_SELECTOR, "div.pla-unit-price").text,
                        "vendor": element.find_element(By.CSS_SELECTOR, "div.pla-extensions-container").text,
                    }
                    shopping_results.append(result)
                except:
                    pass
            
            return shopping_results
            
        except Exception as e:
            self.logger.debug(f"No shopping results found: {str(e)}")
            return None
    
    def _save_serp_result(self, result: SERPResult) -> None:
        """
        Save the SERP result to a JSON file.
        
        Args:
            result: SERPResult object to save
        """
        # Create a filename based on the query and timestamp
        safe_query = result.query.replace(" ", "_").replace("/", "_").replace("\\", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_query}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Convert result to dictionary
        result_dict = {
            "query": result.query,
            "timestamp": result.timestamp,
            "ads_top": result.ads_top,
            "ads_bottom": result.ads_bottom,
            "organic_results": result.organic_results,
            "related_searches": result.related_searches,
            "knowledge_panel": result.knowledge_panel,
            "local_pack": result.local_pack,
            "shopping_results": result.shopping_results
        }
        
        # Save to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"SERP result saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving SERP result: {str(e)}")
    
    def analyze_competitor_ads(self, queries: List[str]) -> Dict[str, Any]:
        """
        Analyze competitor ads across multiple search queries.
        
        Args:
            queries: List of search queries to analyze
            
        Returns:
            Dictionary with competitor ad analysis
        """
        start_time = datetime.now()
        
        try:
            all_ads = []
            competitor_stats = {}
            
            # Scrape SERPs for each query
            for query in queries:
                self.logger.info(f"Analyzing competitor ads for query: {query}")
                
                # Scrape the SERP
                result = self.scrape_serp(query)
                
                # Collect all ads
                ads = result.ads_top + result.ads_bottom
                for ad in ads:
                    ad["query"] = query
                    all_ads.append(ad)
                
                # Add delay between requests
                time.sleep(2)
            
            # Extract unique competitors
            for ad in all_ads:
                display_url = ad["display_url"]
                domain = display_url.split("/")[0]
                
                if domain not in competitor_stats:
                    competitor_stats[domain] = {
                        "ad_count": 0,
                        "queries": set(),
                        "top_position_count": 0,
                        "ad_texts": []
                    }
                
                competitor_stats[domain]["ad_count"] += 1
                competitor_stats[domain]["queries"].add(ad["query"])
                competitor_stats[domain]["ad_texts"].append({
                    "title": ad["title"],
                    "description": ad["description"]
                })
                
                if ad["position"] == "top":
                    competitor_stats[domain]["top_position_count"] += 1
            
            # Convert sets to lists for JSON serialization
            for domain in competitor_stats:
                competitor_stats[domain]["queries"] = list(competitor_stats[domain]["queries"])
            
            # Sort competitors by ad_count
            sorted_competitors = sorted(
                competitor_stats.items(), 
                key=lambda x: x[1]["ad_count"], 
                reverse=True
            )
            
            # Create the result
            result = {
                "total_ads_found": len(all_ads),
                "queries_analyzed": len(queries),
                "timestamp": datetime.now().isoformat(),
                "competitors": dict(sorted_competitors)
            }
            
            # Save the analysis
            filename = f"competitor_ad_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.results_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Competitor ad analysis saved to {filepath}")
            self._track_execution(start_time, True)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing competitor ads: {str(e)}")
            self._track_execution(start_time, False)
            raise
        finally:
            self._close_driver()
    
    def analyze_serp_features(self, queries: List[str]) -> Dict[str, Any]:
        """
        Analyze SERP features across multiple search queries.
        
        Args:
            queries: List of search queries to analyze
            
        Returns:
            Dictionary with SERP feature analysis
        """
        start_time = datetime.now()
        
        try:
            feature_stats = {
                "queries_analyzed": len(queries),
                "features_presence": {
                    "top_ads": 0,
                    "bottom_ads": 0,
                    "knowledge_panel": 0,
                    "local_pack": 0,
                    "shopping_results": 0
                },
                "queries_with_features": {
                    "top_ads": [],
                    "bottom_ads": [],
                    "knowledge_panel": [],
                    "local_pack": [],
                    "shopping_results": []
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Scrape SERPs for each query
            for query in queries:
                self.logger.info(f"Analyzing SERP features for query: {query}")
                
                # Scrape the SERP
                result = self.scrape_serp(query)
                
                # Check for features
                if result.ads_top:
                    feature_stats["features_presence"]["top_ads"] += 1
                    feature_stats["queries_with_features"]["top_ads"].append(query)
                
                if result.ads_bottom:
                    feature_stats["features_presence"]["bottom_ads"] += 1
                    feature_stats["queries_with_features"]["bottom_ads"].append(query)
                
                if result.knowledge_panel:
                    feature_stats["features_presence"]["knowledge_panel"] += 1
                    feature_stats["queries_with_features"]["knowledge_panel"].append(query)
                
                if result.local_pack:
                    feature_stats["features_presence"]["local_pack"] += 1
                    feature_stats["queries_with_features"]["local_pack"].append(query)
                
                if result.shopping_results:
                    feature_stats["features_presence"]["shopping_results"] += 1
                    feature_stats["queries_with_features"]["shopping_results"].append(query)
                
                # Add delay between requests
                time.sleep(2)
            
            # Calculate percentages
            feature_stats["features_percentage"] = {
                feature: (count / len(queries)) * 100
                for feature, count in feature_stats["features_presence"].items()
            }
            
            # Save the analysis
            filename = f"serp_features_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.results_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(feature_stats, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"SERP features analysis saved to {filepath}")
            self._track_execution(start_time, True)
            
            return feature_stats
            
        except Exception as e:
            self.logger.error(f"Error analyzing SERP features: {str(e)}")
            self._track_execution(start_time, False)
            raise
        finally:
            self._close_driver()
    
    def track_keyword_rankings(
        self, keywords: List[str], domain: str, 
        store_history: bool = True
    ) -> Dict[str, Any]:
        """
        Track organic rankings for specific keywords and domain.
        
        Args:
            keywords: List of keywords to track
            domain: Domain to track rankings for
            store_history: Whether to store historical ranking data
            
        Returns:
            Dictionary with ranking data
        """
        start_time = datetime.now()
        
        try:
            ranking_data = {
                "domain": domain,
                "keywords_tracked": len(keywords),
                "timestamp": datetime.now().isoformat(),
                "rankings": {}
            }
            
            # Scrape SERPs for each keyword
            for keyword in keywords:
                self.logger.info(f"Tracking ranking for keyword: {keyword}")
                
                # Scrape the SERP
                result = self.scrape_serp(keyword)
                
                # Find the domain in organic results
                found = False
                position = None
                
                for org_result in result.organic_results:
                    if domain in org_result["url"]:
                        found = True
                        position = org_result["position"]
                        break
                
                # Store the ranking data
                ranking_data["rankings"][keyword] = {
                    "found": found,
                    "position": position,
                    "date": datetime.now().strftime("%Y-%m-%d")
                }
                
                # Add delay between requests
                time.sleep(2)
            
            # Save the rankings
            filename = f"rankings_{domain.replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.results_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(ranking_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Ranking data saved to {filepath}")
            
            # Store historical data if requested
            if store_history:
                self._store_historical_rankings(domain, ranking_data)
            
            self._track_execution(start_time, True)
            return ranking_data
            
        except Exception as e:
            self.logger.error(f"Error tracking keyword rankings: {str(e)}")
            self._track_execution(start_time, False)
            raise
        finally:
            self._close_driver()
    
    def _store_historical_rankings(self, domain: str, ranking_data: Dict[str, Any]) -> None:
        """
        Store historical ranking data for a domain.
        
        Args:
            domain: Domain to store rankings for
            ranking_data: Current ranking data
        """
        history_file = os.path.join(self.results_dir, f"ranking_history_{domain.replace('.', '_')}.json")
        
        # Create or load history
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading ranking history: {str(e)}")
                history = {"domain": domain, "history": []}
        else:
            history = {"domain": domain, "history": []}
        
        # Add current data to history
        history_entry = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "rankings": ranking_data["rankings"]
        }
        
        history["history"].append(history_entry)
        
        # Save updated history
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Updated ranking history for {domain}")
        except Exception as e:
            self.logger.error(f"Error saving ranking history: {str(e)}")
    
    def __del__(self):
        """Ensure WebDriver is closed when the service is destroyed"""
        self._close_driver() 