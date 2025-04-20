# Google Ads Autonomous Management System

A comprehensive, autonomous Google Ads management system that performs everything a senior Google Ads professional would do, including campaign auditing, keyword optimization, bid management, and performance monitoring.

## Implementation Status

### Completed Services ✅
- ExpertFeedbackService
- SERPScraperService 
- ReinforcementLearningService (Enhanced with PPO, DQN, A2C, SAC)
- SelfPlayService
- ContextualSignalService
- TrendForecastingService
- LTVBiddingService
- PortfolioOptimizationService

### In Progress ⚠️
- BanditService (Enhancing with Thompson Sampling and UCB)
- CI/CD Pipeline Setup

### Coming Soon 🔜
- LandingPageOptimizationService
- GraphOptimizationService
- VoiceQueryService

## Architecture

This system uses a modular service-based architecture where each aspect of Google Ads management is handled by a dedicated service. The core services include:

1. **AuditService** - Analyzes campaign and account structure, identifies inefficiencies and opportunities for improvement
2. **KeywordService** - Manages keywords, discovers new high-intent keywords, and optimizes existing ones
3. **NegativeKeywordService** - Identifies and manages negative keywords to improve targeting precision
4. **BidService** - Handles bid optimization using various strategies (target CPA, target ROAS)
5. **ReinforcementLearningService** - Uses RL algorithms to optimize bids and budgets for maximum ROI
6. **BanditService** - Implements multi-armed bandit algorithms for dynamic budget allocation
7. **CausalInferenceService** - Measures true causal impact of campaign changes and experiments
8. **MetaLearningService** - Learns from historical strategy performance to improve future optimizations
9. **ForecastingService** - Predicts future performance metrics and identifies emerging search trends
10. **CreativeService** - Generates and tests ad copy, manages ad rotation and performance
11. **SchedulerService** - Orchestrates tasks and schedules optimizations with flexible scheduling patterns
12. **QualityScoreService** - Monitors and improves Quality Score metrics
13. **AudienceService** - Manages audience targeting and bid modifiers
14. **ReportingService** - Generates performance reports and insights
15. **AnomalyDetectionService** - Detects performance anomalies and triggers alerts
16. **DataPersistenceService** - Manages data storage and retrieval
17. **SimulationService** - Simulates changes to campaigns and predicts their impact
18. **ExpertFeedbackService** - Enables human experts to review and provide feedback on AI-generated recommendations
19. **SERPScraperService** - Scrapes search engine results pages to gather competitive intelligence
20. **LTVBiddingService** - Optimizes bidding based on customer lifetime value predictions
21. **PortfolioOptimizationService** - Optimizes budget allocation across campaigns using convex optimization
22. **SelfPlayService** - Uses agent vs agent competition to discover robust bidding strategies through self-play
23. **ContextualSignalService** - Integrates external contextual data to enhance targeting and bidding decisions
24. **TrendForecastingService** - Provides advanced trend analysis and forecasting beyond basic prediction

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/google_ads_agent.git
cd google_ads_agent
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your credentials:
- Copy `.env.template` to `.env`
- Fill in your Google Ads API credentials and Google AI API key

```
# Google Ads API credentials
GOOGLE_ADS_CLIENT_ID=your_client_id
GOOGLE_ADS_CLIENT_SECRET=your_client_secret
GOOGLE_ADS_DEVELOPER_TOKEN=your_developer_token
GOOGLE_ADS_REFRESH_TOKEN=your_refresh_token
GOOGLE_ADS_LOGIN_CUSTOMER_ID=your_login_customer_id
GOOGLE_ADS_CUSTOMER_ID=your_customer_id

# Google AI API key for Gemini
GOOGLE_AI_API_KEY=your_google_ai_api_key
```

## Usage

### Command Line Interface

The main entry point is `ads_agent.py`, which provides a command-line interface for different actions:

```bash
# Run a comprehensive account audit
python ads_agent.py --action audit --days 30

# Discover new keywords for a specific campaign
python ads_agent.py --action keywords --campaign 123456789

# Analyze keyword performance
python ads_agent.py --action performance --days 90

# Run scheduled optimization
python ads_agent.py --action optimize

# Train reinforcement learning bidding policy
python ads_agent.py --action train_rl_policy --episodes 2000

# Generate bid recommendations using reinforcement learning
python ads_agent.py --action rl_bid_recommendations --exploration 0.05

# Initialize self-play agent population
python ads_agent.py --action initialize_self_play_population

# Run self-play tournament to discover optimal strategies
python ads_agent.py --action run_self_play_tournament

# Evolve the self-play agent population
python ads_agent.py --action evolve_self_play_population

# Get the elite strategy from self-play
python ads_agent.py --action get_elite_strategy

# Generate self-play strategy report
python ads_agent.py --action generate_self_play_strategy_report

# Analyze strategy performance using meta-learning
python ads_agent.py --action analyze_strategies

# Analyze specific service strategies
python ads_agent.py --action analyze_strategies --service bid_service

# Forecast metrics for the next 30 days
python ads_agent.py --action forecast_metrics --days 30 --metrics clicks,impressions,conversions,cost

# Forecast budget requirements for a campaign
python ads_agent.py --action forecast_budget --campaign 123456789 --target-metric conversions --target-value 100

# Detect emerging and declining search trends
python ads_agent.py --action forecast_trends

# Get demand forecasts from Google Ads Insights
python ads_agent.py --action forecast_demand

# Get contextual signals for a campaign
python ads_agent.py --action get_contextual_signals --campaign 123456789 --location "New York" --industry "Retail"

# Apply contextual signal-based optimizations
python ads_agent.py --action apply_contextual_optimizations --campaign 123456789

# Generate advanced trend report
python ads_agent.py --action generate_trend_report --campaign 123456789 --lookback_days 90 --forecast_horizon medium_term

# Detect emerging keyword trends
python ads_agent.py --action detect_emerging_trends --campaign 123456789 --min_growth_rate 0.2

# Discover trending keywords in an industry
python ads_agent.py --action discover_trending_keywords --industry "Technology" --location "California" --limit 20

# Start the scheduler service
python run_agent.py --mode scheduler

# Run simulation to predict impact of changes
python ads_agent.py --action simulate --campaign 123456789 --change "increase_bids_by_10_percent"

# Optimize budget allocation across campaigns
python ads_agent.py --action optimize_portfolio --objective conversions --budget_limit 1000

# Analyze keywords across campaigns to identify overlaps and cannibalization
python ads_agent.py --action cross_campaign_keywords

# Apply portfolio optimization recommendations
python ads_agent.py --action apply_portfolio --recommendations_file portfolio_recommendations.json
```

### Streamlit Web Interface

The system also includes a Streamlit web interface for more interactive management:

```bash
streamlit run app.py
```

This will launch a web-based interface where you can:
- View campaign and keyword performance
- Run audits and analyses
- Schedule optimizations
- Apply recommendations
- Generate reports
- Manage scheduled tasks

## Contextual Signal Service

The Contextual Signal Service enriches ad targeting and optimization with external contextual signals that impact consumer behavior and campaign performance.

### Key Features

- **Weather Data Integration** - Weather conditions affect consumer behavior; optimize bids accordingly
- **News and Events Analysis** - Capture the impact of breaking news and local events on search patterns
- **Industry Trend Signals** - Track industry-specific trends and their influence on campaign performance
- **Economic Indicators** - Use economic data to adjust bidding strategies based on market conditions
- **Social Media Trends** - Incorporate social media sentiment and volume for more targeted optimizations
- **Seasonality Factors** - Automatically account for seasons, holidays, and time-based patterns

### How It Works

1. The service collects data from various external APIs (weather, news, economic indicators)
2. It analyzes the relevance of these signals to specific keywords and campaigns
3. Signal-based recommendations are generated for bid adjustments and budget allocation
4. These optimizations can be applied automatically or reviewed first

### Usage

```bash
# Get contextual signals for a specific location and industry
python ads_agent.py --action get_contextual_signals --location "New York" --industry "Retail"

# Apply contextual signal-based optimizations to a campaign
python ads_agent.py --action apply_contextual_optimizations --campaign 123456789

# Get weather-based recommendations
python ads_agent.py --action get_contextual_recommendations --signal_type weather
```

### Configuration

Contextual signal settings can be configured in the `.env` file:

```
# Contextual Signal Service Configuration
WEATHER_API_KEY=your_openweathermap_api_key
NEWS_API_KEY=your_newsapi_key
TRENDS_API_KEY=your_trends_api_key
ECONOMIC_API_KEY=your_economic_api_key
SOCIAL_API_KEY=your_social_api_key
CONTEXTUAL_SIGNAL_CACHE_HOURS=6
CONTEXTUAL_SIGNAL_ENABLED=true
```

For more details, see the [Contextual Signal Service README](services/contextual_signal_service/README.md).

## Trend Forecasting Service

The Trend Forecasting Service provides advanced trend forecasting and analysis capabilities for Google Ads campaigns, going beyond basic forecasting to identify emerging trends, seasonal patterns, and generate comprehensive trend reports.

### Key Features

- **Advanced Time Series Forecasting** - Multiple models including Prophet, SARIMA, and ensemble methods
- **Emerging Trend Detection** - Identify keywords showing accelerating growth patterns
- **Seasonal Pattern Identification** - Discover daily, weekly, monthly, and seasonal patterns
- **Comprehensive Trend Reports** - Generate visualizations and actionable insights
- **Trending Keyword Discovery** - Find trending keywords in specific industries and locations

### How It Works

1. The service collects historical performance data for keywords and campaigns
2. It trains and evaluates multiple forecasting models to select the best approach
3. Advanced analysis detects emerging trends and seasonal patterns
4. Visualizations and reports are generated to provide actionable insights
5. The service can recommend keywords based on industry trends

### Usage

```bash
# Generate a comprehensive trend report
python ads_agent.py --action generate_trend_report --campaign 123456789 --lookback_days 90

# Forecast performance for a specific keyword
python ads_agent.py --action forecast_keyword --keyword "office furniture" --horizon medium_term --metric clicks

# Detect emerging trends in a campaign
python ads_agent.py --action detect_emerging_trends --campaign 123456789 --min_growth_rate 0.2

# Identify seasonal patterns
python ads_agent.py --action identify_seasonal_patterns --campaign 123456789 --metric clicks

# Discover trending keywords in an industry
python ads_agent.py --action discover_trending_keywords --industry "Technology" --location "California"
```

### Configuration

Trend forecasting settings can be configured in the `.env` file:

```
# Trend Forecasting Service Configuration
TREND_FORECASTING_ENABLED=true
TREND_FORECAST_SHORT_TERM_DAYS=7
TREND_FORECAST_MEDIUM_TERM_DAYS=30
TREND_FORECAST_LONG_TERM_DAYS=90
TREND_FORECAST_DEFAULT_MODEL=prophet
TREND_FORECAST_USE_EXTERNAL_SIGNALS=true
TREND_FORECAST_MIN_GROWTH_RATE=0.2
```

For more details, see the [Trend Forecasting Service README](services/trend_forecasting_service/README.md).

## Self Play Service

The system includes a SelfPlayService that employs agent vs agent competition to discover robust bidding strategies through self-play techniques inspired by competitive AI research.

### Key Features

- **Population-based Training (PBT)** - Maintains a population of competing bidding strategies that evolve over time
- **Tournament-style Competition** - Pits strategies against each other in simulated environments to determine the most effective approaches
- **Evolutionary Optimization** - Uses natural selection principles to evolve the population, keeping the best strategies and combining their attributes
- **Strategy Distillation** - Extracts insights from successful strategies for broader application
- **Robustness Evaluation** - Tests strategies against various market conditions to ensure they're robust
- **Self-Play Variations** - Implements techniques for effective self-play learning
- **Transfer Learning** - Applies successful strategies from one context to others

### How It Works

1. The service initializes a population of competing agents with different hyperparameters
2. Agents compete in tournaments within a simulated Google Ads environment
3. Tournament results determine each agent's fitness score
4. The population evolves through selection, crossover, and mutation
5. The best strategies are extracted and can be deployed in real campaigns
6. The process continues iteratively, with strategies becoming increasingly sophisticated

### Usage

```bash
# Initialize the self-play agent population
python ads_agent.py --action initialize_self_play_population

# Run a tournament between competing agents
python ads_agent.py --action run_self_play_tournament

# Evolve the agent population based on tournament results
python ads_agent.py --action evolve_self_play_population

# Get the current best strategy (elite strategy)
python ads_agent.py --action get_elite_strategy

# Generate a report on strategy evolution
python ads_agent.py --action generate_self_play_strategy_report
```

### Configuration

Self-play settings can be configured in the `.env` file:

```
# Self Play Settings
SELF_PLAY_POPULATION_SIZE=10
SELF_PLAY_TOURNAMENT_SIZE=3
SELF_PLAY_ELITISM_COUNT=2
SELF_PLAY_MUTATION_RATE=0.1
SELF_PLAY_CROSSOVER_PROBABILITY=0.3
```

For more details, see the [Self Play Service README](services/self_play_service/README.md).

## Expert Feedback Service

The Expert Feedback Service enables human experts to review, approve, and provide feedback on the AI-generated recommendations. This service helps to establish a human-in-the-loop workflow for ensuring high-quality recommendations.

**Key Features:**
- Submit recommendations for expert review
- Review and approve/reject recommendations
- Modify recommendations with expert insights
- Learn from expert feedback to improve future recommendations

**Example Usage:**
```python
# Submit bid adjustment recommendations for expert review
recommendations = [
    {
        "keyword_id": "123456789",
        "keyword_text": "example keyword",
        "current_bid": 1.0,
        "recommended_bid": 1.5,
        "confidence": 0.85,
        "rationale": "Keyword shows good conversion rate"
    }
]

result = agent.submit_recommendations_for_review(
    recommendation_type="bid_adjustments",
    recommendations=recommendations,
    priority="high"
)

# Get pending reviews for an expert
pending_reviews = agent.get_pending_expert_reviews(expert_id="expert1")

# Apply expert feedback (approve, reject, or modify)
agent.apply_expert_feedback(
    submission_id="abc123",
    expert_id="expert1",
    action="approve",
    feedback={"comment": "Good recommendations"}
)

# Learn from expert feedback to improve future recommendations
agent.learn_from_expert_feedback()
```

## SERP Scraper Service

The SERP Scraper Service provides functionality to scrape and analyze search engine results pages (SERPs) to gather competitive intelligence, track organic rankings, and analyze ad positions and content.

**Key Features:**
- Scrape Google search results for specific queries
- Analyze competitor ad copy, positions, and extensions
- Track organic rankings for specific keywords and domains
- Analyze SERP features (knowledge panels, local packs, shopping results)
- Store historical data for trend analysis

**Example Usage:**
```python
# Scrape a single SERP for a query
serp_data = agent.scrape_single_serp(
    query="google ads management software",
    location="New York"
)

# Analyze competitor ads across multiple queries
competitor_analysis = agent.analyze_serp_competitors([
    "ppc management software",
    "google ads automation",
    "ai google ads management"
])

# Track organic rankings for your domain
ranking_data = agent.track_keyword_rankings(
    keywords=[
        "ppc software",
        "google ads management",
        "automated bid management"
    ],
    domain="yourdomain.com"
)

# Analyze SERP features across queries
feature_analysis = agent.analyze_serp_features([
    "local plumber near me",
    "best restaurants",
    "digital marketing agency"
])
```

## Services

The Google Ads Agent is composed of the following services:

### Expert Feedback Service

The Expert Feedback Service enables human experts to review, approve, and provide feedback on the AI-generated recommendations. This service helps to establish a human-in-the-loop workflow for ensuring high-quality recommendations.

**Key Features:**
- Submit recommendations for expert review
- Review and approve/reject recommendations
- Modify recommendations with expert insights
- Learn from expert feedback to improve future recommendations

**Example Usage:**
```python
# Submit bid adjustment recommendations for expert review
recommendations = [
    {
        "keyword_id": "123456789",
        "keyword_text": "example keyword",
        "current_bid": 1.0,
        "recommended_bid": 1.5,
        "confidence": 0.85,
        "rationale": "Keyword shows good conversion rate"
    }
]

result = agent.submit_recommendations_for_review(
    recommendation_type="bid_adjustments",
    recommendations=recommendations,
    priority="high"
)

# Get pending reviews for an expert
pending_reviews = agent.get_pending_expert_reviews(expert_id="expert1")

# Apply expert feedback (approve, reject, or modify)
agent.apply_expert_feedback(
    submission_id="abc123",
    expert_id="expert1",
    action="approve",
    feedback={"comment": "Good recommendations"}
)

# Learn from expert feedback to improve future recommendations
agent.learn_from_expert_feedback()
```

### SERP Scraper Service

The SERP Scraper Service provides functionality to scrape and analyze search engine results pages (SERPs) to gather competitive intelligence, track organic rankings, and analyze ad positions and content.

**Key Features:**
- Scrape Google search results for specific queries
- Analyze competitor ad copy, positions, and extensions
- Track organic rankings for specific keywords and domains
- Analyze SERP features (knowledge panels, local packs, shopping results)
- Store historical data for trend analysis

**Example Usage:**
```python
# Scrape a single SERP for a query
serp_data = agent.scrape_single_serp(
    query="google ads management software",
    location="New York"
)

# Analyze competitor ads across multiple queries
competitor_analysis = agent.analyze_serp_competitors([
    "ppc management software",
    "google ads automation",
    "ai google ads management"
])

# Track organic rankings for your domain
ranking_data = agent.track_keyword_rankings(
    keywords=[
        "ppc software",
        "google ads management",
        "automated bid management"
    ],
    domain="yourdomain.com"
)

# Analyze SERP features across queries
feature_analysis = agent.analyze_serp_features([
    "local plumber near me",
    "best restaurants",
    "digital marketing agency"
])
```

### Landing Page Optimization Service

The Landing Page Optimization Service provides tools for analyzing and optimizing landing pages to improve conversion rates and user experience, which ultimately enhances the effectiveness of Google Ads campaigns.

**Key Features:**
- Landing page performance analysis
- A/B testing implementation and analysis
- Page speed optimization
- Element analysis to identify high-impact page components
- Form optimization for lead generation
- Content recommendations for higher conversions

**Example Usage:**
```python
# Analyze a landing page for optimization opportunities
analysis = agent.analyze_landing_page(
    url="https://example.com/landing-page",
    days=30
)

# Create an A/B test for a landing page
test = agent.create_a_b_test(
    url="https://example.com/original",
    variant_urls=["https://example.com/variant1", "https://example.com/variant2"],
    test_name="Homepage Hero Section Test",
    duration_days=14
)

# Get A/B test results
results = agent.get_a_b_test_results(test_id="test_12345")

# Analyze specific page elements for conversion impact
element_analysis = agent.analyze_page_elements(
    url="https://example.com/landing-page"
)

# Get page speed optimization recommendations
speed_recommendations = agent.optimize_for_page_speed(
    url="https://example.com/landing-page"
)

# Get form optimization recommendations
form_recommendations = agent.optimize_form_conversion(
    url="https://example.com/landing-page"
)
```

### Graph Optimization Service

The Graph Optimization Service applies graph theory algorithms to Google Ads campaign optimization, enabling sophisticated analysis of keyword relationships, campaign structure, and optimization opportunities.

**Key Features:**
- Keyword relationship graph analysis
- Campaign structure visualization and optimization
- Keyword cluster identification
- Ad group structure optimization
- PageRank for keyword importance
- Community detection to find thematic clusters

**Example Usage:**
```python
# Build a graph representing keyword relationships
graph = agent.build_keyword_relationship_graph(
    days=30,
    campaign_id="123456789",
    min_impressions=100
)

# Build a graph representing campaign structure
structure_graph = agent.build_campaign_structure_graph(
    days=30,
    account_id="123456789"
)

# Identify keyword clusters for ad group optimization
clusters = agent.identify_keyword_clusters(
    graph_id="graph_12345",
    min_cluster_size=3,
    similarity_threshold=0.5
)

# Get recommendations for optimizing ad group structure
recommendations = agent.optimize_adgroup_structure(
    cluster_min_size=3,
    similarity_threshold=0.5,
    max_recommendations=10
)
```

### Voice Query Service

The Voice Query Service helps optimize Google Ads campaigns for voice search queries, which are becoming increasingly important with the growth of voice assistants and voice-based search.

**Key Features:**
- Voice search pattern detection
- Voice search keyword generation
- Voice query analysis
- Voice-specific recommendations
- Conversational pattern analysis
- Question word analysis and optimization

**Example Usage:**
```python
# Analyze search terms for voice search patterns
analysis = agent.analyze_search_terms_for_voice_patterns(
    days=30,
    campaign_id="123456789",
    min_impressions=10
)

# Generate voice search-friendly keyword variations
keywords = agent.generate_voice_search_keywords(
    seed_keywords=["running shoes", "nike sneakers"],
    question_variations=True,
    conversational_variations=True,
    location_intent=True
)

# Get voice search optimization recommendations
recommendations = agent.get_voice_search_recommendations(
    analysis_id="analysis_12345"
)

# Save custom voice patterns for industry-specific optimization
patterns = {
    "industry_specific": ["dental appointment", "legal advice"],
    "local_intents": ["stores in my area", "nearby location"],
    "question_variations": ["tell me about", "I'd like to know"]
}
agent.save_custom_patterns(patterns)
```

## Development

### Adding a New Service

To add a new service:

1. Create a new directory under `services/`
2. Implement the service class inheriting from `BaseService`
3. Add the service to `services/__init__.py`
4. Initialize the service in `AdsAgent._initialize_services()`

### Running Tests

```bash
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Ads API
- Google Gemini API for AI-powered optimization

# Google Ads Creative Optimization Service

This service provides advanced creative optimization and testing capabilities for Google Ads campaigns.

## Features

### 1. Creative Element Analysis
- Analyzes performance of individual creative elements (headlines, descriptions)
- Identifies patterns in high-performing ads
- Provides actionable recommendations for improvement
- Uses NLP techniques to analyze text characteristics

### 2. Automated Creative Testing
- Sets up and monitors A/B tests for ad creatives
- Calculates statistical significance of results
- Provides confidence intervals for performance metrics
- Generates recommendations based on test results

### 3. Performance Monitoring
- Real-time tracking of creative performance
- Automated detection of underperforming ads
- Creative fatigue detection
- Statistical significance testing

### 4. Advanced Analytics
- Text similarity analysis
- Performance pattern recognition
- Audience segment analysis
- Multi-variate testing support

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
GOOGLE_ADS_CLIENT_ID=your_client_id
GOOGLE_ADS_CLIENT_SECRET=your_client_secret
GOOGLE_ADS_DEVELOPER_TOKEN=your_developer_token
GOOGLE_ADS_REFRESH_TOKEN=your_refresh_token
GOOGLE_ADS_LOGIN_CUSTOMER_ID=your_login_customer_id
```

## Usage

### Analyzing Creative Elements

```python
from services.creative_service import CreativeService

# Initialize service
creative_service = CreativeService(client, customer_id)

# Analyze creatives
results = creative_service.analyze_creative_elements(['creative_id_1', 'creative_id_2'])

# Get recommendations
recommendations = results['recommendations']
```

### Setting Up Creative Tests

```python
# Setup test
test = creative_service.setup_creative_experiment(
    ad_group_id='123456789',
    elements={
        'headlines': [
            'Amazing Offer - 50% Off',
            'Limited Time Deal - Save Now'
        ],
        'descriptions': [
            'Shop our collection today.',
            'Exclusive deals await.'
        ]
    },
    confidence_level=0.95
)

# Monitor test results
results = creative_service.monitor_creative_test(test)
```

## Best Practices

1. **Sample Size**: Ensure sufficient data before making decisions (minimum 1000 impressions per variant)
2. **Test Duration**: Run tests for at least 2 weeks to account for day-of-week variations
3. **Significance Level**: Use 95% confidence level for most tests
4. **Control Group**: Always include a control variant in tests
5. **Isolation**: Test one element at a time for clear results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details
