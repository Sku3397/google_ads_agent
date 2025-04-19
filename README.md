# Google Ads Autonomous Management System

A comprehensive, autonomous Google Ads management system that performs everything a senior Google Ads professional would do, including campaign auditing, keyword optimization, bid management, and performance monitoring.

## Architecture

This system uses a modular service-based architecture where each aspect of Google Ads management is handled by a dedicated service. The core services include:

1. **AuditService** - Analyzes campaign and account structure, identifies inefficiencies and opportunities for improvement
2. **KeywordService** - Manages keywords, discovers new high-intent keywords, and optimizes existing ones
3. **NegativeKeywordService** - Identifies and manages negative keywords to improve targeting precision
4. **BidService** - Handles bid optimization using various strategies (target CPA, target ROAS)
5. **CreativeService** - Generates and tests ad copy, manages ad rotation and performance
6. **QualityScoreService** - Monitors and improves Quality Score metrics
7. **AudienceService** - Manages audience targeting and bid modifiers
8. **ReportingService** - Generates performance reports and insights
9. **AnomalyDetectionService** - Detects performance anomalies and triggers alerts
10. **SchedulerService** - Orchestrates tasks and schedules optimizations
11. **DataPersistenceService** - Manages data storage and retrieval

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

## Scheduling Optimizations

### Configuration

Schedule settings can be modified in the `config.py` file or through the web interface.

```python
# Default scheduler settings
SCHEDULER_SETTINGS = {
    "audit_frequency": "daily",      # hourly, daily, weekly
    "audit_hour": 4,                 # Hour to run (0-23)
    "audit_minute": 0,               # Minute to run (0-59)
    "bid_optimization_frequency": "hourly",
    "keyword_optimization_frequency": "daily",
    "reporting_frequency": "weekly",
    "reporting_day": "monday"        # For weekly schedules
}
```

### Running as a Service

For continuous operation, you can run the agent as a background service:

```bash
python run_agent.py --mode service --frequency daily --hour 4 --minute 0
```

This will run the optimizations daily at 4:00 AM.

## Reading Reports and Alerts

### Reports Location

Reports are stored in the `reports/` directory, organized by type:
- `reports/audit/` - Account structure audits
- `reports/keywords/` - Keyword analysis and suggestions
- `reports/performance/` - Performance reports
- `reports/anomalies/` - Detected anomalies

### Alert Configuration

Alerts can be configured in the `.env` file:

```
# Alert settings
ENABLE_EMAIL_ALERTS=true
ALERT_EMAIL=your_email@example.com
ENABLE_SLACK_ALERTS=false
SLACK_WEBHOOK_URL=your_slack_webhook_url
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