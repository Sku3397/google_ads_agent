# Google Ads Optimization Agent

An intelligent agent that connects to the Google Ads API, fetches campaign data, and uses GPT-4 to provide optimization suggestions.

## Features

- Connects to Google Ads API to fetch campaign performance metrics
- Analyzes campaign data using OpenAI's GPT-4
- Provides actionable optimization suggestions
- Supports scheduled runs (daily or weekly)
- Modern Streamlit GUI with interactive dashboard
- Chat interface for natural language interaction
- Advanced error reporting and logging
- Beautiful data visualizations

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the `.env.template` file to `.env` and fill in your API credentials:

```bash
cp .env.template .env
```

## Configuration

You need to set up the following environment variables in your `.env` file:

### Google Ads API Credentials
- `GOOGLE_ADS_CLIENT_ID`: Your Google Ads API client ID
- `GOOGLE_ADS_CLIENT_SECRET`: Your Google Ads API client secret
- `GOOGLE_ADS_DEVELOPER_TOKEN`: Your Google Ads API developer token
- `GOOGLE_ADS_REFRESH_TOKEN`: Your Google Ads API refresh token
- `GOOGLE_ADS_LOGIN_CUSTOMER_ID`: Your Google Ads login customer ID
- `GOOGLE_ADS_CUSTOMER_ID`: The customer ID you want to analyze

### OpenAI API Credentials
- `OPENAI_API_KEY`: Your OpenAI API key

## Usage

### GUI Application (Recommended)

Run the Streamlit web application:

```bash
streamlit run app.py
```

Or use the provided batch file on Windows:

```bash
run.bat
```

The GUI provides the following sections:
- **Dashboard**: Overview of campaign performance with key metrics
- **Campaign Analysis**: Detailed analysis and visualization of campaign data
- **Optimization**: GPT-4 powered optimization suggestions
- **Chat Assistant**: Natural language interface to interact with the agent
- **Scheduler**: Configure automated analysis on a schedule
- **System Logs**: View detailed logs and error reports

### Command-line Interface (Legacy)

For command-line usage:

Run a one-time analysis for the last 30 days:

```bash
python main.py
```

Run a one-time analysis for a custom number of days:

```bash
python main.py --days 60
```

Schedule a daily analysis at 9:00 AM:

```bash
python main.py --schedule
```

Schedule a daily analysis at a custom time:

```bash
python main.py --schedule --hour 14 --minute 30
```

## Testing

The application includes comprehensive test suites to ensure functionality and stability:

### Running All Tests

Use the provided batch file on Windows:

```bash
test_all.bat
```

Or run individual test files:

```bash
py test_comprehensive.py   # Comprehensive functionality tests
py test_logger.py          # Logger tests
py test_ads_api.py         # Google Ads API tests
py test_app.py             # Main application tests
py test_command_pattern.py # Command pattern tests
py test_command_direct.py  # Direct command tests
```

### Test Coverage

- **test_comprehensive.py**: End-to-end testing of all major components
- **test_logger.py**: Tests for UTF-8 encoding and proper logging
- **test_ads_api.py**: Tests for Google Ads API integration
- **test_app.py**: Tests for the main application functionality
- **test_command_pattern.py**: Tests for the command pattern implementation
- **test_command_direct.py**: Direct command execution tests

## Troubleshooting

### Python Command Issues

If you encounter "Python was not found" errors, try using the `py` command instead:

```bash
py app.py
```

Or for Python 3 specifically:

```bash
python3 app.py
```

### Missing Packages

If you encounter missing package errors, install the requirements:

```bash
py -m pip install -r requirements.txt
```

### API Connection Issues

Ensure your `.env` file contains valid API credentials. Check the logs directory for detailed error messages.

## Project Structure

- `app.py`: Main Streamlit GUI application
- `main.py`: Legacy CLI interface
- `ads_api.py`: Google Ads API connection
- `optimizer.py`: GPT-4 integration for analysis
- `scheduler.py`: Automation with the schedule library
- `config.py`: Configuration loading
- `logger.py`: Enhanced error reporting and logging
- `chat_interface.py`: Natural language chat interface
- `.env`: API keys and secrets

## Chat Commands

The chat assistant supports the following command types:
- **Fetch Data**: "Get campaign data for the last 14 days"
- **Analyze**: "Analyze my campaigns and give optimization suggestions"
- **Custom Query**: "Find campaigns with low CTR"
- **Schedule**: "Schedule daily optimization at 2:30pm"
- **Help**: "Show available commands"

## Future Enhancements

Future versions will support:
- Automatic campaign changes through the API
- Real-time notification system
- Integration with other advertising platforms
- Machine learning-based suggestions to complement GPT-4
- Multi-user support with authentication 