import argparse
from config import load_config
from ads_api import GoogleAdsAPI
from optimizer import AdsOptimizer
from scheduler import AdsScheduler

def run_optimization(days=30):
    """
    Run the optimization process.
    
    Args:
        days (int): Number of days to look back for campaign data
        
    Returns:
        str: Optimization suggestions
    """
    print(f"Fetching campaign data for the last {days} days...")
    
    # Load configuration
    config = load_config()
    
    # Initialize APIs
    ads_api = GoogleAdsAPI(config['google_ads'])
    optimizer = AdsOptimizer(config['openai'])
    
    # Get campaign data
    campaigns = ads_api.get_campaign_performance(days_ago=days)
    print(f"Fetched data for {len(campaigns)} campaigns.")
    
    # Get optimization suggestions
    suggestions = optimizer.get_optimization_suggestions(campaigns)
    
    return suggestions

def main():
    """Main function to parse command-line arguments and run the app."""
    parser = argparse.ArgumentParser(description='Google Ads Optimization Agent')
    parser.add_argument('--days', type=int, default=30, help='Number of days to look back for campaign data')
    parser.add_argument('--schedule', action='store_true', help='Run the scheduler instead of a one-time optimization')
    parser.add_argument('--hour', type=int, default=9, help='Hour to run the scheduled task (0-23)')
    parser.add_argument('--minute', type=int, default=0, help='Minute to run the scheduled task (0-59)')
    
    args = parser.parse_args()
    
    if args.schedule:
        print("Starting scheduled optimization...")
        scheduler = AdsScheduler(lambda: print(run_optimization(args.days)))
        scheduler.schedule_daily(hour=args.hour, minute=args.minute)
        scheduler.start()
    else:
        # Run optimization once
        suggestions = run_optimization(args.days)
        print("\nOptimization Suggestions:")
        print(suggestions)

if __name__ == "__main__":
    main() 