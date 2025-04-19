import argparse
import math # Import math for ceiling division
import logging # Import logging
# print("DEBUG: main.py starting import...") # DEBUG
from config import load_config
from ads_api import GoogleAdsAPI
from optimizer import AdsOptimizer
from scheduler import AdsScheduler
# print("DEBUG: main.py imports complete.") # DEBUG

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_optimization(days=90):
    """
    Run the optimization process, fetching data for 90 days.
    Processes keywords campaign by campaign to manage API token limits.

    Args:
        days (int): Number of days to look back for data.

    Returns:
        list: Combined list of optimization suggestions from all campaigns.
    """
    print(f"Fetching campaign data for the last {days} days...")

    # Load configuration
    config = load_config()

    # Initialize APIs
    ads_api = GoogleAdsAPI(config['google_ads'])
    optimizer = AdsOptimizer(config['google_ai'])

    # Get ALL campaign data
    all_campaigns = ads_api.get_campaign_performance(days_ago=days)
    if not all_campaigns:
        print("No campaign data found. Skipping optimization.")
        return []
    print(f"Fetched data for {len(all_campaigns)} campaigns.")

    # Filter for ENABLED campaigns only
    campaigns = [c for c in all_campaigns if c.get('status') == 'ENABLED']
    if not campaigns:
        print("No ENABLED campaigns found in the specified period. Skipping optimization.")
        return []
    print(f"Processing {len(campaigns)} ENABLED campaigns...")

    all_suggestions = []
    MAX_KEYWORDS_PER_CALL = 100 # Max keywords to send in one OpenAI API call

    # Loop through each ENABLED campaign
    for i, campaign in enumerate(campaigns):
        campaign_id = campaign.get('id')
        campaign_name = campaign.get('name', f'ID {campaign_id}')
        print(f"\nProcessing Campaign {i + 1}/{len(campaigns)}: '{campaign_name}' (ID: {campaign_id})")

        if not campaign_id:
            print(f"  WARN: Skipping campaign {i+1} due to missing ID.")
            continue

        try:
            # Get keywords for ONLY the current campaign
            print(f"  Fetching keywords for campaign {campaign_id}...")
            keywords = ads_api.get_keyword_performance(days_ago=days, campaign_id=str(campaign_id))
            print(f"  Fetched {len(keywords)} keywords for this campaign.")

            # Check if keyword batching is needed for this campaign
            if len(keywords) > MAX_KEYWORDS_PER_CALL:
                print(f"  Keyword count ({len(keywords)}) exceeds limit ({MAX_KEYWORDS_PER_CALL}). Processing keywords in batches...")
                num_batches = math.ceil(len(keywords) / MAX_KEYWORDS_PER_CALL)
                campaign_batch_suggestions = []

                for j in range(num_batches):
                    start_index = j * MAX_KEYWORDS_PER_CALL
                    end_index = start_index + MAX_KEYWORDS_PER_CALL
                    keyword_batch = keywords[start_index:end_index]

                    print(f"    Processing keyword batch {j + 1}/{num_batches} ({len(keyword_batch)} keywords) for campaign '{campaign_name}'...")
                    try:
                        # Get suggestions for the current keyword batch
                        batch_suggestions = optimizer.get_optimization_suggestions([campaign], keywords=keyword_batch)
                        if isinstance(batch_suggestions, list):
                            # Filter out parser errors before extending
                            valid_batch_suggestions = [s for s in batch_suggestions if s.get('action_type') != 'ERROR']
                            if valid_batch_suggestions:
                                campaign_batch_suggestions.extend(valid_batch_suggestions)
                                print(f"      Received {len(valid_batch_suggestions)} valid suggestions for batch {j + 1}.")
                            elif batch_suggestions: # Log if only errors were returned by parser
                                logging.warning(f"      Optimizer/parser returned only errors for batch {j + 1}: {batch_suggestions}")
                        else:
                            logging.warning(f"      Received unexpected suggestion format for batch {j + 1}: {batch_suggestions}")
                    except Exception as batch_e:
                        logging.error(f"      ERROR processing keyword batch {j + 1} for campaign '{campaign_name}': {str(batch_e)}")
                        # Optionally add a specific error suggestion for the failed batch
                        campaign_batch_suggestions.append({
                            "index": len(all_suggestions) + len(campaign_batch_suggestions) + 1,
                            "title": f"Error Processing Keyword Batch {j + 1} for Campaign {campaign_name}",
                            "action_type": "ERROR", "entity_type": "system", "entity_id": f"campaign_{campaign_id}_batch_{j+1}",
                            "change": f"Investigate batch processing error: {str(batch_e)}",
                            "rationale": f"Processing failed for keyword batch {j+1} in campaign {campaign_id}.",
                            "status": "error"
                        })
                # Add all collected suggestions for this campaign's batches
                all_suggestions.extend(campaign_batch_suggestions)
                print(f"  Finished processing {num_batches} keyword batches for campaign '{campaign_name}'. Total suggestions for campaign: {len(campaign_batch_suggestions)}.")

            elif len(keywords) > 0:
                # Process all keywords at once if count is within limit and > 0
                print(f"  Getting optimization suggestions for campaign {campaign_id} (all {len(keywords)} keywords at once)...")
                campaign_suggestions = optimizer.get_optimization_suggestions([campaign], keywords=keywords)
                if isinstance(campaign_suggestions, list):
                    valid_suggestions = [s for s in campaign_suggestions if s.get('action_type') != 'ERROR']
                    if valid_suggestions:
                        print(f"  Received {len(valid_suggestions)} valid suggestions for campaign '{campaign_name}'.")
                        all_suggestions.extend(valid_suggestions)
                    elif campaign_suggestions: # Log if only errors were returned by parser
                         logging.warning(f"  Optimizer/parser returned only errors for campaign '{campaign_name}': {campaign_suggestions}")
                else:
                    logging.warning(f"  Received unexpected suggestion format for campaign '{campaign_name}': {campaign_suggestions}")
            else:
                # No keywords to process for this campaign
                print(f"  No keywords found or processed for campaign '{campaign_name}'. Skipping suggestions.")

        except ValueError as ve:
             # Catch potential ValueError from invalid campaign_id in get_keyword_performance
             print(f"  ERROR: Invalid input processing campaign '{campaign_name}': {str(ve)}")
             all_suggestions.append({
                "index": len(all_suggestions) + 1,
                "title": f"Error Processing Campaign {campaign_name}",
                "action_type": "ERROR", "entity_type": "system", "entity_id": f"campaign_{campaign_id}",
                "change": f"Investigate input error: {str(ve)}",
                "rationale": f"Invalid input encountered for campaign {campaign_id} ({campaign_name}).",
                "status": "error"
            })
        except Exception as e:
            print(f"  ERROR: Failed to get suggestions for campaign '{campaign_name}': {str(e)}")
            # Optionally add a custom error suggestion to the list
            all_suggestions.append({
                "index": len(all_suggestions) + 1,
                "title": f"Error Processing Campaign {campaign_name}",
                "action_type": "ERROR", "entity_type": "system", "entity_id": f"campaign_{campaign_id}",
                "change": f"Investigate processing error: {str(e)}",
                "rationale": f"Processing failed for campaign {campaign_id} ({campaign_name}).",
                "status": "error"
            })

    print(f"\nFinished processing all campaigns. Total suggestions generated: {len(all_suggestions)}")
    # TODO: Consider re-indexing suggestions if strict sequential numbering is needed.
    return all_suggestions

def main():
    """Main function to parse command-line arguments and run the app."""
    # print("DEBUG: main() function started.") # DEBUG
    parser = argparse.ArgumentParser(description='Google Ads Optimization Agent')
    parser.add_argument('--days', type=int, default=90, help='Number of days to look back for campaign data')
    parser.add_argument('--schedule', action='store_true', help='Run the scheduler instead of a one-time optimization')
    parser.add_argument('--hour', type=int, default=9, help='Hour to run the scheduled task (0-23)')
    parser.add_argument('--minute', type=int, default=0, help='Minute to run the scheduled task (0-59)')
    
    args = parser.parse_args()
    # print(f"DEBUG: Parsed args: {args}") # DEBUG

    if args.schedule:
        print("Starting scheduled optimization...")
        scheduler = AdsScheduler(lambda: print(run_optimization(args.days)))
        scheduler.schedule_daily(hour=args.hour, minute=args.minute)
        scheduler.start()
    else:
        # Run optimization once
        # print("DEBUG: Calling run_optimization...") # DEBUG
        suggestions = run_optimization(args.days)
        print("\nOptimization Suggestions:")
        # Pretty print the suggestions list
        if isinstance(suggestions, list):
             if suggestions:
                 # Attempt to pretty-print if suggestions is a list of dicts
                 try:
                     import json
                     print(json.dumps(suggestions, indent=2))
                 except ImportError:
                     # Fallback if json import fails (less likely for standard Python)
                     print(suggestions)
             else:
                 print("No suggestions generated.")
        else:
            # Print raw if it's not a list (e.g., an error string)
            print(suggestions)

if __name__ == "__main__":
    # print("DEBUG: Script execution starting (__name__ == '__main__').") # DEBUG
    main()
    # print("DEBUG: Script execution finished.") # DEBUG 
    # print("DEBUG: Script execution finished.") # DEBUG 