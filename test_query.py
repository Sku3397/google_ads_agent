import sys
import logging
import inspect
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

# Make sure we're using a fresh import
if 'ads_api' in sys.modules:
    del sys.modules['ads_api']

try:
    import ads_api
    
    print(f"ads_api loaded from: {ads_api.__file__}")
    
    # Get the source code of the get_keyword_performance method
    source = inspect.getsource(ads_api.GoogleAdsAPI.get_keyword_performance)
    
    # Extract the SQL-like query string
    query_match = re.search(r'query\s*=\s*f"""([\s\S]*?)"""', source)
    
    if query_match:
        query_template = query_match.group(1).strip()
        print("\nQUERY TEMPLATE FOUND:")
        print("-" * 80)
        for i, line in enumerate(query_template.split('\n')):
            print(f"{i+1:3d} | {line}")
        print("-" * 80)
        
        # Search for specific fields
        fields_of_interest = [
            "metrics.average_position",
            "metrics.top_impression_percentage",
            "metrics.search_impression_share",
            "metrics.search_top_impression_share" 
        ]
        
        print("\nSEARCHING FOR FIELDS OF INTEREST:")
        for field in fields_of_interest:
            if field in query_template:
                print(f"✅ FOUND: {field}")
                # Find the line
                for i, line in enumerate(query_template.split('\n')):
                    if field in line:
                        print(f"  → Line {i+1}: {line.strip()}")
            else:
                print(f"❌ NOT FOUND: {field}")
                
        print("\nVERIFYING FIELD RETRIEVAL IN METHOD CODE:")
        fields = [
            ("metrics.top_impression_percentage", "top_impression_pct"), 
            ("metrics.search_impression_share", "search_impression_share"),
            ("metrics.search_top_impression_share", "search_top_impression_share")
        ]
        
        for api_field, code_field in fields:
            retrieval_pattern = f"'{code_field}'.*?{api_field}"
            if re.search(retrieval_pattern, source, re.DOTALL):
                print(f"✅ Field '{api_field}' is correctly retrieved as '{code_field}'")
            else:
                print(f"❌ Field '{api_field}' retrieval not found or not matching '{code_field}'")
    else:
        print("Could not find query template in the method source code.")
    
except Exception as e:
    print(f"Error: {e}")

print("\nTest completed.") 