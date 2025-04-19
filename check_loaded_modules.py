import sys
import os
import importlib
import importlib.util
import inspect

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("sys.path:", sys.path)

# Try to import and inspect ads_api
print("\nTrying to import ads_api...")
try:
    import ads_api
    print("ads_api loaded from:", ads_api.__file__)
    
    # Check if the module has get_keyword_performance 
    if hasattr(ads_api, 'GoogleAdsAPI'):
        print("Found GoogleAdsAPI class")
        # Create an instance to inspect (this will fail without proper config)
        # But we can inspect the class
        cls = ads_api.GoogleAdsAPI
        if hasattr(cls, 'get_keyword_performance'):
            print("Found get_keyword_performance method")
            # Get the source code
            try:
                source = inspect.getsource(cls.get_keyword_performance)
                print("Method source code (first 500 chars):", source[:500])
                # Look for average_position
                if "metrics.average_position" in source:
                    print("WARNING: Found deprecated metrics.average_position in the source code!")
                    # Find the line with the deprecated field
                    for i, line in enumerate(source.split('\n')):
                        if "metrics.average_position" in line:
                            print(f"Line {i+1}: {line.strip()}")
                else:
                    print("metrics.average_position not found in the source code")
            except Exception as e:
                print("Error getting source code:", str(e))
        else:
            print("get_keyword_performance method not found")
    else:
        print("GoogleAdsAPI class not found")
except ImportError as e:
    print("Error importing ads_api:", str(e))
except Exception as e:
    print("Unexpected error:", str(e))

# Look for any modules with 'ads' in the name
print("\nSearching for modules with 'ads' in the name:")
for module_name in sys.modules:
    if 'ads' in module_name.lower():
        module = sys.modules[module_name]
        print(f"Module: {module_name}")
        print(f"  Path: {getattr(module, '__file__', 'Unknown')}")

# Check __pycache__ for any compiled ads_api files
print("\nChecking for compiled ads_api files in __pycache__:")
for path in sys.path:
    pycache_dir = os.path.join(path, '__pycache__')
    if os.path.exists(pycache_dir):
        for file in os.listdir(pycache_dir):
            if 'ads_api' in file.lower():
                full_path = os.path.join(pycache_dir, file)
                print(f"Found: {full_path} (Size: {os.path.getsize(full_path)} bytes)")

# Additional check - look in current directory for __pycache__
current_dir_pycache = os.path.join(os.getcwd(), '__pycache__')
if os.path.exists(current_dir_pycache):
    print("\nChecking current directory __pycache__:")
    for file in os.listdir(current_dir_pycache):
        if 'ads_api' in file.lower():
            full_path = os.path.join(current_dir_pycache, file)
            print(f"Found: {full_path} (Size: {os.path.getsize(full_path)} bytes)")
else:
    print("\nNo __pycache__ found in current directory")

print("\nDiagnostic check complete.") 