@echo off
echo Running all tests for Google Ads Optimization Agent...
echo.

echo Running comprehensive tests...
py test_comprehensive.py

echo.
echo Running logger tests...
py test_logger.py

echo.
echo Running ads API tests...
py test_ads_api.py

echo.
echo Running app tests...
py test_app.py

echo.
echo Running command pattern tests...
py test_command_pattern.py

echo.
echo Running command direct tests...
py test_command_direct.py

echo.
echo All tests completed!
pause 