@echo off
echo Running tests for Google Ads Autonomous Management System...
echo.

echo Running unit tests...
python -m pytest test_reinforcement_learning_service.py -v

echo.
echo Tests completed.
echo.

pause 