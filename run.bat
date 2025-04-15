@echo off
echo Starting Google Ads Optimization Agent...
echo If you encounter API errors, please check API_FIXES.md for troubleshooting information.
echo.

py app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error starting application. If Python is not in your PATH, please run:
    echo python3 app.py
    echo.
    echo You can also try installing the required packages with:
    echo py -m pip install -r requirements.txt
    echo.
    echo For Google Ads API compatibility issues, see API_FIXES.md
    echo.
    pause
) 