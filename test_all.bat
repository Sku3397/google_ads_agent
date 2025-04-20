@echo off
echo Running all quality checks...

echo.
echo Running flake8...
flake8 .
if %errorlevel% neq 0 (
    echo Flake8 check failed!
    exit /b %errorlevel%
)

echo.
echo Running black check...
black --check .
if %errorlevel% neq 0 (
    echo Black check failed!
    exit /b %errorlevel%
)

echo.
echo Running mypy...
mypy .
if %errorlevel% neq 0 (
    echo Mypy check failed!
    exit /b %errorlevel%
)

echo.
echo Running pytest with coverage...
pytest --maxfail=1 --disable-warnings -q --cov=. --cov-report=xml --cov-fail-under=90
if %errorlevel% neq 0 (
    echo Tests failed!
    exit /b %errorlevel%
)

echo.
echo All quality checks passed successfully! 