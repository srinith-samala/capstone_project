@echo off
title GridShield-AI - Energy Theft Detection System
color 0A

echo.
echo =====================================================
echo   GRIDSHIELD-AI - BDS-33 CAPSTONE PROJECT
echo   Energy Theft and Meter Tamper Detection System
echo =====================================================
echo.

:: Set working directory to where this bat file lives
cd /d "%~dp0"

echo [1/4] Checking Python installation...
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.11+ from https://python.org
    pause
    exit /b 1
)
echo       Python OK.

echo.
echo [2/4] Installing / checking required packages...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies. Check requirements.txt and internet connection.
    pause
    exit /b 1
)
echo       Packages OK.

echo.
echo [3/4] Checking for pre-trained model artifacts...
if not exist "models_saved\clf_model.pkl" (
    echo       No trained model found. Running full pipeline...
    echo       This will take 30-60 seconds depending on your machine.
    echo.
    python -m src.cli run_all
    if errorlevel 1 (
        echo ERROR: Pipeline training failed. Check logs\app.log for details.
        pause
        exit /b 1
    )
    echo       Pipeline completed successfully!
) else (
    echo       Pre-trained models found. Skipping training.
)

echo.
echo [4/4] Launching GridShield-AI Dashboard...
echo.
echo  *** Dashboard will open in your browser at: http://localhost:8501 ***
echo.
echo  Press Ctrl+C in this window to stop the dashboard server.
echo.

:: Launch streamlit - opens browser automatically
python -m streamlit run src/dashboard/app.py --server.port=8501 --browser.gatherUsageStats=false

pause
