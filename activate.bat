@echo off
echo.
echo ============================================================
echo   Blessing EA Optimizer - Virtual Environment
echo   Rafal Wisniewski - Data and AI Solutions
echo ============================================================
echo.
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated
echo.
echo Available commands:
echo   streamlit run dashboard.py  - Start dashboard
echo   python setup.py             - Reinstall packages
echo   deactivate                  - Exit venv
echo.
cmd /k