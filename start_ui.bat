@echo off
setlocal

cd /d "%~dp0"

echo Mimo ASR local UI
echo =================
echo This window runs the local service at http://127.0.0.1:7860
echo Close this terminal window or press Ctrl+C to stop the service and release port 7860.
echo.

where py >nul 2>nul
if %ERRORLEVEL%==0 (
  set "PYTHON_BOOTSTRAP=py -3"
) else (
  where python >nul 2>nul
  if %ERRORLEVEL%==0 (
    set "PYTHON_BOOTSTRAP=python"
  ) else (
    echo Python was not found. Please install Python 3.9 or newer, then run this file again.
    pause
    exit /b 1
  )
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment in .venv ...
  %PYTHON_BOOTSTRAP% -m venv .venv
  if %ERRORLEVEL% neq 0 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
  )
)

set "PYTHON=.venv\Scripts\python.exe"

echo Installing project dependencies into .venv ...
"%PYTHON%" -m pip install --upgrade pip
if %ERRORLEVEL% neq 0 (
  pause
  exit /b 1
)
"%PYTHON%" -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
  pause
  exit /b 1
)

if not exist ".env" (
  if exist ".env.example" (
    copy ".env.example" ".env" >nul
  ) else (
    > ".env" echo MIMO_API_KEY=
    >> ".env" echo HF_TOKEN=
  )
  echo.
  echo Created .env. You can fill API keys in the web UI settings.
)

if not exist "output" mkdir output

echo.
echo Opening http://127.0.0.1:7860 ...

set "MIMO_ASR_OPEN_BROWSER=1"
"%PYTHON%" app.py

endlocal
