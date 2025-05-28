@echo off
echo Starting BeefAI project setup for Windows...
setlocal

REM --- Configuration ---
set "PYTHON_CMD_PREFERRED=py -3.11"
set "PYTHON_CMD_FALLBACK=python"
set "PYTHON_CMD="
set "VENV_DIR=.venv"
set "REQUIREMENTS_FILE=requirements.txt"

REM --- Python Executable Detection ---
echo Detecting Python executable...
REM Try preferred Python command
%PYTHON_CMD_PREFERRED% -c "import sys" >NUL 2>NUL
if %errorlevel% equ 0 (
    set "PYTHON_CMD=%PYTHON_CMD_PREFERRED%"
)

REM If preferred not found, try fallback
if "%PYTHON_CMD%"=="" (
    %PYTHON_CMD_FALLBACK% -c "import sys" >NUL 2>NUL
    if %errorlevel% equ 0 (
        set "PYTHON_CMD=%PYTHON_CMD_FALLBACK%"
        echo Warning: Preferred Python command ('%PYTHON_CMD_PREFERRED%') not found or not working.
        echo Using fallback: '%PYTHON_CMD_FALLBACK%'. Ensure it is Python 3.11+.
    )
)

REM If still no Python command found, exit
if "%PYTHON_CMD%"=="" (
    echo Error: No working Python command found (tried '%PYTHON_CMD_PREFERRED%' and '%PYTHON_CMD_FALLBACK%').
    echo Please install Python 3.11 and ensure it's in PATH.
    goto :eof_error
)
echo Using Python command: %PYTHON_CMD%

REM --- Python Version Check ---
echo Verifying Python version...
%PYTHON_CMD% -c "import sys; ver = sys.version_info; assert ver >= (3, 11), f'Python 3.11+ required, found {ver.major}.{ver.minor}.{ver.micro}'" >NUL 2>NUL
if %errorlevel% neq 0 (
    echo Error: Python version is not 3.11 or newer.
    %PYTHON_CMD% -c "import sys; ver = sys.version_info; print(f'Found Python {ver.major}.{ver.minor}.{ver.micro}')"
    goto :eof_error
)
echo Python version check successful:
for /f "delims=" %%i in ('%PYTHON_CMD% --version') do @echo   %%i

REM --- Virtual Environment Setup ---
if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Virtual environment '%VENV_DIR%' already exists.
) else (
    echo Creating virtual environment in '%VENV_DIR%'...
    %PYTHON_CMD% -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo Error: Failed to create virtual environment.
        goto :eof_error
    )
    echo Virtual environment created successfully.
)

REM --- Activate Virtual Environment ---
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if not defined VIRTUAL_ENV (
    echo Error: Failed to activate virtual environment. Command was: call "%VENV_DIR%\Scripts\activate.bat"
    goto :eof_error
)
echo Virtual environment activated.
echo Pip version:
pip --version

REM --- Install Dependencies ---
if exist "%REQUIREMENTS_FILE%" (
    echo Installing dependencies from %REQUIREMENTS_FILE%...
    pip install -r "%REQUIREMENTS_FILE%"
    if %errorlevel% neq 0 (
        echo Error: Failed to install dependencies from %REQUIREMENTS_FILE%.
        echo Please check the error messages above.
        echo The SSL/TLS error during PyTorch download (DECRYPTION_FAILED_OR_BAD_RECORD_MAC) might be due to network issues, proxy/firewall, or outdated SSL libraries.
        echo You can try the following:
        echo   1. Run this script again (the error might be transient).
        echo   2. Ensure your internet connection is stable.
        echo   3. Temporarily disable any VPN, proxy, or overly aggressive antivirus/firewall software.
        echo   4. Upgrade pip, setuptools, and wheel inside the activated virtual environment:
        echo      pip install --upgrade pip setuptools wheel
        echo      Then, try installing requirements again: pip install -r "%REQUIREMENTS_FILE%"
        echo   5. Manually download the problematic .whl file (e.g., torch-2.7.0+cu126 from the PyTorch website)
        echo      and install it using: pip install path\to\downloaded_file.whl
        echo      Then re-run: pip install -r "%REQUIREMENTS_FILE%" (to get other packages).
        goto :eof_error_pip
    )
    echo Dependencies installed successfully.
) else (
    echo Error: %REQUIREMENTS_FILE% not found.
    goto :eof_error
)

REM --- Post-installation NLTK downloads (optional) ---
REM echo Checking for NLTK data (e.g., 'punkt')...
REM %PYTHON_CMD% -c "import nltk; nltk.download('punkt', quiet=True)"
REM echo NLTK 'punkt' check/download attempted.

REM --- Setup Complete ---
echo.
echo BeefAI project setup is complete!
echo The virtual environment '%VENV_DIR%' is active in this window.
echo To deactivate, run: deactivate
echo To run the main application (example): python main.py
goto :eof_success

:eof_error
echo.
echo Setup failed. Please review the error messages.
if defined VIRTUAL_ENV (
    echo Deactivating virtual environment before exiting due to error.
    call "%VENV_DIR%\Scripts\deactivate.bat"
)
endlocal
exit /b 1

:eof_error_pip
echo.
echo Pip installation failed. The virtual environment is still active for troubleshooting.
echo Review the suggestions above. You might want to try 'pip install --upgrade pip' 
echo and then retry installing requirements ('pip install -r requirements.txt').
endlocal
exit /b 1

:eof_success
endlocal
exit /b 0