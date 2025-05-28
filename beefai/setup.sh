#!/bin/bash
# Cross-platform setup script for Linux and macOS (Bash)

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting BeefAI project setup..."

# --- Python Version Check ---
PYTHON_CMD_PREFERRED="python3.11"
PYTHON_CMD_FALLBACK="python3"
PYTHON_CMD=""

# Check for preferred Python command
if command -v $PYTHON_CMD_PREFERRED &> /dev/null; then
    PYTHON_CMD=$PYTHON_CMD_PREFERRED
# Else, check for fallback Python command
elif command -v $PYTHON_CMD_FALLBACK &> /dev/null; then
    PYTHON_CMD=$PYTHON_CMD_FALLBACK
    echo "Warning: '$PYTHON_CMD_PREFERRED' not found. Using '$PYTHON_CMD_FALLBACK'."
    echo "Please ensure your Python version is 3.11 or newer."
else
    echo "Error: Neither '$PYTHON_CMD_PREFERRED' nor '$PYTHON_CMD_FALLBACK' found in PATH."
    echo "Please install Python 3.11+ or ensure 'python3' points to it."
    exit 1
fi

echo "Using Python command: $($PYTHON_CMD --version)"

# Verify Python version is 3.11 or newer
$PYTHON_CMD -c "import sys; min_version = (3, 11); current_version = sys.version_info; assert current_version >= min_version, f'Python {min_version[0]}.{min_version[1]}+ is required, but found {current_version.major}.{current_version.minor}.{current_version.micro}'"
if [ $? -ne 0 ]; then
    echo "Python version check failed. Please ensure you have Python 3.11 or newer."
    exit 1
fi
echo "Python version check successful."

# --- Virtual Environment Setup ---
VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' already exists. Re-activating."
else
    echo "Creating virtual environment in '$VENV_DIR'..."
    $PYTHON_CMD -m venv $VENV_DIR
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
    echo "Virtual environment created successfully."
fi

# --- Activate Virtual Environment ---
# Activation script path varies between OS (POSIX vs Windows)
# This script is for POSIX (Linux/macOS)
ACTIVATE_SCRIPT="$VENV_DIR/bin/activate" # Correct path for Linux/macOS

if [ ! -f "$ACTIVATE_SCRIPT" ]; then
    echo "Error: Activation script not found at $ACTIVATE_SCRIPT."
    echo "This might happen if the venv was created on a different OS or is corrupted."
    echo "If on Windows using Git Bash, the path might be '$VENV_DIR/Scripts/activate'."
    exit 1
fi

echo "Activating virtual environment..."
source "$ACTIVATE_SCRIPT"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment."
    exit 1
fi
echo "Virtual environment activated. Pip version: $(pip --version)"

# --- Install/Upgrade Pip ---
echo "Ensuring pip is up-to-date..."
pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "Error: Failed to upgrade pip."
    exit 1
fi

# --- Install Dependencies ---
REQUIREMENTS_FILE="requirements.txt"
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing dependencies from $REQUIREMENTS_FILE..."
    # The --extra-index-url for PyTorch should be handled by pip reading requirements.txt directly
    pip install -r $REQUIREMENTS_FILE
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies from $REQUIREMENTS_FILE."
        echo "Check the error messages above. If it's PyTorch related, ensure your CUDA/CPU setup matches the PyTorch build."
        exit 1
    fi
    echo "Dependencies installed successfully."
else
    echo "Error: $REQUIREMENTS_FILE not found in the current directory ($(pwd))."
    exit 1
fi

# --- Post-installation NLTK downloads (if TextProcessor uses them uncommented) ---
# NLTK download is handled by TextProcessor if uncommented.
# echo "Checking for NLTK data (e.g., 'punkt', 'cmudict')..."
# $PYTHON_CMD -c "import nltk; nltk.download('punkt', quiet=True, download_dir='./nltk_data'); nltk.download('cmudict', quiet=True, download_dir='./nltk_data');"
# echo "NLTK data check/download attempted to ./nltk_data (if TextProcessor needs them and they aren't system-wide)."

# --- Setup Complete ---
echo ""
echo "BeefAI project setup is complete!"
echo "The virtual environment '$VENV_DIR' is activated."
echo "To run the main application (example): $PYTHON_CMD beefai/main.py"
echo "To run the Flask web app (example, if you add one): flask run"
echo "To deactivate the virtual environment, run: deactivate"

exit 0