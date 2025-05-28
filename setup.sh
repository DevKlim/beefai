#!/bin/bash
    # Cross-platform setup script for Linux and macOS
    
    # Exit immediately if a command exits with a non-zero status.
    set -e
    
    echo "Starting BeefAI project setup for Linux/macOS..."
    
    # --- Python Version Check ---
    PYTHON_CMD_PREFERRED="python3.11"
    PYTHON_CMD_FALLBACK="python3"
    PYTHON_CMD=""
    
    if command -v $PYTHON_CMD_PREFERRED &> /dev/null; then
        PYTHON_CMD=$PYTHON_CMD_PREFERRED
    elif command -v $PYTHON_CMD_FALLBACK &> /dev/null; then
        PYTHON_CMD=$PYTHON_CMD_FALLBACK
        echo "Warning: python3.11 not found. Using 'python3'. Please ensure it's version 3.11 or newer."
    else
        echo "Error: Neither python3.11 nor python3 found in PATH."
        echo "Please install Python 3.11 or ensure 'python3' points to it."
        exit 1
    fi
    
    echo "Using Python command: $PYTHON_CMD"
    
    # Verify Python version is 3.11 or newer
    $PYTHON_CMD -c "import sys; assert sys.version_info >= (3, 11), f'Python 3.11+ is required, but found {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'"
    if [ $? -ne 0 ]; then
        echo "Python version check failed. Exiting."
        exit 1
    fi
    echo "Python version check successful: $($PYTHON_CMD --version)"
    
    # --- Virtual Environment Setup ---
    VENV_DIR=".venv"
    
    if [ -d "$VENV_DIR" ]; then
        echo "Virtual environment '$VENV_DIR' already exists."
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
    echo "Activating virtual environment..."
    # Corrected path for Linux/macOS, Git Bash on Windows would use Scripts/activate
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
    elif [ -f "$VENV_DIR/Scripts/activate" ]; then # For Git Bash / Windows Python venvs
        source "$VENV_DIR/Scripts/activate"
    else
        echo "Error: Could not find activation script in .venv/bin/activate or .venv/Scripts/activate"
        exit 1
    fi

    if [ $? -ne 0 ]; then
        echo "Error: Failed to activate virtual environment."
        exit 1
    fi
    echo "Virtual environment activated. Pip version: $(pip --version)"
    
    # --- Install Dependencies ---
    REQUIREMENTS_FILE="requirements.txt"
    if [ -f "$REQUIREMENTS_FILE" ]; then
        echo "Installing dependencies from $REQUIREMENTS_FILE..."
        pip install -r $REQUIREMENTS_FILE
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install dependencies from $REQUIREMENTS_FILE."
            exit 1
        fi
        echo "Dependencies installed successfully."
    else
        echo "Error: $REQUIREMENTS_FILE not found."
        exit 1
    fi
    
    # --- Post-installation NLTK downloads (handled by TextProcessor on first import if needed) ---
    echo "NLTK data (e.g., 'punkt', 'cmudict') will be checked/downloaded by TextProcessor if/when it's used."
    
    # --- Setup Complete ---
    echo ""
    echo "BeefAI project setup is complete!"
    echo "The virtual environment '$VENV_DIR' is activated."
    echo "To deactivate, run: deactivate"
    echo "To run the main application (example): python main.py"
    
    exit 0