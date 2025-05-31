#!/bin/bash
    # BeefAI Project Setup Script for Linux/macOS

    # Exit immediately if a command exits with a non-zero status.
    set -e

    echo "Starting BeefAI project setup..."
    echo "This script will guide you through setting up the Python environment,"
    echo "installing dependencies, and preparing for alignment tools."

    # --- System Prerequisites Check (Informational) ---
    echo ""
    echo "INFO: Please ensure you have the following system prerequisites installed:"
    echo "  - Git"
    echo "  - Python 3.11 or newer (this script will check your 'python3' or 'python3.11' command)"
    echo "  - Conda (Miniconda or Anaconda) for managing specific tool environments (optional, but helpful)"
    echo "  - Build essentials (e.g., 'build-essential' on Debian/Ubuntu, or Xcode Command Line Tools on macOS) if building Python packages from source."
    echo "  - FFmpeg (for audio conversion and whisper-timestamped, e.g., 'sudo apt install ffmpeg' or 'brew install ffmpeg')"
    echo "  - espeak-ng (for phonemizer, e.g., 'sudo apt install espeak-ng' or 'brew install espeak-ng')"
    echo ""
    read -p "Press [Enter] to continue if prerequisites are met..."

    # --- Python Version Check ---
    PYTHON_CMD_PREFERRED="python3.11"
    PYTHON_CMD_FALLBACK="python3"
    PYTHON_CMD=""

    if command -v $PYTHON_CMD_PREFERRED &> /dev/null; then
        PYTHON_CMD=$PYTHON_CMD_PREFERRED
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

    $PYTHON_CMD -c "import sys; min_version = (3, 11); current_version = sys.version_info; assert current_version >= min_version, f'Python {min_version[0]}.{min_version[1]}+ is required, but found {current_version.major}.{current_version.minor}.{current_version.micro}. Python version check FAILED.'"
    if [ $? -ne 0 ]; then
        exit 1
    fi
    echo "Python version check successful."

    # --- Project Virtual Environment Setup ---
    VENV_DIR=".venv"
    if [ -d "$VENV_DIR" ]; then
        echo "Project virtual environment '$VENV_DIR' already exists."
    else
        echo "Creating project virtual environment in '$VENV_DIR'..."
        $PYTHON_CMD -m venv $VENV_DIR
        if [ $? -ne 0 ]; then echo "Error: Failed to create project virtual environment."; exit 1; fi
        echo "Project virtual environment created successfully."
    fi

    echo "Activating project virtual environment..."
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
    elif [ -f "$VENV_DIR/Scripts/activate" ]; then # For Git Bash / Windows Python venvs
        source "$VENV_DIR/Scripts/activate"
    else
        echo "Error: Could not find activation script in $VENV_DIR/bin/activate or $VENV_DIR/Scripts/activate"; exit 1;
    fi
    echo "Project virtual environment activated."

    # --- Install/Upgrade Pip ---
    echo "Ensuring pip is up-to-date in project venv..."
    pip install --upgrade pip
    if [ $? -ne 0 ]; then echo "Error: Failed to upgrade pip."; exit 1; fi

    # --- Install Project Dependencies ---
    REQUIREMENTS_FILE="requirements.txt"
    if [ -f "$REQUIREMENTS_FILE" ]; then
        echo "Installing project dependencies from $REQUIREMENTS_FILE..."
        pip install -r $REQUIREMENTS_FILE
        if [ $? -ne 0 ]; then echo "Error: Failed to install dependencies from $REQUIREMENTS_FILE."; exit 1; fi
        echo "Project dependencies installed successfully."
    else
        echo "Error: $REQUIREMENTS_FILE not found in $(pwd)."
        echo "Please create it with necessary packages (e.g., torch, librosa, pyphen, phonemizer, whisper-timestamped)."
        exit 1
    fi

    # --- Whisper-Timestamped (Forced Aligner) Setup ---
    echo ""
    echo "--- Whisper-Timestamped (Forced Aligner) Information ---"
    echo "'whisper-timestamped' has been installed via requirements.txt."
    echo "It requires FFmpeg to be installed on your system and in PATH."
    echo "Whisper models (e.g., base, small) will be downloaded automatically on first use."
    echo "You can test whisper-timestamped by trying to run 'whisper_timestamped --help' or using it on a short audio file."
    echo ""

    # --- Phonemizer Dependency Check ---
    echo ""
    echo "--- Phonemizer (Phoneme Generation) Information ---"
    echo "'phonemizer' has been installed via requirements.txt."
    echo "It often relies on 'espeak-ng' (or 'festival' for some backends) being installed on your system."
    echo "Please ensure 'espeak-ng' is installed and in your PATH for the default backend to work."
    echo "  Debian/Ubuntu: sudo apt install espeak-ng"
    echo "  macOS: brew install espeak-ng"
    echo ""

    # --- Demucs and Audio Separator (Optional Music Separation Tools) ---
    echo ""
    echo "--- Optional Music Separation Tools ---"
    echo "For source separation (vocals/instrumental), consider installing:"
    echo "1. Demucs: 'pip install demucs'"
    echo "   (May require PyTorch with CUDA if GPU acceleration is desired. Your requirements.txt already handles PyTorch.)"
    echo "2. Audio Separator (UVR Models): 'pip install audio-separator'"
    echo "   (Refer to its documentation for model downloads if needed: https://github.com/KaraokeBox/audio-separator)"
    echo "These can be installed into your project's virtual environment ('$VENV_DIR')."
    echo ""
    read -p "Attempt to install demucs and audio-separator now? (y/n): " install_sep_tools
    if [[ "$install_sep_tools" == "y" || "$install_sep_tools" == "Y" ]]; then
        echo "Installing demucs..."
        pip install demucs
        echo "Installing audio-separator..."
        pip install audio-separator
        echo "Demucs and audio-separator installation attempted."
    else
        echo "Skipping automatic installation of demucs and audio-separator."
    fi


    # --- Setup Complete ---
    echo ""
    echo "--------------------------------------------------------------------"
    echo "BeefAI project basic setup is complete!"
    echo "Project virtual environment '$VENV_DIR' is currently active."
    echo ""
    echo "Next Steps & Reminders:"
    echo "  - Ensure FFmpeg AND espeak-ng are installed system-wide."
    echo "  - Set your GENIUS_ACCESS_TOKEN in a .env file at the project root for lyrics download (see .env.example)."
    echo "    Example .env file content: GENIUS_ACCESS_TOKEN=\"YOUR_ACTUAL_TOKEN_HERE\""
    echo "  - To run data preparation or training scripts: python scripts/your_script_name.py"
    echo "  - To deactivate the project virtual environment: deactivate"
    echo "--------------------------------------------------------------------"

    exit 0