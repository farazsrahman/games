#!/bin/bash
# Launcher script for Streamlit app
# Automatically activates virtual environment if it exists

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if venv exists and activate it
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if streamlit is available
if ! command -v streamlit &> /dev/null; then
    echo "Error: streamlit command not found."
    echo "Please make sure:"
    echo "  1. Your virtual environment is activated, or"
    echo "  2. Streamlit is installed: pip install streamlit"
    exit 1
fi

echo "Starting Streamlit app..."
streamlit run streamlit/app.py

