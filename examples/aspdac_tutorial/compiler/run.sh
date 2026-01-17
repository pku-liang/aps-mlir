#!/bin/bash
# Run the ASP-DAC Tutorial Demo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Install streamlit if not present
pip install -q streamlit

# Run the app
echo "Starting ASP-DAC Tutorial Demo..."
echo "Open http://localhost:8501 in your browser"
echo ""
streamlit run app.py --server.headless true
