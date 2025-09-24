#!/bin/bash

# Forex Trading Platform Startup Script
# This script starts the Streamlit web application

echo "🚀 Starting Hybrid LSTM-Transformer Forex Trading Platform..."
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if Streamlit is installed
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "❌ Streamlit is not installed"
    echo "💡 Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if required files exist
if [ ! -f "app.py" ]; then
    echo "❌ app.py not found in current directory"
    exit 1
fi

echo "✅ All requirements met"
echo "🌐 Starting web application on http://localhost:8501"
echo "📊 The trading dashboard will open in your default browser"
echo ""
echo "⚠️  RISK WARNING: This is a trading platform. Please read the disclaimer carefully."
echo "💡 Press Ctrl+C to stop the application"
echo ""

# Start Streamlit app
streamlit run app.py --server.headless true --server.port 8501