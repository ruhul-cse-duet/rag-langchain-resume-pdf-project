#!/bin/bash

# Quick start script for RAG Resume Chatbot

echo "🚀 Starting RAG Resume Chatbot..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Run Streamlit app
echo "🎉 Starting Streamlit app..."
streamlit run app.py

