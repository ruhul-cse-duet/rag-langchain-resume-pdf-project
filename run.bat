@echo off
REM Quick start script for RAG Resume Chatbot (Windows)

echo 🚀 Starting RAG Resume Chatbot...

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Run Streamlit app
echo 🎉 Starting Streamlit app...
streamlit run app.py

