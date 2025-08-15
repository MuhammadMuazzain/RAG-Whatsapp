#!/bin/bash

echo "🚀 Starting Vitiligo Chatbot..."

# Start Ollama in the background
echo "Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to initialize..."
sleep 15

# Check if Ollama is running
if ! ps -p $OLLAMA_PID > /dev/null; then
    echo "❌ Ollama failed to start"
    exit 1
fi

# Pull the Mistral model
echo "Downloading Mistral model (this may take a few minutes)..."
ollama pull mistral || {
    echo "⚠️ Failed to pull Mistral model, continuing anyway..."
}

# Start the FastAPI application
echo "Starting FastAPI application on port ${PORT:-8000}..."
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}