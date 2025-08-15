# Optimized Dockerfile for Railway deployment
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Create necessary directories
RUN mkdir -p vector_store

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=http://127.0.0.1:11434
ENV PORT=8000

# Create a startup script
RUN echo '#!/bin/bash\n\
echo "Starting Ollama service..."\n\
ollama serve &\n\
echo "Waiting for Ollama to start..."\n\
sleep 10\n\
echo "Pulling Mistral model..."\n\
ollama pull mistral\n\
echo "Starting FastAPI application..."\n\
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}' > /app/start.sh && \
    chmod +x /app/start.sh

# Expose the port (Railway will override this)
EXPOSE 8000

# Run the startup script
CMD ["/bin/bash", "/app/start.sh"]