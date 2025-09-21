#!/bin/bash
set -e

echo "Starting Deep Researcher Agent..."

# Debug environment variables
echo "=== Debug Info ==="
uv run  debug_env.py

# Function to wait for Milvus to be ready
wait_for_milvus() {
    echo "Waiting for Milvus to be ready..."
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        # Use the internal container port 9091 for health check
        if curl -f http://${MILVUS_HOST}:9091/healthz > /dev/null 2>&1; then
            echo "Milvus is ready!"
            return 0
        fi
        
        echo "Attempt $attempt/$max_attempts: Milvus not ready yet, waiting 10 seconds..."
        sleep 10
        attempt=$((attempt + 1))
    done
    
    echo "Error: Milvus failed to become ready after $max_attempts attempts"
    exit 1
}

# Wait for Milvus
wait_for_milvus

# Additional wait to ensure Milvus is fully initialized
echo "Waiting additional 15 seconds for Milvus full initialization..."
sleep 15

# Check if data directory has PDFs, if so, run ingestion
if [ -n "$(find data -name '*.pdf' 2>/dev/null)" ]; then
    echo "Found PDF files, attempting ingestion..."
    
    # Try fresh_start.py first, fallback to ingest.py
    if [ -f "fresh_start.py" ]; then
        echo "Running fresh_start.py..."
        uv run fresh_start.py
    elif [ -f "ingest.py" ]; then
        echo "Running ingest.py..."
        uv run  ingest.py
    else
        echo "Warning: No ingestion script found (fresh_start.py or ingest.py)"
        echo "You can upload PDFs through the web interface instead."
    fi
else
    echo "No PDF files found in data directory, skipping ingestion"
fi

# Start the Gradio application
# Check which main file exists and use it
# if [ -f "main.py" ]; then
#     echo "Starting Gradio application from main.py..."
#     uv run  main.py
if [ -f "app.py" ]; then
    echo "Starting Gradio application from app.py..."
    uv run  app.py
else
    echo "Error: No main.py or app.py found!"
    ls -la  # Show what files are available
    exit 1
fi