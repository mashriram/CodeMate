# Production-Ready Dockerfile for the Deep Researcher Agent
FROM ghcr.io/astral-sh/uv:bookworm-slim

# Set environment variables for uv installation
# ENV UV_HOME=/opt/uv
# ENV PATH="/opt/uv/bin:${PATH}"

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt update && apt install -y \
    curl \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libcairo2-dev \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    fontconfig \
    fonts-dejavu-core \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Install uv
# RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN uv init 
RUN uv venv 
# RUN source .venv/bin/activate
# Install Python dependencies using uv
RUN uv pip install  --no-cache -r requirements.txt

# Copy the rest of the application
COPY . .

# Make entrypoint script executable
RUN chmod +x entrypoint.sh

# Create data directory
RUN mkdir -p data

# Expose the Gradio port
EXPOSE 7860

# Health check for the app
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860 || exit 1

# Use the entrypoint script
CMD ["./entrypoint.sh"]