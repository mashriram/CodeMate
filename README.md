# Deep Researcher Agent

An AI-powered research assistant that uses RAG (Retrieval-Augmented Generation) to analyze PDF documents and generate comprehensive research reports. The system combines document ingestion, vector search, and multi-stage AI processing to deliver accurate, cited research outputs.

## Features

- **PDF Document Processing**: Automatically extracts and chunks PDF content
- **Vector Database Search**: Uses Milvus for semantic similarity search
- **Multi-Stage AI Pipeline**: Planning → Research → Drafting → Revision
- **Interactive Web Interface**: Gradio-based UI for easy interaction
- **Comprehensive Citations**: All responses include proper source citations
- **Export Options**: Generate reports in Markdown and PDF formats

## Prerequisites

### System Requirements
- Python 3.12 or higher
- Milvus vector database (standalone installation)
- At least 4GB RAM (8GB recommended)
- 2GB free disk space

### Required Services
- **Milvus**: Vector database for document storage and search
- **Groq API**: For LLM inference (free tier available)

## Installation

### 1. Install Milvus

Download and start Milvus using Docker Compose:

```bash
# Download Milvus configuration
wget https://github.com/milvus-io/milvus/releases/download/v2.6.2/milvus-standalone-docker-compose.yml -O docker-compose-milvus.yml

# Start Milvus
docker compose -f docker-compose-milvus.yml up -d
```

Verify Milvus is running:
```bash
curl http://localhost:9091/healthz
```

### 2. Install UV Package Manager

UV is a fast Python package manager and project manager. Install it:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### 3. Clone and Setup Project

```bash
# Clone the repository
git clone <your-repo-url>
cd CodeMate

# Initialize UV project
uv init

# Install dependencies
uv add -r requirements.txt
```

Alternatively, if you prefer a virtual environment approach:

```bash
# Create and activate virtual environment with UV
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```env
# Required: Groq API key for LLM inference
GROQ_API_KEY=your_groq_api_key_here

# Optional: Milvus configuration (defaults shown)
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

**Get your Groq API key:**
1. Visit [console.groq.com](https://console.groq.com)
2. Sign up/login and create an API key
3. Add it to your `.env` file

### 4. Prepare Your Documents

Create a `data` directory and add your PDF files:

```bash
mkdir data
# Copy your PDF files to the data directory
cp /path/to/your/documents/*.pdf data/
```

## Usage

### 1. Initial Data Ingestion

Process your PDF documents into the vector database:

```bash
# If using UV project mode
uv run python ingest.py

# If using UV venv
source .venv/bin/activate
python ingest.py
```

This will:
- Extract text from all PDFs in the `data/` directory
- Split content into searchable chunks
- Generate embeddings and store in Milvus
- Create search indices for fast retrieval

### 2. Start the Application

Launch the Gradio web interface:

```bash
# If using UV project mode
uv run python main.py

# If using UV venv
python main.py
```

The application will be available at `http://localhost:7860`

### 3. Using the Research Agent

1. **Upload Documents** (optional): Add more PDFs through the web interface
2. **Plan Research**: Enter your research question and click "Plan Research"
3. **Execute Plan**: Review the generated plan and click "Execute Plan"
4. **Download Results**: Export your research report as Markdown or PDF

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Files     │    │   Text Chunks    │    │  Milvus Vector  │
│   (data/)       │───▶│   + Metadata     │───▶│   Database      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐            │
│  Gradio Web UI  │◀──▶│  Research Agent  │            │
│  (port 7860)    │    │  (LangGraph)     │◀───────────┘
└─────────────────┘    └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Groq LLM API   │
                       │ (llama-3.1-8b)   │
                       └──────────────────┘
```

## Configuration

### Key Configuration Files

- **`config.py`**: Main configuration settings
- **`prompts.py`**: AI prompts for different pipeline stages
- **`requirements.txt`**: Python dependencies

### Customizable Settings

```python
# config.py
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # Embedding model
LLM_MODEL = "llama-3.1-8b-instant"         # Groq LLM model
CHUNK_SIZE = 1024                          # Document chunk size
CHUNK_OVERLAP = 120                        # Overlap between chunks
COLLECTION_NAME = "research_docs_v1"       # Milvus collection name
```

## Troubleshooting

### Common Issues

**1. Milvus Connection Errors**
```bash
# Check if Milvus is running
docker ps | grep milvus

# Restart Milvus if needed
docker compose -f docker-compose-milvus.yml restart
```

**2. Groq API Errors**
```bash
# Verify your API key
export GROQ_API_KEY=your_key_here
python -c "import os; print('API Key set:', bool(os.getenv('GROQ_API_KEY')))"
```

**3. PDF Processing Issues**
- Ensure PDFs are text-based (not scanned images)
- Check file permissions in the `data/` directory
- Verify PDF files are not password-protected

**4. Memory Issues**
- Reduce `CHUNK_SIZE` in `config.py` for large documents
- Process fewer documents at once
- Increase system RAM if possible

### Debug Mode

Enable debug logging by setting environment variable:
```bash
# If using UV project mode
DEBUG=1 uv run python main.py

# If using UV venv
export DEBUG=1
python main.py
```

### Testing Search Functionality

Use the test script to verify everything works:
```bash
# If using UV project mode
uv run python test_search.py

# If using UV venv
python test_search.py
```

## Performance Optimization

### For Large Document Collections
- Increase Milvus index parameters in `agent.py`:
  ```python
  search_params = {
      "metric_type": "COSINE",
      "params": {"nprobe": 20}  # Increase for better recall
  }
  ```

### For Better Response Quality
- Use a more powerful Groq model:
  ```python
  LLM_MODEL = "llama-3.1-70b-versatile"  # Larger model
  ```

### For Faster Processing
- Reduce chunk size and increase batch processing
- Use SSD storage for Milvus data
- Ensure sufficient RAM allocation

## Quick Start with UV

If you want to get started quickly with UV:

```bash
# 1. Start Milvus
wget https://github.com/milvus-io/milvus/releases/download/v2.6.2/milvus-standalone-docker-compose.yml -O docker-compose-milvus.yml
docker compose -f docker-compose-milvus.yml up -d

# 2. Setup project with UV
git clone <your-repo-url>
cd CodeMate
uv sync  # Install dependencies from pyproject.toml

# 3. Create .env file
echo "GROQ_API_KEY=your_groq_api_key_here" > .env

# 4. Add PDFs to data directory
mkdir data
# Copy your PDF files here

# 5. Run ingestion and start app
uv run python ingest.py
uv run python main.py
```

Open `http://localhost:7860` in your browser and start researching!

```
CodeMate/
├── agent.py              # Main research agent logic
├── app.py               # Alternative main file name
├── main.py              # Gradio web interface
├── config.py            # Configuration settings
├── data_handler.py      # PDF processing and embedding
├── prompts.py           # AI prompts
├── ingest.py            # Initial data ingestion script
├── fresh_start.py       # Clean re-ingestion script
├── test_search.py       # Search functionality tests
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create this)
├── data/                # PDF documents directory
└── README.md           # This file
```

## API Costs

**Groq API Usage** (approximate):
- Research planning: ~500 tokens
- Document search: ~1,000 tokens per query
- Report generation: ~2,000-5,000 tokens
- Report revision: ~1,000-3,000 tokens

Total per research report: ~4,500-9,500 tokens

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the debug output when running with `DEBUG=1`
3. Ensure all prerequisites are properly installed
4. Verify your API keys and service configurations