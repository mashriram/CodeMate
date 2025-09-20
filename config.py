import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- LLM and Embedding Model Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Recommended embedding model for speed and accuracy
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
# LLM model for planning, synthesis, and refinement
LLM_MODEL = "openai/gpt-oss-20b"


# --- Vector Database Configuration ---
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")  # Use "milvus-standalone" in Docker
MILVUS_PORT = "19530"
COLLECTION_NAME = "research_docs_v1"


# --- Data Ingestion Configuration ---
DATA_DIRECTORY = "data"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 120


# --- Application Configuration ---
APP_TITLE = "Deep Researcher Agent"
