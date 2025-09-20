from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from pymilvus import connections, utility
import config

from langchain_core.embeddings import Embeddings
from fastembed.embedding import DefaultEmbedding as FastEmbedDefaultEmbedding
from typing import List


# --- LangChain-Compatible Embedding Wrapper ---
# This is still needed for the initial ingestion with LangChain's Milvus class.
class FastEmbedEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = FastEmbedDefaultEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.embed(texts)
        return [list(embedding) for embedding in embeddings]

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.embed([text])[0]
        return list(embedding)


def process_and_embed_pdfs(file_paths: list[str]):
    """
    Loads, splits, embeds, and ingests a list of PDF files into Milvus.
    """
    print(f"--- Processing {len(file_paths)} PDF file(s) ---")
    all_chunks = []
    for file_path in file_paths:
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
            )
            chunks = text_splitter.split_documents(docs)
            all_chunks.extend(chunks)
            print(f"Loaded and split '{file_path}' into {len(chunks)} chunks.")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

    if not all_chunks:
        print("No processable content found in the provided files.")
        return 0, 0

    print(f"Total chunks to ingest: {len(all_chunks)}")

    # Initialize the new wrapper embedding model
    embedding_model = FastEmbedEmbeddings(model_name=config.EMBEDDING_MODEL)

    # Ingest into Milvus
    print("Starting ingestion into Milvus...")
    Milvus.from_documents(
        documents=all_chunks,
        embedding=embedding_model,
        connection_args={"host": config.MILVUS_HOST, "port": config.MILVUS_PORT},
        collection_name=config.COLLECTION_NAME,
    )

    print(
        f"--- Successfully ingested {len(all_chunks)} chunks from {len(file_paths)} file(s) ---"
    )
    return len(file_paths), len(all_chunks)
