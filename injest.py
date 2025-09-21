from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from pymilvus import connections, utility
import config
import os
from langchain_core.embeddings import Embeddings
from fastembed.embedding import DefaultEmbedding as FastEmbedDefaultEmbedding
from typing import List
import numpy as np


# --- LangChain-Compatible Embedding Wrapper ---
class FastEmbedEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = FastEmbedDefaultEmbedding(model_name=model_name)
        print(f"Initialized FastEmbed model: {model_name}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        print(f"Embedding {len(texts)} documents")
        try:
            # Generate embeddings
            embeddings = list(self.model.embed(texts))

            # Convert to list of lists of floats
            result = []
            for embedding in embeddings:
                if isinstance(embedding, np.ndarray):
                    result.append(embedding.tolist())
                else:
                    result.append(list(embedding))

            print(f"Successfully generated {len(result)} document embeddings")
            return result

        except Exception as e:
            print(f"Error in embed_documents: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        print(f"Embedding query: {text[:50]}...")
        try:
            # Generate embedding for single query
            embeddings = list(self.model.embed([text]))
            embedding = embeddings[0]

            # Convert to list of floats
            if isinstance(embedding, np.ndarray):
                result = embedding.tolist()
            else:
                result = list(embedding)

            print(
                f"Successfully generated query embedding with dimension: {len(result)}"
            )
            return result

        except Exception as e:
            print(f"Error in embed_query: {e}")
            raise


def process_and_embed_pdfs(file_paths: List[str]):
    """
    Loads, splits, embeds, and ingests a list of PDF files into Milvus.
    """
    print(f"--- Processing {len(file_paths)} PDF file(s) ---")
    all_chunks = []

    for file_path in file_paths:
        try:
            print(f"Loading PDF: {file_path}")
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            if not docs:
                print(f"No documents loaded from {file_path}")
                continue

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
            )
            chunks = text_splitter.split_documents(docs)

            # Clean and standardize metadata for each chunk
            for chunk in chunks:
                # Keep only essential metadata fields
                clean_metadata = {
                    "source": os.path.basename(file_path),
                    "page": chunk.metadata.get("page", 0),
                }

                # Ensure page is an integer
                if isinstance(clean_metadata["page"], str):
                    try:
                        clean_metadata["page"] = int(clean_metadata["page"])
                    except (ValueError, TypeError):
                        clean_metadata["page"] = 0

                chunk.metadata = clean_metadata

            all_chunks.extend(chunks)
            print(f"Loaded and split '{file_path}' into {len(chunks)} chunks.")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

    if not all_chunks:
        print("No processable content found in the provided files.")
        return 0, 0

    print(f"Total chunks to ingest: {len(all_chunks)}")

    try:
        # Initialize the embedding model
        embedding_model = FastEmbedEmbeddings(model_name=config.EMBEDDING_MODEL)

        # Test embedding functionality
        test_text = "This is a test embedding."
        test_embedding = embedding_model.embed_query(test_text)
        print(f"Test embedding successful. Dimension: {len(test_embedding)}")

        # Ingest into Milvus with simple configuration
        print("Starting ingestion into Milvus...")

        vector_store = Milvus.from_documents(
            documents=all_chunks,
            embedding=embedding_model,
            connection_args={"host": config.MILVUS_HOST, "port": config.MILVUS_PORT},
            collection_name=config.COLLECTION_NAME,
        )

        print(
            f"--- Successfully ingested {len(all_chunks)} chunks from {len(file_paths)} file(s) ---"
        )

        # Test search functionality
        print("Testing search functionality...")
        test_results = vector_store.similarity_search("test", k=1)
        print(f"Search test returned {len(test_results)} results")

        return len(file_paths), len(all_chunks)

    except Exception as e:
        print(f"Error during ingestion: {e}")
        raise
