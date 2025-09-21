#!/usr/bin/env python3
"""
Complete fresh start script - drops everything and creates clean ingestion
"""

import os
from pymilvus import connections, utility
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
import config
from data_handler import FastEmbedEmbeddings


def completely_fresh_ingestion():
    """Completely fresh ingestion with minimal metadata"""
    print("=== COMPLETE FRESH START ===")

    try:
        # Connect and drop everything
        connections.connect("default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)

        if utility.has_collection(config.COLLECTION_NAME):
            utility.drop_collection(config.COLLECTION_NAME)
            print(f"Dropped collection: {config.COLLECTION_NAME}")

        connections.disconnect("default")
        print("Disconnected from Milvus")

        # Find PDF files
        pdf_files = [
            os.path.join(config.DATA_DIRECTORY, f)
            for f in os.listdir(config.DATA_DIRECTORY)
            if f.endswith(".pdf")
        ]

        if not pdf_files:
            print("No PDF files found!")
            return

        print(f"Found {len(pdf_files)} PDF files")

        # Process documents with absolutely minimal metadata
        all_texts = []
        all_metadatas = []

        for file_path in pdf_files:
            try:
                print(f"Processing: {os.path.basename(file_path)}")

                # Load PDF
                loader = PyPDFLoader(file_path)
                docs = loader.load()

                if not docs:
                    continue

                # Split text
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
                )
                chunks = text_splitter.split_documents(docs)

                # Extract only text and minimal metadata
                for chunk in chunks:
                    all_texts.append(chunk.page_content)

                    # Minimal metadata - only what we absolutely need
                    metadata = {
                        "source": os.path.basename(file_path),
                        "page": int(chunk.metadata.get("page", 0)),
                    }
                    all_metadatas.append(metadata)

                print(f"  Added {len(chunks)} chunks")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        if not all_texts:
            print("No text content found!")
            return

        print(f"Total chunks: {len(all_texts)}")

        # Create embeddings
        embedding_model = FastEmbedEmbeddings(model_name=config.EMBEDDING_MODEL)

        # Create vector store using from_texts (simpler than from_documents)
        print("Creating vector store...")
        vector_store = Milvus.from_texts(
            texts=all_texts,
            embedding=embedding_model,
            metadatas=all_metadatas,
            connection_args={"host": config.MILVUS_HOST, "port": config.MILVUS_PORT},
            collection_name=config.COLLECTION_NAME,
        )

        print("Successfully created vector store!")

        # Test search
        print("Testing search...")
        results = vector_store.similarity_search("hackathon", k=3)

        if results:
            print(f"Search successful! Found {len(results)} results:")
            for i, doc in enumerate(results, 1):
                source = doc.metadata.get("source", "N/A")
                page = doc.metadata.get("page", "N/A")
                preview = (
                    doc.page_content[:100] + "..."
                    if len(doc.page_content) > 100
                    else doc.page_content
                )
                print(f"  {i}. [{source}, p.{page}] {preview}")
        else:
            print("Search returned no results")

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_agent_functionality():
    """Test the agent's search tool"""
    print("\n=== Testing Agent Functionality ===")

    try:
        from agent import vector_database_search

        result = vector_database_search.invoke(
            {"query": "hackathon problem statements"}
        )

        if "No information found" in result or "error" in result.lower():
            print(f"Agent test failed: {result}")
        else:
            print("Agent test successful!")
            print(f"Result preview: {result[:200]}...")

    except Exception as e:
        print(f"Agent test error: {e}")


def main():
    print("Starting complete fresh ingestion...")

    success = completely_fresh_ingestion()

    if success:
        test_agent_functionality()
        print("\nFresh start complete! You can now run your main application.")
    else:
        print("\nFresh start failed. Check the errors above.")


if __name__ == "__main__":
    main()
