#!/usr/bin/env python3
"""
Complete fresh start script using existing data_handler code
"""

import os
from pymilvus import connections, utility
import config
from data_handler import process_and_embed_pdfs


def completely_fresh_ingestion():
    """Use the existing data handler for fresh ingestion"""
    print("=== COMPLETE FRESH START ===")

    try:
        # Connect to Milvus and drop existing collection
        print(f"Connecting to Milvus at {config.MILVUS_HOST}:{config.MILVUS_PORT}")
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
            return False

        print(f"Found {len(pdf_files)} PDF files")

        # Use the existing process_and_embed_pdfs function
        docs_processed, chunks_ingested = process_and_embed_pdfs(pdf_files)

        if chunks_ingested > 0:
            print(
                f"Successfully processed {docs_processed} files and ingested {chunks_ingested} chunks"
            )
            return True
        else:
            print("No chunks were ingested")
            return False

    except Exception as e:
        print(f"Error during fresh ingestion: {e}")
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
    print(f"Milvus Host: {config.MILVUS_HOST}")
    print(f"Milvus Port: {config.MILVUS_PORT}")
    print(f"Collection Name: {config.COLLECTION_NAME}")

    success = completely_fresh_ingestion()

    if success:
        test_agent_functionality()
        print("\nFresh start complete!")
    else:
        print("\nFresh start failed. Check the errors above.")
        # Don't exit with error code - let the main app start anyway
        print("Continuing to start the main application...")


if __name__ == "__main__":
    main()
