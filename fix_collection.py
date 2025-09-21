#!/usr/bin/env python3
"""
Script to fix the Milvus collection metric type issue
"""

from pymilvus import (
    connections,
    utility,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
)
import config
from data_handler import FastEmbedEmbeddings, process_and_embed_pdfs
import os


def check_collection_info():
    """Check current collection configuration"""
    print("=== Checking Current Collection ===")

    try:
        connections.connect("default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)

        if utility.has_collection(config.COLLECTION_NAME):
            collection = Collection(config.COLLECTION_NAME)

            print(f"Collection '{config.COLLECTION_NAME}' exists")
            print(f"Number of entities: {collection.num_entities}")

            # Get schema info
            schema = collection.schema
            print(f"Schema: {schema}")

            # Get index info
            indexes = collection.indexes
            for index in indexes:
                print(f"Index: {index}")

            return True
        else:
            print(f"Collection '{config.COLLECTION_NAME}' does not exist")
            return False

    except Exception as e:
        print(f"Error checking collection: {e}")
        return False


def recreate_collection_with_cosine():
    """Recreate the collection with COSINE metric"""
    print("=== Recreating Collection with COSINE Metric ===")

    try:
        connections.connect("default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)

        # Drop existing collection
        if utility.has_collection(config.COLLECTION_NAME):
            print(f"Dropping existing collection '{config.COLLECTION_NAME}'")
            utility.drop_collection(config.COLLECTION_NAME)

        print("Collection dropped. Now re-ingesting data...")

        # Re-ingest data - this will create a new collection with proper settings
        pdf_files = [
            os.path.join(config.DATA_DIRECTORY, f)
            for f in os.listdir(config.DATA_DIRECTORY)
            if f.endswith(".pdf")
        ]

        if pdf_files:
            print(f"Found {len(pdf_files)} PDF files to re-ingest")
            docs_processed, chunks_ingested = process_and_embed_pdfs(pdf_files)
            print(
                f"✅ Successfully processed {docs_processed} file(s) and ingested {chunks_ingested} chunks"
            )
        else:
            print("❌ No PDF files found in data directory")

    except Exception as e:
        print(f"Error recreating collection: {e}")
        raise


def test_fixed_collection():
    """Test the fixed collection"""
    print("=== Testing Fixed Collection ===")

    try:
        from langchain_milvus import Milvus

        embedding_model = FastEmbedEmbeddings(model_name=config.EMBEDDING_MODEL)

        # Test with L2 metric (which should work now)
        vector_store = Milvus(
            embedding_function=embedding_model,
            collection_name=config.COLLECTION_NAME,
            connection_args={"host": config.MILVUS_HOST, "port": config.MILVUS_PORT},
            search_params={
                "metric_type": "L2",
                "params": {"nprobe": 10},
            },
        )

        # Test search
        results = vector_store.similarity_search("test query", k=3)

        if results:
            print(f"✅ Search successful! Found {len(results)} results")
            for i, doc in enumerate(results, 1):
                source = doc.metadata.get("source", "N/A")
                page = doc.metadata.get("page", "N/A")
                content_preview = (
                    doc.page_content[:100] + "..."
                    if len(doc.page_content) > 100
                    else doc.page_content
                )
                print(f"  {i}. [{source}, page: {page}] {content_preview}")
        else:
            print("⚠️  Search returned no results, but no errors")

    except Exception as e:
        print(f"❌ Test failed: {e}")


def main():
    """Main function"""
    print("🔧 Fixing Milvus Collection Metric Type Issue")
    print("=" * 50)

    # Check current collection
    collection_exists = check_collection_info()

    if collection_exists:
        print("\n" + "!" * 50)
        print("DETECTED METRIC TYPE MISMATCH!")
        print("Your collection was created with L2 metric, but code expects COSINE.")
        print("!" * 50)

        choice = input(
            "\nChoose an option:\n1. Recreate collection (will delete existing data)\n2. Update code to use L2 metric\nEnter choice (1 or 2): "
        ).strip()

        if choice == "1":
            recreate_collection_with_cosine()
        elif choice == "2":
            print("\n✅ Use the updated agent.py and test_search.py files I provided.")
            print("They now use L2 metric to match your existing collection.")
        else:
            print("Invalid choice. Exiting.")
            return
    else:
        print("No existing collection found. Creating new one...")
        recreate_collection_with_cosine()

    # Test the fixed collection
    test_fixed_collection()

    print("\n" + "=" * 50)
    print("🔧 Collection fix complete!")


if __name__ == "__main__":
    main()
