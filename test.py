#!/usr/bin/env python3
"""
Test script to debug search functionality
"""

from langchain_milvus import Milvus
from data_handler import FastEmbedEmbeddings
from pymilvus import connections, utility, Collection
import config


def test_milvus_connection():
    """Test basic Milvus connection and collection status"""
    print("=== Testing Milvus Connection ===")

    try:
        connections.connect("default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        print(f"✅ Connected to Milvus at {config.MILVUS_HOST}:{config.MILVUS_PORT}")

        # Check if collection exists
        print(
            utility.list_collections()
        )  # This will raise an error if connection fails
        if utility.has_collection(config.COLLECTION_NAME):
            print(f"✅ Collection '{config.COLLECTION_NAME}' exists")

            # Get collection info
            collection = Collection(config.COLLECTION_NAME)
            collection.load()  # Load collection into memory
            print(f"📊 Collection stats: {collection.num_entities} documents")

        else:
            print(f"❌ Collection '{config.COLLECTION_NAME}' does not exist")
            return False

    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

    return True


def test_embedding_model():
    """Test the embedding model functionality"""
    print("\n=== Testing Embedding Model ===")

    try:
        embedding_model = FastEmbedEmbeddings(model_name=config.EMBEDDING_MODEL)

        # Test query embedding
        test_query = "What are the benefits of machine learning?"
        embedding = embedding_model.embed_query(test_query)

        print(f"✅ Query embedding successful")
        print(f"📊 Embedding dimension: {len(embedding)}")
        print(f"📊 First 5 values: {embedding[:5]}")

        return embedding_model

    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return None


def test_vector_store_search():
    """Test vector store search functionality"""
    print("\n=== Testing Vector Store Search ===")

    try:
        # Initialize embedding model
        embedding_model = FastEmbedEmbeddings(model_name=config.EMBEDDING_MODEL)

        # Initialize vector store
        vector_store = Milvus(
            embedding_function=embedding_model,
            collection_name=config.COLLECTION_NAME,
            connection_args={"host": config.MILVUS_HOST, "port": config.MILVUS_PORT},
            search_params={
                "metric_type": "L2",
                "params": {"nprobe": 10},
            },
        )

        print("✅ Vector store initialized")

        # Test search with different queries
        test_queries = [
            "machine learning",
            "benefits",
            "technology",
            "research",
            "analysis",
        ]

        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            results = vector_store.similarity_search(query, k=3)

            if results:
                print(f"✅ Found {len(results)} results")
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
                print("❌ No results found")

    except Exception as e:
        print(f"❌ Vector store search error: {e}")
        import traceback

        traceback.print_exc()


def test_agent_tool():
    """Test the agent's vector database search tool"""
    print("\n=== Testing Agent Tool ===")

    try:
        from agent import vector_database_search

        test_query = "What are the key benefits?"
        result = vector_database_search.invoke({"query": test_query})

        if "No information found" in result or "error" in result.lower():
            print(f"❌ Tool returned: {result}")
        else:
            print(f"✅ Tool search successful")
            print(f"📊 Result length: {len(result)} characters")
            print(f"📄 Preview: {result[:200]}...")

    except Exception as e:
        print(f"❌ Agent tool error: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Run all tests"""
    print("🧪 Starting Milvus Search Debug Tests")
    print("=" * 50)

    # Test 1: Basic connection
    if not test_milvus_connection():
        print("❌ Basic connection failed. Check Milvus server and configuration.")
        return

    # Test 2: Embedding model
    embedding_model = test_embedding_model()
    if not embedding_model:
        print("❌ Embedding model failed. Check model configuration.")
        return

    # Test 3: Vector store search
    test_vector_store_search()

    # Test 4: Agent tool
    test_agent_tool()

    print("\n" + "=" * 50)
    print("🧪 Testing complete")


if __name__ == "__main__":
    main()
