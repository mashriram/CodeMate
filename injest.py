import os
from pymilvus import connections, utility
import config
from data_handler import process_and_embed_pdfs


def main():
    """
    Main function to perform the initial, clean ingestion of all documents
    in the DATA_DIRECTORY.
    """
    print("--- Starting Initial Data Ingestion ---")

    # Establish connection to Milvus
    try:
        connections.connect("default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        print("Successfully connected to Milvus.")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return

    # Drop old collection for a fresh start
    if utility.has_collection(config.COLLECTION_NAME):
        print(
            f"Collection '{config.COLLECTION_NAME}' found. Dropping for a fresh start."
        )
        utility.drop_collection(config.COLLECTION_NAME)

    # Find all PDF files in the data directory
    pdf_files = [
        os.path.join(config.DATA_DIRECTORY, f)
        for f in os.listdir(config.DATA_DIRECTORY)
        if f.endswith(".pdf")
    ]

    if not pdf_files:
        print(f"No PDF files found in '{config.DATA_DIRECTORY}'. Skipping ingestion.")
    else:
        # Process and embed the documents
        process_and_embed_pdfs(pdf_files)

    # Disconnect from Milvus
    connections.disconnect("default")
    print("Disconnected from Milvus.")
    print("--- Initial Data Ingestion Complete ---")


if __name__ == "__main__":
    main()
