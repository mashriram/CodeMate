import os
import config

print("=== Environment Debug Info ===")
print(f"MILVUS_HOST env var: {os.getenv('MILVUS_HOST')}")
print(f"MILVUS_PORT env var: {os.getenv('MILVUS_PORT')}")
print(f"GROQ_API_KEY env var: {'SET' if os.getenv('GROQ_API_KEY') else 'NOT SET'}")

print(f"\nConfig values:")
print(f"config.MILVUS_HOST: {config.MILVUS_HOST}")
print(f"config.MILVUS_PORT: {config.MILVUS_PORT}")
print(f"config.GROQ_API_KEY: {'SET' if config.GROQ_API_KEY else 'NOT SET'}")
print(f"config.COLLECTION_NAME: {config.COLLECTION_NAME}")

print(f"\nCurrent working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")

if os.path.exists("data"):
    print(f"Files in data directory: {os.listdir('data')}")
else:
    print("Data directory does not exist")
