# src/ingest.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("Please add QDRANT_URL and QDRANT_API_KEY to your .env file")

# Define the path to the data directory
DATA_PATH = "data/"
# Define the path to the embedding model
EMBED_MODEL = "all-MiniLM-L6-v2"
# Define the Qdrant collection name
COLLECTION_NAME = "kira_knowledge_base"

def main():
    """
    Main function to load, split, and ingest documents into Qdrant.
    """
    print("--- Starting Document Ingestion ---")

    # 1. Load documents from the specified directory
    print(f"Loading documents from: {DATA_PATH}")
    loader = DirectoryLoader(DATA_PATH, glob="*.txt", show_progress=True)
    documents = loader.load()
    if not documents:
        print("No documents found. Exiting.")
        return
    print(f"Loaded {len(documents)} documents.")

    # 2. Split the documents into smaller chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # 3. Initialize the embedding model
    print(f"Initializing embedding model: {EMBED_MODEL}")
    # This will download the model from Hugging Face and run it locally
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # 4. Ingest the chunks into Qdrant
    print(f"Ingesting chunks into Qdrant collection: {COLLECTION_NAME}...")
    # This will create the collection if it doesn't exist
    Qdrant.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
	api_key=QDRANT_API_KEY,  # URL of your running Qdrant instance
        collection_name=COLLECTION_NAME,
        force_recreate=True # Use True to start with a fresh collection each time
    )

    print("--- Ingestion Complete ---")

if __name__ == "__main__":
    main()
