# src/retriever.py
from langchain_qdrant import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

# Define the constants from our ingestion script
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "alex_knowledge_base"
QDRANT_URL = "http://localhost:6333"

def main():
    """
    Main function to initialize and test the retriever.
    """
    print("--- Initializing Retriever ---")

    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Initialize the Qdrant client
    client = QdrantClient(url=QDRANT_URL)

    # Create a Qdrant vector store object
    # This object will be used to perform searches
    qdrant_store = Qdrant(
        client=client, 
        collection_name=COLLECTION_NAME, 
        embeddings=embeddings
    )

    # Create a retriever from the vector store
    # 'k' determines how many results to fetch
    retriever = qdrant_store.as_retriever(search_kwargs={"k": 3})

    print("--- Retriever Initialized Successfully ---")
    print("\n--- Testing the Retriever with a Sample Query ---")

    # Define a sample query
    # Change this to a topic you know is in your documents!
    sample_query = "What is LangGraph?"

    print(f"Sample Query: '{sample_query}'")

    # Use the retriever to find relevant documents
    retrieved_docs = retriever.invoke(sample_query)

    print("\n--- Retrieved Documents ---")
    if not retrieved_docs:
        print("No documents found. The database might be empty or the query is too obscure.")
    else:
        for i, doc in enumerate(retrieved_docs):
            print(f"\n--- Document {i+1} ---")
            print(f"Source: {doc.metadata.get('source', 'N/A')}")
            print(f"Content: \n{doc.page_content[:500]}...") # Print the first 500 chars

if __name__ == "__main__":
    main()
