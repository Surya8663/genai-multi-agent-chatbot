# src/rag_tester.py
import os
from dotenv import load_dotenv

from langchain_qdrant import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq # <-- CHANGED IMPORT

# --- 1. SET UP THE ENVIRONMENT ---
# Bypassing .env file for now. REMEMBER to remove the hardcoded key before submission.
os.environ["GROQ_API_KEY"] = "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"

if "GROQ_API_KEY" not in os.environ:
    raise ValueError("Groq API Key was not set correctly.")

# --- 2. DEFINE CONSTANTS ---
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "alex_knowledge_base"
QDRANT_URL = "http://localhost:6333"
# We're using Meta's LLaMA 3 8B model - fast and powerful.
LLM_MODEL = "llama-3.1-8b-instant" # Using the latest Llama 3.1 model

# --- 3. INITIALIZE THE RETRIEVER ---
# Make sure your Qdrant Docker container is running!
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
client = QdrantClient(url=QDRANT_URL)
qdrant_store = Qdrant(
    client=client, 
    collection_name=COLLECTION_NAME, 
    embeddings=embeddings
)
retriever = qdrant_store.as_retriever(search_kwargs={"k": 3})

# --- 4. DEFINE THE PROMPT TEMPLATE ---
template = """
You are Alex, a former data journalist now working as a GenAI Analyst.
Your goal is to provide clear, practical, and grounded answers based on the context provided.
Do not mention that you are an AI. Present the information as if you've analyzed it yourself.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# --- 5. INITIALIZE THE LLM ---
llm = ChatGroq(model_name=LLM_MODEL, temperature=0.7) # <-- CHANGED LLM

# --- 6. BUILD THE RAG CHAIN ---
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 7. RUN THE CHAIN ---
def main():
    print("--- Testing the Full RAG Chain with Groq LLaMA 3 ---")
    sample_question = "What is LangGraph and how is it different from LangChain?"
    print(f"Query: {sample_question}\n")
    print("--- Generating Answer... ---")
    response = rag_chain.invoke(sample_question)
    print("\n--- Alex's Answer ---")
    print(response)

if __name__ == "__main__":
    main()
