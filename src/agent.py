# src/agent.py
import os
from typing import TypedDict
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_qdrant import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langgraph.graph import StateGraph, END

# --- 1. SETUP ---
os.environ["GROQ_API_KEY"] = "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"
if "GROQ_API_KEY" not in os.environ or "PASTE_YOUR_GROQ_API_KEY_HERE" in os.environ["GROQ_API_KEY"]:
    raise ValueError("Groq API Key was not set correctly.")

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "alex_knowledge_base"
QDRANT_URL = "http://localhost:6333"
LLM_MODEL = "llama-3.1-8b-instant"

# --- 2. STATE DEFINITION ---
class GraphState(TypedDict):
    question: str
    answer: str
    clarifying_questions: str

# --- 3. COMPONENT SETUP ---
llm = ChatGroq(model_name=LLM_MODEL, temperature=0.7)
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
client = QdrantClient(url=QDRANT_URL)
qdrant_store = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embeddings)
retriever = qdrant_store.as_retriever(search_kwargs={"k": 3})

rag_prompt = PromptTemplate.from_template(
    "You are Alex, a GenAI Analyst. Use context to answer the question.\nCONTEXT: {context}\nQUESTION: {question}\nANSWER:"
)
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | rag_prompt | llm | StrOutputParser()

clarification_prompt = PromptTemplate.from_template(
    'You are Alex, a GenAI Analyst. Generate 2-3 clarifying questions for the user\'s question.\nUSER\'S QUESTION: "{question}"\nYOUR CLARIFYING QUESTIONS:'
)
clarification_chain = clarification_prompt | llm | StrOutputParser()

# --- 4. GRAPH NODES ---
def retrieve_and_generate_node(state):
    print("--- RAG NODE ---")
    answer = rag_chain.invoke(state["question"])
    return {"answer": answer, "clarifying_questions": ""}

def clarification_node(state):
    print("--- CLARIFICATION NODE ---")
    questions = clarification_chain.invoke({"question": state["question"]})
    return {"answer": "", "clarifying_questions": questions}

def router_decision(state):
    print("--- ROUTER ---")
    question = state["question"]
    if len(question.split()) > 8 or "?" in question:
        print("Decision: RAG")
        return "go_to_rag"
    else:
        print("Decision: Clarify")
        return "go_to_clarify"

# --- 5. GRAPH CONSTRUCTION ---
workflow = StateGraph(GraphState)
workflow.add_node("clarify", clarification_node)
workflow.add_node("rag", retrieve_and_generate_node)

workflow.add_conditional_edges(
    "__start__", # Special entry point name
    router_decision,
    {
        "go_to_clarify": "clarify",
        "go_to_rag": "rag",
    },
)

workflow.add_edge("clarify", END)
workflow.add_edge("rag", END)

app = workflow.compile()

# --- 6. TESTING ---
def main():
    print("\n--- TEST CASE 1: AMBIGUOUS ---")
    inputs = {"question": "Tell me about RAG.", "answer": "", "clarifying_questions": ""}
    result = app.invoke(inputs)
    print("Final State:", result)

    print("\n--- TEST CASE 2: SPECIFIC ---")
    inputs = {"question": "What is LangGraph and how is it different?", "answer": "", "clarifying_questions": ""}
    result = app.invoke(inputs)
    print("Final State:", result)

if __name__ == "__main__":
    main()
