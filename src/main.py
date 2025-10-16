# src/main.py (THE ABSOLUTE FINAL, CORRECTED, AND COMPLETE VERSION)
import os
from typing import TypedDict, Optional, List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_qdrant import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults
# --- 1. SETUP & CONSTANTS ---
from dotenv import load_dotenv
load_dotenv() # This line loads the .env file

# Securely load API keys from the environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY or not TAVILY_API_KEY:
    raise ValueError("API keys for Groq and Tavily must be set in your .env file.")

# Set the keys for the libraries to use
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
# --- Error Checking for Keys ---
if "PASTE_YOUR" in os.environ.get("GROQ_API_KEY", "") or not os.environ.get("GROQ_API_KEY"):
    raise ValueError("Groq API Key is missing. Please paste it into src/main.py.")
if "PASTE_YOUR" in os.environ.get("TAVILY_API_KEY", "") or not os.environ.get("TAVILY_API_KEY"):
    raise ValueError("Tavily API Key is missing. Please paste it into src/main.py.")

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "kira_knowledge_base"
QDRANT_URL = "http://localhost:6333"
LLM_MODEL = "llama-3.1-8b-instant"

# --- 2. LANGGRAPH STATE DEFINITION ---
class GraphState(TypedDict):
    chat_history: List[Dict[str, Any]]
    question: str
    documents: Optional[List[Document]] = None
    answer: Optional[str] = None
    clarifying_questions: Optional[str] = None

# --- 3. INITIALIZE COMPONENTS ---
llm = ChatGroq(model_name=LLM_MODEL, temperature=0.7)
router_llm = ChatGroq(model_name=LLM_MODEL, temperature=0.0)
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
client = QdrantClient(url=QDRANT_URL)
qdrant_store = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embeddings)
retriever = qdrant_store.as_retriever(search_kwargs={"k": 3})
web_search_tool = TavilySearchResults(k=3)

def format_chat_history(chat_history: List[Dict[str, Any]]) -> str:
    buffer = ""
    for message in chat_history:
        role = "Human" if message["role"] == "user" else "AI"
        buffer += f'{role}: {message["content"]}\n'
    return buffer

# --- THE FINAL, MOST ROBUST ROUTER PROMPT & CHAIN ---
router_prompt_template = """
You are an expert at routing a user's question to the correct specialist. This is your only job.
Based on the conversation history and the latest user question, you must decide one of three options. This is your primary instruction.

1.  **WEB_SEARCH**: Choose this if the user is asking a general knowledge question (like history, science, news, people), a real-time event, or a topic completely outside of Generative AI. If you are unsure, default to this option.
2.  **KNOWLEDGE_BASE**: Choose this ONLY if the user is asking a specific question about a GenAI topic like LangChain, LangGraph, RAG, vector databases, or agents.
3.  **CLARIFY**: Choose this ONLY if the question is a simple greeting ("hello"), is completely nonsensical, or is too short to understand (e.g., "tell me stuff").

You must respond with a single word: "WEB_SEARCH", "KNOWLEDGE_BASE", or "CLARIFY".

CONVERSATION HISTORY:
{chat_history}

LATEST USER QUESTION: "{question}"
Your Decision:
"""
router_prompt = PromptTemplate.from_template(router_prompt_template)
router_chain = router_prompt | router_llm | StrOutputParser()

# --- Other Chains ---
rag_prompt = PromptTemplate.from_template("You are Kira, a GenAI Analyst. Use the following context and chat history to answer the user's question.\nCHAT HISTORY:\n{chat_history}\nCONTEXT:\n{context}\nLATEST USER QUESTION: {question}\nANSWER:")
full_rag_chain = rag_prompt | llm | StrOutputParser()

web_search_prompt = PromptTemplate.from_template("You are Kira, a GenAI Analyst. Use the following web search results and chat history to answer the user's question.\nCHAT HISTORY:\n{chat_history}\nSEARCH RESULTS:\n{context}\nLATEST USER QUESTION: {question}\nANSWER:")
full_web_search_chain = web_search_prompt | llm | StrOutputParser()

clarification_prompt = PromptTemplate.from_template("You are Kira, a GenAI Analyst. Use the chat history to understand the context, then generate 2-3 brief follow-up questions for the latest user question.\nCHAT HISTORY:\n{chat_history}\nLATEST USER QUESTION: \"{question}\"\nYOUR CLARIFYING QUESTIONS:")
full_clarification_chain = clarification_prompt | llm | StrOutputParser()

# --- 4. LANGGRAPH NODE FUNCTIONS ---
def retrieve_and_generate_node(state):
    question = state["question"]; chat_history = state["chat_history"]; formatted_history = format_chat_history(chat_history); documents = retriever.invoke(question); formatted_context = "\n\n".join(doc.page_content for doc in documents); answer = full_rag_chain.invoke({"chat_history": formatted_history, "context": formatted_context, "question": question}); return {"documents": documents, "answer": answer}
def web_search_node(state):
    question = state["question"]; chat_history = state["chat_history"]; formatted_history = format_chat_history(chat_history); docs = web_search_tool.invoke({"query": question}); web_context = "\n".join([d["content"] for d in docs]); answer = full_web_search_chain.invoke({"chat_history": formatted_history, "context": web_context, "question": question}); documents = [Document(page_content=d["content"], metadata={"source": d["url"]}) for d in docs]; return {"documents": documents, "answer": answer}
def clarification_node(state):
    question = state["question"]; chat_history = state["chat_history"]; formatted_history = format_chat_history(chat_history); questions = full_clarification_chain.invoke({"chat_history": formatted_history, "question": question}); return {"clarifying_questions": questions}
def router_decision(state):
    question = state["question"]; chat_history = state["chat_history"]; formatted_history = format_chat_history(chat_history); decision = router_chain.invoke({"chat_history": formatted_history, "question": question});
    if "WEB_SEARCH" in decision: return "go_to_web_search"
    elif "KNOWLEDGE_BASE" in decision: return "go_to_rag"
    else: return "go_to_clarify"

# --- 5. COMPILE THE AGENT ---
workflow = StateGraph(GraphState); workflow.add_node("clarify", clarification_node); workflow.add_node("rag", retrieve_and_generate_node); workflow.add_node("web_search", web_search_node); workflow.add_conditional_edges("__start__", router_decision, {"go_to_clarify": "clarify", "go_to_rag": "rag", "go_to_web_search": "web_search"}); workflow.add_edge("clarify", END); workflow.add_edge("rag", END); workflow.add_edge("web_search", END); agent_app = workflow.compile()

# --- 6. SETUP FASTAPI APP ---
app = FastAPI(title="Kira - GenAI Chatbot API")
class ChatRequest(BaseModel): messages: List[Dict[str, Any]]
class DocumentModel(BaseModel): page_content: str; metadata: dict = Field(default_factory=dict)
class ChatResponse(BaseModel): answer: Optional[str] = None; clarifying_questions: Optional[str] = None; context: Optional[List[DocumentModel]] = None
@app.post("/chat", response_model=ChatResponse)
def chat_with_agent(request: ChatRequest):
    chat_history = request.messages[:-1]; question = request.messages[-1]['content']; inputs = {"chat_history": chat_history, "question": question}; result = agent_app.invoke(inputs);
    if result.get("documents"): result["context"] = [DocumentModel(page_content=doc.page_content, metadata=doc.metadata) for doc in result["documents"]]
    return result
@app.get("/")
def read_root(): return {"message": "Welcome to Kira's Agent API."}
