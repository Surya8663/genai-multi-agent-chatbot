# src/app.py (THE FINAL, ALL-IN-ONE, GUARANTEED-TO-WORK VERSION)
import os
import streamlit as st
from typing import TypedDict, Optional, List, Dict, Any
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_qdrant import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults

# --- 1. SETUP & CONSTANTS ---
# This loads your .env file with your API keys
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY or not TAVILY_API_KEY:
    st.error("API keys for Groq and Tavily must be set in your .env file. The app cannot start.")
    st.stop()

# Set the keys for the libraries to use
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "kira_knowledge_base"
QDRANT_URL = "http://localhost:6333"
LLM_MODEL = "llama-3.1-8b-instant"

# --- AGENT LOGIC ---

# 2. LANGGRAPH STATE DEFINITION
class GraphState(TypedDict):
    chat_history: List[Dict[str, Any]]; question: str; documents: Optional[List[Document]] = None; answer: Optional[str] = None; clarifying_questions: Optional[str] = None

# This decorator caches the agent so it doesn't reload on every interaction.
@st.cache_resource
def get_agent_app():
    # 3. INITIALIZE COMPONENTS
    llm = ChatGroq(model_name=LLM_MODEL, temperature=0.7)
    router_llm = ChatGroq(model_name=LLM_MODEL, temperature=0.0)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    client = QdrantClient(url=QDRANT_URL)
    qdrant_store = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embeddings)
    retriever = qdrant_store.as_retriever(search_kwargs={"k": 3})
    web_search_tool = TavilySearchResults(k=3)

    def format_chat_history(chat_history: List[Dict[str, Any]]) -> str:
        buffer = "";
        for message in chat_history: buffer += f'{"Human" if message["role"] == "user" else "AI"}: {message["content"]}\n'
        return buffer

    # 4. PROMPTS AND CHAINS
    router_prompt = PromptTemplate.from_template("""You are an expert at routing a user's question. Based on the conversation history and the latest user question, you must decide one of three options:
    1.  **WEB_SEARCH**: Choose this if the user is asking a general knowledge question (like history, science, news, people), a real-time event, or a topic completely outside of Generative AI. If you are unsure, default to this option.
    2.  **KNOWLEDGE_BASE**: Choose this ONLY if the user is asking a specific question about a GenAI topic like LangChain, LangGraph, RAG, vector databases, or agents.
    3.  **CLARIFY**: Choose this ONLY if the question is a simple greeting ("hello"), is completely nonsensical, or is too short to understand (e.g., "tell me stuff").
    You must respond with a single word: "WEB_SEARCH", "KNOWLEDGE_BASE", or "CLARIFY".
    CONVERSATION HISTORY:\n{chat_history}\nLATEST USER QUESTION: "{question}"\nYour Decision:""")
    router_chain = router_prompt | router_llm | StrOutputParser()

    rag_prompt = PromptTemplate.from_template("You are Kira, a GenAI Analyst. Use the following context and chat history to answer the user's question.\nCHAT HISTORY:\n{chat_history}\nCONTEXT:\n{context}\nLATEST USER QUESTION: {question}\nANSWER:")
    full_rag_chain = rag_prompt | llm | StrOutputParser()
    web_search_prompt = PromptTemplate.from_template("You are Kira, a GenAI Analyst. Use the following web search results and chat history to answer the user's question.\nCHAT HISTORY:\n{chat_history}\nSEARCH RESULTS:\n{context}\nLATEST USER QUESTION: {question}\nANSWER:")
    full_web_search_chain = web_search_prompt | llm | StrOutputParser()
    clarification_prompt = PromptTemplate.from_template("You are Kira, a GenAI Analyst. Use the chat history to understand the context, then generate 2-3 brief follow-up questions for the latest user question.\nCHAT HISTORY:\n{chat_history}\nLATEST USER QUESTION: \"{question}\"\nYOUR CLARIFYING QUESTIONS:")
    full_clarification_chain = clarification_prompt | llm | StrOutputParser()

    # 5. LANGGRAPH NODE FUNCTIONS
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

    # 6. COMPILE THE AGENT
    workflow = StateGraph(GraphState); workflow.add_node("clarify", clarification_node); workflow.add_node("rag", retrieve_and_generate_node); workflow.add_node("web_search", web_search_node); workflow.add_conditional_edges("__start__", router_decision, {"go_to_clarify": "clarify", "go_to_rag": "rag", "go_to_web_search": "web_search"}); workflow.add_edge("clarify", END); workflow.add_edge("rag", END); workflow.add_edge("web_search", END);
    agent_app = workflow.compile()
    return agent_app

try:
    agent_app = get_agent_app()
except Exception as e:
    st.error(f"Failed to initialize the agent. Please check your Qdrant connection and API keys. Error: {e}")
    st.stop()

# --- STREAMLIT UI CODE ---
st.markdown("""<style>.stApp {background-image: linear-gradient(to right top, #ff7e5f, #feb47b, #907AD6, #2A2F4F); background-size: cover; background-attachment: fixed; color: #EAEAEA;} [data-testid="stSidebar"] {background-color: white;} [data-testid="stSidebar"] * {color: black !important;} .stChatMessage {background-color: rgba(62, 59, 108, 0.8); border-radius: 10px;} @keyframes fadeIn {from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); }} .fade-in {animation: fadeIn 1.5s ease-out;}</style>""", unsafe_allow_html=True)
st.set_page_config(page_title="Kira - GenAI Analyst", page_icon="🤖", layout="wide")

with st.sidebar:
    st.header("🤖 About Kira"); st.markdown("""Kira is a multi-agent chatbot designed to be your personal Generative AI expert. Kira can answer questions from a specialized knowledge base, search the web for real-time information, and understands the history of your conversation."""); st.subheader("How to Use"); st.markdown("""1. Ask a broad question (e.g., "Tell me about agents").\n2. Answer Kira's clarifying questions to get a more specific answer.\n3. Ask a follow-up question like "How do I install it?". Kira will remember the context!""");
    if st.button("Clear Conversation History"): st.session_state.messages = []; st.rerun()

st.markdown('<h1 class="fade-in">Chat with Kira, your GenAI Analyst</h1>', unsafe_allow_html=True)
st.caption("I'm ready to help you navigate the world of Generative AI.")

if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = [{"role": "assistant", "content": "Hi there! I'm Kira. Ask me about a GenAI topic."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message and "context" in message and message["context"]:
            with st.expander("See Sources"):
                # Handle both list of dicts and list of Document objects
                for doc in message["context"]:
                    if isinstance(doc, dict):
                        st.info(f"Source: {doc.get('metadata', {}).get('source', 'N/A')}")
                        st.markdown(f"> {doc.get('page_content', '')}")
                    else: # Fallback for Document objects if they appear
                        st.info(f"Source: {doc.metadata.get('source', 'N/A')}")
                        st.markdown(f"> {doc.page_content}")


if prompt := st.chat_input("What are you looking into?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.spinner("Kira is thinking..."):
        try:
            chat_history = st.session_state.messages[:-1]
            question = st.session_state.messages[-1]['content']
            inputs = {"chat_history": chat_history, "question": question}
            result = agent_app.invoke(inputs)

            assistant_message = {}
            if result.get("answer"):
                response_text = result["answer"]; assistant_message = {"role": "assistant", "content": response_text};
                # Ensure context is stored in a JSON-serializable format
                if result.get("documents"): assistant_message["context"] = [doc.dict() for doc in result.get("documents", [])]
            elif result.get("clarifying_questions"):
                response_text = result["clarifying_questions"]; assistant_message = {"role": "assistant", "content": response_text}
            else:
                response_text = "Sorry, I encountered an issue."; assistant_message = {"role": "assistant", "content": response_text}

            st.session_state.messages.append(assistant_message)
            st.rerun()
        except Exception as e:
            st.error(f"An error occurred: {e}")
