# src/app.py (FINAL VERSION without Logo - Text Title Fade-In)
import streamlit as st
import requests

# --- CONFIGURATION ---
FASTAPI_URL = "http://127.0.0.1:8000/chat"
# LOGO_PATH is no longer needed

# --- CUSTOM CSS ---
st.markdown(
    """
    <style>
    /* Main app area styles */
    .stApp {
        background-image: linear-gradient(to right top, #ff7e5f, #feb47b, #907AD6, #2A2F4F);
        background-size: cover;
        background-attachment: fixed;
        color: #EAEAEA;
    }

    /* Sidebar styles for high contrast */
    [data-testid="stSidebar"] {
        background-color: white;
    }
    [data-testid="stSidebar"] * {
        color: black !important;
    }

    /* Chat message styles */
    .stChatMessage {
        background-color: rgba(62, 59, 108, 0.8);
        border-radius: 10px;
    }

    /* CSS Animation for the title fade-in */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeIn 1.5s ease-out;
    }
    /* Logo specific styles are now removed */
    </style>
    """,
    unsafe_allow_html=True
)

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Kira - GenAI Analyst", 
    page_icon="🤖", # Revert to default robot icon
    layout="wide"
)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🤖 About Kira")
    st.markdown("""
    Kira is a multi-agent chatbot designed to be your personal Generative AI expert. 
    
    Kira can answer questions from a specialized knowledge base, search the web for real-time information, and understands the history of your conversation.
    """)
    
    st.subheader("How to Use")
    st.markdown("""
    1.  Ask a broad question (e.g., "Tell me about agents").
    2.  Answer Kira's clarifying questions to get a more specific answer.
    3.  Ask a follow-up question like "How do I install it?". Kira will remember the context!
    """)

    if st.button("Clear Conversation History"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT INTERFACE ---
# Reverted to fade-in text title
st.markdown('<h1 class="fade-in">Chat with Kira, your GenAI Analyst</h1>', unsafe_allow_html=True)
st.caption("I'm ready to help you navigate the world of Generative AI.")

# --- CHAT HISTORY MANAGEMENT ---
if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = [{"role": "assistant", "content": "Hi there! I'm Kira. Ask me about a GenAI topic."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message and message["context"]:
            with st.expander("See Sources"):
                for doc in message["context"]:
                    st.info(f"Source: {doc.get('metadata', {}).get('source', 'N/A')}")
                    st.markdown(f"> {doc.get('page_content', '')}")

# --- USER INPUT HANDLING ---
if prompt := st.chat_input("What are you looking into?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Kira is thinking..."):
        try:
            request_data = {"messages": st.session_state.messages}
            
            response = requests.post(FASTAPI_URL, json=request_data)
            response.raise_for_status()
            result = response.json()

            assistant_message = {}
            if result.get("answer"):
                response_text = result["answer"]
                assistant_message = {"role": "assistant", "content": response_text}
                if result.get("context"): assistant_message["context"] = result["context"]
            elif result.get("clarifying_questions"):
                response_text = result["clarifying_questions"]
                assistant_message = {"role": "assistant", "content": response_text}
            else:
                response_text = "Sorry, I encountered an issue."
                assistant_message = {"role": "assistant", "content": response_text}

            st.session_state.messages.append(assistant_message)
            st.rerun()

        except Exception as e:
            st.error(f"An error occurred: {e}")
