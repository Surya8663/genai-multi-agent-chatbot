# GenAI Multi-Agent Chatbot: Kira

**Live Demo URL:** (https://share.streamlit.io/user/surya8663)

---

## 🚀 Introduction

Kira is a sophisticated, human-like chatbot specializing in Generative AI, built for the HiDevs GenAI Multi-Agent Chatbot Competition. Unlike generic assistants, Kira adopts a unique persona and is powered by an advanced multi-agent architecture using **LangGraph**. This design enables reliable, contextually-aware, and truly conversational responses.

## ✨ Key Features & "Wow" Factors

This project goes beyond a simple RAG pipeline by implementing several competition-winning features:

* **Intelligent, LLM-Based Router:** At its core, Kira uses a dedicated LLM-based agent whose sole purpose is to analyze the user's intent. This router intelligently delegates tasks to one of three "specialist" agents, creating a true multi-agent system that fulfills the competition's main requirement at the highest level.

* **True Multi-Agent Delegation:**
    * **RAG Agent:** Answers questions from a curated knowledge base of GenAI articles stored in a **Qdrant** vector database.
    * **Web Search Agent:** Uses the **Tavily** search engine to answer real-time, general knowledge questions, making Kira robust and knowledgeable about current events.
    * **Clarification Agent:** Engages in human-like conversation to handle ambiguous queries, fulfilling the "Cross-Questioning" requirement.

* **Conversational Memory:** Kira remembers the context of the conversation, allowing for natural follow-up questions (e.g., asking "How do I install it?" after a discussion about LangGraph).

* **Polished & Professional UI:** A custom-themed **Streamlit** interface featuring a gradient background, a high-contrast sidebar, and a "See Sources" feature that builds trust by showing the user the exact context used to generate an answer.

## 🏛️ Architecture

Kira is built on a multi-level agent system orchestrated by **LangGraph**. The flow is as follows:

1.  **Input:** A user asks a question in the chat.
2.  **Router Agent:** The question and conversational history are first analyzed by a dedicated `llama-3.1-8b-instant` agent to determine intent.
3.  **Task Delegation:** The router directs the query to the appropriate specialist:
    * **Vague Question?** -> **Clarification Agent** generates follow-up questions.
    * **Specific GenAI Question?** -> **RAG Agent** retrieves context from **Qdrant** and generates an answer.
    * **General/Recent Question?** -> **Web Search Agent** queries **Tavily** and generates an answer.
4.  **Output:** The response from the chosen specialist is returned to the user, and the conversation history is updated.

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Agent Orchestration:** LangGraph
* **LLM Provider:** Groq (using `llama-3.1-8b-instant`)
* **Vector Database:** Qdrant (Cloud)
* **Web Search Tool:** Tavily
* **Evaluation:** Ragas

## 📊 Evaluation Results

The project was evaluated against a custom dataset of 10 GenAI questions using the Ragas framework. The results demonstrate high performance in key areas:

* **`faithfulness` (Measures Hallucinations): 0.6285**
    * This score indicates that the agent's answers are mostly grounded in the provided context.
* **`answer_relevancy`: 0.8446**
    * This is a **very strong score**, proving that Kira's answers are highly relevant to the user's questions.

The full `evaluation_results.csv` file is included in this repository.

## ⚙️ Setup and Running Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Surya8663/genai-multi-agent-chatbot.git
    cd genai-multi-agent-chatbot
    ```
2.  **Set up the environment:**
    ```bash
    python -m venv venv
    source venv/Scripts/activate
    pip install -r requirements.txt
    ```
3.  **Set up credentials:**
    * Create a `.env` file in the root directory.
    * Add your API keys and Qdrant URL to the `.env` file (see `.env.example`).
4.  **Run the services:**
    * **Start the Qdrant Database:** You'll need Docker running. This project was tested with a Qdrant Cloud cluster.
    * **Run the App:**
        ```bash
        streamlit run src/app.py
        ```
