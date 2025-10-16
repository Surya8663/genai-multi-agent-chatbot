# src/clarification_agent.py
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# --- 1. SET UP THE ENVIRONMENT ---
# Bypassing .env file for now. REMEMBER to remove the hardcoded key before submission.
os.environ["GROQ_API_KEY"] = "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"

if "GROQ_API_KEY" not in os.environ or "PASTE_YOUR_ACTUAL_GROQ_API_KEY_HERE" in os.environ["GROQ_API_KEY"]:
    raise ValueError("Groq API Key was not set correctly. Please paste your key into the script.")

# --- 2. DEFINE THE PROMPT TEMPLATE ---
template = """
You are Alex, a helpful GenAI Analyst. A user has asked you a question.
Your goal is to clarify their intent before providing a detailed answer.

Based on the user's question, generate 2-3 brief, insightful follow-up questions that would help you understand their specific needs better.
Frame these questions as if you are a curious and thoughtful analyst. Do not try to answer the original question.

USER'S QUESTION:
"{question}"

YOUR CLARIFYING QUESTIONS:
"""
prompt = PromptTemplate(template=template, input_variables=["question"])

# --- 3. INITIALIZE THE LLM ---
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.5)

# --- 4. BUILD THE CLARIFICATION CHAIN ---
clarification_chain = prompt | llm | StrOutputParser()

# --- 5. RUN THE CHAIN WITH A SAMPLE QUESTION ---
def main():
    print("--- Testing the Clarification Agent ---")

    sample_question = "Tell me about RAG."

    print(f"Original Question: '{sample_question}'\n")
    print("--- Generating Clarifying Questions... ---")

    response = clarification_chain.invoke({"question": sample_question})

    print("\n--- Alex's Clarifying Questions ---")
    print(response)

if __name__ == "__main__":
    main()
