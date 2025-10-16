# evaluate.py (The Absolute Final Victorious Version)
import os
import requests
import pandas as pd
import json
import time
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_groq import ChatGroq
# NEW: Import our local embedding model
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- CONFIGURATION ---
FASTAPI_URL = "http://127.0.0.1:8000/chat"
EVAL_QUESTIONS_FILE = "evaluation_dataset.json"
RESULTS_CSV_FILE = "evaluation_results.csv"
EVALUATION_LLM = "llama-3.1-8b-instant"
# NEW: Define the embedding model we'll use for Ragas
EMBED_MODEL = "all-MiniLM-L6-v2"

# Set the Groq API key for this script
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY must be set in your .env file to run evaluation.")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
def main():
    """
    Runs the full evaluation pipeline.
    """
    if "PASTE_YOUR_GROQ_API_KEY_HERE" in os.environ["GROQ_API_KEY"]:
        raise ValueError("Please paste your Groq API key into the script before running.")

    print("--- Starting Evaluation Pipeline ---")

    # 1. Load test questions
    # (This part is the same)
    try:
        with open(EVAL_QUESTIONS_FILE, 'r') as f:
            eval_data = json.load(f)
        questions = [item['question'] for item in eval_data]
        print(f"Loaded {len(questions)} questions from '{EVAL_QUESTIONS_FILE}'")
    except FileNotFoundError:
        print(f"Error: Evaluation file '{EVAL_QUESTIONS_FILE}' not found.")
        return

    # 2. Call agent and gather results
    # (This part is the same)
    results = []
    print("--- Calling Agent API for each question (with delays) ---")
    for i, question in enumerate(questions):
        print(f"Processing question {i+1}/{len(questions)}: '{question}'")
        try:
            response = requests.post(FASTAPI_URL, json={"question": question})
            response.raise_for_status()
            api_result = response.json()

            contexts = [doc['page_content'] for doc in api_result.get("context", []) or []]

            results.append({
                "question": question,
                "answer": api_result.get("answer", ""),
                "contexts": contexts 
            })
        except requests.exceptions.RequestException as e:
            print(f"  > API call failed for question '{question}': {e}")

        time.sleep(2)

    if not results:
        print("No results were gathered from the API. Please check the FastAPI server. Exiting.")
        return

    # 3. Evaluate using Ragas
    print("--- Evaluating results with Ragas ---")
    dataset = Dataset.from_list(results)

    # NEW: We are creating instances of our LLM and Embedding Model for Ragas
    groq_llm = ChatGroq(model_name=EVALUATION_LLM)
    local_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    metrics = [faithfulness, answer_relevancy]

    # NEW: We are telling Ragas to use BOTH our Groq LLM and our local Embeddings
    score = evaluate(dataset, metrics=metrics, llm=groq_llm, embeddings=local_embeddings)

    print("--- Ragas Evaluation Complete ---")
    print(score)

    # 4. Save to CSV
    df = score.to_pandas()
    df.to_csv(RESULTS_CSV_FILE, index=False)
    print(f"\n--- VICTORY! Evaluation complete! Results saved to '{RESULTS_CSV_FILE}' ---")

if __name__ == "__main__":
    main()
