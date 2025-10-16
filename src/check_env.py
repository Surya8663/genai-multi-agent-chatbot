# src/check_env.py
import os
from dotenv import load_dotenv

print("--- Starting Environment Check ---")

# 1. Print the current directory to see where the script is running from
cwd = os.getcwd()
print(f"Current Working Directory: {cwd}")

# 2. Check if the .env file exists in this directory
env_path = os.path.join(cwd, '.env')
print(f"Looking for .env file at: {env_path}")
if os.path.exists(env_path):
    print(">>> SUCCESS: .env file FOUND!")
else:
    print(">>> FAILURE: .env file NOT FOUND in this directory.")
    print(">>> Please make sure your .env file is in the main 'genai-multi-agent-chatbot' folder.")

# 3. Try to load the .env file
load_successful = load_dotenv()
if load_successful:
    print(">>> SUCCESS: dotenv loaded the file.")
else:
    print(">>> WARNING: dotenv did not load the file (this can happen if the file is empty).")


# 4. Check for the specific environment variable
api_key = os.environ.get("GOOGLE_API_KEY")

print("\n--- Checking for the GOOGLE_API_KEY variable ---")
if api_key:
    print(">>> SUCCESS: The GOOGLE_API_KEY was loaded successfully!")
    # Let's print the first 5 and last 5 characters to confirm it's not empty
    print(f"   Loaded Key starts with: '{api_key[:5]}' and ends with: '{api_key[-5:]}'")
else:
    print(">>> FAILURE: The GOOGLE_API_KEY could NOT be loaded from the environment.")
    print(">>> This means the variable name is likely misspelled in your .env file or the file is empty.")

print("\n--- End of Check ---")
