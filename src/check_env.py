import os
from dotenv import load_dotenv

print("--- Starting .env File Check ---")

# 1. Check where the script is running and if the .env file exists
cwd = os.getcwd()
print(f"1. Current Directory: {cwd}")
env_path = os.path.join(cwd, '.env')
print(f"   Looking for .env file at: {env_path}")
if os.path.exists(env_path):
    print("   [SUCCESS] .env file FOUND.")
else:
    print("   [FAILURE] .env file NOT FOUND in this directory.")
    print("      -> FIX: Make sure your .env file is in the main 'genai-multi-agent-chatbot' folder, NOT inside 'src'.")

# 2. Try to load the file
load_successful = load_dotenv()
if load_successful:
    print("2. [SUCCESS] The dotenv library was able to load the file.")
else:
    print("2. [FAILURE] The dotenv library could NOT load the file. This often means the file is empty or has formatting errors.")

# 3. Check for the specific keys
print("3. Checking for the API keys:")
groq_key = os.getenv("GROQ_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

if groq_key:
    print("   [SUCCESS] Found the GROQ_API_KEY.")
else:
    print("   [FAILURE] Could NOT find the GROQ_API_KEY.")
    print("      -> FIX: Make sure the line 'GROQ_API_KEY=\"...\"' exists and is spelled correctly.")

if tavily_key:
    print("   [SUCCESS] Found the TAVILY_API_KEY.")
else:
    print("   [FAILURE] Could NOT find the TAVILY_API_KEY.")
    print("      -> FIX: Make sure the line 'TAVILY_API_KEY=\"...\"' exists and is spelled correctly.")

print("\n--- Check Complete ---")
