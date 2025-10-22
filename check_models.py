import google.genai as genai
import os
from dotenv import load_dotenv

# Load API key from your .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in .env file.")
else:
    try:
        # Use the NEW client-based syntax to configure
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        print("--- Available Models for your API Key ---")
        
        # Use the NEW client-based syntax to list models
        # We will just print the name, as that's all we need.
        for m in client.models.list():
            print(f"Model name: {m.name}")

        print("------------------------------------------")
        print("\nFind a model name in the list above (e.g., 'models/gemini-pro')")
        print("and paste it into the 'model_name' variable in your main.py")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nThis might be an API key error or a connection issue.")