import os
from dotenv import load_dotenv
from google import genai

# Load variables from .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found.")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="In one sentence, explain what RAG means in AI."
)

print("\nLLM TEST")
print("=" * 50)
print(response.text)