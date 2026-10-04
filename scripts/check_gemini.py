"""List Gemini models visible to our key (no generation, no cost). Never prints the key."""
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"].strip())
names = sorted(m.name for m in client.models.list())
print(len(names), "models visible")
for n in names:
    if "gemini" in n and any(t in n for t in ("2.5", "flash", "pro")):
        print(" ", n)
