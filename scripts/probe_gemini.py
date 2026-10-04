"""Probe which Gemini models accept a 1-token-ish call on this key. Tiny free-tier usage."""
import os, sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"].strip())
for m in sys.argv[1:]:
    try:
        out = client.models.generate_content(
            model=m, contents="Reply with the single word: ok",
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=200),
        )
        u = out.usage_metadata
        print(f"OK   {m}: {out.text!r} in={u.prompt_token_count} out={u.candidates_token_count} think={getattr(u,'thoughts_token_count',None)}")
    except Exception as e:
        print(f"FAIL {m}: {type(e).__name__} {str(e)[:160]}")
