"""Tiny judge sanity test: 4 hand-made cases against a Gemini model. Free-tier, a handful of calls.

Usage: python scripts/test_judge.py [model]
"""

import os
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

sys.path.insert(0, ".")
from src.judge import build_judge_prompt, parse_judge_response  # noqa: E402

load_dotenv()
model = sys.argv[1] if len(sys.argv) > 1 else "gemini-2.5-flash"
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"].strip())

CASES = [
    ("who wrote yakety yak", "The song was written by Jerry Leiber and Mike Stoller.", ["Jerry Leiber and Mike Stoller"], "answer_attempt, score 4-5"),
    ("who wrote yakety yak", "REFUSE_AMBIGUOUS_QUERY", None, "REFUSE_AMBIGUOUS_QUERY"),
    ("when did the east india company reach india", "The company reached India in 1492.", ["1611"], "answer_attempt, score 1-3"),
    ("who wrote yakety yak", "I cannot answer, the passage lists two different writers for the song.", None, "unclear: refusal w/o code"),
]

for q, resp, refs, expect in CASES:
    t0 = time.time()
    out = client.models.generate_content(
        model=model,
        contents=build_judge_prompt(q, resp, refs),
        config=types.GenerateContentConfig(temperature=0.0),
    )
    cls, score, expl, status = parse_judge_response(out.text or "")
    u = out.usage_metadata
    print(f"expect[{expect}] -> {cls} | score={score} | parse={status} | "
          f"tokens in={u.prompt_token_count} out={u.candidates_token_count} think={getattr(u, 'thoughts_token_count', None)} | {time.time()-t0:.1f}s")
