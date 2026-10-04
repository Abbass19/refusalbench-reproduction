"""Tests the generation fallback: a failing batch is retried one by one, and a permanently failing
example is skipped without losing the others. Run: python tests/selftest_runner.py"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.runner import read_jsonl, run_generation  # noqa: E402

CFG = {"batch_size": 4, "provider": "fake", "model": "fake", "temperature": 1.0, "top_p": 1.0,
       "max_new_tokens": 8, "seed": 0, "precision": "4bit"}


def ex(i):
    return {"id": f"id{i}", "question": f"q{i}", "context": f"c{i}", "reference_answers": None,
            "expected_behavior": "REFUSE_OTHER", "uncertainty_type": "t", "intensity": "HIGH"}


class OomOnBatches:
    """Fails for any batch bigger than 1 (like an out-of-memory), and always fails for example id3."""

    def generate_batch(self, prompts):
        if len(prompts) > 1:
            raise RuntimeError("OutOfMemoryError")
        if "q3" in prompts[0]:
            raise RuntimeError("OutOfMemoryError")
        return ["ok"]


d = tempfile.mkdtemp()
out, err = os.path.join(d, "o.jsonl"), os.path.join(d, "e.jsonl")
examples = [ex(i) for i in range(8)]
logs = []
stats = run_generation(examples, OomOnBatches(), out, err, CFG, log=logs.append, prompt_fn=lambda e: e["question"])
rows = read_jsonl(out)
assert sorted(r["example_id"] for r in rows) == sorted(f"id{i}" for i in range(8) if i != 3), "only id3 may be missing"
assert [r["example_id"] for r in read_jsonl(err)] == ["id3"]
assert stats["generated"] == 7 and stats["failed"] == 1

# second pass with a healthy model picks up only the missing example, no duplicates
class Healthy:
    def generate_batch(self, prompts):
        return ["fine"] * len(prompts)


run_generation(examples, Healthy(), out, err, CFG, log=logs.append, prompt_fn=lambda e: e["question"])
ids = [r["example_id"] for r in read_jsonl(out)]
assert len(ids) == 8 and len(set(ids)) == 8, ids
print("selftest_runner: all assertions passed")
