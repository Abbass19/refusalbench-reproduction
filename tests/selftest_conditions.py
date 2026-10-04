"""Tests for the novelty conditions. Run: python tests/selftest_conditions.py   (no network, no GPU, no API)"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.conditions import STATE_TO_CODE, build_prompt_for, extract_effective  # noqa: E402
from src.prompt_builder import build_prompt  # noqa: E402
from src.schema_adapter import REFUSAL_CODES  # noqa: E402

ex = {"question": "who wrote it", "context": "Some passage."}

# ---- prompts
assert build_prompt_for("baseline", ex) == build_prompt(ex), "baseline prompt must stay verbatim"
fo, cf, base = (build_prompt_for(c, ex) for c in ("format_only", "category_first", "baseline"))
assert "FINAL:" in fo and "EVIDENCE_STATE" not in fo
assert "EVIDENCE_STATE:" in cf and "FINAL:" in cf
for p in (base, fo, cf):  # same definitions in all conditions, query and context inserted
    assert "who wrote it" in p and "Some passage." in p
    for code in REFUSAL_CODES:
        assert code in p
section = base.split("**REFUSAL CODES:**")[1].split("Provide your response below:")[0]
assert section in fo and section in cf, "the definitions text must be identical across conditions"
assert set(STATE_TO_CODE.values()) <= set(REFUSAL_CODES)

# ---- baseline extraction is the identity
assert extract_effective("baseline", "anything")["effective"] == "anything"

# ---- format_only
assert extract_effective("format_only", "thinking\nFINAL: REFUSE_OTHER")["effective"] == "REFUSE_OTHER"
assert extract_effective("format_only", "FINAL: first\nFINAL: second")["effective"] == "second"
assert extract_effective("format_only", "**FINAL:** `The answer is 5`")["effective"] == "The answer is 5"
r = extract_effective("format_only", "no final line here")
assert r["effective"] == "no final line here" and "no_final_line" in r["flags"]

# ---- category_first: a refusal state decides the outcome
r = extract_effective("category_first", "EVIDENCE_STATE: CONTRADICTORY\nFINAL: REFUSE_CONTRADICTORY_CONTEXT")
assert r["effective"] == "REFUSE_CONTRADICTORY_CONTEXT" and r["flags"] == []
r = extract_effective("category_first", "EVIDENCE_STATE: epistemic_mismatch\nFINAL: REFUSE_OTHER")
assert r["effective"] == "REFUSE_NONFACTUAL_QUERY" and "final_conflicts_with_state" in r["flags"]
r = extract_effective("category_first", "EVIDENCE_STATE: AMBIGUOUS\nFINAL: The answer is 5")
assert r["effective"] == "REFUSE_AMBIGUOUS_QUERY" and "final_conflicts_with_state" in r["flags"]
# CLEAR: the FINAL line is the answer
r = extract_effective("category_first", "EVIDENCE_STATE: CLEAR\nFINAL: 1611")
assert r["effective"] == "1611" and r["flags"] == [] and r["state"] == "CLEAR"
r = extract_effective("category_first", "EVIDENCE_STATE: CLEAR\nFINAL: REFUSE_OTHER")
assert r["effective"] == "REFUSE_OTHER" and "clear_but_refused" in r["flags"]
r = extract_effective("category_first", "EVIDENCE_STATE: CLEAR\nIt was 1611.")
assert "1611" in r["effective"] and "no_final_line" in r["flags"]
# malformed
r = extract_effective("category_first", "I think it is fine.\nFINAL: 42")
assert r["effective"] == "42" and "no_valid_state" in r["flags"]
r = extract_effective("category_first", "")
assert "no_valid_state" in r["flags"] and "no_final_line" in r["flags"]

print("selftest_conditions: all assertions passed")
