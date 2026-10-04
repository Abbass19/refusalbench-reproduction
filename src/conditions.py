"""The three experimental conditions: baseline, format_only (control), category_first (novelty).

All conditions use the SAME definitions text (the REFUSAL CODES section of the original prompt, verbatim),
so a difference between them comes from the structure of the response, not from extra information.
"""

import re

from .prompt_builder import PROMPT_TEMPLATE, build_prompt
from .schema_adapter import REFUSAL_CODES

CONDITIONS = ("baseline", "format_only", "category_first")

# evidence state -> refusal code (note: the benchmark calls the epistemic class NONFACTUAL)
STATE_TO_CODE = {
    "AMBIGUOUS": "REFUSE_AMBIGUOUS_QUERY",
    "CONTRADICTORY": "REFUSE_CONTRADICTORY_CONTEXT",
    "MISSING_INFORMATION": "REFUSE_INFO_MISSING_IN_CONTEXT",
    "FALSE_PREMISE": "REFUSE_FALSE_PREMISE_IN_QUERY",
    "GRANULARITY_MISMATCH": "REFUSE_GRANULARITY_MISMATCH",
    "EPISTEMIC_MISMATCH": "REFUSE_NONFACTUAL_QUERY",
}
VALID_STATES = ["CLEAR"] + list(STATE_TO_CODE)

_TAIL = "Provide your response below:\n        ANSWER:"
assert _TAIL in PROMPT_TEMPLATE, "baseline template changed, conditions must be re-checked"
_HEAD, _ = PROMPT_TEMPLATE.split(_TAIL)

_FORMAT_ONLY_RULE = """**OUTPUT FORMAT:**
        End your response with exactly one final line in this format:
        FINAL: <your answer, or only the refusal code>

        """

_CATEGORY_FIRST_RULE = """**EVIDENCE STATE (do this first):**
        Before giving a final response, decide the evidence state of the context with respect to the query, choosing exactly one of:
        CLEAR (a complete and faithful answer can be derived solely from the context, none of the issues above applies),
        AMBIGUOUS (= REFUSE_AMBIGUOUS_QUERY), CONTRADICTORY (= REFUSE_CONTRADICTORY_CONTEXT),
        MISSING_INFORMATION (= REFUSE_INFO_MISSING_IN_CONTEXT), FALSE_PREMISE (= REFUSE_FALSE_PREMISE_IN_QUERY),
        GRANULARITY_MISMATCH (= REFUSE_GRANULARITY_MISMATCH), EPISTEMIC_MISMATCH (= REFUSE_NONFACTUAL_QUERY).

        **OUTPUT FORMAT:** respond with exactly two lines:
        EVIDENCE_STATE: <one state from the list>
        FINAL: <if CLEAR, your answer; otherwise only the corresponding refusal code>

        """


def build_prompt_for(condition: str, example: dict) -> str:
    if condition == "baseline":
        return build_prompt(example)
    rule = {"format_only": _FORMAT_ONLY_RULE, "category_first": _CATEGORY_FIRST_RULE}[condition]
    template = _HEAD + rule + _TAIL
    return template.format(query=example["question"], context=example["context"])


# ---------------------------------------------------------------- reply extraction
_FINAL_RE = re.compile(r"^[\s*`>#-]*FINAL[\s*`]*:[\s*`]*(.*)$", re.IGNORECASE | re.MULTILINE)
_STATE_RE = re.compile(r"EVIDENCE_STATE[\s*`]*:[\s*`]*([A-Za-z_]+)", re.IGNORECASE)


def _strip(t: str) -> str:
    return t.strip().strip("`*").strip()


def extract_effective(condition: str, raw: str) -> dict:
    """What gets scored. Returns {effective, state, flags}. `raw` is always preserved elsewhere."""
    raw = raw or ""
    out = {"effective": raw, "state": None, "flags": []}
    if condition == "baseline":
        return out

    finals = _FINAL_RE.findall(raw)
    final = _strip(finals[-1]) if finals else None
    if final is None:
        out["flags"].append("no_final_line")

    if condition == "format_only":
        if final is not None:
            out["effective"] = final
        return out

    # category_first: the evidence state decides refusals; for CLEAR the FINAL line is the answer
    m = _STATE_RE.search(raw)
    state = m.group(1).upper() if m else None
    out["state"] = state
    if state in STATE_TO_CODE:
        code = STATE_TO_CODE[state]
        out["effective"] = code
        if final is not None and code not in final:
            out["flags"].append("final_conflicts_with_state")
    elif state == "CLEAR":
        if final is None:
            out["effective"] = _STATE_RE.sub("", raw).strip()
        else:
            out["effective"] = final
        if any(c in (final or "") for c in REFUSAL_CODES):
            out["flags"].append("clear_but_refused")
    else:
        out["flags"].append("no_valid_state")
        if final is not None:
            out["effective"] = final
    return out
