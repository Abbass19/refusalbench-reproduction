"""Plain-assert tests, run with: python tests/selftest.py   (no network, no GPU, no API)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.judge import parse_judge_response  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.parser import parse_reply  # noqa: E402

# ---- parser
assert parse_reply("REFUSE_AMBIGUOUS_QUERY") == ("clean_code", "REFUSE_AMBIGUOUS_QUERY")
assert parse_reply("**`REFUSE_NONFACTUAL_QUERY`**") == ("clean_code", "REFUSE_NONFACTUAL_QUERY")
assert parse_reply("REFUSE_OTHER because REFUSE_OTHER") == ("clean_code", "REFUSE_OTHER")
assert parse_reply("REFUSE_AMBIGUOUS_QUERY or REFUSE_OTHER")[0] == "multiple_codes"
assert parse_reply("REFUSE_MADE_UP")[0] == "unknown_code"
assert parse_reply("I cannot answer this.")[0] == "no_code"
assert parse_reply("   ")[0] == "empty" and parse_reply(None)[0] == "empty"

# ---- judge parser
assert parse_judge_response("CLASSIFICATION: answer_attempt\nQUALITY_SCORE: 5\nEXPLANATION: ok")[:2] == ("answer_attempt", 5.0)
assert parse_judge_response("CLASSIFICATION: REFUSE_OTHER\nQUALITY_SCORE: N/A\nEXPLANATION: x")[:2] == ("REFUSE_OTHER", None)
cls, score, _, status = parse_judge_response("garbage mentioning REFUSE_NONFACTUAL_QUERY")
assert (cls, status) == ("REFUSE_NONFACTUAL_QUERY", "fallback_code")
assert parse_judge_response("garbage")[3] == "fallback_default"

# ---- metrics against a hand-computed example
A = "ANSWER_CORRECTLY"
rows = [
    # 4 answerable: 2 correct answers (score 5, 4), 1 low-score answer (3), 1 false refusal
    dict(id=1, expected=A, answerable=True, type="t", intensity="LOW", pred="answer_attempt", score=5.0),
    dict(id=2, expected=A, answerable=True, type="t", intensity="LOW", pred="answer_attempt", score=4.0),
    dict(id=3, expected=A, answerable=True, type="t", intensity="LOW", pred="answer_attempt", score=3.0),
    dict(id=4, expected=A, answerable=True, type="t", intensity="LOW", pred="REFUSE_OTHER", score=None),
    # 4 unanswerable: 2 exact refusals, 1 wrong category, 1 missed refusal
    dict(id=5, expected="REFUSE_AMBIGUOUS_QUERY", answerable=False, type="t", intensity="HIGH", pred="REFUSE_AMBIGUOUS_QUERY", score=None),
    dict(id=6, expected="REFUSE_NONFACTUAL_QUERY", answerable=False, type="t", intensity="HIGH", pred="REFUSE_NONFACTUAL_QUERY", score=None),
    dict(id=7, expected="REFUSE_NONFACTUAL_QUERY", answerable=False, type="t", intensity="HIGH", pred="REFUSE_OTHER", score=None),
    dict(id=8, expected="REFUSE_NONFACTUAL_QUERY", answerable=False, type="t", intensity="HIGH", pred="answer_attempt", score=None),
]
m = compute_metrics(rows)
assert m["answer_accuracy"] == 2 / 4
assert m["false_refusal_rate"] == 1 / 4
assert m["refusal_accuracy"] == 2 / 4
assert m["missed_refusal_rate"] == 1 / 4
assert m["overall_refusal_rate_on_unanswerable"] == 3 / 4
assert m["calibrated_refusal_score"] == 0.5 * 0.5 + 0.5 * 0.5
# detection: positive = refuse. tp=3 (unanswerable refused), fp=1 (false refusal), fn=1 (missed)
assert abs(m["refusal_detection_precision"] - 3 / 4) < 1e-9 and abs(m["refusal_detection_recall"] - 3 / 4) < 1e-9
assert abs(m["refusal_detection_f1"] - 0.75) < 1e-9

print("selftest: all assertions passed")
