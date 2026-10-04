"""Paper metrics (definitions follow the original repo's run_models.py / paper Appendix D).

Row = one example joined with its judged classification. `pred` is "answer_attempt",
a REFUSE_* code, or "EMPTY". Examples with no judged row are reported as unjudged, not scored.
"""

import csv
from collections import Counter, defaultdict

from .schema_adapter import ANSWER_LABEL, REFUSAL_CODES


def join_rows(target_rows, judged_rows):
    j = {r["example_id"]: r for r in judged_rows}
    rows, unjudged = [], 0
    for t in target_rows:
        jr = j.get(t["example_id"])
        if jr is None:
            unjudged += 1
            continue
        rows.append({
            "id": t["example_id"], "expected": t["expected_behavior"],
            "answerable": t["expected_behavior"] == ANSWER_LABEL,
            "type": t["uncertainty_type"], "intensity": t["intensity"],
            "pred": jr["classification"], "score": jr.get("quality_score"),
        })
    return rows, unjudged


def _rate(num, den):
    return (num / den) if den else None


def compute_metrics(rows):
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    is_ref = lambda r: r["pred"] in REFUSAL_CODES
    answered = lambda r: r["pred"] == "answer_attempt"

    correct_answers = [r for r in ans if answered(r) and r["score"] is not None and r["score"] >= 4]
    correct_refusals = [r for r in unans if r["pred"] == r["expected"]]
    tp = sum(1 for r in unans if is_ref(r))
    fp = sum(1 for r in ans if is_ref(r))
    fn = sum(1 for r in unans if not is_ref(r))
    prec, rec = _rate(tp, tp + fp), _rate(tp, tp + fn)
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec else (0.0 if prec is not None and rec is not None else None)

    m = {
        "n_scored": len(rows), "n_answerable": len(ans), "n_unanswerable": len(unans),
        "n_empty_replies": sum(1 for r in rows if r["pred"] == "EMPTY"),
        "answer_accuracy": _rate(len(correct_answers), len(ans)),
        "refusal_accuracy": _rate(len(correct_refusals), len(unans)),
        "false_refusal_rate": _rate(fp, len(ans)),
        "missed_refusal_rate": _rate(sum(1 for r in unans if answered(r)), len(unans)),
        "overall_refusal_rate_on_unanswerable": _rate(tp, len(unans)),
        "refusal_detection_precision": prec, "refusal_detection_recall": rec, "refusal_detection_f1": f1,
    }
    a, ra = m["answer_accuracy"], m["refusal_accuracy"]
    m["calibrated_refusal_score"] = (0.5 * a + 0.5 * ra) if a is not None and ra is not None else None
    return m


def breakdown(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return {k: compute_metrics(v) for k, v in sorted(groups.items())}


def confusion(rows):
    """expected label -> predicted label counts."""
    cm = defaultdict(Counter)
    for r in rows:
        cm[r["expected"]][r["pred"]] += 1
    return cm


def write_breakdown_csv(path, groups):
    cols = ["group", "n_scored", "n_answerable", "n_unanswerable", "answer_accuracy", "refusal_accuracy",
            "false_refusal_rate", "missed_refusal_rate", "refusal_detection_f1", "calibrated_refusal_score"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for g, m in groups.items():
            w.writerow([g] + [("" if m[c] is None else round(m[c], 4) if isinstance(m[c], float) else m[c])
                              for c in cols[1:]])


def write_confusion_csv(path, cm):
    preds = ["answer_attempt"] + REFUSAL_CODES + ["EMPTY"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["expected \\ predicted"] + preds)
        for exp in [ANSWER_LABEL] + REFUSAL_CODES:
            if exp in cm:
                w.writerow([exp] + [cm[exp].get(p, 0) for p in preds])
