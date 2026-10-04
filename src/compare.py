"""Paired comparison of the three conditions with a cluster bootstrap over source questions.

Examples that come from the same source question are correlated, so we resample whole source questions
(about 100 clusters), not individual examples. All conditions are scored on the same example IDs.
"""

import numpy as np

from .metrics import compute_metrics

METRICS = ["answer_accuracy", "refusal_accuracy", "false_refusal_rate", "missed_refusal_rate",
           "refusal_detection_f1", "calibrated_refusal_score"]
LOWER_IS_BETTER = {"false_refusal_rate", "missed_refusal_rate"}


def restrict_to_common(rows_by_cond):
    common = set.intersection(*(set(r["id"] for r in rows) for rows in rows_by_cond.values()))
    return {c: [r for r in rows if r["id"] in common] for c, rows in rows_by_cond.items()}, len(common)


def paired_bootstrap(rows_by_cond, pairs, B=2000, seed=0):
    """rows_by_cond: {condition: [row with id, source, ...]}. pairs: [(a, b)] meaning a minus b.
    Returns {"point": {cond: metrics}, "diff": {(a,b): {metric: (point, lo, hi)}}, "n_sources", "n_examples"}."""
    rows_by_cond, n = restrict_to_common(rows_by_cond)
    conds = list(rows_by_cond)
    by_src = {c: {} for c in conds}
    for c in conds:
        for r in rows_by_cond[c]:
            by_src[c].setdefault(r["source"], []).append(r)
    sources = sorted(by_src[conds[0]])
    rng = np.random.default_rng(seed)

    point = {c: compute_metrics(rows_by_cond[c]) for c in conds}
    diffs = {p: {m: [] for m in METRICS} for p in pairs}
    for _ in range(B):
        pick = rng.choice(len(sources), size=len(sources), replace=True)
        boot = {c: compute_metrics([r for i in pick for r in by_src[c].get(sources[i], [])]) for c in conds}
        for a, b in pairs:
            for m in METRICS:
                va, vb = boot[a][m], boot[b][m]
                diffs[(a, b)][m].append(np.nan if va is None or vb is None else va - vb)

    out = {}
    for (a, b), per_metric in diffs.items():
        out[(a, b)] = {}
        for m, vals in per_metric.items():
            arr = np.array(vals, dtype=float)
            pa, pb = point[a][m], point[b][m]
            pt = None if pa is None or pb is None else pa - pb
            if np.isnan(arr).all():
                out[(a, b)][m] = (pt, None, None)
            else:
                lo, hi = np.nanpercentile(arr, [2.5, 97.5])
                out[(a, b)][m] = (pt, float(lo), float(hi))
    return {"point": point, "diff": out, "n_sources": len(sources), "n_examples": n}


def _f(x):
    return "n/a" if x is None else f"{x:.3f}"


def _d(t):
    pt, lo, hi = t
    if pt is None or lo is None:
        return "n/a"
    sig = "" if lo <= 0 <= hi else " *"
    return f"{pt:+.3f} [{lo:+.3f}, {hi:+.3f}]{sig}"


def render_report(res, names, B):
    """names: {condition: label}. Markdown."""
    conds = list(res["point"])
    lines = [
        "# Novelty comparison: baseline vs format-only control vs category-first",
        "",
        f"*Generated automatically. {res['n_examples']} paired examples from {res['n_sources']} source questions. "
        f"Intervals are 95% from a paired cluster bootstrap over source questions ({B} resamples). "
        "`*` means the interval excludes zero. Not a verdict until reviewed together.*",
        "",
        "## Metrics per condition",
        "| metric | " + " | ".join(names[c] for c in conds) + " |",
        "|---|" + "---:|" * len(conds),
    ]
    for m in METRICS:
        lines.append(f"| {m} | " + " | ".join(_f(res["point"][c][m]) for c in conds) + " |")
    lines += ["", "## Differences (first minus second)", "For false and missed refusal rate, lower is better.", ""]
    pairs = list(res["diff"])
    lines.append("| metric | " + " | ".join(f"{names[a]} - {names[b]}" for a, b in pairs) + " |")
    lines.append("|---|" + "---:|" * len(pairs))
    for m in METRICS:
        lines.append(f"| {m} | " + " | ".join(_d(res["diff"][p][m]) for p in pairs) + " |")
    lines += ["", "**Headline test:** category-first vs the format-only control. If the gain over the baseline "
              "is mostly matched by the control, the improvement came from output format, not from the evidence-state step."]
    return "\n".join(lines)
