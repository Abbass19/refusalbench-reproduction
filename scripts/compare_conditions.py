"""Compare baseline vs format-only vs category-first on the same examples.

Usage:
  python scripts/compare_conditions.py --baseline-dir <run> --format-only-dir <run> --category-first-dir <run> --out <folder>
The baseline run covers all 1,600; it is restricted to the examples the other conditions share.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compare import paired_bootstrap, render_report  # noqa: E402
from src.dataset_loader import load_refusalbench_nq  # noqa: E402
from src.metrics import join_rows  # noqa: E402
from src.runner import read_jsonl  # noqa: E402


def load(run_dir, source_by_id):
    target = read_jsonl(os.path.join(run_dir, "target_outputs.jsonl"))
    judged = read_jsonl(os.path.join(run_dir, "judged_outputs.jsonl"))
    rows, unjudged = join_rows(target, judged)
    for r in rows:
        r["source"] = source_by_id[r["id"]]
    if unjudged:
        print(f"WARNING {run_dir}: {unjudged} replies are not judged yet and are left out")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--format-only-dir", required=True)
    ap.add_argument("--category-first-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resamples", type=int, default=2000)
    args = ap.parse_args()

    source_by_id = {e["id"]: e["source_id"] for e in load_refusalbench_nq()}
    rows = {"baseline": load(args.baseline_dir, source_by_id),
            "format_only": load(args.format_only_dir, source_by_id),
            "category_first": load(args.category_first_dir, source_by_id)}
    names = {"baseline": "baseline", "format_only": "format-only", "category_first": "category-first"}
    pairs = [("category_first", "format_only"), ("category_first", "baseline"), ("format_only", "baseline")]

    res = paired_bootstrap(rows, pairs, B=args.resamples)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "novelty_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_report(res, names, args.resamples))
    print(open(path, encoding="utf-8").read())
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
