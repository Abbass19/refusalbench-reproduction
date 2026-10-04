"""Smoke test entry point. Stage 1: dry-run only (prints prompts, makes no API calls)."""

import argparse
from collections import Counter
from pathlib import Path

from src.dataset_loader import (
    load_refusalbench_nq,
    load_sample_by_ids,
    save_sample_ids,
    select_smoke_sample,
)
from src.prompt_builder import build_prompt

SAMPLE_FILE = Path("data/smoke_sample_ids.json")
HARD_MAX = 50  # guard: refuse anything near a full-dataset run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20, help=f"number of examples (max {HARD_MAX})")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--model", default="gpt-4o-2024-08-06")
    ap.add_argument("--dry-run", action="store_true", help="print prompts only; no API calls")
    ap.add_argument("--show", type=int, default=2, help="how many full prompts to print in dry-run")
    args = ap.parse_args()

    if not 1 <= args.limit <= HARD_MAX:
        ap.error(f"--limit must be between 1 and {HARD_MAX}")
    if not args.dry_run:
        raise SystemExit("API calls are not enabled yet (stage 1). Re-run with --dry-run.")

    examples = load_refusalbench_nq()
    print(f"Loaded {len(examples)} RefusalBench-NQ examples")

    if SAMPLE_FILE.exists() and args.limit == 20 and args.seed == 0:
        sample = load_sample_by_ids(examples, SAMPLE_FILE)
        print(f"Using saved sample {SAMPLE_FILE}")
    else:
        sample = select_smoke_sample(examples, n=args.limit, seed=args.seed)
        if args.limit == 20 and args.seed == 0:
            save_sample_ids(sample, SAMPLE_FILE, args.seed)
            print(f"Saved sample ids to {SAMPLE_FILE}")

    print(f"\nSample size: {len(sample)}  (provider={args.provider}, model={args.model})")
    print("expected behavior:", dict(Counter(e["expected_behavior"] for e in sample)))
    print("uncertainty type :", dict(Counter(e["uncertainty_type"] for e in sample)))
    print("intensity        :", dict(Counter(e["intensity"] for e in sample)))

    for ex in sample[: args.show]:
        print("\n" + "=" * 90)
        print(f"id={ex['id']}  expected={ex['expected_behavior']}  {ex['uncertainty_type']}/{ex['intensity']}")
        print("--- prompt that WOULD be sent as a single user message ---")
        print(build_prompt(ex))

    chars = sum(len(build_prompt(e)) for e in sample)
    print("=" * 90)
    print(f"DRY RUN: {len(sample)} requests would be sent; ~{chars // 4} input tokens total (rough chars/4). No API calls made.")


if __name__ == "__main__":
    main()
