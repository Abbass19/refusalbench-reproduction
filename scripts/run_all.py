"""ONE CLICK: run the whole baseline. Safe to stop and re-run: it resumes where it left off.

Order: load data -> generate 10 -> check -> generate 100 -> check -> generate the rest (1,600 total)
       -> judge (parse first) -> metrics -> report -> hand-check sheet.

Usage (Colab does this for you):  python scripts/run_all.py --out /content/drive/MyDrive/refusalbench-results
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from collections import Counter

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset_loader import load_refusalbench_nq, select_smoke_sample  # noqa: E402
from src.model_client import make_judge, make_target  # noqa: E402
from src.parser import parse_reply  # noqa: E402
from src.report import write_all  # noqa: E402
from src.runner import now, prompt_hash, read_jsonl, run_generation, run_judging  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen15_7b_baseline.yaml")
    ap.add_argument("--out", default="results", help="base folder; the run lives in <out>/<run_name>")
    ap.add_argument("--fake", action="store_true", help="TEST ONLY: fake target and judge, no GPU, no API")
    ap.add_argument("--dev-subset", type=int, default=0, help="TEST ONLY: use a small stratified subset")
    ap.add_argument("--skip-judge", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.fake:
        cfg = copy.deepcopy(cfg)
        cfg["target"]["provider"], cfg["judge"]["provider"] = "fake", "fake"
        cfg["run_name"] += "_FAKE_TEST"
    run_dir = os.path.join(args.out, cfg["run_name"])
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "run.log")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{now()}  {msg}\n")

    log(f"=== RefusalBench baseline | run folder: {run_dir}")

    # ---- 1. dataset (never modified, never filtered)
    examples = load_refusalbench_nq()
    n_ans = sum(e["is_answerable"] for e in examples)
    manifest = {"source": "aashiqmuhamed/RefusalBench-NQ (test)", "n": len(examples), "answerable": n_ans,
                "refusal_required": len(examples) - n_ans,
                "by_uncertainty_type": dict(Counter(e["uncertainty_type"] for e in examples)),
                "by_intensity": dict(Counter(e["intensity"] for e in examples))}
    if args.dev_subset:
        log(f"!!! DEV SUBSET of {args.dev_subset}: for testing only, not a real run")
        examples = select_smoke_sample(examples, n=args.dev_subset, seed=0)
    elif (manifest["n"], n_ans) != (1600, 536):
        raise SystemExit(f"Dataset is not the expected 1,600 / 536 answerable: {manifest}")
    with open(os.path.join(run_dir, "dataset_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log(f"dataset OK: {manifest['n']} examples, {n_ans} answerable, {manifest['refusal_required']} refusal-required")

    # ---- 2. config lock: refuse to mix settings inside one run folder
    lock = {"target": {k: cfg["target"][k] for k in ("provider", "model", "precision", "temperature",
                                                     "top_p", "max_new_tokens", "seed")},
            "prompt_hash": prompt_hash(), "dev_subset": args.dev_subset}
    lock_path = os.path.join(run_dir, "config.json")
    if os.path.exists(lock_path):
        if json.load(open(lock_path, encoding="utf-8")) != lock:
            raise SystemExit("This run folder was started with different settings. Change run_name to start fresh.")
    else:
        json.dump(lock, open(lock_path, "w", encoding="utf-8"), indent=2)
    with open(os.path.join(run_dir, "config_full.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    out_path = os.path.join(run_dir, "target_outputs.jsonl")
    err_path = os.path.join(run_dir, "errors.jsonl")
    by_id = {e["id"]: e for e in examples}

    # ---- 3. nested stages: 10 -> 100 -> everything
    sample_pool = list(examples) if args.dev_subset else load_refusalbench_nq()
    cumulative, ordered_ids = [], []
    for size in cfg["stages"]:
        for e in select_smoke_sample(sample_pool, n=min(size, len(sample_pool)), seed=0):
            if e["id"] in by_id and e["id"] not in ordered_ids:
                ordered_ids.append(e["id"])
        cumulative.append(list(ordered_ids))
    rest = sorted(i for i in by_id if i not in set(ordered_ids))
    cumulative.append(ordered_ids + rest)
    labels = [f"stage {s}" for s in cfg["stages"]] + ["all remaining"]

    client = None
    for label, ids in zip(labels, cumulative):
        have = {r["example_id"] for r in read_jsonl(out_path)}
        todo = [i for i in ids if i not in have]
        if todo and client is None:
            log(f"loading the target model {cfg['target']['model']} ({cfg['target']['precision']}) ...")
            client = make_target(cfg["target"])
        log(f"--- {label}: {len(ids)} examples, {len(todo)} still to generate")
        stats = run_generation([by_id[i] for i in ids], client, out_path, err_path, cfg["target"], log)
        rows = [r for r in read_jsonl(out_path) if r["example_id"] in set(ids)]

        # automatic checks (no human needed)
        ok_rate = len(rows) / len(ids)
        empty_rate = sum(1 for r in rows if parse_reply(r["raw_response"])[0] == "empty") / max(1, len(rows))
        clean_rate = sum(1 for r in rows if parse_reply(r["raw_response"])[0] == "clean_code") / max(1, len(rows))
        log(f"    check: {len(rows)}/{len(ids)} ok ({ok_rate:.0%}), empty replies {empty_rate:.0%}, "
            f"clean REFUSE_* code {clean_rate:.0%}")
        if ok_rate < 0.8 or empty_rate > 0.3:
            raise SystemExit("STOPPED by automatic check: too many failed or empty replies. "
                             f"See {err_path} and {out_path}. Progress is saved.")
        if label == labels[0]:
            for r in rows[:3]:
                log(f"    sample reply ({r['expected_behavior']}): {r['raw_response'][:150]!r}")
        if stats["sec_per_example"]:
            eta = stats["sec_per_example"] * (len(by_id) - len(read_jsonl(out_path))) / 60
            log(f"    speed {stats['sec_per_example']:.1f}s per example, about {eta:.0f} min left for the rest")

    target_rows = read_jsonl(out_path)
    ids_done = {r["example_id"] for r in target_rows}
    assert len(ids_done) == len(target_rows) == len(by_id), "duplicate or missing IDs in target_outputs.jsonl"
    log(f"generation complete: {len(ids_done)} unique IDs, no gaps, no duplicates")

    # ---- 4. judge (parse first)
    judged_path = os.path.join(run_dir, "judged_outputs.jsonl")
    judge_stats = {"parser": 0, "judge": 0}
    if args.skip_judge:
        log("judge skipped (--skip-judge)")
    else:
        try:
            judge = make_judge(cfg["judge"])
        except RuntimeError as e:
            judge = None
            log(f"judge NOT run: {e}. Add the key (Colab: Secrets, name GEMINI_API_KEY) and run this cell again; "
                "generation will not be repeated.")
        if judge:
            log("--- judging (clean REFUSE_* replies scored by code, the rest by the judge)")
            judge_stats = run_judging(target_rows, judge, cfg["judge"], judged_path, log)
            log(f"    judge stats: {judge_stats}")

    # ---- 5. metrics, report, hand-check sheet
    judged_rows = read_jsonl(judged_path)
    out, assessment = write_all(run_dir, cfg, target_rows, judged_rows, [], judge_stats)
    o = out["overall"]
    log("=== DONE")
    log(f"scored {o['n_scored']}/{len(target_rows)} | answer acc {o['answer_accuracy']} | "
        f"refusal acc {o['refusal_accuracy']} | false refusal {o['false_refusal_rate']} | "
        f"missed refusal {o['missed_refusal_rate']}")
    log(f"assessment (provisional): {assessment}")
    log(f"report: {os.path.join(run_dir, 'reproduction_report.md')}")
    log(f"to validate the judge, fill in {os.path.join(run_dir, 'handcheck_sheet.csv')} (about 30 rows)")


if __name__ == "__main__":
    main()
