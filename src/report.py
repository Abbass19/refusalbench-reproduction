"""Writes metrics.json, CSVs, the hand-check sheet and reproduction_report.md for a run."""

import csv
import json
import os
import platform
import random
import sys
from collections import Counter
from importlib import metadata

from .metrics import (breakdown, compute_metrics, confusion, join_rows, write_breakdown_csv,
                      write_confusion_csv)
from .parser import parse_reply
from .runner import prompt_hash, read_jsonl

# Qwen1.5-7B-Chat on RefusalBench-NQ as reported in the paper (EACL 2026).
PAPER_QWEN7B = {
    "answer_accuracy": (0.561, "exact, Section 4.3 text"),
    "refusal_accuracy": (0.05, "approximate, read from the small Figure 9 plot; text says < 0.17 for all Qwen sizes"),
}
TOLERANCE = 0.05


def env_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ("torch", "transformers", "bitsandbytes", "accelerate", "google-genai", "datasets"):
        try:
            info[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            info[pkg] = None
    try:
        import torch

        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        info["gpu"] = None
    return info


def _f(x):
    return "n/a" if x is None else f"{x:.3f}"


def _table(groups):
    lines = ["| group | n | answer acc | refusal acc | false refusal | missed refusal | detection F1 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for g, m in groups.items():
        lines.append(f"| {g} | {m['n_scored']} | {_f(m['answer_accuracy'])} | {_f(m['refusal_accuracy'])} | "
                     f"{_f(m['false_refusal_rate'])} | {_f(m['missed_refusal_rate'])} | {_f(m['refusal_detection_f1'])} |")
    return "\n".join(lines)


def write_handcheck(path, target_rows, judged_rows, n, seed=0):
    t = {r["example_id"]: r for r in target_rows}
    cands = [j for j in judged_rows if j.get("source") == "judge" and j["example_id"] in t]
    random.Random(seed).shuffle(cands)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["example_id", "question", "reference_answers", "model_reply", "judge_classification",
                    "judge_score", "judge_explanation", "human_ok (y/n)", "human_comment"])
        for j in cands[:n]:
            r = t[j["example_id"]]
            w.writerow([j["example_id"], r["question"], " | ".join(r["reference_answers"] or []),
                        r["raw_response"], j["classification"], j["quality_score"], j["explanation"], "", ""])
    return min(n, len(cands))


def write_all(run_dir, cfg, target_rows, judged_rows, gen_history, judge_stats, condition="baseline",
              missing_ids=None, examples_by_id=None):
    missing_ids = missing_ids or []
    rows, unjudged = join_rows(target_rows, judged_rows)
    overall = compute_metrics(rows)
    by_type, by_int, cm = breakdown(rows, "type"), breakdown(rows, "intensity"), confusion(rows)

    from .conditions import extract_effective

    parse_counts = Counter(parse_reply(extract_effective(condition, r["raw_response"])["effective"])[0]
                           for r in target_rows)
    n_t = len(target_rows) or 1
    clean_rate = parse_counts["clean_code"] / n_t
    env = env_info()

    out = {"overall": overall, "by_uncertainty_type": by_type, "by_intensity": by_int,
           "n_generated": len(target_rows), "n_unjudged": unjudged,
           "reply_parse_counts": dict(parse_counts), "clean_code_rate": clean_rate,
           "judge_stats": judge_stats, "environment": env, "prompt_hash": prompt_hash()}
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    write_breakdown_csv(os.path.join(run_dir, "metrics_by_category.csv"), by_type)
    write_breakdown_csv(os.path.join(run_dir, "metrics_by_intensity.csv"), by_int)
    write_confusion_csv(os.path.join(run_dir, "confusion_matrix.csv"), cm)
    n_hand = write_handcheck(os.path.join(run_dir, "handcheck_sheet.csv"), target_rows, judged_rows,
                             cfg.get("handcheck_size", 30))

    if condition != "baseline":
        flags = Counter(f for j in judged_rows for f in (j.get("flags") or []))
        states = Counter(j.get("evidence_state") for j in judged_rows)
        cond_report = f"""# Condition report: `{condition}` (Qwen1.5-7B-Chat, RefusalBench-NQ subset)

*Generated automatically. Compare conditions with `scripts/compare_conditions.py`. This is not a paper reproduction.*

- Examples generated {len(target_rows)}, scored {len(rows)}, unjudged {unjudged}, empty replies {overall['n_empty_replies']}
- Reply format after extraction: {dict(parse_counts)} (clean code rate {clean_rate:.1%})
- Extraction flags: {dict(flags)}
- Evidence states: {dict(states)}
- Target: `{cfg['target']['model']}` {cfg['target']['precision']}, temperature {cfg['target']['temperature']}, top_p {cfg['target']['top_p']}, seed {cfg['target']['seed']}; judge `{cfg['judge']['model']}`; prompt hash `{prompt_hash()}`

## Results
| metric | value |
|---|---:|
| answerable / unanswerable | {overall['n_answerable']} / {overall['n_unanswerable']} |
| answer accuracy | {_f(overall['answer_accuracy'])} |
| refusal accuracy | {_f(overall['refusal_accuracy'])} |
| false refusal rate | {_f(overall['false_refusal_rate'])} |
| missed refusal rate | {_f(overall['missed_refusal_rate'])} |
| refusal detection F1 | {_f(overall['refusal_detection_f1'])} |
| calibrated refusal score | {_f(overall['calibrated_refusal_score'])} |

### By uncertainty type
{_table(by_type)}

### By intensity
{_table(by_int)}
"""
        with open(os.path.join(run_dir, "condition_report.md"), "w", encoding="utf-8") as f:
            f.write(cond_report)
        return out, "n/a (not a baseline run)"

    # --- comparison with the paper
    cmp_lines = ["| metric | paper (Qwen1.5-7B-Chat) | ours | difference | within +/-0.05 |", "|---|---:|---:|---:|---|"]
    verdicts = []
    for key, (pv, note) in PAPER_QWEN7B.items():
        ours = overall[key]
        if ours is None:
            cmp_lines.append(f"| {key} | {pv} ({note}) | n/a | n/a | n/a |")
            continue
        ok = abs(ours - pv) <= TOLERANCE
        verdicts.append(ok)
        cmp_lines.append(f"| {key} | {pv} ({note}) | {ours:.3f} | {ours - pv:+.3f} | {'yes' if ok else 'no'} |")
    for key in ("false_refusal_rate", "missed_refusal_rate", "calibrated_refusal_score", "refusal_detection_f1"):
        cmp_lines.append(f"| {key} | not extracted from the paper | {_f(overall[key])} | n/a | n/a |")
    if len(verdicts) == 2:
        assessment = ("closely reproduced on both reported metrics" if all(verdicts) else
                      "partially reproduced (one reported metric within tolerance)" if any(verdicts) else
                      "significant deviation on both reported metrics")
    else:
        assessment = "cannot be assessed (metrics missing, judging incomplete)"

    # --- failure patterns (descriptive only)
    wrong = Counter((r["expected"], r["pred"]) for r in rows if not r["answerable"] and r["pred"] != r["expected"])
    top = "\n".join(f"- expected `{e}` but got `{p}`: {c} times" for (e, p), c in wrong.most_common(5)) or "- none"


    # --- missing examples (failed generation), documented rather than hidden
    missing_md = ""
    if missing_ids:
        import statistics as _st

        mex = [examples_by_id[i] for i in missing_ids if i in examples_by_id]
        gotten = [examples_by_id[i] for i in examples_by_id if i not in set(missing_ids)]
        mlen = lambda xs: _st.median(len(e["context"]) + len(e["question"]) for e in xs)  # noqa: E731
        missing_md = f"""
## Missing examples (excluded)
{len(missing_ids)} of {len(examples_by_id)} examples ({len(missing_ids) / len(examples_by_id):.1%}) have no reply because the GPU ran out of
memory on their batch and they were never regenerated. They are excluded from every metric, not counted as errors.
- By expected behavior: {dict(Counter(e["expected_behavior"] for e in mex))}
- By uncertainty type: {dict(Counter(e["uncertainty_type"] for e in mex))}
- Median length (question plus context, characters): missing {mlen(mex):.0f} vs scored {mlen(gotten):.0f}.
  The missing examples are somewhat longer, so the sample is slightly biased toward shorter inputs.
"""
    t = cfg["target"]
    j = cfg["judge"]
    report = f"""# Reproduction report: Qwen1.5-7B-Chat on RefusalBench-NQ

*Generated automatically. The assessment below is provisional until we review it together.*

## Objective
Reproduce, as closely as practical on free resources, the paper's evaluation of Qwen1.5-7B-Chat on the
released RefusalBench-NQ benchmark (1,600 examples), and compare the measured selective-refusal metrics.

## Original setup (paper)
Qwen/Qwen1.5-7B-Chat served with vLLM on A100 GPUs, temperature 1.0, top_p 1.0. The target model replies with an
answer or a `REFUSE_*` code. Judge: Claude-4-Sonnet classifies each reply (answer attempt or refusal code) and scores
answer quality 1-5; a score of 4 or 5 is a correct answer, a refusal is correct on an exact category match.

## Our setup
- Target: `{t['model']}`, precision `{t['precision']}`, temperature {t['temperature']}, top_p {t['top_p']}, max_new_tokens {t['max_new_tokens']}, seed {t['seed']}, batch size {t['batch_size']}
- Prompt: copied verbatim from the original repo (hash `{prompt_hash()}`), sent as a single user message
- Judge: `{j['model']}` ({j['provider']}), temperature {j['temperature']}. Replies with one clean `REFUSE_*` code are scored by code; every other reply goes to the judge
- Environment: {json.dumps(env)}

## Deviations from the paper
1. Quantization: `{t['precision']}` on a Colab T4, the paper ran vLLM on A100 GPUs (likely full precision)
2. Inference stack: Hugging Face transformers instead of vLLM v0.5.1; the chat template adds Qwen's default system prompt
3. Judge: `{j['model']}` instead of Claude-4-Sonnet, and only unclear replies are judged (clean refusal codes are scored by code)
4. max_new_tokens {t['max_new_tokens']} (not stated in the paper); sampling at temperature 1.0 means results vary run to run
5. Dataset: the released Hugging Face version of RefusalBench-NQ

## Data integrity
Generated {len(target_rows)} replies; judged/scored {len(rows)}; unjudged {unjudged}; empty replies {overall['n_empty_replies']}.
Reply format: {dict(parse_counts)} (clean `REFUSE_*` code rate {clean_rate:.1%}).
Judge calls: {judge_stats.get('judge', 0)}, scored by parser: {judge_stats.get('parser', 0)}.

{missing_md}
## Results (overall)
| metric | value |
|---|---:|
| answerable / unanswerable examples | {overall['n_answerable']} / {overall['n_unanswerable']} |
| answer accuracy | {_f(overall['answer_accuracy'])} |
| refusal accuracy (exact category) | {_f(overall['refusal_accuracy'])} |
| false refusal rate | {_f(overall['false_refusal_rate'])} |
| missed refusal rate | {_f(overall['missed_refusal_rate'])} |
| refusal rate on unanswerable (any code) | {_f(overall['overall_refusal_rate_on_unanswerable'])} |
| refusal detection F1 (refuse = positive) | {_f(overall['refusal_detection_f1'])} |
| calibrated refusal score (0.5 x answer + 0.5 x refusal accuracy) | {_f(overall['calibrated_refusal_score'])} |

### By uncertainty type
{_table(by_type)}

### By intensity
{_table(by_int)}

Confusion matrix: `confusion_matrix.csv`.

## Comparison with the paper
{chr(10).join(cmp_lines)}

Metrics the paper reports only in charts we did not extract are marked, not guessed.

## Reproducibility assessment (provisional)
**{assessment}** (rule: a metric counts as reproduced when within +/-{TOLERANCE} of the paper value; the paper's refusal
accuracy is itself an approximate read of a small chart). Not a verdict until we review the results and the
judge hand check (`handcheck_sheet.csv`, {n_hand} items to fill in) together.

## Observed failure patterns (descriptive only)
Most common wrong refusal categories:
{top}
"""
    with open(os.path.join(run_dir, "reproduction_report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    return out, assessment
