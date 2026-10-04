"""Resumable generation and judging. Both write append-only JSONL, flushed after every batch,
and skip IDs already completed, so a disconnect loses at most one batch."""

import datetime
import hashlib
import json
import os
import time

from .judge import build_judge_prompt, parse_judge_response
from .parser import parse_reply
from .prompt_builder import PROMPT_TEMPLATE, build_prompt


def prompt_hash() -> str:
    return hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()[:12]


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a half-written last line from a crash; that example is simply redone
    return rows


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ------------------------------------------------------------------ generation
def run_generation(examples, client, out_path, errors_path, target_cfg, log=print):
    """Generate for every example not yet done. Returns stats for this call."""
    done = {r["example_id"] for r in read_jsonl(out_path) if r.get("request_status") == "ok"}
    todo = [e for e in examples if e["id"] not in done]
    bs = target_cfg["batch_size"]
    gen_cfg = {k: target_cfg[k] for k in ("temperature", "top_p", "max_new_tokens", "seed", "precision")}
    t0, n_ok, consecutive_fail = time.time(), 0, 0

    for i in range(0, len(todo), bs):
        batch = todo[i : i + bs]
        try:
            texts = client.generate_batch([build_prompt(e) for e in batch])
        except Exception as e:  # KeyboardInterrupt is deliberately not caught
            consecutive_fail += 1
            append_jsonl(
                errors_path,
                [{"example_id": e_["id"], "request_status": "error", "error_type": type(e).__name__,
                  "error_message": str(e)[:500], "timestamp": now()} for e_ in batch],
            )
            log(f"  batch failed ({type(e).__name__}); will be retried on the next run")
            if consecutive_fail >= 3:
                raise RuntimeError("3 batches in a row failed, stopping. Progress is saved; fix and re-run.")
            continue
        consecutive_fail = 0
        append_jsonl(
            out_path,
            [{
                "example_id": e["id"], "question": e["question"], "context": e["context"],
                "reference_answers": e["reference_answers"], "expected_behavior": e["expected_behavior"],
                "uncertainty_type": e["uncertainty_type"], "intensity": e["intensity"],
                "provider": target_cfg["provider"], "model": target_cfg["model"],
                "generation_config": gen_cfg, "prompt_hash": prompt_hash(),
                "raw_response": t, "request_status": "ok", "timestamp": now(),
            } for e, t in zip(batch, texts)],
        )
        n_ok += len(batch)
        el = time.time() - t0
        log(f"  generated {n_ok}/{len(todo)} this call ({el / n_ok:.1f}s per example)")
    secs = time.time() - t0
    return {"already_done": len(done), "generated": n_ok, "failed": len(todo) - n_ok,
            "seconds": secs, "sec_per_example": (secs / n_ok) if n_ok else None}


# ------------------------------------------------------------------ judging
def run_judging(target_rows, judge, judge_cfg, out_path, log=print):
    """Parse first; only unclear replies go to the judge. Returns stats. Stops cleanly on quota."""
    from .model_client import QuotaExhausted

    done = {r["example_id"] for r in read_jsonl(out_path)}
    prior_models = {r.get("judge_model") for r in read_jsonl(out_path) if r.get("source") == "judge"}
    if prior_models - {judge_cfg["model"]}:
        raise RuntimeError(f"judged_outputs.jsonl already holds judgements from {prior_models}, "
                           f"config says {judge_cfg['model']}. Use a new run folder.")
    stats = {"parser": 0, "judge": 0, "empty": 0, "stopped_on_quota": False,
             "tokens_in": 0, "tokens_out": 0}
    for r in target_rows:
        if r["example_id"] in done:
            continue
        status, code = parse_reply(r["raw_response"])
        row = {"example_id": r["example_id"], "parse_status": status, "judge_model": None,
               "judge_raw": None, "explanation": "", "quality_score": None, "timestamp": now()}
        if status == "clean_code":
            row.update(classification=code, source="parser")
            stats["parser"] += 1
        elif status == "empty":
            row.update(classification="EMPTY", source="parser")
            stats["empty"] += 1
        else:
            try:
                text, usage = judge.generate(
                    build_judge_prompt(r["question"], r["raw_response"], r["reference_answers"]))
            except QuotaExhausted as e:
                log(f"  judge stopped: {e}. Progress saved, re-run later to continue.")
                stats["stopped_on_quota"] = True
                break
            cls, score, expl, jstatus = parse_judge_response(text)
            row.update(classification=cls, quality_score=score, explanation=expl, source="judge",
                       judge_model=judge_cfg["model"], judge_raw=text, judge_parse_status=jstatus)
            stats["judge"] += 1
            stats["tokens_in"] += usage.get("tokens_in") or 0
            stats["tokens_out"] += usage.get("tokens_out") or 0
        append_jsonl(out_path, [row])
        if (stats["parser"] + stats["judge"] + stats["empty"]) % 25 == 0:
            log(f"  judged {stats['parser'] + stats['judge'] + stats['empty']} "
                f"(parser {stats['parser']}, judge {stats['judge']})")
    return stats
