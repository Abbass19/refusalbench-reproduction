"""Resumable generation and judging. Both write append-only JSONL, flushed after every batch,
and skip IDs already completed, so a disconnect loses at most one batch."""

import datetime
import hashlib
import json
import os
import time

from .judge import build_judge_prompt, parse_judge_response
from .conditions import extract_effective
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
def run_generation(examples, client, out_path, errors_path, target_cfg, log=print, prompt_fn=build_prompt):
    """Generate for every example not yet done. Returns stats for this call."""
    done = {r["example_id"] for r in read_jsonl(out_path) if r.get("request_status") == "ok"}
    todo = [e for e in examples if e["id"] not in done]
    bs = target_cfg["batch_size"]
    gen_cfg = {k: target_cfg[k] for k in ("temperature", "top_p", "max_new_tokens", "seed", "precision")}
    t0, n_ok, consecutive_fail = time.time(), 0, 0

    def make_row(e, t):
        return {
            "example_id": e["id"], "question": e["question"], "context": e["context"],
            "reference_answers": e["reference_answers"], "expected_behavior": e["expected_behavior"],
            "uncertainty_type": e["uncertainty_type"], "intensity": e["intensity"],
            "provider": target_cfg["provider"], "model": target_cfg["model"],
            "generation_config": gen_cfg, "prompt_hash": prompt_hash(),
            "raw_response": t, "request_status": "ok", "timestamp": now(),
            **({"condition": target_cfg["condition"]} if target_cfg.get("condition", "baseline") != "baseline" else {}),
        }

    def free_gpu():
        try:
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    for i in range(0, len(todo), bs):
        batch = todo[i : i + bs]
        # Retries happen OUTSIDE the except blocks: inside them the exception's traceback still holds the failed
        # attempt's GPU tensors, so memory cannot be freed and every retry fails as well. Only strings are kept.
        texts, fail_name = None, None
        try:
            texts = client.generate_batch([prompt_fn(e) for e in batch])
        except Exception as err:  # KeyboardInterrupt is deliberately not caught
            fail_name = type(err).__name__
        if fail_name:
            log(f"  batch failed ({fail_name}); retrying its {len(batch)} examples one by one")
            free_gpu()
            ok_batch, texts = [], []
            for e in batch:
                single, name, msg = None, None, None
                try:
                    single = client.generate_batch([prompt_fn(e)])[0]
                except Exception as err2:
                    name, msg = type(err2).__name__, str(err2)[:500]
                if name:
                    free_gpu()
                    append_jsonl(errors_path, [{"example_id": e["id"], "request_status": "error",
                                                "error_type": name, "error_message": msg, "timestamp": now()}])
                    log(f"  one example failed even alone ({name}); it is retried on the next pass")
                else:
                    ok_batch.append(e)
                    texts.append(single)
            batch = ok_batch
            if not batch:
                consecutive_fail += 1
                if consecutive_fail >= 3:
                    raise RuntimeError("3 batches in a row failed, stopping. Progress is saved; fix and re-run.")
                continue
        consecutive_fail = 0
        append_jsonl(out_path, [make_row(e, t) for e, t in zip(batch, texts)])
        n_ok += len(batch)
        el = time.time() - t0
        log(f"  generated {n_ok}/{len(todo)} this call ({el / n_ok:.1f}s per example)")
    secs = time.time() - t0
    return {"already_done": len(done), "generated": n_ok, "failed": len(todo) - n_ok,
            "seconds": secs, "sec_per_example": (secs / n_ok) if n_ok else None}


# ------------------------------------------------------------------ judging
def run_judging(target_rows, judge, judge_cfg, out_path, log=print, condition="baseline"):
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
        eff = extract_effective(condition, r["raw_response"])
        reply = eff["effective"]
        status, code = parse_reply(reply)
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
                    build_judge_prompt(r["question"], reply, r["reference_answers"]))
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
        if condition != "baseline":
            row.update(effective_reply=reply, evidence_state=eff["state"], flags=eff["flags"])
        append_jsonl(out_path, [row])
        if (stats["parser"] + stats["judge"] + stats["empty"]) % 25 == 0:
            log(f"  judged {stats['parser'] + stats['judge'] + stats['empty']} "
                f"(parser {stats['parser']}, judge {stats['judge']})")
    return stats
