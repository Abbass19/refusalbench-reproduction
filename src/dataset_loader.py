"""Load RefusalBench-NQ and pick a deterministic, stratified smoke-test subset."""

import json
import random
from collections import defaultdict
from pathlib import Path

from .schema_adapter import adapt_row

HF_DATASET = "aashiqmuhamed/RefusalBench-NQ"
HF_SPLIT = "test"


def load_refusalbench_nq() -> list[dict]:
    from datasets import load_dataset  # imported lazily so other modules stay light

    ds = load_dataset(HF_DATASET, split=HF_SPLIT)
    return [adapt_row(r) for r in ds]


def select_smoke_sample(examples: list[dict], n: int = 20, seed: int = 0) -> list[dict]:
    """Stratified, seeded sample.

    About one third answerable (the benchmark's natural ratio is 536/1600), spread
    across the six perturbation classes. Refusal cases are spread across the six
    refusal categories and alternate MEDIUM/HIGH intensity (the released data has
    no LOW-intensity refusal cases).
    """
    rng = random.Random(seed)
    n_answer = round(n * 536 / 1600)
    n_refuse = n - n_answer

    answerable = defaultdict(list)
    refusals = defaultdict(lambda: defaultdict(list))
    for ex in examples:
        if ex["is_answerable"]:
            answerable[ex["uncertainty_type"]].append(ex)
        else:
            refusals[ex["expected_behavior"]][ex["intensity"]].append(ex)

    chosen: list[dict] = []

    # Answerable: round-robin over perturbation classes.
    classes = sorted(answerable)
    for c in classes:
        rng.shuffle(answerable[c])
    for i in range(n_answer):
        chosen.append(answerable[classes[i % len(classes)]].pop())

    # Refusals: round-robin over refusal codes, alternating intensity within each.
    codes = sorted(refusals)
    for code in codes:
        for pool in refusals[code].values():
            rng.shuffle(pool)
    taken = defaultdict(int)
    for i in range(n_refuse):
        code = codes[i % len(codes)]
        intensities = sorted(refusals[code])  # ['HIGH', 'MEDIUM']
        chosen.append(refusals[code][intensities[taken[code] % len(intensities)]].pop())
        taken[code] += 1

    return sorted(chosen, key=lambda e: e["id"])


def save_sample_ids(sample: list[dict], path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": seed, "n": len(sample), "ids": [e["id"] for e in sample]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_sample_by_ids(examples: list[dict], path: Path) -> list[dict]:
    ids = json.loads(path.read_text(encoding="utf-8"))["ids"]
    by_id = {e["id"]: e for e in examples}
    return [by_id[i] for i in ids]
