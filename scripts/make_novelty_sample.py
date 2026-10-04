"""Build the stratified 800-example novelty sample (and the other 800 as a dev pool for prompt pilots).

Strata: answerable x uncertainty type (LOW only), refusal x refusal code x intensity.
Takes half of every stratum (rounding resolved by a seeded draw), so all ~100 source questions stay represented.
Usage: python scripts/make_novelty_sample.py
"""

import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset_loader import load_refusalbench_nq  # noqa: E402

SEED, TARGET = 0, 800
OUT = "data/novelty_sample_ids.json"


def main():
    examples = load_refusalbench_nq()
    rng = random.Random(SEED)
    strata = defaultdict(list)
    for e in examples:
        key = (("answer", e["uncertainty_type"]) if e["is_answerable"] else ("refuse", e["expected_behavior"]),
               e["intensity"])
        strata[key].append(e)

    chosen, leftovers = [], []
    for key in sorted(strata):
        pool = sorted(strata[key], key=lambda e: e["id"])
        rng.shuffle(pool)
        half = len(pool) // 2
        chosen += pool[:half]
        if len(pool) % 2:
            leftovers.append(pool[half])
    rng.shuffle(leftovers)
    chosen += leftovers[: TARGET - len(chosen)]
    assert len(chosen) == TARGET, len(chosen)

    ids = sorted(e["id"] for e in chosen)
    dev = sorted(e["id"] for e in examples if e["id"] not in set(ids))
    sources_all = {e["source_id"] for e in examples}
    per_source = Counter(e["source_id"] for e in chosen)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "n": len(ids), "ids": ids, "dev_pool_ids": dev}, f, indent=1)

    print(f"wrote {OUT}: {len(ids)} sample ids, {len(dev)} dev-pool ids")
    print(f"answerable {sum(e['is_answerable'] for e in chosen)} / refusal {sum(not e['is_answerable'] for e in chosen)}")
    print(f"source questions covered: {len(per_source)}/{len(sources_all)}; per source min {min(per_source.values())} "
          f"max {max(per_source.values())}")
    print("intensity:", dict(Counter(e["intensity"] for e in chosen)))
    print("expected :", dict(Counter(e["expected_behavior"] for e in chosen)))


if __name__ == "__main__":
    main()
