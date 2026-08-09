"""Check that prefix-cached scoring equals cache-free scoring.

The cached path is the one every experiment uses, and a silent cache bug would
be indistinguishable from a real effect. This script re-scores a random sample
of (item, condition, candidate) triples both ways and reports the largest
disagreement. Anything above ~1e-3 nats on a 30-token continuation is a bug,
not floating-point noise.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".hf"))

from slmecho.models import REGISTRY, load_model  # noqa: E402
from slmecho.probe import CORE_CONDITIONS, build_condition_messages, load_items  # noqa: E402
from slmecho.runner import _length_matched_neutral  # noqa: E402
from slmecho.scoring import CachedScorer  # noqa: E402
from slmecho.seeding import seed_everything  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smollm2-135m")
    ap.add_argument("--items", default="data/probe_toolshed.jsonl")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    seed_everything(20260808)
    spec = REGISTRY[args.model]
    tok, model, meta = load_model(spec, num_threads=args.threads)
    print("loaded", meta)
    scorer = CachedScorer(tok, model, spec.chat_kwargs)

    items = load_items(args.items)
    rng = random.Random(0)
    sample = rng.sample(items, min(args.n, len(items)))

    worst = 0.0
    n_checks = 0
    for item in sample:
        pad = _length_matched_neutral(scorer, item)
        for condition in CORE_CONDITIONS:
            msgs = build_condition_messages(item, condition, neutral_obs_padded=pad)
            text = scorer.render(msgs, add_generation_prompt=True)
            prompt_ids = scorer.encode(text)
            node = scorer.node_for_text(text)
            for cand in (item.failed_action, item.gold_action):
                cand_ids = scorer.encode(cand)
                a = scorer.score(node, cand_ids).logprob
                b = scorer.score_uncached(prompt_ids, cand_ids)
                if not (math.isfinite(a) and math.isfinite(b)):
                    raise SystemExit(
                        f"FAIL: non-finite log-probability (cached={a}, uncached={b}) "
                        f"on item {item.item_id}/{condition}. The numerical path is "
                        f"broken; do not trust any results from this configuration."
                    )
                if a > 0 or b > 0:
                    raise SystemExit(f"FAIL: positive log-probability {a}, {b}")
                worst = max(worst, abs(a - b))
                n_checks += 1
    print(f"checked {n_checks} triples; max |cached - uncached| = {worst:.3e}")
    if worst > args.tol:
        raise SystemExit(f"FAIL: disagreement {worst:.3e} exceeds tol {args.tol}")
    print("OK")


if __name__ == "__main__":
    main()
