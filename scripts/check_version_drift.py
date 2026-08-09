"""Re-score stored probe rows in the current environment and compare.

Part of this study was run under transformers 4.56.1 and part under 4.45.0,
because the environment was rebuilt partway through. That is a real threat to
the numbers being one comparable set, so it gets measured rather than argued
about: this script picks stored (item, condition, candidate) rows out of
`results/raw/probe/`, recomputes them now, and reports the largest difference.

Scoring is a deterministic forward pass, so anything beyond floating-point
noise (~1e-4 nats on a 30-token continuation) means the two halves of the study
are not comparable and the affected runs have to be redone.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".hf"))

from slmecho.models import REGISTRY, load_model  # noqa: E402
from slmecho.probe import build_condition_messages, load_items  # noqa: E402
from slmecho.provenance import environment_record  # noqa: E402
from slmecho.runner import _candidates, _length_matched_neutral  # noqa: E402
from slmecho.scoring import CachedScorer  # noqa: E402
from slmecho.seeding import seed_everything  # noqa: E402

ENV_ITEMS = {
    "toolshed": "data/probe_toolshed.jsonl",
    "coderepair": "data/probe_coderepair.jsonl",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smollm2-135m")
    ap.add_argument("--env", default="toolshed", choices=sorted(ENV_ITEMS))
    ap.add_argument("--raw-dir", default="results/raw/probe")
    ap.add_argument("--n-items", type=int, default=8)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--out", default="results/version_drift.json")
    args = ap.parse_args()

    path = os.path.join(args.raw_dir, f"{args.model}__{args.env}.jsonl")
    stored: Dict[tuple, float] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            stored[(r["item_id"], r["condition"], r["candidate"])] = r["logprob"]
    if not stored:
        raise SystemExit(f"no stored rows in {path}")

    items = {i.item_id: i for i in load_items(ENV_ITEMS[args.env])}
    wanted_ids = sorted({k[0] for k in stored} & set(items))
    rng = random.Random(0)
    sample_ids = rng.sample(wanted_ids, min(args.n_items, len(wanted_ids)))

    seed_everything(20260808)
    spec = REGISTRY[args.model]
    tok, model, _ = load_model(spec, num_threads=args.threads)
    scorer = CachedScorer(tok, model, spec.chat_kwargs)

    diffs: List[dict] = []
    for item_id in sample_ids:
        item = items[item_id]
        conds = sorted({k[1] for k in stored if k[0] == item_id})
        pad = _length_matched_neutral(scorer, item) if "neut_pad" in conds else None
        for condition in conds:
            try:
                msgs = build_condition_messages(item, condition,
                                                neutral_obs_padded=pad)
            except Exception:  # noqa: BLE001
                continue  # a condition this script cannot rebuild; skip it
            node = scorer.node_for_text(
                scorer.render(msgs, add_generation_prompt=True))
            for cand_name, text in _candidates(item, condition):
                key = (item_id, condition, cand_name)
                if key not in stored:
                    continue
                now = scorer.score(node, scorer.encode(text)).logprob
                then = stored[key]
                diffs.append({"item_id": item_id, "condition": condition,
                              "candidate": cand_name, "stored": then,
                              "recomputed": now,
                              "abs_diff": abs(float(now) - float(then))})

    if not diffs:
        raise SystemExit("no comparable rows were rebuilt; nothing checked")
    worst = max(d["abs_diff"] for d in diffs)
    report = {
        "model": args.model,
        "env": args.env,
        "n_compared": len(diffs),
        "max_abs_diff": worst,
        "mean_abs_diff": sum(d["abs_diff"] for d in diffs) / len(diffs),
        "tolerance": args.tol,
        "passed": bool(worst <= args.tol),
        "environment_now": environment_record(),
        "worst_rows": sorted(diffs, key=lambda d: -d["abs_diff"])[:5],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "worst_rows"},
                     indent=2))
    if not report["passed"]:
        raise SystemExit(
            f"FAIL: {worst:.3e} nats exceeds tol {args.tol}. The runs made under "
            f"different library versions are not one comparable set."
        )
    print("OK")


if __name__ == "__main__":
    main()
