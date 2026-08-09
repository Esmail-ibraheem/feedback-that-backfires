"""Run the probe (Studies 1 and 2) for one or more models.

Usage:
    python scripts/run_probe.py --models smollm2-135m --env toolshed --conditions core
    python scripts/run_probe.py --models all --env both --conditions all
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(_ROOT, ".hf"))

from slmecho.models import MAIN_LADDER, REGISTRY, load_model  # noqa: E402
from slmecho.probe import (  # noqa: E402
    CORE_CONDITIONS,
    EXTENDED_CONDITIONS,
    load_items,
    stratified_subsample,
)
from slmecho.runner import run_probe  # noqa: E402
from slmecho.scoring import CachedScorer  # noqa: E402
from slmecho.seeding import seed_everything  # noqa: E402

ENV_ITEMS = {
    "toolshed": "data/probe_toolshed.jsonl",
    "coderepair": "data/probe_coderepair.jsonl",
}


from slmecho.provenance import environment_record  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["smollm2-135m"])
    ap.add_argument("--env", default="toolshed", choices=["toolshed", "coderepair", "both"])
    ap.add_argument("--conditions", default="core", choices=["core", "extended", "all"])
    ap.add_argument("--n-items", type=int, default=0,
                    help="thin to N items, balanced over perturbation operators")
    ap.add_argument("--max-per-task", type=int, default=0,
                    help="cap items per task (raises bootstrap cluster count)")
    ap.add_argument("--out-dir", default="results/raw/probe")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=20260808)
    args = ap.parse_args()

    models = MAIN_LADDER if args.models == ["all"] else args.models
    envs = list(ENV_ITEMS) if args.env == "both" else [args.env]
    if args.conditions == "core":
        conditions = CORE_CONDITIONS
    elif args.conditions == "extended":
        conditions = EXTENDED_CONDITIONS
    else:
        conditions = CORE_CONDITIONS + EXTENDED_CONDITIONS

    os.makedirs(args.out_dir, exist_ok=True)
    import torch

    torch.set_num_threads(args.threads)
    env_rec = environment_record()
    print(json.dumps(env_rec), flush=True)

    for model_key in models:
        spec = REGISTRY[model_key]
        seed_everything(args.seed)
        t0 = time.time()
        try:
            tok, model, meta = load_model(spec, num_threads=args.threads)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {model_key}: {type(exc).__name__}: {exc}", flush=True)
            continue
        print(f"[load] {model_key} {meta['n_params']/1e6:.0f}M params "
              f"in {time.time()-t0:.0f}s", flush=True)
        scorer = CachedScorer(tok, model, spec.chat_kwargs)

        for env_name in envs:
            items = load_items(ENV_ITEMS[env_name])
            if args.n_items:
                items = stratified_subsample(
                    items,
                    args.n_items,
                    seed=args.seed,
                    max_per_task=args.max_per_task or None,
                )
            out_path = os.path.join(args.out_dir, f"{model_key}__{env_name}.jsonl")
            info = run_probe(
                model_key,
                scorer,
                items,
                conditions,
                out_path,
                greedy=not args.no_greedy,
            )
            info.update(
                {
                    "env": env_name,
                    "n_items": len(items),
                    "item_ids": [i.item_id for i in items],
                    "conditions": list(conditions),
                    "model_meta": meta,
                    "environment": env_rec,
                    "seed": args.seed,
                }
            )
            with open(out_path.replace(".jsonl", ".meta.json"), "w", encoding="utf-8") as fh:
                json.dump(info, fh, indent=2)
            print(f"[done] {model_key}/{env_name}: {json.dumps(info['n_scored'])} scores "
                  f"in {info['seconds']:.0f}s "
                  f"({info['forward_tokens']} fwd tokens, "
                  f"{1 - info['forward_tokens']/max(info['naive_tokens'],1):.1%} saved)",
                  flush=True)

        del model, scorer
        import gc

        gc.collect()


if __name__ == "__main__":
    main()
