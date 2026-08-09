"""Run the whole probe schedule in one process, resumably.

Phases:
  A  ToolShed, core conditions, full ladder      -> the main scaling result
  B  CodeRepair, core conditions, subset         -> second environment
  C  ToolShed, extended conditions, four models  -> Study-2 manipulations

Each (model, env, condition-set) is independently resumable: the raw JSONL is
keyed by (item, condition, candidate) and existing keys are skipped, so the
script may be re-run at any time — in particular whenever another checkpoint
has finished downloading. A lock file prevents two copies from writing at once.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(_ROOT, ".hf"))

from slmecho.models import REGISTRY, load_model  # noqa: E402
from slmecho.probe import (  # noqa: E402
    CORE_CONDITIONS,
    EXTENDED_CONDITIONS,
    load_items,
    stratified_subsample,
)
from slmecho.runner import run_probe  # noqa: E402
from slmecho.scoring import CachedScorer  # noqa: E402
from slmecho.provenance import environment_record  # noqa: E402
from slmecho.seeding import seed_everything  # noqa: E402

ENV_ITEMS = {
    "toolshed": "data/probe_toolshed.jsonl",
    "coderepair": "data/probe_coderepair.jsonl",
}

# Phase A carries the scaling claim, so it gets the widest ladder we can afford:
# seven checkpoints spanning 135M-1.7B across four families. Phases B and C are
# depth rather than breadth, and are placed on models that sit in cheap stages of
# the launcher (see scripts/launch_probes.py) so they do not extend the critical
# path.
LADDER_A = [
    "smollm2-135m", "smollm2-360m", "qwen2.5-0.5b", "qwen3-0.6b",
    "llama3.2-1b", "qwen2.5-1.5b", "smollm2-1.7b",
]
LADDER_B = ["smollm2-135m", "smollm2-360m", "qwen2.5-0.5b", "qwen3-0.6b"]
LADDER_C = ["smollm2-135m", "qwen2.5-0.5b", "llama3.2-1b"]

#: Default plan, used when configs/probe_schedule.yaml is absent.
_DEFAULT_PLAN = [
    ("A", "toolshed", "core", 100, 3, LADDER_A),
    ("B", "coderepair", "core", 60, 1, LADDER_B),
    ("C", "toolshed", "extended", 100, 3, LADDER_C),
]


def build_plan(config_path: str = "configs/probe_schedule.yaml"):
    """Return [(phase, model, env, condition_set, n_items, max_per_task), ...].

    Reads the YAML when it is present so the schedule can be changed without
    editing code, and falls back to the identical built-in defaults so the
    repository runs with no configuration at all.
    """
    spec = _DEFAULT_PLAN
    if os.path.exists(config_path):
        try:
            import yaml

            cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
            spec = [
                (name, ph["env"], ph["conditions"], ph["n_items"],
                 ph["max_per_task"], ph["models"])
                for name, ph in sorted((cfg.get("phases") or {}).items())
            ] or _DEFAULT_PLAN
        except Exception as exc:  # noqa: BLE001
            print(f"[config] falling back to defaults ({exc})", flush=True)
    plan = []
    for phase, env_name, conds, n_items, mpt, models in spec:
        for m in models:
            plan.append((phase, m, env_name, conds, n_items, mpt))
    return plan


class Lock:
    def __init__(self, path: str):
        self.path = path

    def __enter__(self):
        if os.path.exists(self.path):
            with open(self.path) as fh:
                pid = fh.read().strip()
            raise SystemExit(
                f"another schedule appears to be running (lock {self.path}, pid {pid}). "
                f"Delete the lock if that is stale."
            )
        with open(self.path, "w") as fh:
            fh.write(str(os.getpid()))
        return self

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except OSError:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/raw/probe")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260808)
    ap.add_argument("--phases", default="ABC")
    ap.add_argument("--n-items", type=int, default=0,
                    help="override the item count for every phase in this run. "
                         "The stratified subsample is a prefix, so lowering it "
                         "reuses everything already scored.")
    ap.add_argument("--models", nargs="*", default=None,
                    help="restrict to these model keys (one process per model "
                         "is how we get parallelism: see docs/compute.md)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plan = [p for p in build_plan() if p[0] in args.phases]
    if args.models:
        plan = [p for p in plan if p[1] in set(args.models)]
    if not plan:
        print('[schedule] nothing to do'); return

    env_rec = environment_record()
    print(json.dumps(env_rec), flush=True)

    lock_tag = "-".join(sorted({p[1] for p in plan}))[:60]
    with Lock(os.path.join(args.out_dir, f".lock-{lock_tag}")):
        # Group by model so each checkpoint is loaded at most once.
        by_model = {}
        for entry in plan:
            by_model.setdefault(entry[1], []).append(entry)

        for model_key in [m for m in dict.fromkeys(e[1] for e in plan)]:
            spec = REGISTRY[model_key]
            seed_everything(args.seed)
            t0 = time.time()
            try:
                tok, model, meta = load_model(spec, num_threads=args.threads)
            except Exception as exc:  # noqa: BLE001 - usually "not downloaded yet"
                print(f"[skip] {model_key}: {type(exc).__name__}: {exc}", flush=True)
                continue
            print(f"[load] {model_key} {meta['n_params']/1e6:.0f}M in "
                  f"{time.time()-t0:.0f}s", flush=True)
            scorer = CachedScorer(tok, model, spec.chat_kwargs)

            for phase, _m, env_name, cond_set, n_items, max_per_task in by_model[model_key]:
                conditions = CORE_CONDITIONS if cond_set == "core" else EXTENDED_CONDITIONS
                items = stratified_subsample(
                    load_items(ENV_ITEMS[env_name]),
                    args.n_items or n_items,
                    seed=args.seed,
                    max_per_task=max_per_task,
                )
                out_path = os.path.join(args.out_dir, f"{model_key}__{env_name}.jsonl")
                print(f"[run ] {phase} {model_key}/{env_name}/{cond_set} "
                      f"{len(items)} items", flush=True)
                info = run_probe(
                    model_key, scorer, items, conditions, out_path,
                    greedy=False, log_every=20,
                )
                info.update({
                    "phase": phase, "env": env_name, "condition_set": cond_set,
                    "n_items": len(items), "item_ids": [i.item_id for i in items],
                    "conditions": list(conditions), "model_meta": meta,
                    "environment": env_rec,
                    "seed": args.seed,
                })
                meta_path = out_path.replace(".jsonl", f".{cond_set}.meta.json")
                with open(meta_path, "w", encoding="utf-8") as fh:
                    json.dump(info, fh, indent=2)
                print(f"[done] {phase} {model_key}/{env_name}/{cond_set}: "
                      f"{info['n_scored']} scores in {info['seconds']:.0f}s", flush=True)

            del model, scorer, tok
            gc.collect()
    print("[schedule] complete", flush=True)


if __name__ == "__main__":
    main()
