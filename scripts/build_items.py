"""Materialise the probe item sets to disk.

Item construction involves running programs and simulated tools, so it is done
once and frozen: every model in the ladder then scores byte-identical items.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from slmecho.probe import build_coderepair_items, build_toolshed_items, save_items  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--seed", type=int, default=20260808)
    ap.add_argument("--toolshed-tasks", type=int, default=60)
    ap.add_argument("--toolshed-per-operator", type=int, default=45)
    ap.add_argument("--mbpp-problems", type=int, default=200)
    ap.add_argument("--mbpp-per-operator", type=int, default=60)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ts = build_toolshed_items(
        n_tasks=args.toolshed_tasks,
        seed=args.seed,
        max_per_operator=args.toolshed_per_operator,
    )
    save_items(ts, os.path.join(args.out_dir, "probe_toolshed.jsonl"))

    cr = build_coderepair_items(
        n_problems=args.mbpp_problems,
        seed=args.seed,
        max_per_operator=args.mbpp_per_operator,
        data_dir=os.path.join(args.out_dir, "mbpp"),
    )
    save_items(cr, os.path.join(args.out_dir, "probe_coderepair.jsonl"))

    from slmecho.envs.coderepair import cleanup_runner_files

    cleanup_runner_files()

    summary = {
        "seed": args.seed,
        "toolshed": {
            "n_items": len(ts),
            "n_tasks": len({i.task_id for i in ts}),
            "operators": sorted({i.operator for i in ts}),
            "error_types": sorted({i.error_type for i in ts}),
        },
        "coderepair": {
            "n_items": len(cr),
            "n_tasks": len({i.task_id for i in cr}),
            "operators": sorted({i.operator for i in cr}),
            "error_types": sorted({i.error_type for i in cr}),
        },
    }
    with open(os.path.join(args.out_dir, "items_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
