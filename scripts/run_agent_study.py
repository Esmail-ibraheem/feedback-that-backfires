"""Study 3: end-to-end agent rollouts across harness configurations.

Every (model, env, harness, temperature, seed) combination writes its own JSONL
of full trajectories, so a run can be interrupted and resumed at combination
granularity, and any trajectory in the paper can be traced back to the exact
file that produced it.
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

from slmecho.agent import AgentEngine, repeat_stats  # noqa: E402
from slmecho.harness import HARNESSES  # noqa: E402
from slmecho.models import REGISTRY, load_model  # noqa: E402
from slmecho.seeding import seed_everything  # noqa: E402


def build_env_and_tasks(env_name: str, n_tasks: int, seed: int):
    if env_name == "toolshed":
        from slmecho.envs.toolshed import ToolShedEnv
        from slmecho.envs.toolshed_tasks import generate_agent_tasks

        # echo_action=False matches the probe's primary rendering; `demo` adds a
        # format example, identically for every harness (see ToolShedEnv.demo).
        return ToolShedEnv(
            verbosity="standard", echo_action=False, demo=True
        ), generate_agent_tasks(n_tasks, seed=seed)
    if env_name == "coderepair":
        from slmecho.envs.base import Task
        from slmecho.envs.coderepair import CodeRepairEnv, load_mbpp, run_program

        problems = load_mbpp("data/mbpp")
        tasks = []
        for p in problems:
            if len(tasks) >= n_tasks:
                break
            if len(p["code"]) > 420:
                continue  # keep programs short: CPU decoding is the binding cost
            if not run_program(p["code"], p["test_list"]).ok:
                continue
            tasks.append(
                Task(
                    task_id=f"mbpp-{p['task_id']:04d}",
                    goal=p["prompt"],
                    payload={"tests": p["test_list"]},
                    gold_actions=[p["code"]],
                )
            )
        return CodeRepairEnv(verbosity="standard", echo_action=False), tasks
    raise KeyError(env_name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--env", default="toolshed", choices=["toolshed", "coderepair"])
    # `drop` is clean restart, the remedy proposed by prior work on context
    # contamination; it is a baseline here rather than a strawman.
    ap.add_argument("--harnesses", nargs="+",
                    default=["verbatim", "verbatim+instr", "drop", "abstract",
                             "verbatim+ban", "abstract+ban"])
    ap.add_argument("--n-tasks", type=int, default=60)
    ap.add_argument("--max-steps", type=int, default=8)
    # Must match configs/agent_study.yaml; the released rollouts were produced
    # at 32, which is recoverable from them (truncated actions are exactly this
    # long).
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seeds", nargs="+", type=int, default=[20260808])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out-dir", default="results/raw/agent")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    env, tasks = build_env_and_tasks(args.env, args.n_tasks, 20260808)
    print(f"[env] {args.env}: {len(tasks)} tasks", flush=True)

    for model_key in args.models:
        spec = REGISTRY[model_key]
        seed_everything(20260808)
        try:
            tok, model, meta = load_model(spec, num_threads=args.threads)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {model_key}: {type(exc).__name__}: {exc}", flush=True)
            continue
        print(f"[load] {model_key} {meta['n_params']/1e6:.0f}M", flush=True)
        engine = AgentEngine(
            model,
            tok,
            model_key,
            chat_kwargs=spec.chat_kwargs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        for harness_name in args.harnesses:
            harness = HARNESSES[harness_name]
            for seed in args.seeds:
                tag = (
                    f"{model_key}__{args.env}__{harness_name.replace('+','-')}"
                    f"__t{args.temperature:g}__s{seed}"
                )
                out_path = os.path.join(args.out_dir, f"{tag}.jsonl")
                # Resume at rollout granularity. On CPU a single rollout costs
                # minutes, so a run that only writes when a whole harness
                # finishes throws away hours if it is interrupted.
                done_tasks = set()
                if os.path.exists(out_path):
                    with open(out_path, "r", encoding="utf-8") as rh:
                        for line in rh:
                            if line.strip():
                                try:
                                    done_tasks.add(json.loads(line)["task_id"])
                                except json.JSONDecodeError:
                                    pass
                todo = [t for t in tasks if t.task_id not in done_tasks]
                if not todo:
                    print(f"[have] {tag}", flush=True)
                    continue
                t0 = time.time()
                out_fh = open(out_path, "a", encoding="utf-8", buffering=1)

                def _sink(r, _fh=out_fh, _seed=seed) -> None:
                    rec = r.to_dict()
                    rec.update(repeat_stats(r.steps, args.env))
                    rec.update(
                        {
                            "env": args.env,
                            "temperature": args.temperature,
                            "seed": _seed,
                            "max_steps": args.max_steps,
                            # Recorded per rollout: the first version of this
                            # study left the generation cap only in a config
                            # file, and the config and the manuscript drifted
                            # apart without anything catching it.
                            "max_new_tokens": args.max_new_tokens,
                            "model_params": meta["n_params"],
                        }
                    )
                    _fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

                def _progress(done: int, total: int) -> None:
                    if done % 5 == 0 or done == total:
                        print(f"    {tag}: {done}/{total} rollouts "
                              f"({time.time()-t0:.0f}s)", flush=True)

                rollouts = engine.run(
                    env, todo, harness, max_steps=args.max_steps, seed=seed,
                    progress=_progress, sink=_sink,
                )
                out_fh.close()
                solved = sum(r.solved for r in rollouts)
                print(
                    f"[done] {tag}: solved {solved}/{len(rollouts)} new "
                    f"({len(done_tasks)} resumed) in {time.time()-t0:.0f}s",
                    flush=True,
                )
        del model, engine
        gc.collect()


if __name__ == "__main__":
    main()
