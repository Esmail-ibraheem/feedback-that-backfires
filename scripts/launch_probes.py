"""Launch the probe schedule as several single-threaded processes.

Why processes and not threads: this machine's torch build silently returns NaN
logits when `torch.set_num_threads(n>1)` is called (see `load_model`), so all
intra-op parallelism is off. Running one single-threaded process per model
recovers the CPU's four cores, and it also isolates failures — a crash takes
one model with it, not the run.

Stages exist because memory, not cores, is the binding constraint for the larger
checkpoints: everything is fp32, so a 1.7B model resides in ~7 GB and two of
them plus the OS do not fit in 17 GB.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_EXE = os.path.join(_ROOT, ".venv", "Scripts", "python.exe")

# (stage name, [model keys run concurrently]) — sized so peak fp32 residency
# stays under ~12 GB.
STAGES = [
    ("small", ["smollm2-135m", "smollm2-360m", "qwen2.5-0.5b", "qwen3-0.6b"]),
    ("mid", ["llama3.2-1b", "qwen2.5-1.5b"]),
    ("large", ["smollm2-1.7b"]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="ABC")
    ap.add_argument("--stages", nargs="*", default=None)
    ap.add_argument("--log-dir", default="results/logs")
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    env = dict(os.environ)
    env["HF_HOME"] = os.path.join(_ROOT, ".hf")
    # Keep every worker strictly single-threaded.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[var] = "1"

    for name, models in STAGES:
        if args.stages and name not in args.stages:
            continue
        print(f"=== stage {name}: {models} ===", flush=True)
        procs = []
        for m in models:
            log = open(os.path.join(args.log_dir, f"probe_{m}.log"), "a",
                       encoding="utf-8")
            cmd = [PY_EXE, "scripts/run_schedule.py", "--phases", args.phases,
                   "--models", m]
            procs.append((m, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                              env=env, cwd=_ROOT), log))
            time.sleep(2)
        for m, p, log in procs:
            rc = p.wait()
            log.close()
            print(f"  [{name}] {m} exited rc={rc}", flush=True)
    print("=== all stages done ===", flush=True)


if __name__ == "__main__":
    main()
