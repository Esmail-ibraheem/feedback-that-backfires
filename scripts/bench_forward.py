"""Micro-benchmark: forward throughput vs threads, dtype, and LM-head handling.

Run before committing to an experiment schedule. On a CPU-only box these
choices decide whether a study finishes overnight or never; they are recorded
here so the paper's efficiency section quotes measured numbers rather than
folklore.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(_ROOT, ".hf"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from slmecho.models import REGISTRY  # noqa: E402


def bench(model, n_params, vocab, lengths, threads, tag, keep_logits, rows):
    for nt in threads:
        torch.set_num_threads(nt)
        for L in lengths:
            ids = torch.randint(0, vocab - 1, (1, L))
            kw = {}
            if keep_logits is not None:
                kw["logits_to_keep"] = keep_logits
            with torch.inference_mode():
                try:
                    model(input_ids=ids, use_cache=False, **kw)
                except TypeError:
                    kw = {}
                    model(input_ids=ids, use_cache=False)
                reps = 3 if L > 64 else 6
                t0 = time.time()
                for _ in range(reps):
                    model(input_ids=ids, use_cache=False, **kw)
                dt = (time.time() - t0) / reps
            row = {
                "tag": tag,
                "threads": nt,
                "length": L,
                "ms": dt * 1000,
                "tok_per_s": L / dt,
                "gflops": 2 * n_params * L / dt / 1e9,
            }
            rows.append(row)
            print(
                f"  [{tag}] threads={nt:2d} L={L:4d}  {dt*1000:8.1f} ms  "
                f"{L/dt:8.1f} tok/s  {row['gflops']:6.1f} GFLOP/s",
                flush=True,
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smollm2-135m")
    ap.add_argument("--threads", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--lengths", nargs="+", type=int, default=[1, 32, 256])
    ap.add_argument("--out", default="results/bench_forward.json")
    args = ap.parse_args()

    spec = REGISTRY[args.model]
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    print(json.dumps({"mkldnn": torch.backends.mkldnn.is_available(),
                      "mkl": torch.backends.mkl.is_available()}))

    rows = []
    for dtype_name in ("float32", "bfloat16"):
        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id, torch_dtype=getattr(torch, dtype_name), low_cpu_mem_usage=True
        )
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        vocab = model.config.vocab_size
        bench(model, n_params, vocab, args.lengths, args.threads,
              f"{dtype_name}/full-head", None, rows)
        bench(model, n_params, vocab, args.lengths, args.threads,
              f"{dtype_name}/last-head", 1, rows)
        del model

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"model": args.model, "rows": rows}, fh, indent=2)


if __name__ == "__main__":
    main()
