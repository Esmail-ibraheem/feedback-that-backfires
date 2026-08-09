"""Pre-fetch model weights, smallest first.

Kept as a separate step (rather than lazily downloading inside the experiment
runner) because on this machine the network is roughly two orders of magnitude
slower than the GPU-less inference we do afterwards: mixing the two would make
timing measurements meaningless and would make a failed download look like a
failed experiment.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from slmecho.models import MAIN_LADDER, REGISTRY  # noqa: E402

PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--ladder", action="store_true", help="download MAIN_LADDER")
    args = ap.parse_args()

    keys = args.models or (MAIN_LADDER if args.ladder else MAIN_LADDER)
    keys = sorted(keys, key=lambda k: REGISTRY[k].params_b)

    from huggingface_hub import snapshot_download

    for key in keys:
        spec = REGISTRY[key]
        t0 = time.time()
        print(f"[download] {key} <- {spec.hf_id}", flush=True)
        try:
            path = snapshot_download(spec.hf_id, allow_patterns=PATTERNS)
        except Exception as exc:  # noqa: BLE001
            print(f"[download] FAILED {key}: {type(exc).__name__}: {exc}", flush=True)
            continue
        size = 0
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    size += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        dt = time.time() - t0
        print(
            f"[download] ok {key} {size / 1e6:.0f} MB in {dt:.0f}s "
            f"({size / 1e6 / max(dt, 1e-9):.2f} MB/s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
