"""Where does probe time actually go? Breaks one model's work into phases."""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(_ROOT, ".hf"))

import torch  # noqa: E402

from slmecho.models import REGISTRY, load_model  # noqa: E402
from slmecho.probe import CORE_CONDITIONS, build_condition_messages, load_items, stratified_subsample  # noqa: E402
from slmecho.runner import _length_matched_neutral  # noqa: E402
from slmecho.scoring import CachedScorer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smollm2-135m")
    ap.add_argument("--items", default="data/probe_toolshed.jsonl")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    spec = REGISTRY[args.model]
    tok, model, meta = load_model(spec, num_threads=args.threads)
    print("threads", torch.get_num_threads(), "deterministic", args.deterministic)
    scorer = CachedScorer(tok, model, spec.chat_kwargs)

    items = stratified_subsample(load_items(args.items), args.n, max_per_task=4)
    phases = {"render": 0.0, "pad": 0.0, "seek": 0.0, "score": 0.0}
    t_all = time.time()
    for item in items:
        t = time.time(); pad = _length_matched_neutral(scorer, item); phases["pad"] += time.time() - t
        for cond in CORE_CONDITIONS:
            t = time.time()
            msgs = build_condition_messages(item, cond, neutral_obs_padded=pad)
            text = scorer.render(msgs, add_generation_prompt=True)
            phases["render"] += time.time() - t
            t = time.time(); node = scorer.node_for_text(text); phases["seek"] += time.time() - t
            for cand in (item.failed_action, item.gold_action):
                t = time.time(); scorer.score(node, scorer.encode(cand)); phases["score"] += time.time() - t
    total = time.time() - t_all
    print(f"{len(items)} items in {total:.1f}s -> {total/len(items):.1f}s/item")
    for k, v in phases.items():
        print(f"  {k:8s} {v:7.1f}s  ({v/total:5.1%})")
    print(f"  forward_tokens={scorer.forward_tokens} calls={scorer.forward_calls} "
          f"naive={scorer.naive_tokens} saved={1-scorer.forward_tokens/max(scorer.naive_tokens,1):.1%}")
    print(f"  effective {scorer.forward_tokens/total:.1f} pushed tok/s")


if __name__ == "__main__":
    main()
