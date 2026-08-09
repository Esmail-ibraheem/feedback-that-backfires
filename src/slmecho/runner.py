"""Probe execution: score every (item, condition, candidate) triple for a model.

Scheduling matters here. Conditions that share a system prompt are run in the
same pass so the (long) system prompt is encoded once per pass rather than once
per item, and within a pass items are visited in a fixed order so that the
cache's longest-common-prefix seek does the rest. On the 10-model ladder this
is the difference between roughly 8 hours and roughly 2 days of CPU time.

Results are written incrementally as JSONL and every write is keyed by
(model, item_id, condition, candidate), so an interrupted run can be resumed
by skipping keys that are already present.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .probe import (
    CORE_CONDITIONS,
    EXTENDED_CONDITIONS,
    ProbeItem,
    build_condition_messages,
)

#: Conditions that alter the system prompt, and therefore need their own pass.
_INSTR_CONDITIONS = {"fail_instr"}

#: Conditions under which we also score the distractor actions. The normalised
#: repeat probability only needs the before/after pair, and on a CPU budget each
#: extra candidate is a full (short) forward pass, so we do not score them
#: everywhere.
_DISTRACTOR_CONDITIONS = {"pre", "fail"}

#: Conditions we also decode greedily from, to record what the agent would
#: literally have written next. Free-form decoding is ~50x more expensive per
#: token than scoring on CPU, so it is off by default: the *exact-repeat*
#: statistic is recovered for free from the argmax path inside `score`, and
#: free-form samples are only needed for the qualitative appendix.
_GREEDY_CONDITIONS = {"pre", "fail", "abstract"}


def _condition_passes(conditions: Sequence[str]) -> List[List[str]]:
    plain = [c for c in conditions if c not in _INSTR_CONDITIONS]
    instr = [c for c in conditions if c in _INSTR_CONDITIONS]
    passes = []
    if plain:
        passes.append(plain)
    if instr:
        passes.append(instr)
    return passes


def _candidates(item: ProbeItem, condition: str) -> List[Tuple[str, str]]:
    out = [("failed", item.failed_action), ("gold", item.gold_action)]
    if condition in _DISTRACTOR_CONDITIONS:
        for i, d in enumerate(item.distractors):
            out.append((f"distractor{i}", d))
    return out


def _length_matched_neutral(scorer, item: ProbeItem) -> str:
    """Neutral observation padded with valence-free trace ids to match `obs_fail`.

    Length is matched in *tokens for this model*, not characters, because the
    quantity we are controlling for is context position.
    """
    from .envs.toolshed import ToolShedEnv

    target = len(scorer.encode(item.obs_fail))
    env = ToolShedEnv(verbosity="standard", echo_action=False)
    best, best_gap = item.obs_neutral, abs(len(scorer.encode(item.obs_neutral)) - target)
    for units in range(1, 24):
        cand = env.neutral_observation(item.failed_action, None, filler_units=units)
        gap = abs(len(scorer.encode(cand)) - target)
        if gap < best_gap:
            best, best_gap = cand, gap
        if len(scorer.encode(cand)) >= target:
            break
    return best


def existing_keys(path: str) -> Set[Tuple[str, str, str]]:
    keys: Set[Tuple[str, str, str]] = set()
    if not os.path.exists(path):
        return keys
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a truncated final line from a killed run
            keys.add((r["item_id"], r["condition"], r["candidate"]))
    return keys


def run_probe(
    model_key: str,
    scorer,
    items: Sequence[ProbeItem],
    conditions: Sequence[str],
    out_path: str,
    greedy: bool = True,
    greedy_max_tokens: int = 40,
    resume: bool = True,
    log_every: int = 20,
) -> Dict[str, object]:
    """Score all (item, condition, candidate) triples; append JSONL to `out_path`."""
    done = existing_keys(out_path) if resume else set()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fh = open(out_path, "a", encoding="utf-8", buffering=1)

    greedy_path = out_path.replace(".jsonl", ".greedy.jsonl")
    greedy_done = existing_keys(greedy_path) if resume else set()
    gh = open(greedy_path, "a", encoding="utf-8", buffering=1)

    # Newline ends an action in our format; stopping there keeps greedy decoding
    # cheap without truncating any well-formed call.
    stop_ids = []
    for s in ("\n", "\n\n"):
        ids = scorer.encode(s)
        if len(ids) == 1:
            stop_ids.append(ids[0])

    t0 = time.time()
    n_scored = 0
    for pass_conditions in _condition_passes(conditions):
        for i, item in enumerate(items):
            pad_neutral = None
            if "neut_pad" in pass_conditions:
                pad_neutral = _length_matched_neutral(scorer, item)
            for condition in pass_conditions:
                cands = _candidates(item, condition)
                need_score = [
                    (n, c)
                    for n, c in cands
                    if (item.item_id, condition, n) not in done
                ]
                need_greedy = (
                    greedy
                    and condition in _GREEDY_CONDITIONS
                    and (item.item_id, condition, "__greedy__") not in greedy_done
                )
                if not need_score and not need_greedy:
                    continue
                msgs = build_condition_messages(
                    item, condition, neutral_obs_padded=pad_neutral
                )
                text = scorer.render(msgs, add_generation_prompt=True)
                node = scorer.node_for_text(text)
                for name, cand in need_score:
                    res = scorer.score(node, scorer.encode(cand))
                    fh.write(
                        json.dumps(
                            {
                                "model": model_key,
                                "item_id": item.item_id,
                                "condition": condition,
                                "candidate": name,
                                "logprob": res.logprob,
                                "n_tokens": res.n_tokens,
                                "n_argmax_match": res.n_argmax_match,
                                "is_greedy": res.is_greedy,
                                "ctx_tokens": len(node.ids),
                            }
                        )
                        + "\n"
                    )
                    n_scored += 1
                if need_greedy:
                    out_ids = scorer.greedy(
                        node, max_new_tokens=greedy_max_tokens, stop_ids=stop_ids
                    )
                    text_out = scorer.tok.decode(out_ids, skip_special_tokens=True)
                    gh.write(
                        json.dumps(
                            {
                                "model": model_key,
                                "item_id": item.item_id,
                                "condition": condition,
                                "candidate": "__greedy__",
                                "generation": text_out,
                                "n_tokens": len(out_ids),
                            }
                        )
                        + "\n"
                    )
            if log_every and (i + 1) % log_every == 0:
                el = time.time() - t0
                print(
                    f"  [{model_key}] {i + 1}/{len(items)} items "
                    f"({pass_conditions[0]}...) {el:.0f}s "
                    f"fwd_tok={scorer.forward_tokens} "
                    f"saved={1 - scorer.forward_tokens / max(scorer.naive_tokens, 1):.1%}",
                    flush=True,
                )
    fh.close()
    gh.close()
    return {
        "model": model_key,
        "n_scored": n_scored,
        "seconds": time.time() - t0,
        "forward_tokens": scorer.forward_tokens,
        "forward_calls": scorer.forward_calls,
        "naive_tokens": scorer.naive_tokens,
    }
