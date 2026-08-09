"""Study 3: end-to-end agent rollouts under different harness configurations.

The probe measures a single decision point under a controlled context. This
module lets the models actually drive: they choose their own actions, the
environment executes them, and whatever failures occur are their own. The point
is to test whether the harness manipulations that move the probe's numbers also
move task success and repetition in a real loop.

Rollouts run one at a time but share a key/value cache across steps, because an
agent's context at step t extends its context at step t-1. Re-encoding the whole
transcript every step — what a plain `model.generate` call does — would spend
roughly 95% of this study's compute on tokens it had already seen.

Each rollout is handed to a sink as soon as it finishes, so an interrupted run
keeps everything it completed. On CPU a single rollout can take minutes; losing
a harness's worth of them to a stray kill is not acceptable.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .decoding import BannedSequenceProcessor, tokenize_ban_list
from .harness import HarnessConfig, TrajStep, build_messages, failed_actions


@dataclass
class Rollout:
    task_id: str
    harness: str
    model: str
    steps: List[TrajStep] = field(default_factory=list)
    solved: bool = False
    finished: bool = False
    stop_reason: str = ""
    gen_tokens: int = 0
    prompt_tokens: int = 0
    seconds: float = 0.0
    n_banned_masked: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "harness": self.harness,
            "model": self.model,
            "solved": self.solved,
            "n_steps": len(self.steps),
            "stop_reason": self.stop_reason,
            "gen_tokens": self.gen_tokens,
            "prompt_tokens": self.prompt_tokens,
            "seconds": self.seconds,
            "n_banned_masked": self.n_banned_masked,
            "actions": [s.action for s in self.steps],
            "ok": [s.ok for s in self.steps],
            "error_types": [s.error_type for s in self.steps],
            "observations": [s.observation for s in self.steps],
        }


# --------------------------------------------------------------------------
# Action canonicalisation (for repeat detection)
# --------------------------------------------------------------------------


def canonical_action(action: str, env_name: str) -> str:
    """A normal form used only for *measuring* repetition, never for execution.

    Exact string repetition understates the phenomenon: a model that re-emits
    the same call with different spacing or argument order has still repeated
    itself. For tool calls we parse and sort the keyword arguments; for code we
    strip comments and collapse whitespace.
    """
    if env_name == "toolshed":
        try:
            from .envs.toolshed import parse_action

            p = parse_action(action)
            args = ",".join(f"{k}={v!r}" for k, v in sorted(p.kwargs.items()))
            pos = ",".join(repr(v) for v in p.positional)
            return f"{p.name}({pos}|{args})"
        except Exception:  # noqa: BLE001
            return re.sub(r"\s+", " ", action.strip())
    body = re.sub(r"#.*", "", action)
    return re.sub(r"\s+", " ", body.strip())


def repeat_stats(steps: Sequence[TrajStep], env_name: str) -> Dict[str, float]:
    """Repetition summary for one trajectory."""
    seen_exact: set = set()
    seen_canon: set = set()
    n_fail = 0
    exact_repeats = 0
    canon_repeats = 0
    consecutive = 0
    prev_canon = None
    for s in steps:
        c = canonical_action(s.action, env_name)
        if prev_canon is not None and c == prev_canon:
            consecutive += 1
        prev_canon = c
        if s.ok:
            continue
        n_fail += 1
        if s.action.strip() in seen_exact:
            exact_repeats += 1
        if c in seen_canon:
            canon_repeats += 1
        seen_exact.add(s.action.strip())
        seen_canon.add(c)
    return {
        "n_failed_actions": n_fail,
        "n_exact_repeats": exact_repeats,
        "n_canonical_repeats": canon_repeats,
        "n_consecutive_repeats": consecutive,
        "exact_repeat_rate": exact_repeats / n_fail if n_fail else float("nan"),
        "canonical_repeat_rate": canon_repeats / n_fail if n_fail else float("nan"),
    }


# --------------------------------------------------------------------------
# Batched rollout engine
# --------------------------------------------------------------------------


ACTION_LINE = re.compile(r"^\s*(?:Action\s*:\s*)?(.+?)\s*$")


def extract_action(text: str, env_name: str) -> str:
    """Take one action out of a model turn."""
    text = text.strip()
    if env_name == "coderepair":
        from .envs.coderepair import extract_code

        return extract_code(text)
    # Tool calls: the first non-empty line, minus any "Action:" label or fences.
    for line in text.splitlines():
        line = line.strip().strip("`")
        if not line:
            continue
        line = re.sub(r"^(?:Action|Tool|Call)\s*:\s*", "", line, flags=re.I)
        if line:
            return line
    return text


class AgentEngine:
    """Runs rollouts one at a time, reusing the KV cache across steps.

    The obvious implementation calls `model.generate` once per agent step, which
    re-encodes the whole transcript every time. In an 8-step ToolShed rollout
    that is ~8,400 prefill tokens against ~250 generated ones — the experiment
    would be 95% wasted work. Driving generation through `CachedScorer` instead
    means step t only pays for the tokens that step t actually added, and the
    (long, identical) system prompt is paid for once per model, not once per
    step per rollout. Measured end to end this is ~4x faster than the batched
    `generate` version it replaced, and it is what makes the harness sweep fit
    on a CPU.
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_key: str,
        chat_kwargs: Optional[Dict] = None,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ):
        from .scoring import CachedScorer

        self.scorer = CachedScorer(tokenizer, model, chat_kwargs)
        self.tok = tokenizer
        self.model_key = model_key
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _stop_ids(self, env_name: str) -> List[int]:
        """Extra end-of-turn ids. A newline ends a tool call but not a program."""
        ids: List[int] = []
        vocab = self.tok.get_vocab()
        for t in ("<|im_end|>", "<|eot_id|>", "<|endoftext|>", "<|end|>"):
            if t in vocab:
                ids.append(vocab[t])
        if env_name != "coderepair":
            for s in ("\n", "\n\n"):
                t = self.tok(s, add_special_tokens=False)["input_ids"]
                if len(t) == 1:
                    ids.append(t[0])
        return sorted(set(ids))

    def run(
        self,
        env,
        tasks: Sequence[Any],
        harness: HarnessConfig,
        max_steps: int = 8,
        seed: int = 20260808,
        progress: Optional[Callable[[int, int], None]] = None,
        sink: Optional[Callable[["Rollout"], None]] = None,
    ) -> List[Rollout]:
        env_name = getattr(env, "name", "env")
        stop_ids = self._stop_ids(env_name)
        generator = None
        if self.temperature and self.temperature > 0:
            generator = torch.Generator().manual_seed(seed)

        rollouts: List[Rollout] = []
        for idx, task in enumerate(tasks):
            env.reset(task)
            r = Rollout(task_id=task.task_id, harness=harness.name, model=self.model_key)
            t0 = time.time()
            for _step in range(max_steps):
                msgs = build_messages(
                    env.system_prompt(task),
                    env.user_message(task) if hasattr(env, "user_message") else task.goal,
                    r.steps,
                    harness,
                )
                text = self.scorer.render(msgs, add_generation_prompt=True)
                node = self.scorer.node_for_text(text)
                banned = (
                    tokenize_ban_list(self.tok, failed_actions(r.steps))
                    if harness.ban_failed_actions
                    else []
                )
                out_ids, n_masked = self.scorer.generate(
                    node,
                    max_new_tokens=self.max_new_tokens,
                    stop_ids=stop_ids,
                    banned=banned,
                    temperature=self.temperature,
                    generator=generator,
                )
                r.prompt_tokens += len(node.ids)
                r.gen_tokens += len(out_ids)
                r.n_banned_masked += n_masked
                raw = self.tok.decode(out_ids, skip_special_tokens=True)
                action = extract_action(raw, env_name)
                outcome = env.step(task, action)
                tool = None
                if env_name == "toolshed":
                    try:
                        from .envs.toolshed import parse_action

                        tool = parse_action(action).name
                    except Exception:  # noqa: BLE001
                        tool = None
                r.steps.append(
                    TrajStep(
                        action=action,
                        observation=outcome.observation,
                        ok=outcome.ok,
                        error_type=outcome.error_type,
                        tool=tool,
                    )
                )
                if _is_solved(env, task, outcome):
                    r.solved = True
                    r.stop_reason = "solved"
                    break
                if outcome.info.get("finish"):
                    r.stop_reason = "finish_called"
                    break
            if not r.stop_reason:
                r.stop_reason = "step_limit"
            r.seconds = time.time() - t0
            rollouts.append(r)
            if sink is not None:
                sink(r)
            if progress:
                progress(idx + 1, len(tasks))
        return rollouts


#: Backwards-compatible alias; the batched implementation was replaced.
BatchedAgentEngine = AgentEngine


def _is_solved(env, task, outcome) -> bool:
    try:
        return bool(env.is_solved(task))
    except NotImplementedError:
        return bool(outcome.ok)
