"""Teacher-forced scoring with prefix-KV reuse.

The central measurement of this project is a *difference of log-probabilities of
one fixed string under several contexts*. That is cheap in principle — one
forward pass — but naively it is not: the contexts we compare share a long
common prefix (a system prompt with twelve tool schemas, then the same goal,
then the same successful steps) and differ only in a short suffix. Re-encoding
the shared part for every condition wastes most of the compute, which on a
CPU-only machine is the difference between an overnight run and a week.

`CachedScorer` keeps one KV cache and a record of exactly which token ids it
holds. Asking it to work from a `Node` triggers a seek: crop to the longest
common prefix with the node's ids, then push the remainder. Depth-first
traversal of a context tree therefore costs each shared prefix exactly once,
and — importantly — the answer does not depend on traversal order, so a bug in
the traversal cannot silently corrupt a measurement.

Correctness is checked, not assumed: `score_uncached` recomputes the same
quantity with a single full forward pass and no cache at all, and
`scripts/verify_scoring.py` asserts the two agree to 1e-3 nats on real items.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class ScoreResult:
    logprob: float
    n_tokens: int
    #: Positions where the greedy token equals the target token.
    n_argmax_match: int
    #: True iff greedy decoding from this context reproduces the continuation.
    is_greedy: bool = False


@dataclass(frozen=True)
class Node:
    """A context, identified by the exact token ids that produce it."""

    ids: Tuple[int, ...]
    #: log-softmax over the vocabulary for the token that would come next.
    #: `None` only for the empty root, which callers must extend before scoring.
    next_logprobs: Optional[torch.Tensor] = None

    def __len__(self) -> int:  # noqa: D105
        return len(self.ids)


class CachedScorer:
    """Score continuations under a tree of shared-prefix contexts."""

    def __init__(self, tokenizer, model, chat_kwargs: Optional[Dict] = None):
        self.tok = tokenizer
        self.model = model
        self.chat_kwargs = dict(chat_kwargs or {})
        self._cache = None
        self._cache_ids: List[int] = []
        self._last_logprobs: Optional[torch.Tensor] = None
        self._supports_logits_to_keep = True
        # Instrumentation: how many token positions we actually pushed through
        # the network, versus how many a cache-free implementation would have.
        self.forward_tokens = 0
        self.forward_calls = 0
        self.naive_tokens = 0

    # -- text handling ------------------------------------------------------

    def render(
        self, messages: Sequence[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        return self.tok.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **self.chat_kwargs,
        )

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return self.tok(text, add_special_tokens=add_special_tokens)["input_ids"]

    def node_for_text(self, text: str) -> Node:
        """Build (and cache) the node for a fully rendered prompt string."""
        return self.node_for_ids(self.encode(text))

    def node_for_ids(self, ids: Sequence[int]) -> Node:
        ids = tuple(ids)
        self._seek(ids)
        return Node(ids=ids, next_logprobs=self._last_logprobs)

    # -- cache plumbing -----------------------------------------------------

    def _reset(self) -> None:
        from transformers import DynamicCache

        self._cache = DynamicCache()
        self._cache_ids = []
        self._last_logprobs = None

    def _seek(self, ids: Sequence[int]) -> None:
        """Make the cache hold exactly `ids`, reusing the longest common prefix."""
        if self._cache is None:
            self._reset()
        lcp = 0
        limit = min(len(ids), len(self._cache_ids))
        while lcp < limit and ids[lcp] == self._cache_ids[lcp]:
            lcp += 1
        if lcp < len(self._cache_ids):
            self._cache.crop(lcp)
            del self._cache_ids[lcp:]
            self._last_logprobs = None
        remainder = list(ids[lcp:])
        if remainder:
            lp = self._forward(remainder, all_positions=False)
            self._last_logprobs = lp[-1]
        elif self._last_logprobs is None and self._cache_ids:
            # We cropped exactly to `ids`; the next-token distribution for this
            # position was discarded, so recompute it by replaying one token.
            self._cache.crop(len(self._cache_ids) - 1)
            last = self._cache_ids[-1]
            del self._cache_ids[-1]
            lp = self._forward([last], all_positions=False)
            self._last_logprobs = lp[-1]

    @torch.inference_mode()
    def _forward(self, ids: Sequence[int], all_positions: bool = True) -> torch.Tensor:
        """Push `ids` onto the cache; return log-softmax logits.

        When only the final next-token distribution is needed (extending a
        context rather than scoring through it) we ask the model to compute the
        LM head at the last position only. For the vocabularies in this ladder
        that head is 20-30% of the FLOPs of a forward pass, so this is not a
        micro-optimisation.
        """
        input_ids = torch.tensor([list(ids)], dtype=torch.long)
        kwargs = {}
        if not all_positions and self._supports_logits_to_keep:
            kwargs["logits_to_keep"] = 1
        try:
            out = self.model(
                input_ids=input_ids,
                past_key_values=self._cache,
                use_cache=True,
                **kwargs,
            )
        except TypeError:
            self._supports_logits_to_keep = False
            out = self.model(
                input_ids=input_ids, past_key_values=self._cache, use_cache=True
            )
        self._cache = out.past_key_values
        self._cache_ids.extend(ids)
        self.forward_tokens += len(ids)
        self.forward_calls += 1
        logits = out.logits[0].float()
        # Cheap tripwire. A misconfigured CPU threading path on this machine
        # produced all-NaN logits without raising anything; checking one element
        # per forward costs nothing and turns that into a loud failure.
        if not bool(torch.isfinite(logits[-1, :8]).all()):
            raise RuntimeError(
                "non-finite logits from the model; refusing to record scores"
            )
        return torch.log_softmax(logits, dim=-1)

    # -- public API ---------------------------------------------------------

    def score(self, node: Node, cont_ids: Sequence[int]) -> "ScoreResult":
        """Score `cont_ids` under `node`. The node stays usable afterwards.

        Besides the summed log-probability we return how many positions the
        greedy (argmax) token matches the target. If *every* position matches,
        greedy decoding from this context reproduces the continuation exactly —
        which is the behavioural statement we care about ("the agent would
        re-emit the failed call verbatim"), obtained here for free rather than
        by running a generation loop that costs ~50x more on CPU.
        """
        cont_ids = list(cont_ids)
        self.naive_tokens += len(node.ids) + len(cont_ids)
        if not cont_ids:
            return ScoreResult(0.0, 0, 0)
        self._seek(node.ids)
        total = float(self._last_logprobs[cont_ids[0]])
        n_argmax = 1 if int(torch.argmax(self._last_logprobs)) == cont_ids[0] else 0
        prefix_greedy = n_argmax == 1
        if len(cont_ids) > 1:
            lp = self._forward(cont_ids[:-1], all_positions=True)
            idx = torch.tensor(cont_ids[1:], dtype=torch.long)
            total += float(lp.gather(1, idx.unsqueeze(1)).sum())
            matches = (lp.argmax(dim=-1) == idx).tolist()
            for i, m in enumerate(matches):
                if m:
                    n_argmax += 1
                if prefix_greedy and not m:
                    prefix_greedy = False
            # Roll back so `node` remains the cache's tip for the next call.
            self._cache.crop(len(node.ids))
            del self._cache_ids[len(node.ids) :]
        return ScoreResult(total, len(cont_ids), n_argmax, is_greedy=prefix_greedy)

    @torch.inference_mode()
    def generate(
        self,
        node: Node,
        max_new_tokens: int = 48,
        stop_ids: Optional[Sequence[int]] = None,
        banned: Optional[Sequence[Sequence[int]]] = None,
        temperature: float = 0.0,
        top_p: float = 0.95,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[List[int], int]:
        """Decode a continuation from `node`, optionally under a ban list.

        This is the agent's generation path. It runs on the same prefix cache as
        scoring, which is the whole point: an agent's context at step t extends
        its context at step t-1, so re-encoding the transcript every step (what
        a plain `model.generate` call does) wastes the great majority of the
        compute. Reusing the cache turns the per-step cost from "prefill the
        whole transcript" into "prefill the new turn".

        `banned` is a list of token sequences that may not be produced; the last
        token of a sequence is masked whenever the tokens generated so far match
        its prefix. Returns (token_ids, n_ban_masks_applied).
        """
        self._seek(node.ids)
        stop = set(stop_ids or [])
        if self.tok.eos_token_id is not None:
            stop.add(self.tok.eos_token_id)
        bans = [list(b) for b in (banned or []) if b]
        out_ids: List[int] = []
        n_masked = 0
        logits = self._last_logprobs.clone()
        while len(out_ids) < max_new_tokens:
            if bans:
                for seq in bans:
                    k = len(seq) - 1
                    if k == 0:
                        logits[seq[0]] = float("-inf")
                        n_masked += 1
                    elif len(out_ids) >= k and out_ids[len(out_ids) - k :] == seq[:k]:
                        logits[seq[-1]] = float("-inf")
                        n_masked += 1
            nxt = _pick(logits, temperature, top_p, generator)
            if nxt in stop:
                break
            out_ids.append(nxt)
            if len(out_ids) >= max_new_tokens:
                break
            logits = self._forward([nxt], all_positions=False)[-1].clone()
        # Restore the cache tip to `node` so the caller can keep using it.
        if len(self._cache_ids) > len(node.ids):
            self._cache.crop(len(node.ids))
            del self._cache_ids[len(node.ids) :]
            self._last_logprobs = node.next_logprobs
        return out_ids, n_masked

    def greedy(self, node: Node, max_new_tokens: int = 48,
               stop_ids: Optional[Sequence[int]] = None) -> List[int]:
        return self.generate(node, max_new_tokens, stop_ids)[0]

    # -- verification -------------------------------------------------------

    @torch.inference_mode()
    def score_uncached(
        self, prompt_ids: Sequence[int], cont_ids: Sequence[int]
    ) -> float:
        """Reference implementation: one full forward, no cache reuse."""
        ids = list(prompt_ids) + list(cont_ids)
        out = self.model(
            input_ids=torch.tensor([ids], dtype=torch.long), use_cache=False
        )
        lp = torch.log_softmax(out.logits[0].float(), dim=-1)
        n_p = len(prompt_ids)
        tgt = torch.tensor(list(cont_ids), dtype=torch.long)
        sel = lp[n_p - 1 : n_p - 1 + len(cont_ids)]
        return float(sel.gather(1, tgt.unsqueeze(1)).sum())


def _pick(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    generator: Optional[torch.Generator],
) -> int:
    """Argmax, or nucleus sampling when a temperature is set."""
    if temperature is None or temperature <= 0:
        return int(torch.argmax(logits))
    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    keep = cum <= top_p
    keep[0] = True
    sorted_probs = sorted_probs * keep
    sorted_probs = sorted_probs / sorted_probs.sum()
    choice = torch.multinomial(sorted_probs, 1, generator=generator)
    return int(sorted_idx[choice])


def split_prompt_and_continuation(
    tok, prompt_text: str, continuation: str
) -> Tuple[List[int], List[int]]:
    """Tokenise `prompt_text + continuation` and split at the true boundary.

    Tokenising the two halves separately and concatenating is not in general the
    same as tokenising the concatenation (BPE merges across the seam). We
    tokenise the full string and locate the split by longest common prefix with
    the tokenised prompt. Chat templates end the prompt with a role header and a
    newline, so in practice the split is exact; doing it this way means we would
    notice if that ever stopped being true for some model.
    """
    full = tok(prompt_text + continuation, add_special_tokens=False)["input_ids"]
    head = tok(prompt_text, add_special_tokens=False)["input_ids"]
    k = 0
    while k < len(head) and k < len(full) and head[k] == full[k]:
        k += 1
    return full[:k], full[k:]
