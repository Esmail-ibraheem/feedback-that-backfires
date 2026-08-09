"""Decoder-level suppression of previously-failed actions.

The probe study says the damage done by a failure record is carried by the
*surface form* of the failed call sitting in the context. If that is right, the
intervention that should work is not a better instruction — the model has to
understand and obey an instruction — but a constraint applied to the decoder,
where "do not write this string again" is enforced rather than requested.

`BannedSequenceProcessor` is that constraint. It is the classic bad-words
construction: a banned token sequence is blocked by masking its *final* token
whenever the tokens generated so far match the sequence's prefix. Blocking the
first token instead would be wrong — it would forbid every call that merely
starts the same way, including the corrected one, which typically shares a long
prefix with the failed call.

Two properties matter for the experiment:

* It is *exact*: a banned string cannot be emitted, so the repeat rate for
  banned strings is zero by construction and any residual repetition must be a
  paraphrase. We measure that separately.
* It is *cheap*: one Python pass over the ban list per decoding step, no extra
  model calls, no extra tokens. In the cost accounting it shows up as ~0.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import torch


class BannedSequenceProcessor:
    """A `LogitsProcessor` banning per-row token sequences.

    `banned[row]` is a list of token-id sequences that row may not produce as a
    contiguous suffix of its generated text.
    """

    def __init__(self, banned: Sequence[Sequence[Sequence[int]]], prompt_len: int):
        self.banned = [[list(s) for s in row if s] for row in banned]
        self.prompt_len = prompt_len
        #: Instrumentation for the paper: how often the constraint actually bit.
        self.n_masked = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for row, seqs in enumerate(self.banned):
            if not seqs:
                continue
            generated = input_ids[row, self.prompt_len :].tolist()
            for seq in seqs:
                k = len(seq) - 1
                if k == 0:
                    scores[row, seq[0]] = float("-inf")
                    self.n_masked += 1
                elif len(generated) >= k and generated[len(generated) - k :] == seq[:k]:
                    scores[row, seq[-1]] = float("-inf")
                    self.n_masked += 1
        return scores


def tokenize_ban_list(tokenizer, actions: Iterable[str]) -> List[List[int]]:
    """Token sequences for a ban list, including the leading-whitespace variant.

    A model may write the same call with or without a leading space, which BPE
    turns into a different first token. Banning only one spelling would let the
    other through and would make the intervention look weaker than it is, so we
    ban both.
    """
    out: List[List[int]] = []
    seen = set()
    for a in actions:
        a = a.strip()
        if not a:
            continue
        for variant in (a, " " + a):
            ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
            key = tuple(ids)
            if ids and key not in seen:
                seen.add(key)
                out.append(ids)
    return out
