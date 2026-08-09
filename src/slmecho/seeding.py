"""Seed management.

Every stochastic component in this project draws from an explicit, named
sub-stream rather than from a single global RNG.  The reason is practical: we
frequently re-run *part* of an experiment (say, one model or one harness
variant) and we want the data it sees to be bit-identical to the first run even
if the surrounding loop changed.  Deriving a seed from (base_seed, name) via a
stable hash gives us that.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Optional

import numpy as np

DEFAULT_SEED = 20260808


def derive_seed(base_seed: int, name: str) -> int:
    """Deterministically derive a 32-bit sub-seed from a base seed and a label.

    Uses BLAKE2b rather than Python's ``hash`` because the latter is salted per
    process (PYTHONHASHSEED) and would silently break reproducibility.
    """
    digest = hashlib.blake2b(
        f"{base_seed}:{name}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def rng_for(base_seed: int, name: str) -> random.Random:
    """A `random.Random` bound to the (base_seed, name) sub-stream."""
    return random.Random(derive_seed(base_seed, name))


def np_rng_for(base_seed: int, name: str) -> np.random.Generator:
    """A numpy Generator bound to the (base_seed, name) sub-stream."""
    return np.random.default_rng(derive_seed(base_seed, name))


def seed_everything(seed: int = DEFAULT_SEED, torch_deterministic: bool = True) -> None:
    """Seed python / numpy / torch and pin CPU thread behaviour.

    `torch` is imported lazily so that pure-python parts of the package stay
    importable in environments without it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is optional for data code
        return
    torch.manual_seed(seed)
    if torch_deterministic:
        # We only ever run CPU inference, where the main source of run-to-run
        # drift is thread-count-dependent reduction order.  Pinning the algorithm
        # selection keeps log-probabilities stable to ~1e-6 across reruns.
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_seed(seed: Optional[int]) -> int:
    return DEFAULT_SEED if seed is None else int(seed)
