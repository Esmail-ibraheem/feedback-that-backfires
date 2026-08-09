"""Model registry and loading.

Everything runs on CPU in fp32 (see `research_log.md` for why bf16 is not a
win on this machine). The registry records the exact parameter count reported
by the checkpoint config so that the scaling analysis uses measured, not
advertised, sizes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

# Keep the (large) model cache off the small system drive. Set before any
# transformers/huggingface_hub import so it is picked up.
_DEFAULT_HF_HOME = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), ".hf")
os.environ.setdefault("HF_HOME", _DEFAULT_HF_HOME)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    family: str
    #: Advertised size, used only for labels; the measured count is recorded at
    #: load time and is what the analysis uses.
    label: str
    params_b: float
    dtype: str = "float32"
    #: Some checkpoints emit a reasoning preamble unless told not to.
    chat_kwargs: Optional[Dict[str, object]] = None


REGISTRY: Dict[str, ModelSpec] = {
    m.key: m
    for m in [
        ModelSpec("smollm2-135m", "HuggingFaceTB/SmolLM2-135M-Instruct",
                  "SmolLM2", "SmolLM2-135M", 0.135),
        ModelSpec("smollm2-360m", "HuggingFaceTB/SmolLM2-360M-Instruct",
                  "SmolLM2", "SmolLM2-360M", 0.362),
        ModelSpec("smollm2-1.7b", "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                  "SmolLM2", "SmolLM2-1.7B", 1.711),
        ModelSpec("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B-Instruct",
                  "Qwen2.5", "Qwen2.5-0.5B", 0.494),
        ModelSpec("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct",
                  "Qwen2.5", "Qwen2.5-1.5B", 1.544),
        ModelSpec("qwen2.5-3b", "Qwen/Qwen2.5-3B-Instruct",
                  "Qwen2.5", "Qwen2.5-3B", 3.086),
        ModelSpec("llama3.2-1b", "unsloth/Llama-3.2-1B-Instruct",
                  "Llama-3.2", "Llama-3.2-1B", 1.236),
        ModelSpec("llama3.2-3b", "unsloth/Llama-3.2-3B-Instruct",
                  "Llama-3.2", "Llama-3.2-3B", 3.213),
        ModelSpec("qwen3-0.6b", "Qwen/Qwen3-0.6B", "Qwen3", "Qwen3-0.6B", 0.596,
                  chat_kwargs={"enable_thinking": False}),
        ModelSpec("qwen3-1.7b", "Qwen/Qwen3-1.7B", "Qwen3", "Qwen3-1.7B", 2.031,
                  chat_kwargs={"enable_thinking": False}),
        ModelSpec("olmo2-1b", "allenai/OLMo-2-0425-1B-Instruct",
                  "OLMo-2", "OLMo-2-1B", 1.480),
        ModelSpec("falcon3-1b", "tiiuae/Falcon3-1B-Instruct",
                  "Falcon3", "Falcon3-1B", 1.669),
        ModelSpec("falcon3-3b", "tiiuae/Falcon3-3B-Instruct",
                  "Falcon3", "Falcon3-3B", 3.228),
        ModelSpec("qwen2.5-coder-1.5b", "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                  "Qwen2.5-Coder", "Qwen2.5-Coder-1.5B", 1.544),
    ]
}

#: Ladder used for the main scaling figure — three families with at least two
#: sizes each, so within-family scaling is identifiable.
MAIN_LADDER: List[str] = [
    "smollm2-135m",
    "smollm2-360m",
    "qwen2.5-0.5b",
    "qwen3-0.6b",
    "llama3.2-1b",
    "qwen2.5-1.5b",
    "smollm2-1.7b",
    "qwen3-1.7b",
    "qwen2.5-3b",
    "llama3.2-3b",
]


class NonFiniteLogitsError(RuntimeError):
    """Raised when a freshly loaded model cannot produce finite logits."""


def load_model(spec: ModelSpec, num_threads: Optional[int] = None, self_check: bool = True):
    """Load tokenizer + model for CPU inference. Returns (tokenizer, model, meta).

    `num_threads` is accepted and ignored. On the machine this project was run
    on, calling ``torch.set_num_threads(n)`` with n > 1 makes this torch build
    return **all-NaN logits** — silently, with no error and with unchanged
    wall-clock time, because the arithmetic still happens. We lost a full probe
    run to that before catching it, so the call is gone, the parameter is kept
    only so existing scripts and configs do not break, and every load now ends
    with a forward pass that must produce finite logits. Parallelism comes from
    running several single-threaded processes (see `scripts/run_shard.py`),
    which on a 4-core CPU is both faster and correct.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = getattr(torch, spec.dtype)
    # `torch_dtype` was renamed to `dtype` during the transformers 4.5x series.
    # Passing the wrong one is not ignored: it is forwarded to the model
    # constructor and raises. Try the new name, fall back to the old one.
    try:
        model = AutoModelForCausalLM.from_pretrained(spec.hf_id, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(spec.hf_id, torch_dtype=dtype)
    model.eval()
    model.config.use_cache = True

    if self_check:
        probe = tok("The capital of France is", add_special_tokens=False,
                    return_tensors="pt")
        with torch.inference_mode():
            logits = model(**probe).logits
        if not bool(torch.isfinite(logits).all()):
            raise NonFiniteLogitsError(
                f"{spec.hf_id}: forward pass produced non-finite logits "
                f"(torch {torch.__version__}, threads={torch.get_num_threads()}). "
                f"Refusing to run experiments with a broken numerical path."
            )

    n_params = sum(p.numel() for p in model.parameters())
    meta = {
        "hf_id": spec.hf_id,
        "n_params": int(n_params),
        "n_layers": int(getattr(model.config, "num_hidden_layers", -1)),
        "hidden": int(getattr(model.config, "hidden_size", -1)),
        "vocab": int(getattr(model.config, "vocab_size", -1)),
        "dtype": spec.dtype,
        "attn_implementation": str(getattr(model.config, "_attn_implementation", "")),
        "torch_threads": int(torch.get_num_threads()),
    }
    return tok, model, meta
