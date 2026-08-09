# Running this on a CPU

Everything in this project was produced on a single laptop CPU. This note
records what that forced, because most of the engineering in the repository
exists for these reasons and would look arbitrary otherwise.

## The machine

| | |
| --- | --- |
| CPU | Intel Core i7-6820HQ, 4 physical cores / 8 threads, 2.7 GHz (Skylake) |
| RAM | 17 GB |
| GPU | Quadro M1000M, 2 GB — unusable; the installed torch is a CPU build |
| torch | 2.4.0+cpu |
| transformers | 4.56.1 |
| precision | float32 everywhere |

## Three measurements that shaped the design

Run `python scripts/bench_forward.py` to reproduce these on your own hardware.

**1. bfloat16 is 20x slower than float32 here.** 6 tok/s versus 116 tok/s on
SmolLM2-135M. This CPU has no AVX512-BF16, so oneDNN emulates bf16 arithmetic.
The consequence is that model size is bounded by RAM at 4 bytes per parameter,
which is why the ladder stops below 2B.

**2. A single-token forward costs ~90 ms almost regardless of model size,**
because at batch size one it is dominated by per-layer dispatch rather than
arithmetic. Generating one token therefore costs roughly what scoring thirty
positions costs. This is why the probe is built out of teacher-forced scoring,
and why the "would greedy decoding repeat the failed call?" statistic is
recovered from the argmax path already computed during scoring rather than by
running a generation loop.

**3. Computing the LM head only at the last position saves 16–22%** when merely
extending a context. At these vocabulary sizes (49k–152k) the head is 20–30% of
a forward pass.

## Prefix reuse

The probe's contexts share a ~520-token system prompt and a per-item prefix, and
differ only in a ~50-token tail. `CachedScorer` keeps one KV cache plus the exact
token ids it holds; moving to a new context crops to the longest common prefix
and pushes only the remainder. Measured on the ToolShed probe this removes about
**95%** of the token positions a cache-free implementation would process.

The same cache drives the agent rollouts, where step *t*'s context extends step
*t−1*'s. A plain `model.generate` call per step would re-encode the whole
transcript every time: in an 8-step rollout that is ~8,400 prefill tokens against
~250 generated ones.

## Parallelism: processes, not threads

**Do not call `torch.set_num_threads(n)` with n > 1 on this setup.** On this
machine it makes torch return **all-NaN logits**, silently, with unchanged
wall-clock time and no warning. One complete probe run was lost to this before
it was caught, and the first verification script missed it because
`max(0.0, float('nan'))` returns `0.0` in Python, so comparing two NaNs reported
perfect agreement.

Three guards now exist:

* `load_model` ends with a forward pass that must produce finite logits and
  raises `NonFiniteLogitsError` otherwise;
* `CachedScorer._forward` checks a few elements of every returned logit row;
* `scripts/verify_scoring.py` rejects non-finite and positive log-probabilities
  before comparing cached against uncached scoring.

Parallelism therefore comes from running one **single-threaded process per
model** (`scripts/launch_probes.py`). On four physical cores this is both
correct and faster than the broken threading was.

Stages exist because memory, not cores, binds for the larger checkpoints. In
float32 a 1.7B model resides in ~7 GB, so two of them plus the OS do not fit.

## Rough costs

Measured on this machine, single-threaded, with three other workers competing:

| Work unit | Cost |
| --- | --- |
| ToolShed probe, 100 items, core conditions, 135M | ~25 min |
| ToolShed probe, 100 items, core conditions, 0.5B | ~85 min |
| CodeRepair probe, 60 items, 135M | ~45 min (actions are ~5x longer) |
| One agent rollout, 0.5B, 6 steps | ~70 s |

Everything is resumable at fine granularity: probe results are keyed by
`(item, condition, candidate)`, and rollouts are streamed to disk as they
complete and resumed per task.

## If you have a GPU

The whole study fits comfortably on one modern GPU, and the interesting
extension needs one: the corrective gain is measurable with a few hundred forward
passes per model, so extending the ladder to 8B–70B and asking whether the
inversion survives is cheap for anyone with the hardware.
