# Feedback That Backfires

**Why small language model agents repeat the call they just watched fail, and
what a harness can do about it.**

Agent frameworks append a failed tool call and its error message to the
transcript and ask the model to continue. This repository contains the code,
data and analysis behind a simple measurement of whether that helps:

> Take the moment before an action is attempted and the moment after it has
> failed. Compare the model's probability of writing that exact action. For
> every instruction-tuned model we tested, across two environments, the
> probability goes **up**, by about one nat per action token, on essentially
> every individual item.

The interesting part is *why*. Appending a failed call does two things at once:
it puts the call's token sequence into the context, and it says the call failed.
Scoring the same action against a **valence-free** observation, where the
runtime acknowledges the call and says nothing about how it went, separates
them. Almost
all of the effect is the surface form. The semantic part, which would have to be
large and negative if the story were "the small model doesn't understand the
error", is small and points the wrong way.

That localises the problem in the *harness*, not the model, and it says which
fixes can work, including which plausible ones do not. Telling the model not to
repeat itself does not help: it moves the probe's measured quantity slightly the
wrong way, and in free-running rollouts it leaves the exact repeat rate where it
was. Neither does deleting the failed attempt and retrying from a
clean context: that is the **worst** harness we measured, because it restores
precisely the context that produced the failure. What works is replacing the
failed call with a description the runtime generates from its own error
metadata, which keeps the diagnosis and drops the token sequence.

The principle that survives all six harnesses is narrow enough to be useful:

> The context after a failure must differ from the context before it, and the
> difference must not be the failed action itself.

---

## The central quantity

For a context `C`, an action `a` that fails there, and its observation `o`:

```
G(a) = log P(a | C) − log P(a | C ⊕ (a, o))          # corrective gain
```

`G > 0` means the failure record did its job. `G < 0` is *feedback inversion*.
With a valence-free observation `o∅` and the identity `−G = Δcopy + Δsem`:

```
Δcopy = log P(a | C ⊕ (a, o∅)) − log P(a | C)        # the string being present
Δsem  = log P(a | C ⊕ (a, o))  − log P(a | C ⊕ (a, o∅))   # it being marked failed
```

---

## What's here

```
src/slmecho/
  envs/toolshed.py        simulated tool-use world: 12 typed tools, deterministic errors
  envs/toolshed_tasks.py  task templates + the perturbation operators that make wrong calls
  envs/coderepair.py      MBPP with real subprocess execution and real tracebacks
  probe.py                probe items and the condition definitions
  scoring.py              prefix-KV-reusing teacher-forced scorer (+ its verification path)
  harness.py              harness configurations: verbatim / abstract / drop / +instruction
  decoding.py             decoder-level ban on previously-failed action strings
  agent.py                free-running rollouts, repetition metrics
  analysis.py stats.py    derived measures, cluster bootstrap, scaling fits
  plotting.py tables.py   figure style and LaTeX table generation
scripts/                  one entry point per stage; all resumable
paper/                    LaTeX manuscript; every number is \input from generated files
results/raw/              per-score and per-rollout records (JSONL) + run metadata
research_log.md           what was tried, what failed, and why decisions were made
```

## Installation

No GPU is needed; nothing in this repository uses one.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # or: conda env create -f environment.yml
export HF_HOME=$PWD/.hf        # keep ~25 GB of checkpoints off the system drive
```

## Running it

```bash
python scripts/download_models.py --ladder      # ~14 GB, smallest first
python scripts/build_items.py                   # builds both probe item pools
python scripts/verify_scoring.py --model smollm2-135m   # cached vs uncached agreement
python scripts/launch_probes.py --phases ABC    # Studies 1 and 2
python scripts/run_agent_study.py --models qwen2.5-0.5b --n-tasks 30
python scripts/analyze_probe.py && python scripts/analyze_agent.py
python scripts/make_facts.py && python scripts/make_figures.py
```

Every stage is resumable. Probe results are keyed by
`(item, condition, candidate)` and existing keys are skipped; rollouts are
written one file per `(model, env, harness, temperature, seed)`.

**Run it single-threaded.** `scripts/launch_probes.py` starts one process per
model rather than using intra-op threads. That is not a stylistic choice: on the
machine this was developed on, `torch.set_num_threads(n>1)` makes the installed
PyTorch build return all-NaN logits silently. `load_model` now ends with a
forward pass that must produce finite logits and raises otherwise, so you will
find out immediately if your build has the same problem.

## Hardware we used

One laptop CPU (Intel i7-6820HQ, 4 cores, 17 GB RAM), float32, no GPU. The probe
is built out of teacher-forced scoring rather than generation precisely because
of this: on CPU a single decoding step costs about as much as thirty scored
positions. Prefix reuse across conditions removes ~95% of the token positions a
naive implementation would push through the network.

## Reproducing the numbers in the paper

The manuscript contains no hand-typed results. `scripts/make_facts.py` writes
every inline number into `paper/tables/facts.tex` as LaTeX macros, and anything
not yet measured is emitted as a visible placeholder rather than a plausible
number. Tables come from `scripts/analyze_*.py`; figures from
`scripts/make_figures.py` in both PDF and PNG.

`paper/references.bib` is generated by `scripts/fetch_citations.py`, which
resolves every candidate reference against the arXiv API and writes an audit
trail to `paper/references_audit.json`. A reference that does not resolve is not
written.

The built manuscript is committed as [`paper/main.pdf`](paper/main.pdf). To
rebuild it, any TeX distribution will do; the run of record used
[Tectonic](https://tectonic-typesetting.github.io/) 0.17.0, which is a single
self-contained binary and is not vendored here:

```bash
cd paper && tectonic -X compile main.tex
```

Three static checks run before the compile and catch the things a clean LaTeX
build does not: `scripts/check_latex.py` (macros used but never defined,
`\input` targets that nothing includes, cross-references without labels, missing
figures, citation keys absent from the bibliography), `scripts/check_tables.py`
(rows whose column count disagrees with their `tabular` specification) and
`scripts/check_escapes.py` (unescaped `_` and `#` in generated tables). All
three are part of `scripts/reproduce.sh`.

Two of them exist because of specific bugs that compiled silently: a `\ref`
whose backslash had been eaten was typeset as prose without any undefined
reference, and a `\texttt{neut_pad}` in a generated table halted the build only
much later. `research_log.md` has the full list.

## Checkpoints

All models are public instruction-tuned checkpoints pulled from the Hugging Face
Hub at run time; see `src/slmecho/models.py` for the exact ids. No model is
trained or fine-tuned in this project, so there are no checkpoints to release.

## Citation

See `CITATION.cff`. The paper is
[`paper/main.pdf`](paper/main.pdf); the repository is
<https://github.com/Esmail-ibraheem/feedback-that-backfires>.

## License

MIT for the code (see `LICENSE`). MBPP is used under its own license; ToolShed
tasks and error messages are generated by code in this repository.
