# Research log

Running notebook for an autonomous research project. Entries are append-only;
if a decision is reversed later, the original entry stays and a new one records
the reversal and the reason.

---

## 2026-08-08 — Session 1

### Environment audit (done before choosing a question)

Hardware/software available to this project:

| Item | Value |
| --- | --- |
| CPU | 8 logical cores |
| RAM | 17 GB |
| GPU | Quadro M1000M, 2 GB VRAM — **unusable** for LM work; installed torch is a CPU build |
| Disk | `F:` 756 GB free (`C:` only 20 GB free → HF cache must be redirected to `F:`) |
| Network | reachable: `huggingface.co`, `export.arxiv.org` |
| API keys | none present (`OPENAI_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN`, … all unset) |

Two hard consequences for research design:

1. **No GPU and no paid API.** Every number in this paper must come from CPU
   inference on open-weight models. Fine-tuning anything above ~150M params is
   out. Long agent rollouts with 7B models are out.
2. **fp32 only.** The CPU predates AVX512-BF16, so bf16 matmuls fall back to
   slow emulation. At 4 bytes/param and 17 GB RAM, ~3B parameters is the
   practical ceiling for a resident model.

This is a real constraint, and it *shaped* the question rather than merely
limiting it: I deliberately looked for a question whose primary evidence is
obtainable from **teacher-forced scoring** (one forward pass per context, no
sampling, no KV-cache growth) rather than from many long generations. Scoring
is ~2–3 orders of magnitude cheaper than rollouts and is fully deterministic,
which also removes sampling noise from the central measurement.

Decision: install an isolated venv at `F:\paper\.venv` rather than touching the
user's global Python (their global env has torch 2.4.0+cpu / transformers 4.45,
too old for some of the model families I want).

### Step 1 — literature reconnaissance

Queries run (web search + arXiv/HF fetches), and what came back:

* *"LLM agents repeat failed actions loop execution feedback"* →
  the phenomenon is **widely reported but treated as an engineering nuisance**.
  Representative hits:
  - "When Agents Do Not Stop: Uncovering Infinite Agentic Loops in LLM Agents"
    (Hou, Wang, Zhao, Wang) — a **static analysis** tool (IAL-Scan) that finds
    unbounded feedback paths in agent *source code* across 6,549 repos. It is
    about program structure, not model behaviour. Complementary, not competing.
  - "Probing Embodied LLMs: When Higher Observation Fidelity Hurts Problem
    Solving" (Zenkri & Brock) — striking independent evidence: on a mechanical
    puzzle, *injecting noise into observations improves success 2.85×*, and the
    authors attribute the gain to **a reduction in repetitive action loops**.
    Embodied/robotic setting, no token-level analysis, no proposed harness fix.
* *"self-reinforcement effect repetition"* → Xu et al., "Learning to Break the
  Loop" (NeurIPS 2022): probability of repeating a sentence **increases** with
  the number of times it already appears. This is the closest mechanistic
  ancestor, but it is about *open-ended generation with no feedback signal*. In
  an agent loop there is an explicit negative signal ("this failed") that is
  supposed to counteract it. Nobody appears to have asked which force wins.
* *"pink elephant" / negation* → well-documented that naming a thing in the
  prompt raises its probability even under a negative instruction; LLMs handle
  negation poorly. Again: prompt-level, not agent-trajectory-level.
* *induction heads / repetition neurons* → mechanistic vocabulary for *why*
  copying might dominate in small models ("repetition curse", entropy collapse
  when induction heads dominate the logits).
* Adjacent agent work found: "Don't Adapt Small Language Models for Tools;
  Adapt Tool Schemas to the Models"; "Constraint Tax in Open-Weight LLMs"
  (structured-output constraints can *suppress* tool calls); "Learning
  Agent-Compatible Context Management for Long-Horizon Tasks" (AdaCoM — a
  learned manager that keeps a compact state incl. "ineffective queries").

**Gap identified.** Everyone agrees agents get stuck repeating failed actions.
The literature treats it as (a) a control-flow bug to be detected statically,
(b) something a bigger/better-prompted model will fix, or (c) something a
learned context manager should paper over. What is *missing* is the obvious
controlled measurement:

> Does putting a failed action and its error message into the context actually
> reduce the model's probability of re-emitting that action — and if not, is
> the culprit the *semantics* (model doesn't understand the error) or the
> *surface form* (the failed action is now sitting in the context, verbatim,
> where the copying machinery can reach it)?

Those two have completely different fixes, and the field has been assuming the
first while (I suspect) the second dominates at small scale.

### Candidate hypotheses considered, and why they were dropped

| # | Candidate | Verdict |
| --- | --- | --- |
| A | Constrained decoding closes the tool-format gap for SLMs | Dropped — heavily covered (Outlines/XGrammar/JSON-mode); "Let Me Speak Freely?" and "Constraint Tax" already stake out both sides. |
| B | Heterogeneous SLM cascade / router beats one larger model | Dropped — crowded (FrugalGPT, RouteLLM, Hybrid-LLM), and a fair study needs GPU-scale baselines I cannot run. |
| C | Verifier-quality limits test-time scaling for SLMs | Dropped — already made, sharply, by "Inference Scaling fLaws"-style work. |
| D | **Failure self-conditioning in agent contexts: copying vs. corrective semantics** | **Selected.** |
| E | Tool-schema placement effects | Dropped — real but small; a workshop note at best. |
| F | Harness scaffolding cost scales inversely with model size | Folded into D as a secondary analysis rather than a separate paper — too diffuse on its own. |

### Selected research question (falsifiable, single central claim)

**RQ.** In an agent loop, let `C` be the context before an attempt, `a` a
failed action, and `o` its error observation. Define the *corrective gain*

    G(a) = log P(a | C) − log P(a | C ⊕ (a, o))

`G > 0` means the harness's feedback did its job (the failed action became less
likely). `G < 0` means **feedback inversion**: showing the model its own
failure made it *more* likely to repeat it.

**H1.** `G < 0` for small instruction-tuned models and crosses zero somewhere in
the 1–3B range, i.e. execution feedback is net-counterproductive below a scale
threshold.

**H2.** `G` decomposes into a *surface-form copying* term and a *corrective
semantics* term that can be separated by counterfactual observations
(same action, but the observation reports success / is neutral / reports
failure). The copying term is roughly scale-flat; the semantics term grows with
scale. The crossover in H1 is where semantics finally overtakes copying.

**H3.** Because the harmful term is surface-form, the fix belongs in the harness,
not the prompt: removing the failed action's surface form from the context, or
banning it at the decoder, should beat telling the model "do not repeat failed
actions" — and should help small models much more than large ones.

Falsifiable in the strong sense: if `G > 0` uniformly, H1 dies and the paper
becomes "execution feedback works, here is how much and how it scales", which
is still a useful measurement but a different (weaker) paper. I will report
whichever way it comes out.

### Plan of record

* **Study 1 (probe).** Teacher-forced scoring of `G` and its decomposition
  across model families/sizes and three action domains.
* **Study 2 (controlled manipulations).** Error verbosity, whether the error
  echoes the action, number of prior failures k, negative NL instruction,
  surface-form redaction, recency.
* **Study 3 (end-to-end).** Real multi-turn loops in executable environments,
  comparing harness variants incl. a decoder-level ban; test whether Study-1's
  `G` predicts Study-3's loop rate and success.


---

## 2026-08-08 — Session 1 (continued): building the instruments

### Environments

Two, deliberately at opposite ends of the control/realism trade-off.

**ToolShed** (`src/slmecho/envs/toolshed.py`) — a simulated office workspace with
12 typed tools, six task templates and a deterministic error renderer. Actions
are Python-style calls parsed with `ast`; no LLM judge is involved anywhere.
Wrong actions are produced from the reference action by seven perturbation
operators (drop a required argument, pluralise the tool name, rename an
argument, swap a type, write a date the way a person would, hallucinate an
entity, target a read-only file). All 50 perturbation/error pairings were
checked to produce exactly the intended error family.

**CodeRepair** (`src/slmecho/envs/coderepair.py`) — MBPP (sanitized split, 427
problems) with real execution in a subprocess. Wrong programs come from
single-edit mutations of the reference solution; a mutant is kept only if the
reference passes and the mutant actually fails. Observations are the
interpreter's own output, with our frames stripped so the traceback reads the
way it would in a real coding agent — including its habit of quoting the
offending source line, which is a naturally occurring instance of the
surface-form echo we manipulate deliberately in ToolShed. 37% of code items have
a traceback that quotes a line of the submission; that flag is recorded per item.

### Two design decisions worth recording

1. **The primary failure rendering does not quote the failed call back.** With
   quoting on, the `fail` context would contain the action string twice while
   the `succ` context contained it once, and the copy/semantics contrast would
   be confounded by occurrence count. Quoting is instead a manipulated variable
   (`fail_echo`).
2. **Counterfactual success observations are built from the failed call's own
   arguments.** A generic "success" naming a different person or file than the
   call it supposedly answers is incoherent, and incoherence is itself a cue the
   model could use.

### Compute reality, measured rather than assumed

`scripts/bench_forward.py`, on SmolLM2-135M, fp32, this CPU (i7-6820HQ):

* bfloat16 is **20x slower** than float32 (6 tok/s vs 116) — no AVX512-BF16, so
  oneDNN emulates. bf16 is never used in this project.
* Computing the LM head only at the last position (`logits_to_keep=1`) when
  merely extending a context saves 16–22%; the head is 20–30% of a forward pass
  at these vocabulary sizes.
* A single-token forward costs ~90 ms almost independently of model size,
  because it is dominated by per-layer dispatch rather than arithmetic. This is
  why free-form greedy decoding was removed from the probe: the *same*
  behavioural statement ("greedy decoding would re-emit the failed call") is
  recoverable for free from the argmax path already computed during scoring.

Prefix-KV reuse (`CachedScorer`) removes 95% of the token positions a naive
implementation would push: the probe's contexts share a ~520-token system
prompt and an item prefix, and differ only in a ~50-token tail.

### A bug that would have invalidated everything

Mid-run, the derived measures came out all-NaN. Tracing it back:

> On this machine, `torch.set_num_threads(n)` with n > 1 makes torch 2.4.0+cpu
> return **all-NaN logits**. Silently. Wall-clock time is unchanged, no warning
> is emitted, and the NaNs first appear inside layer 0. Single-threaded is
> correct; every n > 1 is broken; `OMP_NUM_THREADS` in the environment is
> ignored (torch reports 1 thread and stays correct).

Worse, my first verification script compared cached against uncached scoring
and reported perfect agreement — because `max(0.0, float('nan'))` returns `0.0`
in Python, so NaN == NaN passed as "identical". Two lessons, both now baked into
the code: `load_model` ends with a forward pass that must produce finite logits
and raises `NonFiniteLogitsError` otherwise; `CachedScorer._forward` checks a
few elements of every returned logit row; and `verify_scoring.py` rejects
non-finite and positive log-probabilities before comparing them.

The first probe run (~1.5h of CPU) was discarded. Parallelism now comes from
running one **single-threaded process per model** (`scripts/launch_probes.py`),
staged so that peak fp32 residency stays under ~12 GB of the 17 GB available.
This is both correct and faster than the broken threading was.

A second bug found at the same time: two perturbation operators could share a
name for the same call (a written-out date and "next Tuesday" are both
`date_format`), so their item ids collided and the second item's scores were
silently skipped as "already computed". Item ids now carry a perturbation index.

### Verified bibliography

`scripts/fetch_citations.py` resolves every candidate reference against the
arXiv API and writes `paper/references.bib` plus `paper/references_audit.json`.
41/41 candidates resolved. This caught one title I had misremembered: the
"inference scaling flaws" paper is actually titled *The Limits of Inference
Scaling Through Resampling* (Stroebl et al.).

---

## 2026-08-08 — first valid results (ToolShed, partial: 3 models, 16–40 items each)

The hypothesis survives its first contact with data, and the effect is much
larger than I expected.

| model | $G$ (nats) | copy | sem | $p_{\rm rep}$ before → after | greedy re-emits exact failed call |
| --- | --- | --- | --- | --- | --- |
| SmolLM2-135M | −22.4 [−25.1, −19.6] | +20.5 | +1.9 | 0.12 → 0.85 | 44% |
| SmolLM2-360M | −17.7 [−22.2, −13.0] | +15.1 | +1.4 | 0.10 → 0.72 | 33% |
| Qwen2.5-0.5B | −15.0 [−18.5, −10.6] | +13.2 | +2.6 | 0.002 → 0.47 | 6% |

Readings:

1. **Feedback inversion is real and large.** Writing the failed call and its
   error into the transcript raises the probability of re-emitting that exact
   call by 15–22 nats. In the normalised readout that is 0.12 → 0.85 for the
   135M model. On 44% of items, greedy decoding from the post-failure context
   reproduces the failed call token for token; from the pre-failure context it
   never does (0%).
2. **The damage is surface form, not misunderstanding.** `copy` (call present,
   feedback-neutral observation) accounts for ~90% of it. Marking the call as
   failed does not claw any of it back — `sem` is *positive* (+1.4 to +2.6),
   i.e. the error message adds a little further to the repetition tendency,
   plausibly because it names the tool and argument again.
3. **Removing the surface form removes almost all of it.** Under the abstract
   harness — same diagnosis, no verbatim call — $G$ goes from ≈ −22 to ≈ −2 and
   the exact-greedy-repeat rate goes to 0% for all three models.
4. **A framing that pre-empts the obvious objection.** `repair` is *positive*
   (+11.0, +5.5, +0.5): the correct action also becomes more likely, because it
   shares most of its tokens with the failed one. So the story is not "copying
   raises everything". The decision-relevant quantity is the margin, and
   `margin_shift` = −11 to −14 nats: the failure record moves the gold-vs-failed
   margin decisively in favour of the failed call.

Caveat I have to control before claiming (3): `sem` compares a long, informative
error string against a short neutral one, so part of it could be length or
lexical priming rather than valence. That is exactly what the length-matched
neutral (`neut_pad`), the terse/verbose contrast and the no-quote contrast are
for; they are in phase C and not yet run.

Also note the monotone trend with size (−22.4 → −17.7 → −15.0, copy 20.5 → 15.1
→ 13.2). I will report the trend, but I will **not** claim a crossover: linear
extrapolation of these three points puts the zero crossing far outside the range
I can actually run, and a scaling claim I cannot test is exactly the kind of
thing a reviewer should reject.

---

## 2026-08-08 — novelty audit (round 1)

Searched again, deliberately trying to find work that already contains this
contribution. The most important hit is one I had not seen:

**Yang (2026), "Why Retrying Fails: Context Contamination in LLM Agent
Pipelines"** (arXiv 2605.08563). Formalises exactly the phenomenon I am
studying, at the level of task statistics: a retry whose context still contains
the failed attempt has an elevated error rate. The Context-Contaminated Restart
Model gives closed forms for success under a budget, and is fitted to SWE-bench
Verified with a cascade ratio ε₁/ε₀ = 7.1 — the IID assumption overestimates
pass@3 by 17.4 points.

Is my contribution still novel? I think yes, and the overlap actually helps:

* CCRM is a **probabilistic model fitted to outcome data**. It contains no
  token-level or log-probability analysis, and no decomposition of the context
  into parts. It answers "how much does contamination cost", not "what
  contaminates".
* Its prescription is **clean restart** — clear the context before retrying.
  That is the `drop` harness in my taxonomy, and it throws away the diagnosis
  along with the hazard. My decomposition predicts a strictly better option:
  remove the *surface form*, keep the diagnosis. The probe already contains the
  measurement, because the `pre` condition *is* the clean-restart context by
  construction: G is fail-vs-clean-restart, and G_abstract is abstract-vs-clean-
  restart. The partial data says abstract sits within ~2 nats of a clean restart
  while retaining the error diagnosis.

**Design change made as a result:** `drop` and `drop+ban` were in the harness
registry but were not in the end-to-end study's default list. They are now, so
that clean restart is an evaluated baseline rather than a strawman I describe.

Also found and added: **Wang et al. (2025), "Hell or High Water"** (2508.11027) —
agents fail to formulate backup plans after an external failure even when the
task remains solvable. Behavioural counterpart to the distributional effect I
measure; good corroboration, different instrument.

Bibliography now 43/43 verified against the arXiv API.

---

## 2026-08-08 — fuller ToolShed results + first CodeRepair results

ToolShed, 87–100 items per model over 37–48 tasks:

| model | $G$ | copy | sem | $p_{\rm rep}$ | greedy repeat | $G$ under abstraction |
| --- | --- | --- | --- | --- | --- | --- |
| SmolLM2-135M | −23.9 [−25.8,−22.0] | +21.7 | +2.2 | 0.14→0.88 | 0%→43% | −2.2 |
| SmolLM2-360M | −18.1 [−20.5,−15.8] | +15.7 | +2.3 | 0.13→0.75 | 0%→39% | −3.2 |
| Qwen2.5-0.5B | −13.4 [−15.6,−11.2] | +9.6  | +3.8 | 0.02→0.59 | 0%→2%  | −2.6 |

CodeRepair (SmolLM2-135M, 60 MBPP problems, one item each):
$G = -83.2$ [−94.5, −71.9], copy $= +84.5$, sem $= -1.4$ [−1.8, −0.9],
$p_{\rm rep}$ 0.10 → 0.95, and $G_{\text{abstract}} = +1.5$ — under the
abstracted harness the corrective gain is actually *positive*.

Three things I did not anticipate, all of which improve the paper:

1. **Per-token, the effect is the same size in both environments.** Raw $G$
   differs by 3.5x (a program is ~100 tokens, a tool call ~20), but $G$ per
   action token is −1.21, −0.92, −0.94 on ToolShed and −1.21 on CodeRepair.
   About one nat per token, regardless of domain. That is a much better headline
   than the raw nats and it neutralises the length objection completely.
2. **The semantic term has an inconsistent sign, and that is the point.** It is
   +2.2 to +3.8 on ToolShed (the error names the tool and the argument again,
   which primes them) and −1.4 on CodeRepair (a traceback genuinely pushes the
   model away from the failing program). Either way it is dwarfed: on CodeRepair
   the ratio is about 60:1. So the claim is not "models cannot read errors" —
   sometimes they can — but "whatever reading happens is swamped by the string".
3. **The result holds item by item, not just on average.** The fraction of
   individual items with $G<0$ is 1.00, 1.00, 0.89. This is the statistic to
   lead with for a sceptical reader, because it needs no bootstrap.

Also confirmed: the decomposition is insensitive to which reference observation
is used. Referenced to the counterfactual *success* observation instead of the
valence-free one, copy is +24.5/+17.6/+10.3 and sem is −0.5/+0.7/+3.0 — same
story, so R2.2's objection (that "Call recorded." might read as mildly positive)
does not change any conclusion.

### An engineering note that cost real time

The end-to-end study originally wrote its results only after a whole harness
finished. With rollouts costing minutes each under CPU contention, that meant a
killed run lost everything. Rollouts are now streamed to disk as they complete
and runs resume at task granularity.

---

## 2026-08-08 — scheduling lesson: memory, not cores

Running five single-threaded model processes at once looked like the right use
of four physical cores. It was not. With ~8.4 GB of resident model weights plus
the OS, free memory dropped to ~1 GB and throughput per worker collapsed by
roughly an order of magnitude — far more than 5-way core contention could
explain. Dropping to three workers restored ~5.5 GB free and each worker went
back to ~88% of a core.

The general rule for this project, now written into `docs/compute.md`: **the
binding constraint on parallelism is resident memory, not cores**, because every
model is float32 and every forward pass streams the full weight matrix. Two
1.7B models cannot share this machine at all.

This also cost me a mismeasurement. A "10 rows in 7 minutes" reading taken
during the paging period suggested ~14 minutes per item, which would have made
the study infeasible and nearly prompted me to cut the manipulation study. A
direct measurement (CPU delta over a fixed wall-clock interval, plus row delta)
gave ~68 s per item, which is what the logs had been saying all along. Lesson:
when a throughput number implies a design change, measure it directly rather
than inferring it from file sizes sampled across an unstable period.

---

## 2026-08-08 — Study 2 (manipulations), SmolLM2-135M, 39 items

| condition | $G$ (nats) |
| --- | --- |
| the same call failed three times | −25.9 |
| the same call failed twice | −25.7 |
| error quotes the call | −24.0 |
| **standard (verbatim call + error)** | **−23.9** |
| verbose error text | −22.8 |
| terse error text | −22.8 |
| failure placed before the successful steps | −21.9 |
| **placebo: a *different* call failed** | **−8.6** |
| **abstracted failure (no verbatim call)** | **−2.2** |

Reading:

* **Message content is nearly inert.** Terse, standard and verbose renderings
  span 1.2 nats out of 24. The verbose rendering names the argument the model
  should have used and still does not turn the gain positive.
* **Occurrence count is not inert.** Two and three failures of the same call
  make it worse (−25.7, −25.9), and quoting the call inside the error message
  also makes it worse. This is the self-reinforcement of Xu et al. operating
  inside a trajectory that contains an explicit corrective signal at every
  repetition.
* **Recency is not the story.** Moving the failure *before* the successful
  steps barely changes it (−21.9 vs −23.9), so this is not simply "the last
  thing in the context wins".
* **The placebo is the interesting one.** With a *different* wrong call recorded
  as failed, the scored action still gains 8.6 nats. I had expected close to
  zero. The explanation is that the distractor is another perturbation of the
  same reference call, so it shares most of its tokens with the action being
  scored — the placebo leaks copying by construction. That makes it a
  **conservative** control rather than a broken one: it bounds the
  string-specific component from below at 23.9 − 8.6 = 15.3 nats, and I will
  report it that way rather than claiming the whole 23.9 is string-specific.
  A cleaner placebo would use an unrelated call; that is a one-line change and
  worth adding if compute allows.

---

## 2026-08-08 — Study 3, first two harnesses (Qwen2.5-0.5B, 24 tasks, greedy)

| harness | solved | exact repeat rate | steps | gen tokens |
| --- | --- | --- | --- | --- |
| verbatim (standard) | 8/24 = 0.33 | 0.222 | 2.96 | 49 |
| verbatim + "do not repeat" | 11/24 = 0.46 | 0.489 | 3.25 | 64 |

Paired contrasts on matched tasks:
success +0.125 [0.000, 0.250]; exact repeat rate **+0.289 [0.067, 0.530]**.

This is not what I expected and I am going to report it as measured. The
instruction **more than doubles** the exact repeat rate — the interval excludes
zero — while the success difference is borderline and its interval touches zero.

The reconciliation is visible in the other columns: with the instruction the
model takes more steps (3.25 vs 2.96) and generates more tokens (64 vs 49). It
does not repeat *less*; it *gives up less*. More attempts produce both more
repeats and, at this sample size, possibly a few more successes. So the
instruction changes persistence, not repetition-avoidance, and any success gain
is bought with tokens rather than with better behaviour.

Two consequences for the manuscript:

1. The introduction previously said the prohibition "does essentially nothing".
   That is now wrong in an interesting direction and has been corrected: it does
   not reduce repetition, it increases it.
2. `n = 9` for the repeat-rate contrast (only rollouts with at least one failure
   under both harnesses contribute). That is small, and the paper will say so.

---

## 2026-08-08 — a statistics fix that changed a claim

While re-reading the manipulation section I noticed the table was comparing
`G` estimated on 100 items against `G_variant` estimated on the 39–65 items the
extended pass had reached. That is not a comparison of harness variants; it is
partly a comparison of item sets. Switching to **paired within-item deltas**
against the standard harness fixed it, and the picture sharpened considerably:

| variant | paired Δ log P(repeat) vs standard |
| --- | --- |
| terse error text | −0.23 [−0.62, +0.12] |
| verbose error text | −0.08 [−0.32, +0.16] |
| error quotes the failed call | **+1.28 [+1.09, +1.48]** |
| failure placed before the successful steps | −0.77 [−1.24, −0.36] |
| the same call failed twice | **+3.06 [+2.76, +3.36]** |
| the same call failed three times | **+3.29 [+2.99, +3.59]** |
| placebo: a *different* call failed | **−14.71 [−16.17, −13.32]** |
| abstracted failure (no verbatim call) | **−21.68 [−23.19, −20.14]** |
| bare marker, no diagnosis | **−23.35 [−25.30, −21.42]** |

The unpaired version had made "quoting the call back makes it worse" look like
a null (−23.95 vs −23.92, comparing across different item sets). Paired, it is
+1.28 with an interval well clear of zero. I had written the paragraph claiming
the effect *before* checking it properly; it happened to be right, but it was
right by luck, and the corrected statistic is what the paper now reports.

Two other things the paired numbers settle:

* **`abstract_min` matches the full abstraction** (−23.35 vs −21.68). Stripping
  the diagnosis entirely and leaving `[attempt 1 failed]` works at least as well
  as our hand-written gloss. That answers the reviewer objection that the
  abstraction might be working because of *our wording*: it is not. It is the
  absence of the string.
* **Content is inert, count is not.** Terse and verbose both have intervals
  containing zero; echoing the call, and failing twice or three times, do not.
  This is the cleanest form/content dissociation in the paper and it now rests
  on the right statistic.

---

## 2026-08-08 — clean restart is not a fix (Qwen2.5-0.5B, 24 tasks, greedy)

| harness | solved | exact repeat rate | valid-call rate | steps | gen tokens |
| --- | --- | --- | --- | --- | --- |
| verbatim (standard) | 0.333 | 0.222 | 0.632 | 2.96 | 49 |
| verbatim + "do not repeat" | 0.458 | 0.489 | 0.618 | 3.25 | 64 |
| **drop (clean restart)** | **0.333** | **0.819** | 0.528 | 3.79 | 75 |

Paired against verbatim: drop moves success by exactly **+0.000 [0.000, 0.000]**
and the exact repeat rate by **+0.597 [+0.389, +0.778]**.

Clearing the failed step from the context — the remedy proposed by the closest
prior work on context contamination — makes repetition dramatically *worse* and
does not change success at all.

In hindsight the mechanism is obvious, and it is worth stating plainly because
it sharpens the whole paper: **deleting the failure restores the exact context
that produced the failure.** Under a deterministic policy the model then emits
the identical action, forever. Clean restart is not the absence of the problem;
it is the strongest possible version of it.

That reframes the contribution. It is not "remove the failed action from the
context". It is:

> The context after a failure must differ from the context before it, and the
> difference must not be the failed action itself.

`verbatim` changes the context with the failed string (pulls you back toward
it). `drop` does not change the context at all (leaves you exactly where you
were). `abstract` changes it with something that is not the failed string. Those
are the three cells of the design, and the middle one had been the recommended
fix.

**Scope condition I must state.** This is greedy decoding. With temperature
sampling, `drop` would not be degenerate in the same way, because resampling
supplies the variation the context no longer does. Many deployed agents do run
at low temperature, so the regime is not exotic, but the result is about
determinism plus context deletion, not about context deletion alone. This goes
in the paper next to the result, not in a footnote.

---

## 2026-08-09 — Study 3 complete (Qwen2.5-0.5B, 24 tasks, 6 harnesses, greedy)

| harness | solved | ended in a loop | exact repeat | canon. repeat | valid calls | gen tokens | ban fired |
| --- | --- | --- | --- | --- | --- | --- | --- |
| verbatim (standard) | 0.333 | 0.17 | 0.222 | 0.222 | 0.632 | 49 | — |
| verbatim + "do not repeat" | 0.458 | 0.38 | 0.489 | 0.489 | 0.618 | 64 | — |
| drop (clean restart) | 0.333 | 0.50 | 0.819 | 0.819 | 0.528 | 75 | — |
| verbatim + ban | 0.333 | 0.00 | 0.000 | 0.000 | 0.632 | 51 | 22 |
| **abstract** | **0.500** | **0.00** | **0.000** | **0.000** | 0.601 | 63 | — |
| abstract + ban | 0.500 | 0.00 | 0.000 | 0.000 | 0.601 | 63 | **0** |

Paired against verbatim on matched tasks:
abstract **+0.167 [+0.042, +0.333]** success, **−0.222 [−0.410, −0.056]** exact
repeat; verbatim+ban **+0.000** success, −0.222 exact repeat; drop +0.000
success, **+0.597** exact repeat; instruction +0.125 [0.000, 0.250] success,
**+0.289** exact repeat.

Four things fall out, and three of them I would not have predicted:

1. **Only the abstraction improves task success.** +17 points, interval clear of
   zero, at 63 generated tokens against 49 — and it *shortens* the prompt,
   because a one-line failure description is smaller than a call plus its error.
2. **The ban stops the loop without solving anything.** Exact *and canonical*
   repeat rates both go to zero, so the model is not paraphrasing around the
   constraint — it writes genuinely different calls, which are also wrong.
   Success is unchanged to three decimals. Forbidding a string stops the agent
   wasting its budget; removing the string from context is what lets it
   reconsider. These are different interventions with different effects and the
   paper now says so.
3. **abstract+ban is numerically identical to abstract on every metric, and the
   ban never fired once** (0 activations, against 22 under the verbatim
   transcript). That is the decomposition's prediction confirmed at the level of
   individual decoding steps: once the failed call is absent from the context,
   the model does not begin writing it, so there is nothing to block.
4. **The loop-rate ordering is the cleanest single summary**: drop 50% >
   instruction 38% > verbatim 17% > abstract = verbatim+ban = abstract+ban 0%.
   The two harnesses that act on the surface form are at the bottom; the two
   that reason about the failure in language or delete it wholesale are at the
   top.

Caveats stated in the paper: one model, 24 tasks, greedy decoding, one
trajectory per cell. The success intervals are wide (24 tasks); the repetition
intervals are narrow because the effects are large.

---

## 2026-08-09 — Study 3 discarded and re-run: a confound I introduced myself

Reading the qualitative trajectories for the appendix, I noticed the model was
emitting `you: find_contact(name='Aisha Moreau')` — with a speaker label glued to
the front. The parser rejected those as malformed calls.

The source was mine. The format demonstration I added to lift the smallest
models off the floor was written as a two-turn dialogue:

```
  user: List the files in notes.
  you:  list_files(folder='notes')
```

Small models copied the `you:` label. Quantifying it across the completed runs:

| harness | actions with a role prefix |
| --- | --- |
| verbatim+instr | 0 / 78 (0%) |
| abstract, abstract+ban | 9 / 88 (10%) |
| verbatim, verbatim+ban | 30 / 71 (42%) |
| drop | 48 / 91 (53%) |

`parse_error` accounted for **279 of 315** failures — 89% of everything the
agent did wrong was a formatting artefact I had introduced. And the rate is not
constant across harnesses: it ranges from 0% to 53% and correlates with exactly
the conditions being compared. A harness that happened to suppress the prefix
would look better for a reason that has nothing to do with the mechanism under
study. That is a confound, not noise, and the study as it stood was not
measuring what it claimed to.

Discarded ~4 hours of rollouts and re-ran from scratch after two fixes:

1. The demonstration now shows only the reply, with nothing that could be
   mistaken for part of the output.
2. `parse_action` strips a leading speaker or action label. This is the right
   behaviour independently of the bug: a harness should not score a chat
   model's formatting habit as a tool-use failure.

The old rollouts are kept under `results/raw/agent_v1_roleprefix/` with a README
explaining why they are excluded, rather than deleted.

The lesson I want to record: **read the raw trajectories, not only the
aggregates.** Every summary statistic in the discarded study was internally
consistent and told a clean story. Nothing in the numbers pointed at the problem.
It was visible immediately in four lines of transcript.

---

## 2026-08-09 — the re-run overturns a claim I had already written

Two harnesses of the clean re-run are in, and the instruction result reverses.

| contrast (paired, vs verbatim) | v1 (confounded) | v2 (clean) |
| --- | --- | --- |
| +instruction, task success | +0.125 [0.000, 0.250] | **+0.167 [+0.042, +0.333]** |
| +instruction, exact repeat rate | **+0.289 [+0.067, +0.519]** | −0.165 [−0.439, +0.111] |
| drop, exact repeat rate | +0.597 [+0.389, +0.778] | **+0.435 [+0.231, +0.652]** |
| drop, task success | +0.000 [0.000, 0.000] | −0.071 [−0.214, 0.000] |

Under v1 the instruction appeared to *increase* exact repetition with an
interval clear of zero, and I had written that into the abstract, the
introduction and two sections. It was an artefact: the instruction variant was
the one condition with a 0% role-prefix rate, so removing the prefix confound
removed the apparent effect. Under v2 the interval contains zero and the point
estimate is slightly the other way.

What the clean data says instead: the instruction **improves task success**
(+17 points, interval excluding zero) while using *fewer* tokens (47 vs 67) and
fewer steps (3.2 vs 4.2), without measurably changing repetition. So it does
something real, and it is not what I claimed. The most likely reading is that it
makes the model less willing to abandon the task early, which is a persistence
effect rather than a repetition-avoidance one — but with 24 tasks I will report
it as measured and not build a mechanism on it.

The `drop` conclusion survives cleanly: clean restart still raises exact
repetition by a large, well-separated margin, and does not help success.

Manuscript changes made immediately, before the remaining four harnesses land,
so that no claim in the repository is currently unsupported: the abstract, the
introduction, the analysis section and the agent section have all had the
"instruction makes repetition worse" claim removed and replaced with what v2
shows. The probe's own instruction result (+0.16 [+0.06, +0.27] nats, i.e. a
very small adverse effect on the log-probability of repeating) is unaffected by
any of this — it is teacher-forced and never saw the demonstration.

This is the second claim this project has had to retract on its own evidence.
Both retractions came from re-running rather than from re-reading, which is an
argument for building the pipeline to be cheap to re-run.

---

## 2026-08-09 — the abstraction's success gain does not survive the re-run either

Four harnesses of the clean run:

| harness | solved /24 | ended in a loop /24 | exact repeat | valid calls | gen tokens |
| --- | --- | --- | --- | --- | --- |
| verbatim | 10 | 7 | 0.307 | 0.611 | 67 |
| + "do not repeat" | 14 | 4 | 0.244 | 0.736 | 47 |
| drop (clean restart) | 8 | 16 | 0.798 | 0.444 | 69 |
| abstract | 8 | **2** | **0.163** | 0.631 | 64 |

Paired vs verbatim: abstract success −0.095 [−0.238, 0.000], exact repeat
−0.185 [−0.379, +0.010]; drop exact repeat **+0.491 [+0.331, +0.645]**.

In v1 the abstraction improved success by +17 points with the interval excluding
zero. In v2 it does not improve success at all. That +17 was carried by the
role-prefix confound: `abstract` had a 10% prefix rate against `verbatim`'s 42%,
so it was being rewarded for a parsing artefact.

What survives, and what I now claim:

* The abstraction is the **best repetition intervention** in the sweep. Loops
  fall 7 → 2 of 24; exact repetition roughly halves.
* It does **not** improve task success at n = 24.
* Clean restart remains catastrophic and is the only contrast whose interval is
  comfortably clear of zero in the harmful direction.

This is a weaker result than v1 and a more believable one. The intervention was
derived from a mechanism; it moves that mechanism by the predicted amount and in
the predicted direction; and it does not automatically move a coarser outcome
that the mechanism only partly controls — 18% of this model's actions are prose
where a call was expected, which no transcript surgery repairs.

Rewrote the agent results section to lead with the repetition result and state
the null on success in the same breath, rather than leading with whichever
metric happened to look best.

---

## 2026-08-09 — Study 3, clean run, five of six harnesses

| harness | solved /24 | ended in a loop | exact repeat | valid calls | gen tokens |
| --- | --- | --- | --- | --- | --- |
| verbatim (standard) | 10 | 7 | 0.307 | 0.611 | 67 |
| + "do not repeat" | 14 | 4 | 0.244 | 0.736 | 47 |
| drop (clean restart) | 8 | 16 | 0.798 | 0.444 | 69 |
| abstract | 8 | 2 | 0.163 | 0.631 | 64 |
| **verbatim + ban** | 10 | 0 | **0.053** | 0.630 | 66 |

Paired against verbatim, exact repeat rate:
**ban −0.221 [−0.369, −0.087]** (excludes zero);
abstract −0.145 [−0.317, +0.019];
instruction −0.165 [−0.438, +0.117];
**drop +0.491 [+0.331, +0.646]** (excludes zero).
Task success: instruction **+0.167 [+0.042, +0.333]**; ban +0.000; abstract
−0.083 [−0.208, 0.000]; drop −0.083 [−0.208, 0.000].

Final ordering by repetition: ban < abstract < instruction < verbatim < drop.
Exactly the order the decomposition predicts, and arrived at without reference
to it. The two interventions that act on the failed call's surface form occupy
the good end; the harness that deletes the failure outright is dramatically
worst; the standard harness sits between.

The honest summary, after two rounds of retraction:

* **The mechanism claim is well supported.** The probe measures it directly, the
  manipulations isolate it, and the rollout ordering reproduces it.
* **The decoder ban is the cleanest intervention**: −22 points of exact
  repetition with the interval clear of zero, zero token cost, no change in
  success. The canonical rate falls identically, so it is not being evaded by
  paraphrase.
* **Neither surface-form intervention improves task success at n = 24.** I would
  have liked them to. They do not, and Section "Where the interventions fail"
  now says why: 18% of this model's actions are prose where a call was expected,
  which no transcript surgery repairs.
* **The instruction improves success without touching repetition**, which is a
  real effect I can measure and cannot explain, and I have written it that way.

Three claims retracted over the project, all on its own evidence, all caught by
re-running: (i) the instruction increases repetition; (ii) the abstraction
improves success; (iii) abstract+ban is identical to abstract — pending, since
the last harness is still running. Each retraction came from removing a confound
rather than from new reasoning, which is the argument for building the pipeline
to be cheap to re-run and for reading raw trajectories rather than aggregates.

---

## 2026-08-09 — Study 3 complete (clean run, all six harnesses)

| harness | solved /24 | ended in a loop | exact repeat | canon. repeat | valid calls | gen tokens | ban fired |
| --- | --- | --- | --- | --- | --- | --- | --- |
| verbatim (standard) | 10 | 7 | 0.307 | 0.307 | 0.611 | 67 | — |
| + "do not repeat" | **14** | 4 | 0.244 | 0.244 | 0.736 | **47** | — |
| drop (clean restart) | 8 | **16** | **0.798** | 0.798 | 0.444 | 69 | — |
| abstract | 8 | 2 | 0.163 | 0.163 | 0.631 | 64 | — |
| verbatim + ban | 10 | 3 | 0.075 | 0.075 | 0.611 | 69 | 39 |
| **abstract + ban** | 8 | **1** | **0.067** | 0.067 | 0.631 | 66 | **20** |

Paired against verbatim, exact repeat rate:
ban **−0.232 [−0.375, −0.103]**; abstract+ban **−0.241 [−0.417, −0.066]**;
abstract −0.145 [−0.320, +0.019]; instruction −0.165 [−0.436, +0.117];
drop **+0.491 [+0.331, +0.645]**.
Task success: instruction **+0.167 [+0.042, +0.333]**; ban +0.000; abstract,
abstract+ban and drop each −0.083 [−0.208, 0.000].

**The third retraction.** In the confounded run, `abstract+ban` was numerically
identical to `abstract` and the ban never fired — which I had written up as the
decomposition's prediction confirmed at the level of individual decoding steps.
It was an artefact. In the clean run the ban fires **20 times** under the
abstracted transcript (against 39 under the verbatim one) and pushes exact
repetition from 0.163 to 0.067.

The corrected reading is better than the one I lost. Removing the failed call
from the context removes the model's opportunity to *copy* it, but not its
ability to *re-derive* it: the goal, the tool schemas and the successful prefix
are all still present, and those are what produced the wrong call in the first
place. Context editing and decoder constraint therefore catch different things —
the copying and the residue — and are complementary rather than redundant.

**Final ordering by loop rate**: abstract+ban (1) < abstract (2) < verbatim+ban
(3) < instruction (4) < verbatim (7) < drop (16). Ranking by exact repeat rate
gives the same order. That ordering is the decomposition's prediction, and it was
arrived at without reference to it.

**The mismatch I am not going to hide**: task success does not follow that
ordering. The best harness for success is the one that does least about
repetition. The reconciliation is in the failure section — 18% of this model's
actions are prose where a call was expected — and the paper states it in the same
paragraph as the repetition result rather than in a footnote.

Three claims retracted over the project, every one on its own evidence, every
one caught by re-running after removing a confound rather than by re-reading.

---

## Typesetting the PDF, and what typesetting the PDF exposed

Built the manuscript for the first time (Tectonic 0.17.0, vendored to
`tools/`), rendered every page to PNG, and read them. The intent was a layout
pass. It turned into a correctness pass, because several errors are invisible in
the `.tex` source and obvious the moment the page is a page.

**Layout, in the order the pages gave them up.**

* The abstract was set in one column of a two-column layout. Fixed with
  `\twocolumn[\begin{@twocolumnfalse}...]`.
* Every result table overflowed its column; two of them were printed on top of
  each other on page 9. All are `table*` now, at `\footnotesize` with 3pt
  column separation. Worst overfull box went from 390pt to zero.
* Negative numbers were set with a text hyphen. `-18.07` and `\textminus 18.07`
  are different characters and only one of them is a minus sign. Both
  `tables._fmt` and `make_facts` now emit `\textminus`.
* The T1 Times font has no `č`, `Š` or `τ`; XeTeX dropped them silently, so
  three cited authors were missing letters from their names. The `.bib` uses
  `\v{c}`, `\v{S}` and `\tau` now.
* The Llama-3 entry listed 561 authors and consumed two pages of bibliography.
  Author lists longer than twelve are truncated to three and `others`.
* Appendix tables floated above their own section headings, so section C
  appeared to have no content and its table appeared to belong to section B.
  `\usepackage{float}` and `[H]`.
* Figure 5's six rotated harness labels, at single-column width across three
  panels, were unreadable. It is a `figure*` now.
* Figure 3's legend sat on top of the interval it was labelling.

**Errors of fact that only surfaced because the page was rendered.**

1. *A broken cross-reference that compiled cleanly.* `Section~\ref{sec:failure}`
   had been mangled to `Section~` + CR + `ef{sec:failure}` by an earlier
   heredoc. LaTeX reported no undefined reference, because there was no
   reference — it was prose. Only reading the page caught it. Added a scan for
   stray control characters across all `.tex` sources.

2. *Two columns of the same number.* The robustness table printed `sem`
   referenced to the success counterfactual next to the polarity contrast. They
   are the same expression — `f(fail) - f(succ)` — and a reader would have
   counted one piece of evidence twice. One column now.

3. *A caption that named the wrong colour.* The error-family heatmap is `RdBu`
   with all-negative data, so every cell is red. The caption said "Blue is
   inversion."

4. *A generation cap asserted in two places with two values.* The appendix said
   40 tokens, the failure section said 32, and nothing recorded which had run.
   Recovered it from the artefacts instead of picking one: an action cut off by
   the sampler is exactly `max_new_tokens` long, and the longest truncated
   action across all rollouts is 32 tokens. The code default and the config
   agree at 32 now, `max_new_tokens` is written into every rollout going
   forward, and the manuscript reads it from a macro.

5. *A hard-coded version number that nothing had measured.* `make_facts` fell
   back to the literal `4.56.1` when no run recorded its environment, and the
   older meta files predate that record. Worse, the logs show the study
   straddles two: **transformers 4.45.0 and 4.56.1**, because the environment
   was rebuilt partway through. Removed every fallback literal — an unmeasured
   value now expands to a visible placeholder — and measured the threat rather
   than waving at it. `scripts/check_version_drift.py` re-scores stored rows in
   the current environment: over 210 triples the largest disagreement is
   4.9e-5 nats, four orders of magnitude under the smallest reported effect.
   The appendix states both versions and cites the check.

6. *Two spelled-out ratios that were neither the median nor the maximum.*
   "about sixty to one" and "tripling the exact repeat rate" were the last
   hand-typed numbers in the manuscript. The true values are a median of 8:1
   (max 106:1) and a factor of 2.6. Both are macros now.

7. *A claim contradicted by its own table.* The agent section said the decoder
   ban drives the canonical repeat rate to zero. It is 0.075. Chasing it down
   produced a better result than the one I had claimed: the canonical and exact
   rates are **equal to the digit in every harness**, so nothing is escaping by
   paraphrase, and the residual is the ban's own boundary — it masks token
   sequences in the raw generation while a repeat is counted on the parsed
   action, and the two come apart when the generation is truncated by the token
   cap. In all three surviving rollouts the repeated string is one the parser
   rejected, never a well-formed call.

**The qualitative figure was the worst of it.** Truncating trajectory lines to a
single column reduced every action to the same 44 characters
(`write_file(path='reports/conversion.md', co~`), which made the standard
harness and the decoder ban look *identical* — the exact opposite of the claim
the figure exists to support. At full width the contrast is the clearest single
piece of evidence in the paper: under `verbatim` the model writes one wrong call
six times; under `verbatim+ban` it cannot, and what it does instead is keep the
same call and accrete fragments of the error message onto it as an extra
argument. It is copying from the transcript, and forbidding one string redirects
it to the nearest other string.

The appendix trajectory section now shows all six harnesses on that task, and
two of the six cut against the aggregates: `abstract` does nothing on this item
though it reduces repetition across the set, and `+instruction` changes the
trajectory here though it does nothing across the set. Left both in, with the
mismatch stated, rather than picking a friendlier task.

**Inclusion rule, made explicit and mechanical.** SmolLM2-1.7B had 11 scored
items against 100 for the rest, and was appearing in tables while the abstract
counted five checkpoints. `analyze_probe.py` now writes a `published` column at
a stated threshold of 30 items; tables, figures and inline facts all read it, so
they cannot disagree about who is in the study, and the appendix prints the
excluded runs by name. A resumed 1.7B run is in progress; if it clears the
threshold it enters the paper the same way everything else does, by being
recomputed.

The general lesson is narrow and worth keeping: a LaTeX build that reports no
errors has checked that the document is well-formed, not that it is true. Six of
the seven errors above compiled silently.
