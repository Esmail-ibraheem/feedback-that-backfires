# Adversarial review round 1

Written against the manuscript and repository as of the first complete probe
results. Each reviewer is played straight — the goal is to find what is actually
wrong, not to produce criticism that is easy to answer. Every point ends with
what was **done** about it, and points that were not acted on say why.

---

## Reviewer 1 — hostile NeurIPS reviewer

**R1.1 (novelty).** *"You have rediscovered that language models copy from their
context. Induction heads have been understood since 2022, and sentence-level
self-reinforcement since Xu et al. The 'agent' framing is decoration on a known
phenomenon."*

This is the objection to beat, and the paper must beat it in the introduction
rather than in a rebuttal. The response is not that copying is new; it is that
(a) the harness *deliberately inserts* the string, believing this helps, and
(b) a strong, ground-truth corrective signal is present and demonstrably fails
to counteract it. Prior work on repetition studies open-ended generation with no
competing signal. The interesting quantity here is which force wins, and nobody
had measured it.

**Done:** the introduction now states this contrast explicitly; the related-work
section separates "repetition without feedback" from "repetition despite
feedback" and Table 1 makes the distinction a column.

**R1.2 (the effect is mechanical).** *"Of course $\log P$ of a string rises once
the string is in the context. You could have predicted the sign without running
anything. A 22-nat effect is just a long string being copied."*

Partly fair, and the paper is stronger for conceding the predictable part and
isolating the unpredictable part. Three defences, all now in the text:

1. The correct action also rises (`repair` is positive), so the result is not
   "copying lifts everything". What matters is the margin, and the margin moves
   *against* the correct action.
2. The normalised repeat probability is computed over a fixed candidate set, so
   a uniform lift cancels.
3. The greedy readout concerns the argmax, which is immune to uniform lifts by
   construction.

**Done:** added the placebo condition `fail_other`, where a *different* wrong
call is recorded as having failed. If "a failure occurred" were the operative
variable, this would look like the failure condition; the surface-form account
predicts it looks like the baseline. This is the cleanest available test and it
did not exist in the first design.

**R1.3 (scale).** *"1.7B is not an agent. Nothing here tells me about the models
people deploy."*

Correct and not fixable with the available hardware (CPU only, float32, 17 GB).
The honest response is to characterise the regime we can measure, report the
monotone trend, and refuse to extrapolate a crossing point.

**Done:** the results section now explicitly declines to name a zero crossing,
and the limitations section says the ladder stops where it does because of
memory. The probe is cheap enough (a few hundred forward passes per model) that
we note anyone with a GPU can settle the question.

**R1.4 (single environment).** *"ToolShed is yours. You wrote the tasks, the
errors and the perturbations."*

**Done:** CodeRepair exists precisely for this — MBPP problems, mutations of
reference solutions, and observations produced by the Python interpreter. The
setup section states why a purpose-built environment was necessary at all
(counterfactual observations) rather than treating it as a convenience.

---

## Reviewer 2 — skeptical ACL reviewer

**R2.1 (metric).** *"Summed log-probabilities over whole strings are
length-confounded and hard to interpret. 22 nats of what?"*

**Done:** three responses now in the paper. The comparison is between two
contexts scoring an *identical* string, so length cancels within a contrast;
a per-token version is reported in the robustness table; and the two headline
readouts (normalised repeat probability, exact greedy repeat) are dimensionless.

**R2.2 (the neutral observation is not neutral).** *"'Call recorded.' plausibly
reads as mild success. If so your `copy` term has absorbed part of the semantic
term and the decomposition is not clean."*

This is the sharpest methodological point raised in this round.

**Done:** the decomposition is now also reported against the counterfactual
*success* observation as the reference (`copy_succ`, `sem_succ`). The two
references bracket any reasonable notion of neutrality; if the conclusion holds
under both, the exact choice does not matter. A length-matched neutral
(`neut_pad`) additionally separates content from position.

**R2.3 (clusters).** *"Cluster bootstrap over 15–30 tasks is not much better
than no bootstrap at all."*

**Done:** the number of clusters is reported in every summary row, item counts
per model are stated, and an assumption-light item-level sign statistic
(fraction of items with $G<0$) is added so the headline does not rest on the
bootstrap alone. Item subsampling caps items per task specifically to raise the
cluster count.

**R2.4 (the fix is trivial).** *"Banning strings you have already seen fail is an
obvious engineering trick. Where is the science?"*

The science is not the trick; it is the argument that predicts the trick will
work and that predicts the more natural remedy (an instruction) will not. A fix
derived from a measurement is worth more than the same fix proposed by intuition,
and the paper is careful to report where the ban fails.

**Done:** the canonical repeat rate is reported alongside the exact rate,
precisely to show how much of the ban's apparent success is displacement into
paraphrase. The failure-analysis section leads with this.

**R2.5 (missing baseline).** *"Prior work on context contamination recommends
clearing the context before retrying. You do not compare against it."*

Correct at the time it was raised, and a serious omission.

**Done:** `drop` (clean restart) and `drop+ban` are now evaluated harnesses in
the end-to-end study, and the probe already contains the corresponding
comparison for free: the `pre` condition *is* the clean-restart context, so $G$
is exactly "contaminated versus clean restart" and $G_{\text{abstract}}$ is
"abstracted versus clean restart".

---

## Reviewer 3 — expert in agent harnesses

**R3.1 (positioning against CCRM).** *"Yang (2026) already showed that retrying
in a contaminated context has a 7.1x elevated error rate, on SWE-bench Verified,
with a formal model. What is left?"*

The mechanism and the fix. CCRM is a probabilistic model fitted to outcome data;
it contains no token-level analysis and does not distinguish which part of the
context contaminates. Its prescription — clear the context — is exactly the
intervention our decomposition says is unnecessarily lossy, because it discards
the diagnosis along with the string.

**Done:** CCRM has its own paragraph in related work and a row in Table 1, and
`drop` is an evaluated baseline (see R2.5).

**R3.2 (the abstraction is hand-written).** *"You wrote the glosses. Maybe the
abstraction works because your descriptions are good, not because the string is
gone. That would not generalise."*

**Done:** added `abstract_min`, which supplies only `[attempt 1 failed]` — no
call and no diagnosis. If it matches the full abstraction, the benefit is the
absence of the string, which is what the decomposition predicts and what
generalises.

**R3.3 (harness realism).** *"Real harnesses do not append a bare error string;
they use tool roles, structured error objects, and often truncate."*

Partly addressed, partly a limitation. We use a `user` role for observations
uniformly because several small models ship chat templates without a tool role,
and varying the role per model would confound the comparison. Verbosity and
quoting are manipulated variables, which covers the largest realistic axis of
variation.

**Not done:** structured/JSON error objects are untested. Noted as a limitation.

**R3.4 (floor effects).** *"If your smallest models solve nothing, your success
numbers are noise."*

Real risk: with the original task set, SmolLM2-135M solved zero tasks under
every harness.

**Done:** the end-to-end task mix now interleaves single-call tasks with the
multi-call ones, and a one-line format demonstration is appended to the system
prompt identically in every condition. Both changes are stated in the setup
section, with the reason. The probe is unaffected, since it is teacher-forced.

**R3.5 (seeds).** *"One greedy trajectory per (model, harness, task) is one
sample."*

Accepted as a limitation. Greedy removes decoding noise from a harness contrast,
which is the right default for this comparison, but it means intervals over
rollouts come from resampling tasks rather than seeds.

**Not done (compute):** a sampled multi-seed run is the obvious next addition and
is supported by the code (`--temperature`, `--seeds`).

---

## Actions taken, in order of importance

1. Placebo condition `fail_other` — a different call fails (R1.2).
2. `drop` / clean restart added as an evaluated baseline (R2.5, R3.1).
3. `abstract_min` — abstraction with no diagnosis (R3.2).
4. Decomposition re-referenced against the success counterfactual (R2.2).
5. Per-token gain, item-level sign statistic, cluster counts reported (R2.1, R2.3).
6. Easy tasks + format demonstration to escape the floor (R3.4).
7. CCRM given its own related-work paragraph and table row (R3.1).
8. Refusal to extrapolate a scale crossing point (R1.3).

---

# Round 2 — self-review after the first complete results

Re-read of the manuscript against the data actually in `results/`, looking
specifically for claims that had drifted from the evidence.

## What was wrong, and what changed

**A statistic that was not the right statistic.** The manipulation table
compared `G` estimated on 100 items against each variant's `G` estimated on the
39–65 items the extended pass had reached. That is partly a comparison of item
sets. Everything in that section is now a **paired within-item delta** against
the standard harness. The change was not cosmetic: "quoting the call back makes
it worse" looked like a null under the unpaired comparison (−23.95 vs −23.92)
and is +1.28 [+1.09, +1.48] when paired. The paragraph had been written before
the check, and was right by luck.

**A claim that the data contradicted.** The introduction said the
natural-language prohibition "does essentially nothing". In free-running
rollouts it more than doubles the exact repeat rate (+29 points, interval
excluding zero) while moving success by an amount whose interval touches zero.
The text now says what was measured, including the mechanism visible in the
other columns: the instruction makes the agent persist longer rather than
repeat less.

**An objection that was answered better than expected.** `abstract_min` — the
bare marker `[attempt 1 failed]`, no call and no diagnosis — matches the full
abstraction (−23.35 vs −21.68). So the intervention does not depend on the
quality of glosses we wrote, which was R3.2's concern and would have been the
main threat to it generalising.

**A control that turned out to be conservative rather than clean.** The placebo
recovers −14.71 of the effect rather than all of it, because the distractor is
another perturbation of the *same* reference call and therefore shares most of
its tokens. Reported as a lower bound on the string-specific component rather
than as a clean zero, with the tighter design named as the refinement.

## What is still open

* **Scale.** Nothing above 1.7B. The fit's zero crossing is an extrapolation
  several times beyond the largest model tested and the paper declines to claim
  it.
* **One trajectory per (model, harness, task).** Greedy decoding; intervals over
  rollouts come from resampling tasks, not seeds.
* **Structured error objects.** Real harnesses sometimes pass JSON error objects
  rather than strings. Untested.
* **A cleaner placebo** using an unrelated call.
