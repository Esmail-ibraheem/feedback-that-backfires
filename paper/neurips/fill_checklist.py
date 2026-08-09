"""Fill the NeurIPS paper checklist from a table of answers.

The checklist is a required part of the submission and is read by reviewers, so
it is generated from one place rather than hand-edited into the manuscript. The
skeleton is the official template's own text, unmodified; only the Answer and
Justification lines are substituted, in question order.
"""

from __future__ import annotations

import re

# (answer macro, justification). Order matches the official checklist.
ANSWERS = [
    ("Yes",
     "The abstract and introduction state the measured effect, its "
     "decomposition, and which interventions do and do not act on it, with the "
     "model range and environments named. Claims about scale are confined to "
     "the range tested."),
    ("Yes",
     "Section~\\ref{sec:ws-limits} states the parameter ceiling, the "
     "single-model rollout study, the synthetic construction of the failing "
     "actions, and the instability of the scaling extrapolation."),
    ("NA",
     "The paper contains no theoretical results; the decomposition in "
     "Eq.~\\ref{eq:ws-decomp} is an identity that follows from the definitions "
     "by construction."),
    ("Yes",
     "The environments, perturbation operators, model checkpoints, conditions, "
     "seeds and statistical procedure are specified, and the released code "
     "reproduces every number end to end on a CPU."),
    ("Yes",
     "Code, item sets, raw per-score records and run metadata are released, "
     "with a single documented command sequence. The link is withheld here for "
     "anonymity and will be provided on acceptance."),
    ("Yes",
     "No training is performed. Scoring is teacher-forced and deterministic; "
     "rollouts use greedy decoding with a stated token cap and step budget, all "
     "recorded per run."),
    ("Yes",
     "Every interval is a 95\\% cluster bootstrap over tasks with 10{,}000 "
     "resamples, clustered because items from one task share a prompt. The "
     "headline family is Holm-corrected across models."),
    ("Yes",
     "Section~\\ref{sec:ws-setup} names the CPU, the memory limit that sets the "
     "parameter ceiling, and float32 as the numeric format; per-run wall-clock "
     "and token counts are in the released metadata."),
    ("Yes",
     "The work uses public instruction-tuned checkpoints and a public program "
     "repair dataset, trains nothing, and involves no human subjects or "
     "personal data."),
    ("Yes",
     "The paper notes that suppressing repeated failed actions makes an agent "
     "more persistent, which is not always desirable in a side-effecting "
     "environment, and that cheaper local agents lower the cost of running "
     "systems without oversight."),
    ("NA",
     "No models or datasets with high misuse potential are released. The "
     "released artefacts are analysis code, constructed probe items and "
     "recorded log-probabilities."),
    ("Yes",
     "The model checkpoints and MBPP are cited, and the licence terms of the "
     "released code and of MBPP are stated in the repository."),
    ("Yes",
     "The constructed environment, the probe item pools and the analysis code "
     "are documented in the repository, including the operator definitions and "
     "the exact prompts, which are emitted from the code rather than "
     "transcribed."),
    ("NA", "The paper involves no crowdsourcing and no human subjects."),
    ("NA", "The paper involves no human subjects, so no IRB review applies."),
    ("NA",
     "Language models are the object of study rather than a component of the "
     "method. No LLM was used as part of the core methodology."),
]


def main() -> None:
    text = open("checklist_skeleton.tex", encoding="utf-8").read()
    answers = iter(ANSWERS)

    def sub_answer(_m):
        macro, _just = next(answers)
        return f"\\item[] Answer: \\answer{macro}{{}}"

    text = re.sub(r"\\item\[\] Answer: \\answerTODO\{\}[^\n]*", sub_answer, text)

    just = iter(ANSWERS)

    def sub_just(_m):
        _macro, j = next(just)
        return f"\\item[] Justification: {j}"

    text = re.sub(r"\\item\[\] Justification: \\justificationTODO\{\}", sub_just, text)

    assert "answerTODO" not in text and "justificationTODO" not in text, "unfilled"
    # The official template contains curly quotes, and T1 Times has no glyph
    # for them, so XeTeX drops them silently. Use the TeX ligatures instead.
    for bad, good in (("“", "``"), ("”", "''"),
                      ("‘", "`"), ("’", "'"),
                      ("–", "--"), ("—", "---")):
        text = text.replace(bad, good)
    open("checklist.tex", "w", encoding="utf-8", newline="\n").write(text)
    print(f"wrote checklist.tex ({len(ANSWERS)} answers)")


if __name__ == "__main__":
    main()
