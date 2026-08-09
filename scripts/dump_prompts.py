"""Emit the appendix's prompt/condition/operator reference tables from the code.

Prompts printed in a paper drift from the prompts a repository actually uses.
These are generated, so they cannot.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from slmecho import tables  # noqa: E402
from slmecho.envs.coderepair import SYSTEM_PROMPT as CODE_PROMPT  # noqa: E402
from slmecho.envs.toolshed import ToolShedEnv  # noqa: E402
from slmecho.envs.toolshed_tasks import _TOOL_TYPOS, generate_tasks  # noqa: E402
from slmecho.harness import ERROR_GLOSS, NEGATIVE_INSTRUCTION, abstract_failure  # noqa: E402
from slmecho.probe import CONDITION_DOC, ProbeItem, load_items  # noqa: E402


# Courier at \scriptsize fits about this many characters across the one-column
# appendix text block. Anything longer runs off the page instead of wrapping,
# because `verbatim` never breaks a line.
WRAP = 106

# The typewriter font has no glyph for these; XeTeX drops them silently, which
# would misrepresent the prompt. Transliterate instead, and say so in the text.
TRANSLITERATE = {"—": "---", "–": "--", "‘": "'", "’": "'",
                 "“": '"', "”": '"', "→": "->", " ": " "}


def _wrap(line: str) -> list:
    """Hard-wrap one line, marking continuations with a trailing backslash."""
    if len(line) <= WRAP:
        return [line]
    out, rest = [], line
    while len(rest) > WRAP:
        cut = rest.rfind(" ", 0, WRAP - 2)
        if cut <= 0:
            cut = WRAP - 2
        out.append(rest[:cut] + " \\")
        rest = rest[cut:].lstrip()
    out.append(rest)
    return out


def verbatim_block(title: str, body: str, label: str) -> str:
    body = body.replace("\t", "    ")
    for src, dst in TRANSLITERATE.items():
        body = body.replace(src, dst)
    wrapped = [w for line in body.rstrip().split("\n") for w in _wrap(line)]
    return "\n".join(
        [
            f"\\paragraph{{{title}}}",
            r"\begin{scriptsize}\begin{verbatim}",
            *wrapped,
            r"\end{verbatim}\end{scriptsize}",
        ]
    )


def main() -> None:
    os.makedirs("paper/tables", exist_ok=True)

    # ---- conditions -----------------------------------------------------
    lines = [
        # [H], not [h]: these tables are the entire content of their appendix
        # section, and a float that drifts to the previous page arrives above
        # the heading that introduces it.
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\begin{tabular}{l p{0.62\linewidth}}", r"\toprule",
        r"Condition & What the transcript contains \\", r"\midrule",
    ]
    for name, desc in CONDITION_DOC.items():
        # `_` is a maths subscript in LaTeX even inside \texttt.
        lines.append(f"\\texttt{{{name.replace('_', chr(92) + '_')}}} & {desc} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Every probe condition. All conditions for one item share "
              r"the same system prompt, goal and successful prefix, byte for byte.}",
              r"\label{tab:conditions}", r"\end{table}"]
    tables.write("paper/tables/condition_reference.tex", "\n".join(lines))

    # ---- operators ------------------------------------------------------
    items = load_items("data/probe_toolshed.jsonl") if os.path.exists(
        "data/probe_toolshed.jsonl") else []
    examples = {}
    for it in items:
        examples.setdefault(it.operator, (it.gold_action, it.failed_action,
                                          it.error_type))
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        # The two `l` columns hold operator and error names up to ~20 characters
        # each, so the call columns cannot have 0.30 of the width apiece.
        r"\begin{tabular}{l p{0.27\linewidth} p{0.27\linewidth} l}", r"\toprule",
        r"Operator & Reference call & Perturbed call & Error family \\", r"\midrule",
    ]
    def esc(s: str) -> str:
        return s.replace("_", r"\_").replace("&", r"\&")

    LIMIT = 88

    def breakable(s: str) -> str:
        """A tool call is one long unhyphenatable word to LaTeX.

        Without explicit break points it overruns the column rather than
        wrapping. Punctuation inside the call is where a reader would break it
        anyway. Over-long examples are cut, but visibly: a call that just stops
        mid-argument reads as a bug in the table rather than as an excerpt.
        """
        out = esc(s[:LIMIT]) + (r"\,\ldots{}" if len(s) > LIMIT else "")
        for ch in ("(", ",", "=", r"\_"):
            out = out.replace(ch, ch + r"\allowbreak{}")
        return out

    for op in sorted(examples):
        gold, bad, err = examples[op]
        lines.append(
            f"\\texttt{{{esc(op)}}} & \\texttt{{\\scriptsize {breakable(gold)}}} & "
            f"\\texttt{{\\scriptsize {breakable(bad)}}} & "
            f"\\texttt{{\\scriptsize {esc(err)}}} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Perturbation operators used to build failing actions in "
              r"ToolShed, with one instantiated example each.}",
              r"\label{tab:operators}", r"\end{table}"]
    tables.write("paper/tables/operator_reference.tex", "\n".join(lines))

    # ---- prompts --------------------------------------------------------
    env = ToolShedEnv(verbosity="standard", echo_action=False)
    env_demo = ToolShedEnv(verbosity="standard", echo_action=False, demo=True)
    task = generate_tasks(1)[0]
    blocks = [
        verbatim_block("ToolShed system prompt (probe)", env.system_prompt(task), "p1"),
        verbatim_block(
            "Format demonstration appended in the agent study",
            ToolShedEnv.DEMO,
            "p2",
        ),
        verbatim_block("CodeRepair system prompt", CODE_PROMPT, "p3"),
        verbatim_block("Do-not-repeat instruction", NEGATIVE_INSTRUCTION, "p4"),
    ]
    if items:
        it = items[0]
        blocks.append(
            verbatim_block(
                "Failure abstraction (example)",
                abstract_failure(it.failed_step(), 1),
                "p5",
            )
        )
    glosses = "\n".join(f"{k:24s} -> {v}" for k, v in sorted(ERROR_GLOSS.items()))
    blocks.append(verbatim_block("Error-family glosses used by the abstraction",
                                 glosses, "p6"))
    tables.write("paper/tables/prompts.tex", "\n\n".join(blocks))

    print("wrote condition_reference.tex, operator_reference.tex, prompts.tex")


if __name__ == "__main__":
    main()
