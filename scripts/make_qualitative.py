"""Pull representative trajectories out of the rollout logs for the paper.

Selection is by a stated rule rather than by eye: for the qualitative pair we
take the task with the most exact repeats under the standard harness that also
has a rollout under the comparison harness, which is the clearest instance of
the behaviour rather than the most flattering one.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from slmecho import tables  # noqa: E402

# The whole point of these excerpts is the difference between one action and
# the next, and truncating to a single column reduces every line to the same
# `write_file(path='reports/conversion.md', co~`. Both blocks therefore get the
# full page width: the in-text pair as a `figure*`, the appendix in a
# \onecolumn appendix. That holds ~115 characters of \scriptsize typewriter,
# less a 10-character "NN. [ERR] " prefix.
MAXLEN = 98

#: The order the harnesses are discussed in, so the appendix reads the same way
#: as the tables rather than alphabetically.
HARNESS_ORDER = ["verbatim", "verbatim-instr", "drop", "abstract",
                 "verbatim-ban", "abstract-ban"]


def esc(s: str) -> str:
    return s.replace("\\", r"\textbackslash ").replace("_", r"\_").replace(
        "&", r"\&").replace("%", r"\%").replace("#", r"\#").replace(
        "$", r"\$").replace("{", r"\{").replace("}", r"\}")


def load(raw_dir: str) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.jsonl"))):
        parts = os.path.basename(path)[: -len(".jsonl")].split("__")
        if len(parts) < 3:
            continue
        key = f"{parts[0]}|{parts[2]}"
        out[key] = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return out


def render(rollout: dict, title: str) -> str:
    lines = [r"\textbf{" + esc(title) + "}", r"\begin{scriptsize}\begin{verbatim}"]
    lines.append(f"goal: {rollout.get('task_id')}  solved={rollout['solved']}")
    for i, (a, ok) in enumerate(zip(rollout["actions"], rollout["ok"]), 1):
        flag = "ok " if ok else "ERR"
        body = " ".join(a.strip().split())
        if len(body) > MAXLEN:
            body = body[: MAXLEN - 1] + "~"  # ~ marks a truncated action
        lines.append(f"{i}. [{flag}] {body}")
    lines += [r"\end{verbatim}\end{scriptsize}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/raw/agent")
    ap.add_argument("--tables-dir", default="paper/tables")
    args = ap.parse_args()

    data = load(args.raw_dir)
    if not data:
        for name in ("qualitative_ban", "qualitative_appendix"):
            tables.write(os.path.join(args.tables_dir, f"{name}.tex"),
                         "% no rollouts available yet\n")
        print("no rollouts yet; wrote placeholders")
        return

    models = sorted({k.split("|")[0] for k in data})
    blocks: List[str] = []
    pair_block: Optional[str] = None
    for model in models:
        base = data.get(f"{model}|verbatim")
        if not base:
            continue
        worst = max(base, key=lambda r: (r.get("n_exact_repeats", 0), r["n_steps"]))

        # In-text pair: the standard harness against the decoder ban.
        for alt_name in ("verbatim-ban", "abstract-ban", "abstract", "drop"):
            alt = data.get(f"{model}|{alt_name}")
            match = [r for r in (alt or []) if r["task_id"] == worst["task_id"]]
            if not match:
                continue
            if pair_block is None:
                pair_block = "\n\n".join([
                    render(worst, f"{model}, harness=verbatim"),
                    render(match[0], f"{model}, harness={alt_name}"),
                ])
            break

        # Appendix: the same task under every harness we ran, which is the
        # comparison the aggregate tables summarise. Showing only the pair
        # already in the body would make the appendix pure duplication.
        rendered = []
        for h in HARNESS_ORDER:
            rolls = data.get(f"{model}|{h}")
            match = [r for r in (rolls or []) if r["task_id"] == worst["task_id"]]
            if match:
                rendered.append(render(match[0], f"{model}, harness={h}"))
        if len(rendered) > 1:
            blocks.append("\n\n".join(rendered))

    if pair_block:
        pair_block = "\n".join([
            r"\begin{figure*}[t]",
            r"\centering",
            r"\begin{minipage}{\textwidth}",
            pair_block,
            r"\end{minipage}",
            r"\caption{The same task and model under the standard harness and "
            r"under the decoder ban. Selected by rule, not by eye: the task "
            r"with the most exact repeats under the standard harness for which "
            r"a matching rollout exists under the comparison harness. Actions "
            r"are shown as the parser received them.}",
            r"\label{fig:qualitative-ban}",
            r"\end{figure*}",
        ])
    tables.write(os.path.join(args.tables_dir, "qualitative_ban.tex"),
                 pair_block or "% no matching pair found yet\n")
    tables.write(os.path.join(args.tables_dir, "qualitative_appendix.tex"),
                 ("\n\n" + r"\medskip" + "\n\n").join(blocks) or "% none\n")
    print(f"wrote qualitative tables from {len(blocks)} pairs")


if __name__ == "__main__":
    main()
