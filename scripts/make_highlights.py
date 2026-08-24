"""Emit the Elsevier Highlights file from the same facts the manuscript uses.

Elsevier wants highlights in a separate editable file with "Highlights" in the
name: three to five bullets, each at most 85 characters including spaces. The
numbers are pulled from paper/tables/facts.tex rather than retyped, so a re-run
of the experiments cannot leave the highlights quoting a stale value, and the
length limit is checked rather than counted by eye.
"""

from __future__ import annotations

import re
import sys

LIMIT = 85
FACTS = "paper/tables/facts.tex"
OUT = "paper/Highlights.txt"

BULLETS = [
    "Small language model agents repeat a tool call they have just watched fail",
    "Execution feedback lowers the correct call's log-odds in all {NumModels} models tested",
    "A decomposition attributes {CopyShare} of the effect to copying, not to inference",
    "Removing the failed call from the transcript removes {AbstractRemoved} of the penalty",
    "Blocking exact repeats cuts agent repetition from {AgentBaseRepeat} to {AgentVerbBanRepeat}",
]

MACRO = re.compile(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$")


def detex(v: str) -> str:
    """Highlights are plain text: no macros, no maths, a real minus sign."""
    v = v.replace(r"\textminus ", "-").replace(r"\textminus", "-")
    v = v.replace(r"\%", "%").replace(r"\,", "").replace("~", " ")
    v = re.sub(r"\\ci\{([^}]*)\}\{([^}]*)\}", r"[\1, \2]", v)
    v = re.sub(r"\$([^$]*)\$", r"\1", v)
    return v.strip()


def load_facts(path: str) -> dict:
    facts = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = MACRO.match(line.strip())
            if m:
                facts[m.group(1)] = detex(m.group(2))
    return facts


def main() -> None:
    facts = load_facts(FACTS)
    lines = [b.format(**facts) for b in BULLETS]
    over = [t for t in lines if len(t) > LIMIT]
    if over:
        for t in over:
            print(f"TOO LONG ({len(t)} > {LIMIT}): {t}", file=sys.stderr)
        raise SystemExit(1)
    assert 3 <= len(lines) <= 5, len(lines)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("Highlights\n\n")
        fh.write("\n".join("- " + t for t in lines) + "\n")
    for t in lines:
        print(f"{len(t):3d}  {t}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
