"""Check generated LaTeX tables for column-count consistency.

A `tabular` whose rows disagree with its column specification is one of the few
LaTeX errors a reference/macro check will not catch, and it is easy to introduce
when a table is emitted by code. Without a TeX installation on the development
machine, this stands in for the compile.

Parsing LaTeX with regexes is a losing game, so this is deliberately
conservative: it uses a brace-balanced scanner for the column specification, and
it *skips* (rather than guesses at) tables whose cells contain constructs that
make naive `\\\\` splitting wrong, reporting them as unchecked.
"""

from __future__ import annotations

import glob
import os
import re
import sys
from typing import List, Optional, Tuple

SKIP_IF_PRESENT = (r"\makecell", r"\multirow", r"\begin{tabular}")


def balanced(text: str, start: int) -> Tuple[str, int]:
    """Return the contents of the brace group beginning at `start`, and its end."""
    assert text[start] == "{"
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
    return text[start + 1 :], len(text)


def spec_columns(spec: str) -> int:
    spec = re.sub(r"@\{[^}]*\}", "", spec)
    spec = re.sub(r"\*\{(\d+)\}\{([^}]*)\}",
                  lambda m: m.group(2) * int(m.group(1)), spec)
    spec = re.sub(r"[pmb]\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", "P", spec)
    return sum(1 for ch in spec if ch in "lcrP")


def row_columns(row: str) -> int:
    n = len(row.split("&"))
    for cell in row.split("&"):
        m = re.search(r"\\multicolumn\{(\d+)\}", cell)
        if m:
            n += int(m.group(1)) - 1
    return n


def check_file(path: str) -> Tuple[int, int, List[str]]:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    problems: List[str] = []
    checked = skipped = 0
    for m in re.finditer(r"\\begin\{tabular\}", text):
        brace = text.index("{", m.end())
        spec, body_start = balanced(text, brace)
        end = text.find(r"\end{tabular}", body_start)
        body = text[body_start:end if end != -1 else len(text)]
        if any(tok in body for tok in SKIP_IF_PRESENT):
            skipped += 1
            continue
        checked += 1
        ncols = spec_columns(spec)
        for raw in body.split(r"\\"):
            row = re.sub(r"\\(toprule|midrule|bottomrule|hline)\b", "", raw).strip()
            row = re.sub(r"\\cmidrule(\([^)]*\))?\{[^}]*\}", "", row).strip()
            if not row or row.startswith("%"):
                continue
            got = row_columns(row)
            if got != ncols:
                problems.append(
                    f"{os.path.basename(path)}: {got} columns but tabular "
                    f"declares {ncols} -- {row[:66]}"
                )
    return checked, skipped, problems


def main() -> int:
    all_problems: List[str] = []
    checked = skipped = 0
    for path in sorted(glob.glob(os.path.join("paper", "tables", "*.tex"))):
        c, s, p = check_file(path)
        checked += c
        skipped += s
        all_problems += p
    print(f"checked {checked} tabular environment(s); "
          f"skipped {skipped} using constructs this checker cannot parse")
    if all_problems:
        print(f"{len(all_problems)} problem(s):")
        for p in all_problems[:20]:
            print("  !", p)
        return 1
    print("all checked tables have consistent column counts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
