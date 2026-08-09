"""Repair `\\ref` control sequences mangled into carriage returns.

Editing LaTeX from Python here-documents is a trap: `"\\ref"` inside a
double-quoted heredoc string becomes CR + "ef". This scans the manuscript for
that signature (and the same accident for `\\r`-initial control words) and
repairs it, so the damage cannot survive to a compile.
"""

from __future__ import annotations

import glob
import os
import re
import sys

CR = chr(13)
# Control words that begin with `r` and are plausible in this manuscript.
WORDS = ["ref", "right", "rule", "raggedright", "renewcommand", "rowcolor"]


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "paper"
    files = glob.glob(os.path.join(root, "**", "*.tex"), recursive=True)
    fixed = 0
    for path in files:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            text = fh.read()
        original = text
        for w in WORDS:
            text = text.replace(CR + w + "{", "\\" + w + "{")
            text = text.replace(CR + w + " ", "\\" + w + " ")
            text = text.replace(CR + "\n" + w + "{", "\\" + w + "{")
        # A bare `~ef{` or `~ight{` is the same accident with the CR stripped by
        # a later normalisation pass.
        for w in WORDS:
            text = re.sub(r"~\s*" + w[1:] + r"\{", "~\\\\" + w + "{", text)
        if text != original:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
            fixed += 1
            print(f"repaired {path}")
    print(f"{fixed} file(s) repaired" if fixed else "nothing to repair")
    return 0


if __name__ == "__main__":
    sys.exit(main())
