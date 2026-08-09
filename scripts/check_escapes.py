"""Find unescaped LaTeX special characters in generated tables.

`_`, `&`, `%`, `#` and `$` are all active in LaTeX, and a generated table that
embeds a Python identifier or a percentage will fail to compile — or, worse,
compile into something subtly wrong. A structural checker that only looks at
macros and references does not see this; a compile does, which is why this
exists alongside `check_latex.py` for environments without a TeX installation.
"""

from __future__ import annotations

import glob
import os
import re
import sys

# Characters that must be escaped in ordinary text, with the contexts where they
# are legitimately bare.
VERBATIM_ENVS = (r"\begin{verbatim}", r"\end{verbatim}")


def strip_verbatim(text: str) -> str:
    out, keep = [], True
    for line in text.splitlines():
        if VERBATIM_ENVS[0] in line:
            keep = False
        if keep:
            out.append(line)
        if VERBATIM_ENVS[1] in line:
            keep = True
    return "\n".join(out)


def problems_in(text: str) -> list[str]:
    text = strip_verbatim(text)
    # Remove comments and maths, where these characters are legal.
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(r"\$[^$]*\$", "", text)
    found = []
    for m in re.finditer(r"(?<!\\)_", text):
        line = text[: m.start()].count("\n") + 1
        found.append(f"line {line}: unescaped underscore")
    for m in re.finditer(r"(?<!\\)#", text):
        line = text[: m.start()].count("\n") + 1
        found.append(f"line {line}: unescaped hash")
    return found


def main() -> int:
    total = 0
    for path in sorted(glob.glob(os.path.join("paper", "tables", "*.tex"))):
        with open(path, encoding="utf-8") as fh:
            probs = problems_in(fh.read())
        if probs:
            total += len(probs)
            print(f"{os.path.basename(path)}:")
            for p in probs[:6]:
                print("   ", p)
            if len(probs) > 6:
                print(f"    ... and {len(probs) - 6} more")
    if total:
        print(f"\n{total} unescaped special character(s) found")
        return 1
    print("no unescaped special characters in generated tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
