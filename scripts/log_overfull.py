"""Report overfull boxes in a LaTeX log with the file that produced each one.

`tectonic` prints the warnings but not always the enclosing file, and the log's
own parenthesis nesting is the only record of which \\input we were inside. This
walks that nesting so a 100pt overfull can be traced to one generated table
rather than hunted for by eye.
"""

from __future__ import annotations

import re
import sys

OVERFULL = re.compile(r"Overfull \\([hv])box \(([\d.]+)pt too (wide|high)\)")
OPEN = re.compile(r"\(([^()\s]+)")


def main() -> None:
    path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    stack: list[str] = []
    hits: list[tuple[float, str, str]] = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = OVERFULL.search(line)
        if m and float(m.group(2)) >= threshold:
            hits.append((float(m.group(2)), m.group(1), stack[-1] if stack else "?"))
        for ch in line:
            if ch == "(":
                stack.append("?")
            elif ch == ")" and stack:
                stack.pop()
        for name in OPEN.findall(line):
            if stack:
                stack[-1] = name
    for size, kind, where in sorted(hits, reverse=True):
        print(f"{size:8.1f}pt  {kind}box  {where}")
    print(f"-- {len(hits)} boxes over {threshold}pt")


if __name__ == "__main__":
    main()
