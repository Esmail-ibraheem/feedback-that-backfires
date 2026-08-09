"""Rewrite em-dash constructions in the manuscript as explicit substitutions.

An em-dash pair is almost always a parenthetical, and in English prose it has
several natural replacements (a comma pair, a colon, parentheses, or two
sentences) that are not interchangeable. So this is a table of hand-written
rewrites rather than a search-and-replace: each entry names the old text and
what it becomes.

Matching collapses whitespace, so a rewrapped paragraph does not silently
break an entry; a miss is a hard error rather than a skipped edit.
"""

from __future__ import annotations

import re
import sys
from typing import List, Sequence, Tuple


def apply(path: str, pairs: Sequence[Tuple[str, str]]) -> int:
    s = open(path, encoding="utf-8").read()
    missing: List[str] = []
    for old, new in pairs:
        pat = re.compile(r"\s+".join(re.escape(w) for w in old.split()))
        s, n = pat.subn(lambda m, _n=new: _n, s, count=1)
        if n == 0:
            missing.append(old[:70])
    if missing:
        for m in missing:
            print(f"  !! NOT FOUND in {path}: {m}")
        sys.exit(1)
    open(path, "w", encoding="utf-8", newline="\n").write(s)
    left = s.count("---")
    print(f"  {path}: {left} em-dashes left")
    return left
