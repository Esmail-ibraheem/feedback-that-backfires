"""Static checks on the manuscript that catch what a missing TeX install cannot.

No LaTeX toolchain was available on the machine this was written on, so the
manuscript could not be compiled during development. These checks stand in for
the errors a compile would have surfaced: macros used but never defined, files
`\\input` but absent, `\\ref`s with no matching `\\label`, figures that do not
exist, and citation keys missing from the bibliography.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Dict, List, Set

# Control sequences supplied by LaTeX itself or by the packages we load.
BUILTIN = {
    "Require", "State", "If", "ElsIf", "Else", "EndIf", "For", "ForAll", "EndFor",
    "While", "EndWhile", "Return", "Comment", "Procedure", "EndProcedure",
    "Delta", "Vert", "Cref", "Sigma", "Omega", "Lambda", "Gamma", "Theta", "Phi",
    "Pi", "Psi", "Xi", "Upsilon", "Big", "Bigg", "Large", "LARGE", "Huge",
    "Leftarrow", "Rightarrow", "Leftrightarrow", "Longrightarrow",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", default="paper")
    args = ap.parse_args()
    P = args.paper_dir
    problems: List[str] = []

    tex_files = [os.path.join(P, "main.tex")] + sorted(
        glob.glob(os.path.join(P, "sections", "*.tex"))
    )
    body = ""
    for f in tex_files:
        with open(f, encoding="utf-8") as fh:
            body += fh.read() + "\n"

    # Only files the document actually pulls in may contribute definitions.
    # Scanning every file in tables/ regardless of whether it is \input would
    # hide exactly the bug this check exists to catch: a definitions file that
    # is present on disk but never included. (It did hide one.)
    included = set()
    for rel in re.findall(r"\\input\{([^}]+)\}", body):
        path = os.path.join(P, rel if rel.endswith(".tex") else rel + ".tex")
        included.add(os.path.normpath(path))
    # One level of nesting is enough for this manuscript.
    for path in list(included):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for rel in re.findall(r"\\input\{([^}]+)\}", fh.read()):
                    nested = os.path.join(P, rel if rel.endswith(".tex") else rel + ".tex")
                    included.add(os.path.normpath(nested))

    table_body = ""
    orphans = []
    for f in sorted(glob.glob(os.path.join(P, "tables", "*.tex"))):
        if os.path.normpath(f) in included:
            with open(f, encoding="utf-8") as fh:
                table_body += fh.read() + "\n"
        else:
            orphans.append(os.path.basename(f))
    everything = body + table_body

    # --- 1. inputs exist -------------------------------------------------
    for rel in sorted(set(re.findall(r"\\input\{([^}]+)\}", everything))):
        path = os.path.join(P, rel if rel.endswith(".tex") else rel + ".tex")
        if not os.path.exists(path):
            problems.append(f"missing \\input target: {rel} ({path})")

    # --- 2. macros defined ----------------------------------------------
    defined: Set[str] = set(
        re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", everything)
    )
    used = set(re.findall(r"\\([A-Z][A-Za-z]+)", body))
    unknown = sorted(u for u in used - defined - BUILTIN)
    for u in unknown:
        problems.append(f"macro used but not defined: \\{u}")

    # --- 3. refs resolve --------------------------------------------------
    labels = set(re.findall(r"\\label\{([^}]+)\}", everything))
    for r in sorted(set(re.findall(r"\\ref\{([^}]+)\}", everything))):
        if r not in labels:
            problems.append(f"\\ref to undefined label: {r}")

    # --- 4. figures exist -------------------------------------------------
    for g in sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
                                   everything))):
        stem = os.path.join(P, g)
        if not (os.path.exists(stem + ".pdf") or os.path.exists(stem + ".png")
                or os.path.exists(stem)):
            problems.append(f"missing figure: {g}")

    # --- 5. citations exist ----------------------------------------------
    bib_path = os.path.join(P, "references.bib")
    keys: Set[str] = set()
    if os.path.exists(bib_path):
        with open(bib_path, encoding="utf-8") as fh:
            keys = set(re.findall(r"@\w+\{([^,]+),", fh.read()))
    cited: Set[str] = set()
    for m in re.findall(r"\\cite[a-z]*\{([^}]+)\}", everything):
        cited |= {k.strip() for k in m.split(",")}
    for c in sorted(cited - keys):
        problems.append(f"citation key not in references.bib: {c}")
    unused = sorted(keys - cited)

    # --- 6. unresolved placeholders --------------------------------------
    todos = re.findall(r"\\todo\{([^}]*)\}", body)
    pending = re.findall(r"RESULT PENDING EXPERIMENT", everything)

    if orphans:
        print(f"  note: {len(orphans)} table file(s) never \\input by the "
              f"document: {', '.join(orphans)}")
    print(f"checked {len(tex_files)} section files, "
          f"{len(glob.glob(os.path.join(P, 'tables', '*.tex')))} generated tables "
          f"({len([f for f in glob.glob(os.path.join(P, 'tables', '*.tex')) if os.path.normpath(f) in included])} included)")
    print(f"  macros defined: {len(defined)}   labels: {len(labels)}   "
          f"bib keys: {len(keys)}   cited: {len(cited)}")
    if unused:
        print(f"  note: {len(unused)} verified references not yet cited: "
              f"{', '.join(unused[:8])}{' ...' if len(unused) > 8 else ''}")
    if todos:
        print(f"  {len(todos)} \\todo markers in the prose:")
        for t in todos:
            print(f"      - {t[:90]}")
    if pending:
        print(f"  {len(pending)} 'RESULT PENDING EXPERIMENT' placeholders "
              f"(expected while experiments are still running)")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print("  !", p)
        return 1
    print("\nno structural problems found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
