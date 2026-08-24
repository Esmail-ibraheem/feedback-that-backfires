"""Assemble the arXiv source package for the two-column build.

arXiv compiles the source rather than accepting a PDF, so the package has to
contain every file main.tex reaches and nothing else: a stray second top-level
.tex would be compiled too, and the other three builds in paper/ are exactly
that. The file list is therefore walked from main.tex rather than typed out, so
it cannot fall behind a section or a figure that gets added later.

Two arXiv-specific things happen to the copy of main.tex:

  * a guarded \\pdfoutput=1 goes on line 1, which is how arXiv is told to run
    pdflatex rather than latex+dvips. The guard tests for \\pdftexversion, not
    for \\pdfoutput: XeTeX defines \\pdfoutput too, and setting it there makes
    hyperref load hpdftex.def and the build die. Guarding on the engine keeps
    the shipped file compiling under both, which matters because a package
    nobody can rebuild is a package nobody can check.
  * main.bbl is shipped, because arXiv does not run BibTeX. references.bib goes
    in as well. arXiv ignores it, but it costs 30 KB and makes the downloaded
    source rebuildable from scratch, which for this paper is the point.

Usage:  python scripts/make_arxiv.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import zipfile

PAPER = "paper"
MAIN = "main.tex"
OUT_DIR = os.path.join("dist", "arxiv")
ZIP_PATH = os.path.join("dist", "arxiv-feedback-that-backfires.zip")
TECTONIC = os.path.join("tools", "tectonic.exe")

PDFOUTPUT = r"\ifdefined\pdftexversion\pdfoutput=1\fi  % arXiv: build with pdflatex"

INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")
GRAPHIC_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
COMMENT_RE = re.compile(r"(?<!\\)%.*")

GRAPHIC_EXT = (".pdf", ".png", ".jpg", ".jpeg", ".eps")


def strip_comments(text: str) -> str:
    return "\n".join(COMMENT_RE.sub("", line) for line in text.splitlines())


def walk(root: str) -> tuple:
    """Every .tex and every image main.tex reaches, transitively."""
    tex, images, queue = [], [], [root]
    while queue:
        rel = queue.pop(0)
        if rel in tex:
            continue
        path = os.path.join(PAPER, rel)
        if not os.path.exists(path):
            raise SystemExit(f"missing input: {path}")
        tex.append(rel)
        body = strip_comments(open(path, encoding="utf-8").read())
        for name in INPUT_RE.findall(body):
            queue.append(name if name.endswith(".tex") else name + ".tex")
        for name in GRAPHIC_RE.findall(body):
            if name.lower().endswith(GRAPHIC_EXT):
                images.append(name)
                continue
            # \includegraphics{figures/fig2_scaling}: pdflatex resolves the
            # extension itself, and prefers .pdf. Ship only that one, not the
            # .png beside it, which would double the package for nothing.
            for ext in GRAPHIC_EXT:
                if os.path.exists(os.path.join(PAPER, name + ext)):
                    images.append(name + ext)
                    break
            else:
                raise SystemExit(f"no file found for figure: {name}")
    return tex, sorted(set(images))


def copy(rel: str) -> None:
    src, dst = os.path.join(PAPER, rel), os.path.join(OUT_DIR, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    tex, images = walk(MAIN)
    extra = ["references.bib", "main.bbl"]
    for rel in extra:
        if not os.path.exists(os.path.join(PAPER, rel)):
            raise SystemExit(
                f"missing {rel}: build main.tex once with --keep-intermediates"
            )

    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)
    for rel in tex + images + extra:
        copy(rel)

    main_path = os.path.join(OUT_DIR, MAIN)
    body = open(main_path, encoding="utf-8").read()
    with open(main_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(PDFOUTPUT + "\n" + body)

    # Rebuild the copy in place. This is the only check that matters: if the
    # package is missing a table or a figure, this fails.
    if os.path.exists(TECTONIC):
        r = subprocess.run(
            [os.path.abspath(TECTONIC), "-X", "compile", MAIN, "--keep-intermediates"],
            cwd=OUT_DIR, capture_output=True, text=True,
        )
        if r.returncode != 0:
            sys.stderr.write(r.stderr[-3000:])
            raise SystemExit("the packaged source does not build")
        print("verified: the packaged source builds on its own")
        for junk in os.listdir(OUT_DIR):
            if os.path.splitext(junk)[1] in (".aux", ".log", ".out", ".blg", ".pdf"):
                os.remove(os.path.join(OUT_DIR, junk))

    os.makedirs(os.path.dirname(ZIP_PATH), exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, _, names in os.walk(OUT_DIR):
            for name in sorted(names):
                full = os.path.join(dirpath, name)
                z.write(full, os.path.relpath(full, OUT_DIR).replace(os.sep, "/"))

    with zipfile.ZipFile(ZIP_PATH) as z:
        names = z.namelist()
    print(f"{len(names)} files, {os.path.getsize(ZIP_PATH) / 1024:.0f} KB -> {ZIP_PATH}")
    for n in names:
        print("  ", n)


if __name__ == "__main__":
    main()
