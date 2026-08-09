"""LaTeX table generation.

Tables are written to `paper/tables/*.tex` and `\\input{}` by the manuscript, so
the only way a number reaches the paper is by coming out of `results/raw/`.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


def _fmt(x: float, digits: int = 2) -> str:
    """Format a number, using a real minus sign rather than a hyphen.

    A hyphen is visibly too short for a minus and, in a column of signed
    quantities, reads as a dash. `\\textminus` is the text-mode minus and picks
    up the surrounding font.
    """
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    s = f"{x:.{digits}f}"
    return s.replace("-", r"\textminus ", 1) if s.startswith("-") else s


def ci_cell(mean: float, lo: float, hi: float, digits: int = 2, bold_if_excludes_zero: bool = True) -> str:
    """`mean [lo, hi]`, bolded when the interval excludes zero."""
    if not np.isfinite(mean):
        return "--"
    body = f"{_fmt(mean, digits)}\\,\\ci{{{_fmt(lo, digits)}}}{{{_fmt(hi, digits)}}}"
    if bold_if_excludes_zero and np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0):
        return f"\\textbf{{{body}}}"
    return body


def write(path: str, body: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body.rstrip() + "\n")
    return path


PREAMBLE_HINT = r"""% Requires in the preamble:
% \newcommand{\ci}[2]{{\scriptsize\,[#1,\,#2]}}
"""


def measures_by_model(
    summary: pd.DataFrame,
    measures: Sequence[str],
    headers: Dict[str, str],
    env: str,
    caption: str,
    label: str,
    digits: int = 2,
    order: Optional[Sequence[str]] = None,
) -> str:
    """One row per model, one column per measure, cells as mean [CI]."""
    sub = summary[summary["env"] == env].copy()
    if sub.empty:
        return f"% no rows for env={env}\n"
    sub = sub.sort_values("params_b")
    models = list(dict.fromkeys(sub["model"]))
    if order:
        models = [m for m in order if m in models]

    col_spec = "l r " + " ".join(["c"] * len(measures))
    lines = [
        PREAMBLE_HINT,
        # `table*` spans both columns. Each cell is "mean [lo, hi]" and there are
        # five or six measures; that does not fit a 3.25in column at any font
        # size a reader would accept, and silently overflows into the neighbour.
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        # Wide tables need every point of horizontal slack they can get.
        r"\setlength{\tabcolsep}{3pt}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Model & Params & " + " & ".join(headers[m] for m in measures) + r" \\",
        r"\midrule",
    ]
    for m in models:
        rows = sub[sub["model"] == m]
        label_txt = rows["label"].iloc[0]
        params = rows["params_b"].iloc[0]
        cells = []
        for meas in measures:
            r = rows[rows["measure"] == meas]
            if r.empty:
                cells.append("--")
            else:
                cells.append(
                    ci_cell(
                        float(r["mean"].iloc[0]),
                        float(r["ci_lo"].iloc[0]),
                        float(r["ci_hi"].iloc[0]),
                        digits,
                    )
                )
        lines.append(f"{label_txt} & {params:.2f}B & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def conditions_table(
    summary: pd.DataFrame,
    measures: Sequence[str],
    headers: Dict[str, str],
    env: str,
    caption: str,
    label: str,
    digits: int = 2,
) -> str:
    """Study-2 style table: rows are manipulations, columns are models."""
    sub = summary[(summary["env"] == env) & (summary["measure"].isin(measures))].copy()
    if sub.empty:
        return f"% no rows for env={env}\n"
    sub = sub.sort_values("params_b")
    models = list(dict.fromkeys(sub["model"]))
    # Only a subset of the ladder ran the extended conditions, so most models
    # would contribute a column of dashes that pushes the table off the page.
    # Keep those with at least half the rows populated.
    models = [
        m for m in models
        if sub[(sub["model"] == m)]["measure"].nunique() >= max(2, len(measures) // 2)
    ] or models
    labels = {m: sub[sub["model"] == m]["label"].iloc[0] for m in models}
    col_spec = "l " + " ".join(["c"] * len(models))
    lines = [
        PREAMBLE_HINT,
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        # Wide tables need every point of horizontal slack they can get.
        r"\setlength{\tabcolsep}{3pt}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Harness variant & " + " & ".join(labels[m] for m in models) + r" \\",
        r"\midrule",
    ]
    for meas in measures:
        cells = []
        for m in models:
            r = sub[(sub["model"] == m) & (sub["measure"] == meas)]
            cells.append(
                "--"
                if r.empty
                else ci_cell(
                    float(r["mean"].iloc[0]),
                    float(r["ci_lo"].iloc[0]),
                    float(r["ci_hi"].iloc[0]),
                    digits,
                )
            )
        lines.append(f"{headers.get(meas, meas)} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def dataframe_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    col_format: Optional[str] = None,
    float_digits: int = 2,
) -> str:
    body = df.to_latex(
        index=False,
        escape=False,
        float_format=lambda v: f"{v:.{float_digits}f}",
        column_format=col_format or ("l" * df.shape[1]),
    )
    body = body.replace(r"\toprule", r"\toprule").replace("\n\n", "\n")
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            body.rstrip(),
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\end{table}",
        ]
    )
