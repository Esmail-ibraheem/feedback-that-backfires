"""Turn raw probe scores into the quantities the paper reports.

Everything downstream of this module (tables, figures, statistics) reads the
frames produced here, and this module reads only `results/raw/`. No number in
the manuscript is transcribed by hand.

Sign conventions, fixed once and used everywhere:

* All log-probabilities are **summed over the tokens of the candidate string**,
  so a difference between two conditions is the log-odds change of emitting
  that exact string.
* ``G`` (corrective gain) ``= logP(a_fail | pre) - logP(a_fail | fail)``.
  Positive means the harness did what it is supposed to do: after seeing the
  call fail, the model is less likely to write it again. Negative means the
  opposite, which we call *feedback inversion*.
* ``copy`` and ``sem`` decompose ``-G``: ``-G = copy + sem``, where
  ``copy = logP(a_fail | neut) - logP(a_fail | pre)`` is what merely *placing
  the call in the context* does, and ``sem = logP(a_fail | fail) -
  logP(a_fail | neut)`` is what *marking it as failed* adds on top.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .stats import Estimate, cluster_bootstrap

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_raw(paths: Sequence[str]) -> pd.DataFrame:
    rows: List[dict] = []
    for p in paths:
        if not os.path.exists(p):
            continue
        env = os.path.basename(p).split("__")[1].split(".")[0]
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                r["env"] = env
                rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # A resumed run can append a duplicate row; the last write wins.
    df = df.drop_duplicates(
        subset=["model", "env", "item_id", "condition", "candidate"], keep="last"
    )
    return df


def item_metadata(items_paths: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for env, path in items_paths.items():
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                it = json.loads(line)
                rows.append(
                    {
                        "env": env,
                        "item_id": it["item_id"],
                        "task_id": it["task_id"],
                        "template": it["template"],
                        "operator": it["operator"],
                        "error_type": it["error_type"],
                        "step_index": it["step_index"],
                        "n_prefix_steps": len(it["prefix_actions"]),
                        "traceback_quotes_source": (it.get("meta") or {}).get(
                            "traceback_quotes_source", False
                        ),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Wide frame + derived measures
# --------------------------------------------------------------------------


def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, env, item); columns are `<condition>__<candidate>`."""
    lp = df.pivot_table(
        index=["model", "env", "item_id"],
        columns=["condition", "candidate"],
        values="logprob",
        aggfunc="first",
    )
    lp.columns = [f"lp__{c}__{k}" for c, k in lp.columns]
    gr = df.pivot_table(
        index=["model", "env", "item_id"],
        columns=["condition", "candidate"],
        values="is_greedy",
        aggfunc="first",
    )
    gr.columns = [f"greedy__{c}__{k}" for c, k in gr.columns]
    nt = df.pivot_table(
        index=["model", "env", "item_id"],
        columns=["condition", "candidate"],
        values="n_tokens",
        aggfunc="first",
    )
    nt.columns = [f"ntok__{c}__{k}" for c, k in nt.columns]
    ctx = df.pivot_table(
        index=["model", "env", "item_id"],
        columns=["condition"],
        values="ctx_tokens",
        aggfunc="first",
    )
    ctx.columns = [f"ctx__{c}" for c in ctx.columns]
    return pd.concat([lp, gr, nt, ctx], axis=1).reset_index()


def _col(wide: pd.DataFrame, cond: str, cand: str, prefix: str = "lp") -> pd.Series:
    name = f"{prefix}__{cond}__{cand}"
    if name not in wide.columns:
        return pd.Series(np.nan, index=wide.index)
    return wide[name]


def derive(wide: pd.DataFrame) -> pd.DataFrame:
    """Add the paper's derived measures to a wide frame."""
    out = wide.copy()
    f = lambda c: _col(out, c, "failed")  # noqa: E731
    g = lambda c: _col(out, c, "gold")  # noqa: E731

    out["G"] = f("pre") - f("fail")
    out["G_pad"] = f("pre") - f("fail")  # same numerator; pad enters `copy_pad`
    out["copy"] = f("neut") - f("pre")
    out["sem"] = f("fail") - f("neut")
    out["copy_pad"] = _col(out, "neut_pad", "failed") - f("pre")
    out["sem_pad"] = f("fail") - _col(out, "neut_pad", "failed")
    out["polarity"] = f("fail") - f("succ")
    out["G_abstract"] = f("pre") - f("abstract")
    # Paired within-item contrast against the standard harness. This is the
    # right statistic for the manipulation study: comparing two `G` columns
    # estimated on different item counts would confound the manipulation with
    # which items happened to be scored.
    out["delta_abstract"] = f("abstract") - f("fail")
    out["repair"] = g("fail") - g("pre")
    out["repair_abstract"] = g("abstract") - g("pre")
    out["margin_pre"] = g("pre") - f("pre")
    out["margin_fail"] = g("fail") - f("fail")
    out["margin_abstract"] = g("abstract") - f("abstract")
    out["margin_shift"] = out["margin_fail"] - out["margin_pre"]

    # Study-2 contrasts (present only when the extended conditions were run).
    for cond in ("fail_echo", "fail_terse", "fail_verbose", "fail_instr",
                 "fail_k2", "fail_k3", "fail_early", "fail_other",
                 "abstract_min"):
        out[f"G_{cond}"] = f("pre") - f(cond)
        out[f"delta_{cond}"] = f(cond) - f("fail")

    # Two robustness variants of the decomposition.
    #  - against the counterfactual *success* observation rather than the
    #    valence-free one, in case "Call recorded." reads as mildly positive;
    #  - per-token, so the numbers do not scale with action length.
    out["copy_succ"] = f("succ") - f("pre")
    # Identical to `polarity` by construction -- referencing the semantic term
    # to the success counterfactual *is* the polarity contrast. Kept under both
    # names because each reads naturally in its own context, but the paper
    # reports it once.
    out["sem_succ"] = f("fail") - f("succ")
    ntok = _col(out, "fail", "failed", prefix="ntok")
    with np.errstate(invalid="ignore", divide="ignore"):
        out["G_per_token"] = out["G"] / ntok
        out["copy_per_token"] = out["copy"] / ntok
        out["sem_per_token"] = out["sem"] / ntok
    out["G_negative"] = (out["G"] < 0).astype(float)

    # Normalised probability of re-emitting the failed call, over the candidate
    # set actually scored in that condition.
    for cond in ("pre", "fail", "abstract", "neut", "succ"):
        cols = [
            c
            for c in out.columns
            if c.startswith(f"lp__{cond}__")
        ]
        if len(cols) < 2:
            continue
        mat = out[cols].to_numpy(dtype=float)
        # Rows where nothing was scored under this condition stay NaN rather
        # than becoming a spurious 0/0.
        usable = np.isfinite(mat).any(axis=1)
        ex = np.zeros_like(mat)
        denom = np.full(mat.shape[0], np.nan)
        if usable.any():
            sub = mat[usable]
            mx = np.nanmax(sub, axis=1, keepdims=True)
            e = np.exp(sub - mx)
            e[np.isnan(e)] = 0.0
            ex[usable] = e
            denom[usable] = e.sum(axis=1)
        idx = cols.index(f"lp__{cond}__failed")
        out[f"p_repeat_{cond}"] = ex[:, idx] / np.where(denom > 0, denom, np.nan)
    if "p_repeat_pre" in out and "p_repeat_fail" in out:
        out["p_repeat_shift"] = out["p_repeat_fail"] - out["p_repeat_pre"]

    for cond in ("pre", "fail", "abstract"):
        gc = f"greedy__{cond}__failed"
        if gc in out.columns:
            out[f"greedy_repeat_{cond}"] = out[gc].astype("float")
    return out


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

#: Measures reported in the main tables, with a one-line gloss each.
MEASURES: Dict[str, str] = {
    "G": "corrective gain: logP(fail|pre) - logP(fail|after failure)",
    "copy": "surface-form term: effect of the call being in context at all",
    "sem": "semantic term: extra effect of marking that call as failed",
    "copy_pad": "surface-form term, length-matched neutral observation",
    "sem_pad": "semantic term, length-matched neutral observation",
    "polarity": "logP(fail|failure obs) - logP(fail|success obs)",
    "G_abstract": "corrective gain under the abstracted-failure harness",
    "repair": "change in logP of the correct call after the failure",
    "margin_shift": "change in (gold - failed) log-prob margin",
    "p_repeat_pre": "normalised P(re-emit failed call) before the attempt",
    "p_repeat_fail": "normalised P(re-emit failed call) after the failure",
    "p_repeat_shift": "change in normalised P(re-emit failed call)",
    "greedy_repeat_pre": "greedy decoding emits the failed call, before",
    "greedy_repeat_fail": "greedy decoding emits the failed call, after",
    "greedy_repeat_abstract": "greedy emits the failed call, abstracted harness",
    "copy_succ": "surface-form term, referenced to the success counterfactual",
    "sem_succ": "semantic term, referenced to the success counterfactual",
    "G_per_token": "corrective gain per action token",
    "copy_per_token": "surface-form term per action token",
    "sem_per_token": "semantic term per action token",
    "G_negative": "fraction of items with a negative corrective gain",
    "delta_abstract": "paired change vs the standard harness, abstracted failure",
    "delta_fail_echo": "paired change vs standard, error quotes the call",
    "delta_fail_terse": "paired change vs standard, terse error",
    "delta_fail_verbose": "paired change vs standard, verbose error",
    "delta_fail_instr": "paired change vs standard, + do-not-repeat instruction",
    "delta_fail_k2": "paired change vs standard, call failed twice",
    "delta_fail_k3": "paired change vs standard, call failed three times",
    "delta_fail_early": "paired change vs standard, failure placed earlier",
    "delta_fail_other": "paired change vs standard, a different call failed",
    "delta_abstract_min": "paired change vs standard, bare failure marker",
}


def summarise(
    derived: pd.DataFrame,
    meta: pd.DataFrame,
    measures: Optional[Sequence[str]] = None,
    group_cols: Sequence[str] = ("model", "env"),
    n_boot: int = 10000,
    seed: int = 20260808,
) -> pd.DataFrame:
    """Cluster-bootstrap summary of each measure, clustered on task_id."""
    measures = list(measures or MEASURES)
    df = derived.merge(meta, on=["env", "item_id"], how="left")
    rows = []
    for keys, sub in df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        for m in measures:
            if m not in sub.columns:
                continue
            vals = sub[m].to_numpy(dtype=float)
            clusters = sub["task_id"].fillna("na").to_numpy()
            mask = np.isfinite(vals)
            if mask.sum() < 3:
                continue
            est = cluster_bootstrap(
                vals[mask], clusters[mask], n_boot=n_boot, seed=seed
            )
            row = dict(zip(group_cols, keys))
            row["measure"] = m
            row.update(est.as_dict())
            rows.append(row)
    return pd.DataFrame(rows)


def attach_model_info(df: pd.DataFrame) -> pd.DataFrame:
    """Join model family / label / parameter count onto a summary frame.

    Returns an empty frame unchanged: partial runs legitimately produce
    summaries with nothing in them, and that should not be an error.
    """
    from .models import REGISTRY

    if df.empty or "model" not in df.columns:
        return df

    info = pd.DataFrame(
        [
            {
                "model": k,
                "family": s.family,
                "label": s.label,
                "params_b": s.params_b,
                "log_params": float(np.log10(s.params_b * 1e9)),
            }
            for k, s in REGISTRY.items()
        ]
    )
    return df.merge(info, on="model", how="left")
