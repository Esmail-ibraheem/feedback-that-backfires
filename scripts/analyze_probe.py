"""Turn raw probe scores into processed frames, LaTeX tables and figures.

Reads only `results/raw/probe/*.jsonl` and `data/probe_*.jsonl`; writes
`results/processed/*.csv`, `paper/tables/*.tex` and `paper/figures/*`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from slmecho.analysis import (  # noqa: E402
    MEASURES,
    attach_model_info,
    derive,
    item_metadata,
    load_raw,
    summarise,
    to_wide,
)
from slmecho import tables  # noqa: E402
from slmecho.stats import cluster_bootstrap, holm, weighted_least_squares  # noqa: E402

ENV_NAME = {"toolshed": "ToolShed (tool calls)",
            "coderepair": "CodeRepair (MBPP program repair)"}

ITEM_FILES = {
    "toolshed": "data/probe_toolshed.jsonl",
    "coderepair": "data/probe_coderepair.jsonl",
}

# Five columns is what fits a two-column layout without shrinking the font;
# margin and repair move to the robustness table.
MAIN_MEASURES = ["G", "G_per_token", "copy", "sem", "G_abstract"]
# `sem_succ` is deliberately absent: it is the same expression as `polarity`,
# and printing one quantity in two columns invites the reader to treat it as
# two pieces of evidence.
ROBUST_MEASURES = ["G_negative", "margin_shift", "repair", "copy_succ",
                   "polarity"]
MAIN_HEADERS = {
    "G": r"$G$",
    "copy": r"copy",
    "sem": r"sem",
    "polarity": r"polarity",
    "margin_shift": r"$\Delta$margin",
    "G_per_token": r"$G$/token",
    "repair": r"repair",
    "G_abstract": r"$G_{\text{abs}}$",
    "p_repeat_pre": r"$p_{\text{rep}}$ before",
    "p_repeat_fail": r"$p_{\text{rep}}$ after",
    "p_repeat_shift": r"$\Delta p_{\text{rep}}$",
    "greedy_repeat_pre": r"greedy before",
    "greedy_repeat_fail": r"greedy after",
    "greedy_repeat_abstract": r"greedy abs.",
}

# The manipulation table reports *paired* changes against the standard harness
# on the same items, which is the only fair comparison when different variants
# were scored on different subsets.
EXT_MEASURES = [
    "delta_fail_terse",
    "delta_fail_verbose",
    "delta_fail_echo",
    "delta_fail_early",
    "delta_fail_instr",
    "delta_fail_k2",
    "delta_fail_k3",
    "delta_fail_other",
    "delta_abstract",
    "delta_abstract_min",
]
EXT_HEADERS = {
    "delta_fail_terse": r"terse error text",
    "delta_fail_verbose": r"verbose error text",
    "delta_fail_echo": r"error quotes the failed call",
    "delta_fail_early": r"failure placed before the successful steps",
    "delta_fail_instr": r"+ ``do not repeat'' instruction",
    "delta_fail_k2": r"the same call failed twice",
    "delta_fail_k3": r"the same call failed three times",
    "delta_fail_other": r"placebo: a \emph{different} call failed",
    "delta_abstract": r"abstracted failure (no verbatim call)",
    "delta_abstract_min": r"\quad bare marker, no diagnosis",
}


def write_by_error_table(out_dir: str, tables_dir: str, keep=None) -> None:
    """Appendix table: corrective gain broken down by the error family provoked."""
    err_path = os.path.join(out_dir, "probe_by_error_type.csv")
    if not os.path.exists(err_path):
        return
    be = pd.read_csv(err_path)
    be = be[(be["measure"] == "G") & (be["env"] == "toolshed")]
    if keep is not None:
        # Same inclusion rule as everywhere else in the manuscript.
        be = be[be.apply(lambda r: (r["model"], r["env"]) in keep, axis=1)]
    if be.empty:
        return
    models = list(dict.fromkeys(be.sort_values("params_b")["model"]))
    order = be.groupby("error_type")["mean"].mean().sort_values().index
    lines = [
        tables.PREAMBLE_HINT,
        # [H]: keep it under its own appendix heading rather than letting it
        # float up onto the previous page.
        r"\begin{table}[H]",
        r"\centering",
        # Sized by the wrapper; \small overruns the appendix text block.
        r"\tblfontsmall",
        r"\setlength{\tabcolsep}{\tblsepsmall}",
        "\\begin{tabular}{l " + " ".join(["c"] * len(models)) + "}",
        r"\toprule",
        "Error family & "
        + " & ".join(be[be["model"] == m]["label"].iloc[0] for m in models)
        + r" \\",
        r"\midrule",
    ]
    for et in order:
        cells = []
        for m in models:
            row = be[(be["model"] == m) & (be["error_type"] == et)]
            cells.append(
                "--"
                if row.empty
                else tables.ci_cell(
                    float(row["mean"].iloc[0]),
                    float(row["ci_lo"].iloc[0]),
                    float(row["ci_hi"].iloc[0]),
                    1,
                )
            )
        safe = et.replace("_", r"\_")
        lines.append(f"\\texttt{{{safe}}} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Corrective gain (nats) by error family on ToolShed, sorted by "
        r"pooled mean. Rows are the error the failing call provoked; a broadly "
        r"uniform column argues against any explanation resting on the wording "
        r"of one error type.}",
        r"\label{tab:by-error}",
        r"\end{table}",
    ]
    tables.write(os.path.join(tables_dir, "probe_by_error.tex"), "\n".join(lines))


def _write_partial_note(tables_dir: str, partial, min_items: int) -> None:
    """A sentence naming any run that was excluded for being incomplete.

    Written whether or not there are any, so the appendix can \\input{} it
    unconditionally and a silently-dropped checkpoint is impossible.
    """
    os.makedirs(tables_dir, exist_ok=True)
    if not partial:
        body = (f"Every model/environment cell reported here reached at least "
                f"{min_items} scored items; none were excluded on that ground.")
    else:
        listed = "; ".join(
            f"{lab} on {ENV_NAME.get(env, env)} ($n={n}$)" for lab, env, n in partial
        )
        body = (f"The following runs were still filling up when the results were "
                f"frozen and are excluded from every table, average and fitted "
                f"line in this paper, on a pre-set threshold of {min_items} "
                f"scored items: {listed}. Their raw scores are in the release.")
    tables.write(os.path.join(tables_dir, "partial_runs.tex"), body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/raw/probe")
    ap.add_argument("--out-dir", default="results/processed")
    ap.add_argument("--tables-dir", default="paper/tables")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument(
        "--min-items", type=int, default=30,
        help="model/env cells with fewer scored items than this are kept in the "
             "CSVs but excluded from the manuscript's tables and averages, so a "
             "checkpoint that is still filling up cannot move a headline number",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.raw_dir, "*__*.jsonl")))
    paths = [p for p in paths if not p.endswith(".greedy.jsonl")]
    raw = load_raw(paths)
    if raw.empty:
        print("no raw probe results yet")
        return
    meta = item_metadata(ITEM_FILES)

    wide = to_wide(raw)
    derived = derive(wide)
    derived.to_csv(os.path.join(args.out_dir, "probe_measures.csv"), index=False)

    all_measures = [m for m in MEASURES if m in derived.columns]
    all_measures += [c for c in derived.columns if c.startswith("G_fail_")]
    all_measures += [c for c in derived.columns if c.startswith("delta_")]
    summary = summarise(derived, meta, measures=sorted(set(all_measures)),
                        n_boot=args.n_boot)
    summary = attach_model_info(summary)
    summary.to_csv(os.path.join(args.out_dir, "probe_summary.csv"), index=False)

    # Holm-adjusted p-values for the headline family (G, per model and env).
    head = summary[summary["measure"] == "G"]
    pvals = {f"{r.model}/{r.env}": r.p for r in head.itertuples()}
    adj = holm(pvals)
    summary["p_holm"] = summary.apply(
        lambda r: adj.get(f"{r['model']}/{r['env']}") if r["measure"] == "G" else np.nan,
        axis=1,
    )
    summary.to_csv(os.path.join(args.out_dir, "probe_summary.csv"), index=False)

    # Per-error-family breakdown (ToolShed only; code errors have their own set).
    for cols, name in (
        (("model", "env", "error_type"), "probe_by_error_type"),
        (("model", "env", "operator"), "probe_by_operator"),
        (("model", "env", "template"), "probe_by_template"),
        (("model", "env", "n_prefix_steps"), "probe_by_depth"),
    ):
        part = summarise(derived, meta, measures=["G", "copy", "sem"],
                         group_cols=cols, n_boot=2000)
        if part.empty:
            print(f"[warn] no rows for breakdown {name}")
            continue
        attach_model_info(part).to_csv(
            os.path.join(args.out_dir, f"{name}.csv"), index=False
        )

    # Everything below this line is what the manuscript sees. A model/env cell
    # that is still filling up stays in the CSVs -- deleting data is worse than
    # reporting it -- but it is not allowed into a table or a fitted line,
    # because an eleven-item estimate next to a hundred-item one reads as if
    # they carry the same weight.
    gcov = summary[summary["measure"] == "G"][["model", "env", "n", "label"]]
    keep = {(r.model, r.env) for r in gcov.itertuples() if r.n >= args.min_items}
    partial = sorted(
        (r.label, r.env, int(r.n)) for r in gcov.itertuples()
        if (r.model, r.env) not in keep
    )
    if partial:
        print(f"[partial] excluded from the manuscript (< {args.min_items} items): "
              + ", ".join(f"{lab}/{env} n={n}" for lab, env, n in partial))
    # Recorded in the CSV as a column rather than applied by each consumer, so
    # the tables, the figures and the inline numbers cannot disagree about who
    # is in the study.
    summary["published"] = summary.apply(
        lambda r: (r["model"], r["env"]) in keep, axis=1)
    summary.to_csv(os.path.join(args.out_dir, "probe_summary.csv"), index=False)
    summary = summary[summary["published"]].reset_index(drop=True)
    _write_partial_note(args.tables_dir, partial, args.min_items)

    # Scaling fit: G against log10(params), pooled and per family.
    fits = []
    for env_name, sub in summary[summary["measure"] == "G"].groupby("env"):
        sub = sub.dropna(subset=["log_params", "mean"])
        if len(sub) >= 3:
            slope, intercept, r2 = weighted_least_squares(sub["log_params"], sub["mean"])
            cross = -intercept / slope if slope else np.nan
            fits.append(
                {
                    "env": env_name,
                    "family": "all",
                    "n_models": len(sub),
                    "slope_per_decade": slope,
                    "intercept": intercept,
                    "r2": r2,
                    "zero_crossing_params_b": float(10**cross / 1e9) if slope else np.nan,
                }
            )
        for fam, fsub in sub.groupby("family"):
            if len(fsub) >= 2:
                slope, intercept, r2 = weighted_least_squares(
                    fsub["log_params"], fsub["mean"]
                )
                cross = -intercept / slope if slope else np.nan
                fits.append(
                    {
                        "env": env_name,
                        "family": fam,
                        "n_models": len(fsub),
                        "slope_per_decade": slope,
                        "intercept": intercept,
                        "r2": r2,
                        "zero_crossing_params_b": float(10**cross / 1e9)
                        if slope
                        else np.nan,
                    }
                )
        # How much the extrapolated zero-crossing depends on any one model.
        # The crossing lies far outside the measured range, so it is a long
        # lever on a short fit; refitting with each model held out says how
        # long. This is the evidence for declining to claim the number, and it
        # is generated rather than asserted.
        if len(sub) >= 4:
            for _, held in sub.iterrows():
                loo = sub[sub["model"] != held["model"]]
                slope, intercept, r2 = weighted_least_squares(
                    loo["log_params"], loo["mean"])
                cross = -intercept / slope if slope else np.nan
                fits.append({
                    "env": env_name,
                    "family": f"loo:-{held['label']}",
                    "n_models": len(loo),
                    "slope_per_decade": slope,
                    "intercept": intercept,
                    "r2": r2,
                    "zero_crossing_params_b": float(10**cross / 1e9)
                    if slope else np.nan,
                })

    pd.DataFrame(fits).to_csv(
        os.path.join(args.out_dir, "probe_scaling_fits.csv"), index=False
    )

    # ---- LaTeX tables ----
    os.makedirs(args.tables_dir, exist_ok=True)
    for env_name in sorted(summary["env"].unique()):
        present = [m for m in MAIN_MEASURES if m in set(summary["measure"])]
        tables.write(
            os.path.join(args.tables_dir, f"probe_main_{env_name}.tex"),
            tables.measures_by_model(
                summary,
                present,
                MAIN_HEADERS,
                env_name,
                caption=(
                    f"Probe results on {ENV_NAME.get(env_name, env_name)}. All "
                    r"quantities are in nats and are "
                    r"differences in the summed log-probability of the \emph{same} "
                    r"action string under two contexts. $G>0$ means the harness's "
                    r"failure record made the failed call less likely; $G<0$ is "
                    r"feedback inversion. Cells give the mean over items with a 95\% "
                    r"cluster bootstrap interval (clustered on task); bold marks "
                    r"intervals excluding zero."
                ),
                label=f"tab:probe-main-{env_name}",
            ),
        )
        tables.write(
            os.path.join(args.tables_dir, f"probe_repeat_{env_name}.tex"),
            tables.measures_by_model(
                summary,
                [
                    m
                    for m in [
                        "p_repeat_pre",
                        "p_repeat_fail",
                        "p_repeat_shift",
                        "greedy_repeat_pre",
                        "greedy_repeat_fail",
                        "greedy_repeat_abstract",
                    ]
                    if m in set(summary["measure"])
                ],
                MAIN_HEADERS,
                env_name,
                caption=(
                    f"Repetition probabilities on {env_name}. $p_{{\\text{{rep}}}}$ is "
                    r"the normalised probability of the failed call among the scored "
                    r"candidate actions; \emph{greedy} is the fraction of items on "
                    r"which greedy decoding reproduces the failed call exactly."
                ),
                label=f"tab:probe-repeat-{env_name}",
                digits=3,
            ),
        )
    robust_present = [m for m in ROBUST_MEASURES if m in set(summary["measure"])]
    if robust_present:
        tables.write(
            os.path.join(args.tables_dir, "probe_robustness.tex"),
            tables.measures_by_model(
                summary, robust_present,
                {"G_negative": r"frac. $G<0$", "margin_shift": r"$\Delta$margin",
                 "repair": r"repair", "copy_succ": r"copy$^{\checkmark}$",
                 "sem_succ": r"sem$^{\checkmark}$", "polarity": r"polarity"},
                "toolshed",
                caption=(
                    r"Robustness and secondary quantities. \emph{frac.\ $G<0$} is "
                    r"the assumption-light item-level sign statistic; "
                    r"$\Delta$margin is the change in the gold-minus-failed "
                    r"log-probability gap; \emph{repair} is the change in the "
                    r"correct action's log-probability. copy$^{\checkmark}$ "
                    r"repeats the surface-form term against the counterfactual "
                    r"\emph{success} observation instead of the valence-free "
                    r"one, in case the latter reads as mildly positive; the "
                    r"matching semantic term referenced the same way is "
                    r"algebraically the polarity contrast, so it appears once, "
                    r"in the last column."
                ),
                # Two decimals: these are tens of nats, and a third digit is
                # far below the width of the interval beside it.
                label="tab:probe-robustness", digits=2,
            ),
        )

    ext_present = [m for m in EXT_MEASURES if m in set(summary["measure"])]
    if len(ext_present) > 2:
        tables.write(
            os.path.join(args.tables_dir, "probe_manipulations.tex"),
            tables.conditions_table(
                summary,
                ext_present,
                EXT_HEADERS,
                "toolshed",
                caption=(
                    r"Harness variants, as \emph{paired} changes in the "
                    r"log-probability of re-emitting the failed call, relative to "
                    r"the standard harness on the same items. Positive means the "
                    r"variant makes repetition \emph{more} likely; negative means "
                    r"less. Every row uses the same items and the same failed "
                    r"call, and only how the failure is written into the "
                    r"transcript changes."
                ),
                label="tab:probe-manipulations",
            ),
        )

    write_by_error_table(args.out_dir, args.tables_dir, keep=keep)

    print(f"models: {sorted(summary['model'].unique())}")
    print(f"envs:   {sorted(summary['env'].unique())}")
    print("\n--- corrective gain G (nats), by model ---")
    show = summary[summary["measure"] == "G"].sort_values(["env", "params_b"])
    for r in show.itertuples():
        print(
            f"  {r.env:11s} {r.label:16s} {r.params_b:5.2f}B  "
            f"G = {r.mean:+7.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]  "
            f"d_z={r.dz:+.2f}  p={r.p:.4f}  n={r.n}/{r.n_clusters} clusters"
        )
    print("\n--- decomposition (copy / sem) ---")
    for env_name in sorted(summary["env"].unique()):
        for meas in ("copy", "sem", "polarity"):
            sub = summary[(summary["env"] == env_name) & (summary["measure"] == meas)]
            for r in sub.sort_values("params_b").itertuples():
                print(
                    f"  {env_name:11s} {r.label:16s} {meas:9s} "
                    f"{r.mean:+7.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"
                )


if __name__ == "__main__":
    main()
