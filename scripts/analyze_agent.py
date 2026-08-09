"""Summarise end-to-end rollouts: success, repetition, and cost by harness.

Comparisons against the standard harness are paired on task, because the same
task set is run under every harness and pairing removes task difficulty from the
contrast.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from slmecho import tables  # noqa: E402
from slmecho.analysis import attach_model_info  # noqa: E402
from slmecho.envs.toolshed import TOOLS_BY_NAME  # noqa: E402
from slmecho.stats import cluster_bootstrap  # noqa: E402

TOOL_NAMES = set(TOOLS_BY_NAME) | {"finish"}

HARNESS_ORDER = [
    "verbatim", "verbatim+instr", "abstract", "drop",
    "verbatim+ban", "abstract+ban", "drop+ban",
]
HARNESS_LABEL = {
    "verbatim": "verbatim (standard)",
    "verbatim+instr": r"\quad + ``do not repeat''",
    "abstract": "abstract",
    "drop": "drop",
    "verbatim+ban": r"verbatim + ban",
    "abstract+ban": r"abstract + ban",
    "drop+ban": r"drop + ban",
}
METRICS = {
    "solved": "task success",
    "stuck": "ended in a loop (step limit + repeated action)",
    "exact_repeat_rate": "exact repeat rate",
    "canonical_repeat_rate": "canonical repeat rate",
    "ok_call_rate": "valid-call rate",
    "n_steps": "steps used",
    "gen_tokens": "generated tokens",
    "prompt_tokens": "prompt tokens",
    "prompt_tokens_per_step": "prompt tokens per step",
    "truncated_rate": "actions cut off by the token budget",
    "prose_rate": "prose emitted where a call was expected",
    "seconds": "seconds / rollout",
}


def load(raw_dir: str) -> pd.DataFrame:
    rows: List[dict] = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.jsonl"))):
        name = os.path.basename(path)[: -len(".jsonl")]
        parts = name.split("__")
        if len(parts) < 5:
            continue
        model, env, harness, temp, seed = parts[:5]
        harness = harness.replace("-", "+") if harness.count("-") else harness
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                ok = r.get("ok") or []
                r["ok_call_rate"] = float(np.mean(ok)) if ok else np.nan
                r["harness"] = harness
                r["model"] = model
                r["env"] = env
                r["temperature"] = float(temp[1:])
                r["seed_tag"] = seed
                r["solved"] = float(bool(r["solved"]))
                # Context length per step is the fair way to compare harnesses:
                # a harness that takes more steps accumulates more prompt tokens
                # even if each of its contexts is shorter.
                steps = r.get("n_steps") or 0
                r["prompt_tokens_per_step"] = (
                    r["prompt_tokens"] / steps if steps else float("nan")
                )
                # Break parse failures into the two kinds that matter: an action
                # cut off by the token budget (our artefact) versus prose where
                # a call was expected (the model's limitation).
                trunc = prose = 0
                for a, e in zip(r.get("actions") or [], r.get("error_types") or []):
                    if e != "parse_error":
                        continue
                    head = a.strip().split("(")[0].strip()
                    if head in TOOL_NAMES and a.count("(") > a.count(")"):
                        trunc += 1
                    elif head not in TOOL_NAMES:
                        prose += 1
                n_act = max(len(r.get("actions") or []), 1)
                r["truncated_rate"] = trunc / n_act
                r["prose_rate"] = prose / n_act
                r["stuck"] = float(
                    r.get("stop_reason") == "step_limit"
                    and r.get("n_canonical_repeats", 0) >= 2
                )
                rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/raw/agent")
    ap.add_argument("--out-dir", default="results/processed")
    ap.add_argument("--tables-dir", default="paper/tables")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    df = load(args.raw_dir)
    if df.empty:
        print("no agent results yet")
        return
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "agent_rollouts.csv"), index=False)

    rows = []
    for (model, env, harness), sub in df.groupby(["model", "env", "harness"]):
        for metric in METRICS:
            if metric not in sub.columns:
                continue
            vals = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(vals)
            if mask.sum() < 3:
                continue
            est = cluster_bootstrap(
                vals[mask], sub["task_id"].to_numpy()[mask], n_boot=args.n_boot
            )
            rows.append(
                dict(model=model, env=env, harness=harness, metric=metric,
                     **est.as_dict())
            )
    summary = attach_model_info(pd.DataFrame(rows))
    summary.to_csv(os.path.join(args.out_dir, "agent_summary.csv"), index=False)

    # Paired contrasts against the standard harness, on matched tasks.
    contrasts = []
    for (model, env), sub in df.groupby(["model", "env"]):
        base = sub[sub["harness"] == "verbatim"].set_index("task_id")
        if base.empty:
            continue
        for harness, hsub in sub.groupby("harness"):
            if harness == "verbatim":
                continue
            hsub = hsub.set_index("task_id")
            common = base.index.intersection(hsub.index)
            if len(common) < 5:
                continue
            for metric in METRICS:
                if metric not in base.columns:
                    continue
                a = pd.to_numeric(hsub.loc[common, metric], errors="coerce").to_numpy(float)
                b = pd.to_numeric(base.loc[common, metric], errors="coerce").to_numpy(float)
                d = a - b
                mask = np.isfinite(d)
                if mask.sum() < 5:
                    continue
                est = cluster_bootstrap(
                    d[mask], np.asarray(common)[mask], n_boot=args.n_boot
                )
                contrasts.append(
                    dict(model=model, env=env, harness=harness, metric=metric,
                         n_paired=int(mask.sum()), **est.as_dict())
                )
    contrast_df = attach_model_info(pd.DataFrame(contrasts))
    contrast_df.to_csv(os.path.join(args.out_dir, "agent_contrasts.csv"), index=False)
    if summary.empty:
        print("no (model, harness, metric) cell has enough rollouts yet")
        return

    # ---- LaTeX ----
    os.makedirs(args.tables_dir, exist_ok=True)
    for env in sorted(summary["env"].unique()):
        sub = summary[summary["env"] == env]
        models = list(dict.fromkeys(sub.sort_values("params_b")["model"]))
        harnesses = [h for h in HARNESS_ORDER if h in set(sub["harness"])]
        lines = [
            tables.PREAMBLE_HINT,
            r"\begin{table*}[t]", r"\centering", r"\small",
            r"\setlength{\tabcolsep}{4pt}",
            "\\begin{tabular}{l " + " ".join(["c c c"] * len(models)) + "}",
            r"\toprule",
            " & " + " & ".join(
                f"\\multicolumn{{3}}{{c}}{{{sub[sub['model']==m]['label'].iloc[0]}}}"
                for m in models
            ) + r" \\",
            "Harness & " + " & ".join(["success & exact rep. & canon. rep."] * len(models)) + r" \\",
            r"\midrule",
        ]
        for h in harnesses:
            cells = []
            for m in models:
                for metric in ("solved", "exact_repeat_rate", "canonical_repeat_rate"):
                    r = sub[(sub["model"] == m) & (sub["harness"] == h)
                            & (sub["metric"] == metric)]
                    cells.append(
                        "--" if r.empty else tables.ci_cell(
                            float(r["mean"].iloc[0]), float(r["ci_lo"].iloc[0]),
                            float(r["ci_hi"].iloc[0]), 2,
                            bold_if_excludes_zero=False,
                        )
                    )
            lines.append(f"{HARNESS_LABEL.get(h, h)} & " + " & ".join(cells) + r" \\")
        lines += [
            r"\bottomrule", r"\end{tabular}",
            r"\caption{End-to-end rollouts on ToolShed. Success is the "
            r"environment's goal predicate; \emph{exact rep.} is the fraction of "
            r"failed actions that repeat an earlier failed action byte for byte, "
            r"and \emph{canon. rep.} applies the same count after parsing and "
            r"normalising the call, so paraphrases of a banned string are still "
            r"counted. Brackets give 95\% cluster bootstrap intervals over tasks.}",
            r"\label{tab:agent-main}", r"\end{table*}",
        ]
        tables.write(os.path.join(args.tables_dir, "agent_main.tex"), "\n".join(lines))

        # Cost table.
        clines = [
            tables.PREAMBLE_HINT,
            r"\begin{table}[t]", r"\centering", r"\small",
            "\\begin{tabular}{l " + " ".join(["r r r r"] * len(models)) + "}",
            r"\toprule",
            " & " + " & ".join(
                f"\\multicolumn{{4}}{{c}}{{{sub[sub['model']==m]['label'].iloc[0]}}}"
                for m in models) + r" \\",
            "Harness & " + " & ".join(
                ["gen & prompt & ctx/step & steps"] * len(models)) + r" \\",
            r"\midrule",
        ]
        for h in harnesses:
            cells = []
            for m in models:
                for metric in ("gen_tokens", "prompt_tokens",
                               "prompt_tokens_per_step", "n_steps"):
                    r = sub[(sub["model"] == m) & (sub["harness"] == h)
                            & (sub["metric"] == metric)]
                    cells.append("--" if r.empty else f"{float(r['mean'].iloc[0]):.0f}")
            clines.append(f"{HARNESS_LABEL.get(h, h)} & " + " & ".join(cells) + r" \\")
        clines += [
            r"\bottomrule", r"\end{tabular}",
            r"\caption{Cost per rollout. \emph{gen} and \emph{prompt} are "
            r"tokens summed over a rollout's steps, so a harness that takes more "
            r"steps accumulates more of both; \emph{ctx/step} divides the prompt "
            r"total by the number of steps and is the fair comparison of context "
            r"size. The decoder ban adds no tokens at all. The abstraction "
            r"produces a slightly shorter context per step while taking more "
            r"steps, because it does not terminate early on a loop. Wall-clock is "
            r"omitted: these runs shared a CPU with other jobs, so it measures "
            r"scheduling rather than the harness.}",
            r"\label{tab:cost}", r"\end{table}",
        ]
        tables.write(os.path.join(args.tables_dir, "agent_cost.tex"), "\n".join(clines))

    print(summary.pivot_table(index=["model", "harness"], columns="metric",
                              values="mean")[
        [c for c in ("solved", "exact_repeat_rate", "canonical_repeat_rate",
                     "ok_call_rate", "n_steps", "gen_tokens")
         if c in set(summary["metric"])]].round(3).to_string())
    if not contrast_df.empty:
        print("\n--- paired contrasts vs verbatim ---")
        for r in contrast_df[contrast_df["metric"].isin(
                ["solved", "exact_repeat_rate", "canonical_repeat_rate"])].itertuples():
            print(f"  {r.model:14s} {r.harness:16s} {r.metric:24s} "
                  f"{r.mean:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] n={r.n_paired}")


if __name__ == "__main__":
    main()
