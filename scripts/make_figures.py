"""Build every figure in the paper from processed results.

Each figure is a function so it can fail independently: a missing experiment
leaves a gap rather than aborting the whole build.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import matplotlib.pyplot as plt  # noqa: E402

from slmecho.plotting import (  # noqa: E402
    AQUA,
    BLUE,
    COL_WIDTH,
    FAMILY_COLORS,
    FAMILY_MARKERS,
    FULL_WIDTH,
    GREY,
    INK,
    MUTED,
    ORANGE,
    RULE,
    VIOLET,
    band,
    direct_label,
    grouped_bars,
    labels_no_overlap,
    param_axis,
    save,
    use_paper_style,
    zero_rule,
)

ENV_TITLES = {"toolshed": "Tool calls (ToolShed)", "coderepair": "Code repair (MBPP)"}


# --------------------------------------------------------------------------


def _published(summary: pd.DataFrame) -> pd.DataFrame:
    """Drop model/env cells that analyze_probe.py marked as still filling up.

    The flag lives in the CSV so that a plot and a table can never disagree
    about which checkpoints are in the study.
    """
    if "published" not in summary.columns:
        return summary
    return summary[summary["published"].astype(bool)].reset_index(drop=True)


def fig_scaling(summary: pd.DataFrame, out: str) -> Optional[str]:
    """Corrective gain against parameter count.

    Two rows. The top shows raw nats with an independent axis per environment,
    because an MBPP program is roughly five times as many tokens as a tool call
    and a shared axis would flatten one panel to nothing. The bottom shows the
    same quantity per action token, which *is* comparable across environments,
    and there the axis is shared on purpose --- the two panels landing on the
    same value is the point.
    """
    have = set(summary["measure"])
    rows = [m for m in ("G", "G_per_token") if m in have]
    if not rows:
        return None
    envs = [e for e in ("toolshed", "coderepair")
            if e in set(summary[summary["measure"] == rows[0]]["env"])]
    if not envs:
        return None

    fig, axes = plt.subplots(
        len(rows), len(envs),
        figsize=(FULL_WIDTH * (0.58 if len(envs) == 1 else 1.0),
                 2.3 * len(rows)),
        squeeze=False,
    )
    for r, measure in enumerate(rows):
        pertok = measure == "G_per_token"
        band_vals = summary[(summary["measure"] == measure)
                            & (summary["env"].isin(envs))]["mean"]
        for c, env in enumerate(envs):
            ax = axes[r][c]
            d = (summary[(summary["measure"] == measure) & (summary["env"] == env)]
                 .dropna(subset=["params_b"]).sort_values("params_b"))
            if d.empty:
                ax.set_visible(False)
                continue
            zero_rule(ax, 0.0,
                      "feedback suppresses the failed call" if r == 0 else None)
            label_items = []
            for fam, fd in d.groupby("family"):
                fd = fd.sort_values("params_b")
                color = FAMILY_COLORS.get(fam, GREY)
                ax.errorbar(
                    fd["params_b"], fd["mean"],
                    yerr=[fd["mean"] - fd["ci_lo"], fd["ci_hi"] - fd["mean"]],
                    fmt=FAMILY_MARKERS.get(fam, "o") + "-",
                    color=color, ecolor=color, elinewidth=0.9, capsize=1.6,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=3,
                )
                last = fd.iloc[-1]
                label_items.append((last["params_b"], last["mean"], fam, color))
            param_axis(ax, d["params_b"])
            ax.set_xlim(right=d["params_b"].max() * 3.6)
            if r == 0:
                ax.set_title(ENV_TITLES.get(env, env), loc="left", color=INK)
            if r == len(rows) - 1:
                ax.set_xlabel("parameters")
            if pertok and len(band_vals):
                lo, hi = float(band_vals.min()), float(band_vals.max())
                pad = max(0.35, (hi - lo) * 0.6)
                ax.set_ylim(lo - pad, min(0.25, hi + pad))
            ax.grid(axis="y", alpha=0.55)
            ax.set_axisbelow(True)
            labels_no_overlap(ax, label_items)
        axes[r][0].set_ylabel("$G$ (nats)" if not pertok
                              else "$G$ per action token")
    fig.tight_layout(w_pad=1.4, h_pad=1.0)
    return save(fig, out)[0]


def fig_decomposition(summary: pd.DataFrame, out: str, env: str = "toolshed") -> Optional[str]:
    """Where -G comes from: surface-form copying vs. corrective semantics."""
    d = summary[(summary["env"] == env) & (summary["measure"].isin(["copy", "sem", "G"]))]
    if d.empty:
        return None
    piv = d.pivot_table(index=["model", "label", "params_b"], columns="measure",
                        values="mean").reset_index().sort_values("params_b")
    lo = d.pivot_table(index=["model"], columns="measure", values="ci_lo")
    hi = d.pivot_table(index=["model"], columns="measure", values="ci_hi")
    labels = list(piv["label"])
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.62, 2.4))
    zero_rule(ax)
    series = [
        ("copying (call is in context)", list(piv["copy"]),
         [(lo.loc[m, "copy"], hi.loc[m, "copy"]) for m in piv["model"]], BLUE),
        ("semantics (call is marked failed)", list(piv["sem"]),
         [(lo.loc[m, "sem"], hi.loc[m, "sem"]) for m in piv["model"]], ORANGE),
    ]
    grouped_bars(ax, labels, series)
    ax.plot(np.arange(len(labels)), -piv["G"], "o", color=INK, markersize=4,
            zorder=4, label="total $-G$")
    ax.set_ylabel("contribution to $-G$ (nats)")
    ax.tick_params(axis="x", rotation=28)
    for t in ax.get_xticklabels():
        t.set_ha("right")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3,
              handlelength=1.1, columnspacing=1.2, borderaxespad=0)
    ax.grid(axis="y", alpha=0.55)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return save(fig, out)[0]


def fig_repeat(summary: pd.DataFrame, out: str, env: str = "toolshed") -> Optional[str]:
    """Before/after repetition probability, and exact greedy repeats."""
    need = ["p_repeat_pre", "p_repeat_fail", "greedy_repeat_pre", "greedy_repeat_fail"]
    d = summary[(summary["env"] == env) & (summary["measure"].isin(need))]
    if d.empty or d["measure"].nunique() < 2:
        return None
    piv = d.pivot_table(index=["model", "label", "params_b"], columns="measure",
                        values="mean").reset_index().sort_values("params_b")
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH * 0.86, 2.4))

    ax = axes[0]
    for _, r in piv.iterrows():
        if not (np.isfinite(r.get("p_repeat_pre", np.nan))
                and np.isfinite(r.get("p_repeat_fail", np.nan))):
            continue
        up = r["p_repeat_fail"] > r["p_repeat_pre"]
        color = ORANGE if up else AQUA
        ax.plot([0, 1], [r["p_repeat_pre"], r["p_repeat_fail"]], "-o",
                color=color, markersize=3.6, markeredgecolor="white",
                markeredgewidth=0.5)
        direct_label(ax, 1, r["p_repeat_fail"], r["label"], color)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["before the\nattempt", "after it\nfailed"])
    ax.set_xlim(-0.15, 2.3)
    ax.set_ylabel("P(re-emit the failed call)")
    ax.set_title("Normalised repeat probability", loc="left")
    ax.grid(axis="y", alpha=0.55)
    ax.set_axisbelow(True)

    ax = axes[1]
    cols = [c for c in ("greedy_repeat_pre", "greedy_repeat_fail") if c in piv.columns]
    if cols:
        before = list(piv.get("greedy_repeat_pre", []))
        after = list(piv.get("greedy_repeat_fail", []))
        # When the "before" rate is identically zero for every model, a bar
        # series of zeros is just an empty legend entry; say it in words.
        if before and max(before) == 0:
            series = [("after the failure", after, None, ORANGE)]
            ax.annotate("before the attempt: 0% for every model",
                        xy=(0.02, 0.93), xycoords="axes fraction",
                        fontsize=7, color=MUTED)
        else:
            series = [("before", before, None, GREY),
                      ("after the failure", after, None, ORANGE)]
        series = [s for s in series if len(s[1])]
        grouped_bars(ax, list(piv["label"]), series)
        ax.set_ylabel("fraction of items")
        ax.set_title("Greedy decoding reproduces the failed call", loc="left")
        ax.tick_params(axis="x", rotation=28)
        for t in ax.get_xticklabels():
            t.set_ha("right")
        if len(series) > 1:
            ax.legend(loc="upper left")
        ax.grid(axis="y", alpha=0.55)
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=1.6)
    return save(fig, out)[0]


def fig_manipulations(summary: pd.DataFrame, out: str, env: str = "toolshed") -> Optional[str]:
    """Study 2: G under each harness variant."""
    order = [
        ("G_fail_terse", "terse error"),
        ("G", "standard error"),
        ("G_fail_verbose", "verbose error"),
        ("G_fail_echo", "error quotes the call"),
        ("G_fail_instr", "+ do-not-repeat instruction"),
        ("G_fail_early", "failure placed earlier"),
        ("G_fail_k2", "failed twice"),
        ("G_fail_k3", "failed three times"),
        ("G_abstract", "abstracted (no verbatim call)"),
    ]
    d = summary[(summary["env"] == env)]
    have = set(d["measure"])
    order = [(m, lab) for m, lab in order if m in have]
    if len(order) < 3:
        return None
    models = list(
        d[d["measure"] == "G_fail_k2"].sort_values("params_b")["model"].unique()
    )
    if not models:
        return None
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.72, 2.9))
    ys = np.arange(len(order))[::-1]
    colors = [BLUE, ORANGE, AQUA, VIOLET]
    for i, m in enumerate(models):
        md = d[d["model"] == m]
        means, los, his = [], [], []
        for meas, _lab in order:
            r = md[md["measure"] == meas]
            means.append(float(r["mean"].iloc[0]) if len(r) else np.nan)
            los.append(float(r["ci_lo"].iloc[0]) if len(r) else np.nan)
            his.append(float(r["ci_hi"].iloc[0]) if len(r) else np.nan)
        off = (i - (len(models) - 1) / 2) * 0.17
        label = md["label"].iloc[0]
        ax.errorbar(
            means, ys + off,
            xerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
            fmt=FAMILY_MARKERS.get(md["family"].iloc[0], "o"),
            color=colors[i % len(colors)], ecolor=colors[i % len(colors)],
            elinewidth=0.9, capsize=1.6, markersize=4,
            markeredgecolor="white", markeredgewidth=0.5, label=label, zorder=3,
        )
    ax.axvline(0, color="#9a9a9a", linewidth=0.9, linestyle=(0, (4, 3)), zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _m, lab in order])
    ax.set_xlabel("corrective gain $G$ (nats)")
    # Above the axes, not inside them: the bottom-right corner of this panel is
    # where the abstracted-failure row lands, and a legend box there sits on
    # top of the interval it is labelling.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=len(models),
              handlelength=1.1, columnspacing=1.3, borderaxespad=0,
              frameon=False)
    ax.grid(axis="x", alpha=0.55)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return save(fig, out)[0]


def fig_error_types(by_error: pd.DataFrame, out: str, env: str = "toolshed") -> Optional[str]:
    d = by_error[(by_error["env"] == env) & (by_error["measure"] == "G")]
    if d.empty:
        return None
    piv = d.pivot_table(index="error_type", columns="label", values="mean")
    order = piv.mean(axis=1).sort_values().index
    piv = piv.loc[order]
    cols = sorted(piv.columns, key=lambda c: d[d["label"] == c]["params_b"].iloc[0])
    piv = piv[cols]
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.62, 0.24 * len(piv) + 1.5))
    vmax = float(np.nanmax(np.abs(piv.to_numpy())))
    im = ax.imshow(piv.to_numpy(), cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=28, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([i.replace("_", " ") for i in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=6,
                        color="#1a1a1a" if abs(v) < vmax * 0.55 else "white")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("$G$ (nats)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    ax.set_title("Corrective gain by error family", loc="left")
    fig.tight_layout()
    return save(fig, out)[0]


# --------------------------------------------------------------------------
# End-to-end study
# --------------------------------------------------------------------------

HARNESS_ORDER = ["verbatim", "verbatim+instr", "drop", "abstract",
                 "verbatim+ban", "abstract+ban"]
HARNESS_SHORT = {
    "verbatim": "verbatim",
    "verbatim+instr": "+instruction",
    "drop": "drop",
    "abstract": "abstract",
    "verbatim+ban": "verbatim+ban",
    "abstract+ban": "abstract+ban",
}


def fig_agent(summary: pd.DataFrame, out: str) -> Optional[str]:
    """Task success and repetition by harness, one series per model."""
    if summary.empty:
        return None
    models = list(dict.fromkeys(summary.sort_values("params_b")["model"]))
    harnesses = [h for h in HARNESS_ORDER if h in set(summary["harness"])]
    if len(harnesses) < 2:
        return None
    panels = [("solved", "Task success", "fraction of tasks")]
    if "stuck" in set(summary["metric"]):
        panels.append(("stuck", "Ended in a loop", "fraction of rollouts"))
    panels.append(("exact_repeat_rate", "Exact repeat rate",
                   "fraction of failed actions"))
    # Six harness names per panel, rotated, across three panels: at the
    # single-column height this was a row of overlapping smudges. The figure is
    # a `figure*`, so it gets the full text width and enough height for the
    # labels to be read.
    fig, axes = plt.subplots(1, len(panels), figsize=(FULL_WIDTH, 3.1))
    axes = np.atleast_1d(axes)
    palette = [BLUE, ORANGE, AQUA, VIOLET]

    for ax, (metric, title, ylab) in zip(axes, panels):
        series = []
        for i, m in enumerate(models):
            vals, cis = [], []
            for h in harnesses:
                r = summary[(summary["model"] == m) & (summary["harness"] == h)
                            & (summary["metric"] == metric)]
                vals.append(float(r["mean"].iloc[0]) if len(r) else np.nan)
                cis.append((float(r["ci_lo"].iloc[0]), float(r["ci_hi"].iloc[0]))
                           if len(r) else (np.nan, np.nan))
            label = summary[summary["model"] == m]["label"].iloc[0]
            series.append((label, vals, cis, palette[i % len(palette)]))
        grouped_bars(ax, [HARNESS_SHORT.get(h, h) for h in harnesses], series)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylab)
        ax.tick_params(axis="x", rotation=32)
        for t in ax.get_xticklabels():
            t.set_ha("right")
        ax.grid(axis="y", alpha=0.55)
        ax.set_axisbelow(True)
        # A zero bar is invisible; say so rather than leaving a blank slot.
        for h, v in zip(harnesses, series[0][1] if len(series) == 1 else []):
            if v == 0:
                ax.annotate("0", xy=(harnesses.index(h), 0), xytext=(0, 3),
                            textcoords="offset points", ha="center",
                            fontsize=7, color=MUTED)
    if len(models) > 1:
        axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, 1.28),
                       ncol=min(4, len(models)), handlelength=1.1,
                       columnspacing=1.3)
    fig.tight_layout(w_pad=1.5)
    return save(fig, out)[0]


def fig_correlation(probe: pd.DataFrame, agent: pd.DataFrame, out: str) -> Optional[str]:
    """Does the cheap probe predict the expensive rollout?"""
    g = probe[(probe["measure"] == "G") & (probe["env"] == "toolshed")]
    rep = agent[(agent["metric"] == "exact_repeat_rate")
                & (agent["harness"] == "verbatim")]
    if g.empty or rep.empty:
        return None
    merged = g[["model", "label", "params_b", "mean"]].merge(
        rep[["model", "mean"]], on="model", suffixes=("_G", "_rep")
    )
    if len(merged) < 3:
        return None
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.3))
    ax.scatter(merged["mean_G"], merged["mean_rep"], s=34, color=BLUE,
               edgecolor="white", linewidth=0.6, zorder=3)
    for _, r in merged.iterrows():
        direct_label(ax, r["mean_G"], r["mean_rep"], r["label"], MUTED, dx=0.4)
    if len(merged) >= 3:
        from slmecho.stats import weighted_least_squares

        slope, intercept, r2 = weighted_least_squares(merged["mean_G"],
                                                      merged["mean_rep"])
        xs = np.linspace(merged["mean_G"].min(), merged["mean_G"].max(), 20)
        ax.plot(xs, slope * xs + intercept, "-", color=GREY, linewidth=1.0,
                zorder=1)
        ax.annotate(f"$R^2$ = {r2:.2f}", xy=(0.04, 0.9),
                    xycoords="axes fraction", fontsize=7, color=MUTED)
    ax.set_xlabel("corrective gain $G$ (nats, probe)")
    ax.set_ylabel("exact repeat rate\n(free-running rollouts)")
    ax.grid(alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return save(fig, out)[0]


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="results/processed")
    ap.add_argument("--out-dir", default="paper/figures")
    args = ap.parse_args()

    use_paper_style()
    os.makedirs(args.out_dir, exist_ok=True)
    made: List[str] = []

    summary_path = os.path.join(args.processed, "probe_summary.csv")
    if os.path.exists(summary_path):
        summary = _published(pd.read_csv(summary_path))
        for fn, name in (
            (lambda: fig_scaling(summary, os.path.join(args.out_dir, "fig2_scaling")), "scaling"),
            (lambda: fig_decomposition(summary, os.path.join(args.out_dir, "fig3_decomposition")), "decomposition"),
            (lambda: fig_repeat(summary, os.path.join(args.out_dir, "fig4_repeat")), "repeat"),
            (lambda: fig_manipulations(summary, os.path.join(args.out_dir, "fig5_manipulations")), "manipulations"),
        ):
            try:
                p = fn()
                if p:
                    made.append(p)
                else:
                    print(f"[skip] {name}: not enough data yet")
            except Exception as exc:  # noqa: BLE001
                print(f"[fail] {name}: {type(exc).__name__}: {exc}")

    agent_path = os.path.join(args.processed, "agent_summary.csv")
    if os.path.exists(agent_path):
        agent = pd.read_csv(agent_path)
        try:
            p = fig_agent(agent, os.path.join(args.out_dir, "fig7_agent"))
            if p:
                made.append(p)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] agent: {type(exc).__name__}: {exc}")
        if os.path.exists(summary_path):
            try:
                p = fig_correlation(_published(pd.read_csv(summary_path)), agent,
                                    os.path.join(args.out_dir, "fig8_correlation"))
                if p:
                    made.append(p)
            except Exception as exc:  # noqa: BLE001
                print(f"[fail] correlation: {type(exc).__name__}: {exc}")

    err_path = os.path.join(args.processed, "probe_by_error_type.csv")
    if os.path.exists(err_path):
        try:
            # The breakdown CSV has no `published` column of its own; take the
            # model/env pairs from the summary that does.
            be = pd.read_csv(err_path)
            if os.path.exists(summary_path):
                pub = _published(pd.read_csv(summary_path))
                ok = set(zip(pub["model"], pub["env"]))
                be = be[[(m, e) in ok for m, e in zip(be["model"], be["env"])]]
            p = fig_error_types(be,
                                os.path.join(args.out_dir, "fig6_error_types"))
            if p:
                made.append(p)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] error_types: {type(exc).__name__}: {exc}")

    for p in made:
        print("wrote", p)


if __name__ == "__main__":
    main()
