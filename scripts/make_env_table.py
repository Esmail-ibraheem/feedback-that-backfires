"""Environment statistics table + the small helper facts the prose needs."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from slmecho import tables  # noqa: E402
from slmecho.probe import load_items  # noqa: E402


def main() -> None:
    rows = []
    for env, path, desc in (
        ("ToolShed", "data/probe_toolshed.jsonl", "tool calls"),
        ("CodeRepair", "data/probe_coderepair.jsonl", "MBPP programs"),
    ):
        if not os.path.exists(path):
            continue
        items = load_items(path)
        n_tasks = len({i.task_id for i in items})
        ops = len({i.operator for i in items})
        errs = len({i.error_type for i in items})
        act_chars = sum(len(i.failed_action) for i in items) / max(len(items), 1)
        obs_chars = sum(len(i.obs_fail) for i in items) / max(len(items), 1)
        prefix = sum(len(i.prefix_actions) for i in items) / max(len(items), 1)
        quotes = sum(
            1 for i in items if (i.meta or {}).get("traceback_quotes_source")
        )
        rows.append(
            (env, desc, len(items), n_tasks, ops, errs, act_chars, obs_chars,
             prefix, quotes / max(len(items), 1))
        )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{l r r r r r r}",
        r"\toprule",
        r"Environment & items & tasks & ops. & err. & \makecell{action\\(chars)} "
        r"& \makecell{obs.\\(chars)} \\",
        r"\midrule",
    ]
    for env, _desc, n, t, ops, errs, ac, oc, _pf, _q in rows:
        lines.append(f"{env} & {n} & {t} & {ops} & {errs} & {ac:.0f} & {oc:.0f} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Probe item pools. \emph{ops.} is the number of perturbation "
        r"operators used to build failing actions, \emph{err.} the number of "
        r"distinct error families they provoke. Experiments draw a balanced "
        r"subsample from these pools; the drawn size is given per study.}",
        r"\label{tab:envs}",
        r"\end{table}",
    ]
    tables.write("paper/tables/env_stats.tex", "\n".join(lines))
    print("wrote paper/tables/env_stats.tex")
    for r in rows:
        print(" ", r[0], "items", r[2], "tasks", r[3], "ops", r[4], "errs", r[5],
              f"traceback-quotes {r[9]:.0%}")


if __name__ == "__main__":
    main()
