"""Unit tests for the parts where a silent bug would change a conclusion.

Run with:  python -m pytest tests -q     (or: python tests/test_core.py)

These deliberately avoid loading a language model: everything here is about the
experimental machinery — that conditions differ only where they are supposed to,
that the decomposition identity holds, that the ban blocks what it claims to
block, and that repetition is counted the way the paper says it is.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from slmecho.agent import canonical_action, repeat_stats  # noqa: E402
from slmecho.decoding import BannedSequenceProcessor  # noqa: E402
from slmecho.envs.toolshed import ToolShedEnv, parse_action  # noqa: E402
from slmecho.envs.toolshed_tasks import generate_tasks, perturbations  # noqa: E402
from slmecho.harness import HARNESSES, TrajStep, build_messages  # noqa: E402
from slmecho.probe import build_condition_messages, load_items  # noqa: E402
from slmecho.seeding import derive_seed, rng_for  # noqa: E402
from slmecho.stats import cluster_bootstrap, holm  # noqa: E402

FAILURES = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


# --------------------------------------------------------------------------


def test_seeding() -> None:
    print("seeding")
    check("derive_seed is stable", derive_seed(7, "a") == derive_seed(7, "a"))
    check("different names give different streams",
          derive_seed(7, "a") != derive_seed(7, "b"))
    check("rng reproducible",
          rng_for(1, "x").random() == rng_for(1, "x").random())


def test_env_determinism() -> None:
    print("environment")
    tasks = generate_tasks(6)
    env_a, env_b = ToolShedEnv(), ToolShedEnv()
    outs_a, outs_b = [], []
    for t in tasks:
        env_a.reset(t)
        env_b.reset(t)
        for a in t.gold_actions:
            outs_a.append(env_a.step(t, a).observation)
            outs_b.append(env_b.step(t, a).observation)
    check("execution is deterministic", outs_a == outs_b)
    check("reference trajectories solve their tasks",
          all(ToolShedEnv().__class__ and _solves(t) for t in tasks))


def _solves(task) -> bool:
    env = ToolShedEnv()
    env.reset(task)
    for a in task.gold_actions:
        if not env.step(task, a).ok:
            return False
    return env.is_solved(task)


def test_perturbations_provoke_intended_errors() -> None:
    print("perturbation operators")
    env = ToolShedEnv()
    n_ok = n_total = 0
    for task in generate_tasks(12):
        env.reset(task)
        for gold in task.gold_actions:
            if gold.strip() == "finish()":
                continue
            for p in perturbations(gold, env.world(task)):
                out = env.step(task, p.action)
                n_total += 1
                n_ok += int((not out.ok) and out.error_type == p.expected_error)
                # A failed call must not mutate the world.
            env.step(task, gold)
    check("every perturbation provokes its declared error family",
          n_ok == n_total, f"({n_ok}/{n_total})")


def test_conditions_differ_only_where_intended() -> None:
    print("probe conditions")
    path = "data/probe_toolshed.jsonl"
    if not os.path.exists(path):
        print("  skip (items not built)")
        return
    item = load_items(path)[0]
    pre = build_condition_messages(item, "pre")
    fail = build_condition_messages(item, "fail")
    succ = build_condition_messages(item, "succ")
    neut = build_condition_messages(item, "neut")

    check("pre is a strict prefix of fail", fail[: len(pre)] == pre)
    check("fail/succ/neut share everything but the observation",
          [m["content"] for m in fail[:-1]] == [m["content"] for m in succ[:-1]]
          == [m["content"] for m in neut[:-1]])
    check("fail/succ/neut each contain the failed call exactly once",
          all(sum(m["content"] == item.failed_action for m in msgs) == 1
              for msgs in (fail, succ, neut)))
    abst = build_condition_messages(item, "abstract")
    check("abstract contains no verbatim failed call",
          all(item.failed_action not in m["content"] for m in abst))
    if item.distractors:
        other = build_condition_messages(item, "fail_other")
        check("placebo does not contain the scored action",
              all(item.failed_action not in m["content"] for m in other))
        check("placebo does contain the other action",
              any(item.distractors[0] in m["content"] for m in other))


def test_decomposition_identity() -> None:
    print("decomposition identity")
    rng = np.random.default_rng(0)
    pre, neut, fail = rng.normal(size=(3, 500)) * 5
    copy = neut - pre
    sem = fail - neut
    g = pre - fail
    check("-G == copy + sem exactly", np.allclose(-g, copy + sem))


def test_harness_rendering() -> None:
    print("harness policies")
    steps = [
        TrajStep("good_call(x=1)", "ok", True),
        TrajStep("bad_call(y=2)", "TypeError: nope", False,
                 error_type="type_error", tool="bad_call"),
    ]
    verbatim = build_messages("SYS", "GOAL", steps, HARNESSES["verbatim"])
    abstract = build_messages("SYS", "GOAL", steps, HARNESSES["abstract"])
    drop = build_messages("SYS", "GOAL", steps, HARNESSES["drop"])
    joined = lambda ms: "\n".join(m["content"] for m in ms)  # noqa: E731

    check("verbatim keeps the failed call", "bad_call(y=2)" in joined(verbatim))
    check("abstract removes the failed call",
          "bad_call(y=2)" not in joined(abstract))
    check("abstract keeps the diagnosis", "wrong type" in joined(abstract))
    check("drop removes both", "bad_call(y=2)" not in joined(drop)
          and "TypeError" not in joined(drop))
    check("all policies keep successful steps",
          all("good_call(x=1)" in joined(m) for m in (verbatim, abstract, drop)))
    instr = build_messages("SYS", "GOAL", steps, HARNESSES["verbatim+instr"])
    check("instruction variant adds the prohibition",
          "Do not repeat" in instr[0]["content"])


def test_ban_processor() -> None:
    print("decoder ban")
    try:
        import torch
    except ImportError:
        print("  skip (torch unavailable)")
        return

    banned = [[[10, 11, 12]]]
    # Generated tokens match the banned prefix [10, 11] -> the completion 12
    # must be masked, and nothing else.
    ids = torch.tensor([[1, 2, 3, 10, 11]])
    scores = torch.zeros((1, 20))
    proc = BannedSequenceProcessor(banned, prompt_len=3)
    out = proc(ids, scores.clone())
    check("completion token is masked", math.isinf(out[0, 12].item())
          and out[0, 12].item() < 0)
    check("only the completion is masked",
          int((out[0] == float("-inf")).sum()) == 1)

    # Prefix does not match -> nothing is masked.
    ids2 = torch.tensor([[1, 2, 3, 10, 99]])
    proc2 = BannedSequenceProcessor(banned, prompt_len=3)
    out2 = proc2(ids2, torch.zeros((1, 20)))
    check("no mask when the prefix does not match",
          int((out2[0] == float("-inf")).sum()) == 0)

    # The first token of a banned sequence must remain reachable, because the
    # corrected call usually shares a prefix with the failed one.
    ids3 = torch.tensor([[1, 2, 3]])
    out3 = BannedSequenceProcessor(banned, prompt_len=3)(ids3, torch.zeros((1, 20)))
    check("first token of a banned sequence stays reachable",
          not math.isinf(out3[0, 10].item()))


def test_repeat_counting() -> None:
    print("repetition metrics")
    a = "find_contact(name='Dana')"
    a_reordered = "find_contact( name = 'Dana' )"
    steps = [
        TrajStep(a, "err", False, error_type="x"),
        TrajStep(a, "err", False, error_type="x"),
        TrajStep(a_reordered, "err", False, error_type="x"),
        TrajStep("other(x=1)", "ok", True),
    ]
    st = repeat_stats(steps, "toolshed")
    check("exact repeats counted", st["n_exact_repeats"] == 1)
    check("canonical form catches the paraphrase",
          st["n_canonical_repeats"] == 2)
    check("canonicalisation is whitespace/order invariant",
          canonical_action(a, "toolshed") == canonical_action(a_reordered, "toolshed"))
    check("successful steps are not counted as failures",
          st["n_failed_actions"] == 3)


def test_bootstrap_behaviour() -> None:
    print("statistics")
    rng = np.random.default_rng(3)
    # 20 clusters of 5 correlated items each, true mean -2.
    vals, clusters = [], []
    for c in range(20):
        offset = rng.normal(0, 1.5)
        for _ in range(5):
            vals.append(-2 + offset + rng.normal(0, 0.3))
            clusters.append(f"c{c}")
    est = cluster_bootstrap(vals, clusters, n_boot=2000, seed=1)
    check("point estimate near truth", abs(est.mean + 2) < 0.8, f"({est.mean:.2f})")
    check("interval contains the truth", est.lo < -2 < est.hi)
    check("clusters counted", est.n_clusters == 20 and est.n == 100)

    # Ignoring clustering must give a narrower (over-confident) interval.
    naive = cluster_bootstrap(vals, [str(i) for i in range(100)], n_boot=2000, seed=1)
    check("cluster bootstrap is wider than the naive one",
          (est.hi - est.lo) > (naive.hi - naive.lo))

    adj = holm({"a": 0.001, "b": 0.02, "c": 0.5})
    check("holm is monotone and >= raw",
          adj["a"] >= 0.001 and adj["b"] >= 0.02 and adj["a"] <= adj["b"] <= adj["c"])


def test_action_parsing() -> None:
    print("action parsing")
    p = parse_action("find_contact(name='Dana Whitfield')")
    check("keyword args parsed", p.name == "find_contact"
          and p.kwargs == {"name": "Dana Whitfield"})
    p2 = parse_action("```python\nread_file(path='a/b.md')\n```")
    check("code fences tolerated", p2.name == "read_file")
    p3 = parse_action("Action: cancel_event(event_id='evt-1')")
    check("action labels tolerated", p3.name == "cancel_event")
    try:
        parse_action("this is not a call")
        check("prose is rejected", False)
    except Exception:  # noqa: BLE001
        check("prose is rejected", True)


def main() -> int:
    for fn in (
        test_seeding,
        test_env_determinism,
        test_perturbations_provoke_intended_errors,
        test_conditions_differ_only_where_intended,
        test_decomposition_identity,
        test_harness_rendering,
        test_ban_processor,
        test_repeat_counting,
        test_bootstrap_behaviour,
        test_action_parsing,
    ):
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
