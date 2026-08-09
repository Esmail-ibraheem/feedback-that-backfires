"""CodeRepair: single-step program repair with real execution feedback.

ToolShed is deliberately synthetic so that counterfactual observations are
exactly controllable. This environment is the opposite end of the trade-off:
the tasks come from MBPP (Austin et al., 2021), the failures are produced by
actually running the code, and the observations are the interpreter's own
output. Nothing about the error text is authored by us.

It also stresses a different regime. ToolShed actions are ~15 tokens of
structured call syntax; here an action is a whole function body, and a real
Python traceback *quotes the offending source line back at the model*, which is
the most common way an agent runtime re-injects a failed action's surface form
into context without anyone deciding to do so.

The wrong programs are obtained by mutating the reference solution with a fixed
set of operators, not by sampling a model, so that every model in the ladder is
probed on exactly the same items.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import Outcome, Task

MBPP_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    "master/mbpp/sanitized-mbpp.json"
)

SYSTEM_PROMPT = (
    "You are a Python programmer. The user gives a task and the tests that "
    "will be run against your code.\n\n"
    "Rules:\n"
    "- Reply with one Python function definition and nothing else.\n"
    "- Do not include the tests, comments, explanations or markdown fences.\n"
    "- Use exactly the function name that appears in the tests."
)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


def load_mbpp(cache_dir: str) -> List[Dict[str, Any]]:
    """Fetch (once) and return the sanitized MBPP split."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "sanitized-mbpp.json")
    if not os.path.exists(path):
        import urllib.request

        with urllib.request.urlopen(MBPP_URL, timeout=120) as resp:
            data = resp.read()
        with open(path, "wb") as fh:
            fh.write(data)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

#: Executed in a fresh subprocess. Solution and tests are written to real files
#: named solution.py / tests.py so that the interpreter's traceback can quote
#: the offending source line, exactly as it would in a real coding agent. Our
#: own frames are stripped from the traceback for the same reason.
_RUNNER = r"""
import os, sys, traceback

sol_path, tests_path = sys.argv[1], sys.argv[2]
here = os.path.dirname(os.path.abspath(sol_path))


def emit(kind, passed, text=""):
    if text:
        sys.stdout.write(text.rstrip() + "\n")
    sys.stdout.write("@@KIND@@%s\n@@PASSED@@%d\n" % (kind, passed))
    sys.stdout.flush()


def fmt():
    et, ev, tb = sys.exc_info()
    entries = [e for e in traceback.extract_tb(tb)
               if os.path.abspath(e.filename).startswith(here)]
    lines = ["Traceback (most recent call last):"] if entries else []
    lines += [ln.rstrip("\n") for ln in traceback.format_list(entries)]
    lines += [ln.rstrip("\n") for ln in traceback.format_exception_only(et, ev)]
    out = "\n".join(lines)
    return out.replace(here + os.sep, "").replace(here, "").replace('"' + os.sep, '"')


ns = {"__name__": "__solution__"}
src = open(sol_path, "r", encoding="utf-8").read()
try:
    exec(compile(src, sol_path, "exec"), ns)
except SyntaxError:
    emit("syntax_error", 0, fmt()); sys.exit(1)
except Exception:
    emit("runtime_error", 0, fmt()); sys.exit(1)

tsrc = open(tests_path, "r", encoding="utf-8").read()
assert_lines = [i for i, ln in enumerate(tsrc.splitlines(), 1)
                if ln.strip().startswith("assert")]
try:
    exec(compile(tsrc, tests_path, "exec"), ns)
except AssertionError:
    _, _, tb = sys.exc_info()
    lineno = traceback.extract_tb(tb)[-1].lineno if traceback.extract_tb(tb) else 0
    passed = sum(1 for i in assert_lines if i < lineno)
    emit("test_failure", passed, fmt()); sys.exit(1)
except Exception:
    _, _, tb = sys.exc_info()
    ent = [e for e in traceback.extract_tb(tb)
           if os.path.abspath(e.filename) == os.path.abspath(tests_path)]
    lineno = ent[-1].lineno if ent else 0
    passed = sum(1 for i in assert_lines if i < lineno)
    emit("runtime_error", passed, fmt()); sys.exit(1)
emit("ok", len(assert_lines))
"""


@dataclass
class ExecResult:
    ok: bool
    kind: str
    raw: str
    passed: int
    first_failing_test: Optional[str] = None


def run_program(code: str, tests: Sequence[str], timeout: float = 10.0) -> ExecResult:
    """Execute `code` then each test, in a fresh subprocess."""
    with tempfile.TemporaryDirectory() as td:
        # The runner lives outside `td` so that its own frames are excluded by
        # the "starts with the sandbox dir" filter inside it.
        runner = os.path.join(os.path.dirname(td), f"_slmecho_runner_{os.getpid()}.py")
        sol = os.path.join(td, "solution.py")
        tst = os.path.join(td, "tests.py")
        with open(runner, "w", encoding="utf-8") as fh:
            fh.write(_RUNNER)
        with open(sol, "w", encoding="utf-8") as fh:
            fh.write(code)
        with open(tst, "w", encoding="utf-8") as fh:
            fh.write("\n".join(tests) + "\n")
        try:
            proc = subprocess.run(
                [sys.executable, runner, sol, tst],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
        except subprocess.TimeoutExpired:
            return ExecResult(False, "timeout", "TimeoutError: execution timed out", 0)

    kind = "unknown"
    m = re.search(r"@@KIND@@(\w+)", out)
    if m:
        kind = m.group(1)
    passed = 0
    m = re.search(r"@@PASSED@@(\d+)", out)
    if m:
        passed = int(m.group(1))
    clean = re.sub(r"@@(KIND|PASSED)@@\S*\n?", "", out).strip()
    failing = None
    if kind == "test_failure":
        lines = [ln for ln in clean.splitlines() if ln.strip().startswith("assert")]
        failing = lines[0].strip() if lines else None
    return ExecResult(kind == "ok", kind, clean, passed, failing)


def cleanup_runner_files() -> None:
    """Remove the helper runner scripts left in the system temp directory."""
    import glob

    for p in glob.glob(os.path.join(tempfile.gettempdir(), "_slmecho_runner_*.py")):
        try:
            os.remove(p)
        except OSError:
            pass


# --------------------------------------------------------------------------
# Observation rendering
# --------------------------------------------------------------------------


def render_observation(
    result: ExecResult,
    n_tests: int,
    verbosity: str = "standard",
    echo_action: bool = False,
    code: str = "",
) -> str:
    """Render an execution result the way an agent runtime would.

    `echo_action` here means *the whole submission is quoted back*, which some
    harnesses do. A real traceback already quotes the single offending line;
    that is part of the standard rendering and is not controlled by this flag.
    """
    if result.ok:
        body = f"All {n_tests} tests passed."
    elif result.kind == "test_failure":
        if verbosity == "terse":
            body = "Test failed."
        elif verbosity == "verbose":
            body = (
                f"{result.passed}/{n_tests} tests passed, then this assertion "
                f"failed:\n{result.first_failing_test}\nAssertionError"
            )
        else:
            body = f"{result.first_failing_test}\nAssertionError"
    elif result.kind == "timeout":
        body = "TimeoutError: execution timed out."
    else:
        raw = result.raw.strip()
        if verbosity == "terse":
            last = raw.splitlines()[-1] if raw.splitlines() else "Error"
            body = last
        elif verbosity == "verbose":
            body = raw
        else:
            # Drop the interpreter's frame bookkeeping, keep the message and
            # the quoted source line — this is what most runtimes forward.
            keep = [
                ln
                for ln in raw.splitlines()
                if not ln.startswith("Traceback (") and "most recent call last" not in ln
            ]
            body = "\n".join(keep).strip()
    if echo_action and code:
        body = f"{body}\n  submitted:\n{code}"
    return body


def neutral_observation(n_tests: int, filler_units: int = 0, key: str = "") -> str:
    base = "Submission recorded."
    if filler_units > 0:
        import hashlib

        segs = " ".join(
            "trace="
            + hashlib.blake2b(f"{key}:{i}".encode(), digest_size=4).hexdigest()
            for i in range(filler_units)
        )
        base = f"{base} {segs}"
    return base


def success_observation(n_tests: int) -> str:
    return f"All {n_tests} tests passed."


# --------------------------------------------------------------------------
# Mutation operators
# --------------------------------------------------------------------------


@dataclass
class Mutation:
    code: str
    operator: str


def _sub_once(code: str, pattern: str, repl: str) -> Optional[str]:
    new, n = re.subn(pattern, repl, code, count=1)
    return new if n == 1 and new != code else None


def mutate(code: str) -> List[Mutation]:
    """Plausible single-edit bugs, in a fixed order.

    These mirror the mistakes that actually show up in small-model code: an
    inverted comparison, a range that stops one short, a forgotten return, a
    variable typo. Each is a single token-level edit so the wrong program stays
    close to the right one — which is the interesting regime, because it is
    where the model has to *notice* the difference rather than rewrite.
    """
    out: List[Mutation] = []
    body_lines = code.splitlines()

    cand = _sub_once(code, r"(?<![<>=!])<=(?!=)", "<")
    if cand is None:
        cand = _sub_once(code, r"(?<![<>=!])<(?!=)", "<=")
    if cand:
        out.append(Mutation(cand, "flip_comparison"))

    cand = _sub_once(code, r"(?<![<>=!])>=(?!=)", ">")
    if cand is None:
        cand = _sub_once(code, r"(?<![<>=!])>(?!=)", ">=")
    if cand:
        out.append(Mutation(cand, "flip_comparison2"))

    cand = _sub_once(code, r"range\(([^()\n]*?)\)", lambda m: f"range({m.group(1)} - 1)")
    if cand:
        out.append(Mutation(cand, "off_by_one"))

    for pat, rep, op in (
        (r"\[\s*0\s*\]", "[1]", "index_shift"),
        (r"\[\s*-1\s*\]", "[0]", "index_shift"),
        (r"\[::-1\]", "[:]", "drop_reverse"),
        (r"\bTrue\b", "False", "flip_bool"),
        (r"\bnot\s+", "", "drop_negation"),
        (r"\breverse\s*=\s*True", "reverse=False", "flip_bool"),
    ):
        cand = _sub_once(code, pat, rep)
        if cand:
            out.append(Mutation(cand, op))
            break

    # Drop the last `return` -> the function silently returns None.
    for i in range(len(body_lines) - 1, -1, -1):
        stripped = body_lines[i].strip()
        if stripped.startswith("return ") and len(body_lines) > 2:
            indent = body_lines[i][: len(body_lines[i]) - len(body_lines[i].lstrip())]
            mutated = list(body_lines)
            mutated[i] = f"{indent}pass"
            out.append(Mutation("\n".join(mutated), "drop_return"))
            break

    # Rename one occurrence of a local variable -> NameError at runtime.
    names = re.findall(r"\b([a-z_][a-z0-9_]{2,})\b", code)
    seen: Dict[str, int] = {}
    for n in names:
        seen[n] = seen.get(n, 0) + 1
    for n, c in seen.items():
        if c >= 2 and n not in {"return", "range", "for", "def", "not", "and", "len"}:
            occurrences = [m.start() for m in re.finditer(rf"\b{re.escape(n)}\b", code)]
            pos = occurrences[-1]
            cand = code[:pos] + n + "_x" + code[pos + len(n) :]
            out.append(Mutation(cand, "undefined_name"))
            break

    cand = _sub_once(code, r"(?<![+\-*/=<>!])\+(?![+=])", "-")
    if cand:
        out.append(Mutation(cand, "flip_arith"))

    return out


# --------------------------------------------------------------------------
# Environment wrapper (used by the end-to-end agent study)
# --------------------------------------------------------------------------


class CodeRepairEnv:
    name = "coderepair"

    def __init__(self, verbosity: str = "standard", echo_action: bool = False,
                 timeout: float = 10.0):
        self.verbosity = verbosity
        self.echo_action = echo_action
        self.timeout = timeout

    def system_prompt(self, task: Task) -> str:
        return SYSTEM_PROMPT

    def reset(self, task: Task) -> None:  # stateless
        return None

    def user_message(self, task: Task) -> str:
        tests = "\n".join(task.payload["tests"])
        return f"{task.goal}\n\nThe tests are:\n{tests}"

    def step(self, task: Task, action: str) -> Outcome:
        code = extract_code(action)
        res = run_program(code, task.payload["tests"], timeout=self.timeout)
        n = len(task.payload["tests"])
        obs = render_observation(
            res, n, verbosity=self.verbosity, echo_action=self.echo_action, code=code
        )
        # A real traceback names the offending source line; record whether the
        # observation ended up containing a line of the submission verbatim.
        echoed = any(
            len(ln.strip()) > 8 and ln.strip() in obs for ln in code.splitlines()
        )
        return Outcome(
            ok=res.ok,
            observation=obs,
            error_type=None if res.ok else res.kind,
            echoes_action=echoed,
            info={"passed": res.passed, "n_tests": n},
        )

    def is_solved(self, task: Task) -> bool:  # decided per-step by `step`
        raise NotImplementedError


def extract_code(action: str) -> str:
    """Pull a Python program out of a model turn (fences, prose, or bare code)."""
    text = action.strip()
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)
    if fence:
        return fence.group(1).strip()
    if "```" in text:
        text = text.split("```")[-1].strip()
    return text
