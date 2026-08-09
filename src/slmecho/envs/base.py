"""Shared types for executable environments.

An environment supplies three things to the rest of the codebase:

1. **Task instances** — a natural-language goal plus a reset-able world state.
2. **A step function** — given an action *string*, actually execute it and
   return an `Outcome` describing what happened. Crucially the environment, not
   the model, decides whether an action failed; no LLM judge is involved
   anywhere in this project.
3. **Counterfactual observations** — for the probe study we need, for one and
   the same action string, an observation that reports failure, one that
   reports success, and one that is feedback-neutral. Only the environment
   knows how to render those consistently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class Outcome:
    """Result of executing one action string against a world state."""

    ok: bool
    observation: str
    #: Coarse error family (``None`` when ``ok``). Used for per-category
    #: breakdowns in the analysis, never shown to the model.
    error_type: Optional[str] = None
    #: Whether the rendered observation contains the action string verbatim
    #: (e.g. a Python traceback echoing the offending line). This is a key
    #: manipulated variable in Study 2.
    echoes_action: bool = False
    #: Free-form extras for logging.
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Step:
    """One (action, outcome) pair in a trajectory."""

    action: str
    outcome: Outcome


@dataclass
class Task:
    """A single environment task instance."""

    task_id: str
    #: Natural-language goal shown to the model.
    goal: str
    #: Environment-specific payload needed to rebuild the world.
    payload: Dict[str, Any] = field(default_factory=dict)
    #: A reference solution as a list of action strings (used for the probe's
    #: "correct action" and for computing oracle-step counts; never shown to
    #: the model).
    gold_actions: List[str] = field(default_factory=list)
    #: Difficulty bucket, for stratified reporting.
    difficulty: str = "medium"


class Environment(Protocol):
    """Minimal protocol implemented by every environment in `slmecho.envs`."""

    name: str

    def system_prompt(self, task: Task) -> str:
        """Static instructions + tool/API documentation for this task."""

    def reset(self, task: Task) -> None:
        """Restore world state to the task's initial condition."""

    def step(self, task: Task, action: str) -> Outcome:
        """Execute `action`, mutating world state. Must be deterministic."""

    def is_solved(self, task: Task) -> bool:
        """Whether the goal condition currently holds."""

    def neutral_observation(self, action: str, failure: Outcome) -> str:
        """A feedback-neutral observation of the same shape/length class."""

    def success_observation(self, action: str, failure: Outcome) -> str:
        """A counterfactual observation reporting that `action` succeeded."""
