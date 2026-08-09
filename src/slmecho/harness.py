"""Harness configurations: how an agent runtime turns a trajectory into context.

This module is the object of study. Everything a harness decides — whether the
failed call is echoed back verbatim, whether an explicit "do not repeat"
instruction is added, whether previously-failed calls are blocked at the
decoder — is expressed here as a `HarnessConfig`, so that the probe study and
the end-to-end agent study manipulate exactly the same knobs.

Terminology used throughout:

* **verbatim** — the standard ReAct-style transcript: the failed call appears in
  the context as an assistant turn, exactly as the model wrote it, followed by
  the error observation. This is what essentially every agent framework does.
* **abstract** — the failed call is replaced by a harness-generated description
  that preserves *what went wrong* while removing the exact token sequence
  (e.g. ``[attempt 1 failed: create_event — invalid date format]``). The
  description is produced deterministically from the runtime's own error
  metadata; no extra model call is involved.
* **drop** — the failed step is deleted from the context entirely.
* **ban** — a decoding-time constraint that makes previously-failed call
  strings unreachable. Orthogonal to the three above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

NEGATIVE_INSTRUCTION = (
    "Important: some of your earlier tool calls failed. Do not repeat a call "
    "that has already failed; change the call before trying again."
)


@dataclass
class TrajStep:
    """One executed step of an agent trajectory."""

    action: str
    observation: str
    ok: bool
    error_type: Optional[str] = None
    #: Human-readable tool name, when the action parsed.
    tool: Optional[str] = None


@dataclass
class HarnessConfig:
    name: str
    #: How failed steps are represented in context.
    failure_policy: str = "verbatim"  # verbatim | abstract | drop
    #: Append an explicit natural-language prohibition to the system prompt.
    negative_instruction: bool = False
    #: Block previously-failed action strings during decoding (Study 3 only).
    ban_failed_actions: bool = False
    #: Role used for observations. We use "user" everywhere because several of
    #: the small models in our ladder ship chat templates without a tool role,
    #: and switching roles per model would confound the comparison.
    observation_role: str = "user"
    observation_prefix: str = "Observation: "
    #: Keep at most this many trailing steps (None = keep all).
    max_history_steps: Optional[int] = None

    def describe(self) -> str:
        bits = [self.failure_policy]
        if self.negative_instruction:
            bits.append("+instruction")
        if self.ban_failed_actions:
            bits.append("+ban")
        return " ".join(bits)


#: The harness variants compared in the end-to-end study.
HARNESSES: Dict[str, HarnessConfig] = {
    "verbatim": HarnessConfig("verbatim"),
    "verbatim+instr": HarnessConfig("verbatim", negative_instruction=True),
    "abstract": HarnessConfig("abstract", failure_policy="abstract"),
    "drop": HarnessConfig("drop", failure_policy="drop"),
    "verbatim+ban": HarnessConfig("verbatim", ban_failed_actions=True),
    "abstract+ban": HarnessConfig(
        "abstract", failure_policy="abstract", ban_failed_actions=True
    ),
    "drop+ban": HarnessConfig("drop", failure_policy="drop", ban_failed_actions=True),
}
# Give every entry the key as its name (the dataclass default above repeats the
# policy string, which would make result files ambiguous).
for _key, _cfg in HARNESSES.items():
    _cfg.name = _key


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

#: Short, neutral English glosses for each error family. These are what the
#: `abstract` policy shows instead of the failed call. They deliberately carry
#: the *diagnosis* but none of the original argument values.
ERROR_GLOSS: Dict[str, str] = {
    "parse_error": "the call could not be parsed",
    "unknown_tool": "no such tool",
    "missing_argument": "a required argument was missing",
    "unexpected_argument": "an argument name was not recognised",
    "arity_error": "wrong number of arguments",
    "type_error": "an argument had the wrong type",
    "format_error": "an argument was badly formatted",
    "permission_error": "the target was read-only",
    "state_error": "the target was already in the requested state",
    "unsupported_conversion": "that conversion is not supported",
    "contact_not_found": "no such contact",
    "file_not_found": "no such file",
    "folder_not_found": "no such folder",
    "team_not_found": "no such team",
    "event_not_found": "no such event",
    "attendee_not_found": "no such attendee",
    "recipient_not_found": "no such recipient",
    "test_failure": "the code did not pass the tests",
    "runtime_error": "the code raised an exception",
    "syntax_error": "the code did not compile",
}


def abstract_failure(step: TrajStep, index: int) -> str:
    """Deterministic, surface-form-free description of a failed step."""
    gloss = ERROR_GLOSS.get(step.error_type or "", "the call failed")
    tool = step.tool or "the tool"
    return f"[attempt {index} failed: {tool} — {gloss}; that call is not repeatable]"


def build_messages(
    system_prompt: str,
    goal: str,
    steps: Sequence[TrajStep],
    config: HarnessConfig,
) -> List[Dict[str, str]]:
    """Render a trajectory into chat messages under a harness configuration."""
    system = system_prompt
    if config.negative_instruction:
        system = f"{system}\n{NEGATIVE_INSTRUCTION}"
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": goal},
    ]

    kept = list(steps)
    if config.max_history_steps is not None:
        kept = kept[-config.max_history_steps :]

    failed_seen = 0
    for step in kept:
        if step.ok:
            messages.append({"role": "assistant", "content": step.action})
            messages.append(
                {
                    "role": config.observation_role,
                    "content": f"{config.observation_prefix}{step.observation}",
                }
            )
            continue

        failed_seen += 1
        if config.failure_policy == "drop":
            continue
        if config.failure_policy == "abstract":
            messages.append(
                {
                    "role": config.observation_role,
                    "content": abstract_failure(step, failed_seen),
                }
            )
            continue
        # verbatim
        messages.append({"role": "assistant", "content": step.action})
        messages.append(
            {
                "role": config.observation_role,
                "content": f"{config.observation_prefix}{step.observation}",
            }
        )
    return messages


def failed_actions(steps: Sequence[TrajStep]) -> List[str]:
    """Action strings that have already failed — the ban list."""
    return [s.action for s in steps if not s.ok]
