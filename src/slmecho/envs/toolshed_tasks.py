"""Task generation and action perturbation for ToolShed.

Two jobs live here:

1. `generate_tasks` instantiates the six task templates against seeded worlds
   and records a reference (gold) action sequence for each.
2. `perturbations` turns a *correct* action into the kinds of *plausible wrong*
   actions small models actually emit — a dropped required argument, a
   pluralised tool name, a date written out in words, a hallucinated contact.
   Each perturbation is tagged with the error family it will provoke, which is
   what lets us report the probe results broken down by error type.

The perturbations are written by hand rather than sampled from model outputs on
purpose: we need the *same* wrong action to be usable across every model in the
scaling ladder, otherwise the item set differs per model and the scaling curve
confounds item difficulty with model size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..seeding import rng_for
from .base import Task
from .toolshed import TOOLS_BY_NAME, World, build_world, parse_action

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


# --------------------------------------------------------------------------
# Task generation
# --------------------------------------------------------------------------


def _fmt_call(name: str, kwargs: Dict[str, Any]) -> str:
    parts = []
    for k, v in kwargs.items():
        parts.append(f"{k}={v!r}")
    return f"{name}({', '.join(parts)})"


def generate_tasks(n: int, seed: int = 20260808) -> List[Task]:
    """Instantiate `n` tasks, cycling through the six templates."""
    templates = [
        _t_email_file,
        _t_cancel_event,
        _t_schedule,
        _t_roster,
        _t_archive,
        _t_convert,
    ]
    tasks: List[Task] = []
    for i in range(n):
        world = build_world(seed, i)
        rng = rng_for(seed, f"toolshed-task-{i}")
        task = templates[i % len(templates)](i, world, rng, seed)
        tasks.append(task)
    return tasks


def _t_email_file(i: int, world: World, rng, seed: int) -> Task:
    contact = rng.choice(world.contacts)
    path = rng.choice([p for p in world.files if not world.files[p]["readonly"]])
    content = world.files[path]["content"]
    stem = path.split("/")[-1].split(".")[0]
    gold = [
        _fmt_call("find_contact", {"name": contact["name"]}),
        _fmt_call("read_file", {"path": path}),
        _fmt_call(
            "send_message",
            {"to": contact["email"], "subject": stem, "body": content},
        ),
        "finish()",
    ]
    return Task(
        task_id=f"toolshed-{i:04d}",
        goal=(
            f"Send {contact['name']} the contents of {path}. "
            f"Use '{stem}' as the subject."
        ),
        payload={
            "template": "email_file",
            "world_seed": seed,
            "world_index": i,
            "contact_email": contact["email"],
            "file_content": content,
        },
        gold_actions=gold,
        difficulty="medium",
    )


def _t_cancel_event(i: int, world: World, rng, seed: int) -> Task:
    event = rng.choice(world.events)
    gold = [
        _fmt_call("list_events", {"date": event["date"]}),
        _fmt_call("cancel_event", {"event_id": event["id"]}),
        "finish()",
    ]
    return Task(
        task_id=f"toolshed-{i:04d}",
        goal=(
            f"Cancel the '{event['title']}' meeting scheduled on "
            f"{event['date']}."
        ),
        payload={
            "template": "cancel_event",
            "world_seed": seed,
            "world_index": i,
            "event_id": event["id"],
        },
        gold_actions=gold,
        difficulty="easy",
    )


def _t_schedule(i: int, world: World, rng, seed: int) -> Task:
    contact = rng.choice(world.contacts)
    date = f"2026-10-{rng.randint(10, 26):02d}"
    time = f"{rng.choice([9, 11, 13, 16]):02d}:30"
    gold = [
        _fmt_call("find_contact", {"name": contact["name"]}),
        _fmt_call(
            "create_event",
            {
                "title": "1:1",
                "date": date,
                "time": time,
                "attendees": [contact["email"]],
            },
        ),
        "finish()",
    ]
    return Task(
        task_id=f"toolshed-{i:04d}",
        goal=(
            f"Schedule a 1:1 with {contact['name']} on {date} at {time}. "
            f"Title the event '1:1'."
        ),
        payload={
            "template": "schedule",
            "world_seed": seed,
            "world_index": i,
            "contact_email": contact["email"],
            "date": date,
            "time": time,
        },
        gold_actions=gold,
        difficulty="medium",
    )


def _t_roster(i: int, world: World, rng, seed: int) -> Task:
    team = rng.choice(sorted({c["team"] for c in world.contacts}))
    members = [c["name"] for c in world.contacts if c["team"] == team]
    out_path = "reports/roster.md"
    gold = [
        _fmt_call("list_contacts", {"team": team}),
        _fmt_call("write_file", {"path": out_path, "content": ", ".join(members)}),
        "finish()",
    ]
    return Task(
        task_id=f"toolshed-{i:04d}",
        goal=(
            f"Write the names of everyone on the {team} team, comma separated, "
            f"to {out_path}."
        ),
        payload={
            "template": "roster",
            "world_seed": seed,
            "world_index": i,
            "out_path": out_path,
            "team_names": members,
        },
        gold_actions=gold,
        difficulty="medium",
    )


def _t_archive(i: int, world: World, rng, seed: int) -> Task:
    path = rng.choice([p for p in world.files if not world.files[p]["readonly"]])
    date = f"2026-11-{rng.randint(3, 27):02d}"
    gold = [
        _fmt_call("delete_file", {"path": path}),
        _fmt_call("set_reminder", {"text": f"{path} was archived", "date": date}),
        "finish()",
    ]
    return Task(
        task_id=f"toolshed-{i:04d}",
        goal=(
            f"Delete {path}, then set a reminder for {date} saying "
            f"'{path} was archived'."
        ),
        payload={
            "template": "archive",
            "world_seed": seed,
            "world_index": i,
            "path": path,
            "date": date,
        },
        gold_actions=gold,
        difficulty="easy",
    )


def _t_convert(i: int, world: World, rng, seed: int) -> Task:
    pairs = [("km", "mi"), ("kg", "lb"), ("l", "gal")]
    src, dst = rng.choice(pairs)
    # Kept as an int so the reference call reads `value=59`, which is what a
    # competent model would actually emit; `value=59.0` would be an unfair
    # target string in the probe.
    value = rng.randint(5, 90)
    from .toolshed import UNIT_TABLE

    expected = round(value * UNIT_TABLE[(src, dst)], 4)
    out_path = "reports/conversion.md"
    gold = [
        _fmt_call(
            "convert_units", {"value": value, "from_unit": src, "to_unit": dst}
        ),
        _fmt_call("write_file", {"path": out_path, "content": str(expected)}),
        "finish()",
    ]
    return Task(
        task_id=f"toolshed-{i:04d}",
        goal=(
            f"Convert {value:g} {src} to {dst} and write only the resulting "
            f"number to {out_path}."
        ),
        payload={
            "template": "convert",
            "world_seed": seed,
            "world_index": i,
            "out_path": out_path,
            "expected": expected,
        },
        gold_actions=gold,
        difficulty="hard",
    )


# --------------------------------------------------------------------------
# Perturbations: correct action -> plausible wrong action
# --------------------------------------------------------------------------


@dataclass
class Perturbation:
    """A wrong-but-plausible variant of a gold action."""

    action: str
    operator: str
    expected_error: str


_TOOL_TYPOS = {
    "find_contact": "find_contacts",
    "read_file": "read_files",
    "write_file": "save_file",
    "delete_file": "remove_file",
    "list_files": "list_file",
    "list_contacts": "list_contact",
    "list_events": "list_event",
    "create_event": "add_event",
    "cancel_event": "delete_event",
    "send_message": "send_email",
    "set_reminder": "create_reminder",
    "convert_units": "convert_unit",
}

_ARG_TYPOS = {
    "path": "file_path",
    "folder": "directory",
    "name": "contact_name",
    "team": "team_name",
    "date": "day",
    "time": "start_time",
    "attendees": "participants",
    "event_id": "id",
    "to": "recipient",
    "subject": "title",
    "body": "message",
    "text": "note",
    "value": "amount",
    "from_unit": "source_unit",
    "to_unit": "target_unit",
    "content": "text",
}


def perturbations(action: str, world: World) -> List[Perturbation]:
    """All applicable wrong variants of `action`, in a stable order."""
    if action.strip() == "finish()":
        return []
    parsed = parse_action(action)
    spec = TOOLS_BY_NAME.get(parsed.name)
    if spec is None:
        return []
    kw = dict(parsed.kwargs)
    out: List[Perturbation] = []

    # 1. Drop a required argument (the last one, deterministically).
    required = [a.name for a in spec.args if a.required and a.name in kw]
    if len(required) > 1:
        dropped = dict(kw)
        dropped.pop(required[-1])
        out.append(
            Perturbation(
                _fmt_call(spec.name, dropped), "drop_arg", "missing_argument"
            )
        )

    # 2. Misspell the tool name.
    if spec.name in _TOOL_TYPOS:
        out.append(
            Perturbation(
                _fmt_call(_TOOL_TYPOS[spec.name], kw), "tool_typo", "unknown_tool"
            )
        )

    # 3. Rename an argument to a plausible synonym.
    for arg_name in list(kw):
        if arg_name in _ARG_TYPOS:
            renamed = {
                (_ARG_TYPOS[k] if k == arg_name else k): v for k, v in kw.items()
            }
            out.append(
                Perturbation(
                    _fmt_call(spec.name, renamed),
                    "arg_typo",
                    "unexpected_argument",
                )
            )
            break

    # 4. Type confusion: list -> string, number -> string.
    for arg in spec.args:
        if arg.name not in kw:
            continue
        if arg.type == "list[str]" and isinstance(kw[arg.name], list) and kw[arg.name]:
            swapped = dict(kw)
            swapped[arg.name] = kw[arg.name][0]
            out.append(
                Perturbation(
                    _fmt_call(spec.name, swapped), "type_swap", "type_error"
                )
            )
            break
        if arg.type in {"int", "float"} and isinstance(kw[arg.name], (int, float)):
            swapped = dict(kw)
            swapped[arg.name] = f"{kw[arg.name]:g}"
            out.append(
                Perturbation(
                    _fmt_call(spec.name, swapped), "type_swap", "type_error"
                )
            )
            break

    # 5. Date written the way a person would write it.
    for arg in spec.args:
        if arg.name in {"date"} and isinstance(kw.get(arg.name), str):
            m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", kw[arg.name])
            if m:
                y, mo, d = m.groups()
                pretty = f"{MONTHS[int(mo) - 1]} {int(d)}, {y}"
                swapped = dict(kw)
                swapped[arg.name] = pretty
                out.append(
                    Perturbation(
                        _fmt_call(spec.name, swapped), "date_format", "format_error"
                    )
                )
            break

    # 6. Hallucinated entity: a referent that does not exist in this world.
    hall = _hallucinate(spec.name, kw, world)
    if hall is not None:
        out.append(hall)

    return out


def _hallucinate(
    tool: str, kw: Dict[str, Any], world: World
) -> Optional[Perturbation]:
    swapped = dict(kw)
    if tool == "find_contact":
        swapped["name"] = "Alex Ramirez"
        return Perturbation(
            _fmt_call(tool, swapped), "hallucinated_entity", "contact_not_found"
        )
    if tool in {"read_file", "delete_file"}:
        swapped["path"] = "notes/meeting_notes.md"
        if swapped["path"] in world.files:
            swapped["path"] = "notes/archive_2025.md"
        return Perturbation(
            _fmt_call(tool, swapped), "hallucinated_entity", "file_not_found"
        )
    if tool == "cancel_event":
        swapped["event_id"] = "evt-9"
        return Perturbation(
            _fmt_call(tool, swapped), "hallucinated_entity", "event_not_found"
        )
    if tool == "send_message":
        # The classic small-model mistake: pass the person's name, not their
        # address, because the name is what the task instruction contained.
        email = str(kw.get("to", ""))
        for c in world.contacts:
            if c["email"] == email:
                swapped["to"] = c["name"]
                return Perturbation(
                    _fmt_call(tool, swapped), "name_for_email", "format_error"
                )
    if tool == "list_contacts":
        swapped["team"] = "marketing"
        return Perturbation(
            _fmt_call(tool, swapped), "hallucinated_entity", "team_not_found"
        )
    if tool == "list_files":
        swapped["folder"] = "documents"
        return Perturbation(
            _fmt_call(tool, swapped), "hallucinated_entity", "folder_not_found"
        )
    if tool == "convert_units":
        swapped["to_unit"] = "m"
        return Perturbation(
            _fmt_call(tool, swapped), "hallucinated_entity", "unsupported_conversion"
        )
    if tool == "create_event":
        attendees = kw.get("attendees")
        if isinstance(attendees, list) and attendees:
            swapped["attendees"] = ["alex.ramirez@corp.example"]
            return Perturbation(
                _fmt_call(tool, swapped),
                "hallucinated_entity",
                "attendee_not_found",
            )
    if tool == "write_file":
        swapped["path"] = "system/config.ini"
        return Perturbation(
            _fmt_call(tool, swapped), "readonly_target", "permission_error"
        )
    if tool == "set_reminder":
        swapped["date"] = "next Tuesday"
        return Perturbation(
            _fmt_call(tool, swapped), "date_format", "format_error"
        )
    if tool == "list_events":
        swapped["date"] = "tomorrow"
        return Perturbation(
            _fmt_call(tool, swapped), "date_format", "format_error"
        )
    return None


# --------------------------------------------------------------------------
# Task mix for the free-running agent study
# --------------------------------------------------------------------------
#
# The probe is teacher-forced, so it does not matter there whether a model could
# have produced the reference trajectory on its own. The agent study is
# different: if every task needs three chained calls, the smallest models score
# zero everywhere and no harness comparison is possible on a floor. We therefore
# add three single-call templates and interleave them with the standard ones.
# `generate_tasks` is deliberately left untouched so the probe item set built
# from it stays byte-identical.


def _t_lookup(i: int, world: World, rng, seed: int) -> Task:
    contact = rng.choice(world.contacts)
    return Task(
        task_id=f"agent-{i:04d}",
        goal=f"Look up the contact record for {contact['name']}.",
        payload={
            "template": "lookup",
            "world_seed": seed,
            "world_index": i,
            "contact_name": contact["name"],
        },
        gold_actions=[
            _fmt_call("find_contact", {"name": contact["name"]}),
            "finish()",
        ],
        difficulty="easy",
    )


def _t_simple_write(i: int, world: World, rng, seed: int) -> Task:
    text = rng.choice(
        ["backups verified", "keys rotated", "invoice sent", "release tagged"]
    )
    out_path = "notes/status.md"
    return Task(
        task_id=f"agent-{i:04d}",
        goal=f"Create the file {out_path} containing exactly: {text}",
        payload={
            "template": "simple_write",
            "world_seed": seed,
            "world_index": i,
            "out_path": out_path,
            "text": text,
        },
        gold_actions=[
            _fmt_call("write_file", {"path": out_path, "content": text}),
            "finish()",
        ],
        difficulty="easy",
    )


def _t_simple_reminder(i: int, world: World, rng, seed: int) -> Task:
    date = f"2026-12-{rng.randint(3, 27):02d}"
    text = rng.choice(["renew the domain", "file the expenses", "review the roadmap"])
    return Task(
        task_id=f"agent-{i:04d}",
        goal=f"Set a reminder for {date} that says: {text}",
        payload={
            "template": "simple_reminder",
            "world_seed": seed,
            "world_index": i,
            "date": date,
            "text": text,
        },
        gold_actions=[
            _fmt_call("set_reminder", {"text": text, "date": date}),
            "finish()",
        ],
        difficulty="easy",
    )


AGENT_TEMPLATES = [
    _t_lookup,
    _t_cancel_event,
    _t_simple_write,
    _t_archive,
    _t_simple_reminder,
    _t_schedule,
    _t_lookup,
    _t_roster,
    _t_simple_write,
    _t_email_file,
    _t_simple_reminder,
    _t_convert,
]


def generate_agent_tasks(n: int, seed: int = 20260808) -> List[Task]:
    """Interleaved easy/standard tasks for the end-to-end study."""
    tasks: List[Task] = []
    for i in range(n):
        world = build_world(seed, i)
        rng = rng_for(seed, f"agent-task-{i}")
        t = AGENT_TEMPLATES[i % len(AGENT_TEMPLATES)](i, world, rng, seed)
        t.task_id = f"agent-{i:04d}"
        tasks.append(t)
    return tasks
