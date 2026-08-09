"""ToolShed: a deterministic, executable tool-use environment.

Why a purpose-built environment rather than an off-the-shelf agent benchmark?
The central measurement in this project is a *counterfactual* one: we need, for
one fixed action string, an observation that says it failed, one that says it
succeeded, and one that says nothing at all — otherwise surface-form effects
and semantic effects cannot be separated. No public benchmark exposes that, and
faking it on top of one would require hand-writing counterfactual observations,
which is exactly the kind of place where an experimenter's expectations leak
into the data. ToolShed generates all three programmatically from the same
renderer, so the only thing that differs between conditions is the thing we
mean to vary.

Design constraints we held ourselves to:

* Actions are Python-style calls (``find_contact(name="Dana Whitfield")``),
  parsed with `ast` — no LLM judge, no regex heuristics, no partial credit.
* Every error message is produced by the same renderer used in the end-to-end
  agent runs, at three verbosity levels and with/without a verbatim echo of the
  offending call, because those are manipulated variables in Study 2.
* The world is small enough that a 0.5B model can plausibly solve tasks, which
  matters: a floor-effect benchmark measures nothing.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..seeding import rng_for
from .base import Outcome, Task

# --------------------------------------------------------------------------
# Tool schemas
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ArgSpec:
    name: str
    type: str  # "str" | "int" | "float" | "list[str]"
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args: Tuple[ArgSpec, ...]
    returns: str

    def signature(self) -> str:
        parts = []
        for a in self.args:
            parts.append(f"{a.name}: {a.type}" if a.required else f"{a.name}: {a.type} = None")
        return f"{self.name}({', '.join(parts)}) -> {self.returns}"

    def doc(self) -> str:
        lines = [f"- {self.signature()}", f"    {self.description}"]
        for a in self.args:
            flag = "" if a.required else " (optional)"
            lines.append(f"    {a.name}{flag}: {a.description}")
        return "\n".join(lines)


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_RE = re.compile(r"^\d{2}:\d{2}$")

TOOLS: Tuple[ToolSpec, ...] = (
    ToolSpec(
        "list_files",
        "List the files inside a folder.",
        (ArgSpec("folder", "str", True, "folder name, e.g. 'notes'"),),
        "list[str]",
    ),
    ToolSpec(
        "read_file",
        "Return the contents of a file.",
        (ArgSpec("path", "str", True, "full path, e.g. 'notes/todo.md'"),),
        "str",
    ),
    ToolSpec(
        "write_file",
        "Create or overwrite a file.",
        (
            ArgSpec("path", "str", True, "full path"),
            ArgSpec("content", "str", True, "text to store"),
        ),
        "str",
    ),
    ToolSpec(
        "delete_file",
        "Delete a file.",
        (ArgSpec("path", "str", True, "full path"),),
        "str",
    ),
    ToolSpec(
        "find_contact",
        "Look up one contact by full name.",
        (ArgSpec("name", "str", True, "exact full name"),),
        "dict",
    ),
    ToolSpec(
        "list_contacts",
        "List the members of a team.",
        (ArgSpec("team", "str", True, "team name, e.g. 'design'"),),
        "list[dict]",
    ),
    ToolSpec(
        "list_events",
        "List calendar events on a date.",
        (ArgSpec("date", "str", True, "date as YYYY-MM-DD"),),
        "list[dict]",
    ),
    ToolSpec(
        "create_event",
        "Add a calendar event.",
        (
            ArgSpec("title", "str", True, "event title"),
            ArgSpec("date", "str", True, "date as YYYY-MM-DD"),
            ArgSpec("time", "str", True, "start time as HH:MM"),
            ArgSpec("attendees", "list[str]", True, "list of attendee emails"),
        ),
        "str",
    ),
    ToolSpec(
        "cancel_event",
        "Cancel a calendar event by id.",
        (ArgSpec("event_id", "str", True, "event id, e.g. 'evt-4'"),),
        "str",
    ),
    ToolSpec(
        "send_message",
        "Send a message to one recipient email address.",
        (
            ArgSpec("to", "str", True, "recipient email address"),
            ArgSpec("subject", "str", True, "subject line"),
            ArgSpec("body", "str", True, "message body"),
        ),
        "str",
    ),
    ToolSpec(
        "set_reminder",
        "Store a reminder for a date.",
        (
            ArgSpec("text", "str", True, "reminder text"),
            ArgSpec("date", "str", True, "date as YYYY-MM-DD"),
        ),
        "str",
    ),
    ToolSpec(
        "convert_units",
        "Convert a value between two units.",
        (
            ArgSpec("value", "float", True, "numeric value"),
            ArgSpec("from_unit", "str", True, "source unit"),
            ArgSpec("to_unit", "str", True, "target unit"),
        ),
        "float",
    ),
)

TOOLS_BY_NAME: Dict[str, ToolSpec] = {t.name: t for t in TOOLS}

UNIT_TABLE: Dict[Tuple[str, str], float] = {
    ("km", "mi"): 0.621371,
    ("mi", "km"): 1.609344,
    ("kg", "lb"): 2.204623,
    ("lb", "kg"): 0.453592,
    ("c", "f"): 0.0,  # handled specially
    ("l", "gal"): 0.264172,
    ("gal", "l"): 3.785412,
}


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class ToolError(Exception):
    """An error raised by the simulated tool runtime.

    `kind` is the machine-readable family used for stratified analysis;
    `terse`/`standard`/`verbose` are the three renderings of the same failure.
    """

    def __init__(self, kind: str, terse: str, standard: str, verbose: str = ""):
        super().__init__(standard)
        self.kind = kind
        self.terse = terse
        self.standard = standard
        self.verbose = verbose or standard


# --------------------------------------------------------------------------
# Action parsing
# --------------------------------------------------------------------------


@dataclass
class ParsedAction:
    name: str
    kwargs: Dict[str, Any]
    positional: List[Any] = field(default_factory=list)


def parse_action(action: str) -> ParsedAction:
    """Parse ``tool(arg=value, ...)``.

    We accept positional arguments too (small models emit them) and bind them
    to the schema's argument order, because rejecting them on syntax alone
    would conflate 'wrong call' with 'unsupported calling convention'.
    """
    text = action.strip()
    # Models sometimes wrap the call in code fences or prefix it with a label.
    text = re.sub(r"^```(?:python|tool_code)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    # Strip a leading label. Models emit "Action:", and chat-tuned ones
    # sometimes emit a speaker label. A real harness would tolerate both rather
    # than scoring a formatting habit as a tool-use failure, and counting them
    # as failures here would put a formatting artefact into every metric.
    text = re.sub(
        r"^(?:action|tool|call|you|user|assistant|response|reply)\s*:\s*",
        "", text, flags=re.I,
    ).strip()
    if not text:
        raise ToolError(
            "parse_error",
            "Error: empty action.",
            "Error: empty action. Emit exactly one tool call.",
            "Error: empty action. Emit exactly one tool call of the form "
            "tool_name(arg=value).",
        )
    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError as exc:
        raise ToolError(
            "parse_error",
            "Error: syntax error.",
            f"SyntaxError: could not parse the call ({exc.msg}).",
            f"SyntaxError: could not parse the call ({exc.msg}). Expected a "
            f"single call of the form tool_name(arg=value).",
        ) from exc
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ToolError(
            "parse_error",
            "Error: not a tool call.",
            "Error: the action must be a single tool call, e.g. read_file(path='notes/todo.md').",
            "Error: the action must be a single tool call, e.g. "
            "read_file(path='notes/todo.md'). Expressions, assignments and "
            "multiple statements are not accepted.",
        )
    name = node.func.id
    kwargs: Dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg is None:
            raise ToolError(
                "parse_error",
                "Error: **kwargs not supported.",
                "Error: argument unpacking (**) is not supported.",
                "Error: argument unpacking (**) is not supported; pass each "
                "argument explicitly.",
            )
        kwargs[kw.arg] = _literal(kw.value, name)
    positional = [_literal(a, name) for a in node.args]
    return ParsedAction(name=name, kwargs=kwargs, positional=positional)


def _literal(node: ast.AST, tool_name: str) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception as exc:  # noqa: BLE001 - any failure is the same to us
        src = ast.unparse(node)
        raise ToolError(
            "parse_error",
            "Error: bad argument value.",
            f"Error: {tool_name}: argument value {src!r} is not a literal.",
            f"Error: {tool_name}: argument value {src!r} is not a literal. "
            f"Arguments must be plain strings, numbers or lists of strings.",
        ) from exc


# --------------------------------------------------------------------------
# World
# --------------------------------------------------------------------------

FIRST_NAMES = [
    "Dana", "Milo", "Priya", "Ravi", "Elena", "Tomas", "Aisha", "Noor",
    "Hugo", "Ingrid", "Karim", "Lena", "Owen", "Sofia", "Yara", "Bruno",
]
LAST_NAMES = [
    "Whitfield", "Arganda", "Vasquez", "Lindqvist", "Okafor", "Petrova",
    "Hassan", "Bergmann", "Moreau", "Kaneko", "Silva", "Duarte",
]
TEAMS = ["design", "platform", "research", "support"]
FOLDERS = ["notes", "reports", "drafts"]
FILE_STEMS = ["todo", "summary", "handover", "budget", "agenda", "checklist"]


@dataclass
class World:
    files: Dict[str, Dict[str, Any]]
    contacts: List[Dict[str, str]]
    events: List[Dict[str, Any]]
    reminders: List[Dict[str, str]] = field(default_factory=list)
    sent: List[Dict[str, str]] = field(default_factory=list)
    #: Names successfully resolved by find_contact. Read-only tools leave no
    #: trace in world state, so goal checking for lookup tasks needs this.
    resolved: List[str] = field(default_factory=list)

    def clone(self) -> "World":
        import copy

        return copy.deepcopy(self)


def build_world(seed: int, index: int) -> World:
    rng = rng_for(seed, f"toolshed-world-{index}")
    people = []
    used = set()
    for _ in range(8):
        while True:
            first = rng.choice(FIRST_NAMES)
            last = rng.choice(LAST_NAMES)
            full = f"{first} {last}"
            if full not in used:
                used.add(full)
                break
        people.append(
            {
                "name": full,
                "email": f"{first.lower()}.{last.lower()}@corp.example",
                "team": rng.choice(TEAMS),
            }
        )
    # Guarantee every team is non-empty so list_contacts tasks are solvable.
    for i, team in enumerate(TEAMS):
        people[i]["team"] = team

    files: Dict[str, Dict[str, Any]] = {}
    for folder in FOLDERS:
        for stem in rng.sample(FILE_STEMS, 2):
            path = f"{folder}/{stem}.md"
            files[path] = {
                "content": _file_content(rng, stem),
                "readonly": False,
            }
    files["system/config.ini"] = {"content": "mode=standard", "readonly": True}

    base_day = rng.randint(9, 20)
    events = []
    for i in range(3):
        host = rng.choice(people)
        events.append(
            {
                "id": f"evt-{i + 1}",
                "title": rng.choice(
                    ["Design review", "Sprint planning", "Budget sync", "Retro"]
                ),
                "date": f"2026-09-{base_day + i:02d}",
                "time": f"{rng.choice([9, 10, 11, 14, 15]):02d}:00",
                "attendees": [host["email"]],
                "cancelled": False,
            }
        )
    return World(files=files, contacts=people, events=events)


def _file_content(rng, stem: str) -> str:
    topics = {
        "todo": "renew the domain, file the expense report",
        "summary": "throughput improved, latency unchanged",
        "handover": "on-call rotates to the platform team",
        "budget": "hardware 4200, travel 1750",
        "agenda": "roadmap, hiring, incident review",
        "checklist": "backups verified, keys rotated",
    }
    return topics.get(stem, "notes pending")


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


class ToolShedEnv:
    """Executable tool-use environment. Deterministic given (task, actions)."""

    name = "toolshed"

    def __init__(self, verbosity: str = "standard", echo_action: bool = True,
                 demo: bool = False):
        assert verbosity in {"terse", "standard", "verbose"}
        self.verbosity = verbosity
        self.echo_action = echo_action
        #: Append a two-line format demonstration to the system prompt. Off for
        #: the probe (which is teacher-forced, so output format is irrelevant)
        #: and on for the free-running agent study, where without it the
        #: smallest models never emit a parseable call and every harness would
        #: be compared on a floor. It is added identically in every harness
        #: condition, so it cannot favour one of them.
        self.demo = demo
        self._worlds: Dict[str, World] = {}

    # -- world management ---------------------------------------------------

    def reset(self, task: Task) -> None:
        self._worlds[task.task_id] = build_world(
            task.payload["world_seed"], task.payload["world_index"]
        )

    def world(self, task: Task) -> World:
        if task.task_id not in self._worlds:
            self.reset(task)
        return self._worlds[task.task_id]

    # -- prompt -------------------------------------------------------------

    #: A format demonstration for the free-running agent study.
    #:
    #: An earlier version wrote this as a two-turn dialogue with `user:` and
    #: `you:` speaker labels. Small models copied the label into their replies
    #: ("you: find_contact(...)"), which the parser then rejected, and the rate
    #: at which they did so varied by harness — so the demonstration silently
    #: became a confound. It now shows only the reply itself, with nothing that
    #: could be mistaken for part of the output.
    DEMO = (
        "\nA valid reply looks exactly like this, with no prefix or explanation:\n"
        "list_files(folder='notes')\n"
    )

    def system_prompt(self, task: Task) -> str:
        tool_docs = "\n".join(t.doc() for t in TOOLS)
        demo = self.DEMO if self.demo else ""
        return (
            "You are a tool-using assistant working inside a small office "
            "workspace. Solve the user's request by calling tools one at a "
            "time.\n\n"
            "Rules:\n"
            "- Reply with exactly one tool call and nothing else.\n"
            "- Use the form tool_name(arg=value), with keyword arguments.\n"
            "- Dates are YYYY-MM-DD and times are HH:MM.\n"
            "- When the request is fully satisfied, reply with finish().\n\n"
            f"Available tools:\n{tool_docs}\n{demo}"
        )

    # -- execution ----------------------------------------------------------

    def step(self, task: Task, action: str) -> Outcome:
        world = self.world(task)
        try:
            parsed = parse_action(action)
            if parsed.name == "finish":
                return Outcome(
                    ok=True, observation="Session ended.", info={"finish": True}
                )
            spec = TOOLS_BY_NAME.get(parsed.name)
            if spec is None:
                raise self._unknown_tool(parsed.name)
            kwargs = self._bind(spec, parsed)
            result = _DISPATCH[spec.name](world, kwargs)
        except ToolError as err:
            return Outcome(
                ok=False,
                observation=self.render_error(action, err),
                error_type=err.kind,
                echoes_action=self.echo_action,
            )
        return Outcome(ok=True, observation=self.render_result(result))

    def _unknown_tool(self, name: str) -> ToolError:
        near = _closest(name, list(TOOLS_BY_NAME))
        return ToolError(
            "unknown_tool",
            f"Error: unknown tool '{name}'.",
            f"NameError: unknown tool '{name}'. Did you mean '{near}'?",
            f"NameError: unknown tool '{name}'. Did you mean '{near}'? "
            f"The available tools are: {', '.join(TOOLS_BY_NAME)}.",
        )

    def _bind(self, spec: ToolSpec, parsed: ParsedAction) -> Dict[str, Any]:
        kwargs = dict(parsed.kwargs)
        # Bind positionals against the declared order.
        if parsed.positional:
            names = [a.name for a in spec.args]
            if len(parsed.positional) > len(names):
                raise ToolError(
                    "arity_error",
                    f"Error: {spec.name} takes {len(names)} arguments.",
                    f"TypeError: {spec.name}() takes {len(names)} arguments but "
                    f"{len(parsed.positional)} positional were given.",
                    f"TypeError: {spec.name}() takes {len(names)} arguments but "
                    f"{len(parsed.positional)} positional were given. "
                    f"Signature: {spec.signature()}.",
                )
            for name, value in zip(names, parsed.positional):
                if name in kwargs:
                    raise ToolError(
                        "arity_error",
                        f"Error: duplicate argument '{name}'.",
                        f"TypeError: {spec.name}() got multiple values for '{name}'.",
                        f"TypeError: {spec.name}() got multiple values for "
                        f"'{name}'. Signature: {spec.signature()}.",
                    )
                kwargs[name] = value

        declared = {a.name for a in spec.args}
        for key in kwargs:
            if key not in declared:
                near = _closest(key, sorted(declared))
                raise ToolError(
                    "unexpected_argument",
                    f"Error: unexpected argument '{key}'.",
                    f"TypeError: {spec.name}() got an unexpected keyword "
                    f"argument '{key}'. Did you mean '{near}'?",
                    f"TypeError: {spec.name}() got an unexpected keyword "
                    f"argument '{key}'. Did you mean '{near}'? "
                    f"Signature: {spec.signature()}.",
                )
        for arg in spec.args:
            if arg.required and arg.name not in kwargs:
                raise ToolError(
                    "missing_argument",
                    f"Error: missing argument '{arg.name}'.",
                    f"TypeError: {spec.name}() missing required argument "
                    f"'{arg.name}'.",
                    f"TypeError: {spec.name}() missing required argument "
                    f"'{arg.name}' ({arg.description}). "
                    f"Signature: {spec.signature()}.",
                )
            if arg.name in kwargs:
                kwargs[arg.name] = _check_type(spec, arg, kwargs[arg.name])
        return kwargs

    # -- observation rendering ---------------------------------------------

    def render_error(self, action: str, err: ToolError) -> str:
        body = {"terse": err.terse, "standard": err.standard, "verbose": err.verbose}[
            self.verbosity
        ]
        if self.echo_action:
            return f"{body}\n  while calling: {action.strip()}"
        return body

    @staticmethod
    def render_result(result: Any) -> str:
        if isinstance(result, str):
            return result
        return repr(result)

    # -- counterfactual observations ---------------------------------------
    #
    # These are the instruments for the copy/semantics decomposition. All three
    # go through the same wrapper so that framing tokens ("while calling:", the
    # observation role, etc.) are identical across conditions.

    def success_observation(self, action: str, failure: Outcome) -> str:
        """What the runtime would have printed had this exact call worked.

        Built from the *actual* arguments of the failed call wherever possible.
        A generic template would be a subtle confound: an observation naming a
        different person or file than the call it supposedly answers is
        incoherent, and incoherence is itself a signal the model could pick up
        on, which would contaminate the failure-vs-success contrast.
        """
        try:
            parsed = parse_action(action)
        except ToolError:
            return "OK"
        return _success_for(parsed.name, parsed.kwargs, parsed.positional)

    def neutral_observation(
        self, action: str, failure: Outcome, filler_units: int = 0
    ) -> str:
        """A valence-free observation.

        Reports neither success nor failure: the runtime merely acknowledges
        that the call was recorded. `filler_units` appends valence-free hex
        trace segments, which is how we length-match the neutral condition to
        the failure condition without introducing any semantic content.
        """
        base = "Call recorded."
        if filler_units > 0:
            segs = " ".join(
                f"trace={_hex_seg(action, i)}" for i in range(filler_units)
            )
            base = f"{base} {segs}"
        if self.echo_action:
            return f"{base}\n  while calling: {action.strip()}"
        return base

    def is_solved(self, task: Task) -> bool:
        checker = _CHECKERS[task.payload["template"]]
        return checker(self.world(task), task.payload)


def _hex_seg(action: str, i: int) -> str:
    import hashlib

    return hashlib.blake2b(f"{action}:{i}".encode(), digest_size=4).hexdigest()


def _closest(name: str, candidates: Sequence[str]) -> str:
    import difflib

    match = difflib.get_close_matches(name, list(candidates), n=1, cutoff=0.0)
    return match[0] if match else candidates[0]


def _check_type(spec: ToolSpec, arg: ArgSpec, value: Any) -> Any:
    want = arg.type
    if want == "str":
        if not isinstance(value, str):
            raise _type_error(spec, arg, value, "string")
    elif want in {"int", "float"}:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _type_error(spec, arg, value, "number")
        value = float(value)
    elif want == "list[str]":
        if isinstance(value, str):
            raise _type_error(spec, arg, value, "list of strings")
        if not isinstance(value, (list, tuple)) or not all(
            isinstance(v, str) for v in value
        ):
            raise _type_error(spec, arg, value, "list of strings")
        value = list(value)
    return value


def _type_error(spec: ToolSpec, arg: ArgSpec, value: Any, want: str) -> ToolError:
    got = type(value).__name__
    return ToolError(
        "type_error",
        f"Error: '{arg.name}' must be a {want}.",
        f"TypeError: {spec.name}(): argument '{arg.name}' must be a {want}, "
        f"got {got}.",
        f"TypeError: {spec.name}(): argument '{arg.name}' must be a {want}, "
        f"got {got} ({value!r}). Signature: {spec.signature()}.",
    )


# --------------------------------------------------------------------------
# Tool implementations
# --------------------------------------------------------------------------


def _not_found(kind: str, key: str, options: Sequence[str]) -> ToolError:
    near = _closest(key, options) if options else ""
    shown = ", ".join(list(options)[:6])
    return ToolError(
        f"{kind}_not_found",
        f"Error: no {kind} '{key}'.",
        f"LookupError: no {kind} matching '{key}'."
        + (f" Closest match: '{near}'." if near else ""),
        f"LookupError: no {kind} matching '{key}'."
        + (f" Closest match: '{near}'." if near else "")
        + (f" Known {kind}s include: {shown}." if shown else ""),
    )


def _t_list_files(world: World, kw: Dict[str, Any]) -> Any:
    folder = kw["folder"].strip("/")
    hits = sorted(p for p in world.files if p.split("/")[0] == folder)
    if not hits:
        folders = sorted({p.split("/")[0] for p in world.files})
        raise _not_found("folder", folder, folders)
    return hits


def _t_read_file(world: World, kw: Dict[str, Any]) -> Any:
    path = kw["path"]
    if path not in world.files:
        raise _not_found("file", path, sorted(world.files))
    return world.files[path]["content"]


def _t_write_file(world: World, kw: Dict[str, Any]) -> Any:
    path = kw["path"]
    if path in world.files and world.files[path]["readonly"]:
        raise ToolError(
            "permission_error",
            f"Error: '{path}' is read-only.",
            f"PermissionError: '{path}' is read-only and cannot be written.",
            f"PermissionError: '{path}' is read-only and cannot be written. "
            f"Files under system/ are managed by the workspace.",
        )
    world.files[path] = {"content": kw["content"], "readonly": False}
    return f"wrote {len(kw['content'])} characters to {path}"


def _t_delete_file(world: World, kw: Dict[str, Any]) -> Any:
    path = kw["path"]
    if path not in world.files:
        raise _not_found("file", path, sorted(world.files))
    if world.files[path]["readonly"]:
        raise ToolError(
            "permission_error",
            f"Error: '{path}' is read-only.",
            f"PermissionError: '{path}' is read-only and cannot be deleted.",
            f"PermissionError: '{path}' is read-only and cannot be deleted. "
            f"Files under system/ are managed by the workspace.",
        )
    del world.files[path]
    return f"deleted {path}"


def _t_find_contact(world: World, kw: Dict[str, Any]) -> Any:
    name = kw["name"]
    for c in world.contacts:
        if c["name"].lower() == name.lower():
            world.resolved.append(c["name"])
            return dict(c)
    raise _not_found("contact", name, [c["name"] for c in world.contacts])


def _t_list_contacts(world: World, kw: Dict[str, Any]) -> Any:
    team = kw["team"].lower()
    hits = [dict(c) for c in world.contacts if c["team"] == team]
    if not hits:
        raise _not_found("team", kw["team"], sorted({c["team"] for c in world.contacts}))
    return hits


def _require_date(value: str, field_name: str, tool: str) -> str:
    if not DATE_RE.match(value):
        raise ToolError(
            "format_error",
            f"Error: bad {field_name} '{value}'.",
            f"ValueError: {tool}(): '{field_name}' must be YYYY-MM-DD, got "
            f"'{value}'.",
            f"ValueError: {tool}(): '{field_name}' must be YYYY-MM-DD, got "
            f"'{value}'. Example: 2026-09-14.",
        )
    return value


def _t_list_events(world: World, kw: Dict[str, Any]) -> Any:
    date = _require_date(kw["date"], "date", "list_events")
    return [
        {k: v for k, v in e.items() if k != "cancelled"}
        for e in world.events
        if e["date"] == date and not e["cancelled"]
    ]


def _t_create_event(world: World, kw: Dict[str, Any]) -> Any:
    date = _require_date(kw["date"], "date", "create_event")
    time = kw["time"]
    if not TIME_RE.match(time):
        raise ToolError(
            "format_error",
            f"Error: bad time '{time}'.",
            f"ValueError: create_event(): 'time' must be HH:MM, got '{time}'.",
            f"ValueError: create_event(): 'time' must be HH:MM, got '{time}'. "
            f"Example: 14:30.",
        )
    known = {c["email"] for c in world.contacts}
    for a in kw["attendees"]:
        if a not in known:
            raise _not_found("attendee", a, sorted(known))
    new_id = f"evt-{len(world.events) + 1}"
    world.events.append(
        {
            "id": new_id,
            "title": kw["title"],
            "date": date,
            "time": time,
            "attendees": list(kw["attendees"]),
            "cancelled": False,
        }
    )
    return f"created {new_id}"


def _t_cancel_event(world: World, kw: Dict[str, Any]) -> Any:
    eid = kw["event_id"]
    for e in world.events:
        if e["id"] == eid:
            if e["cancelled"]:
                raise ToolError(
                    "state_error",
                    f"Error: '{eid}' already cancelled.",
                    f"StateError: event '{eid}' is already cancelled.",
                    f"StateError: event '{eid}' is already cancelled; cancelling "
                    f"it again has no effect.",
                )
            e["cancelled"] = True
            return f"cancelled {eid}"
    raise _not_found("event", eid, [e["id"] for e in world.events])


def _t_send_message(world: World, kw: Dict[str, Any]) -> Any:
    to = kw["to"]
    known = {c["email"] for c in world.contacts}
    if "@" not in to:
        raise ToolError(
            "format_error",
            f"Error: '{to}' is not an email address.",
            f"ValueError: send_message(): 'to' must be an email address, got "
            f"'{to}'.",
            f"ValueError: send_message(): 'to' must be an email address, got "
            f"'{to}'. Use find_contact() to resolve a name to an address.",
        )
    if to not in known:
        raise _not_found("recipient", to, sorted(known))
    world.sent.append({"to": to, "subject": kw["subject"], "body": kw["body"]})
    return f"sent to {to}"


def _t_set_reminder(world: World, kw: Dict[str, Any]) -> Any:
    date = _require_date(kw["date"], "date", "set_reminder")
    world.reminders.append({"text": kw["text"], "date": date})
    return f"reminder set for {date}"


def _t_convert_units(world: World, kw: Dict[str, Any]) -> Any:
    src = kw["from_unit"].lower()
    dst = kw["to_unit"].lower()
    if (src, dst) not in UNIT_TABLE:
        pairs = ", ".join(f"{a}->{b}" for a, b in list(UNIT_TABLE)[:5])
        raise ToolError(
            "unsupported_conversion",
            f"Error: cannot convert {src} to {dst}.",
            f"ValueError: convert_units(): no conversion from '{src}' to "
            f"'{dst}'.",
            f"ValueError: convert_units(): no conversion from '{src}' to "
            f"'{dst}'. Supported pairs include: {pairs}.",
        )
    if (src, dst) == ("c", "f"):
        return kw["value"] * 9 / 5 + 32
    return round(kw["value"] * UNIT_TABLE[(src, dst)], 4)


_DISPATCH: Dict[str, Callable[[World, Dict[str, Any]], Any]] = {
    "list_files": _t_list_files,
    "read_file": _t_read_file,
    "write_file": _t_write_file,
    "delete_file": _t_delete_file,
    "find_contact": _t_find_contact,
    "list_contacts": _t_list_contacts,
    "list_events": _t_list_events,
    "create_event": _t_create_event,
    "cancel_event": _t_cancel_event,
    "send_message": _t_send_message,
    "set_reminder": _t_set_reminder,
    "convert_units": _t_convert_units,
}

#: Fallback success renderings, used when the failed call carries no usable
#: argument (e.g. a dropped required argument).
_SUCCESS_SHAPES: Dict[str, str] = {
    "list_files": "['notes/agenda.md', 'notes/todo.md']",
    "read_file": "renew the domain, file the expense report",
    "write_file": "wrote 26 characters to the file",
    "delete_file": "deleted the file",
    "find_contact": "{'name': 'Ingrid Moreau', 'email': "
    "'ingrid.moreau@corp.example', 'team': 'design'}",
    "list_contacts": "[{'name': 'Ingrid Moreau', 'email': "
    "'ingrid.moreau@corp.example', 'team': 'design'}]",
    "list_events": "[{'id': 'evt-2', 'title': 'Sprint planning', 'date': "
    "'2026-09-14', 'time': '10:00', 'attendees': ['milo.silva@corp.example']}]",
    "create_event": "created evt-4",
    "cancel_event": "cancelled evt-2",
    "send_message": "sent to the recipient",
    "set_reminder": "reminder set",
    "convert_units": "12.4274",
    "finish": "Session ended.",
}


def _first_str(kwargs: Dict[str, Any], positional: Sequence[Any], *names: str) -> Optional[str]:
    """First string argument matching one of `names`, else the first string at all.

    Perturbed calls frequently rename arguments (`path` -> `file_path`), so we
    fall back to positional order rather than giving up.
    """
    for n in names:
        v = kwargs.get(n)
        if isinstance(v, str):
            return v
    for v in list(kwargs.values()) + list(positional):
        if isinstance(v, str):
            return v
    return None


def _success_for(tool: str, kwargs: Dict[str, Any], positional: Sequence[Any]) -> str:
    """Argument-aware plausible success return for `tool`."""
    s = lambda *names: _first_str(kwargs, positional, *names)  # noqa: E731
    if tool == "find_contact":
        name = s("name", "contact_name") or "Ingrid Moreau"
        slug = name.lower().replace(" ", ".")
        return f"{{'name': {name!r}, 'email': '{slug}@corp.example', 'team': 'design'}}"
    if tool == "list_contacts":
        team = s("team", "team_name") or "design"
        return (
            "[{'name': 'Ingrid Moreau', 'email': 'ingrid.moreau@corp.example', "
            f"'team': {team!r}}}]"
        )
    if tool == "read_file":
        return "renew the domain, file the expense report"
    if tool == "write_file":
        path = s("path", "file_path") or "the file"
        content = kwargs.get("content") or kwargs.get("text") or ""
        n = len(content) if isinstance(content, str) else 26
        return f"wrote {n} characters to {path}"
    if tool == "delete_file":
        return f"deleted {s('path', 'file_path') or 'the file'}"
    if tool == "list_files":
        folder = s("folder", "directory") or "notes"
        return f"['{folder}/agenda.md', '{folder}/todo.md']"
    if tool == "list_events":
        date = s("date", "day") or "2026-09-14"
        return (
            "[{'id': 'evt-2', 'title': 'Sprint planning', 'date': "
            f"{date!r}, 'time': '10:00', "
            "'attendees': ['milo.silva@corp.example']}]"
        )
    if tool == "cancel_event":
        return f"cancelled {s('event_id', 'id') or 'evt-2'}"
    if tool == "send_message":
        return f"sent to {s('to', 'recipient') or 'the recipient'}"
    if tool == "set_reminder":
        return f"reminder set for {s('date', 'day') or '2026-09-14'}"
    if tool == "create_event":
        return "created evt-4"
    if tool == "convert_units":
        return "12.4274"
    return _SUCCESS_SHAPES.get(tool, "OK")


# --------------------------------------------------------------------------
# Goal checkers
# --------------------------------------------------------------------------


def _chk_email_file(world: World, p: Dict[str, Any]) -> bool:
    return any(
        m["to"] == p["contact_email"] and p["file_content"] in m["body"]
        for m in world.sent
    )


def _chk_cancel(world: World, p: Dict[str, Any]) -> bool:
    return any(e["id"] == p["event_id"] and e["cancelled"] for e in world.events)


def _chk_schedule(world: World, p: Dict[str, Any]) -> bool:
    return any(
        (not e["cancelled"])
        and e["date"] == p["date"]
        and e["time"] == p["time"]
        and p["contact_email"] in e["attendees"]
        for e in world.events
    )


def _chk_roster(world: World, p: Dict[str, Any]) -> bool:
    f = world.files.get(p["out_path"])
    if not f:
        return False
    return all(name.split()[0] in f["content"] for name in p["team_names"])


def _chk_archive(world: World, p: Dict[str, Any]) -> bool:
    return p["path"] not in world.files and any(
        r["date"] == p["date"] for r in world.reminders
    )


def _chk_convert(world: World, p: Dict[str, Any]) -> bool:
    f = world.files.get(p["out_path"])
    return bool(f) and str(p["expected"]) in f["content"]


_CHECKERS: Dict[str, Callable[[World, Dict[str, Any]], bool]] = {
    "email_file": _chk_email_file,
    "cancel_event": _chk_cancel,
    "schedule": _chk_schedule,
    "roster": _chk_roster,
    "archive": _chk_archive,
    "convert": _chk_convert,
}


def _chk_lookup(world: World, p: Dict[str, Any]) -> bool:
    """Solved when the agent actually resolved the named contact.

    Read-only tools leave no trace in world state, so the runtime records
    successful resolutions (World.resolved) for goal checking.
    """
    return p["contact_name"].lower() in {n.lower() for n in world.resolved}


def _chk_simple_write(world: World, p: Dict[str, Any]) -> bool:
    f = world.files.get(p["out_path"])
    return bool(f) and p["text"].strip() in f["content"]


def _chk_simple_reminder(world: World, p: Dict[str, Any]) -> bool:
    return any(
        r["date"] == p["date"] and p["text"].strip() in r["text"]
        for r in world.reminders
    )


_CHECKERS.update(
    {
        "lookup": _chk_lookup,
        "simple_write": _chk_simple_write,
        "simple_reminder": _chk_simple_reminder,
    }
)
