# Superseded rollouts (role-prefix confound)

These are the first end-to-end runs. They are kept for transparency and are
**not** used by any analysis or by the manuscript.

The format demonstration appended to the system prompt was originally written as
a two-turn dialogue with `user:` and `you:` speaker labels. Small models copied
the `you:` label into their replies, the parser rejected those as malformed
calls, and `parse_error` ended up accounting for 89% of all failures. Crucially
the rate varied by harness (0% under the instruction variant, ~10% under
`abstract`, 42-53% under `verbatim` and `drop`), so the artefact interacted with
the very comparison the study is about.

Two fixes were made and the study was re-run from scratch:

1. The demonstration now shows only the reply itself, with nothing that could be
   mistaken for part of the output (`ToolShedEnv.DEMO`).
2. `parse_action` strips a leading speaker or action label, because a real
   harness should tolerate one rather than score a formatting habit as a
   tool-use failure.

See `research_log.md`, entry 2026-08-09.
