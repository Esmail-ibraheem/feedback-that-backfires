"""The substitution table itself. Run once; kept in the repo as the record of
what was changed and why, so the rewrite is auditable rather than a diff.

Every entry replaces an em-dash construction with ordinary punctuation chosen
for that sentence: a comma pair for a light aside, parentheses for a list, a
colon for a definition, or two sentences where the aside was doing too much
work. Re-running after the edits have been applied is a no-op that reports the
entries as missing, which is the intended failure mode.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dedash import apply  # noqa: E402

S = "paper/sections/"

apply(S + "analysis.tex", [
    (r"names the argument the model should have used --- about as informative as an error message can be without solving the task --- and it does not measurably move",
     "names the argument the model should have used, which is about as informative\nas an error message can be without solving the task, and it does not\nmeasurably move"),
    (r"When the runtime echoes the offending call inside the error message --- as a Python traceback does by default, and as several agent frameworks do by convention --- the failed call appears twice instead of once,",
     "When the runtime echoes the offending call inside the error message, as a\nPython traceback does by default and as several agent frameworks do by\nconvention, the failed call appears twice instead of once,"),
    (r"helps by \DEarlyCI\ nats --- a real effect, and about three percent of the total.",
     "helps by \\DEarlyCI\\ nats, a real effect and about three percent of the\ntotal."),
    (r"where it also fails to reduce repetition --- but where it does change behaviour in a way this measurement does not anticipate.",
     "where it also fails to reduce repetition, but where it does change behaviour\nin a way this measurement does not anticipate."),
    (r"Replacing the verbatim call with a runtime-generated description of the failure --- same diagnosis, no token sequence --- moves the log-probability of repeating",
     "Replacing the verbatim call with a runtime-generated description of the\nfailure, which keeps the diagnosis and drops the token sequence, moves the\nlog-probability of repeating"),
    (r"the placebo recovers \DPlaceboCI\ nats --- a large majority of the effect --- but not all of it.",
     "the placebo recovers \\DPlaceboCI\\ nats, a large majority of the effect,\nbut not all of it."),
    (r"and we wrote those glosses --- so perhaps it works because the glosses are good, which would not generalise.",
     "and we wrote those glosses, so perhaps it works because the glosses are good,\nwhich would not generalise."),
    (r"against \DAbstractMatchedCI\ for the full abstraction on the same models --- if anything slightly more.",
     "against \\DAbstractMatchedCI\\ for the full abstraction on the same models, if\nanything slightly more."),
    (r"That does not make the diagnosis worthless --- it is what the model needs in order to write a \emph{different} call rather than merely a different string --- but it is not what is doing the work here.",
     "That does not make the diagnosis worthless, since it is what the model needs\nin order to write a \\emph{different} call rather than merely a different\nstring, but it is not what is doing the work here."),
    (r"The surface-form term is unmoved --- \MeanCopy\ against a short neutral, \CopyPad\ against a length-matched one --- so the effect the paper rests on is not an artefact of context length.",
     "The surface-form term is unmoved, at \\MeanCopy\\ against a short neutral and\n\\CopyPad\\ against a length-matched one, so the effect the paper rests on is\nnot an artefact of context length."),
    (r"so the correct action --- which usually differs from the failed one by a single argument name or value --- should also become more likely, and it does.",
     "so the correct action, which usually differs from the failed one by a single\nargument name or value, should also become more likely, and it does."),
])

apply(S + "agent.tex", [
    (r"A task counts as solved only if the environment's goal predicate holds --- there is no partial credit and no model judges anything.",
     "A task counts as solved only if the environment's goal predicate holds. There\nis no partial credit and no model judges anything."),
    (r"\emph{drop} (the failed step deleted --- clean restart),",
     r"\emph{drop} (the failed step deleted, i.e.\ clean restart),"),
    (r"\paragraph{Telling the model not to repeat does not reduce repetition --- but it does something else.}",
     "\\paragraph{Telling the model not to repeat does not reduce repetition, but it\ndoes something else.}"),
    (r"and the most plausible readings --- the instruction makes the model more careful in general, or less willing to abandon a task early --- are not about repetition at all.",
     "and the most plausible readings, that the instruction makes the model more\ncareful in general or less willing to abandon a task early, are not about\nrepetition at all."),
    (r"A deterministic policy in an identical context emits an identical action --- so clean restart does not remove the problem, it guarantees it.",
     "A deterministic policy in an identical context emits an identical action, so\nclean restart does not remove the problem, it guarantees it."),
    (r"so the model is not paraphrasing its way around the constraint --- it writes genuinely different calls.",
     "so the model is not paraphrasing its way around the constraint; it writes\ngenuinely different calls."),
    (r"What they do not do is convert that into task success at this scale --- which Section~\ref{sec:failure} explains:",
     "What they do not do is convert that into task success at this scale, which\nSection~\\ref{sec:failure} explains:"),
    (r"the repeated string is in every case one the parser rejected --- unterminated prose, or a call with an unbalanced parenthesis --- never a well-formed call.",
     "the repeated string is in every case one the parser rejected, either\nunterminated prose or a call with an unbalanced parenthesis, and never a\nwell-formed call."),
    (r"The context edit and the decoder constraint therefore catch different things --- one the copying, the other the residue --- and a harness that wants neither should do both.",
     "The context edit and the decoder constraint therefore catch different things,\none the copying and the other the residue, and a harness that wants neither\nshould do both."),
    (r"and the standard harness sits between --- which is the ordering the decomposition predicts, arrived at without reference to it.",
     "and the standard harness sits between. That is the ordering the decomposition\npredicts, arrived at without reference to it."),
    (r"What we cannot yet do is test whether the probe \emph{predicts across models} --- whether a model with a more negative $\Gain$ loops more in practice.",
     "What we cannot yet do is test whether the probe \\emph{predicts across models},\nthat is, whether a model with a more negative $\\Gain$ loops more in practice."),
])
