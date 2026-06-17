from __future__ import annotations

import json

from .types import Question


_FINAL_FORMAT_REQUIREMENT = "Your final answer MUST end with this exact format:"
_BOXED_OUTPUT_MARKER = "\\boxed{"


DEFAULT_PROMPT_TEMPLATES: dict[str, str] = {
    "agent_role": "You are an agent that can predict future events.",
    "guidance": (
        "Do not use any other format. Do not refuse to make a prediction. "
        'Do not say "I cannot predict the future." You must make a clear '
        "prediction based on the best data currently available, using the box "
        "format specified above."
    ),
    "yes_no_output_format": (
        "Your task is to predict whether the event will occur based on your analysis.\n"
        "Your prediction will be scored based on its accuracy. You will only receive "
        "points if your answer is correct.\n"
        f"{_FINAL_FORMAT_REQUIREMENT}\n"
        "\\boxed{Yes} or \\boxed{No}"
    ),
    "binary_named_output_format": (
        "Your task is to predict which of the two outcomes will occur based on your analysis.\n"
        "Your prediction will be scored based on its accuracy. You will only receive "
        "points if your answer is correct.\n"
        f"{_FINAL_FORMAT_REQUIREMENT}\n"
        "\\boxed{<options[0]>} or \\boxed{<options[1]>}"
    ),
    "yes_no_prompt_template": (
        "{agent_role}\n\n"
        "Event to predict: {event}\n"
        "Resolution date: {end_time} (GMT+8). Use this date to identify the "
        "event instance: it is when the event occurs, is decided, or has an "
        "observable result. For recurring or periodically reported events, do "
        "not substitute another cycle.\n\n"
        "{output_format}\n"
        "{guidance}"
    ),
    "binary_named_prompt_template": (
        "{agent_role}\n\n"
        "Event to predict: {event}\n"
        "Resolution date: {end_time} (GMT+8). Use this date to identify the "
        "event instance: it is when the event occurs, is decided, or has an "
        "observable result. For recurring or periodically reported events, do "
        "not substitute another cycle.\n\n"
        "{output_format}\n"
        "{guidance}"
    ),
    "multiple_choice_single_prompt_template": (
        "{agent_role}\n\n"
        "Event to predict: {event}\n"
        "Resolution date: {end_time} (GMT+8). Use this date to identify the "
        "event instance: it is when the event occurs, is decided, or has an "
        "observable result. For recurring or periodically reported events, do "
        "not substitute another cycle.\n\n"
        "Options:{outcomes_block}\n\n"
        "{output_format}\n"
        "{guidance}"
    ),
    "multiple_choice_multi_prompt_template": (
        "{agent_role}\n\n"
        "Event to predict: {event}\n"
        "Resolution date: {end_time} (GMT+8). Use this date to identify the "
        "event instance: it is when the event occurs, is decided, or has an "
        "observable result. For recurring or periodically reported events, do "
        "not substitute another cycle.\n\n"
        "Options:{outcomes_block}\n\n"
        "{output_format}\n"
        "{guidance}"
    ),
    "outcomes_block_rule": (
        "Empty for yes_no/binary_named. For multiple_choice: '\\n' + "
        "'A. <label>\\nB. <label>\\n...' built from options."
    ),
    "multiple_choice_single_output_format": (
        "This is a SINGLE-ANSWER question: exactly ONE of the listed options is correct.\n"
        "Your prediction will be scored on strict equality with the unique correct "
        "letter; choosing the wrong letter, or selecting more than one letter, scores zero.\n"
        f"{_FINAL_FORMAT_REQUIREMENT}\n"
        "the single correct letter inside the box, e.g. \\boxed{A}.\n"
        "Do NOT list more than one letter, even if you believe two outcomes are tied "
        "; pick the one you find most likely."
    ),
    "multiple_choice_multi_output_format": (
        "This is a MULTI-SELECT question: ONE OR MORE of the listed options can be correct.\n"
        "Your prediction will be scored on strict equality with the FULL set of correct "
        "letters: any extra letter, any missing letter, or any wrong letter scores zero. "
        "You must include ALL correct options and NO incorrect options.\n"
        f"{_FINAL_FORMAT_REQUIREMENT}\n"
        "listing all correct option(s) you have identified, separated by commas, within the box.\n"
        "For example: \\boxed{A} for a single correct option, or \\boxed{B, C} "
        "for multiple correct options."
    ),
}


REQUIRED_PROMPT_TEMPLATE_KEYS: tuple[str, ...] = tuple(DEFAULT_PROMPT_TEMPLATES)


# Lowercase / symbolic labels are easily eaten by markdown (backticks /
# underscores / asterisks). With >26 options the labels land on `[`, `\`, `]`,
# `^`, `_`, `` ` ``, `a`, `b`, ... so we uniformly wrap any non-A-Z label in
# backticks to prevent the markdown processor from swallowing them in the
# prompt the LLM sees.
_BACKTICK_SAFE_ASCII_RANGE = set(range(ord("A"), ord("Z") + 1))


# Reflection protocol. Appended as a tail paragraph to the user message; it is
# NOT part of prompt_templates (so prompt_templates_hash is unaffected) but is
# persisted in the user_prompt field so the run is fully reproducible.
# Design goal: use prompt engineering to pull the model from "1 web_search and
# answer directly" to ">=3 distinct angles + reflection after each + opposite-
# direction self-check", driven mostly by the protocol itself rather than a
# hard minimum count.
#
# BELIEF_PROTOCOL: a tail-attached section parallel to the reflection
# protocol, requiring the LLM to emit a strict <belief>...</belief> JSON block
# before \\boxed{...}, so probability-family metrics (Brier / NLL / MBS / ECE /
# calibration curves) can be computed directly. Likewise NOT included in
# prompt_templates_hash; enabled via Settings.BELIEF_PROTOCOL with the
# fingerprint recorded separately in run_meta.belief_protocol_*.
REFLECTION_PROTOCOL = """\

---
**Forecasting Protocol — follow every step before producing the final \\boxed{...} answer.**

1. **Decompose.** Privately list the sub-questions whose answers, taken together, settle this prediction. Note the entities, dates, quantities, and definitions that need to be pinned down.

2. **Plan distinct angles.** Before issuing any `web_search`, list at least three SEPARATE investigation angles that use different keywords and look at different evidence types — for example direct event reporting, official statements/data, betting markets or polls, expert analysis, base rates from similar past events, contradictory or skeptical sources. Re-phrasing the same query is NOT a new angle.

3. **Search iteratively, reflect after every result.** Each time `web_search` returns:
   - In your reasoning, paraphrase what the most relevant snippets actually said (do not copy them).
   - Tag each result as relevant / partially relevant / off-topic, and note its publication date.
   - Identify what new sub-question or contradiction the result raises.
   - Pick the next query to fill the largest remaining gap or cover an angle you have NOT yet exercised. Avoid near-duplicate queries.

4. **Cross-validate.** Do not commit to an answer until at least two independent sources (different domains, different publication dates) corroborate the key facts driving it. If only one source supports the conclusion, search again from a fresh angle.

5. **Stress-test the opposite.** Explicitly ask "What is the strongest case for the OPPOSITE outcome?" If you cannot articulate it from the evidence you have gathered, run a search aimed at that counter-case. Then weigh both sides.

6. **Calibrate, then commit.** State your confidence (low / medium / high), the single most likely failure mode of your prediction, and the most decisive piece of evidence. Only after this self-check, output the final answer in the exact required `\\boxed{...}` format on the last line.

**Quality bar.** A confident answer with only one search is almost always under-researched on this benchmark; multi-angle searches with explicit reflection consistently outperform one-shot guesses. Use the `web_search` tool generously — your search budget is large enough to support thorough investigation. Do NOT skip the protocol even if the answer feels obvious.
"""


# Belief protocol. Appended AFTER reflection protocol when both are enabled.
# Like reflection it's a tail-attachment to the user message: NOT part of
# `prompt_templates`, so `prompt_templates_hash` is unchanged; the fingerprint
# lives in `run_meta.belief_protocol_text` / `belief_protocol_hash`.
#
# Design intent: convert the model's implicit "evidence -> chosen letter" into
# an explicit, scoreable probability vector, so the analysis layer can compute
# proper scoring rules (Brier / NLL / MBS), calibration (ECE / Murphy
# decomposition), and behavioral metrics (belief evolution, confidence
# diagnosis) without re-prompting. Boxed answer remains for backward compat.
BELIEF_PROTOCOL = """\

---
**Belief Protocol — emit a structured probability block BEFORE every `\\boxed{...}` answer.**

After completing the forecasting protocol above, output a `<belief>...</belief>` block on its own line(s), THEN the final `\\boxed{...}` answer. The belief block is the object that gets scored by proper scoring rules (Brier / NLL); the boxed answer is kept only for backward compatibility. Probabilities matter — being well-calibrated beats being overconfident, even when the boxed letter is the same.

**Required JSON schema** (strict; emit valid JSON only — no comments, no trailing commas):

```
<belief>{
  "version": "v4.0",
  "probabilities": { "<letter>": <float in [0, 1]>, ... },
  "confidence": "low" | "medium" | "high",
  "key_evidence":     [ "<= 280 chars per bullet, 1-4 bullets",  ... ],
  "counterevidence":  [ "<= 280 chars per bullet, 0-3 bullets",  ... ],
  "open_questions":   [ "<= 280 chars per bullet, 0-3 bullets",  ... ],
  "decision_rule": "argmax" | "multi-select@<threshold>"
}</belief>
```

**Probability rules.**
- Use the SAME letter labels (`A`, `B`, ...) shown in the question's outcomes block. Emit one entry per outcome — every letter present in the question MUST appear as a key.
- For single-answer questions (yes_no / binary_named / multiple_choice with one true label): the values MUST sum to `1.0` (tolerance `1e-3`).
- For multi-select questions: each value is an INDEPENDENT Bernoulli (probability that THAT outcome is in the true set); values do NOT need to sum to 1.
- Each value MUST lie in `[0, 1]`. Avoid the boundary unless you really mean it: `0.99` says "I'd accept long odds against this being wrong"; pick `0.85` if you would not.

**Calibration norms.**
- Calibrated probabilities are the goal. If you searched once and feel "maybe 60%", emit `0.6`, not `0.9`. Overconfidence is penalised quadratically (Brier) and exponentially (NLL).
- `confidence` is your subjective bucket; pair it honestly with the spread of `probabilities`. `"high"` + flat distribution is incoherent.
- `key_evidence` lists the strongest 1-4 facts that DROVE your top probability. `counterevidence` is the best case AGAINST your top choice — emit it even when you reject it; an empty `counterevidence` on a non-trivial question is a red flag.
- `decision_rule` is `"argmax"` for single-answer; for multi-select, use `"multi-select@<t>"` with `<t>` your inclusion threshold (e.g. `"multi-select@0.5"`).

**Output order (exact).**
1. Your reasoning / search reflection (free text — anywhere in the message).
2. The `<belief>{...}</belief>` JSON block.
3. The final `\\boxed{...}` answer on the LAST line.

If you cannot produce a valid belief block, emit your boxed answer anyway — the boxed path is independent and will still be scored. But a missing or malformed belief block costs you the probabilistic family of metrics for this sample, so default to emitting one whenever your reasoning supports it.
"""


# ---------------------------------------------------------------------------
# Harness status injections (unified format).
#
# Every harness-injected user message during the ReAct loop shares ONE format:
# a status header on the first line + a scenario-specific directive below it.
# This eliminates the steady drift between the historical "penultimate /
# last-step / budget-exhausted / step-position" branches (each had its own
# tone, its own step counter, and overlapping content), and gives the LLM a
# single, predictable shape to parse.
#
# Header:
#   [Harness status] step k/N (R remaining) · web_search s/C used (M left).
#
# Builders below all call `_build_status_header` for line one, then append
# the directive that matches the scenario. The directive is the only place
# the four scenarios diverge — header math is shared so the step counter,
# search counter, and "remaining" math cannot drift.
# ---------------------------------------------------------------------------


def _build_status_header(
    *,
    current_step: int,
    max_steps: int,
    searches_done: int,
    max_search_calls: int,
) -> str:
    """Single-line status banner used by every harness injection.

    `current_step` is 1-indexed and refers to the step the LLM is ABOUT TO
    issue (matches every existing builder's convention). `max_search_calls`
    can be 0 (web_search disabled in this run); we emit a clear "disabled"
    note in that case rather than printing `0/0`.
    """
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    if current_step < 1 or current_step > max_steps:
        raise ValueError(
            f"current_step must be in [1, {max_steps}]; got {current_step}"
        )
    if max_search_calls < 0:
        raise ValueError(f"max_search_calls must be >= 0, got {max_search_calls}")
    if searches_done < 0:
        raise ValueError(f"searches_done must be >= 0, got {searches_done}")
    steps_remaining = max_steps - current_step + 1  # incl. the step we're entering
    if max_search_calls > 0:
        searches_left = max(max_search_calls - searches_done, 0)
        search_clause = (
            f"web_search {searches_done}/{max_search_calls} used "
            f"({searches_left} left)"
        )
    else:
        search_clause = "web_search disabled"
    return (
        f"[Harness status] step {current_step}/{max_steps} "
        f"({steps_remaining} remaining, including this one) · {search_clause}."
    )


def _build_nudge_message(
    *,
    current_step: int,
    max_steps: int,
    searches_done: int,
    max_search_calls: int,
    min_required: int,
) -> str:
    """Soft search-floor reminder when the LLM tries to finalise too early.

    Phrased as concrete process feedback (not scolding). The message never
    reveals `end_date` / training-cutoff / other information-barrier internals
    — it only references public counters (search count, the public minimum
    from `.env`) and the public step counter. Includes the unified status
    header so the model never has to count messages to know where it stands.
    """
    header = _build_status_header(
        current_step=current_step,
        max_steps=max_steps,
        searches_done=searches_done,
        max_search_calls=max_search_calls,
    )
    return (
        f"{header}\n"
        f"You attempted to give a final answer after only {searches_done} "
        f"web_search call(s), but the forecasting protocol requires consulting "
        f"at least {min_required} sources from DIFFERENT angles before "
        "committing. Continue the investigation: pick a NEW angle you have not "
        "yet exercised (e.g. an opposing viewpoint, an independent data source, "
        "or a base-rate comparable), issue another `web_search`, then reflect "
        "on what it adds before you finalise. Do not repeat a near-duplicate "
        "of any previous query."
    )


# Budget-awareness protocol (static, appended once to the initial user prompt).
# Pairs with the runtime status injections below — this section gives the model
# the global plan up front; the runtime injectors then deliver the position-
# specific reminder at penultimate / last / budget-exhausted / continuation
# moments. The static text is NOT part of `prompt_templates`
# (`prompt_templates_hash` stays bit-stable); its presence is recorded per
# sample via the `user_prompt` field and run-wide via `Settings.config_snapshot`.
def build_budget_awareness_protocol(*, max_steps: int, max_search_calls: int) -> str:
    """Render the budget-awareness tail attached to the initial user prompt.

    `max_steps` is `REACT_MAX_STEPS` and `max_search_calls` is the cell-local
    `REACT_MAX_SEARCH_CALLS` int (the dispatcher has already downcast the
    grid list to a single int by the time this is called). When
    `max_search_calls == 0`, web_search is unavailable; we still emit the
    step-budget guidance because the boxed-answer pacing matters either way.
    """
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    if max_search_calls < 0:
        raise ValueError(f"max_search_calls must be >= 0, got {max_search_calls}")
    search_clause = (
        f"and at most **{max_search_calls}** `web_search` call(s)"
        if max_search_calls > 0
        else "(web_search is disabled in this run)"
    )
    return (
        "\n\n---\n"
        "**Budget Awareness — your harness has hard limits.**\n\n"
        f"You may take at most **{max_steps}** reasoning/tool steps {search_clause}. "
        "Each assistant turn — whether it issues tool calls or plain content — counts "
        "as one step against this budget.\n\n"
        "**How the loop exits.**\n"
        "The loop stops as soon as your reply contains a parseable `\\boxed{...}` answer. "
        "Replies without `\\boxed{...}` (tool-call turns or intermediate content turns) "
        "keep the loop running and each cost one step. There is no penalty for "
        "separating reasoning from your final commitment — just include `\\boxed{...}` "
        "when you are ready.\n\n"
        "**Pacing rules.**\n"
        "- Commit with `\\boxed{...}` as soon as your evidence is sufficient.\n"
        "- When `web_search` is removed (budget exhausted or step limit reached), "
        "include `\\boxed{...}` in your next reply.\n"
        f"- Step {max_steps} is the hard deadline: the harness strips all tools and "
        "forces a final reply — if it still lacks `\\boxed{...}`, the sample scores zero.\n\n"
        "**Status feedback.**\n"
        "Before each of your turns the harness may inject a short user message "
        "starting with `[Harness status] step k/N ...` carrying the live step "
        "and search counters. Tool error replies likewise include a `status` "
        "field. Read those — they are the source of truth for your remaining "
        "budget so you do not have to count messages yourself.\n"
    )


def build_penultimate_step_warning(
    *, current_step: int, max_steps: int, searches_done: int, max_search_calls: int
) -> str:
    """Soft reminder injected as a user message inside the LOOKAHEAD window
    (typically the second-to-last step).

    Branches on whether the search budget is already exhausted:
    - Exhausted (or web_search disabled): tools will be stripped THIS turn;
      the model can no longer search and must commit on this reply or the next.
    - Remaining: model may issue one last decisive search, or commit early.

    Both branches use the unified status header so the model always has an
    up-to-date snapshot of (step, searches) without doing arithmetic.
    """
    header = _build_status_header(
        current_step=current_step,
        max_steps=max_steps,
        searches_done=searches_done,
        max_search_calls=max_search_calls,
    )
    if max_search_calls == 0 or searches_done >= max_search_calls:
        # Budget already exhausted (or search disabled): tools are being stripped
        # this turn. The model cannot defer searching — it must commit now or
        # on the next (final) step.
        return (
            f"{header}\n"
            f"You are at the second-to-last step. The `web_search` tool schema "
            "has been removed (no search budget left for this run). Include "
            "`\\boxed{...}` in this reply, or the harness will force a final "
            "reply on the next (final) step. The next step is the hard deadline."
        )
    return (
        f"{header}\n"
        "You are at the second-to-last step. You may issue ONE more "
        "`web_search` if a single decisive query is still missing, OR commit "
        "with `\\boxed{...}` now. The NEXT step strips `web_search` and is "
        "the hard deadline."
    )


def build_last_step_force_finalisation(
    *,
    current_step: int,
    max_steps: int,
    searches_done: int,
    max_search_calls: int,
) -> str:
    """Hard cutoff message paired with `tools=[]` on the final step.

    Same tone and contract as the legacy `REACT_FINAL_ANSWER_RETRY` bail-out
    ("Time to commit") — kept consistent on purpose so traces stay comparable
    across runs that use either mechanism. Status header keeps the step /
    search counters visible so the model can ground its base-rate guess in
    what it actually did or did not get to investigate.
    """
    header = _build_status_header(
        current_step=current_step,
        max_steps=max_steps,
        searches_done=searches_done,
        max_search_calls=max_search_calls,
    )
    return (
        f"{header}\n"
        f"Harness cutoff: this is the final step ({current_step} of "
        f"{max_steps}) and the `web_search` tool schema has been removed. "
        "Stop investigating and output your final `\\boxed{...}` answer NOW. "
        "An empty reply or a reply without `\\boxed{...}` scores zero — even "
        "an uneducated base-rate guess in the box beats no answer."
    )


def build_search_budget_exhausted_commit(
    *,
    current_step: int,
    max_steps: int,
    searches_done: int,
    max_search_calls: int,
) -> str:
    """Injected once when the search budget is hit and tools are being stripped.

    Unlike the step-limit cutoffs this fires mid-budget: the model may still
    have remaining steps but no more searches. The directive guides the model
    to commit immediately or in the next step rather than burning the leftover
    turns commenting.
    """
    header = _build_status_header(
        current_step=current_step,
        max_steps=max_steps,
        searches_done=searches_done,
        max_search_calls=max_search_calls,
    )
    return (
        f"{header}\n"
        "Search budget exhausted — the `web_search` tool schema has been "
        "removed for all remaining steps. Include `\\boxed{...}` in this "
        "reply or the next to commit your final answer. The loop continues "
        "until `\\boxed{...}` appears or the step limit is reached."
    )


def build_continuation_after_unboxed_content(
    *,
    current_step: int,
    max_steps: int,
    searches_done: int,
    max_search_calls: int,
) -> str:
    """Injected when the previous step returned content but no parseable
    `\\boxed{...}` and we still have steps left.

    Replaces the old inline "Harness: step N complete — no \\boxed detected"
    user-message string in react.py. Keeping it in `prompts.py` means the
    full set of harness injections share both the status-header math and a
    consistent voice. The directive is intentionally permissive ("continue
    or commit") because a no-boxed content turn is normal mid-flight: the
    model is free to keep reasoning, search again, or wrap up — but it must
    know exactly where it stands first.
    """
    header = _build_status_header(
        current_step=current_step,
        max_steps=max_steps,
        searches_done=searches_done,
        max_search_calls=max_search_calls,
    )
    return (
        f"{header}\n"
        "Your previous reply did not contain a parseable `\\boxed{...}` "
        "answer, so the loop continues. Either keep investigating (more "
        "reasoning or another `web_search`) or include `\\boxed{...}` in "
        "your next reply to commit."
    )


def build_tool_error_status(
    *,
    current_step: int,
    max_steps: int,
    searches_done: int,
    max_search_calls: int,
    next_step_strips_tools: bool,
) -> str:
    """Status string attached to `tool_error` payloads.

    Tool-error turns happen INSIDE the assistant→tool→assistant cycle, so we
    cannot inject a user message there without breaking message ordering.
    Instead we surface the live status as a JSON `status` field inside the
    tool message payload, mirroring the wording the user-side injectors use.
    """
    if max_search_calls > 0:
        searches_left = max(max_search_calls - searches_done, 0)
        budget = (
            f"web_search {searches_done}/{max_search_calls} used "
            f"({searches_left} left)"
        )
    else:
        budget = "web_search disabled"
    suffix = (
        " The next step strips `web_search`; commit `\\boxed{...}` then."
        if next_step_strips_tools
        else " Output `\\boxed{...}` once your evidence is sufficient."
    )
    return (
        f"step {current_step}/{max_steps}, {budget}.{suffix}"
    )


def index_to_letter(i: int) -> str:
    if i < 0:
        raise ValueError(f"index must be >= 0, got {i}")
    return chr(ord("A") + i)


def letter_to_index(letter: str) -> int:
    if len(letter) != 1:
        raise ValueError(f"letter must be a single character, got {letter!r}")
    return ord(letter) - ord("A")


def _format_outcome_label(letter: str, label: str) -> str:
    """Wrap non-letter labels (>26 options: '[' '\\' ']' '^' '_' '`' 'a'..) in
    backticks so markdown doesn't swallow them, and pad with a leading space so
    a label that starts with a space / underscore still renders.
    """
    if ord(letter) in _BACKTICK_SAFE_ASCII_RANGE:
        return f"{letter}. {label}"
    return f"`{letter}`. {label}"


def _build_outcomes_block(options: list[str]) -> str:
    lines = [_format_outcome_label(index_to_letter(i), label) for i, label in enumerate(options)]
    return "\n" + "\n".join(lines)


def _assert_outer_template_contract(
    template_key: str,
    prompt_template: str,
    *,
    requires_outcomes_block: bool,
) -> None:
    required_slots = ("agent_role", "event", "end_time", "output_format", "guidance")
    missing = [slot for slot in required_slots if f"{{{slot}}}" not in prompt_template]
    if requires_outcomes_block and "{outcomes_block}" not in prompt_template:
        missing.append("outcomes_block")
    if missing:
        raise ValueError(
            f"{template_key} missing required slot(s): {', '.join(missing)}"
        )
    if _FINAL_FORMAT_REQUIREMENT in prompt_template or _BOXED_OUTPUT_MARKER in prompt_template:
        raise ValueError(
            f"{template_key} must keep final-answer format instructions inside output_format"
        )


def _assert_output_format_contract(q: Question, output_format: str) -> None:
    if _FINAL_FORMAT_REQUIREMENT not in output_format:
        raise ValueError(
            f"{q.question_type}/{q.choice_type} output_format must include "
            f"{_FINAL_FORMAT_REQUIREMENT!r}"
        )
    if _BOXED_OUTPUT_MARKER not in output_format:
        raise ValueError(
            f"{q.question_type}/{q.choice_type} output_format must include a boxed answer"
        )
    if q.question_type == "yes_no":
        required = ("\\boxed{Yes}", "\\boxed{No}")
        missing = [marker for marker in required if marker not in output_format]
        if missing:
            raise ValueError(
                "yes_no output_format missing required boxed option(s): "
                + ", ".join(missing)
            )
    elif q.question_type == "binary_named":
        if "<options[" in output_format:
            raise ValueError("binary_named output_format still contains option placeholders")
        if output_format.count(_BOXED_OUTPUT_MARKER) < 2:
            raise ValueError("binary_named output_format must box both named outcomes")


def render_user_prompt(
    q: Question,
    templates: dict[str, str],
    *,
    budget_awareness_protocol: str | None = None,
    reflection_protocol: str | None = None,
    belief_protocol: str | None = None,
) -> str:
    """Assemble the single user message handed to the LLM for one sample.

    All template text lives in `templates` (synced by the loader). This
    function selects the outer template by `q.question_type` and, for
    multiple-choice questions, by `q.choice_type`.

    `budget_awareness_protocol` (if provided) is appended FIRST so the model
    has end-to-end budget awareness before reading the methodology layers.
    `reflection_protocol` and `belief_protocol` follow in that order. None of
    these are part of `prompt_templates`, so the run's `prompt_templates_hash`
    is unchanged by these runtime additions; their presence is
    recorded via `Settings.config_snapshot` in `run_meta` and per-sample via
    the `user_prompt` field.
    """
    options = json.loads(q.options)

    if q.question_type == "yes_no":
        prompt_template_key = "yes_no_prompt_template"
        outcomes_block = ""
        output_format = templates["yes_no_output_format"]
        requires_outcomes_block = False

    elif q.question_type == "binary_named":
        prompt_template_key = "binary_named_prompt_template"
        if len(options) != 2:
            raise ValueError(
                f"binary_named question {q.id!r} must have exactly 2 options, got {len(options)}"
            )
        outcomes_block = ""
        output_format = (
            templates["binary_named_output_format"]
            .replace("<options[0]>", options[0])
            .replace("<options[1]>", options[1])
        )
        requires_outcomes_block = False

    elif q.question_type == "multiple_choice":
        outcomes_block = _build_outcomes_block(options)
        if q.choice_type == "single":
            prompt_template_key = "multiple_choice_single_prompt_template"
            output_format = templates["multiple_choice_single_output_format"]
        elif q.choice_type == "multi":
            prompt_template_key = "multiple_choice_multi_prompt_template"
            output_format = templates["multiple_choice_multi_output_format"]
        else:
            raise ValueError(
                f"multiple_choice question {q.id!r} has unknown choice_type {q.choice_type!r}"
            )
        requires_outcomes_block = True

    else:
        raise ValueError(f"unknown question_type: {q.question_type!r}")

    prompt_template = templates[prompt_template_key]
    _assert_outer_template_contract(
        prompt_template_key,
        prompt_template,
        requires_outcomes_block=requires_outcomes_block,
    )
    _assert_output_format_contract(q, output_format)

    rendered = prompt_template.format(
        agent_role=templates["agent_role"],
        event=q.event,
        end_time=q.end_time,
        outcomes_block=outcomes_block,
        output_format=output_format,
        guidance=templates["guidance"],
    )
    if rendered.count(_FINAL_FORMAT_REQUIREMENT) != 1:
        raise ValueError(
            "rendered prompt must contain exactly one final-answer format requirement"
        )
    if budget_awareness_protocol:
        rendered = rendered + budget_awareness_protocol
    if reflection_protocol:
        rendered = rendered + reflection_protocol
    if belief_protocol:
        rendered = rendered + belief_protocol
    return rendered
