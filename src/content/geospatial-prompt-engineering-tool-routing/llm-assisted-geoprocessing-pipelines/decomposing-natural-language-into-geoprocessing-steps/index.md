---
title: Decomposing Natural Language into Geoprocessing Steps
description: Turn an analysis request into a typed, checkable plan of geoprocessing steps with named inputs, before any operation runs.
slug: decomposing-natural-language-into-geoprocessing-steps
type: howto
breadcrumb: Decomposing a Request
datePublished: 2025-03-26
dateModified: 2026-08-11
---

# Decomposing Natural Language into Geoprocessing Steps

A request like "find schools within 500 m of a flood zone that lost power last night" is not one operation — it is a buffer, a spatial join, an attribute filter, and a temporal filter, in a specific order. Executing an LLM's free-form answer directly is how pipelines end up running `ST_Union` on the wrong layer or buffering in degrees. This guide converts natural language into an ordered, typed, and validated geoprocessing plan *before* a single operation runs, as part of [LLM-assisted geoprocessing pipelines](/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/).

The plan is the contract. Rather than let the model emit SQL or Python, it emits a structured list of steps, each naming one operation from a fixed allowlist and supplying typed arguments. A deterministic parser then validates every step — unknown op, missing argument, wrong type, or a reference to a layer that does not exist all fail closed. Only a fully valid plan advances to execution.

## When to Use This Approach

Use plan decomposition whenever the model's output will drive real geoprocessing rather than just answer a question in prose. The typed-plan indirection costs a little latency and prompt engineering, but it converts an open-ended code-generation risk into a bounded schema-validation problem.

| Style | Model emits | Safety | Best for |
|---|---|---|---|
| Direct execution | Raw SQL / Python | Low — arbitrary ops | Throwaway prototypes |
| Typed plan (this page) | Op + args JSON | High — allowlist + schema | Production agents |
| Fixed template | Slot values only | Highest — no op choice | Narrow, known queries |

Choose the typed plan when the space of legitimate requests is broad but the space of legitimate *operations* is small and known. If you only ever run two or three query shapes, a fixed template is stricter and simpler. If you genuinely need arbitrary computation, you have a much larger sandboxing problem than this page addresses. Downstream, each planned step should still be checked at runtime — see [validating intermediate geoprocessing outputs](/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/validating-intermediate-geoprocessing-outputs/).

<figure class="diagram">
<svg viewBox="16 38 748 188" role="img" aria-labelledby="dnl-two-t dnl-two-d" xmlns="http://www.w3.org/2000/svg"><title id="dnl-two-t">A named plan against a single generated call</title><desc id="dnl-two-d">A plan of named steps can be inspected, corrected and partially re-run, while one opaque call can only be accepted or thrown away.</desc><rect x="16" y="38" width="748" height="188" fill="#ffffff"/><rect x="30" y="52" width="340" height="160" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="410" y="52" width="340" height="160" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="200" y="84">a plan of named steps</text><text x="580" y="84">one generated call</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="200" y="114">each step is inspectable</text><text x="200" y="140">a wrong step is correctable</text><text x="200" y="166">re-running is partial</text><text x="580" y="114">opaque until it finishes</text><text x="580" y="140">wrong means start again</text><text x="580" y="166">no partial recovery</text></g></svg>
<figcaption><b>The plan is the artefact worth producing.</b> It survives review, correction and re-execution, none of which an opaque call supports.</figcaption>
</figure>

## Implementation

The parser below accepts the model's JSON plan, validates each step against an operation allowlist with an argument schema, and rejects the whole plan on any violation. When parsing or validation fails, it returns a deterministic single-step fallback plan that the executor can run safely.

```python
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("geo_planner")

# Allowlist: op name -> (required arg name -> validator predicate).
Validator = Callable[[Any], bool]
OP_SCHEMA: dict[str, dict[str, Validator]] = {
    "buffer":     {"layer": lambda v: isinstance(v, str),
                   "distance_m": lambda v: isinstance(v, (int, float)) and 0 < v <= 50000},
    "spatial_join": {"left": lambda v: isinstance(v, str),
                     "right": lambda v: isinstance(v, str),
                     "predicate": lambda v: v in {"intersects", "within", "dwithin"}},
    "attr_filter": {"layer": lambda v: isinstance(v, str),
                    "column": lambda v: isinstance(v, str) and v.isidentifier(),
                    "op": lambda v: v in {"=", "!=", ">", "<", ">=", "<="},
                    "value": lambda v: isinstance(v, (int, float, str, bool))},
}
KNOWN_LAYERS = {"schools", "flood_zones", "outages", "parcels"}


@dataclass
class Step:
    op: str
    args: dict[str, Any]


@dataclass
class Plan:
    steps: list[Step] = field(default_factory=list)
    fallback: bool = False


class PlanError(Exception):
    pass


def _validate_step(raw: Any, idx: int) -> Step:
    if not isinstance(raw, dict) or "op" not in raw:
        raise PlanError(f"step {idx}: not an object with an 'op'")
    op = raw["op"]
    schema = OP_SCHEMA.get(op)
    if schema is None:
        raise PlanError(f"step {idx}: op '{op}' not in allowlist")
    args = raw.get("args", {})
    if not isinstance(args, dict):
        raise PlanError(f"step {idx}: args must be an object")
    missing = set(schema) - set(args)
    if missing:
        raise PlanError(f"step {idx}: missing args {sorted(missing)}")
    for name, check in schema.items():
        if not check(args[name]):
            raise PlanError(f"step {idx}: arg '{name}' failed validation")
    # Cross-check any layer reference against the known catalogue.
    for key in ("layer", "left", "right"):
        if key in args and args[key] not in KNOWN_LAYERS:
            raise PlanError(f"step {idx}: unknown layer '{args[key]}'")
    return Step(op=op, args=args)


def _safe_fallback() -> Plan:
    # Deterministic minimal plan: return the base layer with no transformation.
    return Plan(steps=[Step(op="attr_filter",
                            args={"layer": "schools", "column": "id",
                                  "op": ">", "value": 0})],
                fallback=True)


def parse_plan(model_output: str) -> Plan:
    try:
        payload = json.loads(model_output)
        raw_steps = payload["steps"]
        if not isinstance(raw_steps, list) or not raw_steps:
            raise PlanError("plan has no steps")
        if len(raw_steps) > 12:
            raise PlanError("plan exceeds step budget")
        steps = [_validate_step(s, i) for i, s in enumerate(raw_steps)]
        log.info("accepted plan with %d steps", len(steps))
        return Plan(steps=steps)
    except (json.JSONDecodeError, KeyError, TypeError, PlanError) as exc:
        log.warning("plan rejected (%s); using deterministic fallback", exc)
        return _safe_fallback()


if __name__ == "__main__":
    demo = json.dumps({"steps": [
        {"op": "buffer", "args": {"layer": "flood_zones", "distance_m": 500}},
        {"op": "spatial_join", "args": {"left": "schools", "right": "flood_zones",
                                        "predicate": "dwithin"}},
    ]})
    plan = parse_plan(demo)
    for s in plan.steps:
        print(s.op, s.args)
```

Every argument is checked for type *and* range (a buffer distance is capped, a column name must be a valid identifier), and every layer reference is confirmed against a known catalogue. A malformed or hallucinated plan never reaches the executor; it degrades to the fallback, which the orchestrator can flag for a re-prompt. This mirrors the safe-degradation posture in [multi-step spatial agent orchestration](/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/).

<figure class="diagram">
<svg viewBox="13 38 753 272" role="img" aria-labelledby="dnl-amb-t dnl-amb-d" xmlns="http://www.w3.org/2000/svg"><title id="dnl-amb-t">Where a request is ambiguous and what to do about it</title><desc id="dnl-amb-d">Most requests under-specify units, reference systems and what counts as nearby, and each of these has a defensible default that must be stated rather than assumed silently.</desc><rect x="13" y="38" width="753" height="272" fill="#ffffff"/><rect x="30" y="52" width="700" height="46" rx="6" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="50" y="80" fill="#1f2937" font-size="12.5">units: metres or feet — state the assumption</text><rect x="30" y="108" width="640" height="46" rx="6" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="50" y="136" fill="#1f2937" font-size="12.5">reference system: which projection the distance is measured in</text><rect x="30" y="164" width="560" height="46" rx="6" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="50" y="192" fill="#1f2937" font-size="12.5">nearby: a threshold the reader did not give</text><rect x="30" y="220" width="460" height="46" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="50" y="248" fill="#1f2937" font-size="12.5">silently assumed: the answer is confidently wrong</text><text x="390" y="294" fill="#1f2937" font-size="13" text-anchor="middle">An assumption stated in the plan is checkable; the same assumption applied silently is not</text></svg>
<figcaption><b>Ambiguity is not a failure of the request.</b> People speak in approximations, and the job of the plan is to turn each approximation into a stated value rather than a hidden one.</figcaption>
</figure>

## Validation & Testing

- **Unknown ops fail closed.** Feed a plan containing `{"op": "rm_rf"}` and assert `parse_plan` returns a plan with `fallback=True` and never raises.
- **Type and range enforcement.** Assert a `buffer` with `distance_m` of `-5` or `"500"` is rejected, while `500` is accepted — proving both the type check and the `0 < v <= 50000` bound fire.
- **Layer catalogue guard.** Assert a `spatial_join` referencing `right="census_blocks"` (not in `KNOWN_LAYERS`) yields the fallback, confirming hallucinated layer names cannot slip through.

## Gotchas & Edge Cases

- **Valid JSON, invalid semantics.** The model may emit a well-formed `spatial_join` before the `buffer` that produces its right layer. Add an ordering pass that checks each step's inputs are produced by an earlier step or exist in the catalogue, not just that each step is individually well-typed.
- **Silent arg coercion.** Accepting `"500"` and casting it to `500` invites unit and precision drift. Reject the wrong type outright and force the model to emit numbers as numbers.
- **Unbounded plans.** Without a step budget, a runaway model can emit hundreds of ops. Cap plan length (here, 12) and treat the overflow as a rejection, not a truncation.
- **Allowlist drift.** As new ops are added, the schema and the executor can fall out of sync, letting a planned op reach an executor that cannot run it. Generate the allowlist from the executor's registered handlers so the two share one source of truth.

<figure class="diagram">
<svg viewBox="16 32 748 214" role="img" aria-labelledby="dnl-check-t dnl-check-d" xmlns="http://www.w3.org/2000/svg"><title id="dnl-check-t">What each step declares before it runs</title><desc id="dnl-check-d">Naming the inputs, outputs and preconditions of every step is what lets the plan be validated as a whole before any of it executes.</desc><rect x="16" y="32" width="748" height="214" fill="#ffffff"/><rect x="30" y="46" width="360" height="86" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="52" y="76" fill="#1f2937" font-size="13" font-weight="600">inputs</text><text x="52" y="102" fill="#5b6471" font-size="12">named, with types</text><text x="52" y="122" fill="#5b6471" font-size="12">so a mismatch is visible</text><rect x="410" y="46" width="340" height="86" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="432" y="76" fill="#1f2937" font-size="13" font-weight="600">output</text><text x="432" y="102" fill="#5b6471" font-size="12">named, so the next step can use it</text><text x="432" y="122" fill="#5b6471" font-size="12">and it can be cached</text><rect x="30" y="146" width="360" height="86" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="52" y="176" fill="#1f2937" font-size="13" font-weight="600">preconditions</text><text x="52" y="202" fill="#5b6471" font-size="12">same reference system, valid geometry</text><text x="52" y="222" fill="#5b6471" font-size="12">checked before running</text><rect x="410" y="146" width="340" height="86" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="432" y="176" fill="#1f2937" font-size="13" font-weight="600">cost estimate</text><text x="432" y="202" fill="#5b6471" font-size="12">rough, from row counts</text><text x="432" y="222" fill="#5b6471" font-size="12">so the plan can be budgeted</text></svg>
<figcaption><b>A plan whose steps declare this much can be checked without executing.</b> That is the difference between finding a mismatch in a second and finding it after four minutes of overlay.</figcaption>
</figure>

## Operating This Step Over Time

Plans drift toward length. A model that produced four steps for a class of request will produce seven once the prompt has accumulated examples, and the extra steps are usually decomposition of things that were fine as single operations. Tracking the median step count per request type is a cheap way to notice, and the fix is to prune examples rather than to instruct against it.

The failures worth logging are the ones where the plan was well-formed and the answer was wrong, because those are the cases the validation cannot catch. Almost all of them turn out to be a stated assumption that was wrong rather than a step that was — which is an argument for surfacing the assumptions to the reader rather than only recording them.

Vocabulary shifts too. New operations get added to the available set and the prompt's examples do not mention them, so plans keep using an older, longer route to the same result. Reviewing which operations are never planned is the check that catches it.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should the plan be shown to the user before it runs?</span></summary><p>For anything expensive or destructive, yes; for a cheap read, no. The value of showing it is that stated assumptions get corrected before four minutes of computation rather than after, and the cost is a confirmation step in a conversation that felt fluid. Tying the confirmation to estimated cost gets both — the plan appears exactly when it is worth a reader's attention.</p></details>

<details class="faq-item"><summary><span>What if the request genuinely cannot be decomposed?</span></summary><p>Say so and name what is missing. A request that refers to data that is not available, or asks for an operation that is not implemented, has no valid plan, and producing a plausible-looking one is much worse than refusing. The useful refusal names the specific gap — this layer, that operation — rather than reporting that the request was not understood.</p></details>

<details class="faq-item"><summary><span>How many steps is too many?</span></summary><p>More than about eight usually means the request contained several questions. Splitting it and answering them in sequence produces better answers and much better failure behaviour, because a failure in one no longer discards the others. It also tends to reveal that the reader wanted two of the three parts.</p></details>

<details class="faq-item"><summary><span>Does the plan need to be a formal structure?</span></summary><p>It needs to be machine-readable, which in practice means a list of objects with named operations and arguments. Prose plans are pleasant to read and cannot be validated, and validation before execution is most of the value. Rendering the structured plan back into a sentence for the reader gives both.</p></details>

## Related

- Up to the section: [LLM-Assisted Geoprocessing Pipelines](/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/)
- [Validating Intermediate Geoprocessing Outputs](/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/validating-intermediate-geoprocessing-outputs/)
- [Multi-Step Spatial Agent Orchestration](/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/)
- [Chaining Geoprocessing Tools with State Checkpoints](/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/chaining-geoprocessing-tools-with-state-checkpoints/)
