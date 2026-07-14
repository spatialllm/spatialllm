# Decomposing Natural Language into Geoprocessing Steps

A request like "find schools within 500 m of a flood zone that lost power last night" is not one operation — it is a buffer, a spatial join, an attribute filter, and a temporal filter, in a specific order. Executing an LLM's free-form answer directly is how pipelines end up running `ST_Union` on the wrong layer or buffering in degrees. This guide converts natural language into an ordered, typed, and validated geoprocessing plan *before* a single operation runs, as part of [LLM-assisted geoprocessing pipelines](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/).

The plan is the contract. Rather than let the model emit SQL or Python, it emits a structured list of steps, each naming one operation from a fixed allowlist and supplying typed arguments. A deterministic parser then validates every step — unknown op, missing argument, wrong type, or a reference to a layer that does not exist all fail closed. Only a fully valid plan advances to execution.

## When to Use This Approach

Use plan decomposition whenever the model's output will drive real geoprocessing rather than just answer a question in prose. The typed-plan indirection costs a little latency and prompt engineering, but it converts an open-ended code-generation risk into a bounded schema-validation problem.

| Style | Model emits | Safety | Best for |
|---|---|---|---|
| Direct execution | Raw SQL / Python | Low — arbitrary ops | Throwaway prototypes |
| Typed plan (this page) | Op + args JSON | High — allowlist + schema | Production agents |
| Fixed template | Slot values only | Highest — no op choice | Narrow, known queries |

Choose the typed plan when the space of legitimate requests is broad but the space of legitimate *operations* is small and known. If you only ever run two or three query shapes, a fixed template is stricter and simpler. If you genuinely need arbitrary computation, you have a much larger sandboxing problem than this page addresses. Downstream, each planned step should still be checked at runtime — see [validating intermediate geoprocessing outputs](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/validating-intermediate-geoprocessing-outputs/).

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

Every argument is checked for type *and* range (a buffer distance is capped, a column name must be a valid identifier), and every layer reference is confirmed against a known catalogue. A malformed or hallucinated plan never reaches the executor; it degrades to the fallback, which the orchestrator can flag for a re-prompt. This mirrors the safe-degradation posture in [multi-step spatial agent orchestration](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/).

## Validation & Testing

- **Unknown ops fail closed.** Feed a plan containing `{"op": "rm_rf"}` and assert `parse_plan` returns a plan with `fallback=True` and never raises.
- **Type and range enforcement.** Assert a `buffer` with `distance_m` of `-5` or `"500"` is rejected, while `500` is accepted — proving both the type check and the `0 < v <= 50000` bound fire.
- **Layer catalogue guard.** Assert a `spatial_join` referencing `right="census_blocks"` (not in `KNOWN_LAYERS`) yields the fallback, confirming hallucinated layer names cannot slip through.

## Gotchas & Edge Cases

- **Valid JSON, invalid semantics.** The model may emit a well-formed `spatial_join` before the `buffer` that produces its right layer. Add an ordering pass that checks each step's inputs are produced by an earlier step or exist in the catalogue, not just that each step is individually well-typed.
- **Silent arg coercion.** Accepting `"500"` and casting it to `500` invites unit and precision drift. Reject the wrong type outright and force the model to emit numbers as numbers.
- **Unbounded plans.** Without a step budget, a runaway model can emit hundreds of ops. Cap plan length (here, 12) and treat the overflow as a rejection, not a truncation.
- **Allowlist drift.** As new ops are added, the schema and the executor can fall out of sync, letting a planned op reach an executor that cannot run it. Generate the allowlist from the executor's registered handlers so the two share one source of truth.

## Related

- Up to the section: [LLM-Assisted Geoprocessing Pipelines](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/)
- [Validating Intermediate Geoprocessing Outputs](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/validating-intermediate-geoprocessing-outputs/)
- [Multi-Step Spatial Agent Orchestration](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/)
- [Chaining Geoprocessing Tools with State Checkpoints](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/chaining-geoprocessing-tools-with-state-checkpoints/)
