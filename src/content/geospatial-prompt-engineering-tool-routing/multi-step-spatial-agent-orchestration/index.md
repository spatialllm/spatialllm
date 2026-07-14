# Multi-Step Spatial Agent Orchestration

Multi-step spatial agent orchestration is the discipline of running an agent that plans a sequence of geoprocessing calls, executes them one at a time, and validates the intermediate geometry after every step so a single bad operation cannot silently poison the rest of the chain. Part of the [Geospatial Prompt Engineering & Tool Routing](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/) area, this guide addresses the failure mode unique to multi-step spatial work: errors that are invisible at the step boundary but compound into a corrupted final result. It gives you an orchestrator with explicit state, checkpoints, and rollback.

A single tool call can be validated in isolation, but a chain of them accumulates risk. A buffer that quietly produces an empty geometry, a reprojection that shifts a feature off the map, an intersection that returns a `GeometryCollection` instead of a polygon — each is locally plausible and globally fatal. The orchestrator's job is to treat every intermediate geometry as untrusted, gate the transition between steps behind a validation predicate, checkpoint the last good state, and fall back deterministically the moment an invariant breaks rather than pressing on into nonsense.

<figure class="diagram">
<svg viewBox="0 0 860 300" role="img" aria-labelledby="msao-t msao-d" xmlns="http://www.w3.org/2000/svg">
  <title id="msao-t">Multi-step spatial agent orchestration loop</title>
  <desc id="msao-d">Plan step, invoke geoprocessing tool, validate intermediate geometry, checkpoint state, and continue or roll back run as a guarded loop so no bad step corrupts the chain.</desc>
  <defs>
    <marker id="msao-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/>
    </marker>
  </defs>
  <g fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2">
    <rect x="15" y="55" width="150" height="80" rx="8"/>
    <rect x="183" y="55" width="150" height="80" rx="8"/>
    <rect x="351" y="55" width="150" height="80" rx="8"/>
    <rect x="519" y="55" width="150" height="80" rx="8"/>
    <rect x="687" y="55" width="150" height="80" rx="8"/>
  </g>
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#msao-arrow)">
    <line x1="166" y1="95" x2="181" y2="95"/>
    <line x1="334" y1="95" x2="349" y2="95"/>
    <line x1="502" y1="95" x2="517" y2="95"/>
    <line x1="670" y1="95" x2="685" y2="95"/>
  </g>
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#msao-arrow)">
    <line x1="90" y1="137" x2="90" y2="202"/>
    <line x1="258" y1="137" x2="258" y2="202"/>
    <line x1="426" y1="137" x2="426" y2="202"/>
    <line x1="594" y1="137" x2="594" y2="202"/>
    <line x1="762" y1="137" x2="762" y2="202"/>
  </g>
  <rect x="15" y="205" width="822" height="58" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/>
  <g fill="#1f2937" font-size="13" text-anchor="middle">
    <text x="90" y="90"><tspan x="90" dy="0">Plan</tspan><tspan x="90" dy="16">step</tspan></text>
    <text x="258" y="90"><tspan x="258" dy="0">Invoke</tspan><tspan x="258" dy="16">tool</tspan></text>
    <text x="426" y="90"><tspan x="426" dy="0">Validate</tspan><tspan x="426" dy="16">geometry</tspan></text>
    <text x="594" y="90"><tspan x="594" dy="0">Checkpoint</tspan><tspan x="594" dy="16">state</tspan></text>
    <text x="762" y="90"><tspan x="762" dy="0">Continue</tspan><tspan x="762" dy="16">or roll back</tspan></text>
  </g>
  <text x="426" y="240" fill="#1f2937" font-size="15" font-weight="600" text-anchor="middle">Guarded per-step execution</text>
  <text x="426" y="287" fill="#5b6471" font-size="12" text-anchor="middle">On failure: rollback to last checkpoint · deterministic fallback</text>
</svg>
</figure>

## Foundational Principles

**Every intermediate geometry is untrusted until validated.** The output of step N is the input to step N+1, so an unchecked geometry propagates its defect down the whole chain. The orchestrator must run a validation predicate — non-empty, valid, correct type, plausible extent — before any step is allowed to consume the previous step's result.

**State is explicit and checkpointed, never implicit.** The agent's progress cannot live only in local variables or model context. Each accepted step writes a checkpoint holding the validated geometry and the step index, so the chain can resume from the last good point and roll back cleanly on failure instead of restarting from scratch.

**Failure is a first-class branch, not an exception you hope never fires.** When a step's output fails validation, the orchestrator rolls back to the last checkpoint and takes a deterministic fallback — a narrower operation, a safe default, or a clean abort — rather than feeding the model a broken geometry and asking it to improvise.

## Step-by-Step Implementation Pipeline

### 1. Model the plan and the checkpointed state

Represent the plan as an ordered list of typed steps and the run as a state object that carries the last validated geometry plus a stack of checkpoints. This makes rollback a pop, not a rerun.

```python
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any
from shapely.geometry.base import BaseGeometry

@dataclass
class Step:
    name: str
    tool: Callable[[BaseGeometry, Dict[str, Any]], BaseGeometry]
    params: Dict[str, Any]

@dataclass
class Checkpoint:
    step_index: int
    geometry: BaseGeometry

@dataclass
class OrchestrationState:
    current: BaseGeometry
    checkpoints: List[Checkpoint] = field(default_factory=list)
    step_index: int = 0

    def commit(self, geom: BaseGeometry) -> None:
        self.checkpoints.append(Checkpoint(self.step_index, geom))
        self.current = geom
        self.step_index += 1

    def rollback(self) -> Optional[Checkpoint]:
        return self.checkpoints[-1] if self.checkpoints else None
```

Because each `commit` snapshots the validated geometry, the run always has a well-defined point to return to. Deciding *which* fallback to take when a rollback fires draws on [Fallback Routing for Geospatial Queries](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/).

### 2. Run each step behind a geometry validation gate

The core loop invokes a tool, validates its output against explicit predicates, and only then commits a checkpoint. Any failure rolls back and triggers the deterministic fallback.

```python
import logging
logger = logging.getLogger("spatial_orchestrator")

def is_geometry_sound(geom: Optional[BaseGeometry]) -> bool:
    if geom is None or geom.is_empty:
        return False
    if not geom.is_valid:
        return False
    # Reject implausible planetary extents (WGS84 degrees).
    minx, miny, maxx, maxy = geom.bounds
    if not (-180 <= minx <= maxx <= 180 and -90 <= miny <= maxy <= 90):
        return False
    return True

def run_chain(
    plan: List[Step],
    state: OrchestrationState,
    fallback: Callable[[str, OrchestrationState], Dict[str, Any]],
) -> Dict[str, Any]:
    for step in plan:
        try:
            result = step.tool(state.current, step.params)
        except Exception as exc:
            logger.error("step %s raised: %s", step.name, exc)
            return fallback(f"TOOL_ERROR:{step.name}", state)

        if not is_geometry_sound(result):
            logger.warning("step %s produced unsound geometry", step.name)
            last = state.rollback()
            reason = f"VALIDATION_FAIL:{step.name}@{last.step_index if last else -1}"
            return fallback(reason, state)

        state.commit(result)
        logger.info("committed checkpoint %d after %s", state.step_index - 1, step.name)

    return {"status": "ok", "geometry": state.current, "steps": state.step_index}
```

The gate is deliberately strict: an empty or invalid geometry, or one whose bounds escape the globe, halts the chain before the next step consumes it. This same per-step validation discipline underlies [LLM-Assisted Geoprocessing Pipelines](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/), which frames the broader pipeline these agents run inside.

### 3. Persist checkpoints so a chain can resume

For long or asynchronous chains, in-memory checkpoints are not enough — a crashed worker must resume from the last validated geometry. Serialize each checkpoint to durable storage keyed by run id, and key each step by an input hash so a resumed run skips already-completed, idempotent steps rather than recomputing them. The concrete persistence and resume mechanics are worked out in [Chaining Geoprocessing Tools with State Checkpoints](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/chaining-geoprocessing-tools-with-state-checkpoints/).

## Failure Modes & Root Causes

**Silent empty-geometry propagation.** An intersection with no overlap returns an empty geometry, which every subsequent step happily processes into more emptiness. The root cause is treating "no error raised" as "success"; the fix is the explicit non-empty gate before each commit.

**Lost work on transient failure.** A network blip in step 7 of a 10-step chain restarts everything when there is no checkpoint. The root cause is implicit state; the fix is a durable checkpoint after every validated step.

**CRS drift across steps.** One tool returns metres, the next assumes degrees, and the extent check is the only thing standing between the agent and a feature launched into orbit. The root cause is unvalidated CRS assumptions between steps; the fix is asserting the working CRS as part of the soundness gate.

**Rollback to a corrupt checkpoint.** If a defective geometry was committed because the gate was too weak, rollback returns to a poisoned state. The root cause is an under-specified validation predicate; the fix is to make `is_geometry_sound` conservative and to never commit without it.

## Production Validation Protocols

1. Assert `is_geometry_sound` is called before every `commit` — enforce with a test that patches the gate and confirms no checkpoint is written when it returns `False`.
2. Assert that a tool raising an exception yields a deterministic fallback and leaves the checkpoint stack unchanged from the last good state.
3. Simulate a mid-chain failure and assert the run resumes from the correct checkpoint index rather than step zero.
4. Track per-step validation-failure rate and rollback count as observability KPIs; a rising rollback rate signals a drifting planner or a flaky tool.
5. Assert idempotency: replaying a step with an identical input hash produces an identical checkpoint and no duplicate side effects.

## Related

- Up to the area overview: [Geospatial Prompt Engineering & Tool Routing](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/)
- [Chaining Geoprocessing Tools with State Checkpoints](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/chaining-geoprocessing-tools-with-state-checkpoints/)
- [Fallback Routing for Geospatial Queries](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/)
- [LLM-Assisted Geoprocessing Pipelines](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/)
- [Spatial Function-Calling Schemas](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/)
