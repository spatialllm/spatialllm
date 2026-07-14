# Chaining Geoprocessing Tools with State Checkpoints

Chaining geoprocessing tools with state checkpoints means persisting the validated geometry between each tool in an agent's chain so the run can be validated, resumed after a crash, and rolled back without recomputing work — and so replaying a step keyed by its input hash is a safe no-op. This guide sits under [Multi-Step Spatial Agent Orchestration](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/) and focuses on the durable-state layer that makes long spatial chains recoverable in production.

The problem is recovery. An agent that runs eight geoprocessing steps in memory loses everything when its worker is preempted, and a naive retry re-executes side-effecting steps twice. Durable checkpoints keyed by a content hash of each step's input turn the chain into something you can pause, inspect, resume from the exact point of failure, and replay safely.

## When to Use This Approach

Checkpoint persistence earns its complexity when a chain is long enough, slow enough, or side-effecting enough that losing progress is expensive. For a two-step, read-only transform, in-memory state is fine.

| Signal | Checkpoint to durable store | Keep in memory |
| --- | --- | --- |
| Chain length | Many steps, minutes-long | Two or three fast steps |
| Failure cost | Re-running is expensive/slow | Re-running is trivial |
| Side effects | Steps write to stores/APIs | Pure, idempotent reads |
| Execution model | Async workers, may be preempted | Single synchronous call |

For the async execution model that makes resumable chains necessary, see [Async vs Sync Geoprocessing Workflows](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/).

## Implementation

The executor below persists each validated geometry as GeoJSON to a checkpoint store keyed by `run_id` and step index. Each step is keyed by a hash of its serialized input geometry plus its parameters; if a checkpoint for that key already exists, the step is skipped (idempotent resume). On failure, the chain returns the last durable checkpoint and a deterministic fallback rather than a half-written result. The store here is a simple JSON-file backend, but the interface is the contract you would implement against Redis or a database.

```python
import json, hashlib, logging
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional
from shapely.geometry import shape, mapping
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger("checkpoint_chain")

def step_key(geom: BaseGeometry, params: Dict[str, Any]) -> str:
    payload = json.dumps(
        {"geom": mapping(geom), "params": params}, sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

class CheckpointStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str, key: str) -> Path:
        return self.root / f"{run_id}__{key}.geojson"

    def load(self, run_id: str, key: str) -> Optional[BaseGeometry]:
        p = self._path(run_id, key)
        if not p.exists():
            return None
        try:
            return shape(json.loads(p.read_text())["geometry"])
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("corrupt checkpoint %s: %s", p.name, exc)
            return None  # treat corruption as a miss; step will recompute

    def save(self, run_id: str, key: str, geom: BaseGeometry) -> None:
        tmp = self._path(run_id, key).with_suffix(".tmp")
        tmp.write_text(json.dumps({"geometry": mapping(geom)}))
        tmp.replace(self._path(run_id, key))  # atomic swap

def validated(geom: Optional[BaseGeometry]) -> bool:
    return geom is not None and not geom.is_empty and geom.is_valid

def run_checkpointed_chain(
    run_id: str,
    seed: BaseGeometry,
    steps: List[Dict[str, Any]],
    store: CheckpointStore,
    fallback: Callable[[str, BaseGeometry], Dict[str, Any]],
) -> Dict[str, Any]:
    current = seed
    for i, step in enumerate(steps):
        key = step_key(current, step["params"])
        cached = store.load(run_id, key)
        if cached is not None and validated(cached):
            logger.info("resume: step %d hit checkpoint %s", i, key)
            current = cached
            continue
        try:
            result = step["tool"](current, step["params"])
        except Exception as exc:
            logger.error("step %d (%s) failed: %s", i, step["name"], exc)
            return fallback(f"TOOL_ERROR:{step['name']}", current)

        if not validated(result):
            logger.warning("step %d produced invalid geometry", i)
            return fallback(f"VALIDATION_FAIL:{step['name']}", current)

        store.save(run_id, key, result)  # durable before advancing
        current = result

    return {"status": "ok", "run_id": run_id, "geometry": mapping(current)}
```

Two details make this safe under real failure. The `save` writes to a temporary file and atomically `replace`s the target, so a crash mid-write never leaves a half-written checkpoint that a resume would trust. And because the checkpoint key is a hash of the *input* geometry and parameters, re-running the chain after a partial failure recomputes only the steps past the last durable checkpoint — earlier steps hit their cached GeoJSON and are skipped. When a fallback fires, choosing the recovery path connects to the schema-bounded calls described in [Spatial Function-Calling Schemas](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/).

## Validation & Testing

- **Resume skips completed work.** Run a three-step chain, delete the third checkpoint, rerun with the same `run_id`, and assert a spy on step one and two records zero invocations while step three runs exactly once.
- **Atomic writes survive a crash.** Simulate a failure between `write_text` and `replace` and assert `load` returns `None` (a miss) rather than a truncated geometry, proving no corrupt state is ever trusted.
- **Idempotent key stability.** Assert `step_key(g, p) == step_key(g, p)` across processes for identical input, and that a changed parameter yields a different key — guarding against stale-checkpoint reuse.
- **Invalid intermediate halts the chain.** Feed a step that returns an empty geometry and assert the run returns the deterministic fallback with the last valid `current` unchanged.

## Gotchas & Edge Cases

**Coordinate precision in the hash.** Serializing full float precision makes two geometrically identical geometries hash differently after a round-trip through GeoJSON. Round coordinates to a fixed grid before hashing so resume actually hits the cache.

**Non-idempotent tools behind the cache.** The skip-on-hit logic assumes a step is a pure function of its input. A tool that also writes to an external store must make that write idempotent (upsert keyed by the same hash), or the resume will skip the geometry but miss the side effect.

**Unbounded checkpoint growth.** Every step of every run writes a file; without a TTL or a sweep keyed on `run_id`, the store grows without limit. Expire checkpoints once a run reaches a terminal state.

**CRS not captured in the checkpoint.** Storing bare GeoJSON drops the SRID, so a resumed step may assume the wrong CRS. Persist the CRS alongside the geometry and assert it on load.

## Related

- Up to the section overview: [Multi-Step Spatial Agent Orchestration](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/)
- [Spatial Function-Calling Schemas](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/)
- [Async vs Sync Geoprocessing Workflows](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/)
