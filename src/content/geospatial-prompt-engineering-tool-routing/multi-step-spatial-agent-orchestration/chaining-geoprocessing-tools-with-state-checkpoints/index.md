---
title: Chaining Geoprocessing Tools with State Checkpoints
description: Persist step outputs by reference so a long chain resumes from its last checkpoint instead of replaying everything that already succeeded.
slug: chaining-geoprocessing-tools-with-state-checkpoints
type: howto
breadcrumb: State Checkpoints
datePublished: 2025-04-09
dateModified: 2026-08-11
---

# Chaining Geoprocessing Tools with State Checkpoints

Chaining geoprocessing tools with state checkpoints means persisting the validated geometry between each tool in an agent's chain so the run can be validated, resumed after a crash, and rolled back without recomputing work — and so replaying a step keyed by its input hash is a safe no-op. This guide sits under [Multi-Step Spatial Agent Orchestration](/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/) and focuses on the durable-state layer that makes long spatial chains recoverable in production.

The problem is recovery. An agent that runs eight geoprocessing steps in memory loses everything when its worker is preempted, and a naive retry re-executes side-effecting steps twice. Durable checkpoints keyed by a content hash of each step's input turn the chain into something you can pause, inspect, resume from the exact point of failure, and replay safely.

## When to Use This Approach

Checkpoint persistence earns its complexity when a chain is long enough, slow enough, or side-effecting enough that losing progress is expensive. For a two-step, read-only transform, in-memory state is fine.

| Signal | Checkpoint to durable store | Keep in memory |
| --- | --- | --- |
| Chain length | Many steps, minutes-long | Two or three fast steps |
| Failure cost | Re-running is expensive/slow | Re-running is trivial |
| Side effects | Steps write to stores/APIs | Pure, idempotent reads |
| Execution model | Async workers, may be preempted | Single synchronous call |

For the async execution model that makes resumable chains necessary, see [Async vs Sync Geoprocessing Workflows](/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/).

<figure class="diagram">
<svg viewBox="16 38 748 188" role="img" aria-labelledby="cgt-two-t cgt-two-d" xmlns="http://www.w3.org/2000/svg"><title id="cgt-two-t">Re-running from a checkpoint against re-running the chain</title><desc id="cgt-two-d">A checkpointed chain resumes at the step that failed; an unchecked one repeats every step before it, paying for work that already succeeded.</desc><rect x="16" y="38" width="748" height="188" fill="#ffffff"/><rect x="30" y="52" width="340" height="160" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="410" y="52" width="340" height="160" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="200" y="84">resume from the checkpoint</text><text x="580" y="84">replay the whole chain</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="200" y="114">only the failed step re-runs</text><text x="200" y="140">cost is proportional to the fault</text><text x="200" y="166">state is inspectable</text><text x="580" y="114">every earlier step repeats</text><text x="580" y="140">cost is proportional to depth</text><text x="580" y="166">no intermediate state</text></g></svg>
<figcaption><b>The deeper the chain, the larger the difference.</b> A failure at step six of seven costs one step with checkpoints and six without, and that is exactly where failures are most likely.</figcaption>
</figure>

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

Two details make this safe under real failure. The `save` writes to a temporary file and atomically `replace`s the target, so a crash mid-write never leaves a half-written checkpoint that a resume would trust. And because the checkpoint key is a hash of the *input* geometry and parameters, re-running the chain after a partial failure recomputes only the steps past the last durable checkpoint — earlier steps hit their cached GeoJSON and are skipped. When a fallback fires, choosing the recovery path connects to the schema-bounded calls described in [Spatial Function-Calling Schemas](/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/).

<figure class="diagram">
<svg viewBox="16 32 748 214" role="img" aria-labelledby="cgt-what-t cgt-what-d" xmlns="http://www.w3.org/2000/svg"><title id="cgt-what-t">What a checkpoint has to record</title><desc id="cgt-what-d">A reference to the output, the inputs that produced it, the parameters used and a fingerprint — enough to decide whether the checkpoint is still valid.</desc><rect x="16" y="32" width="748" height="214" fill="#ffffff"/><rect x="30" y="46" width="360" height="86" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="52" y="76" fill="#1f2937" font-size="13" font-weight="600">output reference</text><text x="52" y="102" fill="#5b6471" font-size="12">not the data itself</text><text x="52" y="122" fill="#5b6471" font-size="12">the state stays small</text><rect x="410" y="46" width="340" height="86" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="432" y="76" fill="#1f2937" font-size="13" font-weight="600">input references</text><text x="432" y="102" fill="#5b6471" font-size="12">what produced it</text><text x="432" y="122" fill="#5b6471" font-size="12">so staleness is detectable</text><rect x="30" y="146" width="360" height="86" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="52" y="176" fill="#1f2937" font-size="13" font-weight="600">parameters</text><text x="52" y="202" fill="#5b6471" font-size="12">tolerance, distance, projection</text><text x="52" y="222" fill="#5b6471" font-size="12">a change invalidates it</text><rect x="410" y="146" width="340" height="86" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="432" y="176" fill="#1f2937" font-size="13" font-weight="600">fingerprint</text><text x="432" y="202" fill="#5b6471" font-size="12">of inputs and parameters</text><text x="432" y="222" fill="#5b6471" font-size="12">the reuse decision in one value</text></svg>
<figcaption><b>The fingerprint is what makes reuse safe.</b> Without it, resuming from a checkpoint means trusting that nothing upstream has changed, which is an assumption nobody checks.</figcaption>
</figure>

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

<figure class="diagram">
<svg viewBox="16 38 728 212" role="img" aria-labelledby="cgt-stale-t cgt-stale-d" xmlns="http://www.w3.org/2000/svg"><title id="cgt-stale-t">When a checkpoint must be discarded</title><desc id="cgt-stale-d">A checkpoint is reusable only while its inputs and parameters are unchanged; anything else makes it a cached wrong answer.</desc><rect x="16" y="38" width="728" height="212" fill="#ffffff"/><rect x="30" y="52" width="700" height="46" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="50" y="80" fill="#1f2937" font-size="12.5">inputs and parameters unchanged: reuse it</text><rect x="30" y="108" width="600" height="46" rx="6" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="50" y="136" fill="#1f2937" font-size="12.5">a parameter changed: discard and recompute</text><rect x="30" y="164" width="700" height="46" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="50" y="192" fill="#1f2937" font-size="12.5">an upstream layer was updated: discard the whole tail</text><text x="390" y="238" fill="#1f2937" font-size="13" text-anchor="middle">Reusing a stale checkpoint produces an answer that is wrong and fast</text></svg>
<figcaption><b>Discarding the tail matters as much as discarding the step.</b> A changed input invalidates everything computed from it, and stopping at the first checkpoint leaves the rest confidently stale.</figcaption>
</figure>

## Operating This Step Over Time

Checkpoint storage grows without anyone noticing, because each individual checkpoint is small and nothing deletes them. A time-to-live tied to how long a conversation can plausibly resume — hours, not weeks — keeps it bounded, and expiring a checkpoint is safe by construction since the worst case is recomputation.

The number worth watching is the reuse rate. Checkpoints that are written and never read are pure overhead, and a low reuse rate usually means either that failures are rare, which is good news that should let you checkpoint less, or that the fingerprint is too strict and is invalidating things needlessly.

Watch for chains that always fail at the same step. That is a step-level problem being absorbed by the checkpoint machinery, which makes it cheap enough to stop being investigated — which is precisely how a systematic failure becomes permanent.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should every step be checkpointed?</span></summary><p>Only the expensive ones. A checkpoint costs a write and a fingerprint computation, and for a step that runs in milliseconds that is more than recomputation. The rule that works is to checkpoint any step whose cost exceeds the cost of storing and validating its output, which in practice means the overlays and dissolves and not the filters.</p></details>

<details class="faq-item"><summary><span>Where should intermediate results be stored?</span></summary><p>Wherever the next step will read them from, which usually means the same store as the input data — a temporary table beside the source rather than a file beside the process. That keeps the resume path identical to the original path, and it avoids the case where a checkpoint is unreachable from the worker that picks the chain back up.</p></details>

<details class="faq-item"><summary><span>How does this interact with a failed step that is not retryable?</span></summary><p>The checkpoint stays valid and the chain stops there. That is the useful case: the earlier work is preserved, the failure is attributable to one named step, and a corrected parameter can resume from the last good state rather than from the beginning. A non-retryable failure is a reason to keep the checkpoints, not to discard them.</p></details>

<details class="faq-item"><summary><span>Can checkpoints be shared between conversations?</span></summary><p>Technically yes, through the fingerprint, and it is worth doing for expensive steps over shared reference data. The caution is permissions: a checkpoint keyed only on inputs and parameters will happily serve one user's result to another, so the fingerprint has to include whatever scopes the data, or the sharing has to be limited to layers everyone may read.</p></details>

<details class="faq-item"><summary><span>What happens when a checkpoint's storage disappears?</span></summary><p>Treat a missing checkpoint exactly like an invalid one and recompute from the last surviving state. The failure mode to avoid is a resume that reads a reference, finds nothing, and reports an error to the user about internal storage — a missing checkpoint is an optimisation that did not apply, not a fault worth surfacing. Counting the misses, however, is worth doing: a rising rate usually means expiry is set shorter than conversations actually last.</p></details>

## Related

- Up to the section overview: [Multi-Step Spatial Agent Orchestration](/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/)
- [Spatial Function-Calling Schemas](/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/)
- [Async vs Sync Geoprocessing Workflows](/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/)
