# Validating Intermediate Geoprocessing Outputs

When a geoprocessing chain runs buffer, then clip, then dissolve, a single invalid geometry produced in step one silently poisons every step after it — and the only symptom is a wrong answer with no error. This guide adds assertions *between* steps — geometry validity, CRS agreement, and non-empty, plausibly-sized results — so a broken intermediate halts the chain at its source instead of propagating. It is a companion technique within [LLM-assisted geoprocessing pipelines](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/).

The failure this prevents is error propagation. A dissolve over a self-intersecting buffer output may not throw; it may return a subtly wrong multipolygon that the next stage happily consumes. By the time a human notices, the provenance is lost. A between-step gate converts a late, mysterious wrong answer into an early, precise failure with the exact step, input, and reason attached.

## When to Use This Approach

Add inter-step validation to any chain of two or more geoprocessing operations, especially when the steps or their arguments came from a model rather than a fixed script. The check is cheap relative to the operations it guards, and the diagnostic value is highest exactly where a model can plan a plausible-but-wrong sequence.

| Check point | Catches | Cost |
|---|---|---|
| Input only | Bad user/model input | Cheapest, misses mid-chain corruption |
| Between every step (this page) | Propagated invalidity, CRS drift, empty results | Low, high diagnostic value |
| Final output only | Nothing about *where* it broke | Cheap, poor provenance |

Validate between every step when correctness matters more than a few milliseconds per stage. If a chain is short and every op is known-safe, an input-plus-output check may suffice. The between-step gate pairs naturally with a typed plan from [decomposing natural language into geoprocessing steps](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/decomposing-natural-language-into-geoprocessing-steps/), where each step is already named and its expected output known. For CRS handling specifically, lean on [coordinate reference system normalization](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/).

## Implementation

The validator below runs after each step. It confirms the result is a non-empty GeoDataFrame, every geometry is valid, the CRS matches the declared working CRS, and the total area sits within an order-of-magnitude sanity band relative to the previous step. Any violation halts the chain and returns a deterministic fallback carrying the last known-good state.

```python
import logging
from dataclasses import dataclass
from typing import Callable, Optional
import geopandas as gpd

log = logging.getLogger("interstep_validator")


class StepViolation(Exception):
    def __init__(self, step: str, reason: str):
        super().__init__(f"[{step}] {reason}")
        self.step, self.reason = step, reason


@dataclass
class GateResult:
    ok: bool
    gdf: Optional[gpd.GeoDataFrame]
    halted_at: Optional[str] = None
    reason: Optional[str] = None


def check_step(
    gdf: gpd.GeoDataFrame,
    step_name: str,
    working_crs: str,
    prev_area: Optional[float],
    area_ratio_limit: float = 100.0,
) -> float:
    """Assert one intermediate output; return its total area for the next comparison."""
    if gdf is None or len(gdf) == 0:
        raise StepViolation(step_name, "empty result set")
    if gdf.crs is None:
        raise StepViolation(step_name, "missing CRS")
    if gdf.crs.to_string() != working_crs:
        raise StepViolation(step_name, f"CRS {gdf.crs.to_string()} != {working_crs}")

    invalid = gdf[~gdf.geometry.is_valid]
    if len(invalid) > 0:
        raise StepViolation(step_name, f"{len(invalid)} invalid geometries")
    if gdf.geometry.is_empty.any():
        raise StepViolation(step_name, "contains empty geometries")

    area = float(gdf.geometry.area.sum())
    if prev_area is not None and prev_area > 0:
        ratio = max(area, 1e-9) / prev_area
        if ratio > area_ratio_limit or ratio < 1.0 / area_ratio_limit:
            raise StepViolation(
                step_name, f"area changed {ratio:.1f}x vs prior step (limit {area_ratio_limit}x)"
            )
    return area


def run_validated_chain(
    seed: gpd.GeoDataFrame,
    steps: list[tuple[str, Callable[[gpd.GeoDataFrame], gpd.GeoDataFrame]]],
    working_crs: str = "EPSG:3857",
) -> GateResult:
    """Execute steps in order, gating each output; halt to last-good state on violation."""
    current = seed
    last_good = seed
    prev_area: Optional[float] = None
    try:
        prev_area = check_step(current, "seed", working_crs, None)
    except StepViolation as v:
        log.error("seed failed validation: %s", v)
        return GateResult(ok=False, gdf=None, halted_at="seed", reason=v.reason)

    for name, fn in steps:
        try:
            current = fn(last_good)
            prev_area = check_step(current, name, working_crs, prev_area)
            last_good = current
        except StepViolation as v:
            log.error("halting chain: %s", v)
            # Deterministic fallback: return the last validated state, not the corrupt one.
            return GateResult(ok=False, gdf=last_good, halted_at=v.step, reason=v.reason)
        except Exception as exc:  # operation itself threw
            log.exception("step '%s' raised", name)
            return GateResult(ok=False, gdf=last_good, halted_at=name, reason=str(exc))

    return GateResult(ok=True, gdf=last_good)
```

The fallback returns `last_good` — the most recent validated GeoDataFrame — so a caller always receives a coherent geometry set plus the exact step and reason that stopped the chain. That payload is what an orchestrator feeds back for a re-plan or surfaces to an operator.

## Validation & Testing

- **Invalid geometry halts immediately.** Insert a step that returns a bowtie (self-intersecting) polygon and assert `run_validated_chain` returns `ok=False` with `halted_at` equal to that step's name.
- **CRS drift is caught.** Have a step reproject to `EPSG:4326` while `working_crs` is `EPSG:3857`; assert the gate halts with a CRS reason and that `gdf` equals the pre-drift `last_good`.
- **Area sanity band.** Feed a step whose output area is 1000x the prior step and assert it halts on the ratio limit, while a 2x change passes — proving the band flags implausible blow-ups without over-triggering.

## Gotchas & Edge Cases

- **Legitimately empty results.** Some queries correctly return nothing (no schools in the flood zone). Treating every empty set as a violation causes false halts; let the caller declare per-step whether empty is a valid terminal state.
- **Area check across mixed geometry types.** Point and line layers have zero area, so the ratio band is meaningless for them. Gate the area comparison on polygonal geometry types and fall back to feature-count sanity for others.
- **Validity in a geographic CRS.** `is_valid` and `area` computed in `EPSG:4326` are misleading — area is in square degrees and validity ignores the sphere. Run these checks in the projected `working_crs`, consistent with [normalizing mixed CRS data before LLM ingestion](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/normalizing-mixed-crs-data-before-llm-ingestion/).
- **Repairing instead of halting.** Silently `make_valid`-ing every intermediate hides the upstream bug that produced the invalidity. Halt and report first; repair only as an explicit, logged policy the caller opts into.

## Related

- Up to the section: [LLM-Assisted Geoprocessing Pipelines](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/)
- [Decomposing Natural Language into Geoprocessing Steps](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/decomposing-natural-language-into-geoprocessing-steps/)
- [Evaluation and Benchmarking for Spatial LLMs](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/)
- [Building Regression Test Harnesses for Spatial Agents](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/building-regression-test-harnesses-for-spatial-agents/)
