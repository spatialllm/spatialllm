# Evaluation and Benchmarking for Spatial LLMs

A spatial model that produces fluent, well-formed geometry can still be wrong in ways no language metric detects: a polygon in the right shape but the wrong place, a coordinate invented from nothing, a boundary that crosses itself. This guide, part of the [Spatial LLM Architecture & Core Concepts](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/) area, defines the correctness signals that actually matter for spatial agents and shows how to wire them into a repeatable harness that gates deployments.

<figure class="diagram">
<svg viewBox="0 0 880 300" role="img" aria-labelledby="evb-t evb-d" xmlns="http://www.w3.org/2000/svg">
  <title id="evb-t">Spatial evaluation harness</title>
  <desc id="evb-d">Predictions and ground truth flow through validity repair, geometric overlap, topology checks, and a scored gate.</desc>
  <defs>
    <marker id="evb-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/>
    </marker>
  </defs>
  <g fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2">
    <rect x="15" y="55" width="150" height="80" rx="8"/>
    <rect x="188" y="55" width="150" height="80" rx="8"/>
    <rect x="361" y="55" width="150" height="80" rx="8"/>
    <rect x="534" y="55" width="150" height="80" rx="8"/>
    <rect x="707" y="55" width="150" height="80" rx="8"/>
  </g>
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#evb-arrow)">
    <line x1="166" y1="95" x2="186" y2="95"/>
    <line x1="339" y1="95" x2="359" y2="95"/>
    <line x1="512" y1="95" x2="532" y2="95"/>
    <line x1="685" y1="95" x2="705" y2="95"/>
  </g>
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#evb-arrow)">
    <line x1="90" y1="137" x2="90" y2="202"/>
    <line x1="263" y1="137" x2="263" y2="202"/>
    <line x1="436" y1="137" x2="436" y2="202"/>
    <line x1="609" y1="137" x2="609" y2="202"/>
    <line x1="782" y1="137" x2="782" y2="202"/>
  </g>
  <rect x="15" y="205" width="842" height="58" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/>
  <g fill="#1f2937" font-size="13" text-anchor="middle">
    <text x="90" y="90"><tspan x="90" dy="0">Pair pred</tspan><tspan x="90" dy="16">and truth</tspan></text>
    <text x="263" y="90"><tspan x="263" dy="0">Repair</tspan><tspan x="263" dy="16">validity</tspan></text>
    <text x="436" y="90"><tspan x="436" dy="0">Spatial</tspan><tspan x="436" dy="16">IoU</tspan></text>
    <text x="609" y="90"><tspan x="609" dy="0">Topology</tspan><tspan x="609" dy="16">count</tspan></text>
    <text x="782" y="90"><tspan x="782" dy="0">Gate on</tspan><tspan x="782" dy="16">score</tspan></text>
  </g>
  <text x="436" y="240" fill="#1f2937" font-size="15" font-weight="600" text-anchor="middle">Measure placement, invention, and legality separately</text>
  <text x="436" y="287" fill="#5b6471" font-size="12" text-anchor="middle">Metrics: IoU · hallucination rate · topology violations</text>
</svg>
</figure>

## Foundational Principles

1. **Fluency is not correctness.** Perplexity, BLEU, and answer-formatting checks say nothing about whether a geometry lands on the real feature. Spatial evaluation must compare against ground-truth geometry with geometric operators, not string overlap. A model that routes uncertain cases through [fallback routing for geospatial queries](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/) should be scored on what it returns, including its abstentions.

2. **Every metric names the failure it catches.** Placement error, coordinate invention, and topological illegality are distinct pathologies with distinct fixes. Reporting one aggregate score hides which one is regressing. Track spatial IoU for placement, a hallucination rate for invention, and a topology-violation rate for legality as separate, independently gated numbers.

3. **The harness is deterministic and pinned.** The same predictions against the same truth set must yield identical scores on every run and every machine. That means pinned GEOS and PROJ versions, a fixed CRS for area computation, and repair steps that are themselves deterministic. Non-reproducible scores make regressions unfalsifiable, which is worse than having no harness.

## Step-by-Step Implementation Pipeline

### 1. Pair predictions to ground truth and repair validity

Before any metric, align each prediction to its truth geometry and make both valid. An LLM-emitted polygon frequently self-intersects; scoring it raw either crashes the operator or silently returns a misleading area. Repair once, deterministically, and record that a repair happened so it can be audited. Precise IoU computation is covered in [measuring spatial IoU for LLM-generated geometries](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/measuring-spatial-iou-for-llm-generated-geometries/).

```python
import logging
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid
from shapely.errors import GEOSException

logger = logging.getLogger("spatial_eval")


def prepare(geom: BaseGeometry) -> BaseGeometry | None:
    """Return a valid geometry, or None if it cannot be repaired."""
    if geom is None or geom.is_empty:
        return None
    if geom.is_valid:
        return geom
    try:
        repaired = make_valid(geom)
        return repaired if not repaired.is_empty else None
    except GEOSException as exc:
        logger.warning("make_valid failed: %s", exc)
        return None
```

### 2. Score placement with spatial IoU

Intersection-over-union quantifies how much of the predicted footprint coincides with the truth footprint. It is defined as

$$\mathrm{IoU} = \frac{\lvert A \cap B \rvert}{\lvert A \cup B \rvert}$$

for predicted area $A$ and truth area $B$. Compute it in an equal-area CRS so the units are honest, and return `0.0` on an empty union rather than dividing by zero.

```python
def spatial_iou(pred: BaseGeometry, truth: BaseGeometry) -> float:
    """IoU in the caller's equal-area CRS; 0.0 on any degenerate case."""
    p, t = prepare(pred), prepare(truth)
    if p is None or t is None:
        return 0.0
    try:
        inter = p.intersection(t).area
        union = p.union(t).area
    except GEOSException as exc:
        logger.warning("IoU overlay failed: %s", exc)
        return 0.0
    return inter / union if union > 0.0 else 0.0
```

### 3. Count topology violations across the set

Legality is a separate axis from placement. Count self-intersections, unclosed rings, and other invalidities across the prediction set so a model that scores well on IoU but emits illegal geometry is still flagged. Feed embedding-level regressions back through [spatial embedding models](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/spatial-embedding-models/) when the violation rate correlates with a specific geometry family.

```python
def evaluate_batch(pairs: list[tuple[BaseGeometry, BaseGeometry]],
                   iou_floor: float = 0.5) -> dict:
    """Aggregate IoU and topology-violation rate over prediction/truth pairs."""
    if not pairs:
        return {"n": 0, "mean_iou": 0.0, "topology_violation_rate": 0.0, "pass": False}

    ious, violations = [], 0
    for pred, truth in pairs:
        try:
            if pred is None or not pred.is_valid:
                violations += 1
            ious.append(spatial_iou(pred, truth))
        except GEOSException as exc:
            logger.error("pair evaluation failed, scoring as miss: %s", exc)
            violations += 1
            ious.append(0.0)                       # deterministic: a failure is a zero

    n = len(pairs)
    mean_iou = sum(ious) / n
    viol_rate = violations / n
    return {
        "n": n,
        "mean_iou": mean_iou,
        "topology_violation_rate": viol_rate,
        "pass": mean_iou >= iou_floor and viol_rate == 0.0,
    }
```

## Failure Modes & Root Causes

**Area measured in degrees.** Computing IoU on geometry stored in EPSG:4326 makes area a meaningless product of degrees, and the score varies with latitude. Root cause: no reprojection to an equal-area CRS before the overlay.

**Invalid geometry scored as a crash or a zero-by-accident.** An unrepaired self-intersecting polygon can throw inside `intersection`, aborting the whole batch, or return an empty result that reads as a legitimate miss. Root cause: skipping the `make_valid` gate and conflating a repair failure with a genuine low score.

**Hallucinations invisible to IoU.** A confidently invented geometry over open ocean may still overlap a truth polygon partially and post a non-zero IoU. Root cause: relying on placement overlap alone without a dedicated invention check, covered under detecting hallucinated coordinates below.

**Regression masked by aggregation.** A rising topology-violation rate hides behind a stable mean IoU when reported as one blended number. Root cause: collapsing independent failure axes into a single score.

## Production Validation Protocols

1. Reproject predictions and truth to a fixed equal-area CRS before any area computation, and assert the CRS in the harness.
2. Gate three numbers independently: mean IoU above a floor, topology-violation rate at zero, and coordinate-hallucination rate below a ceiling.
3. Pin GEOS and PROJ versions in the evaluation container so scores are identical across machines.
4. Store per-pair results, not just aggregates, so a regression can be traced to the offending predictions.
5. Run the harness on every model or prompt change in CI and block the deployment on any gate failure, using a stable, versioned truth corpus.

## Related

- Up to the parent area: [Spatial LLM Architecture & Core Concepts](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/)
- [Measuring Spatial IoU for LLM-Generated Geometries](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/measuring-spatial-iou-for-llm-generated-geometries/)
- [Detecting Hallucinated Coordinates in LLM Output](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/detecting-hallucinated-coordinates-in-llm-output/)
- [Building Regression Test Harnesses for Spatial Agents](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/building-regression-test-harnesses-for-spatial-agents/)
- [Fallback Routing for Geospatial Queries](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/)
- [Spatial Embedding Models](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/spatial-embedding-models/)
