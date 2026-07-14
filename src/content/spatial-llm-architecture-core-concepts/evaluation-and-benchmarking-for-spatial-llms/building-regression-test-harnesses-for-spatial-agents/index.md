# Building Regression Test Harnesses for Spatial Agents

A spatial agent that passed every test last week can silently drift this week — a prompt tweak, a model version bump, or a library upgrade nudges a generated polygon a few metres and no one notices until a customer does. A golden-dataset regression harness freezes known-good geometry outputs, re-runs them in CI on every change, and fails the build the moment a result drifts past tolerance. This guide builds that harness. It is part of the [evaluation and benchmarking](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/) area within the [spatial LLM architecture](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/) work.

The core idea is to treat each agent behaviour as a case with a pinned expectation: a stored reference geometry, a minimum overlap threshold, and a hard zero on topology violations. Comparison must be tolerant — exact float equality is hopeless across platforms — but bounded, so genuine regressions still trip.

## When to Use This Approach

Reach for a golden-dataset harness once an agent's output is stable enough to snapshot and important enough that undetected drift would hurt. Early prototypes change too fast to pin; a shipped agent that generates service areas, catchment polygons, or extracted footprints is exactly the case for it.

Choose the comparison mode per case type:

| Output type | Primary assertion | Tolerance knob |
| --- | --- | --- |
| Polygon / multipolygon | IoU ≥ threshold | per-case min IoU |
| Point set | max pairwise distance ≤ ε | metres |
| Topology contract | zero invalidities | none — hard zero |
| Attribute join | exact set equality | none |

Do not snapshot raw WKT strings and diff them as text: coordinate ordering, precision, and ring winding all vary without any semantic change, producing a wall of false failures. Compare geometry by geometry, using overlap. The overlap metric is intersection-over-union, $\mathrm{IoU}=\frac{|A\cap B|}{|A\cup B|}$, which is 1.0 for identical shapes and degrades smoothly as the result drifts — see [measuring spatial IoU](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/measuring-spatial-iou-for-llm-generated-geometries/) for the metric itself.

## Implementation

The harness below is a pytest-style runner parametrized over cases loaded from a golden directory. Each case pairs an input with a stored reference geometry and a per-case IoU floor. The comparison repairs invalid geometries before scoring (so a topology bug surfaces as a topology assertion, not a crash), and a missing or unreadable reference degrades to an explicit skip-with-warning rather than a false pass.

```python
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from shapely.geometry import shape, mapping
from shapely.validation import make_valid
from shapely import wkt

GOLDEN = Path(__file__).parent / "golden"

@dataclass(frozen=True)
class SpatialCase:
    case_id: str
    payload: dict
    reference: object          # shapely geometry
    min_iou: float

def load_cases(golden_dir: Path = GOLDEN):
    cases = []
    for path in sorted(golden_dir.glob("*.json")):
        try:
            spec = json.loads(path.read_text())
            ref = shape(spec["reference_geojson"])
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # A corrupt fixture must fail loudly at collection, not pass silently.
            raise RuntimeError(f"Bad golden fixture {path.name}: {e}") from e
        cases.append(SpatialCase(
            case_id=path.stem,
            payload=spec["payload"],
            reference=ref,
            min_iou=float(spec.get("min_iou", 0.98)),
        ))
    if not cases:
        raise RuntimeError(f"No golden cases found in {golden_dir}")
    return cases

def geometry_iou(a, b) -> float:
    a = a if a.is_valid else make_valid(a)
    b = b if b.is_valid else make_valid(b)
    inter = a.intersection(b).area
    union = a.union(b).area
    if union == 0.0:
        # Two empty/degenerate geometries: identical iff both empty.
        return 1.0 if (a.is_empty and b.is_empty) else 0.0
    return inter / union

def topology_violations(geom) -> int:
    return 0 if geom.is_valid else 1

def run_case(case: SpatialCase, agent: Callable[[dict], object]):
    """Execute one case. Returns (iou, violations, produced_geom)."""
    try:
        produced = agent(case.payload)
    except Exception as e:
        # Agent crash is a regression: score it as total miss, do not raise past here.
        return 0.0, 1, None
    if isinstance(produced, str):
        produced = wkt.loads(produced)
    return geometry_iou(produced, case.reference), topology_violations(produced), produced
```

Wire it into pytest with parametrization so every case is a named test node — a failure report points straight at the offending `case_id`:

```python
import pytest
from my_agent import build_service_area   # the agent under test

CASES = load_cases()

@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_spatial_regression(case):
    iou, violations, produced = run_case(case, build_service_area)
    assert violations == 0, f"{case.case_id}: {violations} topology violation(s)"
    assert iou >= case.min_iou, (
        f"{case.case_id}: IoU {iou:.4f} below floor {case.min_iou:.4f} — geometry drifted"
    )
```

The `min_iou` floor is deliberately per-case: a crisp administrative-boundary case can demand 0.999, while a fuzzy isochrone tolerates 0.95. Store thresholds in the fixture, not the code, so tightening a bound is a data change reviewable in a pull request. When the agent depends on an external backend, pin the route so tests exercise a deterministic path — see [routing LLM calls to GeoPandas vs PostGIS](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/geopandas-postgis-tool-routing/routing-llm-calls-to-geopandas-vs-postgis-backends/).

## Validation & Testing

The harness itself needs guarding, or a broken comparator will greenlight real regressions:

```python
from shapely.geometry import box

def test_iou_identical_is_one():
    assert geometry_iou(box(0, 0, 1, 1), box(0, 0, 1, 1)) == 1.0

def test_iou_detects_drift():
    # A 5% shift must drop IoU well below a 0.98 floor.
    assert geometry_iou(box(0, 0, 1, 1), box(0.05, 0, 1.05, 1)) < 0.98

def test_corrupt_fixture_fails_collection(tmp_path):
    (tmp_path / "bad.json").write_text("{ not json")
    with pytest.raises(RuntimeError):
        load_cases(tmp_path)
```

In CI, run the suite on every pull request and block merge on any failure. Emit each case's IoU as a metric so you can chart drift over time — a case creeping from 0.999 toward its 0.98 floor is an early warning before it actually breaks the build.

## Gotchas & Edge Cases

**CRS mismatch between fixture and output.** IoU computed on mixed CRS is meaningless — areas do not correspond. Store the reference CRS in the fixture and assert the produced geometry matches before scoring, reprojecting via [CRS normalization](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/) if needed.

**Nondeterministic agents.** If the model samples with temperature > 0, the same input yields different geometries and the harness flickers. Pin the seed, set temperature to 0 for the regression path, or run N samples and assert the median IoU clears the floor.

**Stale golden data.** When a deliberate improvement legitimately changes output, the harness fails by design. Regenerate the fixtures in the same pull request and review the geometry diff visually — never bulk-overwrite goldens to make red turn green, or the harness stops meaning anything.

**Empty-geometry traps.** An agent that returns an empty polygon scores IoU 1.0 against an empty reference but 0.0 against a real one. Add an explicit non-empty assertion for cases whose reference has area.

## Related

- Up to the area overview: [Evaluation & Benchmarking for Spatial LLMs](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/)
- [Measuring Spatial IoU for LLM-Generated Geometries](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/measuring-spatial-iou-for-llm-generated-geometries/)
- [Detecting Hallucinated Coordinates in LLM Output](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/detecting-hallucinated-coordinates-in-llm-output/)
