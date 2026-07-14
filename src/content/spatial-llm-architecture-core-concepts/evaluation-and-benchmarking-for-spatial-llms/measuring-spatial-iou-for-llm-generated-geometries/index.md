# Measuring Spatial IoU for LLM-Generated Geometries

Intersection-over-union is the workhorse metric for asking whether a model put a shape in the right place, but LLM-generated geometry breaks the naive implementation: it self-intersects, arrives in the wrong CRS, and occasionally has no overlap at all. This guide builds a robust IoU function that repairs invalid input, measures area in honest units, and returns a defined value on every degenerate case, as part of the [evaluation and benchmarking for spatial LLMs](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/) area.

## When to Use This Approach

Use spatial IoU whenever the ground truth is a footprint and you care how well the prediction covers it: extracted parcels, delineated hazard zones, service areas, building outlines. IoU rewards both matching the right region and not over-claiming beyond it. The metric is

$$\mathrm{IoU} = \frac{\lvert A \cap B \rvert}{\lvert A \cup B \rvert}$$

for predicted footprint $A$ and truth footprint $B$, ranging from `0.0` (disjoint) to `1.0` (identical). Reach for a different metric when the truth is a point or line, where IoU collapses to zero-area and a distance measure is more informative.

| Truth geometry | Metric | Reason |
|----------------|--------|--------|
| Polygon / multipolygon | Spatial IoU | Overlap and over-claim both matter |
| Point | Distance error | Areas are zero; IoU is undefined |
| Linestring | Buffered IoU or Hausdorff | Compare corridors, not raw overlap |
| Very small slivers | Area-weighted IoU | Guards against precision noise dominating |

## Implementation

The function projects both geometries to a caller-supplied equal-area CRS, repairs invalidity with `make_valid`, and treats an empty union as a clean `0.0` rather than a division error. It never raises on data quality; a geometry it cannot use scores zero and is logged.

```python
import logging
from typing import Optional

from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid
from shapely.ops import transform
from shapely.errors import GEOSException
from pyproj import Transformer, CRS

logger = logging.getLogger("spatial_iou")


class IoUInputError(ValueError):
    """Raised only for programmer error, never for bad geometry data."""


def _to_equal_area(geom: BaseGeometry, src_epsg: int, area_epsg: int) -> Optional[BaseGeometry]:
    """Reproject into an equal-area CRS so that .area is in real square metres."""
    try:
        if src_epsg == area_epsg:
            return geom
        transformer = Transformer.from_crs(
            CRS.from_epsg(src_epsg), CRS.from_epsg(area_epsg), always_xy=True
        )
        return transform(transformer.transform, geom)
    except Exception as exc:                        # malformed CRS, pyproj failure
        logger.warning("reprojection %s->%s failed: %s", src_epsg, area_epsg, exc)
        return None


def _validate(geom: BaseGeometry) -> Optional[BaseGeometry]:
    """Repair an invalid geometry; return None if it is unusable."""
    if geom is None or geom.is_empty:
        return None
    if geom.is_valid:
        return geom
    try:
        fixed = make_valid(geom)              # deterministic repair
        return fixed if not fixed.is_empty else None
    except GEOSException as exc:
        logger.warning("make_valid guard failed: %s", exc)
        return None


def spatial_iou(
    predicted: BaseGeometry,
    truth: BaseGeometry,
    src_epsg: int = 4326,
    area_epsg: int = 6933,          # World Equal-Area Cylindrical (metres)
) -> float:
    """Return IoU in [0, 1]. Never crashes; returns 0.0 on any degenerate case."""
    if not isinstance(src_epsg, int) or not isinstance(area_epsg, int):
        raise IoUInputError("EPSG codes must be integers")

    pred = _validate(predicted)
    tru = _validate(truth)
    if pred is None or tru is None:
        return 0.0

    pred = _to_equal_area(pred, src_epsg, area_epsg)
    tru = _to_equal_area(tru, src_epsg, area_epsg)
    if pred is None or tru is None:
        return 0.0

    try:
        intersection_area = pred.intersection(tru).area
        union_area = pred.union(tru).area
    except GEOSException as exc:
        logger.warning("overlay failed, scoring as no-overlap: %s", exc)
        return 0.0

    # Fallback: an empty union means both footprints vanished under repair;
    # define IoU as 0.0 rather than dividing by zero.
    if union_area <= 0.0:
        return 0.0

    iou = intersection_area / union_area
    return max(0.0, min(1.0, iou))     # clamp against floating-point overshoot
```

## Validation & Testing

```python
from shapely.geometry import box, Polygon


def test_identical_footprints_score_one():
    a = box(0, 0, 10, 10)
    assert abs(spatial_iou(a, a, src_epsg=6933, area_epsg=6933) - 1.0) < 1e-9


def test_disjoint_footprints_score_zero():
    a, b = box(0, 0, 1, 1), box(50, 50, 51, 51)
    assert spatial_iou(a, b, src_epsg=6933, area_epsg=6933) == 0.0


def test_self_intersecting_prediction_is_repaired_not_crashed():
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])  # invalid
    truth = box(0, 0, 2, 2)
    iou = spatial_iou(bowtie, truth, src_epsg=6933, area_epsg=6933)
    assert 0.0 <= iou <= 1.0            # defined, in range, no exception
```

Run these in CI with pinned GEOS and PROJ versions so the equal-area reprojection and repair produce identical scores across environments.

## Gotchas & Edge Cases

**Area measured in degrees.** Calling `.area` on EPSG:4326 geometry returns square degrees, which vary with latitude and make IoU incomparable across regions. The `_to_equal_area` step exists precisely to avoid this; never skip it because the inputs "look small".

**Empty union from over-aggressive repair.** A degenerate zero-width sliver can survive parsing but collapse to empty under `make_valid`, producing a zero union. The `union_area <= 0.0` guard returns `0.0` instead of raising `ZeroDivisionError`.

**Floating-point overshoot above 1.0.** Overlay arithmetic can yield an intersection marginally larger than the union at the last decimal. The final `min(1.0, ...)` clamp keeps the score in range so downstream thresholds behave.

**MultiPolygon truth with holes.** IoU handles multipart and holed geometry correctly only if both sides are valid; an unrepaired hole becomes a phantom overlap. The `_validate` gate must run before any area is taken.

## Related

- Up to the parent area: [Evaluation and Benchmarking for Spatial LLMs](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/)
- [Detecting Hallucinated Coordinates in LLM Output](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/detecting-hallucinated-coordinates-in-llm-output/)
- [Building Regression Test Harnesses for Spatial Agents](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/building-regression-test-harnesses-for-spatial-agents/)
