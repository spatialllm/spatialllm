# Detecting Hallucinated Coordinates in LLM Output

When a language model emits latitude/longitude pairs, some fraction of them are invented: values outside the valid globe, points dropped in open ocean where a street address was requested, or coordinates far outside the query's region of interest. This guide covers the validation stage that sits between raw model output and any downstream geoprocessing, catching fabricated points before they poison a map, a route, or a spatial join. It belongs to the [evaluation and benchmarking](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/) area of the [spatial LLM architecture](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/) work.

A hallucinated coordinate is rarely syntactically malformed — it parses fine as two floats. The danger is semantic: the number is plausible in isolation but wrong given the request. Detection therefore has to compare each point against explicit, machine-checkable expectations rather than trust that "looks like a coordinate" means "is a real answer."

## When to Use This Approach

Run per-coordinate validation whenever the model, not a trusted geocoder, is the source of the numbers — free-form extraction from documents, agent tool arguments, or synthetic test fixtures. If your coordinates come from a deterministic API you can skip most of this and check only the region-of-interest bound.

The checks fall into four independent categories, and you should enable only the ones your context can actually assert:

| Check | What it catches | Requires |
| --- | --- | --- |
| Range bound | lon ∉ [-180, 180], lat ∉ [-90, 90] | nothing |
| Region bbox | point outside the query's declared area | a query ROI |
| Land/sea mask | ocean point where land expected (or vice versa) | a coarse mask |
| Precision sanity | 12 decimals implying sub-micron accuracy | a source-precision budget |

Range and precision checks are cheap and always safe. The bbox and land/sea checks depend on knowing the request's intent, so they belong in pipelines where the query carries an explicit ROI. Do not run a land/sea filter on a maritime dataset — the "expected surface" must match the domain.

## Implementation

The validator below returns a per-coordinate flag list rather than a single boolean, so callers can log exactly which point failed and why. It takes a region bounding box, an optional expected surface (`"land"` or `"sea"`), and a coarse land mask callable. Every path is deterministic: a flagged coordinate is rejected and replaced by an explicit `None` sentinel, never silently repaired. This complements [CRS normalization](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/), which must run first so that all coordinates are in EPSG:4326 before range logic applies.

```python
from dataclasses import dataclass, field
from typing import Callable, Optional
import math

@dataclass
class CoordFlags:
    lon: float
    lat: float
    out_of_range: bool = False
    outside_roi: bool = False
    wrong_surface: bool = False
    over_precise: bool = False
    reasons: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.out_of_range or self.outside_roi
                    or self.wrong_surface or self.over_precise)

def _decimals(x: float) -> int:
    s = repr(float(x)).split(".")
    return len(s[1]) if len(s) == 2 else 0

def screen_coordinates(
    points,                       # iterable of (lon, lat)
    roi_bbox,                     # (min_lon, min_lat, max_lon, max_lat) or None
    is_land: Optional[Callable[[float, float], bool]] = None,
    expected_surface: Optional[str] = None,   # "land" | "sea" | None
    max_decimals: int = 6,        # ~0.1 m at the equator; more is fabricated
):
    """Flag likely-hallucinated coordinates. Deterministic: no repair."""
    if expected_surface not in (None, "land", "sea"):
        raise ValueError(f"expected_surface must be land/sea/None, got {expected_surface!r}")

    results = []
    for raw in points:
        try:
            lon, lat = float(raw[0]), float(raw[1])
        except (TypeError, ValueError, IndexError):
            # Unparseable: treat as maximally hallucinated, keep pipeline moving.
            f = CoordFlags(lon=math.nan, lat=math.nan, out_of_range=True)
            f.reasons.append("unparseable")
            results.append(f)
            continue

        f = CoordFlags(lon=lon, lat=lat)

        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0) or \
           math.isnan(lon) or math.isnan(lat):
            f.out_of_range = True
            f.reasons.append("range")

        if roi_bbox is not None and not f.out_of_range:
            min_lon, min_lat, max_lon, max_lat = roi_bbox
            if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
                f.outside_roi = True
                f.reasons.append("outside_roi")

        if expected_surface and is_land and not f.out_of_range:
            try:
                on_land = bool(is_land(lon, lat))
            except Exception:
                on_land = None  # mask failure must not fabricate a verdict
            if on_land is not None:
                if expected_surface == "land" and not on_land:
                    f.wrong_surface = True
                    f.reasons.append("in_water")
                elif expected_surface == "sea" and on_land:
                    f.wrong_surface = True
                    f.reasons.append("on_land")

        if max(_decimals(lon), _decimals(lat)) > max_decimals:
            f.over_precise = True
            f.reasons.append("impossible_precision")

        results.append(f)
    return results

def accept_or_fallback(points, fallback=None, **kwargs):
    """Return cleaned coordinates; rejected points become the fallback sentinel."""
    flags = screen_coordinates(points, **kwargs)
    cleaned = [(f.lon, f.lat) if f.ok else fallback for f in flags]
    rejected = sum(1 for f in flags if not f.ok)
    return cleaned, rejected, flags
```

The land mask does not need to be high resolution. A coarse Natural Earth land polygon loaded once into an in-memory spatial index is enough to catch "the model put the café 40 km offshore." Keep the mask callable pure and side-effect free so the whole screen is reproducible in CI. When a coordinate is rejected, hand the request to a real geocoder or a [fallback route for failed spatial queries](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/) rather than shipping the sentinel downstream.

## Validation & Testing

Pin these three assertions so a regression in the screen fails the build:

```python
def test_range_rejects_swapped_lat_lon():
    # 51.5 lon is out of range; a classic lat/lon swap in London.
    cleaned, rejected, _ = accept_or_fallback([(51.5, 0.12)], roi_bbox=None)
    assert rejected == 1 and cleaned == [None]

def test_roi_flags_out_of_region_point():
    roi = (-0.5, 51.2, 0.3, 51.7)          # Greater London
    _, rejected, flags = accept_or_fallback([(2.35, 48.85)], roi_bbox=roi)  # Paris
    assert rejected == 1 and "outside_roi" in flags[0].reasons

def test_precision_budget_flags_fabricated_digits():
    _, rejected, _ = accept_or_fallback(
        [(-0.1276, 51.5072123456)], roi_bbox=None, max_decimals=6)
    assert rejected == 1
```

Measure the screen's own accuracy against a labelled set of known-good and known-hallucinated points. Track the false-negative rate — a hallucination that slips through — since that is the metric that actually costs you downstream. Report it as $\mathrm{FNR}=\frac{\text{hallucinations passed}}{\text{hallucinations total}}$ and gate CI at, say, $\mathrm{FNR} \le 0.01$.

## Gotchas & Edge Cases

**Lat/lon axis order.** The single most common false-positive source is a swapped pair that happens to stay in range (e.g. two mid-latitude values). Cross-check against the ROI: a point that only validates when you swap its axes is almost certainly swapped.

**Antimeridian ROIs.** A bounding box that crosses ±180° (e.g. Fiji) has `min_lon > max_lon`. The naive containment test rejects every valid point. Detect the wrap and split the box into two ranges before testing.

**Coastline resolution.** A coarse land mask flags legitimate points on piers, small islands, or reclaimed land as "in water." Add a tolerance buffer (a few hundred metres) to the mask, or downgrade a wrong-surface flag to a warning when the point is within the buffer.

**Precision is culture-dependent.** Some upstream systems legitimately store seven decimals. Set `max_decimals` from the actual source precision, not a universal constant, or you will flag honest data as fabricated.

## Related

- Up to the area overview: [Evaluation & Benchmarking for Spatial LLMs](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/)
- [Measuring Spatial IoU for LLM-Generated Geometries](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/measuring-spatial-iou-for-llm-generated-geometries/)
- [Building Regression Test Harnesses for Spatial Agents](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/building-regression-test-harnesses-for-spatial-agents/)
- [Fallback Routing for Geospatial Queries](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/)
