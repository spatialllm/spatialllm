---
title: Measuring Spatial IoU for LLM-Generated Geometries
description: Build an intersection-over-union scorer that repairs invalid model output, measures area in an equal-area projection, and returns a defined value on every degenerate case.
slug: measuring-spatial-iou-for-llm-generated-geometries
type: howto
breadcrumb: Measuring Spatial IoU
datePublished: 2025-04-22
dateModified: 2026-08-11
---

# Measuring Spatial IoU for LLM-Generated Geometries

Intersection-over-union is the workhorse metric for asking whether a model put a shape in the right place, but model-generated geometry breaks the naive implementation: it self-intersects, arrives in the wrong projection, and occasionally has no overlap at all. This guide builds a robust IoU function that repairs invalid input, measures area in honest units, and returns a defined value on every degenerate case — the scoring primitive the rest of [evaluation and benchmarking for spatial LLMs](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/) is built on. It belongs at the scoring stage of the evaluation harness, after generation and before any threshold is applied.

## When to Use This Approach

Use spatial IoU whenever the ground truth is a footprint and you care both how well the prediction covers it and how much it over-claims beyond it: extracted parcels, delineated hazard zones, service areas, building outlines. The metric is

$$\mathrm{IoU} = \frac{\lvert A \cap B \rvert}{\lvert A \cup B \rvert}$$

for predicted footprint $A$ and truth footprint $B$, ranging from `0.0` (disjoint) to `1.0` (identical). Its virtue is that the denominator grows when the model draws too much, so a prediction that swallows the whole city cannot score well by containing the right building. Reach for a different metric when the truth is a point or a line, where both areas collapse to zero and a distance measure carries the information instead.

<figure class="diagram">
<svg viewBox="60 11 601 238" role="img" aria-labelledby="iou-anat-t iou-anat-d" xmlns="http://www.w3.org/2000/svg"><title id="iou-anat-t">How the IoU ratio is assembled from two footprints</title><desc id="iou-anat-d">The union is every panel of the bar — truth-only area, the shared intersection, and predicted-only over-claim. The numerator is only the middle panel, so over-claiming on either side lowers the score.</desc><rect x="60" y="11" width="601" height="238" fill="#ffffff"/><text x="360" y="36" fill="#5b6471" font-size="13" text-anchor="middle">Union is every panel; the numerator is only the middle one</text><rect x="90" y="70" width="160" height="90" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="250" y="70" width="200" height="90" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="450" y="70" width="160" height="90" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle"><text x="170" y="110">Truth only</text><text x="350" y="110">Intersection</text><text x="530" y="110">Over-claim</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="170" y="132">160 units</text><text x="350" y="132">200 units</text><text x="530" y="132">160 units</text></g><text x="360" y="200" fill="#1f2937" font-size="16" font-weight="600" text-anchor="middle">IoU = 200 / 520 = 0.38</text><text x="360" y="232" fill="#5b6471" font-size="12" text-anchor="middle">A prediction that covers the truth but spills past it is penalised twice over</text></svg>
<figcaption><b>Why the denominator matters.</b> Recall alone would score this prediction 200/360 = 0.56. IoU charges for the 160 units of over-claim as well, which is exactly the behaviour you want from a model that likes to round footprints outward.</figcaption>
</figure>

| Truth geometry | Metric | Reason |
|----------------|--------|--------|
| Polygon / multipolygon | Spatial IoU | Overlap and over-claim both matter |
| Point | Distance error | Areas are zero; IoU is undefined |
| Linestring | Buffered IoU or Hausdorff | Compare corridors, not raw overlap |
| Very small slivers | Area-weighted IoU | Guards against precision noise dominating |
| Ranked candidate set | Mean IoU at rank k | One score per candidate hides ordering quality |

IoU is also the wrong tool when the task is classification dressed up as geometry. If the model is choosing among a fixed set of pre-existing polygons — census tracts, administrative units, delivery zones — then the answer is a label, and a confusion matrix over those labels tells you more than an area ratio ever will. Reserve IoU for the case where the model actually drew the boundary.

## Implementation

The function projects both geometries to a caller-supplied equal-area projection, repairs invalidity with `make_valid`, and treats an empty union as a clean `0.0` rather than a division error. It never raises on data quality; a geometry it cannot use scores zero and is logged. That distinction — programmer error raises, data error scores — is what lets the harness run a thousand cases unattended without a single bad prediction aborting the sweep.

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
    """Reproject into an equal-area projection so that .area is in real square metres."""
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

Four things are load-bearing in that order of operations. Repair runs first, because an invalid ring makes every subsequent area meaningless rather than merely wrong. Reprojection runs second, because repairing after a projection change can move vertices that the repair had just reconciled. The overlay is wrapped, because GEOS raises on inputs that survived `make_valid` but still confound the noding step. And the clamp runs last, because the ratio of two floating-point areas is not guaranteed to land inside the closed unit interval even when the geometry is perfect.

<figure class="diagram">
<svg viewBox="6 44 828 236" role="img" aria-labelledby="iou-pipe-t iou-pipe-d" xmlns="http://www.w3.org/2000/svg"><title id="iou-pipe-t">The four scoring stages and the shared failure lane</title><desc id="iou-pipe-d">Repair, reproject, overlay and clamp run in sequence. Any stage that fails drops into a single deterministic lane that logs the cause and returns a score of zero, so no bad prediction can abort an evaluation sweep.</desc><rect x="6" y="44" width="828" height="236" fill="#ffffff"/><g fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"><rect x="20" y="58" width="170" height="78" rx="8"/><rect x="230" y="58" width="170" height="78" rx="8"/><rect x="440" y="58" width="170" height="78" rx="8"/><rect x="650" y="58" width="170" height="78" rx="8"/></g><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="105" y="90">1 · Repair</text><text x="315" y="90">2 · Reproject</text><text x="525" y="90">3 · Overlay</text><text x="735" y="90">4 · Clamp</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="105" y="112">make_valid</text><text x="315" y="112">equal-area units</text><text x="525" y="112">areas of both sets</text><text x="735" y="112">score into [0, 1]</text></g><g stroke="#5b6471" stroke-width="2" marker-end="url(#iou-arrow)"><line x1="192" y1="97" x2="226" y2="97"/><line x1="402" y1="97" x2="436" y2="97"/><line x1="612" y1="97" x2="646" y2="97"/><line x1="105" y1="138" x2="105" y2="196"/><line x1="315" y1="138" x2="315" y2="196"/><line x1="525" y1="138" x2="525" y2="196"/><line x1="735" y1="138" x2="735" y2="196"/></g><defs><marker id="iou-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/></marker></defs><rect x="20" y="200" width="800" height="66" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="420" y="228" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">Deterministic failure lane</text><text x="420" y="250" fill="#5b6471" font-size="12" text-anchor="middle">unrepairable ring · projection error · empty union — log the cause, return 0.0, never raise</text></svg>
<figcaption><b>One exit for every failure.</b> Each stage drops into the same lane rather than raising its own exception type, so the harness records a comparable zero for a torn polygon, a bad EPSG code and a collapsed union alike.</figcaption>
</figure>

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


def test_over_claim_is_penalised_more_than_partial_cover():
    truth = box(0, 0, 10, 10)
    spilled = box(0, 0, 20, 20)         # covers everything, claims four times the area
    clipped = box(0, 0, 10, 5)          # covers half, claims nothing extra
    assert spatial_iou(spilled, truth, 6933, 6933) < spatial_iou(clipped, truth, 6933, 6933)
```

Run these in CI with pinned GEOS and PROJ versions so the equal-area reprojection and repair produce identical scores across environments. A version bump that changes `make_valid`'s output on a bow-tie will move scores by a few percent, and without pinning that drift is indistinguishable from a regression in the model itself. Record both library versions alongside every score you publish, the same way you would record the model version.

The fourth test is the one worth defending in review. It encodes the property the metric exists for, not an arithmetic identity, and it fails loudly if somebody "optimises" the implementation into a recall calculation.

## Gotchas & Edge Cases

**Area measured in degrees.** Calling `.area` on EPSG:4326 geometry returns square degrees, which vary with latitude and make IoU incomparable across regions. The `_to_equal_area` step exists precisely to avoid this; never skip it because the inputs "look small". A test suite drawn entirely from one city will not catch the omission, because within one city the distortion is a constant factor that cancels in the ratio.

**Empty union from over-aggressive repair.** A degenerate zero-width sliver can survive parsing but collapse to empty under `make_valid`, producing a zero union. The `union_area <= 0.0` guard returns `0.0` instead of raising `ZeroDivisionError`.

**Floating-point overshoot above 1.0.** Overlay arithmetic can yield an intersection marginally larger than the union at the last decimal. The final `min(1.0, ...)` clamp keeps the score in range so downstream thresholds behave.

**MultiPolygon truth with holes.** IoU handles multipart and holed geometry correctly only if both sides are valid; an unrepaired hole becomes a phantom overlap. The `_validate` gate must run before any area is taken.

**Averaging scores across wildly different footprint sizes.** A mean IoU over a set containing both a bus shelter and a national park is dominated by whichever cases happen to be easy at that scale. Report the distribution — median plus the tenth percentile — or weight by truth area deliberately, and say which you did.

<figure class="diagram">
<svg viewBox="0 0 760 220" role="img" aria-labelledby="iou-band-t iou-band-d" xmlns="http://www.w3.org/2000/svg"><title id="iou-band-t">Release bands for an IoU score</title><desc id="iou-band-d">Five score bands running from reject through manual review, borderline and pass to near-exact, each mapped to the action an evaluation harness should take when a case lands in it.</desc><rect x="0" y="0" width="760" height="220" fill="#ffffff"/><text x="380" y="36" fill="#5b6471" font-size="13" text-anchor="middle">Bands are per task — publish them next to the model version</text><rect x="40" y="62" width="136" height="76" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="176" y="62" width="136" height="76" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="312" y="62" width="136" height="76" rx="8" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="448" y="62" width="136" height="76" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="584" y="62" width="136" height="76" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="108" y="92">0.00–0.30</text><text x="244" y="92">0.30–0.50</text><text x="380" y="92">0.50–0.70</text><text x="516" y="92">0.70–0.90</text><text x="652" y="92">0.90–1.00</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="108" y="118">reject</text><text x="244" y="118">review</text><text x="380" y="118">borderline</text><text x="516" y="118">pass</text><text x="652" y="118">near-exact</text></g><text x="380" y="176" fill="#1f2937" font-size="13" text-anchor="middle">A single mean hides the reject tail — gate on a percentile, not an average</text><text x="380" y="200" fill="#5b6471" font-size="12" text-anchor="middle">Suggested gate: tenth percentile above the review band on every release</text></svg>
<figcaption><b>Turning a score into a decision.</b> The bands are a policy, not a property of the metric: a parcel-extraction task and a wildfire-perimeter task will draw them in different places, and both should state where.</figcaption>
</figure>

## Frequently Asked Questions

<details class="faq-item"><summary><span>Which equal-area projection should I pass as area_epsg?</span></summary><p>EPSG:6933 is a reasonable global default because it is equal-area everywhere and needs no per-case selection. For work confined to one country, a national equal-area projection gives less shape distortion at the same area fidelity. What matters is that the same code is used for every case in a comparison set — mixing projections across a benchmark makes the scores incomparable even though each one is individually correct.</p></details>

<details class="faq-item"><summary><span>Should a prediction that fails to parse score zero, or be excluded?</span></summary><p>Score it zero and count it. Excluding unparseable output flatters the model exactly where it is weakest, and two systems with identical mean IoU over their parseable cases can have wildly different failure rates. If you need the two numbers separated, publish the parse rate alongside the mean rather than quietly filtering.</p></details>

<details class="faq-item"><summary><span>Why clamp rather than assert the score is in range?</span></summary><p>The overshoot is a floating-point artefact of taking the ratio of two independently computed areas, not a signal about the geometry. Asserting would abort an otherwise valid sweep over a discrepancy in the fifteenth decimal place. Clamping records the score the geometry deserves; if you want visibility, log when the raw ratio exceeds 1.0 by more than a small epsilon.</p></details>

<details class="faq-item"><summary><span>How does IoU interact with coordinate hallucination?</span></summary><p>Badly, and that is the point of pairing them. A hallucinated coordinate usually lands the footprint in the wrong hemisphere, where IoU is exactly zero and tells you nothing about how wrong the answer was. Run a plausibility screen first — see <a href="/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/detecting-hallucinated-coordinates-in-llm-output/">detecting hallucinated coordinates in model output</a> — and report the two failure classes separately.</p></details>

## Related

- Up to the parent topic: [Evaluation and Benchmarking for Spatial LLMs](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/)
- [Detecting Hallucinated Coordinates in LLM Output](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/detecting-hallucinated-coordinates-in-llm-output/)
- [Building Regression Test Harnesses for Spatial Agents](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/building-regression-test-harnesses-for-spatial-agents/)
- [Setting Release Thresholds for Spatial Agents](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/setting-release-thresholds-for-spatial-agents/)
- [Coordinate Reference System Normalization](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/)
