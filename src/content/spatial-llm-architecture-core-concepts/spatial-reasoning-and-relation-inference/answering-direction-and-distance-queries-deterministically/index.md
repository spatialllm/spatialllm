---
title: Answering Direction and Distance Queries Deterministically
description: Give "how far" and "which way" one implementation each — a stated endpoint rule, a metric projection and a named sector convention — so the same question always answers the same way.
slug: answering-direction-and-distance-queries-deterministically
type: howto
breadcrumb: Direction and Distance
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Answering Direction and Distance Queries Deterministically

"How far is the school from the site" has at least four defensible answers depending on which points you measure between and in what projection, and "is it north of the river" has three depending on what you mean by north. A spatial agent that answers these differently in different code paths is not wrong in any particular instance and is untrustworthy overall. This guide gives each question one implementation, as the measurement half of [spatial reasoning and relation inference](/spatial-llm-architecture-core-concepts/spatial-reasoning-and-relation-inference/).

## When to Use This Approach

Route every distance and direction question through these functions, including the ones that look trivial. The inconsistency this prevents comes from convenience implementations written at individual call sites.

| Question shape | Endpoint rule | Note the answer must carry |
|----------------|---------------|----------------------------|
| "How far from A to B" | Edge to edge | The rule, when either shape is large |
| "How far from here" | Focus point to edge | Nothing extra |
| "Within 500 m of" | Edge to edge, threshold | The threshold and the rule |
| "Which way is B from A" | Representative points | The sector convention |
| "How far along a route" | Network distance | That it is a route, not a straight line |

The last row is the one that most often gets answered by the wrong function. A user asking how far it is to a station means walking distance, and a straight-line answer is smaller by a factor that varies with the street layout — which is the sort of error that produces confident, specific and useless advice.

<figure class="diagram">
<svg viewBox="0 0 780 250" role="img" aria-labelledby="add-four-t add-four-d" xmlns="http://www.w3.org/2000/svg"><title id="add-four-t">Four distances between the same two features</title><desc id="add-four-d">Centroid to centroid, edge to edge, focus to edge and network distance produce four different numbers for the same pair, so the rule has to be stated with the answer.</desc><rect x="0" y="0" width="780" height="250" fill="#ffffff"/><text x="390" y="34" fill="#5b6471" font-size="13" text-anchor="middle">One pair of features, four defensible answers</text><rect x="30" y="56" width="360" height="80" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="410" y="56" width="340" height="80" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="30" y="150" width="360" height="80" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="410" y="150" width="340" height="80" rx="8" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><g fill="#1f2937" font-size="13" font-weight="600"><text x="52" y="86">centroid to centroid — 610 m</text><text x="432" y="86">edge to edge — 240 m</text><text x="52" y="180">focus to edge — 265 m</text><text x="432" y="180">along the street network — 480 m</text></g><g fill="#5b6471" font-size="12"><text x="52" y="112">wrong for large or concave shapes</text><text x="432" y="112">what people usually mean</text><text x="52" y="206">right when one side is a point</text><text x="432" y="206">what people mean when walking</text></g></svg>
<figcaption><b>A factor of two and a half between the extremes.</b> None of these is incorrect; the failure is producing one of them without saying which, so a reader cannot tell whether it answers their question.</figcaption>
</figure>

## Implementation

Both functions project into a locally accurate metric frame, apply a stated rule, and return the value with the rule attached.

```python
import logging
import math
from dataclasses import dataclass
from typing import Optional

from shapely.geometry.base import BaseGeometry
from shapely.ops import transform
from pyproj import CRS, Transformer

log = logging.getLogger("direction_distance")

SECTOR_HALF_WIDTH_DEG = 45.0          # "north" spans 315° to 45°
CARDINALS = {"north": 0.0, "east": 90.0, "south": 180.0, "west": 270.0,
             "northeast": 45.0, "southeast": 135.0, "southwest": 225.0, "northwest": 315.0}


@dataclass(frozen=True)
class Measurement:
    value: Optional[float]
    unit: str
    rule: str
    note: str


def _metric(geom: BaseGeometry) -> Transformer:
    lon, lat = geom.centroid.x, geom.centroid.y
    local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat:.6f} +lon_0={lon:.6f} +datum=WGS84 +units=m +no_defs")
    return Transformer.from_crs(CRS.from_epsg(4326), local, always_xy=True)


def distance(a: BaseGeometry, b: BaseGeometry, rule: str = "edge") -> Measurement:
    """Metric distance under a stated endpoint rule. Never guesses a rule."""
    if a is None or b is None or a.is_empty or b.is_empty:
        return Measurement(None, "m", rule, "one or both geometries are unavailable")
    if rule not in {"edge", "centroid"}:
        return Measurement(None, "m", rule, f"unsupported endpoint rule {rule!r}")

    tf = _metric(a)
    try:
        pa, pb = transform(tf.transform, a), transform(tf.transform, b)
    except Exception as exc:
        log.warning("distance projection failed: %s", exc)
        return Measurement(None, "m", rule, f"could not project for measurement: {exc}")

    if rule == "centroid":
        value = pa.centroid.distance(pb.centroid)
    else:
        value = pa.distance(pb)                     # zero when they touch or overlap

    note = "" if value > 0 else "the shapes touch or overlap, so the distance is zero"
    return Measurement(round(value, 1), "m", rule, note)
```

Rounding to one decimal place is a deliberate statement about what the measurement supports. Reporting 264.8317 metres from data captured to the nearest metre implies a precision that does not exist, and a reader who sees four decimal places will reasonably assume the survey justified them.

Direction needs a convention rather than a rule, and the convention must be reported because the alternatives disagree constantly.

```python
def bearing(a: BaseGeometry, b: BaseGeometry) -> Optional[float]:
    """Bearing from a to b in degrees clockwise from north, or None if undefined."""
    tf = _metric(a)
    try:
        pa, pb = transform(tf.transform, a).centroid, transform(tf.transform, b).centroid
    except Exception as exc:
        log.warning("bearing projection failed: %s", exc)
        return None
    dx, dy = pb.x - pa.x, pb.y - pa.y
    if dx == 0 and dy == 0:
        return None                                  # coincident: no direction exists
    return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0


def direction_holds(subject: BaseGeometry, reference: BaseGeometry,
                    named: str) -> Measurement:
    """Is `subject` `named` of `reference`, under the sector convention?"""
    target = CARDINALS.get(named.lower())
    if target is None:
        return Measurement(None, "deg", "sector", f"unsupported direction {named!r}")
    theta = bearing(reference, subject)
    if theta is None:
        return Measurement(None, "deg", "sector", "the geometries coincide; direction is undefined")
    delta = abs((theta - target + 180.0) % 360.0 - 180.0)
    holds = delta <= SECTOR_HALF_WIDTH_DEG
    return Measurement(round(theta, 0), "deg",
                       f"sector ±{SECTOR_HALF_WIDTH_DEG:.0f}°",
                       "" if holds else f"bearing {theta:.0f}° is outside the {named} sector")
```

Returning the bearing whether or not the claim holds is what makes a refusal useful. A claim that fails at forty-eight degrees is a near miss worth mentioning; one that fails at a hundred and thirty is simply wrong, and the reader can see which they are looking at without asking.

<figure class="diagram">
<svg viewBox="46 46 680 188" role="img" aria-labelledby="add-sect-t add-sect-d" xmlns="http://www.w3.org/2000/svg"><title id="add-sect-t">The sector convention for a cardinal direction</title><desc id="add-sect-d">North spans a ninety-degree sector centred on zero, so a bearing of forty-four degrees is north and forty-six is northeast, with the boundary stated rather than implied.</desc><rect x="46" y="46" width="680" height="188" fill="#ffffff"/><rect x="60" y="60" width="180" height="120" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="256" y="60" width="180" height="120" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="452" y="60" width="180" height="120" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="150" y="94">bearing 12°</text><text x="346" y="94">bearing 46°</text><text x="542" y="94">bearing 130°</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="150" y="122">clearly north</text><text x="150" y="148">claim holds</text><text x="346" y="122">just outside the sector</text><text x="346" y="148">claim fails narrowly</text><text x="542" y="122">southeast</text><text x="542" y="148">claim fails plainly</text></g><text x="390" y="216" fill="#1f2937" font-size="13" text-anchor="middle">Reporting the bearing lets a reader tell the middle case from the right-hand one</text></svg>
<figcaption><b>The middle column is why the bearing is returned.</b> A bare "no" for a bearing of forty-six degrees is technically correct and reads as a contradiction; the number turns it into a boundary case the reader can judge.</figcaption>
</figure>

## Validation & Testing

```python
from shapely.geometry import box, Point


def test_edge_distance_is_zero_for_touching_shapes():
    a, b = box(0, 0, 1, 1), box(1, 0, 2, 1)
    m = distance(a, b, rule="edge")
    assert m.value == 0.0 and "touch" in m.note


def test_centroid_and_edge_rules_differ_for_large_shapes():
    a, b = box(-3.4, 55.8, -3.0, 56.1), box(-2.99, 55.9, -2.98, 55.91)
    assert distance(a, b, "centroid").value > distance(a, b, "edge").value * 2


def test_unsupported_rule_returns_none_not_a_default():
    m = distance(box(0, 0, 1, 1), box(2, 2, 3, 3), rule="nearest_vertex")
    assert m.value is None and "unsupported" in m.note


def test_direction_reports_the_bearing_on_failure():
    m = direction_holds(Point(1, 0), Point(0, 0), "north")
    assert m.value is not None and "outside the north sector" in m.note


def test_coincident_geometries_have_no_direction():
    m = direction_holds(Point(0, 0), Point(0, 0), "north")
    assert m.value is None and "undefined" in m.note
```

The third test is the guard on the whole design. Falling back to a default rule when an unrecognised one is requested would make the function total and would silently answer a different question from the one asked, which is precisely the inconsistency these implementations exist to prevent.

## Gotchas & Edge Cases

**Distance measured in degrees.** The classic error, and it produces numbers of plausible magnitude in temperate latitudes. Projecting first is not optional, and the locally centred projection costs microseconds.

**A global metric projection used for local work.** Cheaper than building one per measurement and wrong by a growing margin away from its centre. If throughput demands a shared projection, choose it per region rather than per world.

**Direction by latitude comparison.** Reporting "north of" whenever any part of the subject lies further north ignores east-west offset entirely, so a feature slightly north and forty kilometres east is described as north. Use a bearing sector.

<figure class="diagram">
<svg viewBox="76 9 678 200" role="img" aria-labelledby="add-proj-t add-proj-d" xmlns="http://www.w3.org/2000/svg"><title id="add-proj-t">Distance error from a shared global projection</title><desc id="add-proj-d">A single global metric projection is accurate near its centre and increasingly wrong away from it, while a projection built per measurement keeps the error negligible everywhere.</desc><rect x="76" y="9" width="678" height="200" fill="#ffffff"/><text x="380" y="34" fill="#5b6471" font-size="13" text-anchor="middle">Error in a 500 m measurement, by distance from the projection centre</text><g fill="#fdeaee" stroke="#b3324f" stroke-width="2"><rect x="90" y="146" width="60" height="26" rx="4"/><rect x="240" y="122" width="60" height="50" rx="4"/><rect x="390" y="86" width="60" height="86" rx="4"/><rect x="540" y="60" width="60" height="112" rx="4"/></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="120" y="192">100 km</text><text x="270" y="192">500 km</text><text x="420" y="192">1500 km</text><text x="570" y="192">3000 km</text></g><text x="680" y="140" fill="#12805c" font-size="12" text-anchor="middle">per-measurement</text><text x="680" y="160" fill="#12805c" font-size="12" text-anchor="middle">projection: flat</text></svg>
<figcaption><b>A systematic bias, not random noise.</b> The error grows with distance from the projection centre, so a corpus spanning a continent measures consistently short in some regions and long in others — which looks like a data problem and is a projection one.</figcaption>
</figure>

**Sector width left implicit.** Ninety degrees is conventional and not universal; some domains use sixty. State the width in the note, and keep it a constant that a reader of the code can find.

**Network distance answered with a straight line.** Routing questions need a routable network, and the difference is large in dense street layouts. Detect the intent — "walk", "drive", "along" — and refuse rather than substituting a straight line.

**Rounding that varies between call sites.** One path reports 264.8 and another 265, and a user comparing two answers concludes the data changed. Round in the measurement function, once, at a precision the data supports.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should edge distance be the default?</span></summary><p>Yes, for most questions, because it matches what people mean by "how far apart" and it returns zero for touching or overlapping shapes, which is the answer a reader expects. Centroid distance is the right default only in a narrow case: comparing many features of similar size where the centroid is a fair representative of each. Whichever you choose, state it — the difference is largest exactly where a reader is least likely to notice.</p></details>

<details class="faq-item"><summary><span>How should a distance to a large region be phrased?</span></summary><p>With the rule in the sentence: "240 m from the nearest edge of the conservation area" rather than "240 m from the conservation area". The extra four words remove the ambiguity entirely and cost nothing, and they pre-empt the most common follow-up question. For very large regions it is also worth naming the part measured to, since the nearest edge may be tens of kilometres from the part the reader had in mind.</p></details>

<details class="faq-item"><summary><span>What about directions between extended shapes?</span></summary><p>Bearing between representative points is a reasonable default and becomes misleading when the shapes are large or interleaved — two long parcels running east to west have a north-south bearing between their centres that describes nothing. Where both shapes are extended, consider reporting the relation between their nearest parts instead, and say that is what you did.</p></details>

<details class="faq-item"><summary><span>Should these answers be cached?</span></summary><p>Only keyed on geometry versions, like topology results. A distance is cheap to compute and expensive to be wrong about after a boundary correction, so caching buys little and risks a stale number that reads as authoritative. Where a distance is computed thousands of times for the same pair, cache it with the version stamp and let a correction invalidate it automatically.</p></details>

## Related

- Up to the parent topic: [Spatial Reasoning and Relation Inference](/spatial-llm-architecture-core-concepts/spatial-reasoning-and-relation-inference/)
- [Verifying Topological Claims with the Nine-Intersection Model](/spatial-llm-architecture-core-concepts/spatial-reasoning-and-relation-inference/verifying-topological-claims-with-de9im/)
- Technique: [Choosing a Canonical Frame for Spatial LLM Pipelines](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/choosing-a-canonical-crs-for-llm-pipelines/)
- Related topic: [Spatial Context Retrieval and Reranking](/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/)
