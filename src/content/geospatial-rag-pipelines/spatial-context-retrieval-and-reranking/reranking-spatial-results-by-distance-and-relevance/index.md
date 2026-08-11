---
title: Reranking Spatial Results by Distance and Relevance
description: Combine a semantic score with a proximity term so nearby, on-topic documents outrank eloquent ones from the wrong region — with a decay shape you can defend.
slug: reranking-spatial-results-by-distance-and-relevance
type: howto
breadcrumb: Distance and Relevance Reranking
datePublished: 2025-03-26
dateModified: 2026-08-11
---

# Reranking Spatial Results by Distance and Relevance

A vector search has returned two hundred candidates that are all, in some sense, about the right subject. Ordering them is now a spatial problem: the document about the site next door should outrank the one about a similar site in another county, without the pipeline collapsing into a nearest-neighbour lookup that ignores what the question actually asked. This guide implements that reranking step for [spatial context retrieval and reranking](/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/).

## When to Use This Approach

Rerank when candidates share a region but differ in position, and when the question is about a place rather than about a policy. If every candidate sits inside a small site, distance carries no information and the reranker only adds noise; if the question is regional, distance to a focus point is actively misleading.

| Question | Distance term | Reason |
|----------|---------------|--------|
| "What is at this address" | Strong | The nearest description is almost always the right one |
| "Flood risk to this street" | Moderate | Catchment documents legitimately sit further away |
| "What does the local plan say" | Weak or none | Policy applies across an area, not at a point |
| "Compare these two sites" | None — two foci | A single distance term biases toward whichever focus it used |

The last row is worth taking seriously. Multi-focus questions break the single-distance model entirely, and the honest handling is to run the retrieval once per focus and merge, rather than to average two distances into a number that describes neither.

<figure class="diagram">
<svg viewBox="26 38 708 192" role="img" aria-labelledby="rsr-pair-t rsr-pair-d" xmlns="http://www.w3.org/2000/svg"><title id="rsr-pair-t">Two candidates that a semantic ranker cannot separate</title><desc id="rsr-pair-d">Two documents with nearly identical semantic scores, one 400 metres from the query focus and one 60 kilometres away. Only the proximity term distinguishes them.</desc><rect x="26" y="38" width="708" height="192" fill="#ffffff"/><rect x="40" y="52" width="320" height="120" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="400" y="52" width="320" height="120" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="200" y="82">local site report</text><text x="560" y="82">county-wide assessment</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="200" y="110">semantic 0.81</text><text x="200" y="132">400 m from the focus</text><text x="200" y="154">fused 0.79</text><text x="560" y="110">semantic 0.83</text><text x="560" y="132">60 km from the focus</text><text x="560" y="154">fused 0.54</text></g><text x="380" y="212" fill="#1f2937" font-size="13" text-anchor="middle">A two-point semantic lead cannot survive a 60 km proximity deficit — and should not</text></svg>
<figcaption><b>The margin the semantic score reports is not meaningful at this scale.</b> Two hundredths of cosine similarity is within the noise of most embedding models; sixty kilometres is not within the noise of anything.</figcaption>
</figure>

## Implementation

The reranker takes scored candidates with geometry, computes edge distance in metres, applies an exponential decay, and returns a fused ordering with both components preserved for inspection.

```python
import logging
import math
from dataclasses import dataclass, replace
from typing import Optional, Sequence

from shapely.geometry.base import BaseGeometry
from shapely.ops import transform
from pyproj import Transformer, CRS

log = logging.getLogger("spatial_rerank")


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    semantic: float                 # cosine similarity in [0, 1]
    geom: Optional[BaseGeometry]    # in EPSG:4326
    distance_m: Optional[float] = None
    fused: Optional[float] = None


def _metric_transformer(focus: BaseGeometry) -> Transformer:
    """Build a locally accurate metric projection centred on the query focus."""
    lon, lat = focus.centroid.x, focus.centroid.y
    local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs(CRS.from_epsg(4326), local, always_xy=True)


def _edge_distance_m(geom: BaseGeometry, focus: BaseGeometry, tf: Transformer) -> Optional[float]:
    """Distance from the focus to the candidate's nearest edge; 0 when it contains the focus."""
    try:
        return transform(tf.transform, geom).distance(transform(tf.transform, focus))
    except Exception as exc:                       # unprojectable or malformed geometry
        log.warning("distance failed for candidate geometry: %s", exc)
        return None


def rerank(
    candidates: Sequence[Candidate],
    focus: Optional[BaseGeometry],
    radius_m: float = 5000.0,
    w_semantic: float = 0.65,
    half_life_m: float = 1500.0,
) -> list[Candidate]:
    """Fuse semantic score with proximity. Falls back to semantic order without a focus."""
    if focus is None or focus.is_empty:
        log.info("no focus — returning semantic ordering unchanged")
        return sorted(candidates, key=lambda c: c.semantic, reverse=True)
    if half_life_m <= 0:
        raise ValueError("half_life_m must be positive")

    tf = _metric_transformer(focus)
    out: list[Candidate] = []
    for cand in candidates:
        distance = _edge_distance_m(cand.geom, focus, tf) if cand.geom is not None else None
        if distance is None:
            # Deterministic fallback: unknown position keeps its semantic score,
            # discounted so it never outranks a candidate known to be nearby.
            fused = round(w_semantic * cand.semantic * 0.8, 6)
            out.append(replace(cand, distance_m=None, fused=fused))
            continue
        if distance > radius_m:
            continue                                # outside the region: not a candidate
        proximity = 0.5 ** (distance / half_life_m)
        fused = round(w_semantic * cand.semantic + (1.0 - w_semantic) * proximity, 6)
        out.append(replace(cand, distance_m=distance, fused=fused))

    return sorted(out, key=lambda c: (c.fused if c.fused is not None else 0.0), reverse=True)
```

The locally centred azimuthal projection is worth the two extra lines. Computing distance in degrees and multiplying by a constant is accurate at the equator and increasingly wrong toward the poles, and the error is a systematic bias in ranking rather than random noise — candidates east and west of the focus are penalised too heavily at high latitude, so northern corpora quietly favour documents to the north and south.

The unknown-distance branch discounts rather than excludes. A chunk with no geometry may still be the best answer, especially in corpora where prose and geometry live in different documents, and dropping it entirely converts a metadata gap into a retrieval gap.

<figure class="diagram">
<svg viewBox="46 38 698 191" role="img" aria-labelledby="rsr-edge-t rsr-edge-d" xmlns="http://www.w3.org/2000/svg"><title id="rsr-edge-t">Centroid distance against edge distance for a large polygon</title><desc id="rsr-edge-d">A catchment polygon containing the query focus has a centroid tens of kilometres away. Centroid distance ranks it as irrelevant; edge distance correctly returns zero because the focus is inside it.</desc><rect x="46" y="38" width="698" height="191" fill="#ffffff"/><rect x="60" y="52" width="300" height="140" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="96" y="150" width="34" height="26" rx="4" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="210" y="88" fill="#1f2937" font-size="12.5" text-anchor="middle">catchment polygon</text><text x="210" y="112" fill="#5b6471" font-size="12" text-anchor="middle">centroid near the middle</text><text x="113" y="212" fill="#12805c" font-size="12" text-anchor="middle">focus</text><rect x="430" y="52" width="300" height="66" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="580" y="80" fill="#1f2937" font-size="12.5" text-anchor="middle">centroid distance: 22 km</text><text x="580" y="102" fill="#5b6471" font-size="12" text-anchor="middle">ranked as irrelevant</text><rect x="430" y="132" width="300" height="66" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="580" y="160" fill="#1f2937" font-size="12.5" text-anchor="middle">edge distance: 0 m</text><text x="580" y="182" fill="#5b6471" font-size="12" text-anchor="middle">the focus is inside it</text></svg>
<figcaption><b>Containment is the case centroid distance gets exactly backwards.</b> The document that governs the site is the one a centroid ranker is most likely to discard, because governing documents cover large areas.</figcaption>
</figure>

## Validation & Testing

```python
from shapely.geometry import Point, box


def test_nearer_candidate_wins_at_equal_semantic_score():
    focus = Point(-3.19, 55.95)
    near = Candidate("near", 0.80, Point(-3.19, 55.951))
    far = Candidate("far", 0.80, Point(-3.30, 55.99))
    ranked = rerank([far, near], focus, radius_m=20000)
    assert ranked[0].chunk_id == "near"


def test_containing_polygon_scores_zero_distance():
    focus = Point(-3.19, 55.95)
    catchment = Candidate("catchment", 0.60, box(-3.4, 55.8, -3.0, 56.1))
    ranked = rerank([catchment], focus, radius_m=20000)
    assert ranked[0].distance_m == 0.0


def test_outside_radius_is_absent_not_zero_scored():
    focus = Point(-3.19, 55.95)
    away = Candidate("away", 0.99, Point(2.35, 48.85))
    assert rerank([away], focus, radius_m=5000) == []


def test_missing_geometry_is_ranked_but_discounted():
    focus = Point(-3.19, 55.95)
    known = Candidate("known", 0.70, Point(-3.191, 55.951))
    unknown = Candidate("unknown", 0.70, None)
    ranked = rerank([unknown, known], focus)
    assert ranked[0].chunk_id == "known" and len(ranked) == 2
```

The first test is a monotonicity property and should ideally be run over generated inputs rather than one pair; the ordering it asserts is the entire contract of the function, and it is the assertion that fails when someone inverts a sign while tuning.

## Gotchas & Edge Cases

**Half-life tuned on one query.** A half-life that makes a demonstration query look perfect will usually be far too short for regional questions, and the symptom — "the reranker ignores anything more than a kilometre away" — is indistinguishable from a bug. Tune against a labelled set spanning both question types, and let the region-sizing logic vary the half-life with the question class rather than fixing it globally.

**Candidates whose geometry is a bounding box.** If chunk geometry is stored as an extent rather than the true shape, edge distance is measured to the box, which for a sparse feature collection can be far from any actual feature. Store true geometry when the store allows it, and treat extent-derived distance as an upper bound rather than a measurement.

**Degenerate focus geometry.** A focus resolved to an empty geometry, or to a point at the origin because a lookup failed, produces confident nonsense. Validate the focus before use, as the function does, and prefer returning semantic order to ranking against a wrong place.

<figure class="diagram">
<svg viewBox="16 42 728 182" role="img" aria-labelledby="rsr-half-t rsr-half-d" xmlns="http://www.w3.org/2000/svg"><title id="rsr-half-t">Half-life choice against question class</title><desc id="rsr-half-d">Three question classes with the proximity half-life each one wants, from a few hundred metres for address questions to several kilometres for catchment questions and effectively none for policy questions.</desc><rect x="16" y="42" width="728" height="182" fill="#ffffff"/><rect x="30" y="56" width="220" height="112" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="270" y="56" width="220" height="112" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="510" y="56" width="220" height="112" rx="8" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="140" y="86">address question</text><text x="380" y="86">catchment question</text><text x="620" y="86">policy question</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="140" y="114">half-life 300 m</text><text x="140" y="140">nearest wins decisively</text><text x="380" y="114">half-life 4 km</text><text x="380" y="140">regional documents survive</text><text x="620" y="114">no distance term</text><text x="620" y="140">proximity is noise here</text></g><text x="380" y="206" fill="#1f2937" font-size="13" text-anchor="middle">One constant cannot serve all three — vary it with the question class</text></svg>
<figcaption><b>The half-life is a question-class parameter.</b> Fixing it globally guarantees that one class of question is served badly, and the class it hurts is whichever one was not open on screen during tuning.</figcaption>
</figure>

**Precision loss from a global projection.** Reprojecting every candidate into one global metric projection is faster than building a local one, and it reintroduces the distortion the local projection avoids. If throughput demands it, at least choose the projection from the focus's region rather than using one for the whole world.

**Ties broken by insertion order.** When two candidates fuse to the same score, the sort is stable and the winner is whichever arrived first — a detail that makes results depend on the upstream query plan. Add a deterministic tiebreaker, such as the chunk identifier, so the same inputs always produce the same order.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should distance be measured to a point or to the query's full geometry?</span></summary><p>To the full geometry when the question is about an area — a parcel, a route, a catchment — because distance from a polygon's edge is the quantity that matches how people think about "near this site". Reducing the focus to its centroid first is a simplification that costs little for compact shapes and a great deal for elongated ones such as a river reach or a road corridor.</p></details>

<details class="faq-item"><summary><span>Is exponential decay the right shape, or would a step function do?</span></summary><p>A step function — full score inside a radius, zero outside — is defensible and easier to explain, and it fails badly at the boundary, where a candidate one metre outside loses everything. Exponential decay avoids the cliff while still expressing that the first kilometre matters most. Keep the hard radius as an eligibility filter and use the decay for ordering within it, which is what the implementation above does.</p></details>

<details class="faq-item"><summary><span>How should the weight change for a regional question?</span></summary><p>Toward the semantic side, and possibly to the point where proximity only breaks ties. For a question about policy that applies across a district, the distance from a focus point to a document about that district is close to meaningless, and a strong proximity weight will rank the nearest irrelevant site report above the policy itself. Vary the weight with the question class rather than trying to find one value that serves both.</p></details>

<details class="faq-item"><summary><span>Does reranking need the original vector scores, or just the ranks?</span></summary><p>The scores, if you are fusing as here; the ranks, if you are using reciprocal rank fusion instead. The distinction matters because a store that returns only ranks forces the second design. Both work; what does not work is treating a rank as if it were a score, which compresses the differences between the top few candidates and exaggerates those further down.</p></details>

## Related

- Up to the parent topic: [Spatial Context Retrieval and Reranking](/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/)
- [Filtering Retrieval by Bounding Box Before Vector Search](/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/filtering-retrieval-by-bounding-box-before-vector-search/)
- Related topic: [Hybrid Spatial and Keyword Retrieval](/geospatial-rag-pipelines/hybrid-spatial-keyword-retrieval/)
- Concept: [Spatial Embedding Models](/spatial-llm-architecture-core-concepts/spatial-embedding-models/)
