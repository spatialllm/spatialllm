---
title: How to Tokenize Polygon Boundaries for Transformer Models
description: Turn a ring of vertices into a token sequence a model reads reliably — canonical ordering, vertex reduction, delta encoding, and a budget that never truncates mid-number.
slug: how-to-tokenize-polygon-boundaries-for-transformer-models
type: howto
breadcrumb: Tokenizing Polygon Boundaries
datePublished: 2025-01-23
dateModified: 2026-08-11
---

# How to Tokenize Polygon Boundaries for Transformer Models

A polygon boundary is an ordered ring of coordinates, and every decision about how that ring becomes text changes both its cost and how reliably a model can work with it. This guide covers the four decisions that matter — where the ring starts, which vertices survive, how the numbers are written, and what happens when the result does not fit — as the working core of [geometry tokenization strategies](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/).

## When to Use This Approach

Apply it whenever a boundary reaches a prompt. If the model only ever needs to know which region something is in, cells or a name are cheaper and better; boundary tokenization is for questions where the shape itself is the subject.

| The model needs to | Send | Because |
|--------------------|------|---------|
| Know which region a point is in | A name or a cell identifier | The boundary is irrelevant to the answer |
| Compare two shapes | Both boundaries, canonical form | Ordering differences read as shape differences |
| Describe a shape's character | A reduced boundary | Vertex count far exceeds what the description needs |
| Reproduce or edit a shape | A structured exact form | Compact text breaks under editing |
| Measure something | Nothing — compute it | A model measuring from vertices is guessing |

The last row is worth stating plainly because it is the most common misuse. A model handed a boundary and asked for its area will produce a number, and the number will be wrong in a way that scales with the shape's complexity.

<figure class="diagram">
<svg viewBox="0 0 780 250" role="img" aria-labelledby="tpb-start-t tpb-start-d" xmlns="http://www.w3.org/2000/svg"><title id="tpb-start-t">The same ring, three serialisations</title><desc id="tpb-start-d">One polygon written from three different starting vertices and two ring directions produces three different token sequences, which defeats caching, deduplication and any comparison between two shapes.</desc><rect x="0" y="0" width="780" height="250" fill="#ffffff"/><text x="390" y="34" fill="#5b6471" font-size="13" text-anchor="middle">One shape, three legal serialisations, three different token sequences</text><rect x="30" y="58" width="230" height="110" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="275" y="58" width="230" height="110" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="520" y="58" width="230" height="110" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="12.5" text-anchor="middle"><text x="145" y="88">starts north-west</text><text x="390" y="88">starts south-east</text><text x="635" y="88">reversed direction</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="145" y="116">cache key A</text><text x="390" y="116">cache key B</text><text x="635" y="116">cache key C</text><text x="145" y="142">identical shape</text><text x="390" y="142">identical shape</text><text x="635" y="142">identical shape</text></g><rect x="30" y="192" width="720" height="42" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="390" y="219" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">Canonicalise once at ingestion: fixed direction, deterministic starting vertex</text></svg>
<figcaption><b>Three cache misses for one shape.</b> Nothing is wrong with any of these serialisations individually; the cost appears in everything built on top — caching, deduplication, change detection and any prompt that asks a model to compare two boundaries.</figcaption>
</figure>

## Implementation

The tokenizer canonicalises, reduces, formats and budgets, in that order, and reports what it did at each step.

```python
import logging
from dataclasses import dataclass

from shapely.geometry.polygon import orient
from shapely.errors import GEOSException
from shapely.validation import make_valid

log = logging.getLogger("boundary_tokenizer")


@dataclass(frozen=True)
class Boundary:
    text: str
    vertices: int
    original_vertices: int
    places: int
    note: str


def canonical_ring(geom):
    """Fixed exterior direction and a deterministic starting vertex."""
    try:
        oriented = orient(geom, sign=1.0)          # exterior counter-clockwise
    except Exception as exc:
        log.info("could not orient geometry (%s); using it as given", exc)
        oriented = geom
    coords = list(oriented.exterior.coords)[:-1]   # drop the repeated closing vertex
    if not coords:
        return []
    start = min(range(len(coords)), key=lambda i: (coords[i][0], coords[i][1]))
    return coords[start:] + coords[:start] + [coords[start]]


def reduce_vertices(geom, target: int, to_metric, from_metric):
    """Simplify toward a vertex target by searching tolerance, preserving topology."""
    if len(geom.exterior.coords) <= target:
        return geom, 0.0
    projected = to_metric(geom)
    lo, hi = 0.0, max(projected.bounds[2] - projected.bounds[0],
                      projected.bounds[3] - projected.bounds[1]) / 20.0
    best, best_tol = projected, 0.0
    for _ in range(12):                            # bisection: 12 steps is plenty
        mid = (lo + hi) / 2
        try:
            candidate = projected.simplify(mid, preserve_topology=True)
        except GEOSException:
            break
        if candidate.is_empty:
            hi = mid
            continue
        if len(candidate.exterior.coords) > target:
            lo = mid
        else:
            best, best_tol, hi = candidate, mid, mid
    if not best.is_valid:
        best = make_valid(best)
    return from_metric(best), best_tol
```

Bisecting on tolerance rather than picking one is what makes the vertex target meaningful. A fixed tolerance produces wildly different vertex counts across features of different sizes, whereas a target vertex count produces a boundary whose cost is predictable — which is what a budget needs.

The formatting step writes the ring as pairs and enforces the budget by reducing further, never by cutting the string.

```python
def tokenize_boundary(geom, places: int, budget: int, count_tokens,
                      to_metric, from_metric, target: int = 64) -> Boundary:
    """Canonical, reduced, budgeted. Never truncates a coordinate list."""
    original = len(geom.exterior.coords)
    reduced, tol = reduce_vertices(geom, target, to_metric, from_metric)
    ring = canonical_ring(reduced)

    def render(r, p) -> str:
        pairs = ", ".join(
            f"{round(x, p):.{p}f}".rstrip('0').rstrip('.') + " " +
            f"{round(y, p):.{p}f}".rstrip('0').rstrip('.')
            for x, y in r)
        return f"POLYGON(({pairs}))"

    text = render(ring, places)
    if count_tokens(text) <= budget:
        note = "" if tol == 0 else f"simplified at {tol:.2f} m"
        return Boundary(text, len(ring), original, places, note)

    for fewer_places in range(places - 1, 1, -1):   # cheapest reduction first
        text = render(ring, fewer_places)
        if count_tokens(text) <= budget:
            return Boundary(text, len(ring), original, fewer_places,
                            f"precision reduced to {fewer_places} places")

    for fewer_vertices in (32, 16, 8):
        reduced, tol = reduce_vertices(geom, fewer_vertices, to_metric, from_metric)
        ring = canonical_ring(reduced)
        text = render(ring, max(2, places - 2))
        if count_tokens(text) <= budget:
            return Boundary(text, len(ring), original, max(2, places - 2),
                            f"reduced to {len(ring)} vertices at {tol:.2f} m")

    minx, miny, maxx, maxy = geom.bounds
    log.info("boundary reduced to its extent to fit %d tokens", budget)
    return Boundary(f"BBOX({minx:.4f} {miny:.4f}, {maxx:.4f} {maxy:.4f})",
                    4, original, 4, "replaced by its bounding extent")
```

Reducing precision before reducing vertices is the right order because precision loss is invisible at the scales most questions care about, while vertex loss changes the shape's outline. Both are reported, so a consumer can tell which happened.

<figure class="diagram">
<svg viewBox="16 24 698 206" role="img" aria-labelledby="tpb-bisect-t tpb-bisect-d" xmlns="http://www.w3.org/2000/svg"><title id="tpb-bisect-t">Fixed tolerance against a vertex target</title><desc id="tpb-bisect-d">A single simplification tolerance leaves a small feature untouched and destroys a large one, while bisecting toward a vertex target gives every feature a comparable, predictable cost.</desc><rect x="16" y="24" width="698" height="206" fill="#ffffff"/><text x="30" y="62" fill="#b3324f" font-size="13" font-weight="600">fixed tolerance</text><rect x="220" y="38" width="90" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="326" y="38" width="230" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="572" y="38" width="60" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="11.5" text-anchor="middle"><text x="265" y="64">12 pts</text><text x="441" y="64">310 pts</text><text x="602" y="64">6 pts</text></g><text x="30" y="152" fill="#12805c" font-size="13" font-weight="600">vertex target</text><rect x="220" y="128" width="120" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="356" y="128" width="130" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="502" y="128" width="126" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="11.5" text-anchor="middle"><text x="280" y="154">64 pts</text><text x="421" y="154">64 pts</text><text x="565" y="154">62 pts</text></g><text x="390" y="212" fill="#1f2937" font-size="13" text-anchor="middle">Predictable cost per feature is what makes a per-layer budget possible at all</text></svg>
<figcaption><b>Cost predictability is the point.</b> A budget cannot allocate across layers when one feature might cost twelve tokens and its neighbour three hundred; a vertex target makes every feature roughly the same size.</figcaption>
</figure>

## Validation & Testing

```python
from shapely.geometry import Polygon


def test_canonical_form_is_rotation_invariant():
    a = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    b = Polygon([(1, 1), (1, 0), (0, 0), (0, 1)])
    assert canonical_ring(a) == canonical_ring(b)


def test_budget_is_met_without_truncation():
    out = tokenize_boundary(COMPLEX_POLYGON, 5, 200, count_tokens, to_m, from_m)
    assert count_tokens(out.text) <= 200
    assert out.text.count("(") == out.text.count(")")


def test_reduction_is_reported():
    out = tokenize_boundary(COMPLEX_POLYGON, 5, 200, count_tokens, to_m, from_m)
    assert out.note and out.vertices < out.original_vertices


def test_vertex_target_is_approximately_met():
    reduced, _ = reduce_vertices(COMPLEX_POLYGON, 64, to_m, from_m)
    assert 32 <= len(reduced.exterior.coords) <= 80
```

The first test is the one worth writing first: rotation invariance is easy to lose to a refactor, and its absence shows up not as an error but as a cache hit rate that quietly falls to zero.

## Gotchas & Edge Cases

**Interior rings dropped silently.** A polygon with holes has more than one ring, and a tokenizer that renders only the exterior produces a shape that is wrong in exactly the region a hole was meant to exclude. Render all rings, and if the budget forces a choice, drop the smallest holes and say so.

**Canonical start chosen by index rather than by geometry.** Starting at the first vertex in the file makes the serialisation depend on the exporter. Choosing the extreme vertex, as above, makes it depend only on the shape.

<figure class="diagram">
<svg viewBox="46 36 628 194" role="img" aria-labelledby="tpb-holes-t tpb-holes-d" xmlns="http://www.w3.org/2000/svg"><title id="tpb-holes-t">A polygon rendered without its interior rings</title><desc id="tpb-holes-d">A parcel with an excluded courtyard rendered from its exterior ring alone claims the courtyard as part of the parcel, which is wrong in precisely the region the hole existed to exclude.</desc><rect x="46" y="36" width="628" height="194" fill="#ffffff"/><rect x="60" y="50" width="240" height="140" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="140" y="96" width="90" height="60" rx="4" fill="#ffffff" stroke="#12805c" stroke-width="2"/><text x="180" y="212" fill="#1f2937" font-size="12.5" text-anchor="middle">the real parcel, with a hole</text><rect x="420" y="50" width="240" height="140" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="540" y="126" fill="#1f2937" font-size="12.5" text-anchor="middle">exterior ring only</text><text x="540" y="212" fill="#1f2937" font-size="12.5" text-anchor="middle">the hole is silently claimed</text></svg>
<figcaption><b>The error is confined to the hole and total within it.</b> Every containment question about the excluded area now answers the opposite of the truth, and the geometry passes every validity check on the way there.</figcaption>
</figure>

**Simplification applied in degrees.** A tolerance in degrees is a different distance at every latitude, so a corpus spanning a continent is reduced unevenly by a constant that looks uniform. Project first.

**A vertex target smaller than the shape needs.** Reducing a complex coastline to eight vertices produces a triangle that is technically a polygon and describes nothing. Floor the target, and prefer the extent rung over an absurd reduction — an extent at least announces itself.

**Rings that are not closed.** Some exporters omit the repeated closing vertex and some readers require it, so a tokenizer that assumes one convention will occasionally emit a ring no parser accepts. Close explicitly when rendering, as the canonical form does.

**Multipart geometries flattened to their largest part.** Common, convenient, and wrong for anything that spans islands or detached parcels. Render the parts that fit, count the ones dropped, and report both.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Is delta encoding worth it for long rings?</span></summary><p>Sometimes, and less often than it looks. Writing each vertex as an offset from the previous one shortens the numbers considerably for a dense ring, which does reduce tokens — but it produces text no standard parser accepts and that a model handles less reliably, since it must reconstruct absolute positions to reason about them. Reserve it for cases where the ring is the whole payload and a custom parser already exists on the other side.</p></details>

<details class="faq-item"><summary><span>Should the boundary or the centroid be sent for a "where is it" question?</span></summary><p>Neither, usually — send the name and let the geometry stay in the database. Where the position genuinely matters, a representative point is cheaper and more robust than a boundary, and unlike a centroid it is guaranteed to lie inside the shape, which matters for concave features where the centroid falls outside.</p></details>

<details class="faq-item"><summary><span>How many vertices does a model actually need?</span></summary><p>Far fewer than most corpora carry. For describing character and arrangement, a few dozen is ample; beyond that the additional vertices mostly encode survey precision that the question never touches. The exception is comparison — two shapes reduced to different vertex counts compare badly — which is another reason to use one target across the corpus rather than per feature.</p></details>

<details class="faq-item"><summary><span>What should happen when a boundary crosses the antimeridian?</span></summary><p>Split it before tokenizing. A ring whose longitudes jump from 179 to −179 is rendered as an enormous shape spanning the globe by every naive reader, including the model. Splitting produces two rings that are each unremarkable, and the note should record that the feature was split so nothing downstream treats them as separate features.</p></details>

<details class="faq-item"><summary><span>Should the vertex count reach the prompt alongside the boundary?</span></summary><p>Yes, when the boundary was reduced, and it costs almost nothing to include. A model told that a shape has been reduced from three hundred vertices to sixty-four will hedge appropriately about fine detail; one handed sixty-four vertices with no context treats them as the whole truth. The same applies to the tolerance: a stated figure in metres is far more useful to a reader than a general note that simplification occurred.</p></details>

## Related

- Up to the parent topic: [Geometry Tokenization Strategies](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/)
- [Coordinate Precision Versus Token Cost](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/coordinate-precision-versus-token-cost/)
- [Well-Known Text, Structured Objects and Cell Identifiers](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/wkt-vs-geojson-vs-h3-for-llm-tokenization/)
- Related topic: [Context-Window Optimization for Maps](/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/)
