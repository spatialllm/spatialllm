---
title: Verifying Topological Claims with DE-9IM
description: Check containment, adjacency and crossing exactly using the nine-intersection matrix, so a model's spatial assertions are confirmed, corrected or marked unverifiable.
slug: verifying-topological-claims-with-de9im
type: howto
breadcrumb: Verifying Topological Claims
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Verifying Topological Claims with DE-9IM

Named predicates like "contains" and "touches" are convenient and occasionally not what a question means. The nine-intersection matrix underneath them describes exactly how two geometries relate, and reading it directly lets you check a claim against the relation the user actually intended rather than the closest available function name. This guide uses it to verify model assertions, as the exact-checking stage of [spatial reasoning and relation inference](/spatial-llm-architecture-core-concepts/spatial-reasoning-and-relation-inference/).

## When to Use This Approach

Use the named predicates for ordinary cases and drop to the matrix when a claim falls between them, when the boundary case matters, or when you need to explain to a reader why a claim failed.

| Claim | Named predicate | When it is not enough |
|-------|-----------------|-----------------------|
| "inside" | contains | A shape touching the boundary from within |
| "next to" | touches | Data digitised separately never touches exactly |
| "crosses" | crosses | Lines that overlap for a stretch rather than crossing |
| "overlaps" | overlaps | Requires equal dimension; a point never overlaps a polygon |
| "covers" | covers | Differs from contains exactly at the boundary |

The distinction between containing and covering is the one that surfaces most often in practice: a parcel whose edge lies exactly on a district boundary is covered by the district and not contained in it, and a system that reports "not inside" for that case will be told, correctly, that it is wrong.

<figure class="diagram">
<svg viewBox="36 36 698 208" role="img" aria-labelledby="d9-cases-t d9-cases-d" xmlns="http://www.w3.org/2000/svg"><title id="d9-cases-t">Contains against covers at a shared boundary</title><desc id="d9-cases-d">A parcel wholly inside a district is both contained and covered; a parcel whose edge lies on the district boundary is covered but not contained, which is the case that produces disputed answers.</desc><rect x="36" y="36" width="698" height="208" fill="#ffffff"/><rect x="50" y="50" width="280" height="150" rx="6" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="110" y="96" width="120" height="70" rx="4" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="190" y="226" fill="#1f2937" font-size="12.5" text-anchor="middle">contained and covered</text><rect x="440" y="50" width="280" height="150" rx="6" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="440" y="96" width="120" height="70" rx="4" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="580" y="226" fill="#1f2937" font-size="12.5" text-anchor="middle">covered, not contained</text><text x="580" y="76" fill="#5b6471" font-size="12" text-anchor="middle">the parcel edge lies on the boundary</text></svg>
<figcaption><b>One shared edge changes the answer.</b> Both parcels are, in ordinary language, inside the district; only one of them satisfies the strict containment predicate, which is why the claim has to be mapped to a relation before it is checked.</figcaption>
</figure>

## Implementation

The verifier maps a natural-language claim to an intersection pattern, evaluates it, and returns a three-valued verdict with the matrix so a failure can be explained.

```python
import logging
from dataclasses import dataclass
from typing import Optional

from shapely.geometry.base import BaseGeometry
from shapely.errors import GEOSException
from shapely.validation import make_valid

log = logging.getLogger("topology_verification")

# Patterns describing how the interiors, boundaries and exteriors of two geometries meet.
# "T" means they must intersect, "F" that they must not, "*" that it does not matter.
PATTERNS = {
    "inside":     "T*F**F***",     # strictly within, boundary not touching the outside
    "covered_by": "T*F**F***",     # same test, applied with covers semantics below
    "adjacent":   "F***T****",     # boundaries meet, interiors do not
    "crosses":    "T*T******",
    "overlaps":   "T*T***T**",
}


@dataclass(frozen=True)
class Verdict:
    holds: Optional[bool]          # None means unverifiable, never a guess
    matrix: Optional[str]
    note: str


def _prepare(geom: BaseGeometry) -> Optional[BaseGeometry]:
    if geom is None or geom.is_empty:
        return None
    if geom.is_valid:
        return geom
    try:
        fixed = make_valid(geom)
        return fixed if not fixed.is_empty else None
    except GEOSException as exc:
        log.warning("geometry could not be repaired: %s", exc)
        return None


def verify(subject: BaseGeometry, reference: BaseGeometry, claim: str,
           tolerance_m: float = 0.0, to_metric=None) -> Verdict:
    """Verify one relation claim exactly. Returns None when it cannot be checked."""
    a, b = _prepare(subject), _prepare(reference)
    if a is None or b is None:
        return Verdict(None, None, "one or both geometries are unavailable or unusable")

    if tolerance_m > 0 and to_metric is not None:
        try:
            a = to_metric(a).buffer(tolerance_m)      # tolerance for separately digitised data
            b = to_metric(b)
        except Exception as exc:
            return Verdict(None, None, f"could not apply tolerance: {exc}")

    pattern = PATTERNS.get(claim)
    if pattern is None:
        return Verdict(None, None, f"no intersection pattern defined for claim {claim!r}")

    try:
        matrix = a.relate(b)
        holds = a.relate_pattern(b, pattern)
    except GEOSException as exc:
        return Verdict(None, None, f"relation computation failed: {exc}")

    if claim == "covered_by":
        holds = b.covers(a)                            # boundary-inclusive containment
    return Verdict(holds, matrix, "" if holds else f"matrix {matrix} does not match {pattern}")
```

Returning the matrix alongside the verdict is what turns a failed check into an explanation. "Not inside" is a bare contradiction; "not inside — the parcel's boundary meets the district's boundary, so it is covered rather than contained" is an answer that resolves the disagreement in one sentence.

The tolerance parameter exists because exact adjacency is rare in real data. Two parcels digitised in different surveys will be a few centimetres apart or a few centimetres overlapping, and a strict boundary-touching test reports neighbours as unrelated. Applying a small buffer to one side, in a metric projection, restores the intended meaning — and the tolerance must be reported with the answer, because "adjacent within half a metre" is a materially different claim from "adjacent".

<figure class="diagram">
<svg viewBox="46 46 708 184" role="img" aria-labelledby="d9-tol-t d9-tol-d" xmlns="http://www.w3.org/2000/svg"><title id="d9-tol-t">Why exact adjacency fails on separately digitised data</title><desc id="d9-tol-d">Two parcels captured in different surveys sit a few centimetres apart, so a strict touching test reports them as unrelated even though every reader would call them neighbours.</desc><rect x="46" y="46" width="708" height="184" fill="#ffffff"/><rect x="60" y="60" width="150" height="120" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="218" y="60" width="150" height="120" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="214" y="212" fill="#1f2937" font-size="12.5" text-anchor="middle">a few centimetres apart on the ground</text><rect x="440" y="60" width="300" height="54" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="590" y="92" fill="#1f2937" font-size="12.5" text-anchor="middle">strict test: not adjacent</text><rect x="440" y="128" width="300" height="54" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="590" y="152" fill="#1f2937" font-size="12.5" text-anchor="middle">with a stated tolerance: adjacent</text><text x="590" y="172" fill="#5b6471" font-size="12" text-anchor="middle">and the tolerance appears in the answer</text></svg>
<figcaption><b>Capture accuracy, not topology.</b> The gap is an artefact of two surveys, and a verification layer that ignores it will spend its time contradicting people about facts they can see on the ground.</figcaption>
</figure>

## Validation & Testing

```python
from shapely.geometry import box, Point


def test_edge_touching_parcel_is_covered_not_contained():
    district = box(0, 0, 10, 10)
    parcel = box(0, 2, 4, 6)                       # shares the western edge
    assert verify(parcel, district, "inside").holds is False
    assert verify(parcel, district, "covered_by").holds is True


def test_unusable_geometry_returns_none_not_false():
    v = verify(None, box(0, 0, 1, 1), "inside")
    assert v.holds is None and "unavailable" in v.note


def test_unknown_claim_is_unverifiable_not_false():
    v = verify(box(0, 0, 1, 1), box(0, 0, 2, 2), "upstream_of")
    assert v.holds is None and "no intersection pattern" in v.note


def test_matrix_is_returned_for_a_failed_claim():
    v = verify(Point(20, 20), box(0, 0, 10, 10), "inside")
    assert v.holds is False and v.matrix and len(v.matrix) == 9
```

The second and third tests are the ones that keep the three-valued contract honest. Every simplification of this function will be tempted to return `False` for an input it could not evaluate, and that single change converts "we do not know" into "we checked and it is not so" across the entire system.

## Gotchas & Edge Cases

**Patterns applied to mismatched dimensions.** Several relations are only defined between geometries of the same dimension: a point never overlaps a polygon, however much it may sit inside one. Check dimensions before applying a pattern, and report a dimension mismatch as unverifiable rather than false.

**Tolerance applied in degrees.** Buffering by a value that looks like half a metre but is expressed in degrees produces a buffer of tens of kilometres. Project before buffering, always, and assert the projection in the same test that asserts the tolerance.

**Buffering both sides.** Applying the tolerance to both geometries doubles it silently. Buffer one side only, and state which.

<figure class="diagram">
<svg viewBox="26 42 708 158" role="img" aria-labelledby="d9-dim-t d9-dim-d" xmlns="http://www.w3.org/2000/svg"><title id="d9-dim-t">Relations that require matching dimensions</title><desc id="d9-dim-d">Overlapping is only defined between geometries of the same dimension, so a point-in-polygon question expressed as an overlap returns false for a point that is plainly inside.</desc><rect x="26" y="42" width="708" height="158" fill="#ffffff"/><rect x="40" y="56" width="330" height="130" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="400" y="56" width="320" height="130" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="205" y="88">same dimension</text><text x="560" y="88">mixed dimension</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="205" y="118">polygon and polygon</text><text x="205" y="144">overlap is meaningful</text><text x="560" y="118">point and polygon</text><text x="560" y="144">overlap is always false</text><text x="560" y="168">use containment instead</text></g></svg>
<figcaption><b>Always false is not the same as never true.</b> The predicate is behaving exactly as specified; the claim was simply expressed with a relation that cannot hold for these two kinds of shape.</figcaption>
</figure>

**Invalid geometry silently repaired into a different answer.** A self-intersecting polygon repaired by splitting into parts can genuinely change a containment result. Report that a repair occurred alongside the verdict, so a surprising answer can be traced to it.

**A claim verified against the wrong reference.** The commonest logical error is checking "is the site in the conservation area" against a different conservation area with a similar name. Verification is only as good as the grounding that supplied the reference, which is why place resolution carries identifiers rather than names.

**Relations checked in a projected frame with a different origin.** Both geometries must be in the same frame before any predicate runs. A mismatch produces a confident answer computed over geometry that is nowhere near each other, and no predicate detects it.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should the matrix be shown to the user?</span></summary><p>Rarely to an end user and always to a developer. The nine-character string is meaningless to most readers and is the fastest possible explanation for anyone debugging a disputed claim. The useful pattern is to translate it into a sentence for the answer — "the boundaries meet but the interiors do not" — while logging the raw matrix for anyone who needs to reproduce the check.</p></details>

<details class="faq-item"><summary><span>What tolerance is defensible for adjacency?</span></summary><p>The capture accuracy of the less accurate of the two datasets, which is usually documented and is almost never zero. Choosing it from the data rather than from convenience makes the resulting claim defensible: "adjacent within the stated accuracy of both sources" is a statement you can put in front of a surveyor. A tolerance chosen because it made a test pass is not.</p></details>

<details class="faq-item"><summary><span>How should relations between more than two geometries be handled?</span></summary><p>As a conjunction of pairwise checks, each verified and reported separately. A claim that a site is inside one area and outside another is two claims, and collapsing them into a single verdict loses the information that one held and the other did not. The answer layer can then assert the part that verified and hedge the part that did not, which is almost always more useful than an all-or-nothing result.</p></details>

<details class="faq-item"><summary><span>Is it worth caching verification results?</span></summary><p>Yes, keyed on the pair of geometry versions rather than on identifiers, because geometry changes and a cached topology result computed against a superseded boundary is exactly the kind of stale answer that erodes trust. With version-keyed caching the cache invalidates itself when either shape is corrected, which is the behaviour you want and is difficult to arrange any other way.</p></details>

One last piece of advice on adoption. Introduce this verifier as a check on answers before introducing it as a source of them: run it silently for a period, log every claim it would have retracted, and read the list. That list is a direct measurement of how often the system currently asserts spatial relations it cannot support, and it is usually more persuasive than any argument for adding the layer.

## Related

- Up to the parent topic: [Spatial Reasoning and Relation Inference](/spatial-llm-architecture-core-concepts/spatial-reasoning-and-relation-inference/)
- [Answering Direction and Distance Queries Deterministically](/spatial-llm-architecture-core-concepts/spatial-reasoning-and-relation-inference/answering-direction-and-distance-queries-deterministically/)
- Related topic: [Topology Rule Enforcement via LLMs](/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/)
- Concept: [Geocoding and Place-Name Resolution](/spatial-llm-architecture-core-concepts/geocoding-and-place-name-resolution/)
