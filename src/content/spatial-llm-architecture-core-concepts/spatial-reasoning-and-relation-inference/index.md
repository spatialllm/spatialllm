---
title: Spatial Reasoning and Relation Inference
description: Verify the spatial relations a model asserts — containment, adjacency, direction and distance — against real geometry, and decide what it may claim when it cannot.
slug: spatial-reasoning-and-relation-inference
type: topic
breadcrumb: Reasoning and Relations
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Spatial Reasoning and Relation Inference

"The depot is north of the river and inside the enterprise zone" is two spatial claims and a conjunction. A language model will produce sentences like that fluently from context that supports neither claim, and nothing about the sentence marks which parts were computed and which were composed. This topic covers turning relation claims into computations over real geometry, and deciding what an agent may say when the geometry is not available.

It sits within [spatial LLM architecture and core concepts](/spatial-llm-architecture-core-concepts/) and depends on two other stages: the positions come from [geocoding and place-name resolution](/spatial-llm-architecture-core-concepts/geocoding-and-place-name-resolution/), and the frames they are expressed in come from [coordinate reference system normalization](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/). Without both, a relation check computes the right predicate over the wrong geometry and produces a confident answer that happens to be false.

<figure class="diagram">
<svg viewBox="16 16 768 242" role="img" aria-labelledby="srr-claim-t srr-claim-d" xmlns="http://www.w3.org/2000/svg"><title id="srr-claim-t">Decomposing a fluent sentence into checkable claims</title><desc id="srr-claim-d">One sentence contains a directional claim, a containment claim and a distance claim; each becomes a separate predicate over real geometry, and each can independently pass, fail or be unverifiable.</desc><rect x="16" y="16" width="768" height="242" fill="#ffffff"/><rect x="140" y="30" width="520" height="48" rx="8" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><text x="400" y="60" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">the depot is north of the river, inside the zone, 300 m from the gate</text><rect x="30" y="126" width="220" height="84" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="290" y="126" width="220" height="84" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="550" y="126" width="220" height="84" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="140" y="156">direction</text><text x="400" y="156">containment</text><text x="660" y="156">distance</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="140" y="182">needs a bearing rule</text><text x="400" y="182">a real predicate</text><text x="660" y="182">needs a metric frame</text></g><g stroke="#5b6471" stroke-width="2" marker-end="url(#srr-claim-a)"><line x1="330" y1="80" x2="150" y2="122"/><line x1="400" y1="80" x2="400" y2="122"/><line x1="470" y1="80" x2="650" y2="122"/></g><defs><marker id="srr-claim-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/></marker></defs><text x="400" y="240" fill="#1f2937" font-size="13" text-anchor="middle">Three claims, three verdicts — a sentence that passes as a whole may not</text></svg>
<figcaption><b>Fluency hides the joins.</b> A single sentence can be two-thirds verified and one-third invented, and unless it is decomposed before checking, the verified parts lend their credibility to the invented one.</figcaption>
</figure>

## Foundational Principles

**A relation is computed or it is not asserted.** Containment, intersection, adjacency and crossing all have exact definitions and exact implementations. If both geometries are available, compute the predicate; if either is missing, the honest output is that the relation could not be checked. There is no useful middle where a model's impression of a relation is treated as evidence.

**Direction is a convention, and the convention must be stated.** "North of" can mean a bearing within forty-five degrees of true north, or a comparison of latitudes, or a relation between the nearest points of two extended shapes. These disagree constantly for real geometry. Pick one, implement it once, and say which one an answer used.

**Distance needs a metric frame and a defined endpoint.** Degrees are not metres; centroids are not edges. A distance answer without both decisions recorded is not reproducible, and the two errors compound — a centroid distance in degrees at high latitude can be wrong by a factor of two.

## Step-by-Step Implementation Pipeline

### 1. Extract the claims before checking any of them

Claim extraction turns prose into a list of typed assertions with named subjects. It is deliberately mechanical: a claim the extractor misses simply goes unchecked, which is safer than a claim it invents.

```python
import logging
import re
from dataclasses import dataclass

log = logging.getLogger("relation_claims")

_RELATIONS = {
    "inside": "contains", "within": "contains", "in the": "contains",
    "north of": "direction", "south of": "direction",
    "east of": "direction", "west of": "direction",
    "next to": "adjacent", "adjacent to": "adjacent", "borders": "adjacent",
    "crosses": "crosses", "intersects": "intersects",
}


@dataclass(frozen=True)
class Claim:
    subject: str
    relation: str          # normalised relation kind
    obj: str
    raw: str


def extract_claims(sentence: str, place_names: frozenset[str]) -> list[Claim]:
    """Pull typed relation claims out of a sentence. Misses are safe; inventions are not."""
    claims: list[Claim] = []
    lowered = sentence.lower()
    for phrase, kind in _RELATIONS.items():
        for m in re.finditer(re.escape(phrase), lowered):
            before, after = sentence[:m.start()], sentence[m.end():]
            subject = _nearest_name(before, place_names, from_end=True)
            obj = _nearest_name(after, place_names, from_end=False)
            if subject and obj:
                claims.append(Claim(subject, kind, obj, phrase))
    if not claims:
        log.info("no relation claim found in %r", sentence[:80])
    return claims
```

### 2. Compute topological relations exactly

For containment, intersection, adjacency and crossing there is an exact answer, obtained from a real predicate over real geometry in a shared frame. The nine-intersection model underlying those predicates, and how to use it to check a claim precisely, is covered in [verifying topological claims](/spatial-llm-architecture-core-concepts/spatial-reasoning-and-relation-inference/verifying-topological-claims-with-de9im/).

```python
from shapely.geometry.base import BaseGeometry
from shapely.errors import GEOSException
from shapely.validation import make_valid


def check_topological(a: BaseGeometry, b: BaseGeometry, kind: str) -> tuple[bool | None, str]:
    """Return (verdict, note). None means unverifiable, never a guess."""
    if a is None or b is None:
        return None, "one or both geometries unavailable"
    try:
        a = a if a.is_valid else make_valid(a)
        b = b if b.is_valid else make_valid(b)
    except GEOSException as exc:
        return None, f"geometry could not be repaired: {exc}"
    try:
        if kind == "contains":
            return b.contains(a), "exact predicate"
        if kind == "intersects":
            return a.intersects(b), "exact predicate"
        if kind == "crosses":
            return a.crosses(b), "exact predicate"
        if kind == "adjacent":
            return a.touches(b) or a.distance(b) < 1e-9, "touching or coincident boundary"
    except GEOSException as exc:
        return None, f"predicate failed: {exc}"
    return None, f"no exact predicate for {kind!r}"
```

Returning `None` rather than `False` for an unverifiable claim is the distinction the whole stage rests on. "Not shown to be true" and "shown to be false" mean different things to a reader, and collapsing them makes an agent sound authoritative about the limits of its own data.

### 3. Define direction once, in code

Direction has no standard predicate, which is precisely why it needs an explicit implementation rather than a per-caller convention. A bearing between representative points, bucketed into named sectors with a stated tolerance, is defensible and reproducible.

```python
import math

SECTOR_HALF_WIDTH_DEG = 45.0        # "north" spans 315° to 45°


def bearing_deg(a: BaseGeometry, b: BaseGeometry, to_metric) -> float | None:
    """Bearing from a to b, in degrees clockwise from north, or None if undefined."""
    try:
        pa, pb = to_metric(a.centroid), to_metric(b.centroid)
    except Exception as exc:
        log.warning("bearing projection failed: %s", exc)
        return None
    dx, dy = pb.x - pa.x, pb.y - pa.y
    if dx == 0 and dy == 0:
        return None                                  # coincident: no direction exists
    return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0


def check_direction(a, b, named: str, to_metric) -> tuple[bool | None, str]:
    """Is a `named` of b, under the stated sector convention?"""
    centres = {"north": 0.0, "east": 90.0, "south": 180.0, "west": 270.0}
    target = centres.get(named)
    if target is None:
        return None, f"unsupported direction {named!r}"
    theta = bearing_deg(b, a, to_metric)             # from the reference to the subject
    if theta is None:
        return None, "geometries coincide; direction is undefined"
    delta = abs((theta - target + 180.0) % 360.0 - 180.0)
    return delta <= SECTOR_HALF_WIDTH_DEG, f"bearing {theta:.0f}°, sector ±{SECTOR_HALF_WIDTH_DEG:.0f}°"
```

Reporting the bearing alongside the verdict costs one string and removes most arguments. A claim that fails at forty-eight degrees is a different situation from one that fails at a hundred and thirty, and a reader can see immediately which they are looking at.

<figure class="diagram">
<svg viewBox="16 9 728 229" role="img" aria-labelledby="srr-dir-t srr-dir-d" xmlns="http://www.w3.org/2000/svg"><title id="srr-dir-t">Three conventions for "north of" disagreeing on one pair</title><desc id="srr-dir-d">A bearing sector, a latitude comparison and a nearest-point rule give different verdicts for the same two shapes, which is why the convention must be chosen once and reported with the answer.</desc><rect x="16" y="9" width="728" height="229" fill="#ffffff"/><text x="380" y="34" fill="#5b6471" font-size="13" text-anchor="middle">Same two shapes, three defensible conventions</text><rect x="30" y="58" width="220" height="120" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="270" y="58" width="220" height="120" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="510" y="58" width="220" height="120" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="140" y="88">bearing sector</text><text x="380" y="88">latitude compare</text><text x="620" y="88">nearest points</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="140" y="116">centroid to centroid</text><text x="140" y="140">within 45 degrees</text><text x="140" y="162">verdict: true</text><text x="380" y="116">any part further north</text><text x="380" y="140">ignores east-west offset</text><text x="380" y="162">verdict: true</text><text x="620" y="116">closest pair of edges</text><text x="620" y="140">for adjacent shapes</text><text x="620" y="162">verdict: false</text></g><text x="380" y="220" fill="#1f2937" font-size="13" text-anchor="middle">Disagreement is normal — the failure is not saying which one produced the answer</text></svg>
<figcaption><b>None of the three is wrong.</b> They answer slightly different questions, and a system that switches between them depending on which code path ran will be inconsistent in a way that looks like a bug in the geometry rather than in the specification.</figcaption>
</figure>

### 4. Measure distance in a metric frame, from a defined endpoint

Distance answers need three decisions recorded: the frame, the endpoint rule, and the rounding. Getting the first wrong is a factor-of-two error at high latitude; getting the second wrong is a tens-of-kilometres error for large shapes.

```python
from shapely.ops import transform
from pyproj import Transformer, CRS


def metric_transformer(focus: BaseGeometry) -> Transformer:
    """A locally accurate equidistant projection centred on the subject."""
    lon, lat = focus.centroid.x, focus.centroid.y
    local = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs")
    return Transformer.from_crs(CRS.from_epsg(4326), local, always_xy=True)


def distance_m(a: BaseGeometry, b: BaseGeometry, endpoint: str = "edge") -> float | None:
    """Distance in metres; edge distance is zero when one shape contains the other."""
    if a is None or b is None:
        return None
    tf = metric_transformer(a)
    try:
        pa, pb = transform(tf.transform, a), transform(tf.transform, b)
    except Exception as exc:
        log.warning("distance projection failed: %s", exc)
        return None
    if endpoint == "centroid":
        return pa.centroid.distance(pb.centroid)
    return pa.distance(pb)                           # edge to edge, 0 if they touch or overlap
```

The choice of edge distance as the default is deliberate: it is the one that matches how people describe proximity, and it returns zero for the containment case, which is the answer a reader expects. Further discussion of when each rule is right is in [answering direction and distance queries deterministically](/spatial-llm-architecture-core-concepts/spatial-reasoning-and-relation-inference/answering-direction-and-distance-queries-deterministically/).

### 5. Assemble a verdict the agent must respect

The checker returns per-claim verdicts, and the agent's output is constrained by them. A claim that failed must not appear as an assertion; a claim that could not be checked must be marked as such rather than dropped.

```python
@dataclass(frozen=True)
class Verdict:
    claim: Claim
    holds: bool | None       # True, False, or None for unverifiable
    note: str


def summarise(verdicts: list[Verdict]) -> dict:
    """A compact instruction set for the answer layer."""
    return {
        "assert": [v.claim.raw for v in verdicts if v.holds is True],
        "retract": [f"{v.claim.subject} is not {v.claim.raw} {v.claim.obj}"
                    for v in verdicts if v.holds is False],
        "hedge": [f"could not check whether {v.claim.subject} is {v.claim.raw} "
                  f"{v.claim.obj} ({v.note})" for v in verdicts if v.holds is None],
    }
```

### 6. Fail the claim, not the turn

A failed relation check should change one sentence, not abort the response. An agent that refuses to answer because one of four claims did not verify is less useful than one that answers three and says plainly that the fourth could not be established. The exception is a claim the rest of the answer depends on — a containment claim that determines which regulations apply — where continuing means building on a foundation known to be false.

```python
DEPENDENT_RELATIONS = {"contains"}     # claims that other statements are built on


def turn_should_stop(verdicts: list[Verdict]) -> bool:
    """Only a falsified load-bearing claim stops the turn."""
    return any(v.holds is False and v.claim.relation in DEPENDENT_RELATIONS
               for v in verdicts)
```

### 7. Handle relations the data cannot support at all

Some relations are asked constantly and cannot be computed from the geometry most corpora hold. "Upstream of" needs a flow network, not two polygons. "Overlooks" needs terrain and, arguably, a viewshed. "Served by" needs a service-area definition that varies by operator. A system that quietly reduces these to a distance check produces answers that are wrong in ways the reader has no way to detect, because the phrasing implies a relation the computation never attempted.

The workable response is a registry of supported relations, with everything outside it routed to an explicit refusal that names what would be needed.

```python
SUPPORTED = {"contains", "intersects", "crosses", "adjacent", "direction", "distance"}

REQUIRES = {
    "upstream": "a hydrological flow network",
    "downstream": "a hydrological flow network",
    "overlooks": "a terrain model and a viewshed computation",
    "served_by": "a service-area definition for the operator in question",
    "accessible_from": "a routable network and a travel-time budget",
}


def route_relation(kind: str) -> tuple[bool, str]:
    """Supported relations proceed; the rest refuse and say what is missing."""
    if kind in SUPPORTED:
        return True, ""
    need = REQUIRES.get(kind)
    if need:
        return False, f"answering {kind!r} needs {need}, which this system does not hold"
    return False, f"{kind!r} is not a relation this system can compute"
```

Naming the missing capability is what turns a refusal into something useful. "I cannot answer that" ends the conversation; "answering that needs a flow network, which this system does not hold" tells the reader whether to look elsewhere or to ask a different question, and it tells the team maintaining the agent what the next capability should be.

### 8. Keep the relation check between the model and the answer

Where this stage sits in the pipeline determines whether it works. Running it before generation, as a set of facts fed into the context, helps and does not constrain — the model may still compose a claim the facts do not support. Running it after generation, as a check on the produced text, constrains but is expensive and awkward to correct.

The arrangement that holds up in practice is both: compute the relations the question implies before generating, put them in context as verified facts, then extract claims from the generated answer and verify that each one is licensed by that fact set. The second pass is cheap because it is checking against facts already computed rather than recomputing geometry.

```python
def licensed(claim: Claim, facts: dict[tuple[str, str, str], bool | None]) -> bool | None:
    """Was this claim already established in the pre-computed fact set?"""
    return facts.get((claim.subject, claim.relation, claim.obj))
```

A claim absent from the fact set is not false; it is a claim about something the pre-pass did not think to check, which usually means the model brought in a place or a relation the question did not mention. That is worth surfacing as a hedge rather than an assertion, and worth logging, because a rising rate of unlicensed claims is the clearest early signal that the pre-pass is asking the wrong questions.

## Failure Modes & Root Causes

**The composed conjunction.** Two verified claims and one invented one, joined by "and", read as a single verified statement. Root cause: checking at sentence granularity rather than claim granularity. Mitigation: decompose first, as step 1 does, and constrain the answer per claim.

**Direction by latitude alone.** A place slightly north and far to the east is reported as "north of", which nobody reading a map would say. Root cause: an implicit convention chosen for convenience. Mitigation: an explicit sector rule, with the bearing reported alongside the verdict.

**Centroid containment.** A point is reported as inside an area because its centroid falls within, or a large area is reported as far away because its centroid is. Root cause: reducing geometry to points before predicates. Mitigation: run predicates on the geometry as stored; reduce only when the question genuinely asks for a representative point.

**Unverifiable rendered as false.** A relation the system could not check is reported as not holding, which is a stronger claim than the evidence supports and one that users act on. Root cause: a two-valued verdict type. Mitigation: three-valued verdicts throughout, as above.

## Production Validation Protocols

1. **Three-valued assertion.** Assert the checker's return type admits `None` and that no code path coerces it to `False`; this is the invariant that keeps "unknown" distinguishable from "untrue".
2. **Convention test.** Assert the direction convention is applied identically from every entry point, using a fixture where the conventions disagree.
3. **Frame agreement gate.** Assert both geometries share a frame before any predicate runs; a mismatch must raise rather than silently compare degrees against metres.
4. **Containment property test.** Generate nested shapes and assert containment is transitive and that edge distance is zero whenever containment holds.
5. **Claim-coverage indicator.** Track the share of extracted claims that could be checked; a falling rate means geometry is going missing upstream, not that the questions changed.
6. **Load-bearing stop test.** Assert that a falsified containment claim stops the turn while a falsified direction claim only retracts a sentence.

<figure class="diagram">
<svg viewBox="0 42 780 204" role="img" aria-labelledby="srr-verdict-t srr-verdict-d" xmlns="http://www.w3.org/2000/svg"><title id="srr-verdict-t">Three verdicts and what the answer layer does with each</title><desc id="srr-verdict-d">A verified claim may be asserted, a falsified claim must be retracted, and an unverifiable claim must be hedged with the reason it could not be checked.</desc><rect x="0" y="42" width="780" height="204" fill="#ffffff"/><rect x="30" y="56" width="230" height="130" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="275" y="56" width="230" height="130" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="520" y="56" width="230" height="130" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="145" y="88">true</text><text x="390" y="88">false</text><text x="635" y="88">unverifiable</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="145" y="118">state it plainly</text><text x="145" y="144">cite the predicate</text><text x="145" y="168">no hedge needed</text><text x="390" y="118">do not state it</text><text x="390" y="144">correct it explicitly</text><text x="390" y="168">stop if load-bearing</text><text x="635" y="118">say it was not checked</text><text x="635" y="144">give the reason</text><text x="635" y="168">never imply either way</text></g><text x="390" y="228" fill="#1f2937" font-size="13" text-anchor="middle">Two-valued logic collapses the third column into the second — a stronger claim than the evidence</text></svg>
<figcaption><b>The middle and right columns are where the value is.</b> Anyone can report a relation that holds; what distinguishes a reliable spatial agent is that it distinguishes a relation it disproved from one it never managed to test.</figcaption>
</figure>

## Frequently Asked Questions

<details class="faq-item"><summary><span>Can a model be trusted to extract claims even if it cannot verify them?</span></summary><p>Yes, and this is a good use of it — extraction is a language task where a miss costs an unchecked claim rather than a wrong one. What matters is that extraction and verification stay separate processes, so the model never both proposes a relation and confirms it. Where a model-based extractor helps most is with phrasing the rule-based one misses, such as "just upstream of" or "on the far side of".</p></details>

<details class="faq-item"><summary><span>What tolerance should adjacency use?</span></summary><p>Whatever tolerance the source data was captured at, which is usually documented and is almost never zero. Two parcels digitised separately will rarely share vertices exactly, so a strict touching predicate reports neighbours as non-adjacent. Use a small positive threshold derived from the data's stated accuracy, and record it in the answer, since "adjacent within 0.5 m" is a materially different claim from "adjacent".</p></details>

<details class="faq-item"><summary><span>How should relations between a point and a line be handled?</span></summary><p>Carefully, because containment is almost never what is meant. A point is essentially never exactly on a line in real data, so a containment check returns false for a point that any reader would say is on the road. Convert the question to a distance-within-tolerance check and say so, rather than reporting a technically correct false.</p></details>

<details class="faq-item"><summary><span>Does this replace the need for evaluation?</span></summary><p>No — it changes what evaluation measures. With verification in place, the interesting metric is no longer how often the model asserts a false relation, which should be near zero, but how often a claim is unverifiable, which measures data coverage rather than model quality. Track both, and see <a href="/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/">evaluation and benchmarking for spatial LLMs</a> for how they fit into a release gate.</p></details>

## Related

- Up to the section overview: [Spatial LLM Architecture and Core Concepts](/spatial-llm-architecture-core-concepts/)
- Technique: [Verifying Topological Claims with the Nine-Intersection Model](/spatial-llm-architecture-core-concepts/spatial-reasoning-and-relation-inference/verifying-topological-claims-with-de9im/)
- Technique: [Answering Direction and Distance Queries Deterministically](/spatial-llm-architecture-core-concepts/spatial-reasoning-and-relation-inference/answering-direction-and-distance-queries-deterministically/)
- Peer topic: [Geocoding and Place-Name Resolution](/spatial-llm-architecture-core-concepts/geocoding-and-place-name-resolution/)
- Peer topic: [Evaluation and Benchmarking for Spatial LLMs](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/)
- Related topic: [Topology Rule Enforcement via LLMs](/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/)
