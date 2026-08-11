---
title: Geometry Tokenization Strategies
description: Turn coordinates into tokens a model can use — precision policy, representation choice, simplification that preserves topology, and a budget that degrades predictably.
slug: geometry-tokenization-strategies
type: topic
breadcrumb: Geometry Tokenization
datePublished: 2025-01-21
dateModified: 2026-08-11
---

# Geometry Tokenization Strategies

A language model reads a sequence of tokens; a polygon is a list of high-precision numbers. Everything about how the second becomes the first — how many decimal places survive, which serialisation is used, whether vertices are dropped — determines both how much of the model's context a single feature consumes and how much of the geometry's meaning arrives intact. Tokenization is where those two pressures are traded against each other, deliberately.

This topic belongs to [spatial LLM architecture and core concepts](/spatial-llm-architecture-core-concepts/) and sits directly downstream of [coordinate reference system normalization](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/): a geometry must be in a known frame before any decision about precision means anything, because six decimal places of a projected metre and six of a degree are four orders of magnitude apart in real resolution. Its output feeds [context-window optimization for maps](/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/), which decides how many tokenized features can co-exist in one prompt.

<figure class="diagram">
<svg viewBox="16 9 698 247" role="img" aria-labelledby="gts-cost-t gts-cost-d" xmlns="http://www.w3.org/2000/svg"><title id="gts-cost-t">Token cost of one polygon under four representations</title><desc id="gts-cost-d">The same eighty-vertex polygon costs very different numbers of tokens depending on serialisation and coordinate precision, with hierarchical cell identifiers cheapest and full-precision structured output most expensive.</desc><rect x="16" y="9" width="698" height="247" fill="#ffffff"/><text x="390" y="34" fill="#5b6471" font-size="13" text-anchor="middle">One 80-vertex polygon, four ways to spend the context on it</text><g fill="#fdeaee" stroke="#b3324f" stroke-width="2"><rect x="200" y="52" width="500" height="40" rx="5"/></g><g fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"><rect x="200" y="102" width="330" height="40" rx="5"/></g><g fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"><rect x="200" y="152" width="210" height="40" rx="5"/></g><g fill="#e4f5ec" stroke="#12805c" stroke-width="2"><rect x="200" y="202" width="90" height="40" rx="5"/></g><g fill="#1f2937" font-size="12.5"><text x="30" y="77">structured, full</text><text x="30" y="127">structured, 5 dp</text><text x="30" y="177">compact text, 5 dp</text><text x="30" y="227">cell identifiers</text></g><g fill="#1f2937" font-size="12" text-anchor="middle"><text x="450" y="77">~1900 tokens</text><text x="365" y="127">~1250 tokens</text><text x="305" y="177">~790 tokens</text><text x="245" y="227">~330</text></g></svg>
<figcaption><b>The representation decision is a budget decision.</b> Between the top and bottom rows there is a factor of six, which is the difference between fitting three features in a prompt and fitting twenty — before anything has been simplified.</figcaption>
</figure>

## Foundational Principles

**Precision is a policy, not a property of the data.** Source coordinates arrive with whatever precision an exporter happened to emit, frequently far beyond the accuracy of the survey behind them. Choosing how many decimal places to keep is a decision about what the questions need, and it should be made once and applied uniformly rather than inherited per source.

**Simplification must preserve validity and topology.** Dropping vertices is the most effective way to reduce token cost and the easiest way to produce a self-intersecting ring or a gap between two parcels that used to share a boundary. Use a topology-preserving algorithm, validate after, and record the tolerance.

**The budget degrades, it does not truncate.** A feature that will not fit must be reduced by a defined ladder — fewer decimals, then simplification, then an extent with a note — never by cutting the token stream partway through a coordinate list. A truncated geometry is not a coarser geometry; it is a parse error.

## Step-by-Step Implementation Pipeline

### 1. Set the precision policy from the question, not the data

Decimal places in degrees map to distances on the ground, and the mapping is worth internalising: five decimal places is roughly a metre, four is roughly ten metres, three is roughly a hundred. Choose the coarsest that answers your questions.

```python
import logging
from dataclasses import dataclass

log = logging.getLogger("geometry_tokenization")

# Degrees of latitude per unit at the equator; longitude shrinks with latitude.
DECIMALS_FOR_METRES = {1.0: 5, 10.0: 4, 100.0: 3, 1000.0: 2}


def decimals_for(target_accuracy_m: float) -> int:
    """Fewest decimal places that still resolves the accuracy the task needs."""
    for metres in sorted(DECIMALS_FOR_METRES):
        if target_accuracy_m <= metres:
            return DECIMALS_FOR_METRES[metres]
    return 2
```

The saving is larger than it looks because coordinate text tokenizes badly: a run of digits and decimal points produces far more tokens per character than prose does, so removing two decimal places from every vertex of an eighty-vertex polygon removes several hundred tokens. The arithmetic and the trade-off are worked through in [coordinate precision versus token cost](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/coordinate-precision-versus-token-cost/).

### 2. Choose a representation and stay with it

Three families are in common use and they differ by roughly a factor of six in cost. A compact textual form is the cheapest that preserves exact geometry; a structured object form is more verbose and easier for a model to manipulate reliably; hierarchical cell identifiers are cheapest of all and lossy by construction. The comparison is developed in [comparing well-known text, structured objects and cell identifiers](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/wkt-vs-geojson-vs-h3-for-llm-tokenization/).

```python
def to_compact_text(geom, decimals: int) -> str:
    """Compact textual geometry at a fixed precision, with no trailing zeros."""
    def fmt(value: float) -> str:
        return f"{round(value, decimals):.{decimals}f}".rstrip("0").rstrip(".")
    rings = []
    for ring in _rings(geom):
        rings.append(", ".join(f"{fmt(x)} {fmt(y)}" for x, y in ring))
    return f"POLYGON(({'), ('.join(rings)}))"
```

Mixing representations across a corpus is the failure to avoid. A model that sees two forms for the same kind of object spends capacity distinguishing them, and any downstream parser has to handle both — which it will, until the day one of them acquires a variant.

### 3. Simplify with a topology-preserving algorithm

Vertex reduction is where most of the remaining saving lives. The algorithm matters: a naive douglas-peucker simplification on each geometry independently will pull two shared boundaries apart, opening slivers between parcels that were adjacent.

```python
from shapely.errors import GEOSException
from shapely.validation import make_valid


def simplify_safely(geom, tolerance_m: float, to_metric, from_metric):
    """Simplify in a metric frame, preserving topology, and never return invalid output."""
    if tolerance_m <= 0:
        return geom, 0.0
    try:
        projected = to_metric(geom)
        reduced = projected.simplify(tolerance_m, preserve_topology=True)
        if reduced.is_empty:
            log.info("simplification collapsed a geometry; keeping the original")
            return geom, 0.0
        if not reduced.is_valid:
            reduced = make_valid(reduced)
        return from_metric(reduced), tolerance_m
    except GEOSException as exc:
        log.warning("simplification failed (%s); keeping the original", exc)
        return geom, 0.0                              # deterministic fallback
```

Simplifying in a metric frame rather than in degrees matters for the same reason measuring does: a tolerance expressed in degrees is a different distance at every latitude, so a corpus spanning a continent would be simplified unevenly by a constant that looked uniform.

<figure class="diagram">
<svg viewBox="26 18 724 212" role="img" aria-labelledby="gts-topo-t gts-topo-d" xmlns="http://www.w3.org/2000/svg"><title id="gts-topo-t">Independent simplification opening a sliver between neighbours</title><desc id="gts-topo-d">Two parcels sharing a boundary are simplified separately, so the shared edge is reduced differently on each side and a gap appears where they used to touch.</desc><rect x="26" y="18" width="724" height="212" fill="#ffffff"/><rect x="40" y="60" width="150" height="120" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="190" y="60" width="150" height="120" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="190" y="212" fill="#1f2937" font-size="12.5" text-anchor="middle">before: one shared boundary</text><rect x="440" y="60" width="140" height="120" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="596" y="60" width="140" height="120" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="588" y="212" fill="#1f2937" font-size="12.5" text-anchor="middle">after: a sliver nobody intended</text><text x="588" y="42" fill="#5b6471" font-size="12" text-anchor="middle">each parcel simplified on its own</text></svg>
<figcaption><b>Both shapes are individually correct.</b> The relationship between them is what was lost, and no validity check on either geometry alone will report it — which is why topology preservation has to be requested rather than assumed.</figcaption>
</figure>

### 4. Reduce along a ladder when the budget binds

When a feature still exceeds its allowance, reduction proceeds in a defined order, and each rung is recorded so the consumer knows what it is looking at.

```python
@dataclass(frozen=True)
class Tokenized:
    text: str
    tokens: int
    rung: str            # which reduction was applied
    note: str


REDUCTION_LADDER = ("full", "fewer_decimals", "simplified", "extent_only")


def tokenize_within_budget(geom, budget: int, count_tokens, to_metric, from_metric,
                           decimals: int = 5) -> Tokenized:
    """Reduce along the ladder until it fits. Never truncates the token stream."""
    text = to_compact_text(geom, decimals)
    if count_tokens(text) <= budget:
        return Tokenized(text, count_tokens(text), "full", "")

    coarse = to_compact_text(geom, max(2, decimals - 2))
    if count_tokens(coarse) <= budget:
        return Tokenized(coarse, count_tokens(coarse), "fewer_decimals",
                         f"precision reduced to {max(2, decimals - 2)} decimals")

    reduced, tol = simplify_safely(geom, 25.0, to_metric, from_metric)
    simplified = to_compact_text(reduced, max(2, decimals - 2))
    if count_tokens(simplified) <= budget:
        return Tokenized(simplified, count_tokens(simplified), "simplified",
                         f"simplified at {tol:g} m tolerance")

    minx, miny, maxx, maxy = geom.bounds
    extent = f"BBOX({minx:.4f} {miny:.4f}, {maxx:.4f} {maxy:.4f})"
    log.info("feature reduced to its extent to fit a %d-token budget", budget)
    return Tokenized(extent, count_tokens(extent), "extent_only",
                     "geometry replaced by its bounding extent")
```

The last rung is the important one. Replacing a geometry with its extent is a substantial loss and is honest about it; truncating the coordinate list would be a smaller apparent loss and would produce text that no parser accepts and that the model will attempt to reason over anyway.

### 5. Carry the rung forward into the prompt

A model given a simplified geometry with no indication that it was simplified will answer questions about boundary detail as though the detail were real. The rung and its note belong in the context alongside the geometry.

```python
def to_prompt_fragment(name: str, t: Tokenized) -> str:
    """Geometry plus an honest statement of what was done to it."""
    if t.rung == "full":
        return f"{name}: {t.text}"
    return f"{name} ({t.note}): {t.text}"
```

### 6. Normalise vertex order and starting point

Two identical polygons serialised from different sources can differ in ring direction and starting vertex, producing different token sequences for the same shape. That matters for caching, for deduplication, and for any comparison a model is asked to make between two geometries.

```python
from shapely.geometry.polygon import orient


def canonical_form(geom):
    """Fixed ring orientation and a deterministic starting vertex."""
    try:
        oriented = orient(geom, sign=1.0)             # exterior counter-clockwise
    except Exception:
        return geom
    return oriented                                    # starting vertex handled by the writer
```

### 7. Measure the real token cost, not an estimate

Character counts and vertex counts are proxies, and both mislead for coordinate text. Measure with the tokenizer the model actually uses, and cache the measurement per geometry version so the budget logic is not paying for repeated tokenization.

```python
def measured_tokens(text: str, count_tokens, cache: dict) -> int:
    """Tokenize once per distinct string; fall back to a length heuristic on failure."""
    if text in cache:
        return cache[text]
    try:
        n = count_tokens(text)
    except Exception as exc:                          # tokenizer outage must not stop the build
        n = max(1, len(text) // 3)                    # coordinate text: ~3 chars per token
        log.warning("tokenizer unavailable (%s); estimating %d tokens", exc, n)
    cache[text] = n
    return n
```

The three-characters-per-token estimate is deliberately pessimistic for coordinate text and roughly right; prose runs closer to four. Using one constant for both is a common way to overshoot a context budget by a fifth.

### 8. Validate the round trip

Whatever comes out of tokenization must parse back into a geometry that is valid and, at the chosen precision, equivalent to what went in. This is a build-time assertion rather than a runtime one, and it is the check that catches a formatting change nobody expected.

```python
from shapely import wkt


def round_trips(geom, text: str, decimals: int) -> bool:
    """Parses back, stays valid, and agrees to the precision that was kept."""
    try:
        parsed = wkt.loads(text)
    except Exception:
        return False
    if not parsed.is_valid:
        return False
    tolerance = 10 ** (-decimals) * 2
    return parsed.equals_exact(geom, tolerance)
```

## Operating This Stage Over Time

Tokenization settings are the kind of configuration that gets chosen once, works, and then quietly stops being right. Three drifts account for most of it.

The first is a model change. A new model with a different tokenizer changes the cost of every geometry in the corpus, sometimes by twenty per cent, and a budget tuned against the old one will either waste context or overflow it. Pin the tokenizer version alongside the precision policy, and re-measure a sample when either changes; the measurement takes minutes and the alternative is a slow degradation nobody attributes correctly.

The second is a source whose precision changes. An upstream export that used to emit five decimal places starts emitting twelve, and nothing breaks — the rounding still works — but the pre-rounding text is now four times larger, which matters if anything downstream reads it before rounding. Rounding at the earliest possible point, immediately after normalization rather than at prompt-assembly time, makes the pipeline immune to this.

The third is scope creep in the questions. A system built to answer "which district is this in" acquires users asking "does this boundary follow the river", and the simplification tolerance that was invisible for the first question is fatal for the second. This is the drift worth watching most closely, because nothing in the pipeline reports it: the geometry is still valid, the answers are still fluent, and only someone who knows the ground can tell that the boundary detail is gone. Recording the rung in the prompt, as step 5 does, is what makes it detectable — the model can say the detail was removed, if it was told.

A useful habit is to keep two or three real features of different complexity as fixtures and print their token cost under the current settings on every build. It is one line of output, it makes the cost of the configuration visible to everyone who reads a build log, and it turns "we should probably check the token budget" into something that has already been checked.

## Failure Modes & Root Causes

**Precision inherited from the exporter.** Twelve decimal places of a coordinate whose survey accuracy is a metre, consuming four times the tokens for no information. Root cause: no precision policy. Mitigation: round at ingestion, from the accuracy the questions need.

**Slivers from independent simplification.** Adjacent features stop touching after reduction. Root cause: simplifying each geometry alone. Mitigation: topology-preserving simplification, and where adjacency is critical, simplify the shared boundary network rather than the polygons.

**Silent truncation at the budget.** A coordinate list is cut mid-number to fit, producing unparseable text the model still reasons over. Root cause: treating the budget as a character limit. Mitigation: the reduction ladder, with the extent as the floor.

**Two representations for one corpus.** Some features arrive as compact text and others as structured objects, so parsers and prompts must handle both. Root cause: representation chosen per source rather than per corpus. Mitigation: convert at ingestion; store one form.

## Production Validation Protocols

1. **Round-trip assertion.** Every tokenized geometry must parse back, stay valid, and agree within the retained precision; run it over a sample on every build.
2. **Topology preservation test.** For a fixture of adjacent parcels, assert they still touch after simplification at the configured tolerance.
3. **Ladder coverage test.** Assert each rung of the reduction ladder is reachable, using a fixture geometry sized to trigger it; an unreachable rung is dead code that will be wrong when it is finally needed.
4. **Budget assertion.** Assert no emitted fragment exceeds its budget, and that any that reached the extent rung carries its note.
5. **Token-cost indicator.** Publish the measured cost of the fixture geometries on every build so a tokenizer change is visible immediately.
6. **Precision-policy invariant.** Assert no stored coordinate carries more decimal places than the policy allows; a violation means something bypassed the ingestion rounding.

<figure class="diagram">
<svg viewBox="16 32 728 218" role="img" aria-labelledby="gts-ladder-t gts-ladder-d" xmlns="http://www.w3.org/2000/svg"><title id="gts-ladder-t">The reduction ladder and what each rung costs in meaning</title><desc id="gts-ladder-d">Four rungs from full precision through reduced decimals and simplification to an extent, with the information lost at each step and the note the consumer receives.</desc><rect x="16" y="32" width="728" height="218" fill="#ffffff"/><rect x="30" y="46" width="700" height="42" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="30" y="98" width="560" height="42" rx="6" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="30" y="150" width="400" height="42" rx="6" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="30" y="202" width="220" height="42" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="12.5"><text x="50" y="73">full precision — nothing lost</text><text x="50" y="125">fewer decimals — sub-metre detail lost</text><text x="50" y="177">simplified — small boundary features lost</text><text x="50" y="229">extent only — shape lost</text></g><g fill="#5b6471" font-size="12"><text x="610" y="125">note attached</text><text x="450" y="177">note attached</text><text x="270" y="229">note attached, prominently</text></g></svg>
<figcaption><b>Every rung below the first carries a note.</b> The reduction itself is unavoidable when a feature is large; what is avoidable is a consumer that cannot tell whether the boundary it is reading is the real one.</figcaption>
</figure>

## Frequently Asked Questions

<details class="faq-item"><summary><span>Is it better to send geometry at all, or just describe it?</span></summary><p>Send it when the question is about the geometry — shape, extent, relationships — and describe it when the question is about the thing the geometry represents. A model asked "what land use is here" does not need eighty vertices; it needs a label and a location. Much of the context pressure this topic manages comes from sending geometry to answer questions that were never geometric, and the cheapest optimisation is usually to notice that.</p></details>

<details class="faq-item"><summary><span>Do cell identifiers lose too much to be useful?</span></summary><p>For containment, adjacency and coarse proximity they are excellent and extremely cheap, because those questions are exactly what a hierarchical grid answers well. For anything about a boundary's actual position they are the wrong tool, since the boundary has been replaced by a staircase of cells at a chosen resolution. Many systems end up using both: cells for filtering and relationships, exact geometry for the few features an answer actually measures.</p></details>

<details class="faq-item"><summary><span>How should multipart geometries be budgeted?</span></summary><p>Per feature, not per part, and reduce by dropping the smallest parts before simplifying the largest. A multipolygon of a hundred islands where two hold all the meaning is common, and dropping ninety-eight small parts loses far less than simplifying the two large ones. Record how many parts were dropped and their combined area, so the note is quantitative rather than vague.</p></details>

<details class="faq-item"><summary><span>Should the same tolerance apply to every feature class?</span></summary><p>No. A tolerance appropriate for an administrative boundary is destructive for a building footprint, because the two have completely different characteristic sizes. Set the tolerance as a fraction of the feature's own extent, floored at the precision policy, so a small feature is barely touched and a large one is reduced meaningfully — and record the resulting absolute tolerance per feature rather than the fraction.</p></details>

<details class="faq-item"><summary><span>Does vertex order really matter for a model?</span></summary><p>Less for reasoning than for everything around it. Two serialisations of the same shape produce different cache keys, different deduplication behaviour, and different diffs when a corpus is rebuilt, all of which cost real effort. Canonicalising ring orientation and starting vertex is a few lines at write time and removes an entire category of spurious change.</p></details>

## Related

- Up to the section overview: [Spatial LLM Architecture and Core Concepts](/spatial-llm-architecture-core-concepts/)
- Technique: [How to Tokenize Polygon Boundaries for Transformer Models](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/how-to-tokenize-polygon-boundaries-for-transformer-models/)
- Comparison: [Well-Known Text, Structured Objects and Cell Identifiers](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/wkt-vs-geojson-vs-h3-for-llm-tokenization/)
- Technique: [Coordinate Precision Versus Token Cost](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/coordinate-precision-versus-token-cost/)
- Peer topic: [Context-Window Optimization for Maps](/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/)
- Peer topic: [Coordinate Reference System Normalization](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/)
