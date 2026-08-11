---
title: Chunk-Boundary Strategies for Spatial Corpora
description: Place retrieval splits where they preserve meaning — never inside a geometry, never between a feature and the reference frame that gives its coordinates meaning.
slug: chunk-boundary-strategies-for-spatial-corpora
type: topic
breadcrumb: Chunk-Boundary Strategies
datePublished: 2025-03-11
dateModified: 2026-08-11
---

# Chunk-Boundary Strategies for Spatial Corpora

Geometry-rich documents break naive chunkers. A fixed-size splitter that cuts a corpus every 800 tokens will sever a polygon mid-coordinate, orphan a feature from the header that gives its numbers meaning, or divorce a geometry from the attributes that describe it. Chunk-boundary strategy for spatial corpora is the discipline of placing splits where they preserve meaning — never inside a geometry, never between a feature and its context.

This topic belongs to [geospatial RAG pipelines](/geospatial-rag-pipelines/) and targets a specific retrieval failure: a chunk that contains half a WKT polygon, or a coordinate block with no reference frame, retrieves and embeds as noise, then poisons downstream reasoning with truncated geometry. The strategies here keep each feature atomic and frame-anchored, so retrieval returns spans a model can actually parse. They lean on [geometry tokenization strategies](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/) for how geometry becomes tokens and on [context-window optimization for maps](/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/) for how much of it a chunk can afford to carry. By the end of this page you should be able to write a splitter that provably never emits an unparseable geometry, and to prove it in continuous integration rather than hoping.

## Foundational Principles

**A geometry is indivisible.** A WKT `POLYGON(...)` or a GeoJSON `Feature` is the atomic unit; a split may fall between features but never inside one. A half-polygon is not a smaller polygon — it is malformed text that embeds as garbage and can crash a parser downstream. This is what separates spatial chunking from prose chunking, where a sentence cut in half still carries most of its meaning and still embeds into roughly the right region of vector space. Coordinates have no such graceful degradation: `POLYGON((-3.19 55.95, -3.18 55.9` is not 90% of a polygon, it is zero polygons and a syntax error.

**The reference frame travels with every chunk.** Coordinates without a frame are meaningless numbers. Each emitted chunk carries the document's projection identifier in its metadata so a retrieved fragment is self-describing, consistent with the anchoring enforced by [coordinate reference system normalization](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/). The practical consequence is that chunk metadata is not optional decoration — it is part of the payload, and a retrieval store that cannot hold structured metadata alongside the vector is the wrong store for this corpus.

**The token budget is a hard ceiling with a defined escape.** Chunks must fit the embedding model's window, but the budget can never justify severing a feature. When a single feature exceeds the budget, the strategy simplifies or isolates it — it does not cut it. Encoding that rule explicitly, as a branch in the code rather than as a comment in a design document, is the difference between a pipeline that degrades predictably and one that quietly corrupts a fraction of a percent of its corpus.

<figure class="diagram">
<svg viewBox="16 9 728 225" role="img" aria-labelledby="cbs-cut-t cbs-cut-d" xmlns="http://www.w3.org/2000/svg"><title id="cbs-cut-t">Legal and illegal cut points across a spatial document</title><desc id="cbs-cut-d">A document strip with alternating prose and geometry regions. Cuts are permitted only at paragraph breaks that fall outside every geometry span; a cut landing inside a coordinate list produces an unparseable fragment.</desc><rect x="16" y="9" width="728" height="225" fill="#ffffff"/><text x="380" y="34" fill="#5b6471" font-size="13" text-anchor="middle">One document, scanned once for no-cut zones before any boundary is chosen</text><rect x="30" y="60" width="700" height="86" rx="8" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="46" y="76" width="118" height="54" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="180" y="76" width="150" height="54" rx="5" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="346" y="76" width="118" height="54" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="480" y="76" width="150" height="54" rx="5" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="646" y="76" width="68" height="54" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="11" text-anchor="middle"><text x="105" y="107">prose</text><text x="255" y="107">geometry span</text><text x="405" y="107">prose</text><text x="555" y="107">geometry span</text><text x="680" y="107">prose</text></g><g fill="#12805c" font-size="13" text-anchor="middle" font-weight="600"><text x="172" y="176">cut ok</text><text x="472" y="176">cut ok</text><text x="638" y="176">cut ok</text></g><text x="255" y="176" fill="#b3324f" font-size="13" text-anchor="middle" font-weight="600">cut forbidden</text><text x="380" y="216" fill="#1f2937" font-size="13" text-anchor="middle">A boundary inside a coordinate list yields text no parser will accept</text></svg>
<figcaption><b>The no-cut map comes first.</b> Boundaries are chosen from the set of paragraph breaks that survive the geometry scan, so the budget can only ever pick among safe offsets — it never gets the chance to propose an unsafe one.</figcaption>
</figure>

## Step-by-Step Implementation Pipeline

### 1. Detect geometry spans before choosing any boundary

The first pass locates the byte ranges occupied by WKT and GeoJSON geometries so later steps treat them as no-cut zones. Detection must be conservative: a false negative severs a geometry, so ambiguous spans are treated as geometry. The token accounting here feeds the budget logic described in [context-window optimization for maps](/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/).

```python
import re
from dataclasses import dataclass

# Matches WKT primitives and GeoJSON geometry objects at a coarse span level.
_WKT = re.compile(r"\b(?:POINT|LINESTRING|POLYGON|MULTIPOLYGON|GEOMETRYCOLLECTION)\s*\([^;]*?\)",
                  re.IGNORECASE | re.DOTALL)
_GEOJSON = re.compile(r'\{[^{}]*"type"\s*:\s*"(?:Polygon|MultiPolygon|Feature)"[^{}]*\}', re.DOTALL)

@dataclass(frozen=True)
class Span:
    start: int
    end: int

def detect_geometry_spans(text: str) -> list[Span]:
    """Return sorted, merged no-cut spans; on regex failure, protect the whole doc."""
    try:
        spans = [Span(m.start(), m.end()) for m in _WKT.finditer(text)]
        spans += [Span(m.start(), m.end()) for m in _GEOJSON.finditer(text)]
    except Exception:                              # pathological input, catastrophic backtracking
        return [Span(0, len(text))]                # conservative: forbid all interior cuts
    spans.sort(key=lambda s: s.start)
    merged: list[Span] = []
    for s in spans:
        if merged and s.start <= merged[-1].end:
            merged[-1] = Span(merged[-1].start, max(merged[-1].end, s.end))
        else:
            merged.append(s)
    return merged
```

The fallback in the exception handler deserves a moment's attention, because it inverts the usual instinct. When span detection fails, the safe answer is not "cut anywhere" but "cut nowhere" — the document is emitted whole and flagged as oversized, which a human or a downstream simplifier can act on. An unparseable chunk, by contrast, is silent: it embeds, it retrieves, and nothing in the pipeline notices until an agent reasons over half a ring.

### 2. Place boundaries only at safe offsets

Candidate split points are the paragraph breaks that fall outside every geometry span. The splitter walks the document accumulating a token count and cuts at the last safe offset before the budget is reached. The mechanics of keeping each feature whole are developed fully in [splitting polygon-heavy documents without severing geometries](/geospatial-rag-pipelines/chunk-boundary-strategies-for-spatial-corpora/splitting-polygon-heavy-documents-without-severing-geometries/).

```python
def is_safe_offset(offset: int, spans: list[Span]) -> bool:
    """A cut is safe only if it lies strictly outside every geometry span."""
    return not any(s.start < offset < s.end for s in spans)

def safe_boundaries(text: str, spans: list[Span]) -> list[int]:
    candidates = [m.start() for m in re.finditer(r"\n\s*\n", text)]
    safe = [c for c in candidates if is_safe_offset(c, spans)]
    if not safe:                                   # no natural break survived
        safe = [len(text)]                         # emit the document whole
    return safe
```

Note that `is_safe_offset` uses strict inequalities. An offset exactly at a span's start or end is a boundary *between* features, which is precisely where a cut belongs. Getting that comparison wrong by one character is the classic way this function starts rejecting every legal boundary in a densely packed feature collection, then silently falls through to emitting the document whole.

### 3. Extend spans to cover the attributes that belong to a geometry

A geometry retrieved without its attribute row is coordinates with no meaning: the model can tell you where the polygon is, but not that it is a flood zone with a 1-in-100-year return period. Before boundaries are chosen, each geometry span is grown to absorb the adjacent attribute block, so the pair travels as a unit.

```python
_ATTR_BLOCK = re.compile(r"(?:^[ \t]*[\w .-]{1,40}:[^\n]*\n){1,12}", re.MULTILINE)

def extend_spans_to_attributes(text: str, spans: list[Span], window: int = 400) -> list[Span]:
    """Grow each geometry span over an attribute block sitting immediately after it."""
    grown: list[Span] = []
    for s in spans:
        tail = text[s.end: s.end + window]
        m = _ATTR_BLOCK.match(tail.lstrip("\n"))
        if not m:
            grown.append(s)                        # nothing adjacent — leave the span alone
            continue
        offset = len(tail) - len(tail.lstrip("\n"))
        grown.append(Span(s.start, s.end + offset + m.end()))
    return grown
```

### 4. Stamp metadata and guard the budget

Each chunk is emitted with the source frame attached and its token count verified. When a chunk containing a single indivisible feature still exceeds the budget, the guard routes it to the simplify-or-isolate fallback rather than forcing a cut. Carrying that metadata correctly is involved enough to have its own page: see [carrying frame and extent metadata into every chunk](/geospatial-rag-pipelines/chunk-boundary-strategies-for-spatial-corpora/carrying-crs-and-extent-metadata-into-every-chunk/).

```python
def emit_chunk(body: str, epsg: int, count_tokens, budget: int) -> dict:
    tokens = count_tokens(body)
    over_budget = tokens > budget
    if over_budget:                                # never cut — flag for simplify/isolate
        log_oversized(body, tokens, budget)
    return {
        "text": body.strip(),
        "epsg": epsg,                              # self-describing chunk
        "token_count": tokens,
        "needs_fallback": over_budget,             # triggers simplify/isolate
    }
```

<figure class="diagram">
<svg viewBox="0 0 780 280" role="img" aria-labelledby="cbs-cmp-t cbs-cmp-d" xmlns="http://www.w3.org/2000/svg"><title id="cbs-cmp-t">Fixed-size chunking compared with geometry-aware chunking</title><desc id="cbs-cmp-d">The upper row shows equal-width chunks whose third boundary falls inside a polygon, producing two unparseable fragments. The lower row shows uneven chunks whose boundaries respect feature edges, so every emitted chunk parses.</desc><rect x="0" y="0" width="780" height="280" fill="#ffffff"/><text x="30" y="58" fill="#b3324f" font-size="13" font-weight="600">Fixed size</text><rect x="200" y="26" width="136" height="52" rx="6" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="340" y="26" width="136" height="52" rx="6" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="480" y="26" width="136" height="52" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="620" y="26" width="136" height="52" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="11" text-anchor="middle"><text x="268" y="57">800 tokens</text><text x="408" y="57">800 tokens</text><text x="548" y="57">ring cut open</text><text x="688" y="57">orphan tail</text></g><text x="390" y="112" fill="#5b6471" font-size="12" text-anchor="middle">Two of four chunks now contain text that no geometry reader will accept</text><text x="30" y="188" fill="#12805c" font-size="13" font-weight="600">Geometry aware</text><rect x="200" y="156" width="180" height="52" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="384" y="156" width="110" height="52" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="498" y="156" width="258" height="52" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="11" text-anchor="middle"><text x="290" y="187">2 features</text><text x="439" y="187">1 feature</text><text x="627" y="187">1 large feature, isolated</text></g><text x="390" y="246" fill="#1f2937" font-size="13" text-anchor="middle">Chunk sizes vary; every chunk parses, and the oversized one is flagged not cut</text></svg>
<figcaption><b>Uneven chunks are the point.</b> Equal-sized chunks look tidy in a dashboard and are the direct cause of the failure above. Feature-atomic chunking trades size uniformity — which nothing downstream actually needs — for parseability, which everything downstream needs.</figcaption>
</figure>

### 5. Choose the unit of retrieval deliberately

Everything above assumes you know what a chunk is *for*. Three units are common in spatial corpora and they behave differently under retrieval, so the choice belongs in the design rather than in whatever the default splitter happened to do.

A **feature chunk** holds one geometry and its attributes. It gives the sharpest retrieval — a query about one parcel returns that parcel — and the weakest context, because the neighbouring features that give a place meaning are in other chunks. It suits catalogues and registers, where each record genuinely stands alone.

A **section chunk** holds a run of features under one heading: all the flood zones in one appendix, all the sites in one survey area. Retrieval is coarser, but the model sees relationships between features it would otherwise have to infer. It suits reports and assessments, where the argument spans several geometries.

A **document chunk** holds the whole file and is only appropriate when the file is small, when its geometries are inseparable, or when the corpus is small enough that retrieval is a formality. It is also the fallback the code above emits when no safe boundary exists, which is worth remembering: that fallback silently changes your retrieval unit, and if it fires often you have a corpus-shape problem rather than a chunker problem.

```python
from enum import Enum

class Unit(str, Enum):
    FEATURE = "feature"
    SECTION = "section"
    DOCUMENT = "document"

def choose_unit(feature_count: int, mean_feature_tokens: int, budget: int) -> Unit:
    """Pick a retrieval unit from corpus shape, with a defensible default."""
    if feature_count == 0:
        return Unit.DOCUMENT                       # nothing spatial to key on
    if mean_feature_tokens > budget // 2:
        return Unit.FEATURE                        # two features will not co-exist anyway
    if feature_count > 40:
        return Unit.SECTION                        # a register long enough to need grouping
    return Unit.SECTION
```

The rule of thumb encoded there is worth stating plainly: when the average feature already consumes half the window, grouping is a fiction and feature chunks are the only honest unit. When features are small, grouping them recovers the context that makes retrieved geometry interpretable. Whichever you choose, record it in the chunk metadata — a corpus that mixes units without saying so produces retrieval scores that cannot be compared across its own documents.


### 6. Make the chunker reproducible

A chunker is part of the index build, and an index build that produces different output from the same input is impossible to debug. Two sources of nondeterminism creep in easily. The first is a token counter that depends on a remote tokenizer service, so the same document chunks differently on a day the service is slow and truncates. The second is set iteration — deduplicating spans or sources through a `set` and then taking "the first" gives an order that is stable within one process and not across builds.

```python
def chunk_document(text: str, epsg: int, count_tokens, budget: int) -> list[dict]:
    """Deterministic end-to-end pass: same input, same chunks, every build."""
    spans = extend_spans_to_attributes(text, detect_geometry_spans(text))
    boundaries = sorted(set(safe_boundaries(text, spans)))      # sorted, not set-ordered
    chunks, start = [], 0
    for cut in boundaries:
        body = text[start:cut]
        if not body.strip():
            continue
        if count_tokens(body) > budget and start != 0:
            chunks.append(emit_chunk(body, epsg, count_tokens, budget))
            start = cut
            continue
        chunks.append(emit_chunk(body, epsg, count_tokens, budget))
        start = cut
    tail = text[start:]
    if tail.strip():
        chunks.append(emit_chunk(tail, epsg, count_tokens, budget))
    return chunks
```

Pin the tokenizer version alongside the chunker version and record both in the index metadata. When retrieval quality moves, the first question is always whether the corpus changed or the pipeline did, and only recorded versions can answer it. The same discipline pays off when a model upgrade brings a new tokenizer: a recorded version tells you immediately that every chunk boundary in the index was computed under the old one and the corpus needs rebuilding rather than patching.

## Failure Modes & Root Causes

**Severed geometry.** A boundary lands inside a coordinate list, emitting a truncated ring. Root cause: chunking on token count without a no-cut map. Mitigation: the span detection in step 1 gated by `is_safe_offset`. The tell in production is a parse-error rate that correlates with document length rather than document source — long documents cross more budget boundaries and so sever more often.

**Orphaned reference frame.** The header declaring the projection lands in chunk one; the geometries land in chunk five with no frame. Root cause: treating that header as ordinary prose. Mitigation: stamp the frame onto every chunk in step 4. This failure is especially nasty because the affected chunks parse cleanly — the numbers are syntactically perfect and geographically meaningless, so nothing raises until an answer is checked against the world.

**Attribute divorce.** A feature's geometry and its attribute table split into different chunks, so retrieval returns coordinates with no description. Root cause: boundaries that ignore feature grouping. Mitigation: the span extension in step 3, run before boundary selection rather than after.

**Budget-forced cut.** An oversized feature tempts a mid-geometry split to satisfy the window. Root cause: treating the budget as absolute. Mitigation: the simplify-or-isolate fallback, never a cut. A single national boundary with 40,000 vertices is the usual culprit, and it is better handled by generalizing the geometry to a tolerance the retrieval task can live with than by pretending it fits.

## Production Validation Protocols

1. **Parse every geometry per chunk.** In continuous integration, load each emitted chunk's geometries with a WKT and GeoJSON reader; any parse failure fails the build — it proves a severed feature escaped.
2. **Frame-presence assertion.** Assert every chunk carries a non-empty `epsg` field; a missing frame is a hard error, not a warning.
3. **Budget-with-escape check.** Assert `token_count <= budget` OR `needs_fallback is True`; no chunk may quietly exceed the window without flagging.
4. **Round-trip feature count.** Sum features across all chunks and assert it equals the source count — no feature dropped, none duplicated by overlap.
5. **Reconstruction test.** Concatenating chunks in order, minus any deliberate overlap, must reproduce the source geometry set exactly.
6. **Oversize-rate budget.** Track the share of chunks with `needs_fallback` set and alert when it drifts; a jump usually means an upstream source started shipping unsimplified geometry.

<figure class="diagram">
<svg viewBox="26 26 694 226" role="img" aria-labelledby="cbs-meta-t cbs-meta-d" xmlns="http://www.w3.org/2000/svg"><title id="cbs-meta-t">The payload of a single self-describing chunk</title><desc id="cbs-meta-d">A chunk record holds the text body plus four metadata fields — projection identifier, bounding extent, source feature identifiers and token count — each of which a retrieval filter or a validation gate depends on.</desc><rect x="26" y="26" width="694" height="226" fill="#ffffff"/><rect x="40" y="46" width="240" height="170" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="160" y="80" fill="#1f2937" font-size="14" font-weight="600" text-anchor="middle">chunk.text</text><text x="160" y="106" fill="#5b6471" font-size="12" text-anchor="middle">whole features only</text><text x="160" y="128" fill="#5b6471" font-size="12" text-anchor="middle">never a partial ring</text><text x="160" y="164" fill="#5b6471" font-size="12" text-anchor="middle">the part that gets</text><text x="160" y="186" fill="#5b6471" font-size="12" text-anchor="middle">embedded</text><rect x="360" y="40" width="320" height="42" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="360" y="92" width="320" height="42" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="360" y="144" width="320" height="42" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="360" y="196" width="320" height="42" rx="6" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="12.5"><text x="376" y="66">epsg — makes the numbers mean a place</text><text x="376" y="118">bbox — lets a filter run before the vector search</text><text x="376" y="170">feature_ids — links the answer back to the source</text><text x="376" y="222">token_count — proves the window was respected</text></g></svg>
<figcaption><b>Metadata is payload, not decoration.</b> Three of these four fields are read by something other than the model: the extent by the pre-filter, the identifiers by the citation layer, the token count by the build gate. Only the text is embedded.</figcaption>
</figure>

Two of these gates deserve to run on every build rather than nightly. The parse check is cheap — a reader over already-extracted spans, not a full geometry engine pass — and it is the only gate that catches the severing bug directly rather than by inference. The feature-count round trip is equally cheap and catches the subtler class of bug where a boundary is legal but a span was consumed twice, which shows up in retrieval as a phantom duplicate that quietly halves the effective diversity of a result set.

The remaining gates are better run over a sample on every build and over the whole corpus on a schedule. Reconstruction in particular is expensive on a large corpus and rarely fails alone: when it does fail, the parse gate has usually failed first and told you where.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should chunks overlap the way prose chunkers overlap?</span></summary><p>Sparingly, and never by a fraction of a feature. Prose overlap works because a duplicated sentence is harmless; duplicating half a polygon reintroduces exactly the fragment this strategy exists to prevent. If you want overlap for context continuity, overlap by whole features — repeat the last feature of the previous chunk — and deduplicate on feature identifier at retrieval time so the same geometry does not occupy two slots in one context window.</p></details>

<details class="faq-item"><summary><span>What if the corpus has no paragraph breaks at all?</span></summary><p>Then the candidate set is empty and `safe_boundaries` returns the whole document, which is the correct conservative answer but a poor retrieval unit. The fix is upstream: convert the source into a form with explicit feature boundaries — one feature per line, or a newline-delimited GeoJSON stream — before chunking. Inventing boundaries inside a structureless blob of coordinates is guesswork, and the guessing is what corrupts the index.</p></details>

<details class="faq-item"><summary><span>Does the extent in chunk metadata need to be exact?</span></summary><p>It needs to be a true superset of the geometries in the chunk, never a subset. A slightly loose bounding box costs a few extra candidates in the pre-filter stage; a tight box that clips one feature's corner silently removes that feature from every spatially filtered query. Compute it from the parsed geometries after chunking rather than carrying it forward from the source document, whose declared extent may be stale.</p></details>

<details class="faq-item"><summary><span>How does this interact with reranking?</span></summary><p>Well, provided chunks are feature-atomic. A reranker that scores by distance needs a defensible position for each candidate, and a chunk holding one or a few whole features has one; a chunk holding two features and half of a third does not. See <a href="/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/">spatial context retrieval and reranking</a> for how the position is used once the chunk carries it.</p></details>

## Related

- Up to the section overview: [Geospatial RAG Pipelines](/geospatial-rag-pipelines/)
- Concept: [Geometry Tokenization Strategies](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/)
- Concept: [Context-Window Optimization for Maps](/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/)
- Technique: [Splitting Polygon-Heavy Documents Without Severing Geometries](/geospatial-rag-pipelines/chunk-boundary-strategies-for-spatial-corpora/splitting-polygon-heavy-documents-without-severing-geometries/)
- Technique: [Carrying Frame and Extent Metadata Into Every Chunk](/geospatial-rag-pipelines/chunk-boundary-strategies-for-spatial-corpora/carrying-crs-and-extent-metadata-into-every-chunk/)
- Peer topic: [Spatial Metadata and Catalog Indexing](/geospatial-rag-pipelines/spatial-metadata-and-catalog-indexing/)
