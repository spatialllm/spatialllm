---
title: Spatial Context Retrieval and Reranking
description: Retrieve candidates that are both semantically relevant and geographically right, then rerank them so proximity and topic pull in the same direction instead of fighting.
slug: spatial-context-retrieval-and-reranking
type: topic
breadcrumb: Retrieval and Reranking
datePublished: 2025-03-25
dateModified: 2026-08-11
---

# Spatial Context Retrieval and Reranking

Semantic similarity does not know where anything is. A vector search for "flood risk to the primary school" will happily return the best-written flood assessment in the corpus, from a catchment two hundred kilometres away, and rank it above the terse local report that actually answers the question. Spatial context retrieval and reranking is the discipline of making place a first-class ranking signal, so that the documents reaching the model are the ones that are both about the right subject and about the right ground.

This topic sits within [geospatial RAG pipelines](/geospatial-rag-pipelines/) and addresses the retrieval failure that survives every other fix: a pipeline whose chunks are perfectly formed, whose reference frames are correctly resolved, and whose top result is still about the wrong place. It depends on chunks that carry a defensible position — see [chunk-boundary strategies for spatial corpora](/geospatial-rag-pipelines/chunk-boundary-strategies-for-spatial-corpora/) — and on a store that can filter on that position, which is one of the selection criteria in [spatial vector store selection](/geospatial-rag-pipelines/spatial-vector-store-selection/).

<figure class="diagram">
<svg viewBox="62 26 657 264" role="img" aria-labelledby="scr-fight-t scr-fight-d" xmlns="http://www.w3.org/2000/svg"><title id="scr-fight-t">Semantic rank against spatial rank for one query</title><desc id="scr-fight-d">Four candidate documents plotted by how well they match the topic and how close they are to the query location. Only the candidate strong on both axes should reach the context window; the two single-axis winners are the ones a naive ranker promotes.</desc><rect x="62" y="26" width="657" height="264" fill="#ffffff"/><rect x="150" y="40" width="480" height="200" rx="8" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="390" y="40" width="240" height="100" rx="0" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="12" text-anchor="middle"><text x="510" y="74">keep — right topic, right place</text><text x="510" y="94">the only quadrant worth</text><text x="510" y="114">spending context on</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="270" y="86">eloquent, far away</text><text x="270" y="106">the ranker's favourite</text><text x="270" y="196">off topic and far</text><text x="510" y="186">nearby but off topic</text><text x="510" y="206">a map sheet, no prose</text></g><text x="390" y="272" fill="#1f2937" font-size="13" text-anchor="middle">semantic score increases to the right · distance to the query decreases upward</text><text x="76" y="140" fill="#5b6471" font-size="12">closer</text><text x="76" y="230" fill="#5b6471" font-size="12">farther</text></svg>
<figcaption><b>Two rankings, one budget.</b> A pure vector ranker fills the window from the left column; a pure distance ranker fills it from the top row. Both spend most of the budget on candidates that fail the other test, which is why fusion rather than sequencing is the right shape for this problem.</figcaption>
</figure>

## Foundational Principles

**Position is a filter before it is a score.** A candidate two hundred kilometres outside the area of interest is not a weak match, it is not a match, and paying to embed-compare it wastes both latency and recall. Run a bounding-box or radius filter in the store, then score what survives. The technique and its pitfalls are covered in [filtering retrieval by bounding box before vector search](/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/filtering-retrieval-by-bounding-box-before-vector-search/).

**Distance is a signal, not a verdict.** Once candidates are inside the region of interest, closer is usually better but not always: a regional policy document that governs the site is more useful than a neighbouring site's report. Reranking must be able to express "near and relevant beats far and relevant beats near and irrelevant" without collapsing into "nearest wins".

**Every score must be reproducible from stored data.** A rank that depends on a live distance computation against a moving reference point cannot be replayed, and a retrieval bug that cannot be replayed cannot be fixed. Store the geometry each candidate was scored against, and the parameters of the fusion, alongside the result.

## Step-by-Step Implementation Pipeline

### 1. Resolve the query's geometry before touching the index

A spatial query has a subject and a place, and the place is frequently implicit: "the primary school" resolves to a point only if the agent knows which school. Resolve it first, from the conversation, from an explicit parameter, or from a gazetteer lookup, and fail loudly when it cannot be resolved rather than searching the whole corpus by accident.

```python
from dataclasses import dataclass
from shapely.geometry.base import BaseGeometry
import logging

log = logging.getLogger("spatial_retrieval")

@dataclass(frozen=True)
class SpatialQuery:
    text: str
    focus: BaseGeometry | None      # resolved place, in EPSG:4326
    radius_m: float

def prepare_query(text: str, focus: BaseGeometry | None, radius_m: float = 5000.0) -> SpatialQuery:
    """Normalise a query; an unresolved place is explicit, never silently global."""
    if focus is None:
        log.info("no spatial focus for %r — falling back to text-only retrieval", text)
        return SpatialQuery(text, None, radius_m)
    if focus.is_empty:
        raise ValueError("focus geometry is empty; resolve the place or pass None")
    return SpatialQuery(text, focus, max(100.0, float(radius_m)))
```

The floor on the radius is not fussiness. A caller that computes a radius from a user's zoom level will eventually pass zero, and a zero-radius filter matches nothing, which surfaces as "the corpus has no documents about this place" — a claim that is both false and very hard to distinguish from the truth.

### 2. Filter in the store, not in the application

The candidate set must be narrowed by the index before vectors are compared. In PostGIS this means an index-aware predicate: the bounding-box operator first, so the spatial index is used, and the exact predicate second, so the answer is right.

```sql
-- Index-aware: && uses the GiST index, ST_DWithin refines what survives.
SELECT chunk_id, embedding, geom
FROM   spatial_chunks
WHERE  geom && ST_Expand(ST_GeomFromEWKB(:focus), :radius_deg)
  AND  ST_DWithin(geom::geography, ST_GeomFromEWKB(:focus)::geography, :radius_m)
ORDER  BY embedding <=> :query_vector
LIMIT  :k;
```

Written the other way round — `ST_DWithin` alone, or the exact predicate before the box — the planner may still find the index, but it may equally scan the table, and the difference on a corpus of any size is between forty milliseconds and forty seconds. The bounding-box pre-filter is cheap enough to be unconditional and is the single most reliable performance decision in this pipeline.

### 3. Fuse the two scores rather than sequencing them

Sequencing — take the top fifty by vector, then sort by distance — sounds reasonable and produces the failure in the opening figure, because anything the vector stage missed is unrecoverable. Fusion scores every surviving candidate on both axes and combines them.

```python
import math

def fuse(semantic: float, distance_m: float, radius_m: float,
         w_semantic: float = 0.65, half_life_m: float = 1500.0) -> float:
    """Combine a cosine similarity in [0,1] with a decaying proximity term."""
    if not 0.0 <= semantic <= 1.0:
        semantic = max(0.0, min(1.0, semantic))       # clamp rather than reject
    if distance_m < 0 or not math.isfinite(distance_m):
        proximity = 0.0                               # unknown position scores as far
    else:
        proximity = 0.5 ** (distance_m / half_life_m)  # 1.0 at the focus, 0.5 per half-life
    if distance_m > radius_m:
        return 0.0                                    # outside the region: not a candidate
    return round(w_semantic * semantic + (1.0 - w_semantic) * proximity, 6)
```

An exponential decay is the right default because it expresses the intuition that the first kilometre matters far more than the tenth. A linear decay makes a candidate at nine kilometres nearly as good as one at eight, which is true for a policy document and false for a site report — and when in doubt, the shape that punishes distance early is the one that fails safely.

### 4. Rerank with a model only where fusion is ambiguous

A cross-encoder rerank is expensive and improves ordering mostly among candidates whose fused scores are close. Reserve it for the ambiguous band and let the clear cases through untouched, which typically cuts reranking cost by an order of magnitude with no measurable quality loss.

```python
def rerank_band(candidates, cross_encoder, band: float = 0.08, top_n: int = 20):
    """Cross-encode only the candidates whose fused scores are within `band` of each other."""
    ranked = sorted(candidates, key=lambda c: c.fused, reverse=True)[:top_n]
    if len(ranked) < 2:
        return ranked
    head = ranked[0].fused
    ambiguous = [c for c in ranked if head - c.fused <= band]
    if len(ambiguous) < 2:
        return ranked
    try:
        scores = cross_encoder([(c.query_text, c.text) for c in ambiguous])
    except Exception as exc:                         # model outage must not empty the result
        log.warning("cross-encoder failed, keeping fused order: %s", exc)
        return ranked
    for cand, s in zip(ambiguous, scores):
        cand.fused = round(0.5 * cand.fused + 0.5 * float(s), 6)
    return sorted(ranked, key=lambda c: c.fused, reverse=True)
```

The exception handler is the important line. A reranker that fails closed — returning nothing when the model is unavailable — converts a degraded answer into no answer, which is almost never the right trade for a retrieval stage that already has a defensible ordering in hand.

### 5. Deduplicate on place as well as on text

Spatial corpora repeat themselves: the same site appears in an assessment, an appendix, and a revision, with near-identical prose and near-identical geometry. Text-level deduplication catches some of this; geometry-level deduplication catches the rest and is what stops a context window filling with four descriptions of one field.

```python
def dedupe_by_place(candidates, min_separation_m: float = 50.0):
    """Keep the best-scoring candidate per place; near-coincident geometries collapse."""
    kept = []
    for cand in sorted(candidates, key=lambda c: c.fused, reverse=True):
        if cand.geom is None:
            kept.append(cand)                        # nothing to compare on: keep it
            continue
        clash = any(
            k.geom is not None and cand.geom.distance(k.geom) * 111_000 < min_separation_m
            for k in kept
        )
        if not clash:
            kept.append(cand)
    return kept
```

The degree-to-metre factor there is a deliberate approximation and should be replaced by a projected distance in any pipeline that spans latitudes; it is written this way to make the unit conversion visible rather than buried. Getting it wrong by the cosine of the latitude is the kind of error that makes deduplication too aggressive near the poles and too lax at the equator.

### 6. Return the evidence, not just the ranking

The consumer of this pipeline is a model that will cite what it was given. Each returned candidate should carry its identifier, its position, both component scores, and the fused total, so the answer can be traced and so a reviewer can see whether an odd answer came from a semantic mismatch or a spatial one.

```python
def to_evidence(cand) -> dict:
    return {
        "chunk_id": cand.chunk_id,
        "semantic": cand.semantic,
        "distance_m": round(cand.distance_m, 1),
        "fused": cand.fused,
        "epsg": cand.epsg,
        "source": cand.source_uri,
    }
```

<figure class="diagram">
<svg viewBox="16 46 788 234" role="img" aria-labelledby="scr-stage-t scr-stage-d" xmlns="http://www.w3.org/2000/svg"><title id="scr-stage-t">How the candidate set narrows at each stage</title><desc id="scr-stage-d">A funnel from the full corpus through the spatial filter, vector comparison, fused ranking, ambiguity-band reranking and place deduplication, showing how many candidates survive each stage and what each stage costs.</desc><rect x="16" y="46" width="788" height="234" fill="#ffffff"/><rect x="30" y="60" width="140" height="150" rx="8" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="190" y="80" width="140" height="110" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="350" y="96" width="140" height="78" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="510" y="106" width="140" height="58" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="670" y="112" width="120" height="46" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="12.5" text-anchor="middle" font-weight="600"><text x="100" y="130">corpus</text><text x="260" y="130">in region</text><text x="420" y="130">top by vector</text><text x="580" y="132">fused rank</text><text x="730" y="132">deduped</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="100" y="152">2 000 000</text><text x="260" y="152">4 100</text><text x="420" y="152">200</text><text x="580" y="152">40</text><text x="730" y="152">8</text></g><g stroke="#5b6471" stroke-width="2" marker-end="url(#scr-stage-a)"><line x1="172" y1="135" x2="186" y2="135"/><line x1="332" y1="135" x2="346" y2="135"/><line x1="492" y1="135" x2="506" y2="135"/><line x1="652" y1="135" x2="666" y2="135"/></g><defs><marker id="scr-stage-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/></marker></defs><text x="410" y="242" fill="#1f2937" font-size="13" text-anchor="middle">The spatial filter does the heavy lifting — everything after it is cheap</text><text x="410" y="266" fill="#5b6471" font-size="12" text-anchor="middle">Illustrative counts for one regional query against a national corpus</text></svg>
<figcaption><b>Order the stages by what they cost.</b> The filter removes 99.8% of the corpus for the price of one index lookup. Every stage after it operates on a set small enough that the expensive work — cross-encoding, geometry distance — is affordable.</figcaption>
</figure>

### 7. Size the region of interest from the question, not from a constant

A single radius constant cannot serve every question. "Which trees are protected on this plot" wants tens of metres; "what is the flood risk to this street" wants a catchment; "what does the local plan say about this area" wants an administrative boundary that may be twenty kilometres across. A pipeline that hard-codes five kilometres answers the first question with noise and the third with nothing.

The practical approach is a small table of question classes, each with a radius rule, and a classifier — often a short model call, sometimes a keyword match — that assigns the class. Crucially, the classifier chooses a *rule*, not a number: rules that resolve to a named boundary produce far better regions than any radius, because they follow the shape of the thing being asked about.

| Question class | Region rule | Typical extent |
|----------------|-------------|----------------|
| Site condition | Parcel geometry, buffered | 30–100 m |
| Immediate impact | Fixed radius around the focus | 250–1000 m |
| Catchment or network | Upstream or connected geometry | 2–20 km |
| Policy and designation | Containing administrative unit | Variable |
| Comparative or statistical | Whole study area | Corpus-wide |

When the classifier is unsure, widen rather than narrow. A region that is too large costs latency and dilutes ranking slightly; a region that is too small removes the correct answer from consideration entirely, and no amount of clever reranking recovers a candidate that was never retrieved. That asymmetry should govern every default in this stage.

Record the chosen region alongside the results. When someone disputes an answer, the first question is almost always "what area did you look at", and a pipeline that cannot answer it will be assumed to have looked at the wrong one.

## Failure Modes & Root Causes

**The eloquent stranger.** A well-written document from the wrong region outranks the terse local one. Root cause: semantic score alone, with no spatial term. Mitigation: fusion in step 3, with the region filter in step 2 as the backstop.

**The empty region.** A query returns nothing because the filter radius was computed from a degenerate input, or because the chunks in that area lack geometry metadata and were excluded by the filter. Root cause: treating "no geometry" as "not here". Mitigation: keep geometry-less chunks in a separate lane that the text-only path can still reach, and floor the radius as in step 1.

**Proximity tyranny.** After adding a distance term, every answer becomes about the nearest feature regardless of subject. Root cause: a weight or decay tuned on a single query. Mitigation: tune against a labelled set spanning both site-specific and region-wide questions, and report both components so the imbalance is visible.

**The four-times-duplicated site.** The context window fills with revisions of one document. Root cause: deduplication on exact text only. Mitigation: place-level deduplication in step 5, plus a preference for the most recent revision when geometries coincide.

## Production Validation Protocols

1. **Filter-first assertion.** Assert in a query test that the executed plan uses the spatial index; a plan regression here is silent and catastrophic for latency.
2. **Component visibility.** Assert every returned candidate carries both component scores; a result set that reports only the fused total cannot be debugged.
3. **Fusion monotonicity.** For fixed semantic score, a nearer candidate must never rank below a farther one; property-test this rather than spot-checking it.
4. **Outside-radius exclusion.** Assert that a candidate beyond the radius scores exactly zero and is absent from results, so the filter and the score agree.
5. **Reranker failure drill.** Disable the cross-encoder in a test and assert results are still returned in fused order.
6. **Labelled retrieval set.** Maintain a small set of queries with known-correct documents and track recall at eight; this is the only gate that measures the thing users care about.

<figure class="diagram">
<svg viewBox="56 9 648 240" role="img" aria-labelledby="scr-decay-t scr-decay-d" xmlns="http://www.w3.org/2000/svg"><title id="scr-decay-t">Proximity weight against distance for two decay shapes</title><desc id="scr-decay-d">Bars comparing an exponential half-life decay with a linear decay at five distances. The exponential shape drops sharply over the first two kilometres and flattens; the linear shape treats near and middling distances as almost equivalent.</desc><rect x="56" y="9" width="648" height="240" fill="#ffffff"/><g fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"><rect x="70" y="60" width="46" height="120" rx="4"/><rect x="200" y="120" width="46" height="60" rx="4"/><rect x="330" y="150" width="46" height="30" rx="4"/><rect x="460" y="165" width="46" height="15" rx="4"/><rect x="590" y="172" width="46" height="8" rx="4"/></g><g fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"><rect x="124" y="60" width="46" height="120" rx="4"/><rect x="254" y="84" width="46" height="96" rx="4"/><rect x="384" y="108" width="46" height="72" rx="4"/><rect x="514" y="132" width="46" height="48" rx="4"/><rect x="644" y="156" width="46" height="24" rx="4"/></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="120" y="202">0 km</text><text x="250" y="202">1.5 km</text><text x="380" y="202">3 km</text><text x="510" y="202">4.5 km</text><text x="640" y="202">6 km</text></g><text x="370" y="34" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">Exponential half-life against linear decay</text><text x="370" y="232" fill="#5b6471" font-size="12" text-anchor="middle">Left bar of each pair is exponential; right bar is linear over the same radius</text></svg>
<figcaption><b>Pick the shape before tuning the weight.</b> The two curves agree at the focus and at nothing else. Most disagreements about "how much should distance matter" are really disagreements about decay shape, and they are easier to settle by looking at this picture than by adjusting a scalar.</figcaption>
</figure>

## Frequently Asked Questions

<details class="faq-item"><summary><span>What distance should a polygon candidate use — centroid or edge?</span></summary><p>Edge distance, almost always. A large catchment polygon whose centroid is forty kilometres away may still contain the query point, and centroid distance would rank it as irrelevant. Edge distance is zero when the focus falls inside the candidate, which is the answer you want. The exception is when candidates vary wildly in size and you specifically want to prefer local documents over regional ones; then a blend of edge distance and area is more honest than pretending centroid distance means something.</p></details>

<details class="faq-item"><summary><span>Should the fusion weight be learned or hand-set?</span></summary><p>Hand-set first, learned only once you have a labelled set large enough to trust. A learned weight fitted on a hundred queries mostly memorises the distribution of those queries' distances, and it moves whenever the corpus's spatial density changes. A hand-set weight with a published rationale is easier to defend, easier to override per query type, and usually within a few points of the learned optimum.</p></details>

<details class="faq-item"><summary><span>How do I handle queries with no place at all?</span></summary><p>Detect them and route them to a text-only path rather than inventing a focus. A question like "what does the regulation say about flood zones" has no location, and forcing one in — from the user's last query, or from their session — produces answers that are subtly scoped to somewhere the user did not ask about. Make the absence explicit in the query object, as step 1 does, so the downstream code branches on it deliberately.</p></details>

<details class="faq-item"><summary><span>How many candidates should actually reach the model?</span></summary><p>Fewer than the window allows. The window sets an upper bound; the useful number is set by how many genuinely distinct places and viewpoints the question needs, which is usually between four and ten. Filling the remaining space with rank-twenty candidates measurably degrades answers, because the model must now decide which of twenty documents to trust and the weakest ones are the ones most likely to contain a confidently phrased irrelevance. Cut the list where the fused score falls off, not where the token budget does.</p></details>

<details class="faq-item"><summary><span>Is reranking worth it if the fusion is already good?</span></summary><p>Only in the ambiguous band, which is why step 4 restricts it there. When fused scores are well separated, a cross-encoder almost always agrees with the existing order and you have paid for a confirmation. When they are clustered — which happens most often on broad regional queries with many similar documents — the cross-encoder is the only signal that can tell them apart, and there it earns its cost several times over.</p></details>

## Related

- Up to the section overview: [Geospatial RAG Pipelines](/geospatial-rag-pipelines/)
- Technique: [Reranking Spatial Results by Distance and Relevance](/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/reranking-spatial-results-by-distance-and-relevance/)
- Technique: [Filtering Retrieval by Bounding Box Before Vector Search](/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/filtering-retrieval-by-bounding-box-before-vector-search/)
- Peer topic: [Hybrid Spatial and Keyword Retrieval](/geospatial-rag-pipelines/hybrid-spatial-keyword-retrieval/)
- Peer topic: [Spatial Vector Store Selection](/geospatial-rag-pipelines/spatial-vector-store-selection/)
- Concept: [Spatial Embedding Models](/spatial-llm-architecture-core-concepts/spatial-embedding-models/)
