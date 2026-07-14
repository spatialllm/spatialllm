# Spatial Context Retrieval and Reranking

Retrieval-augmented spatial systems fail quietly when the nearest vectors in embedding space are the wrong features on the ground. This guide covers a two-stage design that first pulls candidate context by semantic similarity, then reorders it by geographic proximity and topology so the model reasons over evidence that is both relevant and geographically correct. It situates the technique inside the broader [geospatial RAG pipelines](https://www.spatialllm.org/geospatial-rag-pipelines/) area and gives you a reranking layer you can drop in front of any generator.

<figure class="diagram">
<svg viewBox="0 0 880 300" role="img" aria-labelledby="scrr-t scrr-d" xmlns="http://www.w3.org/2000/svg">
  <title id="scrr-t">Retrieve then rerank flow</title>
  <desc id="scrr-d">A query is embedded, matched against a vector store, spatially rescored, and passed to the generator with an audit gate.</desc>
  <defs>
    <marker id="scrr-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/>
    </marker>
  </defs>
  <g fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2">
    <rect x="15" y="55" width="150" height="80" rx="8"/>
    <rect x="188" y="55" width="150" height="80" rx="8"/>
    <rect x="361" y="55" width="150" height="80" rx="8"/>
    <rect x="534" y="55" width="150" height="80" rx="8"/>
    <rect x="707" y="55" width="150" height="80" rx="8"/>
  </g>
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#scrr-arrow)">
    <line x1="166" y1="95" x2="186" y2="95"/>
    <line x1="339" y1="95" x2="359" y2="95"/>
    <line x1="512" y1="95" x2="532" y2="95"/>
    <line x1="685" y1="95" x2="705" y2="95"/>
  </g>
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#scrr-arrow)">
    <line x1="90" y1="137" x2="90" y2="202"/>
    <line x1="263" y1="137" x2="263" y2="202"/>
    <line x1="436" y1="137" x2="436" y2="202"/>
    <line x1="609" y1="137" x2="609" y2="202"/>
    <line x1="782" y1="137" x2="782" y2="202"/>
  </g>
  <rect x="15" y="205" width="842" height="58" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/>
  <g fill="#1f2937" font-size="13" text-anchor="middle">
    <text x="90" y="90"><tspan x="90" dy="0">Embed</tspan><tspan x="90" dy="16">query</tspan></text>
    <text x="263" y="90"><tspan x="263" dy="0">ANN</tspan><tspan x="263" dy="16">recall</tspan></text>
    <text x="436" y="90"><tspan x="436" dy="0">Spatial</tspan><tspan x="436" dy="16">rescore</tspan></text>
    <text x="609" y="90"><tspan x="609" dy="0">Fuse and</tspan><tspan x="609" dy="16">cut</tspan></text>
    <text x="782" y="90"><tspan x="782" dy="0">Generate</tspan><tspan x="782" dy="16">answer</tspan></text>
  </g>
  <text x="436" y="240" fill="#1f2937" font-size="15" font-weight="600" text-anchor="middle">Recall wide, rank tight, cut on a fused score</text>
  <text x="436" y="287" fill="#5b6471" font-size="12" text-anchor="middle">Signals: cosine · distance decay · topology</text>
</svg>
</figure>

## Foundational Principles

Three constraints separate a spatial reranker from a generic text reranker, and all three must hold simultaneously.

1. **Recall and precision are different jobs, done by different signals.** The first stage optimizes for recall: cast a wide net with approximate nearest-neighbour search over [spatial embedding models](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/spatial-embedding-models/) so no correct feature is excluded before rescoring. The second stage optimizes for precision using an exact geometric predicate the embedding never captured. Collapsing both into one similarity number is the root cause of geographically plausible but wrong answers.

2. **Every geometric comparison happens in one metric CRS.** Cosine similarity is unit-free, but distance decay is not. Mixing a candidate stored in degrees with a query buffer computed in metres silently corrupts the ranking. Project both sides to a shared equal-distance CRS (a local UTM zone or an equidistant projection) before any distance is measured, and record which CRS was used so the fusion weight stays meaningful.

3. **The final cut is deterministic and auditable.** Given identical candidates and query, the ordered context handed to the generator must be byte-for-byte reproducible. Ties broken by dictionary iteration order or floating-point noise make evaluation impossible and regressions invisible.

## Step-by-Step Implementation Pipeline

### 1. Wide semantic recall

Pull more candidates than you intend to keep. Over-fetching is cheap in an approximate index and gives the spatial stage room to promote features the embedding ranked low. Validate the vector store response before trusting it: a truncated or empty recall must degrade to a documented fallback, never propagate an empty context that the generator will hallucinate around. Route persistent recall failures through [fallback routing for geospatial queries](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/).

```python
import logging
from dataclasses import dataclass

logger = logging.getLogger("spatial_recall")


@dataclass
class Candidate:
    doc_id: str
    similarity: float          # cosine, already in [-1, 1]
    geom_wkt: str              # candidate geometry, WKT in the shared CRS


class RecallError(RuntimeError):
    pass


def recall_candidates(query_vec, store, k_wide=200, k_floor=20):
    """Stage one: over-fetch by semantic similarity, with a validated fallback."""
    if query_vec is None or not len(query_vec):
        raise RecallError("empty query vector; refusing to query the index")
    try:
        hits = store.search(query_vec, top_k=k_wide)
    except Exception as exc:                       # index down, timeout, bad shape
        logger.error("ANN recall failed: %s", exc)
        hits = store.search_exact(query_vec, top_k=k_floor)  # deterministic brute-force
    if not hits:
        raise RecallError("recall returned no candidates after fallback")
    return [
        Candidate(h["id"], float(h["score"]), h["geom_wkt"])
        for h in hits if h.get("geom_wkt")
    ]
```

### 2. Spatial rescoring and fusion

Combine the semantic score with a proximity signal and, where topology matters, a containment or adjacency bonus. The blended score and its decay parameters are covered in depth in [reranking spatial results by distance and relevance](https://www.spatialllm.org/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/reranking-spatial-results-by-distance-and-relevance/); here the reranker only needs a stable, validated ordering.

```python
import math
from shapely import wkt
from shapely.errors import GEOSException


def rerank(candidates, query_geom_wkt, alpha=0.6, tau_m=1500.0, top_n=8):
    """Stage two: fuse cosine with distance decay; deterministic tie-break."""
    try:
        q = wkt.loads(query_geom_wkt)
    except (GEOSException, ValueError) as exc:
        logger.warning("unparseable query geom (%s); returning semantic order", exc)
        ranked = sorted(candidates, key=lambda c: (-c.similarity, c.doc_id))
        return ranked[:top_n]

    scored = []
    for c in candidates:
        try:
            d = q.distance(wkt.loads(c.geom_wkt))          # metres in shared CRS
        except (GEOSException, ValueError):
            d = float("inf")                               # unusable geom sinks, never crashes
        proximity = math.exp(-d / tau_m) if math.isfinite(d) else 0.0
        fused = alpha * c.similarity + (1.0 - alpha) * proximity
        scored.append((fused, c))

    # sort by fused score desc, then doc_id asc so ties are reproducible
    scored.sort(key=lambda t: (-t[0], t[1].doc_id))
    return [c for _, c in scored[:top_n]]
```

### 3. Index-aware spatial pre-filter in PostGIS

When candidates live in PostGIS rather than a vector store, do the coarse spatial narrowing in the database. The bounding-box operator must run before the exact distance predicate so the GiST index is used; reversing them forces a sequential scan over every row.

```sql
-- Coarse-to-fine: bbox pre-filter via '&&' BEFORE ST_DWithin.
-- :q is the query geometry (metric CRS), :radius is metres.
SELECT c.doc_id,
       ST_Distance(c.geom, :q) AS dist_m
FROM   spatial_context c
WHERE  c.geom && ST_Expand(:q, :radius)     -- index-aware bbox gate first
  AND  ST_DWithin(c.geom, :q, :radius)      -- exact predicate second
ORDER  BY dist_m ASC, c.doc_id ASC          -- deterministic tie-break
LIMIT  200;
```

## Failure Modes & Root Causes

**Semantic-only leakage.** A candidate on the far side of a river scores high on cosine similarity because its description matches the query vocabulary. Without a proximity signal it is served as fact. Root cause: skipping stage two entirely, or setting the fusion weight so semantics dominate.

**Silent CRS mismatch.** Distances computed between a degree-based candidate and a metre-based query produce numbers that are neither wrong enough to crash nor right enough to trust. Root cause: no shared-CRS assertion before `distance()`.

**Recall starvation.** Setting the wide-recall `k` too low means the correct feature never enters stage two, so no amount of clever reranking recovers it. Root cause: tuning the reranker while starving the recall stage.

**Non-deterministic ties.** Two candidates with equal fused scores flip order between runs, making an evaluation harness flap. Root cause: sorting on the score alone with no stable secondary key.

## Production Validation Protocols

1. Assert that query and every candidate share one CRS before any distance is measured; fail closed on mismatch.
2. Gate the wide-recall size: require `len(candidates) >= k_floor`, else trigger the documented fallback.
3. Pin the fusion weight and decay constant in config and snapshot them into every trace so a ranking change is attributable.
4. Add a CI test that feeds a fixed candidate set and asserts a byte-identical ordered `doc_id` list across two runs.
5. Track a geographic-precision KPI (share of served context within the truth radius) alongside the usual semantic hit rate, and alert when it drifts.

## Related

- Up to the parent area: [Geospatial RAG Pipelines](https://www.spatialllm.org/geospatial-rag-pipelines/)
- [Reranking Spatial Results by Distance and Relevance](https://www.spatialllm.org/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/reranking-spatial-results-by-distance-and-relevance/)
- [Spatial Embedding Models](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/spatial-embedding-models/)
- [Fallback Routing for Geospatial Queries](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/)
