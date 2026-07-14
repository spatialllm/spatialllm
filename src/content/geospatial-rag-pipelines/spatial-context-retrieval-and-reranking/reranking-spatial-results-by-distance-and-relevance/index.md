# Reranking Spatial Results by Distance and Relevance

A spatial reranker decides which retrieved features actually reach the model by blending two incompatible signals into one comparable number: an embedding similarity that knows what a feature means and a geographic distance that knows where it is. This guide builds a complete fusion scorer that trades between the two with a tunable weight and a distance-decay term, sitting inside the [spatial context retrieval and reranking](https://www.spatialllm.org/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/) stage of a retrieval pipeline.

## When to Use This Approach

Reach for fusion scoring when semantic recall alone surfaces features that read correctly but sit in the wrong place, and when a hard distance cut-off would throw away a strongly relevant feature that happens to be slightly farther out. A weighted blend keeps both effects on a single dial instead of a brittle threshold.

The fusion score for a candidate is

$$s = \alpha \, \mathrm{sim} + (1-\alpha)\, e^{-d/\tau}$$

where $\mathrm{sim}$ is the normalized cosine similarity, $d$ is the geodesic-or-projected distance in metres, $\tau$ is the decay length that sets how quickly relevance falls off with distance, and $\alpha \in [0,1]$ balances meaning against location.

| Situation | Prefer this | Why |
|-----------|-------------|-----|
| Wrong-place-but-right-words results dominate | Fusion score with moderate `alpha` | Distance term demotes far candidates without discarding them |
| Hard operational radius (service area) | Fusion **after** a `ST_DWithin` pre-filter | Cut-off enforces the rule, fusion orders survivors |
| Distance is irrelevant (thematic lookup) | Pure similarity, `alpha = 1` | Adding a decay term only injects noise |
| Sparse, uneven feature density | Fusion with a larger `tau` | Wider decay avoids starving low-density regions |

## Implementation

The reranker below validates its inputs, computes the fused score, and falls back deterministically at every failure point: an unparseable geometry cannot crash the ranking, and equal scores never reorder between runs. It assumes candidates and query are already in one metric CRS, which the caller asserts upstream during [retrieval-augmented CRS resolution](https://www.spatialllm.org/geospatial-rag-pipelines/retrieval-augmented-crs-resolution/).

```python
import math
import logging
from dataclasses import dataclass, field
from typing import Optional

from shapely import wkt
from shapely.geometry.base import BaseGeometry
from shapely.errors import GEOSException

logger = logging.getLogger("spatial_rerank")


@dataclass
class SpatialHit:
    doc_id: str
    similarity: float          # cosine similarity, expected in [-1, 1]
    geom_wkt: str
    fused: float = field(default=0.0)


class RerankConfig:
    def __init__(self, alpha: float = 0.55, tau_m: float = 1200.0):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if tau_m <= 0.0:
            raise ValueError(f"tau_m must be positive metres, got {tau_m}")
        self.alpha = alpha
        self.tau_m = tau_m


def _norm_similarity(raw: float) -> float:
    """Map cosine [-1, 1] into [0, 1]; clamp anything out of range."""
    if raw is None or math.isnan(raw):
        return 0.0
    return max(0.0, min(1.0, (raw + 1.0) / 2.0))


def _safe_distance(query: BaseGeometry, candidate_wkt: str) -> float:
    """Return metric distance, or +inf if the candidate geometry is unusable."""
    try:
        cand = wkt.loads(candidate_wkt)
    except (GEOSException, ValueError, TypeError) as exc:
        logger.warning("dropping candidate with bad WKT: %s", exc)
        return float("inf")
    if cand.is_empty:
        return float("inf")
    try:
        return query.distance(cand)
    except GEOSException as exc:
        logger.warning("distance computation failed: %s", exc)
        return float("inf")


def rerank_by_distance_and_relevance(
    hits: list[SpatialHit],
    query_geom_wkt: str,
    cfg: Optional[RerankConfig] = None,
    top_n: int = 8,
) -> list[SpatialHit]:
    """Fuse similarity with distance decay; deterministic, never raises on data."""
    cfg = cfg or RerankConfig()

    if not hits:
        logger.info("no hits to rerank; returning empty list")
        return []

    # Fallback: if the query geometry is unusable, degrade to pure similarity
    # order with a stable secondary key rather than failing the request.
    try:
        query = wkt.loads(query_geom_wkt)
        if query.is_empty:
            raise ValueError("empty query geometry")
    except (GEOSException, ValueError, TypeError) as exc:
        logger.error("query geometry invalid (%s); semantic-only fallback", exc)
        ordered = sorted(hits, key=lambda h: (-_norm_similarity(h.similarity), h.doc_id))
        return ordered[:top_n]

    for h in hits:
        sim = _norm_similarity(h.similarity)
        d = _safe_distance(query, h.geom_wkt)
        proximity = math.exp(-d / cfg.tau_m) if math.isfinite(d) else 0.0
        h.fused = cfg.alpha * sim + (1.0 - cfg.alpha) * proximity

    # Primary: fused score descending. Secondary: doc_id ascending, so any tie
    # (including two unusable geometries at fused == alpha*sim) is reproducible.
    hits.sort(key=lambda h: (-h.fused, h.doc_id))
    return hits[:top_n]
```

## Validation & Testing

```python
def test_distance_breaks_semantic_tie():
    cfg = RerankConfig(alpha=0.5, tau_m=1000.0)
    near = SpatialHit("near", 0.8, "POINT (0 100)")
    far = SpatialHit("far", 0.8, "POINT (0 5000)")
    out = rerank_by_distance_and_relevance([far, near], "POINT (0 0)", cfg)
    assert [h.doc_id for h in out] == ["near", "far"]


def test_bad_geometry_never_crashes():
    hits = [SpatialHit("ok", 0.6, "POINT (0 10)"),
            SpatialHit("broken", 0.9, "NOT-WKT")]
    out = rerank_by_distance_and_relevance(hits, "POINT (0 0)")
    assert {h.doc_id for h in out} == {"ok", "broken"}  # broken sinks, not raised


def test_ordering_is_deterministic():
    hits = [SpatialHit(f"d{i}", 0.5, "POINT (0 100)") for i in range(5)]
    a = [h.doc_id for h in rerank_by_distance_and_relevance(list(hits), "POINT (0 0)")]
    b = [h.doc_id for h in rerank_by_distance_and_relevance(list(hits), "POINT (0 0)")]
    assert a == b
```

Wire these into CI so a change in `alpha`, `tau_m`, or the sort key is caught as a behavioural regression rather than a silent ranking drift.

## Gotchas & Edge Cases

**Unnormalized cosine sabotages the blend.** A raw similarity of `-0.9` is genuinely dissimilar, but if fed directly it makes the fused score negative and reorders unpredictably against a proximity term bounded in `[0,1]`. Always map similarity into `[0,1]` first, as `_norm_similarity` does.

**`tau` in the wrong units.** If distances arrive in degrees but `tau_m` is set for metres, `e^{-d/\tau}` saturates near 1 for every candidate and the distance signal vanishes. Assert the CRS is metric before ranking.

**Infinite distance underflow.** `math.exp(-inf / tau)` is `0.0`, which is correct, but only because the guard checks `math.isfinite(d)` first; calling `exp` on a `NaN` distance would poison the sort. Keep the finiteness check.

**Choosing `alpha` blind.** Sweeping `alpha` against a labelled set with a geographic-precision metric beats guessing; a value tuned on one corpus rarely transfers to another with different feature density.

## Related

- Up to the parent area: [Spatial Context Retrieval and Reranking](https://www.spatialllm.org/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/)
- [Spatial Vector Store Selection](https://www.spatialllm.org/geospatial-rag-pipelines/spatial-vector-store-selection/)
- [Retrieval-Augmented CRS Resolution](https://www.spatialllm.org/geospatial-rag-pipelines/retrieval-augmented-crs-resolution/)
