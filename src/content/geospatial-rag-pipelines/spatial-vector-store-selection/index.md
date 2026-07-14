# Spatial Vector Store Selection

Selecting a vector database for spatial embeddings decides whether geographically-scoped retrieval stays both correct and fast under load. This guide sits within [Geospatial RAG Pipelines](https://www.spatialllm.org/geospatial-rag-pipelines/) and addresses one specific failure: a store that ranks on embedding distance alone, then filters by region afterwards, quietly starves recall until scoped queries return half-empty result sets. The reader leaves able to probe a candidate engine's real filtering behaviour and configure a PostGIS-backed store where a bounding-box pre-filter runs before any expensive spatial predicate.

<figure class="diagram">
<svg viewBox="0 0 720 300" role="img" aria-labelledby="svs-t svs-d" xmlns="http://www.w3.org/2000/svg">
  <title id="svs-t">Pre-filter versus post-filter retrieval</title>
  <desc id="svs-d">A geographic pre-filter narrows candidates before approximate nearest-neighbour ranking and preserves recall, whereas a post-filter ranks first and discards out-of-region hits, collapsing the result set.</desc>
  <defs>
    <marker id="svs-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/>
    </marker>
  </defs>
  <g fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2">
    <rect x="20" y="45" width="150" height="70" rx="8"/>
    <rect x="285" y="45" width="150" height="70" rx="8"/>
    <rect x="550" y="45" width="150" height="70" rx="8"/>
    <rect x="20" y="165" width="150" height="70" rx="8"/>
    <rect x="285" y="165" width="150" height="70" rx="8"/>
  </g>
  <rect x="550" y="165" width="150" height="70" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/>
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#svs-arrow)">
    <line x1="171" y1="80" x2="283" y2="80"/>
    <line x1="436" y1="80" x2="548" y2="80"/>
    <line x1="171" y1="200" x2="283" y2="200"/>
    <line x1="436" y1="200" x2="548" y2="200"/>
  </g>
  <g fill="#1f2937" font-size="13" text-anchor="middle">
    <text x="95" y="75"><tspan x="95" dy="0">Query +</tspan><tspan x="95" dy="16">bbox</tspan></text>
    <text x="360" y="75"><tspan x="360" dy="0">Bbox</tspan><tspan x="360" dy="16">pre-filter</tspan></text>
    <text x="625" y="75"><tspan x="625" dy="0">ANN on</tspan><tspan x="625" dy="16">subset</tspan></text>
    <text x="95" y="195"><tspan x="95" dy="0">Query +</tspan><tspan x="95" dy="16">bbox</tspan></text>
    <text x="360" y="195"><tspan x="360" dy="0">ANN on</tspan><tspan x="360" dy="16">everything</tspan></text>
    <text x="625" y="195"><tspan x="625" dy="0">Post-filter</tspan><tspan x="625" dy="16">drops hits</tspan></text>
  </g>
  <text x="360" y="275" fill="#5b6471" font-size="12" text-anchor="middle">Top row preserves recall · bottom row collapses it</text>
</svg>
</figure>

## Foundational Principles

1. **Geography filters before vectors rank.** The store must accept a bounding box or region key and reduce the candidate set *before* the ANN scan. A post-filter that trims after ranking cannot recover neighbours the index never surfaced.
2. **Metadata and geometry share one query path.** Region, layer, and time filters must combine with the geographic envelope in a single indexed predicate. Splitting them across two round trips doubles latency and opens race windows.
3. **Distance semantics are explicit.** Cosine, dot-product, and L2 are not interchangeable for spatial embeddings; the store's configured metric must match the metric the embedding model was trained under, or ranking silently degrades.

## Step-by-Step Implementation Pipeline

### 1. Probe the candidate engine's true filtering behaviour

Vendor documentation often claims metadata filtering without stating whether it runs pre- or post-ANN. Rather than trust the label, measure it: issue a scoped and an unscoped query and confirm the scoped result count holds up. A capability probe turns marketing into a testable assertion.

```python
def probe_filter_mode(store, vector, region_key, k=50):
    """Classify a store as 'pre-filter' or 'post-filter' by observing recall.
    Returns a dict; on any store error returns mode='unknown' for safe fallback."""
    try:
        scoped = store.query(vector, k=k, filter={"region": region_key})
        unscoped = store.query(vector, k=k)
    except Exception as exc:  # narrow in real code to the store's error type
        return {"mode": "unknown", "error": str(exc)}
    scoped_in_region = [h for h in scoped if h.metadata.get("region") == region_key]
    unscoped_in_region = [h for h in unscoped if h.metadata.get("region") == region_key]
    # Pre-filter keeps k full of in-region hits; post-filter yields fewer.
    if len(scoped_in_region) >= len(unscoped_in_region) and scoped_in_region:
        return {"mode": "pre-filter", "recall_at_k": len(scoped_in_region) / k}
    return {"mode": "post-filter", "recall_at_k": len(scoped_in_region) / k}
```

The engine-by-engine outcome of running this probe is compared in [pgvector vs Qdrant vs Milvus for Spatial Embeddings](https://www.spatialllm.org/geospatial-rag-pipelines/spatial-vector-store-selection/pgvector-vs-qdrant-vs-milvus-for-spatial-embeddings/).

### 2. Configure a PostGIS + pgvector store with a bbox pre-filter

When the corpus already lives in PostGIS, colocating embeddings in a `vector` column lets a single query combine the spatial index and the ANN index. The critical detail is ordering: the `&&` bounding-box operator, which uses the GiST spatial index, must appear *before* the `ST_DWithin` predicate so the planner prunes on the cheap index first.

```sql
-- Retrieve the nearest embedded documents within a query envelope.
-- The && bbox pre-filter uses the GiST index BEFORE the exact ST_DWithin test.
SELECT d.doc_id,
       d.embedding <=> :query_vec AS cosine_distance
FROM   spatial_docs d
WHERE  d.geom && ST_MakeEnvelope(:minx, :miny, :maxx, :maxy, 4326)
  AND  ST_DWithin(d.geom::geography, :query_pt::geography, :radius_m)
ORDER  BY d.embedding <=> :query_vec
LIMIT  :k;
```

```python
import psycopg
from psycopg.rows import dict_row

def retrieve_scoped(conn_str, query_vec, envelope, query_pt, radius_m, k=20):
    """Run the bbox-pre-filtered ANN query with a deterministic empty fallback."""
    try:
        with psycopg.connect(conn_str, row_factory=dict_row) as conn:
            rows = conn.execute(SCOPED_SQL, {
                "query_vec": query_vec, "minx": envelope[0], "miny": envelope[1],
                "maxx": envelope[2], "maxy": envelope[3], "query_pt": query_pt,
                "radius_m": radius_m, "k": k,
            }).fetchall()
    except psycopg.Error as exc:
        # Never surface a partial context; return nothing so the caller degrades safely.
        return {"rows": [], "degraded": True, "reason": str(exc)}
    return {"rows": rows, "degraded": False}
```

The index parameters that make this query fast without sacrificing recall are tuned in [Indexing Spatial Embeddings with HNSW and Metadata Filters](https://www.spatialllm.org/geospatial-rag-pipelines/spatial-vector-store-selection/indexing-spatial-embeddings-with-hnsw-and-metadata-filters/).

### 3. Wrap the engine in a capability-aware abstraction

Because production systems outlive any single vendor choice, the retrieval call should target an interface that advertises which filters it can push down. When an engine cannot pre-filter, the abstraction routes to the deterministic PostGIS path rather than degrading silently — an application of [Fallback Routing for Geospatial Queries](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/).

```python
def route_retrieval(engine_caps, vector, envelope, pg_fallback):
    """Prefer a pre-filtering engine; otherwise use the PostGIS bbox path."""
    if engine_caps.get("geo_prefilter"):
        return engine_caps["client"].query(vector, bbox=envelope)
    # Deterministic fallback preserves geographic correctness over raw speed.
    return pg_fallback(vector, envelope)
```

## Failure Modes & Root Causes

**Post-filter recall collapse.** An engine ranks the global top-k, then drops out-of-region hits, leaving three results where twenty were requested. Root cause: filtering applied after ANN. Fix: require a pre-filter mode or fall back to PostGIS.

**Metric mismatch.** Embeddings trained for cosine similarity are queried under L2 in the store, reordering neighbours subtly enough to pass smoke tests but fail spatial recall benchmarks. Root cause: store default metric left unchanged.

**Envelope in the wrong CRS.** A bounding box in Web Mercator metres is passed to a column stored in WGS84 degrees; the `&&` operator matches nothing and the query returns empty. Root cause: no CRS assertion at the boundary, addressed by [normalizing mixed-CRS data before ingestion](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/normalizing-mixed-crs-data-before-llm-ingestion/).

## Production Validation Protocols

1. Run the filter-mode probe against every candidate engine in CI; fail the build if a store expected to pre-filter reports `post-filter`.
2. Assert the query envelope SRID equals the geometry column SRID before dispatch; raise on mismatch rather than returning empty.
3. Benchmark scoped recall@k against an exhaustive brute-force baseline nightly; alert when recall drops below the configured floor.
4. Log the `degraded` flag on every retrieval; a rising degraded rate signals an engine outage before users report missing context.

## Related

- Up to the parent area: [Geospatial RAG Pipelines](https://www.spatialllm.org/geospatial-rag-pipelines/)
- [pgvector vs Qdrant vs Milvus for Spatial Embeddings](https://www.spatialllm.org/geospatial-rag-pipelines/spatial-vector-store-selection/pgvector-vs-qdrant-vs-milvus-for-spatial-embeddings/)
- [Indexing Spatial Embeddings with HNSW and Metadata Filters](https://www.spatialllm.org/geospatial-rag-pipelines/spatial-vector-store-selection/indexing-spatial-embeddings-with-hnsw-and-metadata-filters/)
