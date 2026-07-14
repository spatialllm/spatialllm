# pgvector vs Qdrant vs Milvus for Spatial Embeddings

Choosing between pgvector, Qdrant, and Milvus for a spatial retrieval workload comes down to one question that generic benchmarks ignore: can the engine constrain results to a geographic envelope before it ranks by vector distance? This guide compares the three at the retrieval stage of [Spatial Vector Store Selection](https://www.spatialllm.org/geospatial-rag-pipelines/spatial-vector-store-selection/), where a wrong pick forces either a costly re-index later or a permanent recall penalty on region-scoped queries.

## When to Use This Approach

The decision hinges on where your geometry already lives, how tightly geographic filtering must couple with vector ranking, and how much operational surface your team can absorb. pgvector wins when the corpus is already in PostGIS and you want the spatial index and the ANN index in one transactional query. Qdrant suits payload-rich filtering with a lean single-binary deployment. Milvus targets billion-scale corpora where horizontal sharding matters more than transactional geometry.

| Capability | pgvector (PostGIS) | Qdrant | Milvus |
| --- | --- | --- | --- |
| Native bbox / geo filter | Yes — `&&` on GiST geometry | Geo-radius / geo-bounding-box payload filter | Via scalar payload bounds, no true geometry op |
| Pre-filter before ANN | Yes — planner prunes on GiST first | Yes — filterable HNSW | Partition-key / bounded, filter cost varies |
| Index types | HNSW, IVFFlat (pgvector) | HNSW (filterable) | HNSW, IVF, DiskANN, and more |
| Hybrid filter cost | Low when both indexes align | Low — designed for payload filtering | Moderate — depends on partition design |
| Ops overhead | Lowest if Postgres already run | Low — single service | Highest — distributed components |
| True geometry predicates | Full OGC `ST_*` suite | Points and bboxes only | Effectively point/bbox only |

Reach for pgvector when spatial correctness and transactional joins dominate; reach for Qdrant or Milvus when raw vector scale outgrows a single Postgres node and your geographic filter is a simple radius or box.

## Implementation

The probe below runs a scoped query against whichever engine is configured and verifies the geographic pre-filter actually held. It defaults to the pgvector path — the most spatially exact of the three — and every branch has a deterministic fallback so a misconfigured engine degrades to an empty, flagged result rather than leaking out-of-region context. It builds on the store-selection principles and the [spatial embedding models](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/spatial-embedding-models/) that produced the vectors.

```python
import psycopg
from psycopg.rows import dict_row

PGVECTOR_SCOPED_SQL = """
SELECT d.doc_id, d.embedding <=> %(qvec)s AS dist
FROM   spatial_docs d
WHERE  d.geom && ST_MakeEnvelope(%(minx)s, %(miny)s, %(maxx)s, %(maxy)s, 4326)
  AND  ST_Intersects(d.geom, ST_MakeEnvelope(%(minx)s, %(miny)s, %(maxx)s, %(maxy)s, 4326))
ORDER  BY d.embedding <=> %(qvec)s
LIMIT  %(k)s;
"""

def scoped_retrieve(engine, params, k=20):
    """Dispatch a bbox-scoped ANN query per engine with a safe empty fallback.
    engine: 'pgvector' | 'qdrant' | 'milvus'. Returns (rows, meta)."""
    minx, miny, maxx, maxy = params["bbox"]
    try:
        if engine == "pgvector":
            with psycopg.connect(params["dsn"], row_factory=dict_row) as conn:
                rows = conn.execute(PGVECTOR_SCOPED_SQL, {
                    "qvec": params["qvec"], "minx": minx, "miny": miny,
                    "maxx": maxx, "maxy": maxy, "k": k,
                }).fetchall()
                return rows, {"engine": engine, "prefiltered": True}
        elif engine == "qdrant":
            flt = {"must": [{"key": "loc", "geo_bounding_box": {
                "top_left": {"lon": minx, "lat": maxy},
                "bottom_right": {"lon": maxx, "lat": miny}}}]}
            hits = params["client"].search(
                collection_name=params["collection"],
                query_vector=params["qvec"], query_filter=flt, limit=k)
            return [h.__dict__ for h in hits], {"engine": engine, "prefiltered": True}
        elif engine == "milvus":
            expr = (f"lon >= {minx} && lon <= {maxx} && "
                    f"lat >= {miny} && lat <= {maxy}")
            hits = params["collection_obj"].search(
                data=[params["qvec"]], anns_field="embedding",
                param={"metric_type": "COSINE"}, limit=k, expr=expr)
            return list(hits), {"engine": engine, "prefiltered": True}
        else:
            raise ValueError(f"unknown engine {engine!r}")
    except Exception as exc:  # narrow to each client's error type in production
        # Deterministic fallback: no partial, geographically-unverified context.
        return [], {"engine": engine, "prefiltered": False, "error": str(exc)}
```

## Validation & Testing

- Assert scoped recall parity: for a fixed query, `len(scoped_retrieve(...)[0])` must equal a brute-force count of in-envelope neighbours within a tolerance, per engine.
- Assert CRS agreement: the envelope SRID must equal the stored geometry SRID (pgvector) or the payload coordinate frame (Qdrant/Milvus); fail closed on mismatch.
- Assert fallback safety: inject a bad DSN and confirm `meta["prefiltered"]` is `False` and rows are empty, never a global unscoped result set.

## Gotchas & Edge Cases

**Antimeridian-spanning envelopes.** A bounding box crossing longitude 180 becomes inverted; `ST_MakeEnvelope` and Qdrant's geo box both need it split into two boxes. Detect `minx > maxx` and issue two queries.

**Milvus has no geometry type.** Its bbox is a scalar range on `lon`/`lat` payload fields, so it cannot express polygon containment. Restrict Milvus to point-in-box use and route true polygon predicates to pgvector.

**Metric drift across engines.** pgvector's `<=>` defaults to cosine distance only when the operator class matches; Qdrant and Milvus must be created with `COSINE`. A silent L2 default reorders neighbours. Pin the metric at collection creation and assert it in a health check.

**Post-filter masquerade.** Some Milvus partition strategies apply the `expr` after ANN, collapsing recall. Verify with the filter-mode probe from the parent guide before trusting scoped counts.

## Related

- Up to the parent guide: [Spatial Vector Store Selection](https://www.spatialllm.org/geospatial-rag-pipelines/spatial-vector-store-selection/)
- [Indexing Spatial Embeddings with HNSW and Metadata Filters](https://www.spatialllm.org/geospatial-rag-pipelines/spatial-vector-store-selection/indexing-spatial-embeddings-with-hnsw-and-metadata-filters/)
- [Spatial Context Retrieval & Reranking](https://www.spatialllm.org/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/)
