# Indexing Spatial Embeddings with HNSW and Metadata Filters

Tuning an HNSW index for spatial embeddings means trading build time and memory against recall, then combining that approximate index with a geographic metadata pre-filter so region-scoped queries do not silently lose neighbours. This guide covers the indexing stage of [Spatial Vector Store Selection](https://www.spatialllm.org/geospatial-rag-pipelines/spatial-vector-store-selection/), where mis-set parameters produce a store that benchmarks well globally yet returns thin, unreliable results the moment a bounding-box filter is applied.

## When to Use This Approach

Reach for HNSW with a pre-filter when queries are almost always geographically scoped and recall must stay high inside the scope. The three build-and-search parameters trade off directly, and spatial workloads push them differently than generic text retrieval because the effective candidate pool shrinks after the geographic filter.

| Parameter | Controls | Spatial guidance |
| --- | --- | --- |
| `M` | Graph connectivity per node | Raise toward 24–48; higher branching offsets recall loss inside small filtered subsets |
| `ef_construction` | Build-time candidate breadth | 128–256; higher builds a denser graph so filtered traversal still finds neighbours |
| `ef_search` | Query-time candidate breadth | Scale up with filter selectivity; a tight bbox needs a larger `ef_search` to refill k |

If queries are rarely scoped, default parameters suffice. If a bounding box routinely eliminates 95% of the corpus, raise `ef_search` so the graph walk explores enough nodes to still surface k in-region neighbours.

## Implementation

Recall against an exhaustive search quantifies whether the tuned index is safe for production. For a query, if $R$ is the set of true nearest neighbours inside the geographic filter and $A$ is the set the index returned, recall is

$$\mathrm{recall@k}=\frac{|A \cap R|}{|R|}.$$

The routine below creates a filtered HNSW index in pgvector, runs a scoped query with the `&&` bbox pre-filter ahead of the ANN scan, and measures recall against a brute-force baseline so a bad parameter set fails loudly. The embeddings themselves come from the pipeline in [spatial embedding models](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/spatial-embedding-models/).

```python
import psycopg
from psycopg.rows import dict_row

def build_hnsw(conn, m=32, ef_construction=200):
    """Create a tuned HNSW index; idempotent and safe to re-run."""
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS spatial_docs_hnsw "
            "ON spatial_docs USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = %s, ef_construction = %s);",
            (m, ef_construction))
        conn.execute("CREATE INDEX IF NOT EXISTS spatial_docs_gist "
                     "ON spatial_docs USING gist (geom);")
    except psycopg.Error as exc:
        raise RuntimeError(f"index build failed: {exc}") from exc

# Bbox pre-filter (GiST) runs BEFORE the ANN order-by on the HNSW index.
SCOPED = """
SET LOCAL hnsw.ef_search = %(ef)s;
SELECT d.doc_id
FROM   spatial_docs d
WHERE  d.geom && ST_MakeEnvelope(%(minx)s,%(miny)s,%(maxx)s,%(maxy)s,4326)
  AND  ST_Intersects(d.geom, ST_MakeEnvelope(%(minx)s,%(miny)s,%(maxx)s,%(maxy)s,4326))
ORDER  BY d.embedding <=> %(qvec)s
LIMIT  %(k)s;
"""

BRUTE = """
SELECT d.doc_id
FROM   spatial_docs d
WHERE  d.geom && ST_MakeEnvelope(%(minx)s,%(miny)s,%(maxx)s,%(maxy)s,4326)
ORDER  BY d.embedding <=> %(qvec)s
LIMIT  %(k)s;
"""

def scoped_recall(dsn, qvec, bbox, k=20, ef=120):
    """Return recall@k of the filtered HNSW query vs brute force in-bbox.
    On DB error returns 0.0 so a broken index cannot pass a recall gate."""
    minx, miny, maxx, maxy = bbox
    args = {"qvec": qvec, "minx": minx, "miny": miny,
            "maxx": maxx, "maxy": maxy, "k": k, "ef": ef}
    try:
        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            approx = {r["doc_id"] for r in conn.execute(SCOPED, args).fetchall()}
            with conn.cursor() as cur:
                cur.execute("SET LOCAL enable_indexscan = off;")  # force exact
                exact = {r["doc_id"] for r in
                         cur.execute(BRUTE, args).fetchall()}
    except psycopg.Error:
        return 0.0
    if not exact:
        return 0.0
    return len(approx & exact) / len(exact)
```

## Validation & Testing

- Gate on recall: assert `scoped_recall(...) >= 0.95` for a representative set of tight bounding boxes; a failing gate blocks the index change from shipping.
- Sweep `ef_search`: assert recall is monotonically non-decreasing as `ef_search` rises, confirming the graph is traversable and not corrupted.
- Selectivity stress: assert recall holds when the bbox reduces the corpus to under 5% of rows — the regime where post-filter designs collapse.

## Gotchas & Edge Cases

**Post-filter recall collapse.** If the store applies the geographic filter *after* the ANN scan, a selective bbox leaves k mostly empty. The fix is the pre-filter ordering above; pgvector's planner uses the GiST index first only when the `&&` operator is present, so never drop it in favour of `ST_Intersects` alone.

**`ef_search` too low for tight scopes.** A global default of 40 works for whole-corpus queries but starves a 2% bbox. Scale `ef_search` inversely with filter selectivity at query time.

**Over-large `M` blows memory.** Pushing `M` past 64 inflates index size and build time with diminishing recall gains; profile before raising it.

**Metric mismatch.** The operator class `vector_cosine_ops` must match the embedding model's training metric. An L2 operator class on cosine-trained vectors reorders neighbours undetectably in smoke tests.

## Related

- Up to the parent guide: [Spatial Vector Store Selection](https://www.spatialllm.org/geospatial-rag-pipelines/spatial-vector-store-selection/)
- [pgvector vs Qdrant vs Milvus for Spatial Embeddings](https://www.spatialllm.org/geospatial-rag-pipelines/spatial-vector-store-selection/pgvector-vs-qdrant-vs-milvus-for-spatial-embeddings/)
- [Benchmarking Spatial Embedding Models for Vector GIS](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/spatial-embedding-models/benchmarking-spatial-embedding-models-for-vector-gis/)
