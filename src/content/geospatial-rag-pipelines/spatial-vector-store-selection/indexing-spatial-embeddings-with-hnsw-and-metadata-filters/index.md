---
title: Indexing Spatial Embeddings with HNSW and Metadata Filters
description: Build a graph index that keeps its recall when a bounding box and a date range remove most of the corpus, and verify the filter runs during traversal rather than after it.
slug: indexing-spatial-embeddings-with-hnsw-and-metadata-filters
type: howto
breadcrumb: HNSW with Metadata Filters
datePublished: 2025-04-04
dateModified: 2026-08-11
---

# Indexing Spatial Embeddings with HNSW and Metadata Filters

A graph index answers nearest-neighbour questions by walking edges toward the query. Add a filter that excludes most nodes and the walk can strand itself in a region of the graph where nothing is eligible, returning three results when it was asked for twenty. This guide builds the index and the queries so that does not happen, for the workload described in [spatial vector store selection](/geospatial-rag-pipelines/spatial-vector-store-selection/).

## When to Use This Approach

Use a graph index when the corpus is large enough that exact search is too slow, and the filters are selective. For small corpora, exact search under a filter is simpler, has perfect recall, and is often fast enough — a fact worth confirming before adding an approximate index and its parameters.

| Corpus after filtering | Index | Reason |
|------------------------|-------|--------|
| Under ~50 000 vectors | Exact search | Perfect recall, no parameters, fast enough |
| 50 000 to a few million | Graph index, filtered traversal | The normal case for this workload |
| Tens of millions | Graph index plus quantization | Memory becomes the binding constraint |
| Highly selective, small result | Filter first, then exact | The filter has already solved the problem |

The last row is the one people miss. If a bounding box reduces two million chunks to eight hundred, an exact scan of those eight hundred is microseconds of work and the graph index contributes nothing but risk.

<figure class="diagram">
<svg viewBox="16 38 748 218" role="img" aria-labelledby="hnsw-strand-t hnsw-strand-d" xmlns="http://www.w3.org/2000/svg"><title id="hnsw-strand-t">Why a filtered graph walk strands</title><desc id="hnsw-strand-d">A traversal that ignores the filter walks toward the query through ineligible nodes and exhausts its budget; a traversal that evaluates the filter during the walk steers toward eligible neighbourhoods and returns a full result set.</desc><rect x="16" y="38" width="748" height="218" fill="#ffffff"/><rect x="30" y="52" width="350" height="150" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="205" y="80" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">filter applied after the walk</text><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="205" y="110">walk visits the nearest nodes</text><text x="205" y="134">most fail the filter</text><text x="205" y="158">budget exhausted, 3 of 20 returned</text><text x="205" y="182">recall collapses as the filter tightens</text></g><rect x="400" y="52" width="350" height="150" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="575" y="80" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">filter evaluated during the walk</text><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="575" y="110">walk prefers eligible neighbours</text><text x="575" y="134">ineligible nodes still traversed</text><text x="575" y="158">budget spent usefully, 20 of 20</text><text x="575" y="182">recall holds as the filter tightens</text></g><text x="390" y="238" fill="#1f2937" font-size="13" text-anchor="middle">The difference is where the predicate is evaluated, not how good the index is</text></svg>
<figcaption><b>Ineligible nodes still have to be walked through.</b> That is the subtlety: a filtered traversal does not skip them, it declines to return them while still using their edges to reach eligible regions. A design that removes them from the graph entirely disconnects it.</figcaption>
</figure>

## Implementation

The index definition carries three decisions: the distance operator, the graph connectivity, and the build-time search width. The filter columns need their own indexes, because the planner has to be able to narrow before or during the graph walk.

```sql
-- Vector column plus the metadata the filters use.
ALTER TABLE spatial_chunks
    ADD COLUMN IF NOT EXISTS embedding vector(768);

-- Graph index over cosine distance. m controls connectivity; ef_construction
-- controls how hard the builder searches while wiring each node in.
CREATE INDEX IF NOT EXISTS spatial_chunks_embedding_idx
    ON spatial_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 24, ef_construction = 128);

-- The filters need indexes of their own or the pre-filter is a scan.
CREATE INDEX IF NOT EXISTS spatial_chunks_geom_idx   ON spatial_chunks USING gist (geom);
CREATE INDEX IF NOT EXISTS spatial_chunks_period_idx ON spatial_chunks (captured_at);

-- Query-time search width: higher means better recall and more work.
SET hnsw.ef_search = 120;

SELECT chunk_id
FROM   spatial_chunks
WHERE  geom && :bbox
  AND  ST_Intersects(geom, :region)
  AND  captured_at >= :since
ORDER  BY embedding <=> :qvec
LIMIT  20;
```

The build parameters are chosen higher than the defaults on purpose. Filtered search spends part of its budget walking through ineligible nodes, so a graph with more edges per node and a wider build search gives the traversal more routes into eligible territory. The cost is memory and build time, both of which are covered in [sizing HNSW parameters for spatial recall](/geospatial-rag-pipelines/spatial-vector-store-selection/sizing-hnsw-parameters-for-spatial-recall/).

The application layer's job is to raise the search width when the filter is tight, since a single global value cannot serve both loose and selective queries.

```python
import logging
import math

log = logging.getLogger("hnsw_index")

BASE_EF = 60
MAX_EF = 500


def ef_for_selectivity(selectivity: float, k: int = 20) -> int:
    """Widen the search when the filter removes most of the corpus.

    selectivity is the estimated share of the corpus passing the filter, in (0, 1].
    """
    if not (0.0 < selectivity <= 1.0) or math.isnan(selectivity):
        log.warning("implausible selectivity %r — using the base width", selectivity)
        return BASE_EF                              # deterministic fallback
    # Roughly: to see k eligible nodes, the walk must visit k / selectivity nodes.
    needed = int(k / max(selectivity, 1e-4))
    return max(BASE_EF, min(MAX_EF, needed))


def search(conn, qvec, region, since, k: int = 20, selectivity: float = 1.0):
    ef = ef_for_selectivity(selectivity, k)
    with conn.cursor() as cur:
        try:
            cur.execute("SET LOCAL hnsw.ef_search = %s", (ef,))
        except Exception as exc:                    # older server, or a store without the knob
            log.info("could not set search width (%s) — proceeding with the default", exc)
        cur.execute(FILTERED_QUERY, {"qvec": qvec, "region": region,
                                     "since": since, "k": k})
        rows = cur.fetchall()
    if len(rows) < k:
        log.info("filtered search returned %d of %d at ef=%d", len(rows), k, ef)
    return rows
```

The estimate of selectivity does not need to be accurate, only roughly right in order of magnitude. Deriving it from the region's area relative to the corpus extent, cached per region size band, is enough to distinguish a town-sized query from a national one, which is the distinction that matters.

<figure class="diagram">
<svg viewBox="0 0 760 250" role="img" aria-labelledby="hnsw-ef-t hnsw-ef-d" xmlns="http://www.w3.org/2000/svg"><title id="hnsw-ef-t">Search width against recall for three filter selectivities</title><desc id="hnsw-ef-d">Bars showing that a loose filter reaches full recall at a modest search width while a tight filter needs a much wider search to reach the same recall.</desc><rect x="0" y="0" width="760" height="250" fill="#ffffff"/><text x="380" y="32" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">Recall at twenty, by search width</text><g fill="#e4f5ec" stroke="#12805c" stroke-width="2"><rect x="90" y="70" width="50" height="104" rx="4"/><rect x="290" y="66" width="50" height="108" rx="4"/><rect x="490" y="64" width="50" height="110" rx="4"/></g><g fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"><rect x="146" y="112" width="50" height="62" rx="4"/><rect x="346" y="78" width="50" height="96" rx="4"/><rect x="546" y="68" width="50" height="106" rx="4"/></g><g fill="#fdeaee" stroke="#b3324f" stroke-width="2"><rect x="202" y="146" width="50" height="28" rx="4"/><rect x="402" y="116" width="50" height="58" rx="4"/><rect x="602" y="86" width="50" height="88" rx="4"/></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="171" y="196">ef 60</text><text x="371" y="196">ef 200</text><text x="571" y="196">ef 500</text></g><g fill="#1f2937" font-size="12"><text x="40" y="228">50% pass</text><text x="220" y="228">5% pass</text><text x="380" y="228">0.5% pass</text></g><text x="620" y="228" fill="#5b6471" font-size="12">taller is better</text></svg>
<figcaption><b>Selectivity and search width trade against each other.</b> A single global width is either wasteful for loose filters or inadequate for tight ones, which is why the width belongs in the query rather than in the configuration file.</figcaption>
</figure>

## Validation & Testing

```python
def test_recall_holds_at_high_selectivity(conn):
    truth = brute_force_topk(conn, QVEC, TIGHT_REGION, k=20)
    got = {r[0] for r in search(conn, QVEC, TIGHT_REGION, SINCE, k=20, selectivity=0.005)}
    assert len(truth & got) / len(truth) >= 0.9


def test_full_result_count_under_a_tight_filter(conn):
    rows = search(conn, QVEC, TIGHT_REGION, SINCE, k=20, selectivity=0.005)
    assert len(rows) == 20, f"stranded walk: only {len(rows)} results"


def test_search_width_scales_with_selectivity():
    assert ef_for_selectivity(1.0) == 60
    assert ef_for_selectivity(0.01) > ef_for_selectivity(0.5)
    assert ef_for_selectivity(0.0) == 60          # implausible input falls back, not crashes


def test_filter_indexes_are_used(conn):
    plan = explain(conn, FILTERED_QUERY)
    assert "Seq Scan on spatial_chunks" not in plan
```

The second test is the one that catches stranding directly, and it is easy to omit because a short result set looks like "the corpus has little about this place". Assert the count, not just the overlap.

## Gotchas & Edge Cases

**Filters on unindexed columns.** A metadata predicate with no index forces a scan of whatever the graph returns, which caps throughput and, worse, changes the plan under load. Index every column a filter touches, including the ones added later for a single feature.

**Insert-heavy workloads degrading connectivity.** Nodes added after the graph is built are wired into the graph as it exists, which over time produces a less well-connected structure than a full rebuild would. Track recall over time and rebuild on a schedule rather than waiting for complaints.

<figure class="diagram">
<svg viewBox="56 7 618 233" role="img" aria-labelledby="hnsw-deg-t hnsw-deg-d" xmlns="http://www.w3.org/2000/svg"><title id="hnsw-deg-t">Graph quality degrading across incremental inserts</title><desc id="hnsw-deg-d">Recall measured after a full build and after successive batches of incremental inserts, showing a slow decline that a periodic rebuild restores.</desc><rect x="56" y="7" width="618" height="233" fill="#ffffff"/><text x="380" y="32" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">Filtered recall after a full build, then after months of inserts</text><g fill="#e4f5ec" stroke="#12805c" stroke-width="2"><rect x="70" y="62" width="70" height="106" rx="4"/><rect x="590" y="64" width="70" height="104" rx="4"/></g><g fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"><rect x="200" y="76" width="70" height="92" rx="4"/><rect x="330" y="92" width="70" height="76" rx="4"/></g><g fill="#fdeaee" stroke="#b3324f" stroke-width="2"><rect x="460" y="114" width="70" height="54" rx="4"/></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="105" y="190">build</text><text x="235" y="190">month 1</text><text x="365" y="190">month 2</text><text x="495" y="190">month 3</text><text x="625" y="190">rebuild</text></g><text x="380" y="222" fill="#1f2937" font-size="13" text-anchor="middle">Nothing fails; recall simply erodes until a rebuild restores it</text></svg>
<figcaption><b>The slowest failure in the pipeline.</b> No alert fires, no query errors, and answers get slightly worse each month — which is why scheduled recall measurement, not a scheduled rebuild, is the control that matters.</figcaption>
</figure>

**Quantization applied before measuring.** Compressing vectors before establishing a recall baseline makes it impossible to attribute a later recall problem to compression or to the filter. Measure uncompressed first, then compress and measure again.

**A search width set globally and forgotten.** A width tuned during a demonstration with loose filters will underserve every selective query in production. Set it per query from selectivity, as above, and log when a search returns fewer results than requested.

**Deleted records left in the graph.** Soft deletes that remain as graph nodes consume search budget and can strand a walk exactly like a filter does. Ensure deletions are reflected in the index, and treat a rising deleted-node fraction as a reason to rebuild.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should the geometry filter or the vector index run first?</span></summary><p>Let the planner decide, but give it the information to decide well: an index on the geometry column, current statistics, and a query written with the bounding-box operator so the spatial index is usable. For very selective regions the planner should narrow spatially and scan; for loose ones it should walk the graph. A plan that never changes across those two cases is a sign that one of the indexes is not being considered.</p></details>

<details class="faq-item"><summary><span>How do I estimate selectivity without querying twice?</span></summary><p>From geometry, not from data. The ratio of the query region's area to the corpus extent's area, adjusted by a constant fitted once, is accurate enough to pick a search width. Caching that estimate by region-size band avoids recomputing it and keeps the choice stable, which matters because a search width that fluctuates between queries makes latency graphs unreadable.</p></details>

<details class="faq-item"><summary><span>Does raising connectivity always improve filtered recall?</span></summary><p>Up to a point, after which the extra edges cost memory and build time without helping, because the walk is already reaching eligible regions. The point varies with how spatially clustered the eligible set is: a filter that selects a compact region benefits more from connectivity than one selecting scattered records. Measure rather than assuming, and measure under your tightest realistic filter.</p></details>

<details class="faq-item"><summary><span>Should the index be built before or after the corpus is loaded?</span></summary><p>After, whenever the load is a bulk operation. Building the graph as records arrive wires each node into a partial graph, and the result is measurably worse connected than one built over the finished set — the same degradation that incremental inserts cause over time, compressed into the initial load. Load the vectors, then create the index, then measure. The build takes longer as one operation and produces a better graph than the same work spread across the load.</p></details>

<details class="faq-item"><summary><span>What is a reasonable recall target under filtering?</span></summary><p>Ninety percent at the tightest selectivity you actually serve, which is usually achievable without exotic tuning. Chasing ninety-nine costs disproportionately and rarely changes answers, because the reranking stage downstream reorders the top candidates anyway. What matters far more than the last few points is that recall does not silently fall over time, which is a monitoring question rather than a tuning one.</p></details>

A final note on ordering the work: establish the recall baseline before touching any parameter. Every knob in this guide trades recall against memory or latency, and without a baseline you cannot tell whether a change helped, hurt, or merely moved the cost somewhere you were not measuring. The baseline should be recorded with the corpus size and the filter selectivity it was measured at, because both change what the number means.

## Related

- Up to the parent topic: [Spatial Vector Store Selection](/geospatial-rag-pipelines/spatial-vector-store-selection/)
- [Sizing HNSW Parameters for Spatial Recall](/geospatial-rag-pipelines/spatial-vector-store-selection/sizing-hnsw-parameters-for-spatial-recall/)
- [pgvector vs Qdrant vs Milvus for Spatial Embeddings](/geospatial-rag-pipelines/spatial-vector-store-selection/pgvector-vs-qdrant-vs-milvus-for-spatial-embeddings/)
- Technique: [Filtering Retrieval by Bounding Box Before Vector Search](/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/filtering-retrieval-by-bounding-box-before-vector-search/)
