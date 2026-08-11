---
title: Spatial Vector Store Selection
description: Choose a vector store for geometry-bearing corpora on the criteria that actually bite — metadata filtering, geometry predicates, recall under filter, and rebuild cost.
slug: spatial-vector-store-selection
type: topic
breadcrumb: Vector Store Selection
datePublished: 2025-04-02
dateModified: 2026-08-11
---

# Spatial Vector Store Selection

Choosing a vector store for a spatial corpus is not the benchmark exercise it looks like. Every candidate will report impressive queries per second on a synthetic dataset with no filters; almost none of those numbers survive contact with a workload where every query carries a bounding box, a date range, and a source filter, and where recall under those filters is the thing users actually feel. This topic sets out the criteria that decide the outcome and the tests that expose them.

It belongs to [geospatial RAG pipelines](/geospatial-rag-pipelines/) and follows directly from [spatial context retrieval and reranking](/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/): the ranking strategy described there assumes the store can filter on geometry before it compares vectors, and that assumption eliminates several otherwise reasonable options. The vectors themselves come from the models discussed in [spatial embedding models](/spatial-llm-architecture-core-concepts/spatial-embedding-models/), whose dimensionality drives the memory arithmetic below.

<figure class="diagram">
<svg viewBox="16 32 768 232" role="img" aria-labelledby="svs-crit-t svs-crit-d" xmlns="http://www.w3.org/2000/svg"><title id="svs-crit-t">The four criteria that decide a spatial vector store</title><desc id="svs-crit-d">Metadata filtering strategy, native geometry support, recall under filter and rebuild cost, each with the failure it produces when the store is weak on that axis.</desc><rect x="16" y="32" width="768" height="232" fill="#ffffff"/><rect x="30" y="46" width="360" height="94" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="410" y="46" width="360" height="94" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="30" y="156" width="360" height="94" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="410" y="156" width="360" height="94" rx="8" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><g fill="#1f2937" font-size="14" font-weight="600"><text x="52" y="78">Filtering strategy</text><text x="432" y="78">Native geometry</text><text x="52" y="188">Recall under filter</text><text x="432" y="188">Rebuild cost</text></g><g fill="#5b6471" font-size="12"><text x="52" y="102">pre-filter, post-filter or none</text><text x="52" y="124">post-filter empties selective queries</text><text x="432" y="102">real predicates or a bounding box</text><text x="432" y="124">without it, geometry lives elsewhere</text><text x="52" y="212">the number vendors do not publish</text><text x="52" y="234">falls hardest exactly when you filter</text><text x="432" y="212">hours or minutes to reindex</text><text x="432" y="234">decides how often you dare change</text></g></svg>
<figcaption><b>Four axes, not one.</b> Throughput is the number every vendor leads with and the one least likely to decide the outcome. Each of these four has produced a migration; raw speed rarely has.</figcaption>
</figure>

## Foundational Principles

**Filtered recall is the metric that matters.** A store that returns 98% recall unfiltered and 60% recall when a bounding box removes 99% of the corpus is worse for this workload than a slower store that holds 95% in both cases. Spatial queries are selective by nature, so the filtered case is the normal case, not the edge case.

**Geometry belongs in the same store as the vector, or the join will hurt.** Splitting vectors into a dedicated engine and geometry into a spatial database means every query becomes a two-system join with a candidate list passed between them. It works, and it is a permanent tax on latency and on operational complexity. Prefer a store that can hold both unless something else forces the split.

**Rebuild cost sets your iteration speed.** Embedding models change, chunking strategies change, and each change means reindexing. A store that takes six hours to rebuild a corpus makes those changes quarterly events; one that takes twenty minutes makes them routine. This is a product decision disguised as an infrastructure one.

## Step-by-Step Implementation Pipeline

### 1. Write down the workload before reading any benchmark

The workload determines which criteria matter, and it can be described in five numbers: corpus size, vector dimensionality, expected filter selectivity, query rate, and acceptable tail latency. Without them, every comparison is an argument about anecdotes.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Workload:
    n_chunks: int              # vectors in the index
    dim: int                   # embedding dimensionality
    filter_selectivity: float  # share of corpus surviving a typical filter
    qps: float
    p95_budget_ms: float

    def raw_vector_bytes(self) -> int:
        """Float32 vectors only — index overhead is on top of this."""
        return self.n_chunks * self.dim * 4

def sanity_check(w: Workload) -> list[str]:
    """Flag combinations that no store will satisfy, before shortlisting any."""
    problems = []
    if w.raw_vector_bytes() > 64 * 1024**3:
        problems.append("raw vectors exceed 64 GiB — plan for sharding or quantization")
    if w.filter_selectivity < 0.01 and w.p95_budget_ms < 50:
        problems.append("highly selective filters under 50 ms rules out post-filtering stores")
    if w.dim > 1536 and w.n_chunks > 50_000_000:
        problems.append("dimensionality and corpus size together will dominate cost")
    return problems
```

That last check is worth internalising: memory scales with the product of corpus size and dimensionality, and the two are usually chosen by different people at different times. A team that doubles its corpus and moves from 768 to 1536 dimensions has quadrupled its index memory without anyone making a decision that looked like quadrupling anything. Sizing that trade-off deliberately is the subject of [choosing vector dimensionality for spatial retrieval](/spatial-llm-architecture-core-concepts/spatial-embedding-models/choosing-vector-dimensionality-for-spatial-retrieval/).

### 2. Classify each candidate's filtering strategy

There are three, and the difference between them is the single largest determinant of whether a spatial workload behaves.

**Pre-filtering** applies the metadata predicate first and searches only the surviving subset. Recall is preserved under any selectivity; the cost is that the index must support searching a subset, which not every graph structure does well.

**Post-filtering** searches the whole index for the top candidates, then discards those failing the predicate. It is simple and fast when filters are loose, and it collapses when they are tight: search for 100 candidates in a corpus where 1% match your bounding box, and you can expect roughly one survivor.

**Filtered traversal** — the approach most modern graph indexes take — evaluates the predicate during graph traversal, keeping recall high without materialising the subset. It is the best fit for this workload and worth confirming a candidate actually implements, since the marketing language is similar in all three cases.

### 3. Test recall under a realistic filter, not an unfiltered one

The test that separates candidates is small and rarely run: build a modest index, compute exact nearest neighbours by brute force under a real filter, and compare.

```python
import numpy as np

def recall_at_k(store, vectors, queries, filter_fn, k: int = 10) -> float:
    """Compare the store's filtered results against brute-force truth."""
    hits = total = 0
    for q in queries:
        eligible = [i for i in range(len(vectors)) if filter_fn(i)]
        if not eligible:
            continue                              # no truth to compare against
        sims = np.array([float(vectors[i] @ q) for i in eligible])
        truth = {eligible[i] for i in np.argsort(-sims)[:k]}
        try:
            got = set(store.search(q, k=k, filter=filter_fn))
        except Exception as exc:                  # a store that errors under filter fails the test
            print(f"store raised under filter: {exc}")
            return 0.0
        hits += len(truth & got)
        total += len(truth)
    return hits / total if total else 0.0
```

Run this at three selectivities — loose, typical, and tight — and plot the three numbers. A store whose recall is flat across them is the one to shortlist. A store that reports 0.97, 0.88, 0.41 has told you exactly how it will behave the first time a user asks about a small town.

<figure class="diagram">
<svg viewBox="0 0 760 260" role="img" aria-labelledby="svs-rec-t svs-rec-d" xmlns="http://www.w3.org/2000/svg"><title id="svs-rec-t">Recall against filter selectivity for three filtering strategies</title><desc id="svs-rec-d">Bars showing recall at loose, typical and tight filter selectivity. Pre-filtering and filtered traversal hold recall steady; post-filtering degrades sharply as the filter becomes selective.</desc><rect x="0" y="0" width="760" height="260" fill="#ffffff"/><text x="380" y="34" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">Recall at ten, by how much of the corpus survives the filter</text><g fill="#e4f5ec" stroke="#12805c" stroke-width="2"><rect x="110" y="66" width="52" height="112" rx="4"/><rect x="330" y="70" width="52" height="108" rx="4"/><rect x="550" y="74" width="52" height="104" rx="4"/></g><g fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"><rect x="168" y="68" width="52" height="110" rx="4"/><rect x="388" y="76" width="52" height="102" rx="4"/><rect x="608" y="86" width="52" height="92" rx="4"/></g><g fill="#fdeaee" stroke="#b3324f" stroke-width="2"><rect x="226" y="70" width="52" height="108" rx="4"/><rect x="446" y="118" width="52" height="60" rx="4"/><rect x="666" y="158" width="52" height="20" rx="4"/></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="194" y="200">50% survive</text><text x="414" y="200">5% survive</text><text x="634" y="200">0.5% survive</text></g><g fill="#1f2937" font-size="12"><text x="40" y="232">pre-filter</text><text x="180" y="232">filtered traversal</text><text x="360" y="232">post-filter</text></g><text x="600" y="232" fill="#5b6471" font-size="12">taller is better</text></svg>
<figcaption><b>The number that decides it.</b> All three strategies look identical on an unfiltered benchmark. The right-hand group is the workload a spatial pipeline actually runs, and it is the only group where the choice is visible.</figcaption>
</figure>

### 4. Check what geometry support really means

"Supports geospatial" spans a wide range. At the weak end, a store lets you attach a latitude and longitude and filter by radius around a point — useful for point data, useless for polygons. In the middle, a bounding-box filter over stored extents, which is enough for the pre-filter stage described earlier. At the strong end, real predicates over real geometry, which lets containment and intersection run in the same query as the vector search.

```sql
-- Full-strength: geometry predicate and vector ordering in one statement.
SELECT id, source_uri
FROM   spatial_chunks
WHERE  geom && :bbox                              -- index-aware pre-filter
  AND  ST_Intersects(geom, :region)               -- exact predicate
  AND  captured_at >= :since
ORDER  BY embedding <=> :qvec
LIMIT  20;
```

If a candidate cannot express that query, the containment logic moves into application code, and with it the risk that the filter and the score disagree about which candidates exist. That is not disqualifying, but it should be a deliberate trade rather than a discovery made in month four. The concrete comparison across common options is laid out in [pgvector, Qdrant and Milvus for spatial embeddings](/geospatial-rag-pipelines/spatial-vector-store-selection/pgvector-vs-qdrant-vs-milvus-for-spatial-embeddings/).

### 5. Budget memory honestly, including the index

Raw vectors are the floor, not the total. Graph indexes add links per node; quantized indexes add codebooks; every store adds per-record metadata. A rule of thumb that has held up: budget 1.5 to 2 times the raw vector size for a graph index at default parameters, and confirm it by measurement rather than by trust.

```python
def index_footprint(n: int, dim: int, m: int = 16, bytes_per_link: int = 4) -> dict:
    """Rough memory model: vectors plus graph links, before metadata."""
    vectors = n * dim * 4
    links = n * m * 2 * bytes_per_link       # bidirectional neighbour lists
    return {
        "vectors_gib": round(vectors / 1024**3, 2),
        "links_gib": round(links / 1024**3, 2),
        "total_gib": round((vectors + links) / 1024**3, 2),
    }
```

The connectivity parameter appears twice in that model and three times in practice, since it also drives build time and recall. Choosing it is its own exercise: see [sizing HNSW parameters for spatial recall](/geospatial-rag-pipelines/spatial-vector-store-selection/sizing-hnsw-parameters-for-spatial-recall/).

### 6. Rehearse the rebuild before you depend on it

Reindexing is the operation that will hurt, and it is always attempted for the first time under pressure. Time a full rebuild on representative data, confirm the store can serve queries from the old index while the new one builds, and write down what happens if the rebuild fails halfway. A store that requires downtime to reindex constrains every future decision about embeddings, chunking, and metadata.

### 7. Decide where lexical search will live

Spatial corpora are unusually dependent on exact token matching. Place names, parcel references, scheme identifiers and grid codes are precisely the strings a dense embedding blurs, and they are precisely what users type. A store that also offers a lexical index lets the two run together; one that does not pushes the lexical half into a separate system and turns fusion into a distributed join.

```python
def plan_lexical(store_caps: dict) -> str:
    """Choose where keyword scoring happens, given what the store supports."""
    if store_caps.get("bm25") and store_caps.get("hybrid_fusion"):
        return "in-store hybrid"                  # one query, one ranking, one snapshot
    if store_caps.get("bm25"):
        return "in-store lexical, fuse in application"
    return "external lexical index — accept the join and version both sides"
```

The middle option is more common than either extreme and is perfectly workable, provided the fusion is written once and shared. What goes wrong is two teams fusing differently in two code paths, so the same query ranks differently depending on which entry point it arrived through. The mechanics of combining the two score families are covered in [hybrid spatial and keyword retrieval](/geospatial-rag-pipelines/hybrid-spatial-keyword-retrieval/).

### 8. Weigh the operational surface, not just the query surface

The store will be operated for years by people who did not choose it. Four questions predict most of that experience. Can it be backed up and restored without a full rebuild? Does it support rolling upgrades, or does every version bump mean a maintenance window? Are its metrics good enough to diagnose a slow query, or does observability stop at "requests per second"? And can a single engineer run it, or does it assume a platform team?

None of these appear in a benchmark, and all of them outlast the performance characteristics that dominated the selection meeting. A store that is 20% slower and can be restored from a snapshot in ten minutes is, for most teams, the better engineering choice than one that is faster and can only be recovered by reindexing the corpus from source.

There is one spatial-specific wrinkle worth adding to that list: how the store handles geometry updates. Chunk text rarely changes after ingestion, but geometry does — a parcel is resurveyed, a boundary is corrected, a coordinate error is fixed. If updating a geometry means deleting and reinserting the vector, then a routine data correction triggers index churn, and a bulk correction across a region can degrade graph connectivity enough to be visible in recall. Ask how in-place metadata updates are handled before you find out during a correction.

## Failure Modes & Root Causes

**Empty results on selective queries.** Users report that specific places return nothing while broad questions work. Root cause: post-filtering with a fixed candidate count. Mitigation: pre-filtering or filtered traversal, or an over-fetch multiplier tuned to selectivity — the second being a workaround, not a fix.

**Memory exhaustion after a model upgrade.** The index no longer fits after a move to a larger embedding. Root cause: capacity planned against raw vectors only, and dimensionality treated as a model decision rather than an infrastructure one. Mitigation: the footprint model in step 5, recomputed as part of any embedding change.

**Divergence between filter and score.** The application filters on geometry it holds in memory while the store scores over a different snapshot. Root cause: geometry and vectors in separate systems with independent update paths. Mitigation: single store where possible; a shared version stamp where not.

**Recall that decays silently.** Quality degrades over months with no incident. Root cause: incremental inserts degrading graph connectivity, with no periodic recall measurement. Mitigation: the filtered recall test from step 3, run on a schedule against production data, not only at selection time.

## Production Validation Protocols

1. **Filtered recall gate.** Measure recall at three selectivities on every index build; fail the build if the tight case drops more than five points below baseline.
2. **Plan assertion.** For the representative query, assert the spatial index is used; a plan change is the usual cause of a sudden latency cliff.
3. **Footprint check.** Compare measured index memory against the model in step 5 and alert on a gap greater than 25%.
4. **Rebuild drill.** Rebuild the full index on a schedule and record the wall-clock time as a tracked metric, so a slow drift is visible before it becomes a blocker.
5. **Consistency probe.** Assert that a chunk's geometry in the store matches the geometry in the source of record; drift here explains otherwise inexplicable retrieval results.
6. **Tail latency, not mean.** Gate on the 95th percentile under filter; mean latency hides exactly the queries users complain about.

<figure class="diagram">
<svg viewBox="16 9 742 223" role="img" aria-labelledby="svs-mem-t svs-mem-d" xmlns="http://www.w3.org/2000/svg"><title id="svs-mem-t">How index memory grows with corpus size and dimensionality</title><desc id="svs-mem-d">A grid of index footprints for two corpus sizes against three embedding dimensionalities, showing that a simultaneous increase in both quadruples the memory requirement.</desc><rect x="16" y="9" width="742" height="223" fill="#ffffff"/><text x="390" y="34" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">Approximate index footprint, vectors plus graph links</text><rect x="200" y="52" width="176" height="42" rx="6" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="384" y="52" width="176" height="42" rx="6" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="568" y="52" width="176" height="42" rx="6" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><g fill="#1f2937" font-size="12.5" text-anchor="middle"><text x="288" y="78">384 dimensions</text><text x="472" y="78">768 dimensions</text><text x="656" y="78">1536 dimensions</text></g><rect x="30" y="104" width="160" height="52" rx="6" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="30" y="166" width="160" height="52" rx="6" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><g fill="#1f2937" font-size="12.5" text-anchor="middle"><text x="110" y="136">5 M chunks</text><text x="110" y="198">20 M chunks</text></g><rect x="200" y="104" width="176" height="52" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="384" y="104" width="176" height="52" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="568" y="104" width="176" height="52" rx="6" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="200" y="166" width="176" height="52" rx="6" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="384" y="166" width="176" height="52" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="568" y="166" width="176" height="52" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="288" y="136">8 GiB</text><text x="472" y="136">15 GiB</text><text x="656" y="136">29 GiB</text><text x="288" y="198">31 GiB</text><text x="472" y="198">60 GiB</text><text x="656" y="198">115 GiB</text></g></svg>
<figcaption><b>Two decisions, one bill.</b> The corpus grows because the product succeeds; the dimensionality grows because a better model shipped. Neither change looks like a capacity decision, and together they move a workload from one machine to a cluster.</figcaption>
</figure>

## Frequently Asked Questions

<details class="faq-item"><summary><span>Is a general-purpose database with a vector extension good enough?</span></summary><p>For most spatial corpora, yes, and the reason is this workload specifically. Keeping geometry and vectors in one relational store removes the two-system join, gives you real geometry predicates, and lets transactions cover both. The trade is peak throughput at very large scale, which matters far less here than filtered recall does. Reach for a dedicated engine when the corpus outgrows a single machine or when query rate genuinely dominates, not by default.</p></details>

<details class="faq-item"><summary><span>How much does quantization cost in recall?</span></summary><p>Less than intuition suggests on unfiltered search and more than expected under tight filters, because quantization error and filter selectivity compound: fewer eligible candidates means each approximation error is likelier to displace a true neighbour. Measure it the same way as everything else here — at your real selectivity — and treat any figure quoted without a filter as unrelated to your workload.</p></details>

<details class="faq-item"><summary><span>Should geometry be indexed separately from the vector index?</span></summary><p>Usually yes, and that is not the same as storing it in a different system. A spatial index over the geometry column and a graph index over the vector column can coexist in one store, with the planner using the first to narrow and the second to order. What causes trouble is not two indexes; it is two systems, each with its own snapshot of the truth.</p></details>

<details class="faq-item"><summary><span>How should a shortlist be narrowed when two stores both pass?</span></summary><p>Run both against the same corpus for a fortnight behind a flag and compare the filtered recall curve, the 95th-percentile latency under filter, and the rebuild time. Those three numbers decide it more reliably than any feature matrix, and a fortnight of real traffic surfaces the operational differences — a noisy metric, an awkward upgrade, a confusing failure mode — that no evaluation checklist captures. If the numbers tie, choose the one your team can already operate.</p></details>

<details class="faq-item"><summary><span>What is a reasonable rebuild time to aim for?</span></summary><p>Short enough that you would rebuild to test a hypothesis. In practice that means under an hour for a corpus of a few million chunks, which is achievable on ordinary hardware with parallel index construction. Beyond that, rebuilds become scheduled events, experiments stop happening, and the pipeline ossifies around whatever embedding model was current when it was built.</p></details>

## Related

- Up to the section overview: [Geospatial RAG Pipelines](/geospatial-rag-pipelines/)
- Comparison: [pgvector vs Qdrant vs Milvus for Spatial Embeddings](/geospatial-rag-pipelines/spatial-vector-store-selection/pgvector-vs-qdrant-vs-milvus-for-spatial-embeddings/)
- Technique: [Indexing Spatial Embeddings with HNSW and Metadata Filters](/geospatial-rag-pipelines/spatial-vector-store-selection/indexing-spatial-embeddings-with-hnsw-and-metadata-filters/)
- Technique: [Sizing HNSW Parameters for Spatial Recall](/geospatial-rag-pipelines/spatial-vector-store-selection/sizing-hnsw-parameters-for-spatial-recall/)
- Peer topic: [Spatial Context Retrieval and Reranking](/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/)
- Concept: [Spatial Embedding Models](/spatial-llm-architecture-core-concepts/spatial-embedding-models/)
