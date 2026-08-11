---
title: pgvector vs Qdrant vs Milvus for Spatial Embeddings
description: A decision matrix for three common vector stores judged on what a geometry-bearing corpus needs — real spatial predicates, filtered recall, operational cost and scale.
slug: pgvector-vs-qdrant-vs-milvus-for-spatial-embeddings
type: howto
breadcrumb: Store Comparison
datePublished: 2025-04-03
dateModified: 2026-08-11
---

# pgvector vs Qdrant vs Milvus for Spatial Embeddings

Three stores come up in every spatial retrieval design discussion, and the comparison usually turns on throughput, which is the criterion least likely to decide the outcome. This guide compares them on the four axes a geometry-bearing corpus actually exercises, and says plainly which one each axis favours — the concrete follow-on to [spatial vector store selection](/geospatial-rag-pipelines/spatial-vector-store-selection/).

## When to Use This Approach

Use this comparison when you already know the workload — corpus size, dimensionality, filter selectivity, latency budget — and need to narrow to one candidate. If those numbers are not written down, the comparison will be decided by preference rather than evidence, and no matrix helps with that.

| Capability | Relational store with a vector extension | Dedicated engine, filter-first design | Distributed engine |
|------------|------------------------------------------|---------------------------------------|--------------------|
| Real geometry predicates | Native and complete | Bounding box and radius only | Bounding box and radius only |
| Filtered recall at high selectivity | Strong — the filter is a normal predicate | Strong — filtering is designed in | Strong, with tuning |
| Transactional updates with source data | Native | Separate system | Separate system |
| Horizontal scale beyond one machine | Limited | Moderate | Native |
| Operational burden | Lowest, if you already run the database | Moderate | Highest |

The pattern in that table is consistent: the relational option wins on everything that involves geometry and correctness, the distributed option wins on scale, and the dedicated engine sits between them with the best filtered-search behaviour of the three purpose-built designs.

<figure class="diagram">
<svg viewBox="66 32 688 218" role="img" aria-labelledby="pqm-fit-t pqm-fit-d" xmlns="http://www.w3.org/2000/svg"><title id="pqm-fit-t">Which store fits which corpus size and geometry demand</title><desc id="pqm-fit-d">A two-by-two of corpus scale against how much real geometry work the workload needs, showing where a relational store, a dedicated engine and a distributed engine each fit.</desc><rect x="66" y="32" width="688" height="218" fill="#ffffff"/><rect x="170" y="46" width="280" height="90" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="460" y="46" width="280" height="90" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="170" y="146" width="280" height="90" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="460" y="146" width="280" height="90" rx="8" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="310" y="80">relational + vector</text><text x="600" y="80">split: geometry apart</text><text x="310" y="180">relational, still</text><text x="600" y="180">distributed engine</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="310" y="106">predicates matter, corpus fits</text><text x="600" y="106">huge corpus, real predicates</text><text x="310" y="206">simple filters, corpus fits</text><text x="600" y="206">huge corpus, box filters only</text></g><text x="80" y="96" fill="#5b6471" font-size="12">geometry</text><text x="80" y="116" fill="#5b6471" font-size="12">heavy</text><text x="80" y="196" fill="#5b6471" font-size="12">geometry</text><text x="80" y="216" fill="#5b6471" font-size="12">light</text></svg>
<figcaption><b>Two questions decide three-quarters of cases.</b> Does the corpus fit one machine, and does the workload need real predicates rather than rectangles? The upper-right quadrant is the genuinely hard one and the only place a split architecture earns its complexity.</figcaption>
</figure>

## Implementation

The comparison is only meaningful against your own corpus, so the useful artefact is a harness that runs the same measurements against each candidate through one interface.

```python
import logging
import time
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

log = logging.getLogger("store_bakeoff")


class Store(Protocol):
    def upsert(self, ids: Sequence[str], vectors, metadata) -> None: ...
    def search(self, qvec, k: int, region=None) -> Sequence[str]: ...
    def rebuild(self) -> None: ...


@dataclass(frozen=True)
class Measurement:
    name: str
    recall_loose: float
    recall_tight: float
    p95_ms: float
    rebuild_s: float
    notes: tuple[str, ...]


def measure(store: Store, name: str, queries, truth_loose, truth_tight,
            loose_region, tight_region) -> Measurement:
    """Run one candidate through the four measurements that decide the choice."""
    notes: list[str] = []

    def recall(region, truth) -> float:
        hit = tot = 0
        for q, want in zip(queries, truth):
            try:
                got = set(store.search(q, k=10, region=region))
            except Exception as exc:               # a store that errors under filter fails here
                notes.append(f"search raised under filter: {exc}")
                return 0.0
            hit += len(want & got)
            tot += len(want)
        return round(hit / tot, 4) if tot else 0.0

    latencies = []
    for q in queries:
        start = time.perf_counter()
        try:
            store.search(q, k=10, region=tight_region)
        except Exception as exc:
            notes.append(f"latency probe failed: {exc}")
            break
        latencies.append((time.perf_counter() - start) * 1000)
    latencies.sort()
    p95 = latencies[int(0.95 * (len(latencies) - 1))] if latencies else float("inf")

    start = time.perf_counter()
    try:
        store.rebuild()
        rebuild_s = time.perf_counter() - start
    except Exception as exc:                       # some stores cannot rebuild online at all
        notes.append(f"rebuild unavailable: {exc}")
        rebuild_s = float("inf")

    return Measurement(name, recall(loose_region, truth_loose),
                       recall(tight_region, truth_tight), round(p95, 1),
                       round(rebuild_s, 1), tuple(notes))
```

Two design choices in that harness matter more than the numbers it produces. Errors are recorded as notes rather than raised, because "this store throws when a filter removes everything" is itself a finding and should appear in the comparison rather than aborting it. And rebuild time is measured, not asked about, because it is the figure most often quoted from documentation and least often true of a real corpus on real hardware.

For the relational option specifically, the query that makes it competitive is the one that puts both predicates and the ordering in a single statement:

```sql
SELECT c.chunk_id
FROM   spatial_chunks c
WHERE  c.geom && :bbox                    -- spatial index narrows first
  AND  ST_Intersects(c.geom, :region)     -- exact predicate, no approximation
  AND  c.captured_at >= :since
ORDER  BY c.embedding <=> :qvec           -- vector index orders what survives
LIMIT  20;
```

No other option in this comparison can express that without a round trip, and for most spatial corpora that single fact outweighs the throughput difference.

<figure class="diagram">
<svg viewBox="16 34 768 190" role="img" aria-labelledby="pqm-split-t pqm-split-d" xmlns="http://www.w3.org/2000/svg"><title id="pqm-split-t">The cost of splitting geometry from vectors</title><desc id="pqm-split-d">A single-store query runs one round trip; a split architecture fetches identifiers from the spatial database, passes them to the vector engine as a filter, and pays a second round trip plus a consistency risk.</desc><rect x="16" y="34" width="768" height="190" fill="#ffffff"/><rect x="30" y="48" width="200" height="64" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="290" y="48" width="200" height="64" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="130" y="78" fill="#1f2937" font-size="12.5" text-anchor="middle">one store</text><text x="130" y="98" fill="#5b6471" font-size="12" text-anchor="middle">filter and order together</text><text x="390" y="78" fill="#1f2937" font-size="12.5" text-anchor="middle">one round trip</text><text x="390" y="98" fill="#5b6471" font-size="12" text-anchor="middle">one snapshot of the truth</text><rect x="30" y="146" width="200" height="64" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="290" y="146" width="200" height="64" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="550" y="146" width="220" height="64" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="130" y="176" fill="#1f2937" font-size="12.5" text-anchor="middle">spatial database</text><text x="130" y="196" fill="#5b6471" font-size="12" text-anchor="middle">returns identifiers</text><text x="390" y="176" fill="#1f2937" font-size="12.5" text-anchor="middle">vector engine</text><text x="390" y="196" fill="#5b6471" font-size="12" text-anchor="middle">filters on that list</text><text x="660" y="176" fill="#1f2937" font-size="12.5" text-anchor="middle">two snapshots</text><text x="660" y="196" fill="#5b6471" font-size="12" text-anchor="middle">that can disagree</text><g stroke="#5b6471" stroke-width="2" marker-end="url(#pqm-split-a)"><line x1="232" y1="80" x2="286" y2="80"/><line x1="232" y1="178" x2="286" y2="178"/><line x1="492" y1="178" x2="546" y2="178"/></g><defs><marker id="pqm-split-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/></marker></defs></svg>
<figcaption><b>The second round trip is the smaller cost.</b> The larger one is on the right: two systems updated independently will eventually disagree about which chunks exist, and the resulting retrieval bugs are irreproducible by construction.</figcaption>
</figure>

## Validation & Testing

```python
def test_every_candidate_holds_recall_under_a_tight_filter(results):
    for m in results:
        assert m.recall_tight >= m.recall_loose - 0.05, (
            f"{m.name} loses recall when the filter bites: "
            f"{m.recall_loose} -> {m.recall_tight}")


def test_rebuild_completes_within_the_iteration_budget(results):
    for m in results:
        assert m.rebuild_s < 3600, f"{m.name} rebuild takes {m.rebuild_s}s — too slow to iterate"


def test_no_candidate_errors_under_an_empty_filter(results):
    for m in results:
        assert not any("raised under filter" in n for n in m.notes), m.notes
```

Run the harness against a real slice of the corpus — a few hundred thousand chunks is plenty — rather than against synthetic vectors. Synthetic data is uniformly distributed and real embeddings are emphatically not, and clustered data is exactly where approximate indexes lose recall.

## Gotchas & Edge Cases

**Benchmarks run without filters.** Published numbers almost always measure unfiltered search, which is not the workload. Treat any figure without a stated filter selectivity as unrelated to this decision.

**Recall measured against the index instead of the truth.** Comparing a store's results to its own exhaustive mode measures internal consistency, not recall. Compute truth by brute force over the filtered population.

<figure class="diagram">
<svg viewBox="16 32 748 214" role="img" aria-labelledby="pqm-bake-t pqm-bake-d" xmlns="http://www.w3.org/2000/svg"><title id="pqm-bake-t">The four measurements a bake-off should produce</title><desc id="pqm-bake-d">Loose recall, tight recall, tail latency under filter and rebuild wall clock, with a note on what a poor result in each one means for the workload.</desc><rect x="16" y="32" width="748" height="214" fill="#ffffff"/><rect x="30" y="46" width="360" height="86" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="410" y="46" width="340" height="86" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="30" y="146" width="360" height="86" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="410" y="146" width="340" height="86" rx="8" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><g fill="#1f2937" font-size="13" font-weight="600"><text x="52" y="76">recall, loose filter</text><text x="432" y="76">recall, tight filter</text><text x="52" y="176">tail latency under filter</text><text x="432" y="176">rebuild wall clock</text></g><g fill="#5b6471" font-size="12"><text x="52" y="102">a floor every candidate clears</text><text x="52" y="120">rarely discriminates</text><text x="432" y="102">the number that decides it</text><text x="432" y="120">rarely published by anyone</text><text x="52" y="202">gate on the 95th percentile</text><text x="52" y="220">the mean hides the complaints</text><text x="432" y="202">sets how often you dare change</text><text x="432" y="220">measure it, never ask</text></g></svg>
<figcaption><b>Two of these four are usually skipped.</b> Tight-filter recall and measured rebuild time are the two that change the decision, and both take an afternoon to obtain on a real corpus slice.</figcaption>
</figure>

**A rebuild that cannot run online.** Some configurations require the index to be offline while rebuilding. That is survivable for a corpus that changes quarterly and disqualifying for one that changes daily, so establish it before, not after.

**Dimensionality chosen elsewhere.** The embedding decision drives the memory bill and is often made by a different team on different grounds. Fix the dimensionality before running the comparison, or the winner will change when the model does.

**Geometry precision quietly reduced.** A store that only holds a bounding box for each record turns every containment question into an approximation. That may be acceptable — many workloads only ever ask for rectangles — but it should be an accepted trade, not a discovery.

Run the harness twice, a week apart, on the same data. Vector stores are stateful in ways that benchmarks rarely capture: caches warm, background compaction runs, and a store that looked fastest on a freshly built index may look different once it has absorbed a week of inserts. If the two runs disagree materially, that instability is itself a finding worth more than the absolute numbers.

Record the hardware, the dataset slice and the parameter settings alongside the results. A comparison whose conditions are not written down cannot be repeated when someone asks, six months later, whether the conclusion still holds — and by then the corpus has grown, the embedding has changed, and repeating it is exactly what you want to do.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Is the relational option really competitive at scale?</span></summary><p>Up to the point where the index no longer fits comfortably in memory on one machine, yes, and that point is further away than most teams assume — tens of millions of moderate-dimension vectors sit within reach of ordinary hardware. Beyond it the argument changes, because sharding a relational store for vector search is work that the distributed engines have already done. The mistake is assuming you are past that point before measuring.</p></details>

<details class="faq-item"><summary><span>What if geometry needs are simple — just bounding boxes?</span></summary><p>Then the geometry axis stops discriminating and the decision falls to scale and operational cost, which usually favours whatever your team already runs. Be careful about predicting simplicity, though: workloads that begin with rectangles acquire containment and adjacency questions as soon as users discover the system can answer them, and retrofitting real predicates onto a store that lacks them means a migration.</p></details>

<details class="faq-item"><summary><span>How much does the embedding model constrain the choice?</span></summary><p>Mostly through dimensionality and through whether you need multiple vectors per chunk. High dimensionality pushes toward stores with good quantization support; multi-vector retrieval — several embeddings per chunk — is supported unevenly and is worth confirming early if it is in your plans, since emulating it with duplicate records inflates the corpus and confuses deduplication.</p></details>

<details class="faq-item"><summary><span>Does the choice change if the corpus is mostly points rather than polygons?</span></summary><p>It softens the geometry axis considerably. Point data is well served by radius filters, which every candidate offers, so the containment and intersection advantages of a relational store stop being decisive. The remaining differences are scale and operational cost, and for point-only corpora at large scale the dedicated engines become genuinely attractive. Be honest about whether the corpus will stay point-only, though — an address index that later acquires service areas has acquired polygons.</p></details>

<details class="faq-item"><summary><span>Should the decision be revisited later?</span></summary><p>At each order of magnitude of corpus growth, and whenever the embedding changes. Those are the two events that move the answer. Revisiting on a calendar schedule mostly produces churn, because the criteria that matter here change with the workload rather than with the release notes of the candidates.</p></details>

One more consideration rarely surfaces in comparisons and often decides them in practice: who else needs the data. A corpus that also feeds dashboards, exports and analytical queries benefits enormously from living in a store those consumers already speak, and moving it into a specialised engine means every one of them acquires a second data path. That is a real cost, paid by teams who were not in the selection meeting.

## Related

- Up to the parent topic: [Spatial Vector Store Selection](/geospatial-rag-pipelines/spatial-vector-store-selection/)
- [Indexing Spatial Embeddings with HNSW and Metadata Filters](/geospatial-rag-pipelines/spatial-vector-store-selection/indexing-spatial-embeddings-with-hnsw-and-metadata-filters/)
- [Sizing HNSW Parameters for Spatial Recall](/geospatial-rag-pipelines/spatial-vector-store-selection/sizing-hnsw-parameters-for-spatial-recall/)
- Concept: [Spatial Embedding Models](/spatial-llm-architecture-core-concepts/spatial-embedding-models/)
