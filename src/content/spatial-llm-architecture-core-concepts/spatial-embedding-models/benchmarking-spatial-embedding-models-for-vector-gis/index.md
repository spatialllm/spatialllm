---
title: Benchmarking Spatial Embedding Models for Vector GIS
description: Compare embedding candidates on your own place-oriented queries under a real region filter, and report the numbers that decide it rather than the ones that are published.
slug: benchmarking-spatial-embedding-models-for-vector-gis
type: howto
breadcrumb: Benchmarking Embeddings
datePublished: 2025-02-19
dateModified: 2026-08-11
---

# Benchmarking Spatial Embedding Models for Vector GIS

Published embedding benchmarks measure general text and rank models accordingly. A corpus of survey reports, planning documents and site assessments is not general text, and the ranking frequently reorders on it — usually in favour of a smaller model. This guide runs the comparison that actually decides the choice, as the measurement half of [spatial embedding models](/spatial-llm-architecture-core-concepts/spatial-embedding-models/).

## When to Use This Approach

Benchmark when choosing a model, when a candidate replacement appears, and when the corpus acquires a substantially new kind of document. Not otherwise — the comparison costs a day and its answer is stable between those events.

| Measurement | Decides | Usually published? |
|-------------|---------|--------------------|
| Recall on your queries | Almost everything | No |
| Recall under a region filter | Whether it suits this workload | No |
| Recall on rare place names | Whether the lexical half is doing all the work | No |
| Encoding throughput | Rebuild time | Sometimes |
| Dimensionality | The memory bill | Yes |
| General benchmark score | Little | Always |

The inversion in that table is the point. The one number that is always available is the one least likely to decide the outcome, and the numbers that decide it have to be produced locally.

<figure class="diagram">
<svg viewBox="26 9 588 237" role="img" aria-labelledby="bse-set-t bse-set-d" xmlns="http://www.w3.org/2000/svg"><title id="bse-set-t">What a place-oriented evaluation set looks like</title><desc id="bse-set-d">The query mix that decides an embedding choice for a spatial corpus is dominated by named places, feature types and local descriptions rather than by general paraphrase.</desc><rect x="26" y="9" width="588" height="237" fill="#ffffff"/><text x="390" y="34" fill="#5b6471" font-size="13" text-anchor="middle">The query mix a spatial corpus actually receives</text><rect x="220" y="56" width="380" height="38" rx="5" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="220" y="102" width="280" height="38" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="220" y="148" width="200" height="38" rx="5" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><rect x="220" y="194" width="90" height="38" rx="5" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="12.5"><text x="40" y="81">named place</text><text x="40" y="127">feature type</text><text x="40" y="173">local description</text><text x="40" y="219">general prose</text></g><g fill="#1f2937" font-size="12" text-anchor="middle"><text x="410" y="81">42%</text><text x="360" y="127">28%</text><text x="320" y="173">21%</text><text x="265" y="219">9%</text></g></svg>
<figcaption><b>Nine per cent of this mix resembles a general benchmark.</b> A model ranked on the bottom bar is being ranked on the smallest slice of the workload, which is why local measurement so often reorders the published list.</figcaption>
</figure>

## Implementation

The harness encodes the corpus once per candidate, runs the evaluation set with and without a region filter, and reports both alongside cost.

```python
import logging
import time
from dataclasses import dataclass
from typing import Callable, Sequence

log = logging.getLogger("embedding_benchmark")


@dataclass(frozen=True)
class Candidate:
    name: str
    dim: int
    encode: Callable[[Sequence[str]], object]     # batch encode
    max_input_tokens: int


@dataclass(frozen=True)
class Result:
    name: str
    dim: int
    recall_unfiltered: float
    recall_filtered: float
    recall_rare_names: float
    encode_docs_per_s: float
    notes: tuple[str, ...]


def evaluate(candidate: Candidate, corpus: Sequence[dict], queries: Sequence[dict],
             search, k: int = 10) -> Result:
    """Encode the corpus once, then measure three recall figures and throughput."""
    notes: list[str] = []
    texts = [c["embedding_text"] for c in corpus]

    started = time.monotonic()
    try:
        vectors = candidate.encode(texts)
    except Exception as exc:                        # a candidate that cannot encode is a result
        log.warning("%s failed to encode the corpus: %s", candidate.name, exc)
        return Result(candidate.name, candidate.dim, 0.0, 0.0, 0.0, 0.0,
                      (f"encode failed: {exc}",))
    throughput = len(texts) / max(1e-6, time.monotonic() - started)

    def recall(subset, region_filter) -> float:
        hit = tot = 0
        for q in subset:
            want = set(q["relevant"])
            if not want:
                continue
            try:
                got = set(search(candidate.encode([q["text"]])[0], vectors, k, region_filter))
            except Exception as exc:
                notes.append(f"search failed on {q['id']}: {exc}")
                got = set()
            hit += len(want & got)
            tot += len(want)
        return round(hit / tot, 4) if tot else 0.0

    rare = [q for q in queries if q.get("has_rare_name")]
    return Result(
        candidate.name, candidate.dim,
        recall(queries, None),
        recall(queries, "region"),
        recall(rare, "region") if rare else 0.0,
        round(throughput, 1),
        tuple(notes[:5]),
    )
```

Encoding the corpus once per candidate rather than per query is what makes the harness affordable; the query encoding inside the recall loop is the small cost. Recording throughput at the same time is nearly free and answers the rebuild-time question that will be asked immediately afterwards.

Separating the rare-name recall is the measurement most likely to change a decision. If a candidate's overall recall is competitive but its rare-name recall is poor, the lexical half of a hybrid system is carrying those queries — which is fine, and it means the dense half is contributing less than the headline number suggests.

```python
def summarise(results: Sequence[Result], target_gap: float = 0.02) -> str:
    """Report the comparison in the terms that decide it."""
    if not results:
        return "no candidates evaluated"
    best_filtered = max(results, key=lambda r: r.recall_filtered)
    close = [r for r in results
             if best_filtered.recall_filtered - r.recall_filtered <= target_gap]
    cheapest = min(close, key=lambda r: (r.dim, -r.encode_docs_per_s))
    if cheapest.name != best_filtered.name:
        return (f"{cheapest.name} at dim {cheapest.dim} is within {target_gap:.02f} "
                f"of {best_filtered.name} on filtered recall, at lower cost")
    return f"{best_filtered.name} leads on filtered recall at dim {best_filtered.dim}"
```

<figure class="diagram">
<svg viewBox="16 24 598 226" role="img" aria-labelledby="bse-flip-t bse-flip-d" xmlns="http://www.w3.org/2000/svg"><title id="bse-flip-t">A ranking that reorders under a region filter</title><desc id="bse-flip-d">Two candidates are close on unfiltered recall and separate clearly once a region filter removes most of the corpus, which is the condition the real workload runs under.</desc><rect x="16" y="24" width="598" height="226" fill="#ffffff"/><text x="30" y="62" fill="#5b6471" font-size="12.5">unfiltered</text><rect x="180" y="38" width="420" height="42" rx="5" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="180" y="86" width="404" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="11.5"><text x="200" y="64">model A — 0.93</text><text x="200" y="112">model B — 0.91</text></g><text x="30" y="182" fill="#5b6471" font-size="12.5">filtered</text><rect x="180" y="158" width="240" height="42" rx="5" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="180" y="206" width="380" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="11.5"><text x="200" y="184">model A — 0.61</text><text x="200" y="232">model B — 0.88</text></g></svg>
<figcaption><b>The order reverses where the workload lives.</b> Unfiltered recall separates these candidates by two points; under the filter that every real query carries, it separates them by twenty-seven.</figcaption>
</figure>

## Validation & Testing

```python
def test_encode_failure_is_a_result_not_an_abort():
    class Broken(Candidate):
        pass
    broken = Candidate("broken", 384, lambda _t: (_ for _ in ()).throw(RuntimeError("x")), 512)
    r = evaluate(broken, CORPUS, QUERIES, search)
    assert r.recall_filtered == 0.0 and r.notes


def test_filtered_and_unfiltered_are_reported_separately():
    r = evaluate(CANDIDATE, CORPUS, QUERIES, search)
    assert r.recall_unfiltered != r.recall_filtered or len(QUERIES) < 5


def test_summary_prefers_a_cheaper_candidate_within_the_gap():
    results = [Result("big", 1536, 0.93, 0.90, 0.7, 40.0, ()),
               Result("small", 384, 0.91, 0.89, 0.7, 180.0, ())]
    assert "small" in summarise(results, target_gap=0.02)


def test_rare_name_recall_is_measured_on_a_subset():
    r = evaluate(CANDIDATE, CORPUS, [q for q in QUERIES if q.get("has_rare_name")], search)
    assert 0.0 <= r.recall_rare_names <= 1.0
```

The third test encodes the decision rule rather than a measurement, which is the point of having a summary function at all. Left to a table of numbers, a comparison is settled by whoever reads it first; a rule that prefers the cheaper candidate within a stated gap makes the trade explicit and reviewable.

Record the harness output as a file in the repository rather than as a message in a channel. The comparison will be re-litigated — when a new model appears, when someone new joins, when recall drops — and a stored table with its conditions attached settles the question in minutes where a remembered conclusion restarts it.

## Gotchas & Edge Cases

**Evaluating on queries written by the person choosing the model.** They will unconsciously favour the phrasing the current system handles. Draw queries from real traffic, or from the corpus by someone who has not seen the candidates.

**Corpus sample too small to cluster.** A few thousand chunks drawn at random are more uniformly distributed than the real corpus, which flatters every candidate. Sample by region and keep everything from those regions.

**Chunk text differing between candidates.** If one candidate is fed the chunk body and another the constructed embedding text, the comparison measures the construction. Build the text once and reuse it.

**Input truncation going unnoticed.** A candidate with a shorter input limit silently truncates long chunks and loses their tails. Check the corpus percentile against the limit before encoding, and record it as a note rather than discovering it as poor recall.

<figure class="diagram">
<svg viewBox="40 42 681 180" role="img" aria-labelledby="bse-trunc-t bse-trunc-d" xmlns="http://www.w3.org/2000/svg"><title id="bse-trunc-t">Silent truncation against a candidate&#8217;s input limit</title><desc id="bse-trunc-d">A candidate with a shorter input limit drops the tail of long chunks without error, so its recall is measured on partial documents while its competitor sees them whole.</desc><rect x="40" y="42" width="681" height="180" fill="#ffffff"/><rect x="60" y="56" width="520" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="320" y="82" fill="#1f2937" font-size="12" text-anchor="middle">candidate A sees the whole chunk</text><rect x="60" y="118" width="300" height="42" rx="5" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="366" y="118" width="214" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="210" y="144" fill="#1f2937" font-size="12" text-anchor="middle">candidate B sees this much</text><text x="473" y="144" fill="#1f2937" font-size="12" text-anchor="middle">silently dropped</text><text x="380" y="204" fill="#1f2937" font-size="13" text-anchor="middle">Check the corpus percentile against every candidate&#8217;s limit before encoding</text></svg>
<figcaption><b>No error, no warning, a different measurement.</b> Truncation is a property of the candidate rather than of the corpus, so the comparison is no longer between two models reading the same thing.</figcaption>
</figure>

**Throughput measured on a warm cache.** The first candidate encodes cold and the rest encode warm, which flatters everything after the first. Randomise the order or discard a warm-up batch.

**Query encoding excluded from the throughput figure.** Corpus encoding dominates a rebuild and query encoding dominates the request path, and a candidate can be fast at one and slow at the other. Record both if latency matters to you.

**Normalisation differing between candidates.** Some models return unit vectors and some do not, and comparing cosine similarity across the two without normalising measures the vector lengths. Normalise consistently before searching.

## Frequently Asked Questions

<details class="faq-item"><summary><span>How many queries does the evaluation set need?</span></summary><p>Thirty to fifty for a shortlisting decision, and more if you want to distinguish candidates separated by a point or two. The rare-name subset matters more than the total: ten genuinely hard name queries tell you more about how a candidate will behave on this corpus than a hundred paraphrase pairs. Build the set once and reuse it for every subsequent comparison, including the ones you have not thought of yet.</p></details>

<details class="faq-item"><summary><span>Should the benchmark include the reranker?</span></summary><p>No, and this is a common mistake. A cross-encoder reranking the top candidates masks differences in the retrieval that produced them, so a weak embedding whose top fifty happens to contain the answer scores as well as a strong one. Measure the embedding alone at the depth the reranker consumes, then measure the whole pipeline separately as an end-to-end check.</p></details>

<details class="faq-item"><summary><span>What if two candidates are genuinely tied?</span></summary><p>Take the smaller, faster, or more operationally boring one, in that order. A tie on quality means the decision falls to cost and risk, and a model with a smaller footprint, a permissive licence and a stable release history is worth more than a marginal recall difference you cannot reliably reproduce. Record the tie so the next comparison starts from it.</p></details>

<details class="faq-item"><summary><span>How often should this be repeated?</span></summary><p>When something changes, not on a schedule. New model releases are frequent and mostly irrelevant to a corpus of technical prose; a substantial new document source or a tenfold corpus growth genuinely can reorder the ranking. Keeping the harness runnable is what makes the answer cheap when the question arises, which matters more than running it regularly.</p></details>

<details class="faq-item"><summary><span>Where should the evaluation set live?</span></summary><p>In the repository, versioned with the code that uses it, and treated as an asset rather than as a test fixture. It outlives every model decision it informs, it is the thing that makes future comparisons cheap, and it is the first thing lost when it lives in someone&#8217;s notebook. Record which version of the set produced any published recall figure, because a set that has grown is not comparable to the one that came before it.</p></details>

## Related

- Up to the parent topic: [Spatial Embedding Models](/spatial-llm-architecture-core-concepts/spatial-embedding-models/)
- [Choosing Vector Dimensionality for Spatial Retrieval](/spatial-llm-architecture-core-concepts/spatial-embedding-models/choosing-vector-dimensionality-for-spatial-retrieval/)
- Related topic: [Hybrid Spatial and Keyword Retrieval](/geospatial-rag-pipelines/hybrid-spatial-keyword-retrieval/)
- Related technique: [Indexing Spatial Embeddings with HNSW and Metadata Filters](/geospatial-rag-pipelines/spatial-vector-store-selection/indexing-spatial-embeddings-with-hnsw-and-metadata-filters/)
