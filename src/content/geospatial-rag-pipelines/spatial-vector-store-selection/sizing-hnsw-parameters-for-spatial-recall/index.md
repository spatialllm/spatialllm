---
title: Sizing HNSW Parameters for Spatial Recall
description: Choose graph connectivity, build width and search width from your corpus and filter selectivity, with a measurement loop that tells you when to stop turning the knobs.
slug: sizing-hnsw-parameters-for-spatial-recall
type: howto
breadcrumb: Sizing HNSW Parameters
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Sizing HNSW Parameters for Spatial Recall

Three numbers control a graph index: how many edges each node keeps, how hard the builder searches while wiring a node in, and how wide the search runs at query time. Defaults are chosen for unfiltered benchmarks, and spatial workloads are not unfiltered. This guide sizes all three from measurements against your own corpus, extending the index construction in [indexing spatial embeddings with metadata filters](/geospatial-rag-pipelines/spatial-vector-store-selection/indexing-spatial-embeddings-with-hnsw-and-metadata-filters/).

## When to Use This Approach

Tune when measured recall under your tightest realistic filter falls short of target, or when index memory has become a constraint. Do not tune preemptively: the defaults are reasonable, and a parameter set chosen before there is a recall measurement to improve is a parameter set chosen by taste.

| Symptom | Parameter to move | Direction |
|---------|-------------------|-----------|
| Recall short under tight filters | Connectivity, then search width | Up |
| Recall fine, latency too high | Search width | Down |
| Index memory over budget | Connectivity, then quantization | Down |
| Build takes too long | Build width | Down |
| Recall fine loose, poor tight | Connectivity — not search width | Up |

The last row is the one worth memorising. Widening the search at query time is the reflexive fix and it treats the symptom: if the graph is poorly connected in the regions your filters select, a wider search visits more of the wrong neighbourhood.

<figure class="diagram">
<svg viewBox="16 38 748 188" role="img" aria-labelledby="hp-three-t hp-three-d" xmlns="http://www.w3.org/2000/svg"><title id="hp-three-t">What each of the three parameters costs and buys</title><desc id="hp-three-d">Connectivity buys filtered recall at the cost of memory and build time; build width buys graph quality at the cost of build time only; search width buys recall at the cost of query latency only.</desc><rect x="16" y="38" width="748" height="188" fill="#ffffff"/><rect x="30" y="52" width="230" height="160" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="275" y="52" width="230" height="160" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="520" y="52" width="230" height="160" rx="8" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="145" y="82">connectivity</text><text x="390" y="82">build width</text><text x="635" y="82">search width</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="145" y="112">buys: filtered recall</text><text x="145" y="136">costs: memory, build</text><text x="145" y="166">set once, per corpus</text><text x="145" y="190">the expensive knob</text><text x="390" y="112">buys: graph quality</text><text x="390" y="136">costs: build time only</text><text x="390" y="166">set once, generously</text><text x="390" y="190">the cheap knob</text><text x="635" y="112">buys: recall per query</text><text x="635" y="136">costs: latency only</text><text x="635" y="166">set per query</text><text x="635" y="190">the runtime knob</text></g></svg>
<figcaption><b>Only one of the three is free to change later.</b> Search width is per query and can be tuned in production; the other two are baked into the graph and changing them means a rebuild, which is why they deserve the measurement effort.</figcaption>
</figure>

## Implementation

The tuning loop builds a small index at several parameter settings, measures filtered recall against brute-force truth, and reports the smallest setting that clears the target — the setting you want, since everything above it costs memory for nothing.

```python
import logging
import math
from dataclasses import dataclass
from typing import Callable, Sequence

log = logging.getLogger("hnsw_sizing")


@dataclass(frozen=True)
class Trial:
    m: int
    ef_construction: int
    ef_search: int
    recall: float
    memory_gib: float
    build_s: float


def estimate_memory_gib(n: int, dim: int, m: int, bytes_per_link: int = 4) -> float:
    """Vectors plus bidirectional neighbour lists. Metadata is on top of this."""
    vectors = n * dim * 4
    links = n * m * 2 * bytes_per_link
    return round((vectors + links) / 1024 ** 3, 3)


def sweep(
    build_index: Callable[[int, int], object],
    measure_recall: Callable[[object, int], float],
    n: int,
    dim: int,
    target_recall: float = 0.90,
    m_values: Sequence[int] = (12, 16, 24, 32, 48),
    ef_construction: int = 128,
    ef_search_values: Sequence[int] = (60, 120, 250, 500),
) -> list[Trial]:
    """Sweep connectivity and search width; return every trial for inspection."""
    if not 0.0 < target_recall < 1.0:
        raise ValueError("target_recall must be a fraction between 0 and 1")

    trials: list[Trial] = []
    for m in m_values:
        try:
            index = build_index(m, ef_construction)
        except Exception as exc:                     # out of memory at high connectivity
            log.warning("build failed at m=%d: %s — stopping the sweep here", m, exc)
            break                                    # larger m will fail too
        for ef in ef_search_values:
            try:
                recall = measure_recall(index, ef)
            except Exception as exc:
                log.warning("recall measurement failed at m=%d ef=%d: %s", m, ef, exc)
                recall = 0.0                         # a failed measurement is not a good one
            trials.append(Trial(m, ef_construction, ef, round(recall, 4),
                                estimate_memory_gib(n, dim, m), 0.0))
    return trials


def cheapest_passing(trials: Sequence[Trial], target: float) -> Trial | None:
    """The smallest configuration that clears the target — not the best recall."""
    passing = [t for t in trials if t.recall >= target]
    if not passing:
        log.info("no configuration reached %.2f recall; consider quantization or a rebuild", target)
        return None
    return min(passing, key=lambda t: (t.memory_gib, t.ef_search))
```

The choice to return the *cheapest* passing configuration rather than the best-scoring one is the point of the whole exercise. Recall above target buys nothing a reader will notice, and the difference between connectivity 24 and 48 on a twenty-million-vector corpus is tens of gigabytes of memory spent on results nobody sees.

Measure at your tightest realistic selectivity, not at an average one. A configuration tuned at 10% selectivity will be under-provisioned at 0.5%, and 0.5% is what a query about one neighbourhood in a national corpus actually looks like.

<figure class="diagram">
<svg viewBox="66 7 668 239" role="img" aria-labelledby="hp-mem-t hp-mem-d" xmlns="http://www.w3.org/2000/svg"><title id="hp-mem-t">Memory and recall across connectivity settings</title><desc id="hp-mem-d">Recall rises steeply from low connectivity and flattens, while memory rises linearly, so the useful setting is the knee rather than the highest value tested.</desc><rect x="66" y="7" width="668" height="239" fill="#ffffff"/><text x="380" y="32" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">Filtered recall and index memory against connectivity</text><g fill="#e4f5ec" stroke="#12805c" stroke-width="2"><rect x="80" y="118" width="46" height="56" rx="4"/><rect x="220" y="92" width="46" height="82" rx="4"/><rect x="360" y="72" width="46" height="102" rx="4"/><rect x="500" y="68" width="46" height="106" rx="4"/><rect x="640" y="66" width="46" height="108" rx="4"/></g><g fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"><rect x="132" y="152" width="28" height="22" rx="3"/><rect x="272" y="146" width="28" height="28" rx="3"/><rect x="412" y="134" width="28" height="40" rx="3"/><rect x="552" y="122" width="28" height="52" rx="3"/><rect x="692" y="98" width="28" height="76" rx="3"/></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="116" y="196">m 12</text><text x="256" y="196">m 16</text><text x="396" y="196">m 24</text><text x="536" y="196">m 32</text><text x="676" y="196">m 48</text></g><text x="380" y="228" fill="#1f2937" font-size="12.5" text-anchor="middle">Tall bars are recall; narrow bars are memory — the knee sits at m 24</text></svg>
<figcaption><b>Pick the knee, not the peak.</b> Between connectivity 24 and 48 recall gains a point and memory nearly doubles, which is a trade worth refusing on any corpus large enough for the question to arise.</figcaption>
</figure>

## Validation & Testing

```python
def test_sweep_returns_a_configuration_that_clears_target():
    trials = sweep(build_index, measure_recall, n=2_000_000, dim=768, target_recall=0.9)
    chosen = cheapest_passing(trials, 0.9)
    assert chosen is not None and chosen.recall >= 0.9


def test_cheapest_passing_prefers_memory_over_recall():
    trials = [
        Trial(16, 128, 120, 0.91, 6.5, 0.0),
        Trial(48, 128, 120, 0.97, 14.2, 0.0),
    ]
    assert cheapest_passing(trials, 0.9).m == 16


def test_failed_measurement_scores_zero_not_none():
    trials = sweep(build_index_that_fails_recall, measure_recall_raises, 1000, 128)
    assert all(t.recall == 0.0 for t in trials)


def test_target_recall_must_be_a_fraction():
    try:
        sweep(build_index, measure_recall, 1000, 128, target_recall=90)
    except ValueError:
        return
    raise AssertionError("a target of 90 should be rejected, not treated as 90%")
```

The last test looks pedantic and guards a real mistake: recall expressed as a percentage rather than a fraction silently makes every configuration fail, and the resulting investigation goes looking for an index problem that does not exist.

Rerun the sweep whenever the embedding model or the corpus size changes by an order of magnitude. Those are the two inputs the chosen configuration depends on, and both change without anyone thinking of the index.

## Gotchas & Edge Cases

**Tuning on unfiltered recall.** The defaults are already good unfiltered, so a sweep without filters shows almost no variation and concludes that the parameters do not matter. Apply the real filter, at the real selectivity, or the sweep is measuring the wrong thing.

**Memory estimated without metadata.** The model above counts vectors and links. Real indexes also store identifiers, deletion markers and per-record metadata, which on short vectors can rival the vector data itself. Measure the built index rather than trusting the estimate, and treat the estimate as a planning tool only.

<figure class="diagram">
<svg viewBox="16 46 728 168" role="img" aria-labelledby="hp-order-t hp-order-d" xmlns="http://www.w3.org/2000/svg"><title id="hp-order-t">The order to move the three knobs</title><desc id="hp-order-d">Establish a baseline, then raise search width because it is free to change, then connectivity because it requires a rebuild, then consider quantization only when memory binds.</desc><rect x="16" y="46" width="728" height="168" fill="#ffffff"/><rect x="30" y="60" width="160" height="86" rx="8" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="210" y="60" width="160" height="86" rx="8" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><rect x="390" y="60" width="160" height="86" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="570" y="60" width="160" height="86" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="12.5" text-anchor="middle" font-weight="600"><text x="110" y="92">1 · baseline</text><text x="290" y="92">2 · search width</text><text x="470" y="92">3 · connectivity</text><text x="650" y="92">4 · quantization</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="110" y="118">measure before tuning</text><text x="290" y="118">free, per query</text><text x="470" y="118">needs a rebuild</text><text x="650" y="118">only if memory binds</text></g><text x="380" y="196" fill="#1f2937" font-size="13" text-anchor="middle">Cheapest reversible change first — the last two are commitments</text></svg>
<figcaption><b>Reversibility, not impact, sets the order.</b> Search width can be changed and changed back within a request; connectivity is a rebuild, and quantization changes what the stored vectors are.</figcaption>
</figure>

**Build width traded away to save time.** Build width costs build time and nothing else, so cutting it to speed a rebuild is a false economy — it produces a permanently worse graph in exchange for a one-off saving. If builds are too slow, parallelise them or reduce the corpus, but leave the width generous.

**A sweep that stops at the first passing value.** Sweeping connectivity upward and stopping at the first pass finds a passing configuration, not the cheapest one; recall is not always monotone in connectivity under filtering, and the cheapest pass can sit above a value that failed. Sweep the whole range, then select.

**Different parameters per region.** It is tempting to tune separately for dense urban regions and sparse rural ones. One index has one set of build parameters, so this only works with separate indexes — which is a legitimate design, and a much bigger commitment than a parameter change.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Is connectivity the same knob as the number of neighbours returned?</span></summary><p>No, and conflating them is common. Connectivity is a property of the graph — how many edges each node keeps — fixed at build time. The number of neighbours returned is a query parameter and has no effect on the structure. Raising the result count without raising the search width simply returns more of an inadequately explored neighbourhood.</p></details>

<details class="faq-item"><summary><span>How large a sample is enough for the sweep?</span></summary><p>Large enough to be clustered the way the real corpus is, which usually means a few hundred thousand vectors drawn from real data rather than a uniform random sample. Uniform samples flatter approximate indexes because uniformly distributed points have no hard neighbourhoods. Sampling by region — take everything from a few areas rather than a scatter from everywhere — preserves the clustering that makes the measurement meaningful.</p></details>

<details class="faq-item"><summary><span>Should search width vary with the result count as well as selectivity?</span></summary><p>Yes, and roughly linearly in both. The search must visit enough eligible nodes to fill the result set, so the width scales with the count divided by the selectivity, which is exactly the estimate the previous guide uses. Bound it at both ends, since the formula produces absurd values for a nearly empty filter.</p></details>

<details class="faq-item"><summary><span>When is quantization the right answer instead of tuning?</span></summary><p>When memory is the binding constraint and recall at the cheapest passing configuration still fits comfortably above target. Quantization trades a few points of recall for a large memory saving, so it needs headroom to spend. Applying it when recall is already marginal produces a configuration that fails on exactly the selective queries this whole exercise is about.</p></details>

<details class="faq-item"><summary><span>Does the distance metric affect these choices?</span></summary><p>It affects the absolute numbers but not the shape of the trade-offs. What does matter is consistency: the index must be built with the same metric the queries use, and a mismatch produces results that look plausible and rank wrongly. Assert the metric in the same test that asserts recall, since it is invisible in the output otherwise.</p></details>

One habit makes all of this cheaper: keep the sweep script in the repository next to the index definition, and run it as part of the same job that rebuilds. The alternative — a notebook someone ran once — means the next person to ask "why is m set to 24" has no way to find out, and the safest answer available to them is to leave it alone forever.

## Related

- Up to the parent topic: [Spatial Vector Store Selection](/geospatial-rag-pipelines/spatial-vector-store-selection/)
- [Indexing Spatial Embeddings with HNSW and Metadata Filters](/geospatial-rag-pipelines/spatial-vector-store-selection/indexing-spatial-embeddings-with-hnsw-and-metadata-filters/)
- [pgvector vs Qdrant vs Milvus for Spatial Embeddings](/geospatial-rag-pipelines/spatial-vector-store-selection/pgvector-vs-qdrant-vs-milvus-for-spatial-embeddings/)
- Technique: [Choosing Vector Dimensionality for Spatial Retrieval](/spatial-llm-architecture-core-concepts/spatial-embedding-models/choosing-vector-dimensionality-for-spatial-retrieval/)
