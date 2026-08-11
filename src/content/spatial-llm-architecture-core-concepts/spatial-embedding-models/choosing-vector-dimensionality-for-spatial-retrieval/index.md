---
title: Choosing Vector Dimensionality for Spatial Retrieval
description: Pick the smallest dimensionality that holds recall on your own corpus, model the memory it commits you to, and keep the option to reduce it later.
slug: choosing-vector-dimensionality-for-spatial-retrieval
type: howto
breadcrumb: Choosing Dimensionality
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Choosing Vector Dimensionality for Spatial Retrieval

Dimensionality is chosen by picking a model, and it is the single number that determines the memory footprint of every index and every replica for the life of the system. Choosing it deliberately means measuring recall at several sizes on your own corpus and taking the smallest that holds — which is usually far below the largest available. This guide does that, as the sizing decision within [spatial embedding models](/spatial-llm-architecture-core-concepts/spatial-embedding-models/).

## When to Use This Approach

Measure before committing, and re-measure whenever the corpus grows by an order of magnitude or the model changes. Between those events the number is settled and should be left alone.

| Corpus size | Dimensionality pressure | Typical outcome |
|-------------|-------------------------|-----------------|
| Under a million chunks | Negligible — pick on quality | Any size fits comfortably |
| A few million | Moderate | 384 to 768 usually optimal |
| Tens of millions | Dominant | 384 or quantized 768 |
| Hundreds of millions | Decisive | Reduced or quantized, always |

The second row is where most spatial corpora sit, and it is also where the largest available model is most tempting and least justified: the recall difference between 768 and 1536 dimensions on technical prose is typically under a point, and the memory difference is a factor of two.

<figure class="diagram">
<svg viewBox="16 38 748 198" role="img" aria-labelledby="cvd-two-t cvd-two-d" xmlns="http://www.w3.org/2000/svg"><title id="cvd-two-t">Two independent decisions, one memory bill</title><desc id="cvd-two-d">Corpus growth and a move to a larger embedding are usually decided by different people at different times, and their effects multiply into an index that no longer fits its host.</desc><rect x="16" y="38" width="748" height="198" fill="#ffffff"/><rect x="30" y="52" width="330" height="76" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="410" y="52" width="340" height="76" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="195" y="80" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">the corpus doubled</text><text x="195" y="104" fill="#5b6471" font-size="12" text-anchor="middle">because the product succeeded</text><text x="580" y="80" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">the model got better</text><text x="580" y="104" fill="#5b6471" font-size="12" text-anchor="middle">and doubled the dimensions</text><rect x="30" y="152" width="720" height="70" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="390" y="182" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">the index is now four times the size</text><text x="390" y="206" fill="#5b6471" font-size="12" text-anchor="middle">and nobody made a decision that looked like quadrupling anything</text></svg>
<figcaption><b>Neither change looked like a capacity decision.</b> That is what makes this worth modelling in advance: the two multiply, and the moment they do is the moment a single-machine deployment becomes a cluster.</figcaption>
</figure>

## Implementation

The measurement sweeps dimensionality, records recall and memory together, and returns the cheapest configuration that clears the target.

```python
import logging
from dataclasses import dataclass
from typing import Callable, Sequence

log = logging.getLogger("dimensionality")


@dataclass(frozen=True)
class Point:
    dim: int
    recall: float
    index_gib: float
    note: str = ""


def index_gib(n_chunks: int, dim: int, m: int = 16, bytes_per_link: int = 4) -> float:
    """Vectors plus graph links. Metadata and identifiers are on top of this."""
    vectors = n_chunks * dim * 4
    links = n_chunks * m * 2 * bytes_per_link
    return round((vectors + links) / 1024 ** 3, 2)


def sweep(dims: Sequence[int], n_chunks: int,
          measure_recall: Callable[[int], float]) -> list[Point]:
    """Measure filtered recall at each dimensionality on the real corpus."""
    out: list[Point] = []
    for dim in sorted(dims):
        try:
            recall = measure_recall(dim)
        except Exception as exc:                    # a failed measurement is not a good one
            log.warning("recall measurement failed at dim=%d: %s", dim, exc)
            out.append(Point(dim, 0.0, index_gib(n_chunks, dim), f"measurement failed: {exc}"))
            continue
        out.append(Point(dim, round(recall, 4), index_gib(n_chunks, dim)))
    return out


def cheapest_meeting(points: Sequence[Point], target_recall: float,
                     memory_budget_gib: float | None = None) -> Point | None:
    """Smallest index that clears the recall target and fits the budget."""
    if not 0.0 < target_recall < 1.0:
        raise ValueError("target recall must be a fraction between 0 and 1")
    eligible = [p for p in points if p.recall >= target_recall
                and (memory_budget_gib is None or p.index_gib <= memory_budget_gib)]
    if not eligible:
        log.info("no dimensionality met recall %.3f within the memory budget", target_recall)
        return None
    return min(eligible, key=lambda p: (p.index_gib, p.dim))
```

Measuring recall under the real filter is what makes this exercise meaningful. Unfiltered recall is nearly flat across dimensionality on most corpora, so an unfiltered sweep concludes that the choice does not matter — which is true of the measurement and not of the workload.

Returning `None` rather than the best available when nothing meets the target is deliberate. That condition is a real finding: it means the corpus needs either a better model, a different chunking strategy, or a larger machine, and returning the highest-recall option quietly papers over the choice.

<figure class="diagram">
<svg viewBox="66 7 648 239" role="img" aria-labelledby="cvd-knee-t cvd-knee-d" xmlns="http://www.w3.org/2000/svg"><title id="cvd-knee-t">Recall and memory across four dimensionalities</title><desc id="cvd-knee-d">Recall rises steeply from the smallest size and flattens, while memory grows linearly, so the useful configuration is the knee rather than the largest tested.</desc><rect x="66" y="7" width="648" height="239" fill="#ffffff"/><text x="390" y="32" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">Filtered recall and index memory, measured on one corpus</text><g fill="#e4f5ec" stroke="#12805c" stroke-width="2"><rect x="80" y="122" width="54" height="52" rx="4"/><rect x="240" y="88" width="54" height="86" rx="4"/><rect x="400" y="76" width="54" height="98" rx="4"/><rect x="560" y="72" width="54" height="102" rx="4"/></g><g fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"><rect x="140" y="158" width="32" height="16" rx="3"/><rect x="300" y="146" width="32" height="28" rx="3"/><rect x="460" y="118" width="32" height="56" rx="3"/><rect x="620" y="62" width="32" height="112" rx="3"/></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="120" y="196">256</text><text x="280" y="196">384</text><text x="440" y="196">768</text><text x="600" y="196">1536</text></g><text x="390" y="228" fill="#1f2937" font-size="12.5" text-anchor="middle">Wide bars are recall; narrow bars are memory — the knee sits between 384 and 768</text></svg>
<figcaption><b>The last doubling buys a point and costs half the machine.</b> On technical prose the curve flattens early, and the configuration worth shipping is the one just past the flattening rather than the one at the end of the axis.</figcaption>
</figure>

## Validation & Testing

```python
def test_cheapest_meeting_prefers_smaller_over_better():
    points = [Point(384, 0.91, 6.5), Point(1536, 0.97, 24.1)]
    assert cheapest_meeting(points, 0.90).dim == 384


def test_memory_budget_excludes_a_configuration_that_meets_recall():
    points = [Point(384, 0.88, 6.5), Point(1536, 0.97, 24.1)]
    assert cheapest_meeting(points, 0.90, memory_budget_gib=10.0) is None


def test_failed_measurement_scores_zero_and_is_noted():
    def raising(_dim):
        raise RuntimeError("encoder unavailable")
    points = sweep([384], 1_000_000, raising)
    assert points[0].recall == 0.0 and "failed" in points[0].note


def test_target_must_be_a_fraction():
    try:
        cheapest_meeting([Point(384, 0.9, 1.0)], 90)
    except ValueError:
        return
    raise AssertionError("a target of 90 must be rejected, not read as 90%")


def test_memory_model_scales_with_both_inputs():
    assert index_gib(2_000_000, 768) > index_gib(1_000_000, 768)
    assert index_gib(1_000_000, 1536) > index_gib(1_000_000, 768)
```

The second test is the one that produces a useful conversation rather than a silent compromise. When nothing fits, somebody has to decide between a bigger machine, a smaller corpus and a lower target, and a function that quietly returns the best available option removes that decision from view.

Run the sweep against the same evaluation set used to choose the model, so the two decisions are measured on comparable ground. A dimensionality chosen against a different query set than the model was chosen against produces a configuration that is optimal for neither, and the discrepancy is invisible because both numbers look reasonable in isolation.

## Gotchas & Edge Cases

**Memory estimated from vectors alone.** Graph links, identifiers, deletion markers and per-record metadata are all on top, and on short vectors the overhead can rival the data. Measure the built index and treat the model as planning arithmetic.

**Recall measured on a uniform sample.** Real embeddings are clustered, and clustered data is where approximate indexes lose recall. Sample by region — everything from a few areas — rather than scattering across the corpus.

**Dimensionality chosen with the model and never revisited.** The two are separable in more cases than people expect, and a model that supports shortened outputs makes the choice a slice rather than a migration. Ask about that capability before committing.

**Quantization applied before the baseline exists.** Compressing first makes a later recall problem impossible to attribute. Measure uncompressed, then compress and measure again.

<figure class="diagram">
<svg viewBox="46 42 688 172" role="img" aria-labelledby="cvd-rep-t cvd-rep-d" xmlns="http://www.w3.org/2000/svg"><title id="cvd-rep-t">One index against the memory a deployment actually commits</title><desc id="cvd-rep-d">The sizing conversation is usually about a single index while the deployment runs several replicas, so the real commitment is a multiple of the figure that was discussed.</desc><rect x="46" y="42" width="688" height="172" fill="#ffffff"/><rect x="60" y="56" width="140" height="50" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="130" y="86" fill="#1f2937" font-size="12.5" text-anchor="middle">30 GiB</text><text x="130" y="126" fill="#5b6471" font-size="12" text-anchor="middle">what was discussed</text><rect x="300" y="56" width="140" height="50" rx="6" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="450" y="56" width="140" height="50" rx="6" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="600" y="56" width="120" height="50" rx="6" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="510" y="126" fill="#5b6471" font-size="12" text-anchor="middle">three replicas: 90 GiB committed</text><text x="380" y="196" fill="#1f2937" font-size="13" text-anchor="middle">Multiply before comparing the estimate against a host</text></svg>
<figcaption><b>The estimate and the commitment differ by the replica count.</b> Nothing in the sizing arithmetic mentions replicas, and the deployment topology is usually decided by different people in a different conversation.</figcaption>
</figure>

**Replicas forgotten in the budget.** Three replicas of a 30 GiB index is 90 GiB of memory, and the sizing conversation usually happens about one copy. Multiply before comparing against a host.

**Quantized and unquantized configurations compared on one axis.** They are different points on a different curve, and plotting them together invites a comparison of memory figures that were produced under different assumptions about recall. Sweep uncompressed first, choose, then evaluate compression against that choice.

**A target set higher than the workload needs.** Chasing ninety-nine per cent recall costs disproportionately and rarely changes answers, because the reranking stage reorders the top candidates anyway. Set the target from measured answer quality, not from the metric.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Can dimensionality be reduced after the index is built?</span></summary><p>Sometimes cheaply and sometimes not. Models trained so that a prefix of the vector is itself usable make reduction a slice and a rebuild; otherwise a projection fitted on the corpus can cut the size substantially for a measurable recall cost. Either way it requires re-encoding or at least re-indexing, so the practical question is whether your rebuild is fast enough to make it a routine option rather than a project.</p></details>

<details class="faq-item"><summary><span>Does a larger dimensionality help with spatial questions specifically?</span></summary><p>Not noticeably, because the spatial part of a spatial question is not carried by the embedding at all — it is carried by the metadata filter. What the embedding carries is ordinary technical prose, where the recall curve flattens early. This is one of the reasons spatial corpora tolerate smaller vectors than general-purpose retrieval systems do.</p></details>

<details class="faq-item"><summary><span>How does dimensionality interact with the graph index parameters?</span></summary><p>Additively in memory and mildly in recall. The link structure costs the same regardless of vector width, so at small dimensionalities the links can be a substantial share of the index, which is worth knowing when a sweep shows memory barely falling below 384. Recall interacts less than expected: the connectivity that a filtered workload needs is driven more by filter selectivity than by vector width.</p></details>

<details class="faq-item"><summary><span>Should the choice be the same across environments?</span></summary><p>Yes, and it should be recorded in the index manifest so a mismatch is detectable. A staging environment running a smaller dimensionality to save memory produces recall numbers that do not transfer, and the difference is invisible in every dashboard until someone compares them directly.</p></details>

<details class="faq-item"><summary><span>Is it worth running the sweep on a subset rather than the whole corpus?</span></summary><p>Yes, and a few hundred thousand chunks is usually enough provided the subset is clustered the way the corpus is. Encoding a hundred million chunks four times to compare dimensionalities is a week of compute to answer a question a subset answers in an afternoon. What the subset cannot tell you is absolute memory, so compute that from the full corpus size using the model rather than extrapolating the measured index.</p></details>

<details class="faq-item"><summary><span>Should the sweep include a candidate below 256 dimensions?</span></summary><p>Include one if the corpus is very large, because the curve is steep down there and the memory saving is substantial — and expect it to fail the recall target. Its value is in showing where the curve turns rather than in being a viable option, and a sweep whose lowest point still meets the target has not established that the chosen size is the smallest that does.</p></details>

## Related

- Up to the parent topic: [Spatial Embedding Models](/spatial-llm-architecture-core-concepts/spatial-embedding-models/)
- [Benchmarking Spatial Embedding Models for Vector GIS](/spatial-llm-architecture-core-concepts/spatial-embedding-models/benchmarking-spatial-embedding-models-for-vector-gis/)
- Related technique: [Sizing HNSW Parameters for Spatial Recall](/geospatial-rag-pipelines/spatial-vector-store-selection/sizing-hnsw-parameters-for-spatial-recall/)
- Related topic: [Spatial Vector Store Selection](/geospatial-rag-pipelines/spatial-vector-store-selection/)
