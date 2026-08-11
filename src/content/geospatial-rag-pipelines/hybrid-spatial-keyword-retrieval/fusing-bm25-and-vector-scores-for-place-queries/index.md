---
title: Fusing Keyword and Vector Scores for Place Queries
description: Combine a lexical ranking with a dense ranking using reciprocal rank fusion, so rare toponyms and reference codes survive alongside semantic similarity.
slug: fusing-bm25-and-vector-scores-for-place-queries
type: howto
breadcrumb: Fusing Keyword and Vector Scores
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Fusing Keyword and Vector Scores for Place Queries

Two rankings arrive from two systems with incompatible score scales, and one number has to come out. Getting that combination right is what makes a rare place name findable without destroying the semantic behaviour that handles everything else. This guide implements the fusion step for [hybrid spatial and keyword retrieval](/geospatial-rag-pipelines/hybrid-spatial-keyword-retrieval/), and explains why rank-based fusion is the default worth beating.

## When to Use This Approach

Fuse when both halves genuinely contribute — that is, when your query mix contains both identifier-shaped and description-shaped questions. If every query is one or the other, run the appropriate half and skip the machinery.

| Situation | Approach | Reason |
|-----------|----------|--------|
| Mixed query set, no labelled data | Reciprocal rank fusion | One parameter, no normalisation to get wrong |
| Mixed set, good labelled data | Weighted rank fusion | Weights per query class, fitted and refitted |
| Scores on a shared, stable scale | Score fusion | Rare, but strictly more information |
| One half contributes nothing | Drop it | Fusion cannot rescue a broken index |

Rank fusion discards score magnitude, which sounds like a loss and usually is not. Magnitudes from a lexical index depend on corpus statistics that shift as the corpus grows, so a fusion tuned on them silently drifts; positions do not.

<figure class="diagram">
<svg viewBox="46 9 668 235" role="img" aria-labelledby="fbv-scale-t fbv-scale-d" xmlns="http://www.w3.org/2000/svg"><title id="fbv-scale-t">Why raw score addition lets one half dominate</title><desc id="fbv-scale-d">Lexical scores span a wide unbounded range while cosine similarities cluster in a narrow band, so adding them directly makes the lexical spread decide almost every ordering.</desc><rect x="46" y="9" width="668" height="235" fill="#ffffff"/><text x="390" y="34" fill="#5b6471" font-size="13" text-anchor="middle">Typical spread of each score family across one result set</text><rect x="60" y="60" width="640" height="56" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="380" y="94" fill="#1f2937" font-size="12.5" text-anchor="middle">lexical relevance: 0.4 to 31.7 — unbounded, corpus-dependent</text><rect x="300" y="132" width="160" height="56" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="380" y="166" fill="#1f2937" font-size="12.5" text-anchor="middle">cosine: 0.71 to 0.86</text><text x="150" y="166" fill="#5b6471" font-size="12" text-anchor="middle">nothing here</text><text x="620" y="166" fill="#5b6471" font-size="12" text-anchor="middle">nothing here</text><text x="390" y="226" fill="#1f2937" font-size="13" text-anchor="middle">Added directly, the wide bar decides the order and the narrow one is noise</text></svg>
<figcaption><b>The variance, not the mean, is what dominates.</b> Normalising each family per query fixes this, and needs recomputing whenever either index changes; rank fusion sidesteps it entirely.</figcaption>
</figure>

## Implementation

The fusion takes any number of ranked identifier lists and combines them by summed reciprocal rank, with optional per-list weights supplied by the query classifier.

```python
import logging
from dataclasses import dataclass
from typing import Sequence

log = logging.getLogger("rank_fusion")

RRF_K = 60          # damping constant: larger flattens the advantage of rank one


@dataclass(frozen=True)
class Fused:
    doc_id: str
    score: float
    positions: tuple[tuple[str, int | None], ...]     # (list name, 1-based rank or None)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    names: Sequence[str],
    weights: Sequence[float] | None = None,
    k: int = RRF_K,
) -> list[Fused]:
    """Fuse ranked id lists. Never raises on an empty or partial input."""
    if len(rankings) != len(names):
        raise ValueError("each ranking needs a name for provenance")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")
    if k <= 0:
        raise ValueError("k must be positive")

    scores: dict[str, float] = {}
    where: dict[str, dict[str, int]] = {}
    for ranking, name, weight in zip(rankings, names, weights):
        if not ranking:
            log.info("ranking %r is empty — fusion continues without it", name)
            continue
        for position, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + position)
            where.setdefault(doc_id, {})[name] = position

    out = [
        Fused(doc_id, round(score, 8),
              tuple((n, where.get(doc_id, {}).get(n)) for n in names))
        for doc_id, score in scores.items()
    ]
    # Deterministic ordering: score first, then id, so equal scores never depend
    # on dictionary insertion order.
    out.sort(key=lambda f: (-f.score, f.doc_id))
    return out
```

Three details are load-bearing. Empty rankings are skipped rather than treated as a failure, so a lexical index outage degrades the system to dense-only instead of emptying it. Positions are retained per document, which turns "why did this rank here" into a lookup rather than an investigation. And the sort has an explicit tiebreaker, because equal fused scores are common — two documents each ranked third by one half and absent from the other tie exactly — and without one the order depends on hash iteration.

The damping constant deserves a moment. With `k = 60`, rank one contributes 1/61 and rank ten contributes 1/70, a ratio of about 1.15; with `k = 1`, the same pair is 1/2 against 1/11, a ratio of 5.5. The small constant makes a single half's top result nearly unbeatable, which defeats the purpose of consulting two.

<figure class="diagram">
<svg viewBox="66 7 654 237" role="img" aria-labelledby="fbv-k-t fbv-k-d" xmlns="http://www.w3.org/2000/svg"><title id="fbv-k-t">How the damping constant changes the weight of a top rank</title><desc id="fbv-k-d">Contribution by rank position for a small and a large damping constant, showing that a small constant makes rank one dominate while a large one keeps ranks one to ten comparable.</desc><rect x="66" y="7" width="654" height="237" fill="#ffffff"/><text x="380" y="32" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">Contribution by rank, for two damping constants</text><g fill="#fdeaee" stroke="#b3324f" stroke-width="2"><rect x="80" y="60" width="50" height="112" rx="4"/><rect x="210" y="128" width="50" height="44" rx="4"/><rect x="340" y="150" width="50" height="22" rx="4"/><rect x="470" y="158" width="50" height="14" rx="4"/><rect x="600" y="162" width="50" height="10" rx="4"/></g><g fill="#e4f5ec" stroke="#12805c" stroke-width="2"><rect x="136" y="70" width="50" height="102" rx="4"/><rect x="266" y="74" width="50" height="98" rx="4"/><rect x="396" y="78" width="50" height="94" rx="4"/><rect x="526" y="82" width="50" height="90" rx="4"/><rect x="656" y="86" width="50" height="86" rx="4"/></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="133" y="194">rank 1</text><text x="263" y="194">rank 3</text><text x="393" y="194">rank 6</text><text x="523" y="194">rank 10</text><text x="653" y="194">rank 15</text></g><text x="380" y="226" fill="#1f2937" font-size="12.5" text-anchor="middle">Left bar of each pair: small constant. Right bar: the conventional value.</text></svg>
<figcaption><b>The constant decides how much agreement is worth.</b> A large value rewards documents that both halves place reasonably; a small one lets whichever half is most confident win outright, which is the behaviour hybrid retrieval exists to avoid.</figcaption>
</figure>

## Validation & Testing

```python
def test_agreement_beats_a_single_first_place():
    lexical = ["doc-a", "doc-c", "doc-d"]
    dense = ["doc-b", "doc-c", "doc-a"]
    top = reciprocal_rank_fusion([lexical, dense], ["lexical", "dense"])[0]
    assert top.doc_id == "doc-c"


def test_empty_ranking_degrades_rather_than_empties():
    fused = reciprocal_rank_fusion([[], ["doc-b"]], ["lexical", "dense"])
    assert [f.doc_id for f in fused] == ["doc-b"]


def test_ties_are_broken_deterministically():
    a = reciprocal_rank_fusion([["x", "y"], ["y", "x"]], ["l", "d"])
    b = reciprocal_rank_fusion([["y", "x"], ["x", "y"]], ["l", "d"])
    assert [f.doc_id for f in a] == [f.doc_id for f in b]


def test_positions_are_recorded_for_every_result():
    fused = reciprocal_rank_fusion([["x"], ["y"]], ["lexical", "dense"])
    for f in fused:
        assert dict(f.positions).keys() == {"lexical", "dense"}
```

The first test encodes the property that justifies fusion at all, and it is the one to point at when somebody proposes replacing the whole thing with "just use the lexical result when the query has a code in it".

Run the fusion against a frozen pair of ranking fixtures in continuous integration rather than against live indexes. The fusion is pure and deterministic; testing it through two live search systems tests those systems instead and makes failures ambiguous.

## Gotchas & Edge Cases

**Rankings of different lengths.** A lexical half returning 40 results and a dense half returning 400 gives the dense half more opportunities to accumulate score. Truncate both to the same depth before fusing, or the imbalance becomes an implicit weight nobody chose.

**Identifiers that do not match.** If the two indexes key on different identifiers — one on chunk, one on document — fusion silently produces two disjoint sets and the output is just the concatenation. Assert an overlap in testing; zero overlap on a query both halves answered is a schema bug, not a ranking outcome.

<figure class="diagram">
<svg viewBox="46 46 648 188" role="img" aria-labelledby="fbv-len-t fbv-len-d" xmlns="http://www.w3.org/2000/svg"><title id="fbv-len-t">Unequal ranking depths acting as a hidden weight</title><desc id="fbv-len-d">One ranking truncated at forty results and another at four hundred gives the longer list ten times as many chances to contribute score, an imbalance nobody configured.</desc><rect x="46" y="46" width="648" height="188" fill="#ffffff"/><rect x="60" y="60" width="120" height="52" rx="6" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="60" y="132" width="620" height="52" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="120" y="92" fill="#1f2937" font-size="12.5" text-anchor="middle">lexical: 40</text><text x="370" y="164" fill="#1f2937" font-size="12.5" text-anchor="middle">dense: 400 — ten times the opportunities to accumulate score</text><text x="380" y="216" fill="#1f2937" font-size="13" text-anchor="middle">Truncate both to one depth before fusing, or the depths become the weights</text></svg>
<figcaption><b>The imbalance is invisible in the configuration.</b> Nothing in the weight settings records it, so a fusion that was carefully tuned at equal depths quietly changes behaviour the day one retriever's limit is raised.</figcaption>
</figure>

**Weights applied to the wrong list.** The weights and the rankings are two parallel sequences, and swapping them is invisible: results are still returned, just reliably worse. Pass names alongside, as above, and log the pairing at startup.

**Fusion depth larger than the reranker's budget.** Fusing a thousand candidates and then cross-encoding the top twenty wastes most of the fusion. Fuse to roughly the depth you intend to inspect, plus a margin.

**Duplicate identifiers within one ranking.** A store that returns the same document twice at different positions will have its contribution counted twice. Deduplicate each ranking, keeping the best position, before fusing.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Can a third ranking be added — for example freshness?</span></summary><p>Yes, and rank fusion extends to it naturally, but be careful about what a rank means for a non-relevance signal. Ordering documents by date produces a ranking where position one is simply the newest, not the most relevant, so its reciprocal contribution rewards recency uniformly regardless of whether recency matters for the query. Freshness is usually better applied as a multiplier after fusion, or as a filter before it.</p></details>

<details class="faq-item"><summary><span>How deep should each ranking go?</span></summary><p>Between one hundred and four hundred for most workloads. Too shallow and a document that one half ranks poorly can never be rescued by the other, which is the whole point; too deep and you are fusing noise, plus paying to retrieve it. If the two halves disagree strongly on your query mix, deeper helps; if they mostly agree, it does not.</p></details>

<details class="faq-item"><summary><span>Should the geographic filter be a third ranking?</span></summary><p>No. Distance is a constraint on the population and, separately, a ranking signal applied afterwards — mixing it in as a third fusion input means a strong lexical match can outvote geography, which is precisely the leak the filter exists to prevent. Filter first, fuse the two text rankings, then apply proximity in the reranking stage.</p></details>

<details class="faq-item"><summary><span>Is it worth normalising scores anyway, to compare against rank fusion?</span></summary><p>Once, as an experiment, on a labelled set. Score fusion with per-query normalisation does sometimes win, and knowing by how much tells you whether the maintenance is worth it. What is not worth doing is shipping score fusion without that measurement, because its failure mode — one half quietly dominating after a corpus change — produces no error and degrades slowly.</p></details>

One operational note. Log the fused score alongside both positions for the results that actually reach the model, not for the whole fused list. The full list is large and mostly uninteresting; the handful that were used are the ones someone will ask about, and having their provenance already recorded turns a retrieval complaint into a five-minute lookup instead of a reproduction attempt against indexes that have since changed.

Store that log with the query text and the classifier's label, too. Most fusion complaints turn out to be classification complaints — the query was weighted as descriptive when the user meant it as a lookup — and without the label recorded the two are indistinguishable after the fact.

<details class="faq-item"><summary><span>What happens to a document that only one half returns?</span></summary><p>It gets exactly one contribution and therefore ranks below anything both halves returned at a comparable position — which is the intended behaviour, and the reason a single half's top result does not automatically win. If you find that single-half documents are systematically the right answers for some query class, that is evidence the other half is failing on that class, and the fix is in that index rather than in the fusion.</p></details>

## Related

- Up to the parent topic: [Hybrid Spatial and Keyword Retrieval](/geospatial-rag-pipelines/hybrid-spatial-keyword-retrieval/)
- [Tuning Fusion Weights for Toponym-Heavy Queries](/geospatial-rag-pipelines/hybrid-spatial-keyword-retrieval/tuning-fusion-weights-for-toponym-heavy-queries/)
- Technique: [Reranking Spatial Results by Distance and Relevance](/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/reranking-spatial-results-by-distance-and-relevance/)
- Concept: [Geocoding and Place-Name Resolution](/spatial-llm-architecture-core-concepts/geocoding-and-place-name-resolution/)
