---
title: Tuning Fusion Weights for Toponym-Heavy Queries
description: Fit per-class fusion weights against a labelled query set, verify each half still earns its place, and keep the tuning from overfitting a handful of demonstration queries.
slug: tuning-fusion-weights-for-toponym-heavy-queries
type: howto
breadcrumb: Tuning Fusion Weights
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Tuning Fusion Weights for Toponym-Heavy Queries

Equal weights are a reasonable starting point and a poor destination. Queries dominated by place names want the lexical half to lead; descriptive questions want the dense half to. This guide fits those weights against a labelled set, checks that the fitting has not simply memorised it, and establishes when to stop — the tuning counterpart to [fusing keyword and vector scores](/geospatial-rag-pipelines/hybrid-spatial-keyword-retrieval/fusing-bm25-and-vector-scores-for-place-queries/) within [hybrid spatial and keyword retrieval](/geospatial-rag-pipelines/hybrid-spatial-keyword-retrieval/).

## When to Use This Approach

Tune once you have at least a few dozen labelled queries per class and a measurable gap between fused performance and the better single half. Before that, equal weights with a query classifier are more robust than fitted ones.

| Labelled queries per class | Approach | Risk |
|----------------------------|----------|------|
| Fewer than 20 | Keep equal weights | Fitting memorises the examples |
| 20 to 100 | Coarse grid, three classes | Moderate; validate on a held-out split |
| Over 100 | Finer grid, per-class | Low, if the split is honest |
| Growing continuously | Refit on a schedule | Drift is the main hazard |

The number that matters is per class, not in total. Two hundred queries that are all descriptive tell you nothing about how to weight an identifier lookup, and a fit over the pooled set will confidently produce weights that serve the majority class and fail the minority.

<figure class="diagram">
<svg viewBox="46 9 688 225" role="img" aria-labelledby="tfw-split-t tfw-split-d" xmlns="http://www.w3.org/2000/svg"><title id="tfw-split-t">Why weights must be fitted per query class</title><desc id="tfw-split-d">A single pooled fit lands between the two class optima and serves neither well; per-class fits sit at each optimum, at the cost of needing a classifier at query time.</desc><rect x="46" y="9" width="688" height="225" fill="#ffffff"/><text x="390" y="34" fill="#5b6471" font-size="13" text-anchor="middle">Best lexical weight, by query class</text><rect x="60" y="60" width="200" height="70" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="160" y="88" fill="#1f2937" font-size="12.5" text-anchor="middle">identifier queries</text><text x="160" y="112" fill="#5b6471" font-size="12" text-anchor="middle">best at weight 2.0</text><rect x="520" y="60" width="200" height="70" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="620" y="88" fill="#1f2937" font-size="12.5" text-anchor="middle">descriptive queries</text><text x="620" y="112" fill="#5b6471" font-size="12" text-anchor="middle">best at weight 0.6</text><rect x="290" y="150" width="200" height="70" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="390" y="178" fill="#1f2937" font-size="12.5" text-anchor="middle">one pooled fit: 1.2</text><text x="390" y="202" fill="#5b6471" font-size="12" text-anchor="middle">below both optima, on both classes</text><g stroke="#5b6471" stroke-width="2" marker-end="url(#tfw-split-a)"><line x1="200" y1="132" x2="330" y2="146"/><line x1="580" y1="132" x2="450" y2="146"/></g><defs><marker id="tfw-split-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/></marker></defs></svg>
<figcaption><b>The average of two right answers is a wrong answer.</b> Pooled fitting is not a conservative compromise; it produces a weight that is optimal for a query mix nobody actually sends.</figcaption>
</figure>

## Implementation

The fit is a small grid search per class, scored on recall at a fixed depth, with a held-out split so the reported number is not the one the weights were chosen on.

```python
import logging
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

log = logging.getLogger("fusion_tuning")

GRID = (0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5)


@dataclass(frozen=True)
class Labelled:
    query: str
    query_class: str
    relevant: frozenset[str]


@dataclass(frozen=True)
class Fit:
    query_class: str
    w_lexical: float
    w_dense: float
    train_recall: float
    holdout_recall: float
    n_train: int
    n_holdout: int


def _recall_at(fused_ids: Sequence[str], relevant: frozenset[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(fused_ids[:k]) & relevant) / len(relevant)


def fit_class(
    examples: Sequence[Labelled],
    lexical_search: Callable[[str], Sequence[str]],
    dense_search: Callable[[str], Sequence[str]],
    fuse: Callable[[Sequence[Sequence[str]], Sequence[float]], Sequence[str]],
    k: int = 10,
    holdout_fraction: float = 0.3,
) -> Fit | None:
    """Grid-search weights for one query class, reporting held-out recall."""
    if len(examples) < 8:
        log.info("only %d example(s) for %s — keeping equal weights",
                 len(examples), examples[0].query_class if examples else "?")
        return None                                   # too few to fit: refuse, do not guess

    ordered = sorted(examples, key=lambda e: e.query)   # deterministic split, no shuffling
    cut = max(1, int(len(ordered) * (1 - holdout_fraction)))
    train, holdout = ordered[:cut], ordered[cut:]

    # Retrieve once per query; the rankings do not depend on the weights.
    cache = {}
    for ex in ordered:
        try:
            cache[ex.query] = (list(lexical_search(ex.query)), list(dense_search(ex.query)))
        except Exception as exc:                      # a broken half must not silently score 0
            log.warning("retrieval failed for %r: %s — dropping from the fit", ex.query, exc)

    def mean_recall(subset, w_lex, w_dense) -> float:
        scored = [
            _recall_at(fuse(cache[ex.query], (w_lex, w_dense)), ex.relevant, k)
            for ex in subset if ex.query in cache
        ]
        return sum(scored) / len(scored) if scored else 0.0

    best = max(
        ((w_lex, w_dense, mean_recall(train, w_lex, w_dense))
         for w_lex in GRID for w_dense in GRID),
        key=lambda t: (t[2], -abs(t[0] - t[1])),       # tie-break toward balance
    )
    w_lex, w_dense, train_recall = best
    return Fit(examples[0].query_class, w_lex, w_dense, round(train_recall, 4),
               round(mean_recall(holdout, w_lex, w_dense), 4), len(train), len(holdout))
```

Two guards make the difference between tuning and self-deception. Refusing to fit fewer than eight examples means a class with three demonstration queries keeps its neutral weights instead of acquiring confident nonsense. And the deterministic split — sorted, not shuffled — means rerunning the fit produces the same answer, so a change in the reported recall is a change in the data rather than in the random seed.

The tie-break toward balanced weights is a mild regularisation. When several settings score identically on the training split, the one closest to equal weights is the one least likely to be an artefact of those particular examples.

<figure class="diagram">
<svg viewBox="52 7 656 237" role="img" aria-labelledby="tfw-over-t tfw-over-d" xmlns="http://www.w3.org/2000/svg"><title id="tfw-over-t">Training against held-out recall as the grid gets finer</title><desc id="tfw-over-d">Training recall rises steadily as more weight settings are tried while held-out recall peaks and then declines, marking the point where the fit has begun memorising the training queries.</desc><rect x="52" y="7" width="656" height="237" fill="#ffffff"/><text x="380" y="32" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">Recall on the split the weights were chosen on, and on the one they were not</text><g fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"><rect x="80" y="120" width="50" height="52" rx="4"/><rect x="240" y="102" width="50" height="70" rx="4"/><rect x="400" y="88" width="50" height="84" rx="4"/><rect x="560" y="72" width="50" height="100" rx="4"/></g><g fill="#e4f5ec" stroke="#12805c" stroke-width="2"><rect x="136" y="124" width="50" height="48" rx="4"/><rect x="296" y="108" width="50" height="64" rx="4"/><rect x="456" y="112" width="50" height="60" rx="4"/><rect x="616" y="132" width="50" height="40" rx="4"/></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="133" y="194">3 settings</text><text x="293" y="194">8 settings</text><text x="453" y="194">24 settings</text><text x="613" y="194">64 settings</text></g><text x="380" y="226" fill="#1f2937" font-size="12.5" text-anchor="middle">Left bar: training split. Right bar: held out. They diverge after eight settings.</text></svg>
<figcaption><b>The divergence is the signal to stop.</b> A finer grid always improves the number you fitted on; only the held-out split can tell you whether it improved anything real.</figcaption>
</figure>

## Validation & Testing

```python
def test_small_class_refuses_to_fit():
    tiny = [Labelled(f"q{i}", "identifier", frozenset({"d"})) for i in range(4)]
    assert fit_class(tiny, lex, dense, fuse) is None


def test_fit_is_deterministic():
    a = fit_class(EXAMPLES, lex, dense, fuse)
    b = fit_class(EXAMPLES, lex, dense, fuse)
    assert (a.w_lexical, a.w_dense) == (b.w_lexical, b.w_dense)


def test_holdout_recall_is_reported_and_lower_bounded():
    fit = fit_class(EXAMPLES, lex, dense, fuse)
    assert fit.holdout_recall > 0.0
    assert fit.n_holdout >= 1


def test_fitted_weights_beat_equal_weights_on_holdout():
    fit = fit_class(EXAMPLES, lex, dense, fuse)
    equal = mean_recall_equal_weights(EXAMPLES)
    assert fit.holdout_recall >= equal, "if fitting does not help, keep equal weights"
```

The final test is the one that decides whether to ship the fit at all. If fitted weights do not beat equal weights on held-out queries, the honest conclusion is that the labelled set is too small or the classes are wrong — not that the grid needs to be finer.

## Gotchas & Edge Cases

**A labelled set built from the system's own output.** Marking the current top results as relevant fits the weights to reproduce present behaviour, which is a very effective way to make every future change look worse. Label from the corpus, ideally by someone who did not build the ranker.

**Class labels assigned by the same classifier being tuned.** If the query classifier is itself uncertain, tuning per class on its labels compounds its errors. Label the evaluation queries by hand, and measure the classifier separately.

<figure class="diagram">
<svg viewBox="16 42 728 148" role="img" aria-labelledby="tfw-label-t tfw-label-d" xmlns="http://www.w3.org/2000/svg"><title id="tfw-label-t">Where a labelled query set should come from</title><desc id="tfw-label-d">Labels drawn from the current system reproduce its behaviour; labels drawn from the corpus by an independent reader measure it. Only the second can show an improvement.</desc><rect x="16" y="42" width="728" height="148" fill="#ffffff"/><rect x="30" y="56" width="330" height="120" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="195" y="86" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">labels from current output</text><text x="195" y="116" fill="#5b6471" font-size="12" text-anchor="middle">fits the weights to today</text><text x="195" y="144" fill="#5b6471" font-size="12" text-anchor="middle">every change scores worse</text><rect x="400" y="56" width="330" height="120" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="565" y="86" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">labels from the corpus</text><text x="565" y="116" fill="#5b6471" font-size="12" text-anchor="middle">independent of the ranker</text><text x="565" y="144" fill="#5b6471" font-size="12" text-anchor="middle">improvement is measurable</text></svg>
<figcaption><b>A labelled set built from the system cannot evaluate the system.</b> It encodes the current ranking as ground truth, which makes any genuine improvement look like a regression and any regression that preserves the ordering look fine.</figcaption>
</figure>

**Recall depth chosen after seeing the results.** Fitting at depth ten and reporting at depth twenty inflates every number. Fix the depth first, from how many candidates actually reach the model, and never move it to make a comparison look better.

**Weights that drift out of the classifier's reach.** A fitted weight of 2.5 for identifier queries only helps if the classifier recognises identifier queries. Measure the pipeline end to end — classification plus fusion — not just the fusion given a perfect label.

**Refitting on every corpus change.** Weights are more stable than they feel; refitting weekly mostly tracks noise in a small labelled set. Refit when the query mix changes, when either index is rebuilt with different parameters, or when held-out recall drops — not on a calendar.

## Frequently Asked Questions

<details class="faq-item"><summary><span>How many labelled queries are really needed?</span></summary><p>Around thirty per class is where fitting starts to beat equal weights reliably, and a hundred is comfortable. Below thirty the held-out split is so small that its recall estimate swings by several points between runs, which makes it impossible to tell a real improvement from noise. Building that set is a day of work and is the single highest-value investment in this whole pipeline.</p></details>

<details class="faq-item"><summary><span>Should weights be constrained to sum to a constant?</span></summary><p>Not for rank fusion, where only the ratio matters — scaling both weights by the same factor leaves the ordering identical. Constraining them anyway is harmless and makes the grid smaller, which is a reasonable reason to do it. What matters far more is that the ratio is reported and stored, since that is the number that carries the meaning.</p></details>

<details class="faq-item"><summary><span>What if one class has no examples at all?</span></summary><p>Give it equal weights and log every query that lands in it, so the class either accumulates examples or turns out to be empty in practice. A class defined in the classifier but never observed is a maintenance cost with no benefit, and discovering that is worth as much as tuning the classes that do exist.</p></details>

<details class="faq-item"><summary><span>Can the weights be learned continuously from user behaviour?</span></summary><p>In principle, from clicks or from which retrieved documents an answer actually cited. In practice this is a bigger project than it appears: behavioural signals are biased toward whatever is currently ranked highly, so a naive feedback loop reinforces the existing weights. If you pursue it, hold out a slice of traffic with fixed weights as a control, or the loop has no reference point.</p></details>

A last piece of process advice: store the fitted weights, the labelled set version, the grid, and the held-out recall together as one artefact, and require that artefact to be referenced whenever the weights change. Weights that appear in a configuration file with no provenance become untouchable — nobody knows what they were fitted against, so nobody dares move them, and the tuning that was meant to be routine becomes a one-off from which the system never recovers.

Treat the labelled set itself as versioned data rather than as a fixture. Queries get added, labels get corrected, and a recall figure quoted without the set version it was measured against is not comparable to any other figure, including the one you measured last month.

<details class="faq-item"><summary><span>Do the weights need to be revisited when the embedding model changes?</span></summary><p>Always, and before the new model ships rather than after. A new embedding changes the dense half's score distribution and often its failure modes, so weights fitted against the old one are fitted against a system that no longer exists. Rerunning the fit is cheap once the labelled set exists, which is another reason to treat that set as the durable asset and the weights as a derived value.</p></details>

## Related

- Up to the parent topic: [Hybrid Spatial and Keyword Retrieval](/geospatial-rag-pipelines/hybrid-spatial-keyword-retrieval/)
- [Fusing Keyword and Vector Scores for Place Queries](/geospatial-rag-pipelines/hybrid-spatial-keyword-retrieval/fusing-bm25-and-vector-scores-for-place-queries/)
- Technique: [Setting Release Thresholds for Spatial Agents](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/setting-release-thresholds-for-spatial-agents/)
- Concept: [Evaluation and Benchmarking for Spatial LLMs](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/)
