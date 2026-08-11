---
title: Inferring CRS from Retrieved Spatial Context
description: Turn retrieved neighbours into a scored coordinate reference system hypothesis using magnitude, precision, provenance and geographic agreement — with a flagged fallback.
slug: inferring-crs-from-retrieved-spatial-context
type: howto
breadcrumb: Inferring CRS from Context
datePublished: 2025-03-19
dateModified: 2026-08-11
---

# Inferring CRS from Retrieved Spatial Context

Retrieval gives you neighbours; it does not give you an answer. A handful of documents that mention the same place, each with its own coordinates and its own declared frame, have to be turned into one scored hypothesis about the reference system of the mention in hand. This guide sets out the four signals that do that work and how to combine them without manufacturing false confidence, as the scoring stage of [retrieval-augmented CRS resolution](/geospatial-rag-pipelines/retrieval-augmented-crs-resolution/).

## When to Use This Approach

Use inference when a mention arrives with coordinates but no declared frame, and there is a corpus with enough overlap to supply evidence. Do not use it when the frame is declared — validate the declaration instead — and do not use it when the corpus covers a different region from the mention, where retrieved neighbours are noise wearing the clothes of evidence.

| Signal | What it rules out | Strength alone |
|--------|-------------------|----------------|
| Coordinate magnitude | Frames whose axes cannot hold these numbers | Weak — many frames overlap |
| Decimal precision | Degrees against metres | Moderate — six decimals implies degrees |
| Neighbour agreement | Frames no nearby document uses | Moderate — corpora are biased |
| Declared provenance | Everything but the source's own frame | Strong — when present |

None of the four is decisive on its own, which is the whole difficulty. Magnitude narrows a registry of thousands to a few dozen; precision usually separates geographic from projected; agreement and provenance do the rest. A design that leans on any single one produces confident answers exactly where the evidence is weakest.

<figure class="diagram">
<svg viewBox="16 52 748 178" role="img" aria-labelledby="icrs-sig-t icrs-sig-d" xmlns="http://www.w3.org/2000/svg"><title id="icrs-sig-t">Four signals narrowing the candidate set</title><desc id="icrs-sig-d">Magnitude narrows thousands of frames to dozens, precision separates geographic from projected, neighbour agreement narrows to a handful and provenance selects one, with confidence rising at each stage.</desc><rect x="16" y="52" width="748" height="178" fill="#ffffff"/><rect x="30" y="66" width="170" height="106" rx="8" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="220" y="76" width="170" height="86" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="410" y="86" width="170" height="66" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="600" y="96" width="150" height="46" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="12.5" text-anchor="middle" font-weight="600"><text x="115" y="104">magnitude</text><text x="305" y="110">precision</text><text x="495" y="114">agreement</text><text x="675" y="118">provenance</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="115" y="132">thousands to dozens</text><text x="305" y="134">degrees or metres</text><text x="495" y="136">dozens to a few</text><text x="675" y="138">a few to one</text></g><g stroke="#5b6471" stroke-width="2" marker-end="url(#icrs-sig-a)"><line x1="202" y1="119" x2="216" y2="119"/><line x1="392" y1="119" x2="406" y2="119"/><line x1="582" y1="119" x2="596" y2="119"/></g><defs><marker id="icrs-sig-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/></marker></defs><text x="390" y="212" fill="#1f2937" font-size="13" text-anchor="middle">Confidence should rise across this chain, never before it</text></svg>
<figcaption><b>Each stage is a filter, not a vote.</b> Scoring before the candidate set is narrowed produces high numbers early, which is how a magnitude coincidence ends up presented as a determination.</figcaption>
</figure>

## Implementation

The scorer takes a coordinate pair and a set of retrieved neighbours, applies the four signals in order, and returns a ranked list with an explicit rationale for each candidate.

```python
import logging
import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from pyproj import CRS
from pyproj.exceptions import CRSError

log = logging.getLogger("crs_inference")


@dataclass(frozen=True)
class Neighbour:
    epsg: int
    declared: bool          # the source stated this frame, rather than it being assumed
    similarity: float       # retrieval score in [0, 1]


@dataclass(frozen=True)
class Hypothesis:
    epsg: int
    score: float
    rationale: str


def _looks_geographic(x: float, y: float, decimals: int) -> bool:
    """Degrees are small and precise; projected metres are large and blunt."""
    in_range = -180.0 <= x <= 180.0 and -90.0 <= y <= 90.0
    return in_range and decimals >= 4


def _magnitude_ok(epsg: int, x: float, y: float) -> bool:
    """Reject frames whose own axis ranges cannot hold these numbers."""
    try:
        crs = CRS.from_epsg(epsg)
    except CRSError:
        return False
    if crs.is_geographic:
        return -180.0 <= x <= 180.0 and -90.0 <= y <= 90.0
    # Projected frames: reject values no plausible grid uses.
    return abs(x) < 1.0e7 and abs(y) < 1.0e7 and (abs(x) > 1000.0 or abs(y) > 1000.0)


def infer_crs(
    x: float,
    y: float,
    decimals: int,
    neighbours: Sequence[Neighbour],
) -> list[Hypothesis]:
    """Rank candidate frames for an undeclared coordinate pair. Never raises."""
    if not all(math.isfinite(v) for v in (x, y)):
        log.warning("non-finite coordinate (%r, %r) — no hypothesis possible", x, y)
        return []

    geographic = _looks_geographic(x, y, decimals)
    counts = Counter(n.epsg for n in neighbours)
    declared = {n.epsg for n in neighbours if n.declared}
    best_sim = {n.epsg: 0.0 for n in neighbours}
    for n in neighbours:
        best_sim[n.epsg] = max(best_sim[n.epsg], n.similarity)

    out: list[Hypothesis] = []
    for epsg, count in counts.items():
        if not _magnitude_ok(epsg, x, y):
            continue                                   # magnitude gate: hard exclusion
        try:
            crs = CRS.from_epsg(epsg)
        except CRSError:
            continue
        if crs.is_geographic != geographic:
            continue                                   # precision gate: hard exclusion

        agreement = min(0.30, 0.10 * count)            # capped: three neighbours, then no more
        provenance = 0.35 if epsg in declared else 0.0
        similarity = 0.25 * best_sim[epsg]
        score = round(min(1.0, 0.10 + agreement + provenance + similarity), 3)
        reason = (f"{count} neighbour(s), "
                  f"{'declared' if provenance else 'assumed'} frame, "
                  f"{'geographic' if geographic else 'projected'} magnitudes")
        out.append(Hypothesis(epsg, score, reason))

    return sorted(out, key=lambda h: h.score, reverse=True)
```

The two gates are exclusions rather than penalties, and that asymmetry is deliberate. A frame whose axes cannot hold the numbers is not unlikely, it is impossible, and letting a strong provenance score outweigh an impossibility is how a pipeline produces answers that no amount of downstream validation can rescue.

The floor of 0.10 on the score means even the best-supported hypothesis tops out below the fallback threshold used in the parent topic unless it has both provenance and agreement. That is the intended behaviour: inference from context should rarely reach high confidence on its own, and a design where it usually does is one that has stopped distinguishing evidence from coincidence.

<figure class="diagram">
<svg viewBox="16 36 728 178" role="img" aria-labelledby="icrs-prec-t icrs-prec-d" xmlns="http://www.w3.org/2000/svg"><title id="icrs-prec-t">Decimal precision as a separator between degrees and metres</title><desc id="icrs-prec-d">Coordinate pairs grouped by magnitude and decimal count. Small values with many decimals indicate degrees; large values with few decimals indicate projected metres; small values with no decimals are genuinely ambiguous.</desc><rect x="16" y="36" width="728" height="178" fill="#ffffff"/><rect x="30" y="50" width="220" height="100" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="270" y="50" width="220" height="100" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="510" y="50" width="220" height="100" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="140" y="80">-3.192834, 55.946</text><text x="380" y="80">325612, 673481</text><text x="620" y="80">412, 388</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="140" y="106">six decimals, in range</text><text x="140" y="128">geographic degrees</text><text x="380" y="106">six figures, no decimals</text><text x="380" y="128">projected metres</text><text x="620" y="106">small and blunt</text><text x="620" y="128">genuinely ambiguous</text></g><text x="380" y="196" fill="#1f2937" font-size="13" text-anchor="middle">The third case is the one to flag rather than resolve</text></svg>
<figcaption><b>Precision is evidence, not proof.</b> An exporter that rounds degrees to two decimals produces exactly the ambiguous third case, which is why the gate keys on range and precision together rather than on either alone.</figcaption>
</figure>

## Validation & Testing

```python
def test_impossible_frame_is_excluded_not_ranked_low():
    # Geographic-looking coordinates cannot belong to a projected grid.
    hyps = infer_crs(-3.19, 55.95, 6, [Neighbour(27700, True, 0.95)])
    assert all(h.epsg != 27700 for h in hyps)


def test_declared_provenance_outranks_bare_agreement():
    ns = [Neighbour(4326, True, 0.5), Neighbour(4258, False, 0.5), Neighbour(4258, False, 0.5)]
    top = infer_crs(-3.19, 55.95, 6, ns)[0]
    assert top.epsg == 4326


def test_no_neighbours_yields_no_hypothesis():
    assert infer_crs(-3.19, 55.95, 6, []) == []


def test_non_finite_input_returns_empty_not_raises():
    assert infer_crs(float("nan"), 55.95, 6, [Neighbour(4326, True, 1.0)]) == []
```

The second test encodes the ranking policy explicitly, which matters because it is the assertion that fails when someone raises the agreement cap to "make the scores more decisive". The third is equally important and easy to omit: a scorer that returns a hypothesis with no evidence at all is worse than one that returns nothing, because the caller cannot tell the two apart from the shape of the result.

## Gotchas & Edge Cases

**Corpus bias masquerading as agreement.** If nine tenths of the corpus uses one frame, neighbour agreement will favour it for every mention, including the ones that belong to something else. Cap the agreement contribution, as the code does, and consider normalising by the frame's overall prevalence in the corpus so a common frame has to earn its lead.

<figure class="diagram">
<svg viewBox="46 9 714 225" role="img" aria-labelledby="icrs-bias-t icrs-bias-d" xmlns="http://www.w3.org/2000/svg"><title id="icrs-bias-t">Corpus bias inflating neighbour agreement</title><desc id="icrs-bias-d">A corpus dominated by one frame supplies most neighbours in that frame regardless of the mention, so raw agreement counts favour it everywhere until the contribution is capped and normalised by prevalence.</desc><rect x="46" y="9" width="714" height="225" fill="#ffffff"/><text x="380" y="34" fill="#5b6471" font-size="13" text-anchor="middle">Neighbours returned for a mention that genuinely belongs to the rarer frame</text><rect x="60" y="60" width="400" height="52" rx="6" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="60" y="124" width="100" height="52" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="260" y="92" fill="#1f2937" font-size="12.5" text-anchor="middle">eight neighbours in the corpus-dominant frame</text><text x="110" y="156" fill="#1f2937" font-size="12.5" text-anchor="middle">two correct</text><text x="530" y="92" fill="#5b6471" font-size="12">raw count says: dominant</text><text x="530" y="156" fill="#5b6471" font-size="12">capped count says: unresolved</text><text x="380" y="216" fill="#1f2937" font-size="13" text-anchor="middle">Cap the agreement term, then normalise by how common the frame is overall</text></svg>
<figcaption><b>Agreement measures the corpus, not the mention.</b> Capping the contribution keeps a dominant frame from winning by weight of numbers alone, which is the failure that makes inference look reliable in testing and behave badly in production.</figcaption>
</figure>

**Northing and easting confusion.** A pair swapped at export looks like a plausible position in the same frame, and no signal here detects it. The check that does is geographic: transform to geographic coordinates and ask whether the result falls in the region the document is about. This is the same class of error covered in axis-order handling for declared frames.

**Precision inflation from reprojection.** A projected coordinate reprojected to degrees acquires many decimals that carry no real accuracy, so a document full of six-decimal degrees may be a reprojection of a metre-precision survey. The signal remains useful for typing the frame and should not be read as evidence about the data's accuracy.

**Frames that are equivalent in practice.** Several codes describe datums that differ by centimetres, and choosing between them from retrieved context is not possible. Group such codes and return the group with a note, rather than picking one and implying a distinction the evidence cannot support.

**Neighbours retrieved from the mention's own document.** Self-agreement is not evidence. Exclude neighbours that share a source identifier with the mention before counting, or a single document with a repeated frame will reliably reach the agreement cap on its own.

## Frequently Asked Questions

<details class="faq-item"><summary><span>How many neighbours should be retrieved?</span></summary><p>Enough to see disagreement, which in practice means eight to twelve. Fewer than five and a single unusual source dominates; more than about fifteen and you are mostly adding weakly related documents that inflate the agreement count without adding information. Since the agreement contribution is capped anyway, the marginal value of the tenth neighbour is close to zero.</p></details>

<details class="faq-item"><summary><span>Can a model be asked to do this inference directly?</span></summary><p>It can be asked and it will answer, which is the problem. Frame inference is exactly the task where a plausible-sounding answer is indistinguishable from a correct one without the deterministic checks above, and the model has no way to run them. Use it, if at all, to rank a shortlist the gates have already produced — never to generate the candidates.</p></details>

<details class="faq-item"><summary><span>What should the caller do with two hypotheses of equal score?</span></summary><p>Treat it as an unresolved case and fall back with a flag, recording both. A tie means the evidence does not separate them, and picking the first is arbitrary in a way that will be wrong roughly half the time. If the two are members of an equivalent-datum group, say so in the note — that is a different situation from two genuinely different candidate frames.</p></details>

<details class="faq-item"><summary><span>Does the scoring need to be recomputed when the corpus changes?</span></summary><p>Yes, and this is an argument for caching resolutions with a corpus version rather than indefinitely. Neighbour agreement is a property of the corpus at a moment in time; ingesting a large new source can change which frame dominates a region and therefore change what the same mention would resolve to. Recording the corpus version alongside the resolution makes that change visible instead of confusing.</p></details>

## Related

- Up to the parent topic: [Retrieval-Augmented CRS Resolution](/geospatial-rag-pipelines/retrieval-augmented-crs-resolution/)
- [Resolving Ambiguous EPSG Codes from Document Context](/geospatial-rag-pipelines/retrieval-augmented-crs-resolution/resolving-ambiguous-epsg-codes-from-document-context/)
- Concept: [Coordinate Reference System Normalization](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/)
- Technique: [Detecting Axis-Order Swaps in Coordinate Input](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/detecting-axis-order-swaps-in-coordinate-input/)
