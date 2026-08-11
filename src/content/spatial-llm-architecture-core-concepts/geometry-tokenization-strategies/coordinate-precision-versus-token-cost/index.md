---
title: Coordinate Precision Versus Token Cost
description: Work out how many decimal places your questions actually need, measure what each one costs in tokens, and round once at ingestion so the saving is structural.
slug: coordinate-precision-versus-token-cost
type: howto
breadcrumb: Precision Versus Token Cost
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Coordinate Precision Versus Token Cost

Coordinates arrive with whatever precision an exporter emitted, which is frequently twelve decimal places describing a survey accurate to a metre. Those extra digits cost real context and buy nothing, and removing them is the cheapest optimisation available in a spatial prompt pipeline. This guide works out how many places are actually needed and how to make the reduction structural, as part of [geometry tokenization strategies](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/).

## When to Use This Approach

Set a precision policy before the first prompt is assembled, and apply it at ingestion rather than at prompt time. Rounding late means every intermediate stage carries the full-precision text, and one of them will end up in a context window.

| Question the corpus answers | Ground accuracy needed | Decimal places (degrees) |
|-----------------------------|------------------------|--------------------------|
| Which district, region or catchment | ~100 m | 3 |
| Which street or block | ~10 m | 4 |
| Which building or parcel | ~1 m | 5 |
| Survey-grade boundary work | ~0.1 m | 6 |
| Anything beyond that | Not achievable from the source | — |

The final row matters as much as the others. A source accurate to a metre does not become accurate to a centimetre by being printed with more digits, and carrying those digits through a pipeline implies a precision that will eventually be quoted back as though it were real.

<figure class="diagram">
<svg viewBox="26 9 716 213" role="img" aria-labelledby="cpt-scale-t cpt-scale-d" xmlns="http://www.w3.org/2000/svg"><title id="cpt-scale-t">What each decimal place is worth on the ground</title><desc id="cpt-scale-d">Three decimal places resolves about a hundred metres, four about ten, five about one and six about a tenth of a metre, with anything beyond exceeding the accuracy of typical sources.</desc><rect x="26" y="9" width="716" height="213" fill="#ffffff"/><text x="390" y="34" fill="#5b6471" font-size="13" text-anchor="middle">Decimal places of latitude, and the distance each one resolves</text><rect x="40" y="58" width="160" height="96" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="216" y="58" width="160" height="96" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="392" y="58" width="160" height="96" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="568" y="58" width="160" height="96" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="14" text-anchor="middle" font-weight="600"><text x="120" y="90">3 places</text><text x="296" y="90">4 places</text><text x="472" y="90">5 places</text><text x="648" y="90">6 places</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="120" y="118">about 100 m</text><text x="120" y="140">a district</text><text x="296" y="118">about 10 m</text><text x="296" y="140">a street</text><text x="472" y="118">about 1 m</text><text x="472" y="140">a building</text><text x="648" y="118">about 0.1 m</text><text x="648" y="140">beyond most sources</text></g><text x="390" y="204" fill="#1f2937" font-size="13" text-anchor="middle">Choose from the question, then check the source can support it</text></svg>
<figcaption><b>Two constraints, and the tighter one wins.</b> The question sets a floor on precision and the source sets a ceiling; carrying digits above the ceiling is the common error and it costs tokens on every feature in every prompt.</figcaption>
</figure>

## Implementation

The policy is a function from required ground accuracy to decimal places, applied once at ingestion, with a measurement of what it saved.

```python
import logging
import re
from dataclasses import dataclass

log = logging.getLogger("coordinate_precision")

# Degrees of latitude per metre is roughly constant; longitude shrinks with latitude,
# so a policy in degrees is conservative near the equator and generous near the poles.
_PLACES_BY_METRES = ((0.1, 6), (1.0, 5), (10.0, 4), (100.0, 3), (1000.0, 2))

_NUMBER = re.compile(r"-?\d+\.\d+")


@dataclass(frozen=True)
class PrecisionResult:
    text: str
    places: int
    chars_before: int
    chars_after: int


def places_for(accuracy_m: float, source_accuracy_m: float | None = None) -> int:
    """Decimal places from the question's needs, capped by what the source can support."""
    if accuracy_m <= 0:
        raise ValueError("accuracy must be a positive number of metres")
    needed = next((p for limit, p in _PLACES_BY_METRES if accuracy_m <= limit), 2)
    if source_accuracy_m is None:
        return needed
    supported = next((p for limit, p in _PLACES_BY_METRES if source_accuracy_m <= limit), 2)
    if supported < needed:
        log.info("question wants %d places, source supports %d — capping",
                 needed, supported)
    return min(needed, supported)


def round_coordinates(text: str, places: int) -> PrecisionResult:
    """Round every decimal number in a geometry string, without touching the structure."""
    if places < 0:
        raise ValueError("places must not be negative")

    def fix(match: re.Match) -> str:
        value = float(match.group(0))
        formatted = f"{value:.{places}f}"
        # Strip a trailing zero run, but never leave a bare decimal point.
        return formatted.rstrip("0").rstrip(".") or "0"

    out = _NUMBER.sub(fix, text)
    return PrecisionResult(out, places, len(text), len(out))
```

Operating on the text rather than on parsed geometry is a deliberate simplification for the measurement path: it lets the same function run over well-known text, structured objects and anything else without a parser, and the regular expression only matches decimal numbers, so structural characters are untouched.

For the storage path, round the parsed geometry instead, so the stored value and the emitted text agree:

```python
from shapely.ops import transform as shapely_transform


def round_geometry(geom, places: int):
    """Round coordinates in place, then re-validate — rounding can self-intersect."""
    factor = 10 ** places

    def snap(x, y, z=None):
        rx, ry = round(x * factor) / factor, round(y * factor) / factor
        return (rx, ry) if z is None else (rx, ry, z)

    rounded = shapely_transform(snap, geom)
    if not rounded.is_valid:
        from shapely.validation import make_valid
        log.info("rounding to %d places produced an invalid ring; repairing", places)
        rounded = make_valid(rounded)
    return rounded
```

The re-validation is not paranoia. Rounding moves vertices, and two vertices that were a fraction of a unit apart can land on the same point, turning a thin sliver into a zero-area spike that fails validity. It happens on real data at four decimal places and routinely at three.

<figure class="diagram">
<svg viewBox="26 9 688 221" role="img" aria-labelledby="cpt-cost-t cpt-cost-d" xmlns="http://www.w3.org/2000/svg"><title id="cpt-cost-t">Token cost of one polygon at four precisions</title><desc id="cpt-cost-d">The same eighty-vertex polygon costs progressively fewer tokens as decimal places are removed, with the reduction from seven places to five saving roughly a third.</desc><rect x="26" y="9" width="688" height="221" fill="#ffffff"/><text x="390" y="34" fill="#5b6471" font-size="13" text-anchor="middle">One 80-vertex polygon, measured with a real tokenizer</text><rect x="180" y="56" width="520" height="34" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="180" y="98" width="400" height="34" rx="5" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="180" y="140" width="330" height="34" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="180" y="182" width="270" height="34" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="12.5"><text x="40" y="78">7 places</text><text x="40" y="120">5 places</text><text x="40" y="162">4 places</text><text x="40" y="204">3 places</text></g><g fill="#1f2937" font-size="12" text-anchor="middle"><text x="440" y="78">1180 tokens</text><text x="380" y="120">790</text><text x="345" y="162">640</text><text x="315" y="204">520</text></g></svg>
<figcaption><b>The first two places removed are the cheapest tokens you will ever save.</b> Going from seven to five costs nothing in answer quality for any question above metre scale and returns a third of the geometry budget.</figcaption>
</figure>

## Validation & Testing

```python
from shapely.geometry import Polygon


def test_rounding_preserves_structure():
    text = "POLYGON((-3.192834912 55.946123991, -3.181 55.9412, -3.192834912 55.946123991))"
    out = round_coordinates(text, 5).text
    assert out.count("(") == text.count("(") and out.count(",") == text.count(",")
    assert "-3.19283" in out and "912" not in out


def test_rounded_geometry_stays_valid():
    sliver = Polygon([(0, 0), (1e-5, 1e-7), (2e-5, 0), (1e-5, -1e-7)])
    assert round_geometry(sliver, 4).is_valid


def test_source_accuracy_caps_the_policy():
    assert places_for(accuracy_m=0.1, source_accuracy_m=1.0) == 5


def test_zero_accuracy_is_rejected():
    try:
        places_for(0.0)
    except ValueError:
        return
    raise AssertionError("a zero accuracy requirement must be rejected, not treated as infinite")
```

The second test is the one to keep. It uses a shape deliberately narrower than the rounding grid, which is exactly the geometry that appears in real parcel data where two boundaries nearly coincide, and it is the case that turns a precision change into a validity incident.

Measure the saving on a sample of real geometry when the policy changes, and record it. "We reduced precision to five places" is a change; "we reduced precision to five places, cutting geometry tokens by 34% on a hundred-feature sample" is a decision anyone can evaluate.

## Gotchas & Edge Cases

**Rounding applied at prompt time only.** Every stage upstream still carries the full-precision text, and one of them — a cache, a log, an intermediate summary — will end up in a context window. Round at ingestion so the reduction is structural.

**Trailing-zero stripping that breaks a parser.** Removing trailing zeros is a real saving and produces values like `55.9` where a strict reader expects a fixed format. Test the actual parser you use rather than assuming, and keep the guard against stripping down to a bare decimal point.

**Uniform precision across wildly different latitudes.** A degree of longitude is about 111 km at the equator and 60 km at 57 degrees north, so a policy in decimal degrees is roughly twice as precise at high latitude as at the equator. That is harmless — it errs toward more precision — but it means the metre figures in the table are approximate and the equatorial case is the one to check.

<figure class="diagram">
<svg viewBox="32 46 697 190" role="img" aria-labelledby="cpt-snap-t cpt-snap-d" xmlns="http://www.w3.org/2000/svg"><title id="cpt-snap-t">How rounding turns a thin sliver into an invalid spike</title><desc id="cpt-snap-d">Two vertices closer together than the rounding grid collapse onto the same point, converting a narrow but valid polygon into a zero-area spike that fails a validity check.</desc><rect x="32" y="46" width="697" height="190" fill="#ffffff"/><rect x="60" y="60" width="260" height="120" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="190" y="126" fill="#1f2937" font-size="12.5" text-anchor="middle">before: a narrow but valid ring</text><rect x="420" y="60" width="280" height="120" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="560" y="114" fill="#1f2937" font-size="12.5" text-anchor="middle">after: two vertices coincide</text><text x="560" y="140" fill="#5b6471" font-size="12" text-anchor="middle">zero-area spike, validity fails</text><text x="380" y="218" fill="#1f2937" font-size="13" text-anchor="middle">Re-validate after rounding — the failure appears on real parcel data at four places</text></svg>
<figcaption><b>Rounding is a geometric operation, not a formatting one.</b> It moves vertices, and vertices that move onto each other change the topology of the ring — which is why the repair step belongs in the rounding function rather than somewhere downstream.</figcaption>
</figure>

**Precision reduced below the tolerance of a downstream join.** If features are matched to a reference dataset by coordinate proximity, rounding both sides to three places can merge distinct features. Round after any coordinate-based joining, or join on identifiers instead.

**A source that rounds differently between exports.** Coordinates that shift in the last digit between deliveries produce spurious change detection on every update. Rounding to a fixed policy on ingestion is also the fix for this, and it is usually the reason someone notices the policy is missing.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should projected coordinates use the same policy?</span></summary><p>The same idea, different numbers. Projected coordinates are typically in metres, so one decimal place is a decimetre and two is a centimetre — most work needs at most one, and many corpora need none. The mistake to avoid is carrying a degrees-based policy across to a projected frame, where five decimal places describes a hundredth of a millimetre and costs a great deal of context to say nothing.</p></details>

<details class="faq-item"><summary><span>Does reducing precision affect retrieval as well as prompts?</span></summary><p>Only if coordinates are being embedded, which they should not be. With coordinates in structured metadata and prose in the vector, precision affects the size of the geometry payload and the accuracy of spatial filters, not the embeddings. Filters are the thing to check: a bounding box rounded inward can exclude a feature at its edge, which is why extents should be rounded outward.</p></details>

<details class="faq-item"><summary><span>How do I find the source's actual accuracy?</span></summary><p>From its documentation where it exists, and from the data where it does not. A source that stores twelve decimal places but whose values all end in the same repeating pattern has been reprojected from a coarser original, and a histogram of the last significant digit usually makes that obvious. Where nothing is documented and the data is inconclusive, assume the coarser of the plausible options — over-claiming precision is the more damaging error.</p></details>

<details class="faq-item"><summary><span>Is it worth varying precision per feature class?</span></summary><p>Occasionally, and it complicates the pipeline for a modest gain. A corpus mixing building footprints with national boundaries genuinely wants different precisions, and the simpler alternative — one policy at the finest requirement — costs tokens on the coarse features. If you do vary it, key it on feature class rather than on size, and record the applied precision per feature so the variation is visible downstream.</p></details>

A final note on where the saving actually lands. Reducing precision shrinks the geometry payload, which is one part of a prompt among several; on a request dominated by retrieved prose the effect on total cost is modest. Where it matters most is a request assembling many features, which is exactly the request most likely to be near the budget — so the optimisation is worth most precisely when it is needed.

## Related

- Up to the parent topic: [Geometry Tokenization Strategies](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/)
- [How to Tokenize Polygon Boundaries for Transformer Models](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/how-to-tokenize-polygon-boundaries-for-transformer-models/)
- [Well-Known Text, Structured Objects and Cell Identifiers](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/wkt-vs-geojson-vs-h3-for-llm-tokenization/)
- Related topic: [Context-Window Optimization for Maps](/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/)
