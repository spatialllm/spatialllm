---
title: Summarizing Raster Statistics for LLM Prompts
description: Turn pixels into one honest sentence — class proportions rounded to what the sample supports, with the resolution, the date and the coverage attached.
slug: summarizing-raster-statistics-for-llm-prompts
type: howto
breadcrumb: Summarizing Raster Statistics
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Summarizing Raster Statistics for LLM Prompts

A model cannot read pixels, so the pixels have to become a sentence — and every choice in that conversion is an opportunity to imply precision the data does not have. This guide produces statements a model can quote safely, carrying the resolution, the date and the share of the shape that actually had data, as the reporting stage of [vector-raster hybrid processing](/spatial-llm-architecture-core-concepts/vector-raster-hybrid-processing/).

## When to Use This Approach

Summarise whenever a raster informs an answer. Send the numbers, never the grid: a model handed a matrix of class codes will produce arithmetic that looks like analysis.

| Raster kind | Summary | Never |
|-------------|---------|-------|
| Categorical, e.g. land cover | Class proportions, top few | The full class histogram |
| Continuous, e.g. elevation | Median and range | The mean alone |
| Binary mask | Covered share | A count of pixels |
| Time series | Change between two dates | Every date |
| Any | With resolution, date, coverage | A bare percentage |

The last row is the one that turns a correct number into a usable one. "Sixty-eight per cent impermeable" is unquotable; the same figure with "from thirty-metre data captured in 2023, over 94% of the parcel" can be repeated by a reader without misleading anyone.

<figure class="diagram">
<svg viewBox="16 38 748 184" role="img" aria-labelledby="srs-sent-t srs-sent-d" xmlns="http://www.w3.org/2000/svg"><title id="srs-sent-t">The four parts of a quotable raster statement</title><desc id="srs-sent-d">A usable statement carries the figure, the resolution it was measured at, the date of capture and the share of the shape that had usable data.</desc><rect x="16" y="38" width="748" height="184" fill="#ffffff"/><rect x="30" y="52" width="170" height="100" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="216" y="52" width="170" height="100" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="402" y="52" width="170" height="100" rx="8" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><rect x="588" y="52" width="162" height="100" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="115" y="84">the figure</text><text x="301" y="84">resolution</text><text x="487" y="84">date</text><text x="669" y="84">coverage</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="115" y="112">68% impermeable</text><text x="301" y="112">30 m data</text><text x="487" y="112">captured 2023</text><text x="669" y="112">94% of the shape</text><text x="115" y="136">rounded to fit</text><text x="301" y="136">what it can resolve</text><text x="487" y="136">how current it is</text><text x="669" y="136">what was missing</text></g><text x="390" y="204" fill="#1f2937" font-size="13" text-anchor="middle">Remove any one of the four and the sentence overstates what was measured</text></svg>
<figcaption><b>Three of these four are usually dropped.</b> The figure alone is the part everyone keeps, and it is the part that means least without the other three attached to it.</figcaption>
</figure>

## Implementation

The summariser computes proportions at a precision the sample supports, formats the top few classes, and appends the provenance.

```python
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

log = logging.getLogger("raster_summary")


@dataclass(frozen=True)
class Statement:
    text: str
    classes: tuple[tuple[str, float], ...]
    coverage: float
    cells: int


def digits_for(cells: int) -> int:
    """Decimal places a proportion from this many cells can honestly carry."""
    if cells < 100:
        return 0
    if cells < 10_000:
        return 1
    return 2


def proportions(values: Sequence[int], cells: int) -> list[tuple[int, float]]:
    """Class proportions, rounded to what the sample supports."""
    if not values:
        return []
    digits = digits_for(cells)
    counts = Counter(int(v) for v in values)
    total = sum(counts.values())
    return [(cls, round(100.0 * n / total, digits)) for cls, n in counts.most_common()]


def to_statement(name: str, values: Sequence[int], labels: Mapping[int, str],
                 pixel_size_m: float, captured_year: int | None,
                 coverage: float, top_n: int = 4) -> Statement:
    """One sentence a model can quote, with nothing implied that was not measured."""
    cells = len(values)
    if cells == 0:
        return Statement(f"{name}: no usable raster values within the shape.", (), coverage, 0)

    props = proportions(values, cells)
    named = tuple((labels.get(cls, f"class {cls}"), pct) for cls, pct in props[:top_n])
    body = ", ".join(f"{label} {pct:g}%" for label, pct in named)
    if len(props) > top_n:
        remainder = round(sum(p for _, p in props[top_n:]), digits_for(cells))
        body += f", other classes {remainder:g}%"

    date = f", captured {captured_year}" if captured_year else ""
    cover = "" if coverage > 0.98 else f", from {coverage * 100:.0f}% of the shape"
    text = f"{name}: {body} (from {pixel_size_m:g} m data{date}{cover})."
    if coverage <= 0.5:
        log.info("%s summarised from only %.0f%% coverage", name, coverage * 100)
    return Statement(text, named, coverage, cells)
```

Reporting the remainder rather than silently truncating the class list is what keeps the proportions summing to a hundred. A statement listing four classes that add to seventy-one per cent invites the reader to wonder what the other twenty-nine were, and a model will occasionally invent an answer.

Continuous rasters need different statistics, and the difference matters more than it looks.

```python
import statistics


def continuous_statement(name: str, values: Sequence[float], unit: str,
                         pixel_size_m: float, captured_year: int | None,
                         coverage: float) -> Statement:
    """Median and range, not the mean — edge pixels move a mean and not a median."""
    if not values:
        return Statement(f"{name}: no usable values within the shape.", (), coverage, 0)
    ordered = sorted(values)
    median = statistics.median(ordered)
    lo, hi = ordered[int(0.05 * (len(ordered) - 1))], ordered[int(0.95 * (len(ordered) - 1))]
    date = f", captured {captured_year}" if captured_year else ""
    cover = "" if coverage > 0.98 else f", from {coverage * 100:.0f}% of the shape"
    text = (f"{name}: median {median:.1f} {unit}, most values between {lo:.1f} and "
            f"{hi:.1f} {unit} (from {pixel_size_m:g} m data{date}{cover}).")
    return Statement(text, (), coverage, len(values))
```

A median and a percentile range survive the handful of edge cells that belong to something else, which a mean does not. Clipping a shape almost always includes a few pixels of a neighbouring surface, and those are exactly the values a mean moves toward.

<figure class="diagram">
<svg viewBox="16 42 748 188" role="img" aria-labelledby="srs-prec-t srs-prec-d" xmlns="http://www.w3.org/2000/svg"><title id="srs-prec-t">Decimal places a sample can support</title><desc id="srs-prec-d">A proportion from forty cells cannot distinguish tenths of a per cent, so rounding to the sample size is what stops the statement implying an accuracy the pixels never had.</desc><rect x="16" y="42" width="748" height="188" fill="#ffffff"/><rect x="30" y="56" width="230" height="120" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="275" y="56" width="230" height="120" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="520" y="56" width="230" height="120" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="145" y="88">40 cells</text><text x="390" y="88">2 000 cells</text><text x="635" y="88">80 000 cells</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="145" y="118">whole per cent only</text><text x="145" y="144">&#8220;68%&#8221;</text><text x="390" y="118">one decimal</text><text x="390" y="144">&#8220;67.9%&#8221;</text><text x="635" y="118">two decimals</text><text x="635" y="144">&#8220;67.94%&#8221;</text></g><text x="390" y="212" fill="#1f2937" font-size="13" text-anchor="middle">One cell in forty is 2.5% — printing a tenth of a per cent from it is fiction</text></svg>
<figcaption><b>Rounding is where invented accuracy enters.</b> The arithmetic will happily produce four decimal places from forty cells, and a reader has no way to tell that the last three of them describe nothing.</figcaption>
</figure>

## Validation & Testing

```python
def test_proportions_sum_to_a_hundred_with_the_remainder():
    values = [1] * 50 + [2] * 30 + [3] * 10 + [4] * 6 + [5] * 4
    st = to_statement("cover", values, LABELS, 10.0, 2023, 1.0, top_n=3)
    assert "other classes" in st.text


def test_precision_matches_the_sample_size():
    small = to_statement("cover", [1] * 30 + [2] * 10, LABELS, 30.0, 2023, 1.0)
    assert "." not in small.text.split("%")[0].split()[-1]


def test_low_coverage_is_stated():
    st = to_statement("cover", [1] * 100, LABELS, 10.0, 2023, coverage=0.62)
    assert "62% of the shape" in st.text


def test_empty_values_produce_a_refusal_not_a_zero():
    st = to_statement("cover", [], LABELS, 10.0, 2023, 1.0)
    assert "no usable raster values" in st.text and st.classes == ()


def test_continuous_uses_median_not_mean():
    values = [10.0] * 99 + [10_000.0]              # one outlier from an adjacent surface
    st = continuous_statement("elevation", values, "m", 10.0, 2023, 1.0)
    assert "median 10.0" in st.text
```

The last test is the one that justifies the choice of statistic. A single pixel of a neighbouring surface moves a mean of a hundred values by a hundred metres and moves the median by nothing, and clipped shapes contain those pixels routinely.

Keep the statement object alongside the text rather than discarding it. The prose is what reaches the prompt, and the structured proportions are what a chart, an export or a later comparison needs — and regenerating them from the sentence is both awkward and a source of disagreement between the two surfaces.

## Gotchas & Edge Cases

**Nodata counted as a class.** Missing values become a large proportion of something that was never measured. Exclude them from the denominator and report the coverage separately.

**A class list truncated without a remainder.** Proportions that do not sum to a hundred invite invention. Always state the remainder, even when it is small.

**Percentages formatted with a fixed number of decimals.** A formatting string with two decimal places reports 68.00% from forty cells, which is worse than reporting 68% because the zeros look like measurement. Derive the format from the sample size.

**Coverage omitted when it is high but not complete.** Ninety-four per cent coverage is worth stating and is easy to suppress with a threshold set too low. Suppress only when coverage is effectively total.

**Class labels missing for codes present in the data.** A statement mentioning "class 17" is unusable and usually means the class scheme has changed upstream. Fail the build on an unknown code rather than printing the number.

<figure class="diagram">
<svg viewBox="66 58 612 164" role="img" aria-labelledby="srs-med-t srs-med-d" xmlns="http://www.w3.org/2000/svg"><title id="srs-med-t">Why a median survives edge pixels and a mean does not</title><desc id="srs-med-d">A handful of pixels belonging to an adjacent surface pull a mean far from the body of the distribution while leaving the median where the data actually sits.</desc><rect x="66" y="58" width="612" height="164" fill="#ffffff"/><g fill="#e4f5ec" stroke="#12805c" stroke-width="2"><rect x="80" y="96" width="46" height="78" rx="4"/><rect x="134" y="72" width="46" height="102" rx="4"/><rect x="188" y="82" width="46" height="92" rx="4"/><rect x="242" y="110" width="46" height="64" rx="4"/></g><g fill="#fdeaee" stroke="#b3324f" stroke-width="2"><rect x="600" y="158" width="46" height="16" rx="4"/></g><rect x="152" y="182" width="6" height="26" rx="3" fill="#12805c"/><text x="176" y="204" fill="#12805c" font-size="12">median</text><rect x="330" y="182" width="6" height="26" rx="3" fill="#b3324f"/><text x="352" y="204" fill="#b3324f" font-size="12">mean, dragged right by four pixels</text><text x="623" y="196" fill="#5b6471" font-size="12" text-anchor="middle">edge pixels</text></svg>
<figcaption><b>Clipping always catches a few pixels of something else.</b> Those pixels are a property of the boundary rather than of the shape, and the statistic that ignores them is the one describing what was asked about.</figcaption>
</figure>

**A statement built per pixel rather than per shape.** Summaries are cheap and prompts are not; one sentence per shape is the unit, and a summary per tile or per band multiplies the prompt without adding information.

## Frequently Asked Questions

<details class="faq-item"><summary><span>How many classes should a statement name?</span></summary><p>Three or four, plus the remainder. Beyond that the sentence stops being readable and the tail classes are usually below the precision the sample supports anyway. If a specific minor class matters to the question — a protected habitat type, say — name it explicitly regardless of its rank, and say why it is being singled out.</p></details>

<details class="faq-item"><summary><span>Should the statement include the class codes as well as the labels?</span></summary><p>Only if something downstream needs to key on them, and then in metadata rather than in the sentence. Codes in prose are noise to a reader and an invitation to a model to reason about them numerically. The label is what carries meaning; the code is an implementation detail of the source product.</p></details>

<details class="faq-item"><summary><span>What should be said when two rasters disagree?</span></summary><p>Both statements, with their dates and resolutions, and no attempt to reconcile them. Disagreement between a recent coarse product and an older fine one is information — it may be genuine change, or a resolution artefact — and averaging them destroys the only signal a reader has. If one must be preferred, that choice belongs in the catalog fitness scoring where its reasons are recorded.</p></details>

<details class="faq-item"><summary><span>Is it worth including a small chart instead of a sentence?</span></summary><p>Not in a prompt. A model reads the sentence reliably and a chart not at all, and any chart would have to be described in text to be usable anyway. Charts are worth producing for the human-facing surface alongside the answer, built from the same statement object so the two cannot disagree.</p></details>

<details class="faq-item"><summary><span>Should the statement be generated by code or by the model?</span></summary><p>By code, always, and quoted by the model. A model handed proportions and asked to phrase them will occasionally round differently, drop the coverage caveat, or convert a class name into something more readable and less accurate. Generating the sentence deterministically and instructing the model to quote it verbatim removes an entire category of drift between what was measured and what was said.</p></details>

## Related

- Up to the parent topic: [Vector-Raster Hybrid Processing](/spatial-llm-architecture-core-concepts/vector-raster-hybrid-processing/)
- [Aligning Raster Tiles with Vector Masks](/spatial-llm-architecture-core-concepts/vector-raster-hybrid-processing/aligning-raster-tiles-with-vector-masks-for-llm-context/)
- Related topic: [Context-Window Optimization for Maps](/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/)
- Related topic: [Spatial Metadata and Catalog Indexing](/geospatial-rag-pipelines/spatial-metadata-and-catalog-indexing/)
