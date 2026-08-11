---
title: Detecting Axis-Order Swaps in Coordinate Input
description: Catch latitude-first coordinates before they enter the pipeline, using range, area-of-use and land-mask checks that distinguish a swap from a genuinely unusual position.
slug: detecting-axis-order-swaps-in-coordinate-input
type: howto
breadcrumb: Axis-Order Swaps
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Detecting Axis-Order Swaps in Coordinate Input

A swapped coordinate pair is the cheapest bug in spatial software to create and one of the more expensive to find. Two conventions disagree about which number comes first, both are legitimate, and a system that reads one as the other places features in a mirrored world where most of them fall in the ocean. This guide builds a systematic detector, as part of the ingestion gate described in [coordinate reference system normalization](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/).

## When to Use This Approach

Run the check on every geographic coordinate entering the system, not on a sample. The cases it catches are the ones a spot inspection misses, because obvious swaps are obvious and the damaging ones look plausible.

| Signal | Catches | Misses |
|--------|---------|--------|
| Range check on latitude | Any pair where longitude exceeds 90 | Anything within ±90 in both |
| Area-of-use test | Pairs that fall outside the frame's own region | Regions symmetric about the diagonal |
| Land mask | Pairs landing in open ocean | Coastal and island cases |
| Source consistency | A whole file swapped relative to its siblings | The first file from a new source |

No single test is sufficient, which is why the detector runs them in sequence and reports which one fired. A pair that fails the range check is certain; one that fails only the land mask is a suspicion worth surfacing to a human.

<figure class="diagram">
<svg viewBox="16 38 748 202" role="img" aria-labelledby="axs-cases-t axs-cases-d" xmlns="http://www.w3.org/2000/svg"><title id="axs-cases-t">Three swapped pairs and how detectable each one is</title><desc id="axs-cases-d">A pair with a longitude above ninety is certainly swapped, one landing in open ocean is very likely swapped, and one landing on land in another country is undetectable without source context.</desc><rect x="16" y="38" width="748" height="202" fill="#ffffff"/><rect x="30" y="52" width="230" height="130" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="275" y="52" width="230" height="130" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="520" y="52" width="230" height="130" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="145" y="84">certain</text><text x="390" y="84">very likely</text><text x="635" y="84">undetectable alone</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="145" y="112">latitude above 90</text><text x="145" y="138">arithmetically impossible</text><text x="145" y="164">reject outright</text><text x="390" y="112">lands in open ocean</text><text x="390" y="138">possible but improbable</text><text x="390" y="164">flag for review</text><text x="635" y="112">lands on land elsewhere</text><text x="635" y="138">both readings plausible</text><text x="635" y="164">needs source context</text></g><text x="390" y="222" fill="#1f2937" font-size="13" text-anchor="middle">The third column is why file-level consistency matters more than any per-point test</text></svg>
<figcaption><b>Per-point tests thin out fast.</b> They catch the majority and leave a residue that only a comparison across the file — do these points agree with each other and with the source's declared region? — can resolve.</figcaption>
</figure>

## Implementation

The detector runs cheap tests first and returns a verdict with the reason and a confidence, so the caller can reject the certain cases and queue the suspicious ones.

```python
import logging
from dataclasses import dataclass
from typing import Optional, Sequence

from pyproj import CRS

log = logging.getLogger("axis_order")


@dataclass(frozen=True)
class SwapVerdict:
    swapped: Optional[bool]      # True, False, or None for "cannot tell"
    confidence: float
    reason: str


def check_pair(x: float, y: float, crs: CRS,
               on_land=None) -> SwapVerdict:
    """Test one coordinate pair. x is the first value as read, y the second."""
    if not crs.is_geographic:
        return SwapVerdict(False, 1.0, "projected frame: this test does not apply")

    # 1. Arithmetic impossibility — the strongest signal available.
    if abs(y) > 90.0 and abs(x) <= 90.0:
        return SwapVerdict(True, 1.0, "second value exceeds 90; it cannot be a latitude")
    if abs(x) > 90.0 and abs(y) > 90.0:
        return SwapVerdict(None, 0.0, "both values exceed 90; neither can be a latitude")

    # 2. Area of use — does the swapped reading fit the frame better?
    area = crs.area_of_use
    if area is not None:
        w, s, e, n = area.bounds
        as_read = w <= x <= e and s <= y <= n
        as_swapped = w <= y <= e and s <= x <= n
        if as_swapped and not as_read:
            return SwapVerdict(True, 0.85, "only the swapped reading falls in the frame's region")
        if as_read and not as_swapped:
            return SwapVerdict(False, 0.85, "only the given reading falls in the frame's region")

    # 3. Land mask — weak, and useful exactly where the others are silent.
    if on_land is not None:
        try:
            here, there = on_land(x, y), on_land(y, x)
        except Exception as exc:                    # a mask outage must not block ingestion
            log.info("land mask unavailable (%s); skipping this signal", exc)
            here = there = None
        if here is False and there is True:
            return SwapVerdict(True, 0.55, "the given reading falls in open water, the swap does not")

    return SwapVerdict(None, 0.0, "no signal distinguishes the two readings")
```

The three-valued verdict is what makes this usable. A binary detector forces every ambiguous case into one of two wrong answers; returning `None` lets the caller treat "cannot tell" as its own outcome, which for a file from a known-good source usually means proceeding.

File-level agreement is the signal that resolves most of the residue, and it is much stronger than any per-point test because a source is nearly always consistently swapped or consistently correct.

```python
def check_file(pairs: Sequence[tuple[float, float]], crs: CRS,
               sample: int = 200) -> SwapVerdict:
    """Aggregate per-point verdicts across a file; consistency is the real evidence."""
    if not pairs:
        return SwapVerdict(None, 0.0, "no coordinates to test")
    step = max(1, len(pairs) // sample)
    verdicts = [check_pair(x, y, crs) for x, y in pairs[::step]]
    decided = [v for v in verdicts if v.swapped is not None]
    if not decided:
        return SwapVerdict(None, 0.0, f"no signal across {len(verdicts)} sampled points")
    swapped = sum(1 for v in decided if v.swapped)
    share = swapped / len(decided)
    if share > 0.9:
        return SwapVerdict(True, min(0.99, 0.6 + share / 3),
                           f"{swapped} of {len(decided)} sampled points read as swapped")
    if share < 0.1:
        return SwapVerdict(False, min(0.99, 0.6 + (1 - share) / 3),
                           f"{len(decided) - swapped} of {len(decided)} read correctly")
    return SwapVerdict(None, 0.0,
                       f"inconsistent: {swapped} of {len(decided)} read as swapped — "
                       "the file may mix conventions")
```

The inconsistent case deserves its own outcome rather than a majority vote. A file where a fifth of points look swapped is not a file with a convention problem; it is a file with a data problem, and reordering the whole thing would corrupt the four fifths that were correct.

<figure class="diagram">
<svg viewBox="1 36 778 188" role="img" aria-labelledby="axs-flow-t axs-flow-d" xmlns="http://www.w3.org/2000/svg"><title id="axs-flow-t">What the ingestion gate does with each verdict</title><desc id="axs-flow-d">A certain swap is rejected with an explanation, a likely swap is queued for review, an inconsistent file is rejected as a data problem, and an undecidable case proceeds when the source is trusted.</desc><rect x="1" y="36" width="778" height="188" fill="#ffffff"/><rect x="30" y="50" width="170" height="76" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="216" y="50" width="170" height="76" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="402" y="50" width="170" height="76" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="588" y="50" width="162" height="76" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="12.5" text-anchor="middle" font-weight="600"><text x="115" y="78">certain swap</text><text x="301" y="78">likely swap</text><text x="487" y="78">inconsistent</text><text x="669" y="78">no signal</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="115" y="104">reject, explain</text><text x="301" y="104">queue for review</text><text x="487" y="104">reject as bad data</text><text x="669" y="104">proceed if trusted</text></g><rect x="30" y="158" width="720" height="52" rx="8" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><text x="390" y="190" fill="#1f2937" font-size="13" text-anchor="middle">Never reorder silently — a corrected file that nobody knows was corrected is a future mystery</text></svg>
<figcaption><b>Reordering is the tempting fix and the wrong one.</b> It works, it is invisible, and it leaves an upstream export producing broken data indefinitely — with a downstream system quietly compensating that nobody remembers to remove.</figcaption>
</figure>

## Validation & Testing

```python
from pyproj import CRS

WGS84 = CRS.from_epsg(4326)


def test_impossible_latitude_is_certain():
    v = check_pair(55.95, -3.19, WGS84)          # read as (lon, lat): lat = -3.19, fine
    assert v.swapped is False or v.swapped is None
    v2 = check_pair(-3.19, 155.0, WGS84)         # second value cannot be a latitude
    assert v2.swapped is True and v2.confidence == 1.0


def test_inconsistent_file_is_not_majority_voted():
    pairs = [(-3.19, 55.95)] * 8 + [(155.0, 55.95)] * 2
    v = check_file(pairs, WGS84)
    assert v.swapped is None and "inconsistent" in v.reason


def test_land_mask_outage_does_not_block():
    def broken(_x, _y):
        raise RuntimeError("mask service down")
    v = check_pair(-3.19, 55.95, WGS84, on_land=broken)
    assert v.swapped in (False, None)            # degraded, not failed
```

The second test is the one that protects the design. A majority vote over an inconsistent file is the obvious implementation and it silently corrupts the correct minority; asserting that it refuses instead is what keeps a future simplification honest.

## Gotchas & Edge Cases

**Regions symmetric about the diagonal.** A frame whose area of use spans similar ranges in both axes — near the equator at low longitudes — makes the area-of-use test silent, because both readings fit. This is where file-level consistency and source context do the work.

**A land mask that is too coarse.** A low-resolution mask reports coastal points as water and produces false positives on exactly the data most likely to be coastal. Use the mask as a weak signal only, with a confidence low enough that it queues rather than rejects.

<figure class="diagram">
<svg viewBox="36 36 698 197" role="img" aria-labelledby="axs-mask-t axs-mask-d" xmlns="http://www.w3.org/2000/svg"><title id="axs-mask-t">Why a coarse land mask produces false positives on the coast</title><desc id="axs-mask-d">A coastal point sits in a mask cell classified as water, so a correct coordinate is reported as landing in the sea and flagged as a suspected swap.</desc><rect x="36" y="36" width="698" height="197" fill="#ffffff"/><rect x="60" y="50" width="300" height="140" rx="6" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="60" y="50" width="150" height="140" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="135" y="126" fill="#1f2937" font-size="12" text-anchor="middle">land cell</text><text x="285" y="126" fill="#1f2937" font-size="12" text-anchor="middle">water cell</text><text x="210" y="216" fill="#5b6471" font-size="12" text-anchor="middle">a real coastal site falls in the water cell</text><rect x="430" y="60" width="290" height="56" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="575" y="94" fill="#1f2937" font-size="12.5" text-anchor="middle">mask says water — flagged as suspect</text><rect x="430" y="134" width="290" height="56" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="575" y="168" fill="#1f2937" font-size="12.5" text-anchor="middle">low confidence — queued, not rejected</text></svg>
<figcaption><b>A weak signal must carry a weak confidence.</b> Coastal and island data is disproportionately likely to be interesting and disproportionately likely to trip a coarse mask, so this test earns its place only as a queue trigger.</figcaption>
</figure>

**Files that mix conventions.** Concatenated exports from two systems, one of each convention. The inconsistent verdict catches this; treating it as a swap would corrupt half the file and treating it as correct would corrupt the other half.

**Projected frames caught by the geographic test.** Eastings and northings routinely exceed 90 and are not swapped. Short-circuit on the frame type first, as the implementation does, or every projected file will be rejected.

**Silent reordering as a "fix".** The most damaging response, because it works. The upstream export keeps producing swapped data, the compensation is invisible, and the day someone reads that source directly the two systems disagree with no record of why.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Why not just always reorder to match the library's convention?</span></summary><p>Because the reordering is a guess about which convention the source used, and it is applied to data where the guess cannot be checked. Where a source is genuinely known to be latitude-first, express that as a per-source configuration entry — an explicit declaration with a comment explaining the evidence — rather than as an inference made afresh on every file. The difference is that the configuration is visible and reviewable, and the inference is neither.</p></details>

<details class="faq-item"><summary><span>How many points should a file-level check sample?</span></summary><p>A couple of hundred, spread across the file rather than taken from the front. Front-loaded samples miss the common case where a file is a concatenation and the second half differs, and the check is cheap enough that spreading it costs nothing. Beyond a few hundred the confidence stops improving, because the signal is consistency rather than volume.</p></details>

<details class="faq-item"><summary><span>Does this apply to structured formats that specify an order?</span></summary><p>The specification helps and does not settle it, because exporters get it wrong. A format that mandates longitude-first is still routinely written latitude-first by tools that treated it as latitude-first internally, and the file remains syntactically valid. Run the check regardless; on a compliant file it costs microseconds and returns "no signal", which is exactly the right outcome.</p></details>

<details class="faq-item"><summary><span>What should the rejection message say?</span></summary><p>The two readings and where each one lands. "Read as given, this point is in the South Atlantic; read swapped, it is in Edinburgh" is a message that resolves the question in one line for whoever receives it. A bare "axis order suspected" sends someone to reproduce the check by hand, which is the work the detector was supposed to have done.</p></details>

## Related

- Up to the parent topic: [Coordinate Reference System Normalization](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/)
- [Choosing a Canonical Frame for Spatial LLM Pipelines](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/choosing-a-canonical-crs-for-llm-pipelines/)
- [Normalizing Mixed-Frame Data Before Ingestion](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/normalizing-mixed-crs-data-before-llm-ingestion/)
- Related technique: [Detecting Hallucinated Coordinates in LLM Output](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/detecting-hallucinated-coordinates-in-llm-output/)
