---
title: Detecting Hallucinated Coordinates in LLM Output
description: Screen model-produced positions with cheap, ordered checks — domain, null island, region, land mask and gazetteer agreement — before any geometry engine runs.
slug: detecting-hallucinated-coordinates-in-llm-output
type: howto
breadcrumb: Detecting Hallucinated Coordinates
datePublished: 2025-02-05
dateModified: 2026-08-11
---

# Detecting Hallucinated Coordinates in LLM Output

A model asked for a position will always produce one, and the output is formatted identically whether it was recalled correctly, recalled wrongly, or invented. Screening is the cheap layer that separates those cases before anything expensive or consequential happens, and it is the first metric worth reporting in [evaluation and benchmarking for spatial LLMs](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/).

## When to Use This Approach

Screen every coordinate a model produces, and screen before scoring rather than after. A hallucinated coordinate scores zero on any overlap metric and tells you nothing about the model's geometric ability, so mixing the two failure classes makes both unreadable.

| Check | Cost | Catches |
|-------|------|---------|
| Coordinate domain | Free | Values outside the valid range for the frame |
| Null island | Free | Missing values rendered as zero |
| Expected region | Free | Positions in the wrong country or hemisphere |
| Repeated-digit pattern | Free | Fabricated numbers with implausible structure |
| Land or water mask | Cheap | Buildings in open ocean |
| Gazetteer agreement | A lookup | A real place, but not the one that was named |

The last row is the only expensive one and the only one that catches a plausible position attached to the wrong name — which is the failure that survives every other check and is the most damaging in practice.

<figure class="diagram">
<svg viewBox="16 42 748 192" role="img" aria-labelledby="hal-order-t hal-order-d" xmlns="http://www.w3.org/2000/svg"><title id="hal-order-t">Screening order, cheapest first</title><desc id="hal-order-d">Free arithmetic checks remove most fabrications, a mask removes more, and only the survivors reach the gazetteer lookup, which is the only check that costs a network call.</desc><rect x="16" y="42" width="748" height="192" fill="#ffffff"/><rect x="30" y="56" width="200" height="120" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="250" y="66" width="200" height="100" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="470" y="76" width="180" height="80" rx="8" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><rect x="670" y="86" width="80" height="60" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="12.5" text-anchor="middle" font-weight="600"><text x="130" y="102">arithmetic</text><text x="350" y="108">mask</text><text x="560" y="112">gazetteer</text><text x="710" y="112">pass</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="130" y="128">free, catches most</text><text x="350" y="132">cheap, local</text><text x="560" y="134">one lookup</text></g><g stroke="#5b6471" stroke-width="2" marker-end="url(#hal-order-a)"><line x1="232" y1="116" x2="246" y2="116"/><line x1="452" y1="116" x2="466" y2="116"/><line x1="652" y1="116" x2="666" y2="116"/></g><defs><marker id="hal-order-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/></marker></defs><text x="390" y="216" fill="#1f2937" font-size="13" text-anchor="middle">Ordering by cost keeps the expensive check off the majority of inputs</text></svg>
<figcaption><b>Most fabrications fail an arithmetic test.</b> Ordering the screen this way means the lookup — the only check with a network dependency — runs on a small fraction of the traffic, which matters both for latency and for how gracefully the screen degrades when the gazetteer is unavailable.</figcaption>
</figure>

## Implementation

The screen returns a verdict with the first check that failed, so a report can be broken down by failure kind rather than by a single pass rate.

```python
import logging
import re
from dataclasses import dataclass
from typing import Callable, Optional

log = logging.getLogger("coordinate_screen")

_REPEATED = re.compile(r"(\d)\1{4,}")          # five or more identical digits in a row


@dataclass(frozen=True)
class Screen:
    plausible: bool
    failed_check: Optional[str]
    detail: str


def screen_coordinate(
    lon: float,
    lat: float,
    expected_bbox: tuple[float, float, float, float],
    raw_text: str = "",
    on_land: Optional[Callable[[float, float], bool]] = None,
) -> Screen:
    """Ordered plausibility checks, cheapest first. Never raises."""
    try:
        lon, lat = float(lon), float(lat)
    except (TypeError, ValueError):
        return Screen(False, "parse", "coordinate values are not numbers")

    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        return Screen(False, "domain", f"({lon}, {lat}) is outside the coordinate domain")

    if abs(lon) < 1e-9 and abs(lat) < 1e-9:
        return Screen(False, "null_island", "exactly zero — usually a missing value")

    if raw_text and _REPEATED.search(raw_text):
        return Screen(False, "digit_pattern",
                      "the coordinate text contains an implausible run of repeated digits")

    w, s, e, n = expected_bbox
    if not (w - 1 <= lon <= e + 1 and s - 1 <= lat <= n + 1):
        return Screen(False, "region",
                      f"({lon:.4f}, {lat:.4f}) falls outside the expected region")

    if on_land is not None:
        try:
            if not on_land(lon, lat):
                return Screen(False, "water", "the position falls in open water")
        except Exception as exc:                   # mask outage degrades the screen, not the turn
            log.info("land mask unavailable (%s); skipping that check", exc)

    return Screen(True, None, "")
```

The one-degree slack on the region test is deliberate. Expected extents are usually derived from a corpus or a gazetteer record and are frequently slightly tight, so a strict comparison rejects legitimate positions near the edge. A degree is generous at the scale this check operates on and still catches anything in the wrong country.

The gazetteer agreement check is separate because it needs a name as well as a position, and because it is the only one worth running asynchronously if latency matters.

```python
def agrees_with_gazetteer(name: str, lon: float, lat: float, lookup,
                          tolerance_km: float = 25.0) -> Screen:
    """Does the model's position match where this name actually is?"""
    try:
        records = list(lookup(name))
    except Exception as exc:
        log.info("gazetteer unavailable for agreement check: %s", exc)
        return Screen(True, None, "agreement not checked — gazetteer unavailable")
    if not records:
        return Screen(False, "unknown_name", f"no gazetteer record named {name!r}")
    for rec in records:
        w, s, e, n = rec.bbox
        pad = tolerance_km / 111.0
        if w - pad <= lon <= e + pad and s - pad <= lat <= n + pad:
            return Screen(True, None, "")
    return Screen(False, "name_position_mismatch",
                  f"{name!r} exists but not near ({lon:.4f}, {lat:.4f})")
```

Returning plausible when the gazetteer is unavailable, with a note saying so, is the right degradation. Failing closed would reject every coordinate during an outage, converting a screening gap into a total loss of function; the note is what stops that state from being invisible.

<figure class="diagram">
<svg viewBox="16 32 748 214" role="img" aria-labelledby="hal-kinds-t hal-kinds-d" xmlns="http://www.w3.org/2000/svg"><title id="hal-kinds-t">Failure kinds have different causes and different fixes</title><desc id="hal-kinds-d">Domain and null-island failures indicate a parsing or plumbing bug, region failures indicate recall error, and name-position mismatches indicate genuine fabrication.</desc><rect x="16" y="32" width="748" height="214" fill="#ffffff"/><rect x="30" y="46" width="360" height="86" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="410" y="46" width="340" height="86" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="30" y="146" width="360" height="86" rx="8" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><rect x="410" y="146" width="340" height="86" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13" font-weight="600"><text x="52" y="76">domain / null island</text><text x="432" y="76">wrong region</text><text x="52" y="176">unknown name</text><text x="432" y="176">name-position mismatch</text></g><g fill="#5b6471" font-size="12"><text x="52" y="102">a plumbing bug, not the model</text><text x="52" y="122">check the parsing path first</text><text x="432" y="102">recall error — a real place elsewhere</text><text x="432" y="122">grounding would have caught it</text><text x="52" y="202">the place may not exist</text><text x="52" y="222">or the gazetteer has a gap</text><text x="432" y="202">the most damaging class</text><text x="432" y="222">plausible everywhere except the truth</text></g></svg>
<figcaption><b>One pass rate would hide all four.</b> The first is an engineering bug in your own code, the second is a model limitation, the third may be a data gap, and only the fourth is fabrication — and they are fixed by four different people.</figcaption>
</figure>

## Validation & Testing

```python
UK = (-8.0, 49.0, 2.0, 61.0)


def test_null_island_is_caught_before_the_region_check():
    s = screen_coordinate(0.0, 0.0, UK)
    assert s.failed_check == "null_island"


def test_out_of_domain_is_caught_first():
    s = screen_coordinate(999.0, 0.0, UK)
    assert s.failed_check == "domain"


def test_region_slack_admits_a_coastal_edge_case():
    assert screen_coordinate(2.4, 51.1, UK).plausible


def test_mask_outage_does_not_reject():
    def broken(_x, _y):
        raise ConnectionError("mask down")
    assert screen_coordinate(-3.19, 55.95, UK, on_land=broken).plausible


def test_name_position_mismatch_is_reported_distinctly():
    s = agrees_with_gazetteer("Edinburgh", 2.35, 48.85, lookup_returning_edinburgh)
    assert s.failed_check == "name_position_mismatch"
```

The first two tests pin the ordering, which matters because the reported failure kind drives the investigation. A null-island coordinate reported as a region failure sends someone looking at the model when the bug is in a parser.

Keep the fixtures small and pointed. Each of these tests exists to pin one behaviour, and a fixture that exercises three at once fails ambiguously.

Run the screen over historical model output when you first build it. The distribution of failure kinds across a few thousand past answers is the most informative hour you will spend on this, and it usually reveals that the largest class is a plumbing bug nobody knew about.

## Gotchas & Edge Cases

**Screening after conversion to geometry.** By then a bad coordinate has already produced a valid-looking point, and the geometry engine has spent work on it. Screen the numbers, before parsing.

**A region extent that is too tight.** Derived extents frequently clip legitimate coastal and border positions. The slack in the region test handles it; removing that slack to "tighten the check" reliably produces complaints about correct answers being rejected.

**The digit-pattern check firing on real coordinates.** Some legitimate coordinates contain runs of repeated digits, and the check is a heuristic rather than a proof. Keep it as a screen that flags rather than one that rejects outright where the consequence of a false positive is high.

<figure class="diagram">
<svg viewBox="46 46 688 183" role="img" aria-labelledby="hal-tol-t hal-tol-d" xmlns="http://www.w3.org/2000/svg"><title id="hal-tol-t">Agreement tolerance against what it is meant to catch</title><desc id="hal-tol-d">A generous tolerance around a gazetteer record still separates a position in the right town from one in another country, which is the error class the check exists for.</desc><rect x="46" y="46" width="688" height="183" fill="#ffffff"/><rect x="60" y="60" width="260" height="120" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="120" y="96" width="140" height="50" rx="5" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="190" y="126" fill="#1f2937" font-size="12" text-anchor="middle">the record</text><text x="190" y="212" fill="#5b6471" font-size="12" text-anchor="middle">25 km tolerance around it</text><rect x="420" y="60" width="300" height="56" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="570" y="94" fill="#1f2937" font-size="12.5" text-anchor="middle">a position in the same town: passes</text><rect x="420" y="132" width="300" height="56" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="570" y="166" fill="#1f2937" font-size="12.5" text-anchor="middle">a position in another country: caught</text></svg>
<figcaption><b>A blunt check aimed at a blunt error.</b> The tolerance is generous because the failure it exists to catch is enormous, and tightening it converts a reliable screen into a source of complaints about correct answers.</figcaption>
</figure>

**Treating an unknown name as a fabrication.** A name absent from the gazetteer may be a gap in the gazetteer rather than an invention, especially for small or recently created features. Report it as its own class and let the volume tell you which it is.

**Screening only model output.** Coordinates also arrive from users and from documents, and both produce the same failure classes. The same screen applies, with a different expected extent.

**Failure kinds collapsed in the metric.** A single "hallucination rate" is easy to publish and impossible to act on. Report per kind, always — the aggregate can only tell you whether things got worse.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should a screened-out coordinate stop the answer?</span></summary><p>It should stop that claim, not the answer. An agent that produced one bad coordinate among five useful statements should retract the one and keep the rest, with a note that a position could not be verified. Stopping the whole turn converts a partial failure into a total one, and users experience that as the system being brittle rather than careful.</p></details>

<details class="faq-item"><summary><span>How large should the gazetteer agreement tolerance be?</span></summary><p>Large enough to accommodate a legitimate difference between a centroid and a boundary — twenty-five kilometres is a reasonable default for settlements and generous for buildings. Tightening it starts rejecting correct answers where the gazetteer's record is a point and the model gave a position within the same town. The check is looking for wrong-country errors, not for precision.</p></details>

<details class="faq-item"><summary><span>Is a land mask worth the dependency?</span></summary><p>For corpora about land features, usually yes, because "in the ocean" is a common and unambiguous fabrication signature. Keep the mask coarse and local rather than precise and remote: a low-resolution mask that ships with the application catches the open-ocean cases and never fails, whereas a precise remote service adds latency and a dependency for a marginal gain in coastal accuracy you should not rely on anyway.</p></details>

<details class="faq-item"><summary><span>Does screening remove the need for grounding?</span></summary><p>No — it is the fallback for cases where grounding was skipped or impossible. Grounding replaces a model-produced coordinate with a looked-up one and is strictly better; screening only tells you that a produced coordinate is not obviously wrong. Where both are available, ground first and screen the result as a cheap consistency check.</p></details>

<details class="faq-item"><summary><span>Where should the screen sit relative to the tool-calling loop?</span></summary><p>Between the model&#8217;s output and any tool that consumes a position, so a screened-out coordinate never reaches a geometry engine or a map. Placing it after the tool call means the work is already done and, worse, that a downstream system may have acted on the position before the screen ran. The screen is cheap enough that running it at the boundary costs nothing measurable.</p></details>

## Related

- Up to the parent topic: [Evaluation and Benchmarking for Spatial LLMs](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/)
- [Measuring Spatial IoU for LLM-Generated Geometries](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/measuring-spatial-iou-for-llm-generated-geometries/)
- [Building Regression Test Harnesses for Spatial Agents](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/building-regression-test-harnesses-for-spatial-agents/)
- Concept: [Geocoding and Place-Name Resolution](/spatial-llm-architecture-core-concepts/geocoding-and-place-name-resolution/)
