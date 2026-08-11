---
title: Normalizing Mixed-Frame Data Before LLM Ingestion
description: Bring geometry from many sources into one frame at ingestion — per-source declarations, batched transforms, repair with provenance, and a rejection queue that gets read.
slug: normalizing-mixed-crs-data-before-llm-ingestion
type: howto
breadcrumb: Normalizing Mixed-Frame Data
datePublished: 2025-01-15
dateModified: 2026-08-11
---

# Normalizing Mixed-Frame Data Before LLM Ingestion

Real ingestion is not one file with one frame. It is fourteen sources, four of which declare a projection correctly, six of which declare one that is technically wrong, three of which declare nothing, and one that changes convention halfway through the year. This guide is about running that reality through a single gate without either rejecting everything or quietly assuming your way to a plausible-looking corpus. It is the working implementation of [coordinate reference system normalization](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/).

## When to Use This Approach

Use it at the boundary where external data becomes internal data, and nowhere else. Normalizing repeatedly downstream is wasted work and creates several places where the rules can diverge.

| Source behaviour | Handling | Recorded as |
|------------------|----------|-------------|
| Declares a valid frame | Validate and transform | Declared |
| Declares a wrong frame consistently | Per-source override, with evidence | Overridden |
| Declares nothing, frame known | Per-source default, with evidence | Assumed |
| Declares nothing, frame unknown | Reject | Rejected |
| Mixes frames within one file | Reject the file, not the rows | Rejected |

The distinction between the second and third rows matters operationally. An override says the source is wrong and we know better; an assumption says the source is silent and we have external evidence. Both are declarations someone made, and both should be reviewable — which is why the provenance field records which one applied.

<figure class="diagram">
<svg viewBox="1 38 778 194" role="img" aria-labelledby="nmf-src-t nmf-src-d" xmlns="http://www.w3.org/2000/svg"><title id="nmf-src-t">Where the frame for a geometry actually comes from</title><desc id="nmf-src-d">Four provenance classes — declared by the source, overridden by configuration, assumed from a per-source default, or rejected — each recorded on the stored geometry.</desc><rect x="1" y="38" width="778" height="194" fill="#ffffff"/><rect x="30" y="52" width="170" height="120" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="216" y="52" width="170" height="120" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="402" y="52" width="170" height="120" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="588" y="52" width="162" height="120" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="115" y="84">declared</text><text x="301" y="84">overridden</text><text x="487" y="84">assumed</text><text x="669" y="84">rejected</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="115" y="112">the source said so</text><text x="115" y="138">and it validated</text><text x="301" y="112">the source is wrong</text><text x="301" y="138">configuration says so</text><text x="487" y="112">the source is silent</text><text x="487" y="138">evidence elsewhere</text><text x="669" y="112">no basis at all</text><text x="669" y="138">queued for a person</text></g><text x="390" y="214" fill="#1f2937" font-size="13" text-anchor="middle">The middle two are human decisions — recording which one applied is what makes them auditable</text></svg>
<figcaption><b>Three of these four are legitimate.</b> What separates a disciplined pipeline from a hopeful one is not that it never assumes, but that every assumption is attached to a source, carries its evidence, and appears in the record of every geometry it produced.</figcaption>
</figure>

## Implementation

The ingester resolves a frame per source, transforms in batches for efficiency, and returns both accepted geometries and a rejection list that a person is expected to read.

```python
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError
from shapely.ops import transform as shapely_transform
from shapely.validation import make_valid
from shapely.errors import GEOSException

log = logging.getLogger("mixed_frame_ingest")

CANONICAL_EPSG = 4326


@dataclass(frozen=True)
class SourceRule:
    source_id: str
    override_epsg: Optional[int] = None     # the source declares wrongly
    default_epsg: Optional[int] = None      # the source declares nothing
    evidence: str = ""                      # why we believe either of the above


@dataclass(frozen=True)
class Accepted:
    geometry: object
    source_epsg: int
    provenance: str                          # declared | overridden | assumed
    repaired: bool
    evidence: str


@dataclass(frozen=True)
class Rejected:
    source_id: str
    identifier: str
    reason: str


def resolve_frame(declared: Optional[int], rule: SourceRule) -> tuple[Optional[int], str, str]:
    """Return (epsg, provenance, evidence). None means reject."""
    if rule.override_epsg is not None:
        return rule.override_epsg, "overridden", rule.evidence
    if declared is not None:
        return declared, "declared", ""
    if rule.default_epsg is not None:
        return rule.default_epsg, "assumed", rule.evidence
    return None, "rejected", "source declares no frame and no default is configured"
```

Batching the transforms matters more than it looks on a large ingest. Building a transformer is expensive relative to using one, and a naive loop that constructs a fresh transformer per geometry can spend most of its time on setup.

```python
def ingest(records: Iterable[dict], rules: dict[str, SourceRule]
           ) -> tuple[list[Accepted], list[Rejected]]:
    """Normalize a mixed batch. Returns accepted geometries and an explicit rejection list."""
    accepted: list[Accepted] = []
    rejected: list[Rejected] = []
    by_frame: dict[tuple[int, str, str], list[dict]] = defaultdict(list)

    for rec in records:
        rule = rules.get(rec["source_id"], SourceRule(rec["source_id"]))
        epsg, provenance, evidence = resolve_frame(rec.get("declared_epsg"), rule)
        if epsg is None:
            rejected.append(Rejected(rec["source_id"], rec["id"], evidence))
            continue
        try:
            CRS.from_epsg(epsg)
        except CRSError as exc:
            rejected.append(Rejected(rec["source_id"], rec["id"],
                                     f"EPSG:{epsg} will not construct: {exc}"))
            continue
        by_frame[(epsg, provenance, evidence)].append(rec)

    target = CRS.from_epsg(CANONICAL_EPSG)
    for (epsg, provenance, evidence), group in by_frame.items():
        source = CRS.from_epsg(epsg)
        tf = None if source.equals(target) else Transformer.from_crs(
            source, target, always_xy=True)
        for rec in group:
            geom, repaired = _repair(rec["geometry"])
            if geom is None:
                rejected.append(Rejected(rec["source_id"], rec["id"],
                                         "geometry could not be repaired"))
                continue
            try:
                moved = geom if tf is None else shapely_transform(tf.transform, geom)
            except Exception as exc:
                rejected.append(Rejected(rec["source_id"], rec["id"],
                                         f"transform from EPSG:{epsg} failed: {exc}"))
                continue
            accepted.append(Accepted(moved, epsg, provenance, repaired, evidence))

    log.info("ingest: %d accepted, %d rejected across %d frame group(s)",
             len(accepted), len(rejected), len(by_frame))
    return accepted, rejected


def _repair(geom):
    if geom is None or geom.is_empty:
        return None, False
    if geom.is_valid:
        return geom, False
    try:
        fixed = make_valid(geom)
    except GEOSException:
        return None, False
    return (fixed, True) if not fixed.is_empty else (None, False)
```

The rejection list is a return value rather than a log line because it needs to reach a person. A pipeline that logs rejections and returns only successes will run for months with a fifth of its input silently discarded, and the symptom — "the corpus seems thin in the north" — arrives long after the cause.

<figure class="diagram">
<svg viewBox="16 24 734 208" role="img" aria-labelledby="nmf-batch-t nmf-batch-d" xmlns="http://www.w3.org/2000/svg"><title id="nmf-batch-t">Per-geometry transformers against grouped transforms</title><desc id="nmf-batch-d">Constructing a transformer for every geometry dominates ingest time, while grouping records by source frame builds one transformer per frame and spends the remaining time on the geometry itself.</desc><rect x="16" y="24" width="734" height="208" fill="#ffffff"/><text x="30" y="62" fill="#b3324f" font-size="13" font-weight="600">per geometry</text><rect x="190" y="38" width="440" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="636" y="38" width="100" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="410" y="64" fill="#1f2937" font-size="12" text-anchor="middle">building transformers</text><text x="686" y="64" fill="#1f2937" font-size="12" text-anchor="middle">real work</text><text x="30" y="152" fill="#12805c" font-size="13" font-weight="600">grouped</text><rect x="190" y="128" width="60" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="256" y="128" width="480" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="220" y="154" fill="#1f2937" font-size="11.5" text-anchor="middle">setup</text><text x="496" y="154" fill="#1f2937" font-size="12" text-anchor="middle">real work</text><text x="390" y="214" fill="#1f2937" font-size="13" text-anchor="middle">Same transformations, same results — one transformer per frame instead of per record</text></svg>
<figcaption><b>The grouping is not a micro-optimisation.</b> On a corpus with a handful of source frames it turns an ingest that takes hours into one that takes minutes, which is the difference between reprocessing being routine and being avoided.</figcaption>
</figure>

## Validation & Testing

```python
def test_rejections_are_returned_not_only_logged():
    records = [{"source_id": "unknown", "id": "a", "geometry": POINT, "declared_epsg": None}]
    accepted, rejected = ingest(records, rules={})
    assert not accepted and len(rejected) == 1
    assert "no frame" in rejected[0].reason


def test_provenance_distinguishes_assumed_from_declared():
    rules = {"silent": SourceRule("silent", default_epsg=27700, evidence="agency confirmed 2024")}
    records = [
        {"source_id": "silent", "id": "a", "geometry": POINT, "declared_epsg": None},
        {"source_id": "silent", "id": "b", "geometry": POINT, "declared_epsg": 4326},
    ]
    accepted, _ = ingest(records, rules)
    assert {a.provenance for a in accepted} == {"assumed", "declared"}


def test_one_transformer_per_frame_group(monkeypatch):
    built = []
    original = Transformer.from_crs
    monkeypatch.setattr(Transformer, "from_crs",
                        lambda *a, **k: (built.append(a) or original(*a, **k)))
    ingest(HUNDRED_RECORDS_TWO_FRAMES, rules={})
    assert len(built) <= 2
```

The third test guards a performance property with a correctness-shaped assertion, which is the only way this kind of regression gets caught. Nothing about a per-geometry transformer produces wrong output; it simply makes reprocessing expensive enough that people stop doing it.

## Gotchas & Edge Cases

**An override applied to a source that later fixes itself.** The upstream export starts declaring correctly, the override keeps forcing the old value, and the geometry is now wrong in the opposite direction. Review overrides on a schedule, and log when an override contradicts a declaration rather than silently winning.

**Mixed frames within one file.** Rejecting the file rather than the offending rows is deliberate: a file with two conventions usually means a concatenation, and the rows you can identify as wrong are rarely all of them.

**Repair applied before the frame is known.** Repairing geometry in a frame that later turns out to be wrong bakes the repair into the wrong coordinates. Resolve the frame first, repair second, transform third — the order in the code is the order that survives contact with bad data.

<figure class="diagram">
<svg viewBox="3 24 753 208" role="img" aria-labelledby="nmf-order-t nmf-order-d" xmlns="http://www.w3.org/2000/svg"><title id="nmf-order-t">Why the order of resolve, repair and transform matters</title><desc id="nmf-order-d">Repairing before the frame is known bakes a repair into coordinates that are about to move; resolving first, repairing second and transforming third keeps each step operating on data it can interpret.</desc><rect x="3" y="24" width="753" height="208" fill="#ffffff"/><text x="30" y="62" fill="#12805c" font-size="13" font-weight="600">correct</text><rect x="170" y="38" width="170" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="356" y="38" width="170" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="542" y="38" width="170" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="12" text-anchor="middle"><text x="255" y="64">resolve the frame</text><text x="441" y="64">repair</text><text x="627" y="64">transform</text></g><text x="30" y="152" fill="#b3324f" font-size="13" font-weight="600">wrong</text><rect x="170" y="128" width="170" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="356" y="128" width="170" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="542" y="128" width="170" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="12" text-anchor="middle"><text x="255" y="154">repair</text><text x="441" y="154">resolve the frame</text><text x="627" y="154">transform</text></g><text x="380" y="214" fill="#1f2937" font-size="13" text-anchor="middle">A repair applied in the wrong frame is a correction to coordinates that were about to move</text></svg>
<figcaption><b>Order is not a matter of taste here.</b> Repair changes vertex positions, and vertex positions only mean something once the frame is known — so a repair performed first is a decision made with the wrong information.</figcaption>
</figure>

**Rejections that nobody owns.** A queue with no reader is a discard with extra steps. Route rejections to whoever owns the source, with the source identifier in the message, and track the queue depth as a monitored value.

**Evidence fields left empty.** An assumption with no recorded evidence is indistinguishable from a guess six months later. Make the evidence string required on any rule that overrides or defaults, even if the evidence is only "confirmed by the data owner on this date".

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should a source override be per file or per source?</span></summary><p>Per source, with the file identifier recorded on the accepted geometry so a per-file exception can be traced later. Per-file overrides multiply quickly and end up as a directory of one-off rules nobody can reason about; a per-source rule with a dated evidence note stays reviewable. Where one file genuinely differs, that is usually a sign the source has changed its convention, which is worth handling as a change rather than as an exception.</p></details>

<details class="faq-item"><summary><span>How should the rejection queue be sized and monitored?</span></summary><p>Track depth and age, and alert on age rather than depth. A queue with two hundred items that are all from this morning is a busy ingest; a queue with five items that have been there for three weeks is an ownership problem. Aging is also the signal that distinguishes a genuine data issue from a rule that needs adding — items that sit are usually items nobody knows what to do with.</p></details>

<details class="faq-item"><summary><span>Is it worth normalizing during a bulk backfill differently from streaming ingest?</span></summary><p>The rules should be identical; only the batching changes. A backfill can group aggressively across a whole file, while a streaming ingest groups within a window. What must not differ is the resolution logic — two code paths that decide frames differently will produce a corpus whose geometry depends on when it arrived, which is the hardest kind of inconsistency to diagnose because nothing about the data records it.</p></details>

<details class="faq-item"><summary><span>What should happen when a transform succeeds but produces implausible output?</span></summary><p>Treat it as a rejection with a distinct reason. A transform that lands geometry in the wrong hemisphere has technically succeeded and is certainly wrong, and the round-trip check described in the parent topic is what catches it. Recording it as its own rejection class rather than as a generic failure is what lets you see, later, that one source accounts for all of them.</p></details>

<details class="faq-item"><summary><span>Should accepted geometry keep its original coordinates as well?</span></summary><p>Keep the source frame code and the transformation note, which together let the original be reconstructed, rather than storing both geometries. Duplicating the geometry doubles the storage for a value that is almost never read, and the two copies drift the moment anything edits one of them. The exception is a corpus where the original is legally the record of truth, in which case store the original and treat the canonical copy as derived.</p></details>

## Related

- Up to the parent topic: [Coordinate Reference System Normalization](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/)
- [Choosing a Canonical Frame for Spatial LLM Pipelines](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/choosing-a-canonical-crs-for-llm-pipelines/)
- [Detecting Axis-Order Swaps in Coordinate Input](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/detecting-axis-order-swaps-in-coordinate-input/)
- Related topic: [Retrieval-Augmented CRS Resolution](/geospatial-rag-pipelines/retrieval-augmented-crs-resolution/)
