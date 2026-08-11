---
title: Carrying Frame and Extent Metadata Into Every Chunk
description: Attach the projection, bounding extent and feature identifiers to each retrieval chunk so a fragment retrieved out of context is still self-describing and filterable.
slug: carrying-crs-and-extent-metadata-into-every-chunk
type: howto
breadcrumb: Frame and Extent Metadata
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Carrying Frame and Extent Metadata Into Every Chunk

A chunk is retrieved alone. Whatever context sat around it in the source document — the header declaring the projection, the caption naming the survey, the sentence explaining the units — is gone by the time a model reads it. Metadata is how that context survives the trip. This guide covers computing and attaching the four fields that make a spatial chunk self-describing, as the final stage of [chunk-boundary strategies for spatial corpora](/geospatial-rag-pipelines/chunk-boundary-strategies-for-spatial-corpora/).

## When to Use This Approach

Always, for any corpus containing coordinates. The only real decision is how much to compute rather than copy, and that decision turns on whether the source can be trusted.

| Source declares | Do this | Why |
|-----------------|---------|-----|
| A frame, consistently | Copy it, verify against a sample coordinate | Cheap, and the check catches header drift |
| A frame, inconsistently | Resolve per chunk, record the evidence | One header cannot speak for a merged document |
| Nothing | Resolve from context and flag confidence | A guess recorded as fact is the worst outcome |
| A frame and an extent | Recompute the extent from the geometry anyway | Declared extents go stale; geometry does not |

The last row is the one people skip. A document's declared extent describes what the author thought the file contained, often before an edit removed half of it, and a chunk's extent must describe what the chunk actually holds — otherwise the spatial pre-filter in [spatial context retrieval and reranking](/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/) is filtering on fiction.

<figure class="diagram">
<svg viewBox="16 32 748 198" role="img" aria-labelledby="cme-loss-t cme-loss-d" xmlns="http://www.w3.org/2000/svg"><title id="cme-loss-t">What a chunk loses when it leaves its document</title><desc id="cme-loss-d">A source document carries a projection header, a units note and a survey caption above its features. After chunking, only the coordinates travel unless the context is copied into metadata.</desc><rect x="16" y="32" width="748" height="198" fill="#ffffff"/><rect x="30" y="46" width="300" height="170" rx="8" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><text x="180" y="74" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">source document</text><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="180" y="104">header: EPSG:27700</text><text x="180" y="128">note: distances in metres</text><text x="180" y="152">caption: 2024 site survey</text><text x="180" y="186">…then four hundred features</text></g><rect x="450" y="46" width="300" height="76" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="600" y="74" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">chunk without metadata</text><text x="600" y="100" fill="#5b6471" font-size="12" text-anchor="middle">numbers with no frame, era or units</text><rect x="450" y="140" width="300" height="76" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="600" y="168" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">chunk with metadata</text><text x="600" y="194" fill="#5b6471" font-size="12" text-anchor="middle">frame, extent, ids and period travel too</text><g stroke="#5b6471" stroke-width="2" marker-end="url(#cme-loss-a)"><line x1="332" y1="110" x2="446" y2="86"/><line x1="332" y1="150" x2="446" y2="176"/></g><defs><marker id="cme-loss-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/></marker></defs></svg>
<figcaption><b>Everything above the features is context.</b> It is obvious in the document and invisible in the chunk, which is why metadata is not an optimisation — it is the only mechanism that carries the document's assumptions to the point of use.</figcaption>
</figure>

## Implementation

The function parses each chunk's geometries, computes the true extent from them, verifies the declared frame against that extent, and returns a record that carries everything a downstream filter or citation layer needs.

```python
import logging
from dataclasses import dataclass, asdict
from typing import Iterable, Optional

from shapely import wkt
from shapely.errors import GEOSException
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from pyproj import CRS
from pyproj.exceptions import CRSError

log = logging.getLogger("chunk_metadata")


@dataclass(frozen=True)
class ChunkMeta:
    epsg: int
    bbox: tuple[float, float, float, float]
    feature_ids: tuple[str, ...]
    geometry_count: int
    frame_verified: bool          # False means the frame is copied, not confirmed
    notes: tuple[str, ...]


def _parse_all(literals: Iterable[str]) -> list[BaseGeometry]:
    """Parse what parses; log and skip what does not, never abort the chunk."""
    out = []
    for lit in literals:
        try:
            geom = wkt.loads(lit)
        except (GEOSException, ValueError) as exc:
            log.warning("unparseable geometry skipped during metadata build: %s", exc)
            continue
        if not geom.is_empty:
            out.append(geom)
    return out


def _frame_plausible(epsg: int, bbox) -> tuple[bool, Optional[str]]:
    """Check the extent sits inside the frame's declared area of use."""
    try:
        crs = CRS.from_epsg(epsg)
    except CRSError as exc:
        return False, f"frame will not construct: {exc}"
    if crs.area_of_use is None:
        return False, "frame declares no area of use"
    if not crs.is_geographic:
        return False, "projected frame — extent check needs a transform first"
    w, s, e, n = crs.area_of_use.bounds
    inside = (w - 1 <= bbox[0] and bbox[2] <= e + 1
              and s - 1 <= bbox[1] and bbox[3] <= n + 1)
    return inside, None if inside else "extent falls outside the frame's area of use"


def build_chunk_meta(
    geometry_literals: Iterable[str],
    feature_ids: Iterable[str],
    declared_epsg: int,
) -> ChunkMeta:
    """Compute self-describing metadata for one chunk. Never raises on data quality."""
    geoms = _parse_all(geometry_literals)
    ids = tuple(str(i) for i in feature_ids)
    notes: list[str] = []

    if not geoms:
        # Deterministic fallback: a chunk with no usable geometry still gets a record,
        # with a degenerate extent that no bounding-box filter will match by accident.
        notes.append("no parseable geometry in chunk")
        return ChunkMeta(declared_epsg, (0.0, 0.0, 0.0, 0.0), ids, 0, False, tuple(notes))

    try:
        bounds = unary_union(geoms).bounds
    except GEOSException as exc:
        log.warning("union failed, falling back to per-geometry bounds: %s", exc)
        xs = [b for g in geoms for b in (g.bounds[0], g.bounds[2])]
        ys = [b for g in geoms for b in (g.bounds[1], g.bounds[3])]
        bounds = (min(xs), min(ys), max(xs), max(ys))

    verified, problem = _frame_plausible(declared_epsg, bounds)
    if problem:
        notes.append(problem)

    return ChunkMeta(declared_epsg, tuple(round(v, 7) for v in bounds),
                     ids, len(geoms), verified, tuple(notes))
```

The degenerate extent for a geometry-free chunk is a deliberate choice over `None`. A null extent forces every consumer to branch, and one of them will forget; a zero-area extent at the origin simply never intersects a real query region, which fails safe. The notes field carries the reason so the chunk is not mistaken for a coding error.

The union-failure fallback exists because `unary_union` is the one call here that can fail on geometry that individually parsed fine — self-intersecting rings that survive `wkt.loads` and confound the overlay. Taking the envelope of individual bounds gives the same answer for well-behaved input and a slightly loose answer for the pathological case, which is the right direction to be wrong in.

<figure class="diagram">
<svg viewBox="46 38 678 212" role="img" aria-labelledby="cme-bbox-t cme-bbox-d" xmlns="http://www.w3.org/2000/svg"><title id="cme-bbox-t">Declared extent against computed extent</title><desc id="cme-bbox-d">A document declares a wide regional extent while its remaining features occupy a small corner. Filtering on the declared extent retrieves the chunk for queries it cannot answer.</desc><rect x="46" y="38" width="678" height="212" fill="#ffffff"/><rect x="60" y="52" width="290" height="150" rx="6" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="80" y="140" width="90" height="46" rx="4" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="205" y="80" fill="#1f2937" font-size="12.5" text-anchor="middle">declared extent</text><text x="205" y="106" fill="#5b6471" font-size="12" text-anchor="middle">what the header claims</text><text x="125" y="212" fill="#12805c" font-size="12" text-anchor="middle">computed</text><rect x="420" y="52" width="290" height="150" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="565" y="98" fill="#1f2937" font-size="12.5" text-anchor="middle">computed extent only</text><text x="565" y="124" fill="#5b6471" font-size="12" text-anchor="middle">matches the queries this chunk</text><text x="565" y="146" fill="#5b6471" font-size="12" text-anchor="middle">can actually answer</text><text x="380" y="238" fill="#5b6471" font-size="12" text-anchor="middle">Recompute from geometry — a declared extent is a claim, not a measurement</text></svg>
<figcaption><b>False positives are quiet.</b> An over-wide extent never errors; it just admits the chunk into result sets for a region it says nothing about, where it competes for context space against documents that do.</figcaption>
</figure>

## Validation & Testing

```python
def test_extent_contains_every_geometry():
    meta = build_chunk_meta(LITERALS, IDS, 4326)
    for lit in LITERALS:
        g = wkt.loads(lit)
        assert g.bounds[0] >= meta.bbox[0] - 1e-6
        assert g.bounds[2] <= meta.bbox[2] + 1e-6


def test_geometryless_chunk_gets_a_degenerate_extent():
    meta = build_chunk_meta([], ["f-1"], 4326)
    assert meta.geometry_count == 0
    assert meta.bbox == (0.0, 0.0, 0.0, 0.0)
    assert "no parseable geometry" in " ".join(meta.notes)


def test_implausible_frame_is_flagged_not_rejected():
    # British coordinates declared as a Japanese frame: usable record, honest flag.
    meta = build_chunk_meta(["POINT(-3.19 55.95)"], ["f-2"], 6668)
    assert meta.frame_verified is False and meta.notes
```

The first test is a containment property and should be run over the whole corpus during an index build, not just over a fixture; it is cheap and it is the assertion that catches an off-by-one in extent rounding. Rounding to seven decimal places is roughly a centimetre, comfortably below any real query tolerance, and it keeps the metadata compact — but rounding *inward* would break containment, which is why the rounding happens after the bounds are taken and is tested for.

## Gotchas & Edge Cases

**Antimeridian-crossing extents.** A chunk holding features either side of 180 degrees longitude produces an extent spanning almost the whole world, and every bounding-box filter then matches it. Detect the case — a longitude span above 180 degrees — and store two extents or a flag, rather than one meaningless rectangle.

<figure class="diagram">
<svg viewBox="26 9 708 201" role="img" aria-labelledby="cme-anti-t cme-anti-d" xmlns="http://www.w3.org/2000/svg"><title id="cme-anti-t">An extent computed naively across the antimeridian</title><desc id="cme-anti-d">Two features either side of 180 degrees longitude produce a bounding box spanning nearly the whole world, which matches every spatial query. Storing two extents instead keeps the filter meaningful.</desc><rect x="26" y="9" width="708" height="201" fill="#ffffff"/><text x="380" y="34" fill="#5b6471" font-size="13" text-anchor="middle">Two features at longitude 179 and minus 179</text><rect x="40" y="56" width="680" height="60" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="380" y="82" fill="#1f2937" font-size="12.5" text-anchor="middle">naive extent: minus 179 to 179 — matches every query on the planet</text><text x="380" y="104" fill="#5b6471" font-size="12" text-anchor="middle">the two features are two degrees apart; the box is 358 degrees wide</text><rect x="40" y="140" width="120" height="56" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="600" y="140" width="120" height="56" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="100" y="174" fill="#1f2937" font-size="12" text-anchor="middle">extent A</text><text x="660" y="174" fill="#1f2937" font-size="12" text-anchor="middle">extent B</text><text x="380" y="174" fill="#1f2937" font-size="12.5" text-anchor="middle">two extents, each honest about what it covers</text></svg>
<figcaption><b>The widest possible box is the least useful one.</b> Nothing errors, nothing is logged, and the chunk simply becomes a candidate for every spatially filtered query in the corpus.</figcaption>
</figure>

**Rounding that shrinks the box.** Truncating rather than rounding, or rounding both corners the same way, can pull the maximum corner inside a geometry's true bound. Round the minimum down and the maximum up if you round at all; the containment test above is what catches this.

**Frames that are projected.** The plausibility check short-circuits for projected frames because comparing metres against a degree-based area of use is meaningless. Transform a corner into geographic coordinates first if you want the check to apply, and be explicit that an unverified frame is not a wrong one.

**Identifiers that are not stable.** Feature identifiers assigned by the chunker — index positions, hashes of coordinate text — change whenever the document is re-exported, which breaks every citation made against the previous build. Prefer identifiers from the source, and where none exist, derive one from a stable attribute rather than from position.

**Metadata that outgrows the chunk.** A chunk holding two hundred features accumulates two hundred identifiers, which can rival the text in size and inflate storage across a large corpus. Cap the stored list and record the total count alongside it, so citation still works for small chunks and the count remains honest for large ones.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should the extent be stored as a geometry or as four numbers?</span></summary><p>As a geometry if the store supports it, because that lets a real spatial index serve the filter and lets the predicate be exact. Four numbers work when the store only offers numeric range filters, and they are perfectly adequate for a rectangular pre-filter. What you should not do is store both and let them drift; if you need the numbers for convenience, derive them at read time from the geometry.</p></details>

<details class="faq-item"><summary><span>Does every chunk need a period as well as an extent?</span></summary><p>Any corpus where the same place is described at different times, which is most of them. Without a period, a query about current conditions retrieves a 2009 survey and a 2024 one with equal enthusiasm, and the model has no basis for preferring either. The period does not need to be precise — a year is usually enough — but it does need to be present and to come from the document rather than from the ingestion date.</p></details>

<details class="faq-item"><summary><span>What if the source declares different frames in different sections?</span></summary><p>Then the document has more than one frame and a single document-level value is wrong for some of it. Resolve per chunk, using the nearest preceding declaration, and record which declaration was used in the notes. This happens most often in merged or appended documents, and it is precisely the case where copying one header across every chunk silently corrupts a subset of the corpus.</p></details>

<details class="faq-item"><summary><span>Is it worth verifying the frame on every build, or once at ingestion?</span></summary><p>Every build, because the check is cheap and the inputs change. A registry update can deprecate a code, an upstream correction can move a document's features, and a chunking change can redistribute geometry between chunks. Verification at ingestion tells you the frame was plausible on the day it arrived, which is a weaker statement than it sounds after a year of edits.</p></details>

## Related

- Up to the parent topic: [Chunk-Boundary Strategies for Spatial Corpora](/geospatial-rag-pipelines/chunk-boundary-strategies-for-spatial-corpora/)
- [Splitting Polygon-Heavy Documents Without Severing Geometries](/geospatial-rag-pipelines/chunk-boundary-strategies-for-spatial-corpora/splitting-polygon-heavy-documents-without-severing-geometries/)
- Concept: [Coordinate Reference System Normalization](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/)
- Related topic: [Spatial Metadata and Catalog Indexing](/geospatial-rag-pipelines/spatial-metadata-and-catalog-indexing/)
