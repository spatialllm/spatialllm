---
title: Coordinate Reference System Normalization
description: Make every geometry entering a spatial LLM pipeline arrive in one declared, validated frame — with axis order checked, datum shifts accounted for, and no silent assumptions.
slug: coordinate-reference-system-normalization
type: topic
breadcrumb: CRS Normalization
datePublished: 2025-01-14
dateModified: 2026-08-11
---

# Coordinate Reference System Normalization

Two datasets describing the same street can disagree by two hundred metres and both be correct, because they were recorded against different realisations of the ground. Normalization is the ingestion gate that resolves that disagreement before anything downstream can be misled by it: every geometry arrives in one declared frame, validated, with its transformation recorded.

This topic is foundational to [spatial LLM architecture and core concepts](/spatial-llm-architecture-core-concepts/) and to everything built on it. A model reasoning over mixed frames produces answers that are internally consistent and externally wrong — distances that do not match the map, containment claims that flip depending on which source a feature came from. The gate here is what makes [spatial reasoning and relation inference](/spatial-llm-architecture-core-concepts/spatial-reasoning-and-relation-inference/) meaningful, and its absence is what [retrieval-augmented CRS resolution](/geospatial-rag-pipelines/retrieval-augmented-crs-resolution/) exists to repair after the fact.

<figure class="diagram">
<svg viewBox="0 0 780 260" role="img" aria-labelledby="crs-drift-t crs-drift-d" xmlns="http://www.w3.org/2000/svg"><title id="crs-drift-t">Three sources of the same feature, three positions</title><desc id="crs-drift-d">One building recorded in three datasets sits in three slightly different places because of datum realisation, axis order and rounding, and only a normalization gate reconciles them before they are compared.</desc><rect x="0" y="0" width="780" height="260" fill="#ffffff"/><text x="390" y="34" fill="#5b6471" font-size="13" text-anchor="middle">One building, three datasets, three positions — all of them &#8220;correct&#8221;</text><rect x="30" y="58" width="220" height="106" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="280" y="58" width="220" height="106" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="530" y="58" width="220" height="106" rx="8" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="140" y="88">survey archive</text><text x="390" y="88">open dataset</text><text x="640" y="88">field capture</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="140" y="114">historic datum</text><text x="140" y="138">offset by ~120 m</text><text x="390" y="114">current datum</text><text x="390" y="138">the reference position</text><text x="640" y="114">axis order swapped</text><text x="640" y="138">in the wrong hemisphere</text></g><rect x="30" y="186" width="720" height="46" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="390" y="214" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">Normalization gate: one frame, validated, transformation recorded</text></svg>
<figcaption><b>None of these is a data-quality problem at source.</b> Each dataset is internally consistent and correctly documented; the problem appears only when they meet, which is exactly what an ingestion pipeline arranges.</figcaption>
</figure>

## Foundational Principles

**A geometry with no declared frame is rejected, not assumed.** The most expensive habit in spatial software is treating undeclared coordinates as geographic because they look like degrees. Rejection at the gate produces an error someone fixes in an hour; assumption produces answers that are wrong by a datum shift for as long as the system runs.

**Transformation is recorded, not just performed.** Knowing that a geometry is now in the canonical frame is less useful than knowing what it was and how it got there. Record the source code, the target, and the transformation path, because datum transformations are not unique and two paths between the same pair can differ by a metre.

**Validity is repaired at the gate or the geometry does not enter.** Self-intersections, unclosed rings and zero-area slivers survive parsing and confound every predicate downstream. Repair deterministically at ingestion and record that a repair happened; a geometry that cannot be repaired is a rejection, not a warning.

## Step-by-Step Implementation Pipeline

### 1. Choose one canonical frame and write down why

Everything in the pipeline is stored and compared in one frame, chosen once. Geographic coordinates are the usual answer for storage because they are unambiguous globally and every tool accepts them, with metric work done in a projection chosen per operation. The reasoning behind that split, and when to store projected instead, is in [choosing a canonical frame for spatial LLM pipelines](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/choosing-a-canonical-crs-for-llm-pipelines/).

```python
import logging
from dataclasses import dataclass
from typing import Optional

from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError

log = logging.getLogger("crs_normalization")

CANONICAL_EPSG = 4326          # storage frame: global, unambiguous, universally supported


@dataclass(frozen=True)
class Normalized:
    geometry: object
    source_epsg: int
    target_epsg: int
    transform_note: str
    repaired: bool
```

### 2. Reject the undeclared and validate the declared

The gate's first job is to distinguish "no frame" from "a frame I should check". Both are common; only one is recoverable automatically.

```python
class CRSRejected(ValueError):
    """The geometry cannot enter the pipeline; the caller must supply a frame."""


def require_frame(declared_epsg: Optional[int]) -> CRS:
    """Return a usable frame, or refuse. Never guess."""
    if declared_epsg is None:
        raise CRSRejected("geometry has no declared reference frame; refusing to assume one")
    try:
        crs = CRS.from_epsg(declared_epsg)
    except CRSError as exc:
        raise CRSRejected(f"EPSG:{declared_epsg} will not construct: {exc}") from exc
    if crs.is_deprecated:
        log.warning("EPSG:%s is deprecated; a successor should be preferred", declared_epsg)
    return crs
```

Raising here rather than returning a default is the whole point of the gate. Every downstream stage is entitled to assume that anything it receives has a known frame, and that assumption is only safe if the gate is willing to stop.

### 3. Detect axis order before trusting the numbers

A geographic coordinate pair can be stored latitude-first or longitude-first, and both orders are legal in different conventions. A swapped pair usually lands in the sea or in another hemisphere, which is detectable — and occasionally lands somewhere plausible, which is why the check must be systematic rather than a spot inspection. The full treatment is in [detecting axis-order swaps in coordinate input](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/detecting-axis-order-swaps-in-coordinate-input/).

```python
def axis_order_suspect(x: float, y: float, crs: CRS) -> bool:
    """True when the pair looks swapped for this frame."""
    if not crs.is_geographic:
        return False                                  # projected frames: different check
    if abs(y) > 90.0 and abs(x) <= 90.0:
        return True                                   # y cannot be a latitude; x could be
    area = crs.area_of_use
    if area is None:
        return False
    w, s, e, n = area.bounds
    inside = w <= x <= e and s <= y <= n
    swapped_inside = w <= y <= e and s <= x <= n
    return swapped_inside and not inside
```

### 4. Transform through an explicit path

Datum transformations are not unique: several published paths connect the same pair of frames, differing by up to a metre. Letting the library pick silently means the same input can transform differently on two machines with different grid files installed.

```python
def build_transformer(source: CRS, target: CRS) -> tuple[Transformer, str]:
    """Return a transformer and a note describing the path it will use."""
    transformer = Transformer.from_crs(source, target, always_xy=True)
    note = transformer.description or f"{source.to_epsg()}->{target.to_epsg()}"
    accuracy = getattr(transformer, "accuracy", None)
    if accuracy is not None and accuracy > 1.0:
        log.info("transformation %s has stated accuracy %.1f m", note, accuracy)
    return transformer, note
```

Recording the description turns an irreproducible result into a reproducible one. When two environments disagree about where a feature is, the transformation note is the first thing to compare and usually the answer.

<figure class="diagram">
<svg viewBox="16 16 768 226" role="img" aria-labelledby="crs-gate-t crs-gate-d" xmlns="http://www.w3.org/2000/svg"><title id="crs-gate-t">The normalization gate and its two exits</title><desc id="crs-gate-d">Geometry with a declared, constructible frame passes through validation, axis checking and transformation into the canonical frame; anything else exits to a rejection queue with a stated reason.</desc><rect x="16" y="16" width="768" height="226" fill="#ffffff"/><rect x="30" y="98" width="150" height="60" rx="8" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><text x="105" y="124" fill="#1f2937" font-size="12.5" text-anchor="middle" font-weight="600">incoming</text><text x="105" y="144" fill="#5b6471" font-size="12" text-anchor="middle">any source</text><rect x="220" y="98" width="160" height="60" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="300" y="124" fill="#1f2937" font-size="12.5" text-anchor="middle" font-weight="600">frame declared?</text><text x="300" y="144" fill="#5b6471" font-size="12" text-anchor="middle">and constructible</text><rect x="420" y="30" width="160" height="60" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="500" y="56" fill="#1f2937" font-size="12.5" text-anchor="middle" font-weight="600">axis + validity</text><text x="500" y="76" fill="#5b6471" font-size="12" text-anchor="middle">checked and repaired</text><rect x="620" y="30" width="150" height="60" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="695" y="56" fill="#1f2937" font-size="12.5" text-anchor="middle" font-weight="600">canonical frame</text><text x="695" y="76" fill="#5b6471" font-size="12" text-anchor="middle">path recorded</text><rect x="420" y="168" width="350" height="60" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="595" y="194" fill="#1f2937" font-size="12.5" text-anchor="middle" font-weight="600">rejection queue</text><text x="595" y="214" fill="#5b6471" font-size="12" text-anchor="middle">reason attached; a person fixes the source</text><g stroke="#5b6471" stroke-width="2" marker-end="url(#crs-gate-a)"><line x1="182" y1="128" x2="216" y2="128"/><line x1="382" y1="112" x2="416" y2="70"/><line x1="382" y1="144" x2="416" y2="188"/><line x1="582" y1="60" x2="616" y2="60"/></g><defs><marker id="crs-gate-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/></marker></defs></svg>
<figcaption><b>The rejection queue is a feature.</b> A pipeline with no exit for unusable input has not eliminated bad geometry; it has agreed to process it silently, which moves the failure from ingestion — where it is one person's afternoon — to inference, where it is everyone's problem.</figcaption>
</figure>

### 5. Repair validity deterministically

Repair must be deterministic across environments and must record that it happened, because a repaired geometry is not the geometry the source published and any measurement taken from it inherits that difference.

```python
from shapely.validation import make_valid
from shapely.errors import GEOSException


def repair(geom):
    """Return (geometry, repaired_flag). Unrepairable geometry is a rejection."""
    if geom is None or geom.is_empty:
        raise CRSRejected("empty geometry")
    if geom.is_valid:
        return geom, False
    try:
        fixed = make_valid(geom)
    except GEOSException as exc:
        raise CRSRejected(f"geometry could not be repaired: {exc}") from exc
    if fixed.is_empty:
        raise CRSRejected("geometry collapsed to empty under repair")
    return fixed, True
```

### 6. Assemble the gate

The pieces compose into one function that either returns a fully described normalized geometry or refuses with a reason. Nothing in between.

```python
from shapely.ops import transform as shapely_transform


def normalize(geom, declared_epsg: Optional[int], target_epsg: int = CANONICAL_EPSG) -> Normalized:
    """One entry point; either a fully described result or an explicit rejection."""
    source = require_frame(declared_epsg)
    target = CRS.from_epsg(target_epsg)

    geom, repaired = repair(geom)

    if source.is_geographic:
        rep = geom.representative_point()
        if axis_order_suspect(rep.x, rep.y, source):
            raise CRSRejected(
                f"coordinates appear axis-swapped for EPSG:{declared_epsg}; "
                "correct the source rather than reordering silently")

    if source.equals(target):
        return Normalized(geom, declared_epsg, target_epsg, "no transformation needed", repaired)

    transformer, note = build_transformer(source, target)
    try:
        moved = shapely_transform(transformer.transform, geom)
    except Exception as exc:
        raise CRSRejected(f"transformation {note} failed: {exc}") from exc
    return Normalized(moved, declared_epsg, target_epsg, note, repaired)
```

Refusing to silently reorder a suspected axis swap is a judgement worth keeping. Reordering would be right most of the time and would hide a source-side bug that will keep producing bad data; rejecting sends someone to fix it once. The mechanics of doing that safely, including how to handle a source where the swap is systematic and known, are covered in [normalizing mixed-frame data before ingestion](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/normalizing-mixed-crs-data-before-llm-ingestion/).

### 7. Enforce the same gate at the database layer

Application-side normalization is necessary and not sufficient, because data arrives by other routes. A column typed with a frame, plus a check constraint, makes the invariant structural.

```sql
-- The column type carries the frame; the constraint carries the validity rule.
CREATE TABLE features (
    feature_id  text PRIMARY KEY,
    geom        geometry(Geometry, 4326) NOT NULL,
    source_epsg integer NOT NULL,
    transform_note text NOT NULL,
    CONSTRAINT features_geom_valid CHECK (ST_IsValid(geom))
);

CREATE INDEX features_geom_idx ON features USING gist (geom);

-- Index-aware read: bounding box first, exact predicate second.
SELECT feature_id
FROM   features
WHERE  geom && ST_MakeEnvelope(:w, :s, :e, :n, 4326)
  AND  ST_Intersects(geom, ST_MakeEnvelope(:w, :s, :e, :n, 4326));
```

### 8. Measure in a projection chosen per operation

Storing geographic coordinates does not mean measuring in them. Areas and distances need a projection appropriate to the extent and the question — equal-area for areas, equidistant for distances near a focus — chosen at the point of use rather than baked into storage.

```python
def metric_crs_for(geom) -> CRS:
    """A locally accurate metric frame for measuring near this geometry."""
    lon, lat = geom.centroid.x, geom.centroid.y
    return CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs")


def area_m2(geom) -> float:
    """Area in square metres via an equal-area projection — never from degrees."""
    equal_area = CRS.from_epsg(6933)
    tf = Transformer.from_crs(CRS.from_epsg(CANONICAL_EPSG), equal_area, always_xy=True)
    return shapely_transform(tf.transform, geom).area
```

Calling `.area` on geographic coordinates returns square degrees, a unit whose relationship to square metres varies with latitude. It is the single most common measurement error in spatial code, it produces numbers that look plausible, and it is entirely prevented by never exposing a raw area accessor to callers.

## Failure Modes & Root Causes

**The silent assumption.** Undeclared coordinates are treated as geographic and everything downstream inherits a datum-sized error. Root cause: a default where there should be a rejection. Mitigation: `require_frame` raises, and the rejection queue makes the volume visible.

**The datum shift nobody noticed.** Two sources agree on the frame name and disagree on the realisation, so features sit consistently offset. Root cause: a code that constructs but is superseded. Mitigation: the deprecation warning at the gate, plus periodic re-validation against the current registry.

**The swapped pair that landed somewhere plausible.** A latitude-first source in a region where the swap does not obviously break produces features displaced by a few degrees. Root cause: checking by eye rather than systematically. Mitigation: the axis-order test on every geometry, not on a sample.

**Measurement in degrees.** Areas and distances computed on stored geographic coordinates, producing values that vary with latitude. Root cause: convenience accessors on the storage frame. Mitigation: measurement helpers that project first, and no direct `.area` in application code.

## Production Validation Protocols

1. **Round-trip gate.** Transform a representative point to the source frame and back; reject if the positional error exceeds one metre.
2. **Frame invariant.** Assert every stored geometry reports the canonical frame; a mismatch means something bypassed the gate.
3. **Rejection-rate indicator.** Track rejections by reason and alert on a step change; a spike in undeclared frames usually means an upstream export changed.
4. **Repair-rate indicator.** Track the share of geometries repaired at ingestion; a rise means a source's quality has changed and the repairs are masking it.
5. **Axis-order fixture.** Keep a fixture of known-swapped coordinates from each region you ingest, and assert each is rejected.
6. **Transformation reproducibility.** Assert that the recorded transformation note is identical across build environments; a difference means the grid files differ and positions will too.

<figure class="diagram">
<svg viewBox="16 42 728 192" role="img" aria-labelledby="crs-meas-t crs-meas-d" xmlns="http://www.w3.org/2000/svg"><title id="crs-meas-t">Where each frame belongs in the pipeline</title><desc id="crs-meas-d">Storage uses one global geographic frame, measurement uses a projection chosen per operation, and display uses whatever the map expects — three different frames with three different jobs.</desc><rect x="16" y="42" width="728" height="192" fill="#ffffff"/><rect x="30" y="56" width="220" height="120" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="270" y="56" width="220" height="120" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="510" y="56" width="220" height="120" rx="8" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="140" y="86">storage</text><text x="380" y="86">measurement</text><text x="620" y="86">display</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="140" y="114">one global frame</text><text x="140" y="138">unambiguous, portable</text><text x="140" y="160">never measured in</text><text x="380" y="114">chosen per operation</text><text x="380" y="138">equal-area for areas</text><text x="380" y="160">equidistant for distance</text><text x="620" y="114">whatever the map wants</text><text x="620" y="138">transformed at render</text><text x="620" y="160">never stored back</text></g><text x="380" y="216" fill="#1f2937" font-size="13" text-anchor="middle">Most measurement errors come from using the storage frame for the middle column</text></svg>
<figcaption><b>Three jobs, three frames.</b> The temptation is to pick one frame that is adequate for all three; the result is a system that stores portably, measures inaccurately, and renders after a transformation it did not record.</figcaption>
</figure>

## Operating the Gate Over Time

A normalization gate is easy to build and easy to erode, and the erosion is always for a good reason. Three pressures account for most of it.

The first is the urgent import. A dataset is needed today, it has no declared frame, and someone adds a bypass "temporarily". Bypasses are not temporary, and the geometry they admit is indistinguishable from the rest a month later. The workable answer is not to refuse the import but to make the declaration explicit: a per-source configuration entry recording the assumed frame and the evidence for it. That is still an assumption, but it is a documented one attached to a source rather than an invisible one attached to a code path.

The second is the source that changes without telling you. Exports get regenerated with different settings, an upstream system upgrades its libraries, and a feed that declared its frame for two years stops doing so. The rejection-rate indicator is what catches this, and it needs to be watched as a rate rather than as an absolute — a hundred rejections a day is fine if it has always been a hundred, and alarming if it was two last week.

The third is registry drift. Datum realisations are superseded, codes are deprecated, and transformation grids improve. A geometry normalised three years ago against the then-current transformation may sit half a metre from where the same input would land today. Whether that matters depends entirely on the questions being asked, which is why the transformation note is stored: it lets you determine, later, which records were processed under which path, and re-normalise only those that need it.

None of these is a code problem, and none is solved by making the gate stricter. They are all solved by making its behaviour visible — rejections by reason, repairs by source, transformation paths by build — so that a change in the input is distinguishable from a change in the pipeline. A gate whose statistics nobody looks at eventually becomes a gate whose bypasses nobody remembers.

One more habit is worth adopting early: keep a small corpus of known geometries with independently verified positions, drawn from each region you ingest, and run the full gate over it on every build. It costs seconds, it exercises the transformation path rather than the code around it, and it is the only test that will notice when a library upgrade quietly changes where things are.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should the canonical frame be geographic or projected?</span></summary><p>Geographic for storage in almost every case, because it is unambiguous globally, survives a corpus that grows into new regions, and is what every downstream tool expects by default. A projected canonical frame is defensible when the data will never leave one region and metric operations dominate, but it makes the first expansion into a neighbouring country a migration rather than an ingest. Measure in projections chosen per operation either way.</p></details>

<details class="faq-item"><summary><span>What tolerance is acceptable on the round-trip test?</span></summary><p>One metre is a reasonable default and is generous relative to what a good transformation achieves. Tightening it much below that starts failing on legitimate transformations whose published accuracy is worse — some datum shifts are only accurate to several metres — so the useful pattern is a global tolerance of one metre with per-frame exceptions recorded alongside the reason. What matters is that the tolerance is asserted rather than assumed.</p></details>

<details class="faq-item"><summary><span>Is rejecting undeclared geometry too strict for real data?</span></summary><p>It feels strict for about a week and then pays for itself. The alternative is not "accepting slightly worse data" but "accepting data whose error you cannot bound", and every measurement taken from it inherits that. Where a source genuinely never declares a frame but is known to use one, encode that knowledge as an explicit per-source default with a comment explaining the evidence — which is a declaration, just one you made.</p></details>

<details class="faq-item"><summary><span>What should happen to geometry that was already stored before the gate existed?</span></summary><p>Re-normalise it rather than grandfathering it, and record that you did. Legacy records with unknown provenance are the ones that will produce the inexplicable answer two years from now, and the effort to reprocess them is bounded and one-off. Where the original frame genuinely cannot be recovered, mark those records explicitly so a query can exclude them rather than silently mixing them with verified geometry.</p></details>

<details class="faq-item"><summary><span>How should vertical coordinates be handled?</span></summary><p>As a separate declaration, because a height is meaningless without its own reference surface and the two are frequently mismatched in the same file. If elevations matter to your questions, carry the vertical frame alongside the horizontal one and apply the same rejection rule; if they do not, drop them at the gate rather than storing numbers nobody can interpret.</p></details>

## Related

- Up to the section overview: [Spatial LLM Architecture and Core Concepts](/spatial-llm-architecture-core-concepts/)
- Technique: [Normalizing Mixed-Frame Data Before Ingestion](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/normalizing-mixed-crs-data-before-llm-ingestion/)
- Technique: [Choosing a Canonical Frame for Spatial LLM Pipelines](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/choosing-a-canonical-crs-for-llm-pipelines/)
- Technique: [Detecting Axis-Order Swaps in Coordinate Input](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/detecting-axis-order-swaps-in-coordinate-input/)
- Peer topic: [Geometry Tokenization Strategies](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/)
- Related topic: [Retrieval-Augmented CRS Resolution](/geospatial-rag-pipelines/retrieval-augmented-crs-resolution/)
