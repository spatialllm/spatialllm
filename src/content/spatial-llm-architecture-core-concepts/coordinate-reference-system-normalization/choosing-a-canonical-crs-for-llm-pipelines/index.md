---
title: Choosing a Canonical CRS for Spatial LLM Pipelines
description: Pick one storage frame and a rule for measurement projections, so every geometry in the system is comparable and every metric answer is computed in real units.
slug: choosing-a-canonical-crs-for-llm-pipelines
type: howto
breadcrumb: Choosing a Canonical Frame
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Choosing a Canonical CRS for Spatial LLM Pipelines

Every spatial system converges on one storage frame, whether or not anyone chose it. Choosing deliberately takes an afternoon and settles a decade of arguments about why two distances disagree. This guide works through the decision and the rule that goes with it — one frame for storage, a projection selected per measurement — as the first step of [coordinate reference system normalization](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/).

## When to Use This Approach

Make the choice before the first geometry is stored, and revisit it only when the system's geographic scope changes fundamentally. Retrofitting a different canonical frame onto a populated corpus is a migration, not a configuration change.

| Situation | Storage frame | Reason |
|-----------|---------------|--------|
| Multi-country or growing scope | Global geographic | Survives expansion without migration |
| Single country, metric-heavy work | National projected, or global geographic | Either defensible; projected saves transforms |
| Data arrives in one national grid only | That grid, if scope is fixed | No transform on the hot path |
| Web mapping is the primary consumer | Global geographic, transform at render | Never store the display projection |

The last row is the one that causes the most damage. Storing geometry in a web mapping projection is convenient for one consumer and wrong for every measurement: that projection's distortion grows with latitude and areas computed in it are meaningless.

<figure class="diagram">
<svg viewBox="16 32 748 208" role="img" aria-labelledby="ccrs-three-t ccrs-three-d" xmlns="http://www.w3.org/2000/svg"><title id="ccrs-three-t">Three frames with three jobs</title><desc id="ccrs-three-d">One global geographic frame for storage, a projection chosen per operation for measurement, and whatever the map expects for display, with transformations flowing outward from storage.</desc><rect x="16" y="32" width="748" height="208" fill="#ffffff"/><rect x="290" y="46" width="200" height="72" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="390" y="76" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">storage</text><text x="390" y="98" fill="#5b6471" font-size="12" text-anchor="middle">one global geographic frame</text><rect x="30" y="160" width="230" height="66" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="520" y="160" width="230" height="66" rx="8" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><text x="145" y="188" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">measurement</text><text x="145" y="210" fill="#5b6471" font-size="12" text-anchor="middle">per operation, metric</text><text x="635" y="188" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">display</text><text x="635" y="210" fill="#5b6471" font-size="12" text-anchor="middle">at render, never stored</text><g stroke="#5b6471" stroke-width="2" marker-end="url(#ccrs-three-a)"><line x1="330" y1="122" x2="180" y2="156"/><line x1="450" y1="122" x2="600" y2="156"/></g><defs><marker id="ccrs-three-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/></marker></defs></svg>
<figcaption><b>Transformations point outward and never back.</b> The moment a display projection is written back into storage, the storage frame has two definitions and every comparison in the system becomes conditional.</figcaption>
</figure>

## Implementation

The choice is expressed as configuration plus two functions: one that declares the canonical frame, and one that selects a measurement projection for a given geometry and purpose.

```python
import logging
from dataclasses import dataclass
from typing import Literal

from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError

log = logging.getLogger("canonical_crs")

CANONICAL_EPSG = 4326                      # storage: global, unambiguous, portable
EQUAL_AREA_EPSG = 6933                     # global equal-area, for area measurement

Purpose = Literal["area", "distance", "display"]


@dataclass(frozen=True)
class Projection:
    crs: CRS
    why: str                               # recorded with any number it produces


def canonical() -> CRS:
    """The one frame everything is stored and compared in."""
    try:
        return CRS.from_epsg(CANONICAL_EPSG)
    except CRSError as exc:                # a broken installation, not a data problem
        raise RuntimeError(f"canonical frame EPSG:{CANONICAL_EPSG} unavailable: {exc}") from exc


def measurement_projection(geom, purpose: Purpose) -> Projection:
    """Select a metric projection appropriate to this geometry and this question."""
    if purpose == "area":
        return Projection(CRS.from_epsg(EQUAL_AREA_EPSG),
                          f"equal-area EPSG:{EQUAL_AREA_EPSG}")
    if purpose == "distance":
        lon, lat = geom.centroid.x, geom.centroid.y
        proj = CRS.from_proj4(
            f"+proj=aeqd +lat_0={lat:.6f} +lon_0={lon:.6f} "
            "+datum=WGS84 +units=m +no_defs")
        return Projection(proj, f"azimuthal equidistant centred on {lat:.4f}, {lon:.4f}")
    raise ValueError(f"no measurement projection for purpose {purpose!r}")


def to_projection(geom, projection: Projection):
    """Move geometry from the canonical frame into a measurement projection."""
    tf = Transformer.from_crs(canonical(), projection.crs, always_xy=True)
    from shapely.ops import transform
    try:
        return transform(tf.transform, geom)
    except Exception as exc:
        log.warning("projection to %s failed: %s", projection.why, exc)
        raise
```

Two decisions in that code are worth defending. The measurement projection is built per geometry for distance work, because a projection centred on the thing being measured has negligible distortion nearby and any fixed alternative does not. And the projection carries a `why` string, because a number without the projection that produced it cannot be reproduced or compared to another number produced differently.

Raising on an unknown purpose rather than falling back to the canonical frame is equally deliberate. A silent fallback would compute areas in square degrees, which is the single error this whole arrangement exists to prevent.

<figure class="diagram">
<svg viewBox="26 9 708 221" role="img" aria-labelledby="ccrs-dist-t ccrs-dist-d" xmlns="http://www.w3.org/2000/svg"><title id="ccrs-dist-t">Area error from measuring in the wrong projection</title><desc id="ccrs-dist-d">The same square kilometre measured in an equal-area projection, a web mapping projection and raw degrees, showing that only the first returns a usable figure and the error grows with latitude.</desc><rect x="26" y="9" width="708" height="221" fill="#ffffff"/><text x="380" y="34" fill="#5b6471" font-size="13" text-anchor="middle">One square kilometre at 56 degrees north, measured three ways</text><rect x="40" y="60" width="220" height="110" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="280" y="60" width="220" height="110" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="520" y="60" width="200" height="110" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="150" y="92">equal-area</text><text x="390" y="92">web mapping</text><text x="620" y="92">raw degrees</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="150" y="120">1.00 km²</text><text x="150" y="146">the answer</text><text x="390" y="120">3.20 km²</text><text x="390" y="146">inflated by latitude</text><text x="620" y="120">0.00016</text><text x="620" y="146">square degrees</text></g><text x="380" y="212" fill="#1f2937" font-size="13" text-anchor="middle">Only the middle figure looks like a plausible area, which is what makes it dangerous</text></svg>
<figcaption><b>The wrong answer that looks right.</b> Square degrees are obviously not an area and get caught; a web mapping projection returns square metres of the correct magnitude and is simply wrong by a factor that varies with latitude.</figcaption>
</figure>

## Validation & Testing

```python
from shapely.geometry import box


def test_area_is_measured_in_an_equal_area_projection():
    # A degree-sized box at two latitudes must not report the same area.
    south = box(-3.0, 10.0, -2.0, 11.0)
    north = box(-3.0, 56.0, -2.0, 57.0)
    a_south = to_projection(south, measurement_projection(south, "area")).area
    a_north = to_projection(north, measurement_projection(north, "area")).area
    assert a_south > a_north * 1.5, "equal-area projection is not being applied"


def test_distance_projection_is_centred_on_the_geometry():
    subject = box(-3.2, 55.9, -3.1, 56.0)
    proj = measurement_projection(subject, "distance")
    assert "aeqd" in proj.crs.to_proj4()
    assert "55.9" in proj.why or "55.95" in proj.why


def test_unknown_purpose_raises_rather_than_defaulting():
    try:
        measurement_projection(box(0, 0, 1, 1), "volume")
    except ValueError:
        return
    raise AssertionError("an unknown purpose must not silently use the storage frame")
```

The first test is a property rather than a fixture check: a degree box near the equator genuinely covers more ground than one near the pole, and any implementation that reports them as equal has skipped the projection. It needs no reference values and does not break when a library updates its constants.

Run these against the real configuration rather than a test double. The value of this whole arrangement is that one module decides the frames, and a test that stubs that module is testing something else.

## Gotchas & Edge Cases

**Storing the display projection.** Convenient for one consumer and corrupting for every measurement, because that projection's scale factor varies with latitude. Transform at render, and never write the result back.

**A national grid chosen before the scope was known.** The system expands into a neighbouring country and half the data is now outside the grid's area of use, where its accuracy degrades quietly. Prefer a global frame unless the scope is genuinely fixed by something other than current ambition.

**Per-geometry projections used for comparison.** Two areas measured in two different locally centred projections are not strictly comparable. Use the global equal-area projection for areas, which is the same everywhere, and reserve locally centred projections for distances, where the geometry being measured is the centre.

<figure class="diagram">
<svg viewBox="27 36 707 198" role="img" aria-labelledby="ccrs-scope-t ccrs-scope-d" xmlns="http://www.w3.org/2000/svg"><title id="ccrs-scope-t">A national grid outgrown by the corpus</title><desc id="ccrs-scope-d">A projected national frame serves its own territory well and degrades outside it, so a corpus that expands into neighbouring regions accumulates geometry in the frame&#8217;s poorly-behaved margins.</desc><rect x="27" y="36" width="707" height="198" fill="#ffffff"/><rect x="200" y="50" width="240" height="130" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="440" y="50" width="150" height="130" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="590" y="50" width="130" height="130" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="12.5" text-anchor="middle" font-weight="600"><text x="320" y="90">inside the area of use</text><text x="515" y="90">at the margin</text><text x="655" y="90">outside</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="320" y="120">accurate to centimetres</text><text x="320" y="146">the frame was designed for this</text><text x="515" y="120">accuracy degrades</text><text x="515" y="146">quietly</text><text x="655" y="120">no longer</text><text x="655" y="146">meaningful</text></g><text x="380" y="216" fill="#1f2937" font-size="13" text-anchor="middle">Nothing errors as the corpus expands rightward — the numbers simply stop being right</text></svg>
<figcaption><b>Area of use is a real boundary.</b> A projected frame is a fit to one region, and geometry stored outside it is not slightly worse but progressively meaningless — with no error at the moment the boundary is crossed.</figcaption>
</figure>

**A canonical frame declared but not enforced.** Configuration says one thing and a legacy import path writes another. The column type and check constraint described in the parent topic are what make the declaration structural rather than aspirational.

**Assuming the axis order of the canonical frame.** The global geographic frame is defined latitude-first in its authoritative definition and is handled longitude-first by most software. Pass `always_xy=True` consistently, and test it, or half the pipeline will disagree with the other half about which number is which.

Run the property tests on real geometry from the corpus rather than on synthetic boxes wherever you can. Synthetic shapes are axis-aligned and centred conveniently, which is exactly the configuration in which a projection error is smallest; a real parcel at a real latitude exercises the same code with the distortions that actually occur.

It is also worth asserting the axis-order convention explicitly in this test file rather than trusting it. Passing `always_xy=True` everywhere is the correct habit, and a single call site that omits it produces geometry that is silently transposed — a failure that looks like a data problem and is a configuration one.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Is a projected canonical frame ever the better choice?</span></summary><p>Yes, when the system genuinely will not leave one region and metric operations dominate every request. Storing in a national grid removes a transform from the hot path and makes distances directly computable, which is a real saving on a high-volume workload. The cost is that expansion becomes a migration, and expansion is easier to predict in hindsight than in advance — so take this route when the constraint is external, such as a regulatory boundary, rather than when it merely seems unlikely to change.</p></details>

<details class="faq-item"><summary><span>Which equal-area projection should be used globally?</span></summary><p>Any of the standard global equal-area options works, and the choice matters far less than using one consistently. What matters is that areas across the corpus are computed in the same projection, so that two figures can be compared, and that the projection is equal-area rather than merely metric — a metric projection that is not equal-area will happily report square metres that are wrong by tens of per cent at the edges of its zone.</p></details>

<details class="faq-item"><summary><span>How should the choice be recorded?</span></summary><p>In one module, as a constant with a comment explaining the reasoning, and in the database schema as a typed geometry column. Both matter: the constant is what application code reads, the column type is what stops a different code path from writing something else. A decision recorded only in a design document is one that will be contradicted by the second person to write an import script.</p></details>

<details class="faq-item"><summary><span>Does the canonical frame apply to raster data too?</span></summary><p>Not usefully. Rasters carry their own grids, and reprojecting them to match a vector canonical frame resamples every pixel, which is a loss and — for categorical data — a corruption. The rule for rasters is the inverse: bring the vector to the raster's frame for each operation, as described in <a href="/spatial-llm-architecture-core-concepts/vector-raster-hybrid-processing/">vector-raster hybrid processing</a>, and store rasters as they arrive.</p></details>

<details class="faq-item"><summary><span>What happens to the choice when a new region is added?</span></summary><p>With a global geographic storage frame, nothing — that is the property being bought. With a projected national frame it is a migration: every stored geometry must be transformed, every stored measurement recomputed, and every index rebuilt. Knowing that in advance is what makes the original choice a decision rather than a default, and it is worth writing the migration cost down next to the constant so the trade is visible to whoever inherits it.</p></details>

## Related

- Up to the parent topic: [Coordinate Reference System Normalization](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/)
- [Normalizing Mixed-Frame Data Before Ingestion](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/normalizing-mixed-crs-data-before-llm-ingestion/)
- [Detecting Axis-Order Swaps in Coordinate Input](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/detecting-axis-order-swaps-in-coordinate-input/)
- Concept: [Spatial Reasoning and Relation Inference](/spatial-llm-architecture-core-concepts/spatial-reasoning-and-relation-inference/)
