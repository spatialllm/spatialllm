---
title: Snapping and Noding LLM-Generated Geometries
description: Collapse near-coincident vertices and node unnoded crossings before validity enforcement, using a tolerance derived from the data's own precision.
slug: snapping-and-noding-llm-generated-geometries
type: howto
breadcrumb: Snapping and Noding
datePublished: 2025-04-24
dateModified: 2026-08-11
---

# Snapping and Noding LLM-Generated Geometries

An LLM asked to draft parcel boundaries or a road network will emit vertices that are *almost* coincident and intersections that cross without a shared node — geometry that looks right on a map but fails every topology rule the moment it hits PostGIS. This guide cleans that output with a snap-to-grid and noding pass *before* validity enforcement, so near-duplicate vertices collapse and every crossing gets a node. It is a preprocessing step within [topology rule enforcement via LLMs](/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/).

Two distinct defects dominate model output. First, *near-coincident vertices*: two points meant to be identical differ in the eighth decimal, leaving hairline slivers and gaps. Second, *unnoded intersections*: two lines cross geometrically but share no vertex, so a coverage or polygonization step treats them as unconnected. Snapping fixes the first; noding fixes the second. Run them in that order, then validate.

## When to Use This Approach

Reach for snap-and-node whenever geometries come from a generative source, a digitizer, or a merge of independently-authored layers — anywhere exact coordinate equality cannot be assumed. If your geometries already come from a topologically clean source, skip it; snapping is lossy and should not be applied gratuitously.

| Symptom | Root cause | Fix |
|---|---|---|
| Slivers / gaps at shared edges | Near-coincident vertices | Snap to a tolerance grid |
| "Non-noded intersection" error | Crossings without shared nodes | Node the linework |
| Persisting invalidity after snap/node | Degenerate rings, spikes | `make_valid` fallback |

Pick the snap tolerance from the data's real precision, not arbitrarily: too coarse and you merge genuinely distinct features, too fine and near-duplicates survive. Snapping and noding are prerequisites, not replacements, for [enforcing topological rules in LLM-generated geometries](/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/enforcing-topological-rules-in-llm-generated-geometries/) — they make the input clean enough that rule checks become meaningful. Do this in a projected CRS so tolerance is in meters; see [coordinate reference system normalization](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/).

<figure class="diagram">
<svg viewBox="16 38 748 178" role="img" aria-labelledby="snn-two-t snn-two-d" xmlns="http://www.w3.org/2000/svg"><title id="snn-two-t">Two distinct defects in generated linework</title><desc id="snn-two-d">Near-coincident vertices leave hairline slivers that snapping closes, while crossings without a shared node defeat polygonization and are fixed by noding.</desc><rect x="16" y="38" width="748" height="178" fill="#ffffff"/><rect x="30" y="52" width="340" height="150" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="410" y="52" width="340" height="150" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="200" y="84">near-coincident vertices</text><text x="580" y="84">unnoded crossings</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="200" y="114">two points meant to be one</text><text x="200" y="140">hairline slivers and gaps</text><text x="200" y="168">fixed by snapping</text><text x="580" y="114">lines cross with no shared node</text><text x="580" y="140">polygonization sees no connection</text><text x="580" y="168">fixed by noding</text></g></svg>
<figcaption><b>Two defects, two passes, one order.</b> Snapping first collapses the duplicates that would otherwise each acquire their own node, so noding has less to do and produces fewer artefacts.</figcaption>
</figure>

## Implementation

The routine below runs in Shapely for local cleaning with a PostGIS equivalent for set-wide noding. It snaps vertices to a grid, nodes the linework via a union, guards validity, and falls back to `make_valid` when residual defects remain — never returning an invalid geometry silently.

```python
import logging
from shapely import set_precision
from shapely.geometry import GeometryCollection
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union, polygonize
from shapely.validation import make_valid

log = logging.getLogger("snap_node")


class NodingError(Exception):
    pass


def snap_and_node(geom: BaseGeometry, grid_size_m: float = 0.01) -> BaseGeometry:
    """
    Snap vertices to `grid_size_m`, node all intersections, then guarantee validity.
    Tolerance is in the geometry's CRS units — use a projected CRS (meters).
    """
    if geom is None or geom.is_empty:
        raise NodingError("empty or null geometry")

    # 1. Snap to grid: collapses near-coincident vertices deterministically.
    try:
        snapped = set_precision(geom, grid_size=grid_size_m)
    except Exception as exc:
        log.warning("set_precision failed (%s); falling back to make_valid only", exc)
        return make_valid(geom)

    if snapped.is_empty:
        raise NodingError("geometry collapsed entirely at this grid size")

    # 2. Node the linework: unary_union inserts nodes at every crossing.
    try:
        noded = unary_union(snapped)
    except Exception as exc:
        log.error("noding via unary_union failed: %s", exc)
        return make_valid(snapped)

    # 3. Validity guard with deterministic fallback.
    if not noded.is_valid:
        log.info("noded output invalid; applying make_valid fallback")
        repaired = make_valid(noded)
        if not repaired.is_valid or repaired.is_empty:
            raise NodingError("geometry unrecoverable after snap+node+make_valid")
        return repaired
    return noded


def rebuild_polygons(lines: BaseGeometry, grid_size_m: float = 0.01) -> BaseGeometry:
    """Snap+node linework, then re-polygonize into clean faces."""
    noded = snap_and_node(lines, grid_size_m)
    faces = list(polygonize(noded))
    if not faces:
        log.warning("no polygons formed; returning noded linework")
        return noded
    return unary_union(faces)


# PostGIS equivalent for set-wide cleaning (run after snapping into a projected SRID):
POSTGIS_SNAP_NODE = """
WITH candidates AS (
    -- && bbox pre-filter narrows pairs via the GiST index before the exact test.
    SELECT a.id, ST_Node(
             ST_SnapToGrid(ST_MakeValid(a.geom), 0.01)
           ) AS geom
    FROM edges a
    JOIN edges b
      ON a.geom && b.geom
     AND ST_Intersects(a.geom, b.geom)
     AND a.id < b.id
)
SELECT id, geom
FROM candidates
WHERE ST_IsValid(geom);
"""


if __name__ == "__main__":
    from shapely.geometry import LineString
    # Two lines that cross without a shared node:
    a = LineString([(0, 0), (10, 10)])
    b = LineString([(0, 10), (10.0000001, -0.0000001)])
    cleaned = snap_and_node(unary_union([a, b]), grid_size_m=0.001)
    print("valid:", cleaned.is_valid, "type:", cleaned.geom_type)
```

Order matters: `set_precision` first so noding operates on already-collapsed vertices; `unary_union` to insert nodes; then `make_valid` only as a last resort. The PostGIS variant does the same set-wide, and its self-join leads with the `&&` bbox pre-filter so noding candidate pairs are found through the spatial index rather than a full cross product.

<figure class="diagram">
<svg viewBox="26 46 720 188" role="img" aria-labelledby="snn-tol-t snn-tol-d" xmlns="http://www.w3.org/2000/svg"><title id="snn-tol-t">Choosing a snap tolerance between two failures</title><desc id="snn-tol-d">A tolerance below the data's real precision leaves near-duplicates in place, while one above the smallest genuine feature merges distinct geometry.</desc><rect x="26" y="46" width="720" height="188" fill="#ffffff"/><rect x="40" y="60" width="220" height="110" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="276" y="60" width="220" height="110" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="512" y="60" width="220" height="110" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="150" y="92">too fine</text><text x="386" y="92">from the data</text><text x="622" y="92">too coarse</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="150" y="120">duplicates survive</text><text x="150" y="146">slivers remain</text><text x="386" y="120">capture precision</text><text x="386" y="146">duplicates collapse only</text><text x="622" y="120">distinct features merge</text><text x="622" y="146">data is destroyed</text></g><text x="390" y="216" fill="#1f2937" font-size="13" text-anchor="middle">The safe window is bounded below by precision and above by the smallest real feature</text></svg>
<figcaption><b>Both failures are silent.</b> Too fine leaves the problem you were fixing; too coarse produces valid geometry describing something that was never there.</figcaption>
</figure>

## Validation & Testing

- **Crossings gain a shared node.** Feed two lines that intersect without a common vertex; assert the result's coordinate set includes the intersection point and that `ST_IsValid`/`is_valid` holds.
- **Near-duplicates collapse.** Assert two vertices `0.0000001` apart snap to one at a `0.001` grid, and that vertex count strictly decreases.
- **Fallback recovers invalid input.** Pass a self-intersecting bowtie and assert `snap_and_node` returns a valid geometry (via the `make_valid` path) rather than raising, and that a fully degenerate input raises `NodingError` deterministically.

## Gotchas & Edge Cases

- **Tolerance too coarse merges real features.** A grid larger than the smallest legitimate gap will weld distinct parcels together. Derive `grid_size_m` from the data's true precision and test on the densest region before applying globally.
- **Snapping in a geographic CRS.** A grid of `0.01` in `EPSG:4326` is degrees, not centimeters, and varies with latitude. Reproject to a projected CRS first, consistent with [normalizing mixed CRS data before LLM ingestion](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/normalizing-mixed-crs-data-before-llm-ingestion/).
- **`make_valid` changing geometry type.** Repair can turn a Polygon into a GeometryCollection containing stray lines or points. Filter the collection back to the expected dimension before persisting, or downstream schema constraints will reject it.
- **Order inverted.** Noding before snapping leaves the near-coincident vertices in place, so the union produces micro-slivers instead of clean nodes. Always snap first, then node.

<figure class="diagram">
<svg viewBox="16 24 690 210" role="img" aria-labelledby="snn-order-t snn-order-d" xmlns="http://www.w3.org/2000/svg"><title id="snn-order-t">Why snapping precedes noding</title><desc id="snn-order-d">Noding first creates a node at every crossing including the spurious ones caused by duplicate vertices, so the subsequent snap has more artefacts to remove than it started with.</desc><rect x="16" y="24" width="690" height="210" fill="#ffffff"/><text x="30" y="62" fill="#b3324f" font-size="13" font-weight="600">node first</text><rect x="200" y="38" width="200" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="406" y="38" width="200" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="300" y="64" fill="#1f2937" font-size="12" text-anchor="middle">nodes on spurious crossings</text><text x="506" y="64" fill="#1f2937" font-size="12" text-anchor="middle">snap has more to clean</text><text x="30" y="152" fill="#12805c" font-size="13" font-weight="600">snap first</text><rect x="200" y="128" width="200" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="406" y="128" width="200" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="300" y="154" fill="#1f2937" font-size="12" text-anchor="middle">duplicates collapse</text><text x="506" y="154" fill="#1f2937" font-size="12" text-anchor="middle">noding sees real crossings</text><text x="390" y="216" fill="#1f2937" font-size="13" text-anchor="middle">Order is not a preference — reversing it creates work rather than saving it</text></svg>
<figcaption><b>The order is load-bearing.</b> Noding a layer with duplicate vertices manufactures nodes that the snap then has to remove, and each removal is another chance to move something that mattered.</figcaption>
</figure>

## Operating This Step Over Time

Tolerances chosen once tend to survive the data that justified them. A source that improves its capture precision, or one that is replaced by a coarser product, changes what the right tolerance is, and a constant that was derived carefully two years ago is now a number nobody can defend. Recording the derivation alongside the constant — which source, which stated accuracy, which date — makes a later review a five-minute exercise rather than an archaeology project.

The second thing to watch is the displacement distribution rather than the maximum. A snap where most vertices move a few millimetres and a handful move nearly the whole tolerance is behaving as intended; one where a large share move the maximum is a tolerance set too coarse, and the geometry is being reshaped rather than cleaned. Publishing the distribution on every run costs nothing and makes that visible before anyone notices it on a map.

Finally, keep the pre-snap geometry until the output has been validated. Snapping is lossy and occasionally wrong, and the ability to re-run with a different tolerance without re-fetching the source is worth the storage for as long as the pipeline is being tuned.

<details class="faq-item"><summary><span>Should the tolerance be uniform across a layer?</span></summary><p>Usually yes, because a varying tolerance produces geometry whose accuracy depends on where it is, which is difficult to describe and harder to defend. The exception is a layer assembled from sources of genuinely different precision, where a per-source tolerance is more honest — provided the source of each feature is recorded, so the tolerance applied to it can be stated later.</p></details>

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should snapping run in the storage frame or a projected one?</span></summary><p>A projected one, always, so the tolerance is a distance rather than a coordinate difference. A tolerance expressed in degrees is a different distance at every latitude, which means a corpus spanning a continent is snapped inconsistently by a constant that looks uniform. Project, snap, node, validate, and transform back.</p></details>

<details class="faq-item"><summary><span>How do I know the snap did not move something that mattered?</span></summary><p>Measure the maximum vertex displacement and compare it against the tolerance — they should agree. A displacement larger than the tolerance means something other than snapping moved the geometry, usually a repair applied afterwards, and that is worth investigating before the output is stored.</p></details>

<details class="faq-item"><summary><span>What should happen when snapping produces an invalid geometry?</span></summary><p>Repair it deterministically and record that the repair happened. Collapsing two vertices onto one can turn a thin sliver into a zero-area spike, which is a legitimate consequence of the snap rather than a bug — but the resulting geometry is not the one the source published, and anything measured from it inherits that difference.</p></details>

<details class="faq-item"><summary><span>Is it worth snapping to a grid rather than to neighbours?</span></summary><p>Grid snapping is simpler, faster and slightly more destructive, because it moves every vertex rather than only the ones that were nearly coincident. For generated geometry, where near-duplicates are the dominant defect, it is usually the better trade; for surveyed data it is not, and snapping only the vertices within tolerance of each other preserves more.</p></details>

## Related

- Up to the section: [Topology Rule Enforcement via LLMs](/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/)
- [Enforcing Topological Rules in LLM-Generated Geometries](/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/enforcing-topological-rules-in-llm-generated-geometries/)
- [Coordinate Reference System Normalization](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/)
- [Normalizing Mixed CRS Data Before LLM Ingestion](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/normalizing-mixed-crs-data-before-llm-ingestion/)
