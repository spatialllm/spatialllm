# Snapping and Noding LLM-Generated Geometries

An LLM asked to draft parcel boundaries or a road network will emit vertices that are *almost* coincident and intersections that cross without a shared node — geometry that looks right on a map but fails every topology rule the moment it hits PostGIS. This guide cleans that output with a snap-to-grid and noding pass *before* validity enforcement, so near-duplicate vertices collapse and every crossing gets a node. It is a preprocessing step within [topology rule enforcement via LLMs](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/).

Two distinct defects dominate model output. First, *near-coincident vertices*: two points meant to be identical differ in the eighth decimal, leaving hairline slivers and gaps. Second, *unnoded intersections*: two lines cross geometrically but share no vertex, so a coverage or polygonization step treats them as unconnected. Snapping fixes the first; noding fixes the second. Run them in that order, then validate.

## When to Use This Approach

Reach for snap-and-node whenever geometries come from a generative source, a digitizer, or a merge of independently-authored layers — anywhere exact coordinate equality cannot be assumed. If your geometries already come from a topologically clean source, skip it; snapping is lossy and should not be applied gratuitously.

| Symptom | Root cause | Fix |
|---|---|---|
| Slivers / gaps at shared edges | Near-coincident vertices | Snap to a tolerance grid |
| "Non-noded intersection" error | Crossings without shared nodes | Node the linework |
| Persisting invalidity after snap/node | Degenerate rings, spikes | `make_valid` fallback |

Pick the snap tolerance from the data's real precision, not arbitrarily: too coarse and you merge genuinely distinct features, too fine and near-duplicates survive. Snapping and noding are prerequisites, not replacements, for [enforcing topological rules in LLM-generated geometries](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/enforcing-topological-rules-in-llm-generated-geometries/) — they make the input clean enough that rule checks become meaningful. Do this in a projected CRS so tolerance is in meters; see [coordinate reference system normalization](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/).

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

## Validation & Testing

- **Crossings gain a shared node.** Feed two lines that intersect without a common vertex; assert the result's coordinate set includes the intersection point and that `ST_IsValid`/`is_valid` holds.
- **Near-duplicates collapse.** Assert two vertices `0.0000001` apart snap to one at a `0.001` grid, and that vertex count strictly decreases.
- **Fallback recovers invalid input.** Pass a self-intersecting bowtie and assert `snap_and_node` returns a valid geometry (via the `make_valid` path) rather than raising, and that a fully degenerate input raises `NodingError` deterministically.

## Gotchas & Edge Cases

- **Tolerance too coarse merges real features.** A grid larger than the smallest legitimate gap will weld distinct parcels together. Derive `grid_size_m` from the data's true precision and test on the densest region before applying globally.
- **Snapping in a geographic CRS.** A grid of `0.01` in `EPSG:4326` is degrees, not centimeters, and varies with latitude. Reproject to a projected CRS first, consistent with [normalizing mixed CRS data before LLM ingestion](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/normalizing-mixed-crs-data-before-llm-ingestion/).
- **`make_valid` changing geometry type.** Repair can turn a Polygon into a GeometryCollection containing stray lines or points. Filter the collection back to the expected dimension before persisting, or downstream schema constraints will reject it.
- **Order inverted.** Noding before snapping leaves the near-coincident vertices in place, so the union produces micro-slivers instead of clean nodes. Always snap first, then node.

## Related

- Up to the section: [Topology Rule Enforcement via LLMs](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/)
- [Enforcing Topological Rules in LLM-Generated Geometries](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/enforcing-topological-rules-in-llm-generated-geometries/)
- [Coordinate Reference System Normalization](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/)
- [Normalizing Mixed CRS Data Before LLM Ingestion](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/normalizing-mixed-crs-data-before-llm-ingestion/)
