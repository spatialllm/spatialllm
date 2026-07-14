# Normalizing Mixed CRS Data Before LLM Ingestion

When geospatial datasets enter generative pipelines, coordinate reference system (CRS) heterogeneity is the primary vector for silent degradation. LLMs consume tokenized geometries, bounding boxes, and spatial relationships as structured context. If the underlying coordinate space is inconsistent, tokenization strategies fracture, spatial embeddings diverge, and context windows fill with mathematically incompatible vectors. **Normalizing Mixed CRS Data Before LLM Ingestion** is not a preprocessing luxury; it is a deterministic requirement for reproducible spatial reasoning. This workflow sits at the foundation of [Spatial LLM Architecture & Core Concepts](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/), where coordinate consistency dictates downstream embedding fidelity, attention routing, and query fallback behavior.

## Failure Modes & Root Causes

The most insidious failure mode occurs when mixed CRS datasets bypass explicit validation before entering tokenization layers. A pipeline ingesting GeoJSON (implicitly WGS84/EPSG:4326 per [RFC 7946](https://datatracker.ietf.org/doc/html/rfc7946)) alongside PostGIS exports in UTM zones like EPSG:32633 will produce coordinate magnitudes that differ by orders of magnitude. Tokenizers interpret these raw floats as distinct semantic spaces, causing spatial embedding models to cluster unrelated geometries. Root causes typically include:

- **Implicit CRS assumptions**: `geopandas.read_file()` silently assigns `None` when `.prj` files are missing, causing downstream `to_crs()` calls to fail or default to planar Cartesian math that misaligns with spherical tokenizers.
- **Datum shift degradation**: `pyproj.Transformer` calls without `always_xy=True` introduce sub-meter drift that compounds across batch transformations, corrupting high-precision spatial queries.
- **Anti-meridian crossing artifacts**: Geometries spanning ±180° longitude generate bounding boxes with inverted min/max values, triggering NaN propagation in attention layers and breaking spatial indexing structures like R-trees or H3.
- **Precision bloat**: Raw floating-point coordinates retain 15+ decimal places, inflating token sequences without improving spatial reasoning accuracy. LLMs require bounded precision aligned to geodetic tolerances (typically 6–8 decimal places for meter-scale accuracy).

## Validation & Safety Gates

Before any transformation, enforce strict metadata auditing. Every geometry batch must pass a CRS resolution gate. If the source declares `None` or `UNKNOWN`, apply a deterministic fallback based on geographic centroid heuristics or explicit pipeline configuration. Implement bounds validation immediately after projection: reject coordinates outside `[-180, 180]` for longitude and `[-90, 90]` for latitude in WGS84, or enforce metric bounds for projected systems. This aligns with established [Coordinate Reference System Normalization](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/) protocols that prevent coordinate drift during batch ingestion.

Safety checks must also verify topology preservation. Projection can introduce self-intersections or collapsed rings when crossing zone boundaries. The following validation gate demonstrates explicit error handling, bounds checking, and topology verification:

```python
import geopandas as gpd
import numpy as np
from shapely.validation import make_valid

def validate_geospatial_batch(gdf: gpd.GeoDataFrame, target_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Enforce CRS resolution, bounds validation, and topology gates before LLM ingestion.
    """
    if gdf.crs is None:
        raise ValueError("CRS is undefined. Cannot safely normalize without explicit source projection.")

    # Gate 1: Topology repair
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(lambda geom: make_valid(geom) if geom else None)
    invalid_mask = gdf["geometry"].isna() | gdf["geometry"].is_empty
    if invalid_mask.any():
        gdf = gdf[~invalid_mask]

    # Gate 2: CRS transformation
    gdf = gdf.to_crs(target_crs)

    # Gate 3: Coordinate bounds validation (WGS84 assumed for LLM tokenization)
    bounds = gdf.total_bounds
    if not (-180 <= bounds[0] and bounds[2] <= 180 and -90 <= bounds[1] and bounds[3] <= 90):
        raise ValueError(f"Coordinates exceed WGS84 bounds: {bounds}. Anti-meridian crossing or projection error detected.")

    # Gate 4: Precision truncation (prevents token bloat)
    import shapely
    gdf["geometry"] = gdf["geometry"].apply(
        lambda geom: shapely.set_precision(geom, 1e-7)  # ~1 cm precision
    )

    return gdf
```

## Deterministic Transformation Pipeline

Once validation passes, the transformation step must be stateless, reproducible, and grid-aware. Relying on default PROJ transformations without explicit datum shift parameters introduces non-deterministic behavior across environments. Use `pyproj` directly for batch operations with explicit axis ordering:

```python
import pyproj
from pyproj import Transformer
from shapely.ops import transform

def deterministic_transform(gdf: gpd.GeoDataFrame, target_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Apply deterministic CRS transformation with explicit axis ordering.
    Works for all geometry types (points, lines, polygons).
    """
    if gdf.crs is None:
        raise ValueError("Source CRS is undefined. Cannot transform.")

    try:
        transformer = Transformer.from_crs(
            gdf.crs, target_crs, always_xy=True, accuracy=0.01
        )
        gdf = gdf.copy()
        gdf.geometry = gdf.geometry.apply(
            lambda geom: transform(transformer.transform, geom) if geom and not geom.is_empty else geom
        )
        gdf = gdf.set_crs(target_crs, allow_override=True)
    except pyproj.exceptions.ProjError as e:
        raise RuntimeError(f"CRS transformation failed: {e}. Verify grid files or fallback to centroid approximation.")

    return gdf
```

For polygon/line geometries, `geopandas.to_crs()` is typically the most efficient path; the function above is useful when you need custom per-geometry handling or logging. Always cache transformation grids via `PROJ_NETWORK=ON` or local CDN mirrors to prevent sub-meter drift in distributed training environments.

## Tokenization & Embedding Alignment

Normalized coordinates directly dictate how spatial features map into token space. When CRS heterogeneity is resolved, bounding boxes, centroid vectors, and vertex sequences align with the tokenizer's expected numerical distribution. This prevents spatial embedding models from learning artificial clusters based on projection artifacts rather than true geographic proximity.

Key alignment strategies:
1. **Quantized Bounding Box Tokens**: Convert normalized WGS84 bounds into fixed-length integer sequences using geohash or discrete global grid systems (e.g., H3, S2). This compresses spatial context while preserving topological adjacency.
2. **Vertex Sequence Padding**: Standardize polygon vertex counts to fixed lengths. Pad or truncate after normalization to ensure consistent attention mask lengths across the batch.
3. **Relative Offset Encoding**: Instead of absolute coordinates, encode deltas from a regional centroid. This reduces floating-point variance and improves gradient stability in spatial fine-tuning.

These strategies feed directly into context window optimization, ensuring that spatial tokens do not dominate the prompt budget while maintaining query resolution.

## Pipeline Integration & Next Steps

Integrating CRS normalization into production LLM pipelines requires deterministic CI/CD gates, monitoring hooks, and explicit fallback routing. Follow these actionable next steps:

1. **Pre-Ingestion Hook**: Attach the validation gate as a mandatory step in your data ingestion DAG (Airflow, Dagster, or Prefect). Reject batches that fail bounds or topology checks before they reach the embedding layer.
2. **Drift Monitoring**: Log CRS metadata, transformation accuracy, and precision truncation rates. Set alerts for batches where more than 2% of geometries require topology repair or exceed WGS84 bounds.
3. **Fallback Routing**: When normalization fails or CRS metadata is irrecoverable, route the batch to a fallback handler that extracts centroid coordinates, applies a conservative buffer, and tags the payload with `spatial_confidence: low`. This prevents silent degradation while preserving pipeline throughput.
4. **Embedding Validation**: After tokenization, run a cosine similarity sanity check against a known geographic baseline. If spatial embeddings diverge by more than 15% from the normalized baseline, trigger a re-ingestion with stricter precision thresholds.
5. **Documentation & Versioning**: Pin `pyproj` and `geopandas` versions in your environment. CRS transformation behavior changes across PROJ releases; lock your stack and document the exact grid files used for reproducibility.

By enforcing strict CRS normalization, you eliminate the silent coordinate drift that corrupts spatial reasoning, tokenization efficiency, and attention routing. Treat coordinate consistency as a first-class pipeline constraint, and your geospatial LLM workflows will scale deterministically across heterogeneous data sources.
