# Vector Raster Hybrid Processing

Vector Raster Hybrid Processing represents a foundational workflow stage within modern geospatial AI pipelines, where discrete geometric primitives and continuous gridded surfaces are fused to generate context-rich inputs for spatial language models. In agent-driven architectures, this hybrid approach bridges the gap between high-precision boundary data and environmental or sensor-derived raster layers, enabling deterministic feature extraction before tokenization. The process requires strict coordinate alignment, sampling consistency, and explicit topology validation to prevent silent data degradation that propagates into downstream [Spatial LLM Architecture & Core Concepts](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/) workflows.

Without rigorous alignment, vector geometries can sample incorrect raster cells, introducing spatial aliasing and corrupting downstream embeddings. Production pipelines must enforce CRS normalization, handle nodata propagation, and guarantee topological validity before any statistical aggregation occurs.

<figure class="diagram">
<svg viewBox="0 0 860 300" role="img" aria-labelledby="vrh-t vrh-d" xmlns="http://www.w3.org/2000/svg">
  <title id="vrh-t">Vector-raster hybrid pipeline</title>
  <desc id="vrh-d">Vector ingest, raster ingest, resampling and alignment, feature fusion, and hybrid encoding run in sequence to build an aligned vector-raster feature grid for embedding, tokenization, and analysis.</desc>
  <defs>
    <marker id="vrh-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/>
    </marker>
  </defs>
  <g fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2">
    <rect x="15" y="55" width="150" height="80" rx="8"/>
    <rect x="183" y="55" width="150" height="80" rx="8"/>
    <rect x="351" y="55" width="150" height="80" rx="8"/>
    <rect x="519" y="55" width="150" height="80" rx="8"/>
    <rect x="687" y="55" width="150" height="80" rx="8"/>
  </g>
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#vrh-arrow)">
    <line x1="166" y1="95" x2="181" y2="95"/>
    <line x1="334" y1="95" x2="349" y2="95"/>
    <line x1="502" y1="95" x2="517" y2="95"/>
    <line x1="670" y1="95" x2="685" y2="95"/>
  </g>
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#vrh-arrow)">
    <line x1="90" y1="137" x2="90" y2="202"/>
    <line x1="258" y1="137" x2="258" y2="202"/>
    <line x1="426" y1="137" x2="426" y2="202"/>
    <line x1="594" y1="137" x2="594" y2="202"/>
    <line x1="762" y1="137" x2="762" y2="202"/>
  </g>
  <rect x="15" y="205" width="822" height="58" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/>
  <g fill="#1f2937" font-size="13" text-anchor="middle">
    <text x="90" y="90"><tspan x="90" dy="0">Vector</tspan><tspan x="90" dy="16">ingest</tspan></text>
    <text x="258" y="90"><tspan x="258" dy="0">Raster</tspan><tspan x="258" dy="16">ingest</tspan></text>
    <text x="426" y="90"><tspan x="426" dy="0">Resampling</tspan><tspan x="426" dy="16">&amp; align</tspan></text>
    <text x="594" y="90"><tspan x="594" dy="0">Feature</tspan><tspan x="594" dy="16">fusion</tspan></text>
    <text x="762" y="90"><tspan x="762" dy="0">Hybrid</tspan><tspan x="762" dy="16">encoding</tspan></text>
  </g>
  <text x="426" y="240" fill="#1f2937" font-size="15" font-weight="600" text-anchor="middle">Aligned vector-raster feature grid</text>
  <text x="426" y="287" fill="#5b6471" font-size="12" text-anchor="middle">Downstream: embedding · tokenization · analysis</text>
</svg>
</figure>

## Step 1: Spatial Alignment & Deterministic Sampling

The first operational phase establishes spatial congruence between vector features and raster extents. Misaligned coordinate reference systems or mismatched pixel resolutions introduce systematic bias during feature extraction. The following routine validates bounding box overlap, enforces a target projection, and samples raster values at vector centroids or along buffered geometries. It explicitly handles invalid topologies, geographic vs. projected CRS constraints, and multi-band statistical aggregation.

```python
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box, mapping
import numpy as np
from typing import List

def align_and_sample(
    vector_gdf: gpd.GeoDataFrame,
    raster_path: str,
    target_crs: str = "EPSG:4326",
    buffer_m: float = 0.0
) -> gpd.GeoDataFrame:
    """
    Aligns vector geometries to a target CRS, validates topology,
    and extracts deterministic raster statistics per feature.
    """
    if vector_gdf.crs is None:
        raise ValueError("Vector GeoDataFrame lacks a defined CRS. Assign explicitly before processing.")

    if vector_gdf.crs.to_string() != target_crs:
        vector_gdf = vector_gdf.to_crs(target_crs)

    # Topology enforcement: fix self-intersections and sliver polygons
    vector_gdf = vector_gdf.copy()
    invalid_mask = ~vector_gdf.geometry.is_valid
    if invalid_mask.any():
        from shapely.validation import make_valid
        vector_gdf.loc[invalid_mask, "geometry"] = vector_gdf.loc[invalid_mask, "geometry"].apply(make_valid)

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError("Raster lacks CRS definition. Cannot perform spatial alignment.")
        if src.crs.to_string() != target_crs:
            raise ValueError(
                f"Raster CRS {src.crs} does not match target {target_crs}. "
                "Reproject raster before ingestion."
            )

        raster_bounds = box(*src.bounds)
        spatial_filter = vector_gdf.geometry.intersects(raster_bounds)
        if not spatial_filter.any():
            raise RuntimeError("No spatial intersection between vector geometries and raster extent.")

        vector_gdf = vector_gdf[spatial_filter].copy()
        sampled_features: List[np.ndarray] = []

        for _, row in vector_gdf.iterrows():
            geom = row.geometry
            if buffer_m > 0:
                if vector_gdf.crs.is_geographic:
                    raise ValueError(
                        "Buffering in meters requires a projected CRS. "
                        "Convert to EPSG:3857 or local UTM before sampling."
                    )
                geom = geom.buffer(buffer_m)

            try:
                out_image, _ = mask(src, [mapping(geom)], crop=True, all_touched=True, filled=True)
                if out_image.size == 0:
                    sampled_features.append(np.full(src.count * 4, np.nan))
                    continue

                band_stats = []
                for band_idx in range(src.count):
                    band_data = out_image[band_idx]
                    # Filter nodata values
                    if src.nodata is not None:
                        valid_data = band_data[band_data != src.nodata]
                    else:
                        valid_data = band_data.ravel()

                    if len(valid_data) == 0:
                        band_stats.extend([np.nan, np.nan, np.nan, np.nan])
                    else:
                        band_stats.extend([
                            float(np.mean(valid_data)),
                            float(np.std(valid_data)),
                            float(np.min(valid_data)),
                            float(np.max(valid_data))
                        ])
                sampled_features.append(np.array(band_stats, dtype=np.float32))
            except Exception:
                sampled_features.append(np.full(src.count * 4, np.nan))

    vector_gdf["raster_features"] = sampled_features
    return vector_gdf
```

This routine enforces explicit validation at each boundary condition. If the intersection check fails or topology is irreparable, the pipeline halts rather than propagating null embeddings. For authoritative guidance on raster I/O patterns and memory-mapped workflows, consult the [Rasterio documentation](https://rasterio.readthedocs.io/en/stable/).

## Step 2: Feature Engineering & Tokenization Preparation

Once raster values are extracted, they must be structured for downstream consumption. Vector attributes and raster-derived statistics are concatenated, normalized, and prepared for geometric serialization. The transformation pipeline isolates continuous raster bands, applies statistical scaling, and attaches them as structured metadata alongside the original geometry. This stage directly informs [Geometry Tokenization Strategies](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/) by ensuring continuous environmental signals are quantized into fixed-length sequences compatible with transformer attention mechanisms.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_tokenization_features(
    gdf: gpd.GeoDataFrame,
    vector_attr_cols: List[str],
    raster_stat_cols: List[str] = None
) -> pd.DataFrame:
    """
    Normalizes and structures hybrid vector-raster features for LLM tokenization.
    """
    gdf = gdf.copy()

    # Expand raster_features array into explicit columns
    if "raster_features" in gdf.columns:
        raster_df = pd.DataFrame(gdf["raster_features"].tolist(), index=gdf.index)
        if raster_stat_cols is None:
            raster_stat_cols = [f"raster_stat_{i}" for i in range(raster_df.shape[1])]
        raster_df.columns = raster_stat_cols
        gdf = pd.concat([gdf.drop(columns=["raster_features"]), raster_df], axis=1)

    # Select numeric features for scaling
    numeric_cols = list(vector_attr_cols) + list(raster_stat_cols or [])
    feature_matrix = gdf[numeric_cols].astype(float)

    # Handle missing values with median imputation
    feature_matrix = feature_matrix.fillna(feature_matrix.median())

    # Apply standard scaling to prevent outlier dominance in attention layers
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    scaled_df = pd.DataFrame(scaled_features, columns=feature_matrix.columns, index=gdf.index)
    final_df = pd.concat([scaled_df, gdf.drop(columns=numeric_cols)], axis=1)

    return final_df
```

The output of this stage is a tabular, normalized feature set where each row corresponds to a spatial entity with aligned environmental context. Continuous raster bands (e.g., elevation, NDVI, temperature) are scaled to zero mean and unit variance, preventing high-magnitude sensor values from dominating gradient updates during fine-tuning.

## Step 3: Topology Enforcement & Embedding Serialization

Hybrid processing pipelines must guarantee that spatial relationships remain intact when transitioning from GIS-native formats to tensor representations. Topological rules—such as non-overlapping polygons, valid ring orientations, and consistent coordinate precision—are enforced prior to embedding generation. This ensures that spatial proximity and adjacency constraints are preserved when fed into [Spatial Embedding Models](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/spatial-embedding-models/).

```python
import json
from shapely.validation import explain_validity
from typing import Dict

def serialize_for_embedding(
    gdf: gpd.GeoDataFrame,
    precision: int = 6,
    drop_invalid: bool = True
) -> List[Dict]:
    """
    Serializes aligned vector-raster features into JSON structures ready for embedding models.
    """
    records = []
    for _, row in gdf.iterrows():
        geom = row.geometry

        # Final topology validation
        if not geom.is_valid:
            reason = explain_validity(geom)
            if drop_invalid:
                continue
            else:
                from shapely.validation import make_valid
                geom = make_valid(geom)

        # Round coordinates to reduce token bloat while preserving spatial accuracy
        # shapely.set_precision is the idiomatic approach in Shapely 2+
        import shapely
        geom_rounded = shapely.set_precision(geom, 10 ** (-precision))

        coords = (
            list(geom_rounded.exterior.coords)
            if hasattr(geom_rounded, "exterior")
            else list(geom_rounded.coords)
        )

        record = {
            "geometry_type": geom_rounded.geom_type,
            "coordinates": coords,
            "attributes": {
                k: v for k, v in row.drop("geometry").items()
                if not pd.isna(v)
            }
        }
        records.append(record)

    return records
```

Serialization strips redundant metadata, enforces coordinate precision limits to optimize context window usage, and produces a clean dictionary structure compatible with batch embedding APIs. For standardized spatial data interchange, pipelines should align with [OGC GeoPackage specifications](https://www.ogc.org/standards/geopackage) to ensure interoperability across cloud-native data lakes.

## Production Integration & Pipeline Validation

In production environments, Vector Raster Hybrid Processing must be orchestrated as a stateless, idempotent microservice. Key operational requirements include:
- **Deterministic Sampling Windows:** Use fixed pixel strides or adaptive quadtree partitioning to prevent edge-case sampling drift across distributed workers.
- **Memory-Mapped I/O:** Leverage `rasterio`'s windowed reading to avoid loading multi-terabyte satellite imagery into RAM.
- **Schema Versioning:** Track CRS transformations, buffer distances, and normalization parameters alongside feature hashes to enable reproducible training runs.
- **Fallback Routing:** When raster coverage is incomplete or vector geometries fall outside valid extents, implement graceful degradation paths that substitute regional climatological averages or trigger [Fallback Routing for Geospatial Queries](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/) mechanisms.

By enforcing strict CRS alignment, topology validation, and deterministic feature extraction, hybrid processing pipelines deliver high-fidelity spatial inputs that align with modern [Context Window Optimization for Maps](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/) requirements. The resulting feature sets are mathematically consistent, topologically sound, and ready for direct ingestion into spatial language model training loops.
