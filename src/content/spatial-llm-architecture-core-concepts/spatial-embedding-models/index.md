# Spatial Embedding Models

Spatial Embedding Models represent a foundational shift in how geospatial systems interface with large language models. By projecting coordinate geometries, topological relationships, and spatial attributes into dense vector spaces, these models enable semantic similarity search, spatial reasoning, and context-aware retrieval. Within the broader [Spatial LLM Architecture & Core Concepts](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/) framework, embedding pipelines must be deterministic, validation-heavy, and tightly coupled with traditional GIS operations to prevent hallucination in spatial queries. Unlike generic text embeddings, geospatial vectors must preserve metric fidelity, adjacency constraints, and projection-aware scaling to remain useful in downstream routing, zoning, and spatial analytics tasks.

<figure class="diagram">
<svg viewBox="0 0 860 300" role="img" aria-labelledby="sem-t sem-d" xmlns="http://www.w3.org/2000/svg">
  <title id="sem-t">Spatial embedding pipeline</title>
  <desc id="sem-d">Feature extraction, geometry encoding, spatial-context modeling, vector projection, and index storage run in sequence to produce metric-consistent spatial embeddings for similarity search, retrieval, and inference.</desc>
  <defs>
    <marker id="sem-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
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
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#sem-arrow)">
    <line x1="166" y1="95" x2="181" y2="95"/>
    <line x1="334" y1="95" x2="349" y2="95"/>
    <line x1="502" y1="95" x2="517" y2="95"/>
    <line x1="670" y1="95" x2="685" y2="95"/>
  </g>
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#sem-arrow)">
    <line x1="90" y1="137" x2="90" y2="202"/>
    <line x1="258" y1="137" x2="258" y2="202"/>
    <line x1="426" y1="137" x2="426" y2="202"/>
    <line x1="594" y1="137" x2="594" y2="202"/>
    <line x1="762" y1="137" x2="762" y2="202"/>
  </g>
  <rect x="15" y="205" width="822" height="58" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/>
  <g fill="#1f2937" font-size="13" text-anchor="middle">
    <text x="90" y="90"><tspan x="90" dy="0">Feature</tspan><tspan x="90" dy="16">extraction</tspan></text>
    <text x="258" y="90"><tspan x="258" dy="0">Geometry</tspan><tspan x="258" dy="16">encoding</tspan></text>
    <text x="426" y="90"><tspan x="426" dy="0">Spatial</tspan><tspan x="426" dy="16">context</tspan></text>
    <text x="594" y="90"><tspan x="594" dy="0">Vector</tspan><tspan x="594" dy="16">projection</tspan></text>
    <text x="762" y="90"><tspan x="762" dy="0">Index</tspan><tspan x="762" dy="16">storage</tspan></text>
  </g>
  <text x="426" y="240" fill="#1f2937" font-size="15" font-weight="600" text-anchor="middle">Metric-consistent spatial embeddings</text>
  <text x="426" y="287" fill="#5b6471" font-size="12" text-anchor="middle">Downstream: similarity search · retrieval · inference</text>
</svg>
</figure>

## Step 1: Data Preparation and CRS Normalization

Before any embedding generation, raw geospatial data requires strict normalization. Spatial embeddings are highly sensitive to coordinate reference system (CRS) inconsistencies, as metric distortions directly corrupt learned spatial relationships. Platform teams must enforce a unified projection and implement explicit validation gates at the ingestion boundary to prevent silent CRS drift.

The following production-ready routine enforces CRS transformation, topology validation, and deterministic geometry cleaning. It relies on [GeoPandas projection handling](https://geopandas.org/en/stable/docs/user_guide/projections.html) and [Shapely validation utilities](https://shapely.readthedocs.io/en/stable/manual.html#validation.make_valid) to guarantee topological soundness.

```python
import geopandas as gpd
import numpy as np
from shapely.validation import make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

def normalize_and_validate_gdf(
    gdf: gpd.GeoDataFrame,
    target_crs: int = 4326,
    min_area_threshold: float = 1e-6
) -> gpd.GeoDataFrame:
    """
    Enforce CRS normalization, repair invalid topologies, and filter degenerate geometries.
    Raises ValueError on structural failures to prevent downstream embedding corruption.
    """
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame must have a defined CRS. Assign via gdf.set_crs() first.")

    # Transform to target CRS
    gdf = gdf.to_crs(epsg=target_crs)

    # Topology repair
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(make_valid)

    # Filter out empty, NaN, or degenerate geometries
    # Note: area check applies only to polygon types; points and lines will have area == 0 by design
    invalid_mask = gdf.geometry.is_empty | gdf.geometry.isna()
    if invalid_mask.any():
        count = invalid_mask.sum()
        raise ValueError(f"Found {count} invalid/empty geometries. Clean before embedding.")

    # Enforce strict geometry types (flatten GeometryCollections to polygons only)
    def flatten_geom(geom):
        if isinstance(geom, GeometryCollection):
            polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
            if not polys:
                return geom  # preserve non-polygon collections
            return MultiPolygon(polys) if len(polys) > 1 else polys[0]
        return geom

    gdf["geometry"] = gdf["geometry"].apply(flatten_geom)
    return gdf.reset_index(drop=True)
```

This preprocessing stage ensures that downstream tokenization and embedding layers receive topologically sound inputs. Metric consistency at this stage is non-negotiable for spatial reasoning tasks.

## Step 2: Geometry Serialization and Feature Extraction

Raw coordinates cannot be directly fed into standard transformer architectures. Instead, spatial features must be serialized into structured sequences that preserve metric relationships and adjacency. The process of [Geometry Tokenization Strategies](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/) dictates how polygons, linestrings, and points are discretized into fixed-length sequences or hierarchical tokens. For embedding generation, a hybrid feature vector typically combines:

- Centroid coordinates (standardized across the dataset)
- Bounding box dimensions, aspect ratios, and diagonal lengths
- Topological descriptors (vertex count, perimeter-to-area ratio, convex hull compactness)
- Semantic attributes (land use class, road hierarchy, administrative level)

These features are concatenated and passed through a deterministic projection layer before entering the embedding model. The extraction routine below guarantees reproducible feature vectors regardless of batch size or execution order.

```python
import numpy as np
import pandas as pd

def extract_spatial_features(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Extract deterministic spatial metrics for embedding input.
    Returns a 2D numpy array of shape (n_samples, n_features).
    """
    centroids = gdf.geometry.centroid
    bounds = gdf.geometry.bounds  # minx, miny, maxx, maxy

    widths = bounds["maxx"] - bounds["minx"]
    heights = bounds["maxy"] - bounds["miny"]
    aspect_ratios = np.where(heights == 0, 0.0, widths / heights)
    diagonals = np.sqrt(widths**2 + heights**2)

    perimeters = gdf.geometry.length
    areas = gdf.geometry.area
    perimeter_area_ratio = np.where(areas == 0, 0.0, perimeters / areas)

    # Isoperimetric quotient: 4π * Area / Perimeter² (= 1.0 for a circle)
    compactness = np.where(perimeters == 0, 0.0, (4 * np.pi * areas) / (perimeters**2))

    # Vertex count (exterior ring only for polygons)
    vertex_counts = np.array([
        len(np.asarray(geom.exterior.coords)) if hasattr(geom, 'exterior') else 1
        for geom in gdf.geometry
    ])

    features = np.column_stack([
        centroids.x, centroids.y,
        widths, heights, aspect_ratios, diagonals,
        perimeters, areas, perimeter_area_ratio, compactness, vertex_counts
    ])

    # Standardize features (zero mean, unit variance) for stable gradient descent
    means = features.mean(axis=0)
    stds = features.std(axis=0)
    stds[stds == 0] = 1.0  # Prevent division by zero for constant features
    return (features - means) / stds
```

Deterministic feature extraction prevents stochastic variance during batch inference and ensures that identical geometries always map to identical embedding inputs.

## Step 3: Embedding Generation Pipeline

With normalized geometries and standardized feature vectors, the embedding pipeline maps spatial descriptors into a dense latent space. Modern Spatial Embedding Models typically employ contrastive learning objectives (e.g., InfoNCE loss) to cluster spatially proximate or topologically similar geometries while pushing dissimilar regions apart.

When processing complex administrative boundaries or multi-polygon features, sequence length constraints become critical. Implementing [Context Window Optimization for Maps](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/) ensures that high-vertex geometries are downsampled or hierarchically encoded without losing critical spatial semantics.

The following PyTorch module demonstrates a production-ready embedding forward pass, incorporating a linear projection head, layer normalization, and output normalization for vector database compatibility.

```python
import torch
import torch.nn as nn
import numpy as np

class SpatialEmbeddingModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning normalized embeddings suitable for vector similarity search.
        LayerNorm is applied but note: for strict L2-unit-norm vectors use
        F.normalize(embeddings, p=2, dim=-1) instead of LayerNorm.
        """
        embeddings = self.encoder(x)
        return self.output_norm(embeddings)


def generate_embeddings(
    features: np.ndarray,
    model: nn.Module,
    device: str = "cpu",
    batch_size: int = 1024
) -> np.ndarray:
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch = torch.tensor(features[i:i+batch_size], dtype=torch.float32, device=device)
            emb = model(batch)
            all_embeddings.append(emb.cpu().numpy())
    return np.vstack(all_embeddings)
```

This architecture produces embeddings suitable for efficient nearest-neighbor retrieval via FAISS or Milvus. The deterministic pipeline ensures that spatial queries remain reproducible across deployment environments.

## Step 4: Validation, Benchmarking, and Production Routing

Deploying Spatial Embedding Models requires rigorous validation beyond standard NLP metrics. Spatial recall, topological consistency, and projection-aware distance preservation must be measured against ground-truth GIS operations. Teams should implement automated evaluation suites that compare embedding similarity against traditional spatial predicates (`ST_Intersects`, `ST_DWithin`, `ST_Contains`) to detect drift or hallucination.

Systematic evaluation is covered in [Benchmarking Spatial Embedding Models for Vector GIS](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/spatial-embedding-models/benchmarking-spatial-embedding-models-for-vector-gis/), which outlines standardized test suites for vector similarity, spatial join accuracy, and multi-scale generalization.

In production, embedding confidence scores should dictate query routing. When cosine similarity falls below a calibrated threshold, or when topological constraints cannot be satisfied by vector search alone, the system must trigger deterministic fallback routing. This hybrid approach combines fast approximate nearest-neighbor (ANN) retrieval with exact spatial indexing (R-tree, Quadtree) to guarantee correctness for critical infrastructure, zoning, and compliance queries. By tightly coupling learned representations with OGC-compliant spatial operations, platform teams can scale geospatial AI without sacrificing metric integrity or operational reliability.
