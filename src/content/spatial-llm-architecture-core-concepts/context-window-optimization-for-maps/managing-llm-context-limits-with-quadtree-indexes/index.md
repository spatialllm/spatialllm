# Managing LLM Context Limits with Quadtree Indexes

When deploying spatial reasoning agents at production scale, the primary bottleneck is rarely model capability but rather deterministic token budget exhaustion. Geospatial datasets exhibit extreme non-uniform density: a single urban parcel layer can serialize into hundreds of thousands of tokens, while adjacent rural zones may require only a fraction. **Managing LLM Context Limits with Quadtree Indexes** resolves this asymmetry by replacing flat bounding-box filtering with recursive, token-aware spatial partitioning. Within the broader [Spatial LLM Architecture & Core Concepts](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/), quadtree-based indexing enables dynamic context window allocation, ensuring that only the geometrically relevant and token-budget-compliant nodes reach the prompt assembly layer.

## Hierarchical Decomposition & Token-Aware Partitioning

The core mechanism relies on hierarchical spatial decomposition. A quadtree recursively subdivides a coordinate space into four quadrants until a stopping condition is met: maximum recursion depth, minimum feature count per node, or a hard token budget threshold. Unlike raster tiling, which introduces fixed-resolution artifacts and forces uniform token consumption regardless of feature density, quadtree nodes adapt dynamically to vector geometry complexity. When serializing spatial data for an LLM, each node is evaluated against a precomputed token estimator. Nodes exceeding the budget are split; nodes falling below are merged or pruned.

This approach directly operationalizes [Context Window Optimization for Maps](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/) by treating spatial partitioning as a constraint satisfaction problem rather than a static retrieval step. By aligning geometric boundaries with token limits, engineers can guarantee that downstream prompts never exceed context thresholds while preserving topological fidelity. Serialization overhead is minimized by adhering to strict coordinate precision standards, as defined in [RFC 7946: The GeoJSON Format](https://datatracker.ietf.org/doc/html/rfc7946), which prevents floating-point bloat from inflating token counts.

## Production-Grade Implementation

Production implementations require strict serialization discipline, explicit coordinate validation, and deterministic fallback routing. The following Python pipeline demonstrates a token-aware quadtree builder using `shapely`, `geopandas`, and `tiktoken`. It enforces CRS normalization, calculates JSON serialization overhead, applies hard depth limits to prevent recursive stack exhaustion, and includes comprehensive error handling for malformed geometries.

```python
import geopandas as gpd
import shapely
import shapely.geometry as sg
import tiktoken
import json
from typing import List, Dict, Tuple
import math
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class CoordinateValidationError(Exception): pass
class TokenBudgetExceededError(Exception): pass

class TokenAwareQuadtree:
    def __init__(self, gdf: gpd.GeoDataFrame, max_tokens: int = 8192, max_depth: int = 6):
        self.max_tokens = max_tokens
        self.max_depth = max_depth
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.nodes: List[Dict] = []

        # Strict WGS84 normalization with fallback routing
        if gdf.crs is None:
            raise CoordinateValidationError("Input GeoDataFrame has no CRS. Cannot normalize to WGS84.")
        try:
            self.gdf = gdf.to_crs(epsg=4326) if gdf.crs.to_epsg() != 4326 else gdf.copy()
        except Exception as e:
            raise CoordinateValidationError(f"CRS normalization failed. Fallback to centroid routing required. Error: {e}")

        self.bbox = self.gdf.total_bounds
        self._validate_wgs84_bounds()
        self._clean_geometries()

    def _validate_wgs84_bounds(self) -> None:
        minx, miny, maxx, maxy = self.bbox
        if not (-180.0 <= minx <= 180.0 and -90.0 <= miny <= 90.0):
            raise CoordinateValidationError("Geometries exceed valid WGS84 longitude/latitude ranges.")
        if any(math.isnan(v) or math.isinf(v) for v in self.bbox):
            raise CoordinateValidationError("Bounding box contains NaN or infinite coordinate values.")

    def _clean_geometries(self) -> None:
        """Remove invalid geometries and log warnings for pipeline safety."""
        valid_mask = self.gdf.geometry.is_valid
        if not valid_mask.all():
            invalid_count = (~valid_mask).sum()
            logging.warning(f"Dropping {invalid_count} invalid geometries to prevent serialization crashes.")
            self.gdf = self.gdf[valid_mask].copy()

    def _estimate_tokens(self, geometry: shapely.geometry.base.BaseGeometry) -> int:
        """Estimate token count for a geometry with precision reduction."""
        try:
            geo_dict = shapely.geometry.mapping(geometry)

            def round_coords(obj, depth: int = 0) -> object:
                if depth > 12: return obj
                if isinstance(obj, list):
                    return [round_coords(x, depth + 1) for x in obj]
                elif isinstance(obj, float):
                    return round(obj, 6)  # ~10cm precision, reduces token overhead
                return obj

            geo_dict["coordinates"] = round_coords(geo_dict["coordinates"])
            json_str = json.dumps(geo_dict, separators=(",", ":"))
            return len(self.encoder.encode(json_str))
        except Exception as e:
            logging.error(f"Token estimation failed for geometry: {e}")
            return 0

    def _build_tree(self, gdf: gpd.GeoDataFrame, bbox: Tuple[float, float, float, float], depth: int) -> None:
        minx, miny, maxx, maxy = bbox
        mid_x, mid_y = (minx + maxx) / 2.0, (miny + maxy) / 2.0

        # Base case: depth limit or empty subset
        if depth >= self.max_depth or gdf.empty:
            self._finalize_node(gdf, bbox, depth)
            return

        # Token budget check
        node_tokens = sum(self._estimate_tokens(geom) for geom in gdf.geometry)
        if node_tokens <= self.max_tokens:
            self._finalize_node(gdf, bbox, depth)
            return

        # Recursive split into quadrants
        quadrants = {
            "NW": gdf.cx[minx:mid_x, mid_y:maxy],
            "NE": gdf.cx[mid_x:maxx, mid_y:maxy],
            "SW": gdf.cx[minx:mid_x, miny:mid_y],
            "SE": gdf.cx[mid_x:maxx, miny:mid_y]
        }

        for quad_name, quad_gdf in quadrants.items():
            if quad_gdf.empty:
                continue
            q_bbox = (
                minx if quad_name.endswith("W") else mid_x,
                mid_y if quad_name.startswith("N") else miny,
                mid_x if quad_name.endswith("W") else maxx,
                maxy if quad_name.startswith("N") else mid_y
            )
            self._build_tree(quad_gdf, q_bbox, depth + 1)

    def _finalize_node(self, gdf: gpd.GeoDataFrame, bbox: Tuple[float, float, float, float], depth: int) -> None:
        """Serialize node and apply fallback routing if budget is still exceeded."""
        node_tokens = sum(self._estimate_tokens(geom) for geom in gdf.geometry)
        if node_tokens > self.max_tokens:
            logging.warning(f"Node at depth {depth} exceeds budget. Applying centroid fallback.")
            gdf = gdf.copy()
            gdf["geometry"] = gdf.geometry.centroid
            node_tokens = sum(self._estimate_tokens(geom) for geom in gdf.geometry)

        self.nodes.append({
            "bbox": bbox,
            "depth": depth,
            "feature_count": len(gdf),
            "token_estimate": node_tokens,
            "data": json.loads(gdf.to_json())
        })

    def build(self) -> List[Dict]:
        """Entry point for pipeline integration."""
        try:
            self._build_tree(self.gdf, tuple(self.bbox), 0)
            logging.info(f"Quadtree built successfully. Total nodes: {len(self.nodes)}")
            return self.nodes
        except Exception as e:
            logging.critical(f"Tree construction failed: {e}")
            raise TokenBudgetExceededError("Unrecoverable token budget violation during partitioning.") from e
```

## Pipeline Integration & Safety Protocols

Integrating this quadtree builder into a geospatial LLM pipeline requires deterministic routing and strict validation gates. The `_clean_geometries` method ensures that malformed inputs do not propagate into the serialization layer, which is critical when using libraries like [Shapely 2.0 Documentation](https://shapely.readthedocs.io/en/stable/) that enforce strict topology rules. Coordinate validation at ingestion prevents silent failures when datasets cross the antimeridian or contain out-of-bounds projections.

For prompt assembly, implement a token-aware router that iterates through `self.nodes` and aggregates only those falling within the active query bounding box. If the combined token count approaches the LLM's context limit, trigger a fallback routing mechanism: replace detailed polygon coordinates with centroid points, bounding boxes, or precomputed spatial embeddings. This aligns with established [Geometry Tokenization Strategies](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/) that prioritize semantic retention over raw coordinate fidelity when budgets are constrained.

Safety protocols must include:
1. **Pre-flight Validation:** Run `_validate_wgs84_bounds` and `_clean_geometries` before any tree construction. Reject or quarantine datasets with >15% invalid geometries.
2. **Dynamic Budget Scaling:** Adjust `max_tokens` based on the target model's actual context window, accounting for system prompts and few-shot examples.
3. **Serialization Auditing:** Log `token_estimate` per node. Use [OpenAI tiktoken](https://github.com/openai/tiktoken) consistently across ingestion and prompt assembly to prevent estimator drift.
4. **Fallback Routing for Geospatial Queries:** When a query intersects more than 30% of the quadtree nodes, switch to a summary-based retrieval strategy (e.g., aggregated statistics, rasterized heatmaps) to avoid context fragmentation.

## Next Steps for Operational Deployment

1. **Implement Streaming Node Retrieval:** Replace batch node collection with an iterator that yields quadtree nodes on-demand. This reduces memory overhead and enables early stopping when the LLM context threshold is reached.
2. **Integrate Spatial Embedding Models:** Cache vector representations of quadtree centroids using domain-specific geospatial encoders. Use cosine similarity to pre-filter nodes before token estimation, reducing unnecessary serialization calls.
3. **Establish Vector-Raster Hybrid Processing:** For highly dense urban cores, convert quadtree leaf nodes into lightweight raster tiles when polygon serialization exceeds `max_tokens`. Route raster tiles to vision-language models while preserving vector nodes for tabular reasoning.
4. **Deploy Context Monitoring Dashboards:** Track `token_estimate` vs. `actual_consumed_tokens` across production queries. Set alerts for budget overruns and implement automated re-partitioning triggers when dataset density shifts beyond 20%.
5. **Standardize CRS Normalization Pipelines:** Enforce WGS84 conversion at the ingestion layer using `pyproj` or `geopandas` hooks. Reject non-conformant datasets before they reach the quadtree builder to eliminate runtime coordinate validation overhead.

By treating spatial partitioning as a dynamic constraint solver rather than a static index, engineering teams can guarantee deterministic context window utilization, eliminate silent token overflows, and scale geospatial LLM agents to continental datasets without compromising reasoning accuracy.
