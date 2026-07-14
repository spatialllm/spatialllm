# Fallback Routing for Geospatial Queries

In production-grade spatial AI systems, query reliability is non-negotiable. When a primary geospatial operation fails—whether due to malformed geometry inputs, coordinate reference system mismatches, index fragmentation, or LLM-hallucinated coordinates—the pipeline must degrade gracefully rather than cascade into terminal exceptions. **Fallback Routing for Geospatial Queries** provides a deterministic mechanism to intercept failures, classify error signatures, and route execution to alternative computational pathways. Within the broader [Spatial LLM Architecture & Core Concepts](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/) framework, this pattern ensures that spatial reasoning workflows maintain continuity, preserve semantic intent, and enforce strict computational boundaries.

<figure class="diagram">
<svg viewBox="0 0 860 300" role="img" aria-labelledby="fbr-t fbr-d" xmlns="http://www.w3.org/2000/svg">
  <title id="fbr-t">Fallback routing pipeline</title>
  <desc id="fbr-d">Query intake, primary tool call, health check, fallback routing, and response merge run in sequence to deliver resilient geospatial query routing with error mapping, retries, and a degraded mode.</desc>
  <defs>
    <marker id="fbr-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
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
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#fbr-arrow)">
    <line x1="166" y1="95" x2="181" y2="95"/>
    <line x1="334" y1="95" x2="349" y2="95"/>
    <line x1="502" y1="95" x2="517" y2="95"/>
    <line x1="670" y1="95" x2="685" y2="95"/>
  </g>
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#fbr-arrow)">
    <line x1="90" y1="137" x2="90" y2="202"/>
    <line x1="258" y1="137" x2="258" y2="202"/>
    <line x1="426" y1="137" x2="426" y2="202"/>
    <line x1="594" y1="137" x2="594" y2="202"/>
    <line x1="762" y1="137" x2="762" y2="202"/>
  </g>
  <rect x="15" y="205" width="822" height="58" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/>
  <g fill="#1f2937" font-size="13" text-anchor="middle">
    <text x="90" y="90"><tspan x="90" dy="0">Query</tspan><tspan x="90" dy="16">intake</tspan></text>
    <text x="258" y="90"><tspan x="258" dy="0">Primary</tspan><tspan x="258" dy="16">tool call</tspan></text>
    <text x="426" y="90"><tspan x="426" dy="0">Health</tspan><tspan x="426" dy="16">check</tspan></text>
    <text x="594" y="90"><tspan x="594" dy="0">Fallback</tspan><tspan x="594" dy="16">route</tspan></text>
    <text x="762" y="90"><tspan x="762" dy="0">Response</tspan><tspan x="762" dy="16">merge</tspan></text>
  </g>
  <text x="426" y="240" fill="#1f2937" font-size="15" font-weight="600" text-anchor="middle">Resilient geospatial query routing</text>
  <text x="426" y="287" fill="#5b6471" font-size="12" text-anchor="middle">Downstream: error mapping · retries · degraded mode</text>
</svg>
</figure>

## Error Taxonomy and Deterministic Mapping

The foundation of any robust fallback strategy is a precise error classification matrix. Spatial queries typically fail across four predictable categories: topology violations, empty result sets, CRS misalignment, and execution timeouts. Implementing structured error capture requires wrapping PostGIS and GeoPandas execution blocks in explicit exception handlers that map database error codes and runtime exceptions to routing decisions.

For example, when an LLM generates coordinates that violate planar topology, the system should intercept the `22023` (invalid parameter value) PostgreSQL error and trigger a coordinate sanitization routine. When a spatial join returns zero matches due to overly restrictive predicates, the routing layer should classify this as an `EMPTY_RESULT` and trigger a geometric relaxation step. Deterministic mapping eliminates ad-hoc retry loops by enforcing a strict decision tree:

1. **Capture**: Intercept exceptions via SQLAlchemy/GeoPandas error hooks.
2. **Classify**: Map error signatures to predefined fallback categories using regex or error-code dictionaries.
3. **Route**: Select the corresponding fallback operator based on the classification.
4. **Validate**: Verify fallback output against topology and CRS constraints before returning to the caller.

This classification layer is particularly critical when parsing outputs from [Geometry Tokenization Strategies](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/), where token-to-coordinate decoding frequently introduces floating-point precision drift or self-intersecting polygons.

## CRS Normalization and Topology Enforcement

Geospatial AI pipelines frequently ingest heterogeneous coordinate systems. Without strict normalization, spatial predicates like `ST_Intersects` or `ST_DWithin` will silently fail or return mathematically incorrect results. The routing layer must enforce a canonical CRS (typically `EPSG:4326` for global LLM contexts or a projected system like `EPSG:3857` for local analysis) before any spatial operation executes.

Topology enforcement operates in tandem with CRS normalization. LLM-generated bounding boxes or point sequences often contain duplicate vertices, unclosed rings, or bowtie geometries. The fallback router must apply deterministic repair routines—such as zero-buffering, snap-to-grid, or `make_valid` operations—before re-attempting execution. These constraints align with the [OGC Coordinate Reference Systems Standard](https://www.ogc.org/standard/crs/) and ensure that downstream [Spatial Embedding Models](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/spatial-embedding-models/) receive geometrically sound inputs for vectorization.

## Production-Ready Implementation Pattern

A production-ready routing layer operates as middleware between the LLM prompt parser and the spatial execution engine. The following Python implementation demonstrates deterministic fallback routing using GeoPandas, PostGIS, and structured exception handling. All code enforces CRS alignment and topology validation.

```python
import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine, text, exc as sql_exc
from shapely.geometry import shape, mapping
from shapely.validation import make_valid
import logging
from typing import Dict, Any, Optional, Callable
import pyproj

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class SpatialFallbackRouter:
    """
    Deterministic fallback router for geospatial queries.
    Intercepts failures, normalizes CRS/topology, and routes to alternative execution paths.
    """
    def __init__(self, db_uri: str, target_crs: str = "EPSG:4326"):
        self.engine = create_engine(db_uri, pool_pre_ping=True)
        self.target_crs = pyproj.CRS.from_user_input(target_crs)
        self.fallback_handlers: Dict[str, Callable] = {
            "TOPOLOGY_ERROR": self._sanitize_topology,
            "EMPTY_RESULT": self._relax_predicate,
            "CRS_MISMATCH": self._normalize_crs,
            "TIMEOUT": self._downsample_geometry
        }

    def execute_with_fallback(self, query: str, params: dict, max_fallbacks: int = 2) -> gpd.GeoDataFrame:
        """
        Executes a spatial query with deterministic fallback routing.
        """
        attempt = 0
        current_params = params.copy()
        last_error = None

        while attempt <= max_fallbacks:
            try:
                logger.info(f"Executing query (attempt {attempt + 1})")
                with self.engine.connect() as conn:
                    result = gpd.read_postgis(
                        text(query),
                        conn,
                        params=current_params,
                        geom_col="geom"
                    )
                    if not result.empty:
                        result = self._validate_and_normalize(result)
                        return result
                    else:
                        raise ValueError("EMPTY_RESULT")

            except sql_exc.DatabaseError as e:
                error_code = str(e.orig)
                if "22023" in error_code or "invalid geometry" in error_code.lower():
                    last_error = "TOPOLOGY_ERROR"
                elif "timeout" in error_code.lower() or "canceling statement" in error_code.lower():
                    last_error = "TIMEOUT"
                else:
                    last_error = "UNKNOWN_DB_ERROR"

            except ValueError as e:
                if "EMPTY_RESULT" in str(e):
                    last_error = "EMPTY_RESULT"
                else:
                    last_error = "UNKNOWN_RUNTIME_ERROR"

            except Exception as e:
                if "CRS" in str(e).upper() or "mismatch" in str(e).lower():
                    last_error = "CRS_MISMATCH"
                else:
                    last_error = "UNKNOWN_RUNTIME_ERROR"

            # Route to fallback handler
            if last_error in self.fallback_handlers:
                logger.warning(f"Routing fallback: {last_error}")
                current_params = self.fallback_handlers[last_error](current_params)
                attempt += 1
            else:
                raise RuntimeError(f"Query failed after {attempt + 1} attempts: {last_error}")

        raise RuntimeError("Max fallback attempts exceeded without successful execution.")

    def _sanitize_topology(self, params: dict) -> dict:
        """Applies Shapely make_valid to repair geometries."""
        if "geom" in params:
            geom = params["geom"]
            if hasattr(geom, "is_valid") and not geom.is_valid:
                params["geom"] = make_valid(geom)
                logger.info("Applied topology sanitization (make_valid)")
        return params

    def _relax_predicate(self, params: dict) -> dict:
        """Expands spatial predicates by a fixed tolerance to recover empty results."""
        if "distance" in params:
            params["distance"] = params["distance"] * 1.5
            logger.info(f"Relaxed distance predicate to {params['distance']}")
        elif "buffer" in params:
            params["buffer"] = params["buffer"] * 1.5
        else:
            params.setdefault("fallback_tolerance", 0.001)
            params["fallback_tolerance"] *= 1.5
        return params

    def _normalize_crs(self, params: dict) -> dict:
        """Forces target CRS on input geometries."""
        if "geom" in params:
            gdf = gpd.GeoDataFrame({"geometry": [params["geom"]]}, crs="EPSG:4326")
            gdf = gdf.to_crs(self.target_crs)
            params["geom"] = gdf.iloc[0].geometry
            logger.info(f"Normalized geometry to {self.target_crs.to_string()}")
        return params

    def _downsample_geometry(self, params: dict) -> dict:
        """Simplifies complex geometries to prevent execution timeouts."""
        if "geom" in params:
            geom = params["geom"]
            params["geom"] = geom.simplify(tolerance=0.0001, preserve_topology=True)
            logger.info("Downsampled geometry complexity to avoid timeout")
        return params

    def _validate_and_normalize(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Post-execution validation and CRS enforcement."""
        if gdf.crs is None:
            gdf = gdf.set_crs(self.target_crs)
        elif gdf.crs != self.target_crs:
            gdf = gdf.to_crs(self.target_crs)

        # Repair rather than drop invalid geometries
        invalid_mask = ~gdf.geometry.is_valid
        if invalid_mask.any():
            logger.warning(f"Repairing {invalid_mask.sum()} invalid geometries post-query")
            gdf = gdf.copy()
            gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].apply(make_valid)
        return gdf
```

## Integration with Spatial LLM Pipelines

Fallback routing is not an isolated utility; it is a critical control plane within the broader spatial reasoning stack. When an LLM generates a spatial query, the output typically passes through a parsing layer that converts natural language or structured JSON into executable GIS operations. If the initial execution fails, the fallback router intercepts the error and modifies the execution graph without requiring a full LLM regeneration. This preserves context window budget and reduces inference latency.

The routing layer interacts directly with vectorization and tokenization subsystems. For instance, when a spatial join fails due to empty results, the router can trigger a predicate relaxation that aligns with the semantic tolerance thresholds defined in [Implementing Fallback Routing for Failed Spatial Queries](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/implementing-fallback-routing-for-failed-spatial-queries/). This ensures that the pipeline maintains deterministic behavior while adapting to the inherent noise in generative spatial outputs.

Furthermore, fallback routing must be idempotent and observable. Each routing decision should emit structured telemetry (error class, fallback applied, execution time delta) to a centralized metrics backend. This enables platform teams to track failure modes, refine LLM prompt templates, and adjust CRS normalization strategies based on real-world query distributions.

## Operational Considerations

Deploying fallback routing in production requires adherence to several engineering constraints:

1. **Circuit Breaking**: Implement exponential backoff or circuit breakers around the fallback loop to prevent cascading load on the spatial database during index fragmentation or network degradation.
2. **Geometry Validation Overhead**: Topology repair (`make_valid`, `buffer(0)`) is computationally expensive. Cache validated geometries at the application layer when processing batch LLM outputs.
3. **CRS Drift Monitoring**: Continuously audit the CRS of incoming LLM coordinates. Mismatches often indicate prompt template drift or hallucinated coordinate formats (e.g., mixing WKT with GeoJSON).
4. **Shapely & PostGIS Alignment**: Ensure that the application-side geometry library (e.g., [Shapely Geometry Validation](https://shapely.readthedocs.io/en/stable/manual.html#validation.make_valid)) matches the database-side PostGIS version to avoid silent precision loss during fallback transformations.

By embedding deterministic fallback routing into the geospatial AI pipeline, engineering teams can transform brittle spatial queries into resilient, self-healing workflows. This architecture guarantees that spatial reasoning systems degrade gracefully, maintain strict topological and CRS boundaries, and deliver consistent outputs even under adversarial or hallucinated input conditions.
