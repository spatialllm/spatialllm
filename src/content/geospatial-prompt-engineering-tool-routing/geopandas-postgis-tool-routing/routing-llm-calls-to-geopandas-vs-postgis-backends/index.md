# Routing LLM Calls to GeoPandas vs PostGIS Backends

In production geospatial AI agents, the decision to execute spatial operations in-memory via GeoPandas versus delegating to a PostGIS database is rarely a static configuration. It is a dynamic routing problem that directly impacts latency, memory footprint, and topological correctness. When an LLM generates a tool call, the orchestrator must parse intent, estimate data volume, evaluate spatial complexity, and route to the appropriate execution backend. Misrouting triggers cascading failures: out-of-memory crashes on large `.sjoin()` operations, silent coordinate reference system (CRS) drift during in-memory transformations, or planner misestimates causing PostGIS query timeouts. Implementing deterministic routing requires strict validation layers, explicit complexity scoring, and fallback mechanisms that preserve pipeline integrity.

## Backend Selection Heuristics

The routing decision should be driven by quantifiable metrics rather than heuristic prompt matching. GeoPandas excels for operations requiring Python-native UDFs, iterative geometry refinement, or datasets under ~500k rows where memory overhead remains predictable. PostGIS dominates when operations involve multi-million-row spatial joins, concurrent read/write workloads, or topology validation requiring `ST_IsValid` and `ST_MakeValid` at scale. The routing layer must intercept the LLM's tool call payload, extract the target dataset size, operation type (e.g., `buffer`, `intersection`, `nearest`), and coordinate reference system metadata before dispatch.

A robust routing matrix evaluates three dimensions:
1. **Cardinality & Memory Footprint**: Estimated row count × average geometry size. If projected memory exceeds 70% of worker allocation, route to PostGIS.
2. **Spatial Complexity**: Topology-heavy operations (`ST_Contains`, `ST_Overlaps`, `ST_DWithin` with large radii) benefit from PostGIS spatial indexes and parallel query execution. Simple attribute filters or lightweight transformations (`centroid`, `buffer` with small distances) remain efficient in GeoPandas.
3. **Concurrency & State**: GeoPandas operations are synchronous and block the event loop. PostGIS integrates cleanly with async drivers (`asyncpg`, `SQLAlchemy 2.0+`), enabling non-blocking pipeline execution.

When LLM-generated tool calls bypass these checks, pipelines experience unpredictable degradation. The architecture documented in [Geospatial Prompt Engineering & Tool Routing](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/) establishes baseline patterns for intercepting and validating spatial tool payloads before execution.

## Failure Modes & Root Causes

### Memory Thrashing During In-Memory Joins
LLMs frequently generate `geopandas.sjoin()` calls without estimating index overlap. When two large polygon layers intersect, the underlying PyGEOS/Shapely engine materializes all candidate pairs in RAM. Root cause: missing pre-filtering via bounding box checks or spatial index pruning. Without explicit row-count estimation, the orchestrator defaults to in-memory execution until the OOM killer terminates the worker process.

### Silent CRS Drift & Geometry Degradation
LLMs often assume all inputs share a common projection. In practice, datasets arrive with mixed EPSG codes, `None` CRS metadata, or invalid geometries (self-intersections, ring orientation issues). GeoPandas will silently perform Cartesian math on unprojected coordinates, yielding meter-scale buffers in degree-space. PostGIS handles this more gracefully but requires explicit `ST_Transform` calls. Failing to validate CRS upfront corrupts downstream analytics.

### Planner Misestimates & Index Bypass
PostGIS relies on accurate statistics for query planning. LLM-generated SQL often omits `ANALYZE` triggers or uses non-sargable functions (e.g., `ST_Distance(geom, point) < 100` instead of `ST_DWithin`). This forces sequential scans, causing timeouts on large tables. Routing logic must rewrite naive LLM outputs into index-aware spatial predicates.

## Explicit Validation & Error Handling Implementation

A production routing layer must enforce strict coordinate validation, geometry integrity checks, and deterministic fallback paths. The following implementation demonstrates a hardened router that intercepts LLM tool calls, validates spatial metadata, and dispatches to the optimal backend with explicit error handling.

```python
import geopandas as gpd
from shapely.validation import make_valid
from shapely.errors import TopologicalError
import logging
from typing import Dict, Any, Optional
import asyncpg

logger = logging.getLogger(__name__)

class SpatialRoutingError(Exception):
    """Raised when routing fails due to invalid spatial state or resource constraints."""
    pass

def validate_crs_and_geometry(gdf: gpd.GeoDataFrame, target_epsg: int = 4326) -> gpd.GeoDataFrame:
    """Enforce CRS consistency and repair invalid geometries before routing."""
    if gdf.crs is None:
        raise SpatialRoutingError("Input GeoDataFrame missing CRS metadata. Cannot route safely.")

    if gdf.crs.to_epsg() != target_epsg:
        try:
            gdf = gdf.to_crs(epsg=target_epsg)
        except Exception as e:
            raise SpatialRoutingError(f"CRS transformation failed: {e}") from e

    # Validate and repair geometries
    invalid_mask = ~gdf.geometry.is_valid
    if invalid_mask.any():
        logger.warning(f"Repairing {invalid_mask.sum()} invalid geometries via make_valid()")
        gdf = gdf.copy()
        gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].apply(make_valid)

    return gdf

def estimate_memory_footprint_mb(gdf: gpd.GeoDataFrame) -> float:
    """Approximate RAM usage in MB for in-memory execution."""
    return gdf.memory_usage(deep=True).sum() / (1024 * 1024)

def route_spatial_operation(
    tool_call: Dict[str, Any],
    gdf: Optional[gpd.GeoDataFrame] = None,
    memory_threshold_mb: float = 2048.0
) -> Dict[str, Any]:
    """Deterministic router for LLM-generated spatial tool calls."""
    operation = tool_call.get("operation", "").lower()
    estimated_mem = estimate_memory_footprint_mb(gdf) if gdf is not None else 0.0

    # 1. Validate spatial inputs
    if gdf is not None:
        try:
            gdf = validate_crs_and_geometry(gdf, target_epsg=4326)
        except SpatialRoutingError as e:
            logger.error(f"Validation failed, forcing PostGIS fallback: {e}")
            return {"backend": "postgis", "reason": "crs_or_geometry_validation_failed", "payload": tool_call}

    # 2. Apply routing matrix
    complexity_heavy_ops = {"intersection", "union", "sjoin", "buffer_large", "dwithin"}
    is_complex = (
        operation in complexity_heavy_ops or
        (operation == "buffer" and tool_call.get("distance", 0) > 0.01)
    )

    if estimated_mem > memory_threshold_mb or is_complex:
        return {
            "backend": "postgis",
            "reason": "memory_or_complexity_threshold_exceeded",
            "payload": tool_call,
            "next_steps": "Rewrite to parameterized SQL with ST_Transform and spatial index hints"
        }
    else:
        return {
            "backend": "geopandas",
            "reason": "within_memory_and_complexity_bounds",
            "payload": tool_call,
            "next_steps": "Execute in-memory with explicit try/except fallback to PostGIS"
        }

def execute_with_fallback(routing_decision: Dict[str, Any], gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if routing_decision["backend"] == "geopandas":
        try:
            op = routing_decision["payload"]["operation"]
            if op == "buffer":
                dist = routing_decision["payload"]["distance"]
                return gpd.GeoDataFrame(geometry=gdf.buffer(dist), crs=gdf.crs)
            elif op == "centroid":
                return gpd.GeoDataFrame(geometry=gdf.centroid, crs=gdf.crs)
            return gdf
        except MemoryError as e:
            logger.critical(f"In-memory execution failed: {e}. Initiating PostGIS failover.")
            raise SpatialRoutingError("GeoPandas OOM. Reroute to PostGIS with chunked execution.") from e
    else:
        raise NotImplementedError("PostGIS async execution requires connection pool and SQL generator")
```

This pattern ensures that coordinate validation occurs before any spatial math, memory thresholds trigger proactive offloading, and explicit error handling prevents silent corruption.

## Pipeline Integration & Next Steps

Integrating this routing layer into an AI agent framework requires three structural adjustments:

1. **Pre-Execution Interceptor**: Wrap the LLM tool dispatcher with a middleware function that extracts `operation`, `dataset_id`, and `parameters`. Run `validate_crs_and_geometry()` immediately upon data load. Never trust LLM-generated CRS assumptions.
2. **Dynamic SQL Rewriting**: When routing to PostGIS, do not pass raw LLM SQL. Use a query builder that injects `ST_Transform()`, enforces `ST_DWithin` over `ST_Distance`, and appends `WHERE geom && ST_MakeEnvelope(...)` for bounding box pre-filtering. Consult the official [PostGIS Spatial Functions Reference](https://postgis.net/docs/) for index-aware predicate patterns.
3. **Async Execution & State Management**: Replace synchronous `gdf.to_postgis()` calls with `asyncpg` connection pools. Implement a retry circuit breaker that catches `asyncpg.exceptions.QueryCanceledError` and falls back to chunked GeoPandas execution for intermediate datasets.

### Clear Next Steps for Platform Teams
- **Instrument Routing Metrics**: Log `estimated_mem_mb`, `backend_choice`, and `validation_failures` to your observability stack. Track CRS drift incidents and memory threshold breaches weekly.
- **Enforce Geometry Contracts**: Require all upstream data pipelines to output GeoJSON/Parquet with explicit `crs` fields. Reject datasets with `None` CRS at ingestion.
- **Implement Topology Guardrails**: For operations requiring strict validity, integrate `ST_IsValid` checks into the PostGIS routing path. Refer to [Shapely's Geometry Validity Guide](https://shapely.readthedocs.io/en/stable/manual.html#object.is_valid) for in-memory validation equivalents.
- **Benchmark & Tune Thresholds**: The `memory_threshold_mb` and complexity flags are environment-specific. Run load tests with representative production datasets to calibrate routing boundaries.

By treating spatial routing as a deterministic, validation-first process rather than a prompt-matching heuristic, platform teams can eliminate cascading failures, enforce topological correctness, and scale geospatial AI agents reliably across heterogeneous workloads.
