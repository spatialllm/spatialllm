# Async vs Sync Geoprocessing Workflows

Modern spatial AI systems increasingly rely on dynamic tool routing to execute geospatial operations at scale. Within the broader [Geospatial Prompt Engineering & Tool Routing](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/) paradigm, selecting between synchronous and asynchronous execution models directly impacts latency, resource utilization, and fault tolerance. The decision to implement **Async vs Sync Geoprocessing Workflows** is not merely an architectural preference; it is a deterministic routing choice dictated by operation complexity, data volume, and downstream agent dependencies. This guide provides production-ready Python, GeoPandas, and PostGIS patterns for both models, with explicit CRS enforcement, topology validation, and structured error mapping.

<figure class="diagram">
<svg viewBox="0 0 860 300" role="img" aria-labelledby="asw-t asw-d" xmlns="http://www.w3.org/2000/svg">
  <title id="asw-t">Async vs sync geoprocessing pipeline</title>
  <desc id="asw-d">Job intake, a sync branch, an async queue, a worker pool, and result aggregation run in sequence to support mixed sync/async geoprocessing with tool routing, retries, and monitoring.</desc>
  <defs>
    <marker id="asw-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
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
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#asw-arrow)">
    <line x1="166" y1="95" x2="181" y2="95"/>
    <line x1="334" y1="95" x2="349" y2="95"/>
    <line x1="502" y1="95" x2="517" y2="95"/>
    <line x1="670" y1="95" x2="685" y2="95"/>
  </g>
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#asw-arrow)">
    <line x1="90" y1="137" x2="90" y2="202"/>
    <line x1="258" y1="137" x2="258" y2="202"/>
    <line x1="426" y1="137" x2="426" y2="202"/>
    <line x1="594" y1="137" x2="594" y2="202"/>
    <line x1="762" y1="137" x2="762" y2="202"/>
  </g>
  <rect x="15" y="205" width="822" height="58" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/>
  <g fill="#1f2937" font-size="13" text-anchor="middle">
    <text x="90" y="90"><tspan x="90" dy="0">Job</tspan><tspan x="90" dy="16">intake</tspan></text>
    <text x="258" y="90"><tspan x="258" dy="0">Sync</tspan><tspan x="258" dy="16">branch</tspan></text>
    <text x="426" y="90"><tspan x="426" dy="0">Async</tspan><tspan x="426" dy="16">queue</tspan></text>
    <text x="594" y="90"><tspan x="594" dy="0">Worker</tspan><tspan x="594" dy="16">pool</tspan></text>
    <text x="762" y="90"><tspan x="762" dy="0">Result</tspan><tspan x="762" dy="16">aggregation</tspan></text>
  </g>
  <text x="426" y="240" fill="#1f2937" font-size="15" font-weight="600" text-anchor="middle">Mixed sync/async geoprocessing</text>
  <text x="426" y="287" fill="#5b6471" font-size="12" text-anchor="middle">Downstream: tool routing · retries · monitoring</text>
</svg>
</figure>

## Execution Model Fundamentals in Spatial Context

Synchronous workflows execute spatial operations sequentially, blocking the calling thread until the operation completes. This model aligns with lightweight, deterministic tasks such as coordinate reference system (CRS) transformations, attribute filtering, or small-scale spatial joins where memory footprint and execution time remain bounded.

Asynchronous workflows decouple execution from the calling thread, enabling concurrent I/O, background topology validation, and non-blocking database queries. For spatial AI agents, async patterns are essential when orchestrating multi-step pipelines that involve large raster processing, network analysis, or iterative LLM-driven query refinement. The routing layer must evaluate payload size, predicate complexity, and downstream dependency graphs to dispatch tasks to the appropriate execution model.

## Step-by-Step: Synchronous Pattern with Validation

A robust synchronous workflow begins with explicit schema validation and geometry sanitization before execution. Using GeoPandas, developers must enforce deterministic input checks to prevent silent spatial failures. Invalid geometries trigger undefined behavior in both GeoPandas and PostGIS, making pre-flight validation non-negotiable.

```python
import logging
import geopandas as gpd
from shapely.validation import make_valid
from typing import Tuple

logging.basicConfig(level=logging.INFO)

def sync_geoprocess(
    input_path: str,
    reference_gdf: gpd.GeoDataFrame,
    target_crs: int = 4326
) -> Tuple[gpd.GeoDataFrame, dict]:
    """
    Synchronous spatial join with explicit CRS & topology enforcement.
    Returns processed GDF and an audit dictionary.
    """
    audit = {"invalid_geometries_repaired": 0, "crs_transformed": False}

    gdf = gpd.read_file(input_path)

    # 1. CRS Enforcement
    if gdf.crs is None:
        raise ValueError("Input dataset lacks CRS. Assign before processing.")
    if gdf.crs.to_epsg() != target_crs:
        gdf = gdf.to_crs(epsg=target_crs)
        audit["crs_transformed"] = True

    # 2. Topology & Geometry Validation
    invalid_mask = ~gdf.geometry.is_valid
    invalid_count = int(invalid_mask.sum())
    if invalid_count > 0:
        logging.warning(f"Repairing {invalid_count} invalid geometries...")
        gdf = gdf.copy()
        gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].apply(make_valid)
        audit["invalid_geometries_repaired"] = invalid_count

    # 3. Synchronous Spatial Join
    if not gdf.crs.equals(reference_gdf.crs):
        raise ValueError("CRS mismatch between input and reference layers.")

    result = gpd.sjoin(gdf, reference_gdf, how="inner", predicate="intersects")
    return result, audit
```

Validation must occur before any spatial predicate evaluation. Implementing a pre-flight validator that checks Shapely validity flags prevents downstream pipeline corruption. Always log the count of repaired geometries to maintain auditability. This pattern integrates cleanly with [Topology Rule Enforcement via LLMs](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/), where automated agents generate validation rules that are applied synchronously before committing results to a spatial index.

## Step-by-Step: Asynchronous Pattern with PostGIS & Connection Pooling

When scaling to concurrent spatial queries or integrating with LLM agents that generate dynamic SQL, asynchronous execution becomes mandatory. The `asyncpg` library paired with `asyncio` provides a production-ready foundation for non-blocking PostGIS operations, leveraging connection pooling and parameterized queries to prevent SQL injection and optimize throughput.

```python
import asyncio
import asyncpg
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)

class AsyncSpatialExecutor:
    def __init__(self, dsn: str, pool_size: int = 10):
        self.dsn = dsn
        self.pool_size = pool_size
        self._pool = None

    async def initialize(self):
        """Initialize connection pool. PostGIS extensions must already be installed on the database."""
        self._pool = await asyncpg.create_pool(
            dsn=self.dsn,
            min_size=2,
            max_size=self.pool_size,
            command_timeout=60.0,
            server_settings={"statement_timeout": "30000"}
        )

    async def execute_spatial_query(
        self,
        query: str,
        params: tuple = ()
    ) -> List[Dict[str, Any]]:
        """Execute a parameterized spatial query asynchronously."""
        if not self._pool:
            raise RuntimeError("Pool not initialized. Call initialize() first.")

        async with self._pool.acquire() as conn:
            try:
                # asyncpg returns Record objects; convert to dict for JSON serialization
                rows = await conn.fetch(query, *params)
                return [dict(row) for row in rows]
            except asyncpg.PostgresError as e:
                logging.error(f"PostGIS execution failed: {e}")
                raise

    async def close(self):
        if self._pool:
            await self._pool.close()


async def run_async_pipeline():
    executor = AsyncSpatialExecutor(dsn="postgresql://user:pass@localhost:5432/spatial_db")
    await executor.initialize()

    # Parameterized query prevents SQL injection
    dynamic_sql = """
        SELECT a.id, ST_AsText(a.geom) AS geom_text
        FROM parcels a
        WHERE ST_Intersects(a.geom, ST_SetSRID(ST_MakePoint($1, $2), $3))
        LIMIT 100;
    """

    try:
        results = await executor.execute_spatial_query(
            dynamic_sql, params=(-73.9857, 40.7484, 4326)
        )
        logging.info(f"Retrieved {len(results)} spatial features asynchronously.")
    finally:
        await executor.close()


if __name__ == "__main__":
    asyncio.run(run_async_pipeline())
```

The async model shines when paired with [Prompt-to-Spatial-SQL Generation](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/), where LLMs construct parameterized queries that are safely executed against pooled connections. For deeper implementation patterns on task scheduling and backpressure handling, refer to [Handling Async Spatial Processing in Python Workflows](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/handling-async-spatial-processing-in-python-workflows/).

## Deterministic Routing & Fallback Logic

Production systems should never hardcode execution models. Instead, implement a routing dispatcher that evaluates operation metadata:

1. **Data Volume Threshold:** < 50k rows or < 500MB → Sync. Above threshold → Async.
2. **Operation Type:** CRS transforms, attribute filters, simple predicates → Sync. Network analysis, raster tiling, iterative spatial joins → Async.
3. **Downstream Dependencies:** Blocking required for immediate UI/agent feedback → Sync. Background indexing, batch exports, multi-agent orchestration → Async.

```python
def route_geoprocess(
    row_count: int,
    operation_type: str,
    requires_immediate_result: bool
) -> str:
    """Deterministic router for Async vs Sync Geoprocessing Workflows."""
    if requires_immediate_result and row_count < 50_000:
        return "sync"
    if operation_type in {"network_analysis", "raster_processing", "multi_step_join"}:
        return "async"
    if row_count >= 50_000:
        return "async"
    return "sync"
```

The router should be wrapped in a circuit breaker pattern. If an async task exceeds timeout thresholds or encounters connection pool exhaustion, the system must gracefully degrade to a queued synchronous fallback or return a structured retry token.

## Error Mapping & Fault Tolerance

Spatial operations fail differently across execution models. Synchronous failures raise immediate Python exceptions, making stack traces straightforward to parse. Asynchronous failures often manifest as `asyncpg.PostgresError`, `ConnectionResetError`, or task-level timeouts, requiring structured error mapping.

Implement a unified error mapper that translates spatial API failures into standardized JSON responses:

```python
def map_spatial_error(exception: Exception) -> dict:
    error_map = {
        "InvalidTextRepresentationError": {
            "code": "SPATIAL_PARSE_001",
            "message": "Malformed geometry input or invalid WKT/GeoJSON.",
            "action": "Validate input schema and retry with sanitized payload."
        },
        "QueryCanceledError": {
            "code": "SPATIAL_TIMEOUT_002",
            "message": "Spatial query exceeded statement_timeout.",
            "action": "Reduce bounding box, add spatial index, or route to async batch queue."
        },
        "ValueError": {
            "code": "CRS_MISMATCH_003",
            "message": "Coordinate reference system mismatch detected.",
            "action": "Explicitly transform all layers to a common EPSG before execution."
        }
    }

    exc_name = type(exception).__name__
    return error_map.get(exc_name, {
        "code": "UNKNOWN_SPATIAL_ERR",
        "message": str(exception),
        "action": "Check PostGIS logs and verify topology constraints."
    })
```

Always enforce `ST_IsValid` checks at the database level when routing to PostGIS. The official [PostGIS documentation](https://postgis.net/documentation/) details spatial index optimization and topology validation functions that should be integrated into async query templates. For connection pool tuning and asyncpg best practices, consult the [asyncpg official documentation](https://magicstack.github.io/asyncpg/current/).

## Conclusion

Choosing between Async vs Sync Geoprocessing Workflows is a deterministic routing decision that directly impacts pipeline resilience, agent orchestration, and spatial data integrity. By enforcing strict CRS alignment, pre-flight topology validation, and structured error mapping, platform teams can deploy hybrid execution models that scale predictably. Integrate these patterns into your tool routing layer to ensure that lightweight vector operations remain responsive while heavy spatial computations execute concurrently without blocking downstream AI agents.
