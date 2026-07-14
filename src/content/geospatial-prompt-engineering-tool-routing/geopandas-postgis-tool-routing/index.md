# GeoPandas & PostGIS Tool Routing

Modern spatial AI pipelines require deterministic routing between in-memory Python workflows and database-native execution engines. **GeoPandas & PostGIS Tool Routing** establishes a decision framework that dynamically directs geospatial operations to the most performant and reliable backend based on payload characteristics, operation complexity, and system state. Within the broader [Geospatial Prompt Engineering & Tool Routing](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/) paradigm, this routing layer acts as the execution orchestrator, translating high-level spatial intents into validated, backend-specific execution plans. Platform teams and spatial data scientists must implement explicit validation gates, structured error mapping, and deterministic fallbacks to prevent silent geometry corruption, memory exhaustion, or unbounded query execution.

<figure class="diagram">
<svg viewBox="0 0 860 300" role="img" aria-labelledby="gpp-t gpp-d" xmlns="http://www.w3.org/2000/svg">
  <title id="gpp-t">GeoPandas/PostGIS routing pipeline</title>
  <desc id="gpp-d">Query intake, engine selection, a GeoPandas path, a PostGIS path, and result unification run in sequence to deliver optimal GeoPandas/PostGIS routing for SQL generation, pipelines, and caching.</desc>
  <defs>
    <marker id="gpp-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
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
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#gpp-arrow)">
    <line x1="166" y1="95" x2="181" y2="95"/>
    <line x1="334" y1="95" x2="349" y2="95"/>
    <line x1="502" y1="95" x2="517" y2="95"/>
    <line x1="670" y1="95" x2="685" y2="95"/>
  </g>
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#gpp-arrow)">
    <line x1="90" y1="137" x2="90" y2="202"/>
    <line x1="258" y1="137" x2="258" y2="202"/>
    <line x1="426" y1="137" x2="426" y2="202"/>
    <line x1="594" y1="137" x2="594" y2="202"/>
    <line x1="762" y1="137" x2="762" y2="202"/>
  </g>
  <rect x="15" y="205" width="822" height="58" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/>
  <g fill="#1f2937" font-size="13" text-anchor="middle">
    <text x="90" y="90"><tspan x="90" dy="0">Query</tspan><tspan x="90" dy="16">intake</tspan></text>
    <text x="258" y="90"><tspan x="258" dy="0">Engine</tspan><tspan x="258" dy="16">selection</tspan></text>
    <text x="426" y="90"><tspan x="426" dy="0">GeoPandas</tspan><tspan x="426" dy="16">path</tspan></text>
    <text x="594" y="90"><tspan x="594" dy="0">PostGIS</tspan><tspan x="594" dy="16">path</tspan></text>
    <text x="762" y="90"><tspan x="762" dy="0">Result</tspan><tspan x="762" dy="16">unify</tspan></text>
  </g>
  <text x="426" y="240" fill="#1f2937" font-size="15" font-weight="600" text-anchor="middle">Optimal GeoPandas/PostGIS routing</text>
  <text x="426" y="287" fill="#5b6471" font-size="12" text-anchor="middle">Downstream: SQL generation · pipelines · caching</text>
</svg>
</figure>

## Architectural Decision Matrix for Spatial Backends

Routing is not a static configuration; it is a runtime evaluation of spatial workload constraints. The decision matrix evaluates three primary dimensions before dispatching execution:

1. **Data Volume & Memory Footprint:** GeoPandas operates in-process and scales linearly with available RAM. Operations exceeding ~500 MB of uncompressed geometry payloads typically trigger OS paging or OOM failures. PostGIS leverages disk-backed storage, spatial indexing (GiST/SP-GiST), and parallel query execution, making it the default for datasets with >1M rows or payloads exceeding available heap space.
2. **Operation Topology & Complexity:** Vectorized Python operations (e.g., `GeoSeries.distance`, `sjoin_nearest`, affine transformations) execute faster in-memory for small-to-medium datasets due to zero serialization overhead. Set-based spatial operations (`ST_Intersects`, `ST_DWithin`, `ST_Union` aggregations) benefit from PostGIS query planners, cost-based optimization, and index-aware execution.
3. **Concurrency & Statefulness:** Stateless batch transformations favor GeoPandas. Concurrent multi-tenant workloads, transactional spatial updates, or topology-aware validations require PostGIS connection pooling, MVCC, and row-level locking to maintain data integrity under parallel load.

The routing engine must evaluate these dimensions pre-execution, not after failure. Failing to do so introduces non-deterministic latency spikes and silent topology degradation in downstream AI feature stores.

## Implementation: Deterministic Routing & Validation Gates

A production-grade router requires explicit capability parsing, threshold evaluation, and pre-execution validation. The following pattern implements a deterministic routing controller with strict CRS alignment and topology enforcement.

```python
import geopandas as gpd
from sqlalchemy import create_engine, text
import numpy as np
import logging
from typing import Dict, Any, Optional
from shapely.validation import make_valid

class SpatialRouterError(Exception): pass
class CRSValidationError(SpatialRouterError): pass
class TopologyValidationError(SpatialRouterError): pass

class SpatialRouter:
    def __init__(self, postgis_uri: str, memory_threshold_mb: float = 500.0,
                 row_threshold: int = 500_000, target_crs: str = "EPSG:4326"):
        self.postgis_uri = postgis_uri
        self.memory_threshold = memory_threshold_mb * 1024 * 1024
        self.row_threshold = row_threshold
        self.target_crs = target_crs
        self.engine = create_engine(postgis_uri, pool_size=10, max_overflow=20, pool_pre_ping=True)
        self.logger = logging.getLogger(__name__)

    def _estimate_memory(self, gdf: gpd.GeoDataFrame) -> int:
        return int(gdf.memory_usage(deep=True).sum())

    def _validate_crs(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if gdf.crs is None:
            raise CRSValidationError("GeoDataFrame lacks a defined CRS. Assign EPSG code before routing.")
        if gdf.crs.to_string() != self.target_crs:
            self.logger.warning(f"CRS mismatch: {gdf.crs} vs target {self.target_crs}. Reprojecting.")
            gdf = gdf.to_crs(self.target_crs)
        return gdf

    def _validate_topology(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        invalid_mask = ~gdf.geometry.is_valid
        if invalid_mask.any():
            self.logger.warning(f"Detected {invalid_mask.sum()} invalid geometries. Applying Shapely make_valid.")
            gdf = gdf.copy()
            gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].apply(make_valid)
        if not gdf.geometry.is_valid.all():
            raise TopologyValidationError("Persistent topology errors after repair. Rejecting payload.")
        return gdf

    def route(self, operation: str, gdf: gpd.GeoDataFrame) -> str:
        mem_usage = self._estimate_memory(gdf)
        row_count = len(gdf)

        gdf = self._validate_crs(gdf)
        gdf = self._validate_topology(gdf)

        if mem_usage > self.memory_threshold or row_count > self.row_threshold:
            return "postgis"
        if operation in {"buffer", "centroid", "distance", "sjoin_nearest", "affine_transform"}:
            return "geopandas"
        if operation in {"st_union_agg", "st_intersects", "st_dwithin", "st_difference", "st_snap"}:
            return "postgis"
        return "geopandas"

    def execute(self, operation: str, gdf: gpd.GeoDataFrame, params: Optional[Dict[str, Any]] = None) -> gpd.GeoDataFrame:
        backend = self.route(operation, gdf)
        self.logger.info(f"Routing operation '{operation}' to {backend} backend.")

        if backend == "geopandas":
            return self._exec_geopandas(operation, gdf, params or {})
        else:
            return self._exec_postgis(operation, gdf, params or {})

    def _exec_geopandas(self, operation: str, gdf: gpd.GeoDataFrame, params: Dict[str, Any]) -> gpd.GeoDataFrame:
        if operation == "buffer":
            return gpd.GeoDataFrame(
                geometry=gdf.buffer(params.get("distance", 0.01), cap_style="round", join_style="mitre"),
                crs=gdf.crs
            )
        elif operation == "centroid":
            return gpd.GeoDataFrame(geometry=gdf.centroid, crs=gdf.crs)
        elif operation == "sjoin_nearest":
            return gdf.sjoin_nearest(params["target_gdf"], max_distance=params.get("max_distance"))
        else:
            raise NotImplementedError(f"GeoPandas operation '{operation}' not implemented in router.")

    def _exec_postgis(self, operation: str, gdf: gpd.GeoDataFrame, params: Dict[str, Any]) -> gpd.GeoDataFrame:
        table_name = f"temp_{operation}_{np.random.randint(1000, 9999)}"
        gdf.to_postgis(table_name, self.engine, if_exists="replace", index=False)

        try:
            if operation == "st_union_agg":
                query = text(f"SELECT ST_Union(geometry) AS geometry FROM {table_name}")
            elif operation == "st_intersects":
                target_table = params.get("target_table")
                if not target_table:
                    raise ValueError("'target_table' param required for st_intersects routing.")
                query = text(f"SELECT a.* FROM {table_name} a JOIN {target_table} b ON ST_Intersects(a.geometry, b.geometry)")
            else:
                raise NotImplementedError(f"PostGIS operation '{operation}' not implemented.")

            return gpd.read_postgis(query, self.engine, geom_col="geometry")
        finally:
            with self.engine.connect() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                conn.commit()
```

## Integration with Prompt-Driven Spatial Workflows

The router serves as the execution bridge between natural language spatial intents and deterministic backend dispatch. When an LLM generates a spatial intent, the [Prompt-to-Spatial-SQL Generation](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/) subsystem parses the request into structured operation tokens, CRS requirements, and parameter dictionaries. These tokens are passed directly to the `SpatialRouter.route()` method, which evaluates payload characteristics before committing to an execution path.

Topology validation is particularly critical in LLM-assisted pipelines. Generative models frequently produce malformed coordinate sequences or unclosed rings when synthesizing feature extraction prompts. By integrating [Topology Rule Enforcement via LLMs](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/) into the pre-routing validation gate, the system intercepts invalid geometries before they reach the execution layer. The router's `_validate_topology()` method applies deterministic repair strategies aligned with OGC Simple Features compliance, ensuring that downstream AI models receive clean, spatially consistent inputs.

Dispatch logic must also account for async execution boundaries and connection lifecycle management. The [Routing LLM Calls to GeoPandas vs PostGIS Backends](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/geopandas-postgis-tool-routing/routing-llm-calls-to-geopandas-vs-postgis-backends/) pattern demonstrates how to wrap synchronous Python operations in async-compatible executors while maintaining PostGIS connection pooling. This prevents thread starvation during high-concurrency inference requests and guarantees that spatial queries respect transactional isolation levels.

## Production Hardening & Error Mapping

Production deployments require explicit error mapping to prevent silent failures from propagating into feature stores or downstream model training loops. The router implements structured exception hierarchies (`CRSValidationError`, `TopologyValidationError`) that map directly to HTTP 4xx/5xx responses or retry queues. When a PostGIS query exceeds configured execution limits, the router should catch `sqlalchemy.exc.OperationalError` and fallback to chunked GeoPandas processing, or vice versa when memory thresholds are breached.

CRS standardization must be enforced at ingestion. Spatial operations across mismatched projections yield mathematically invalid results that are difficult to trace post-hoc. The router's `_validate_crs()` method ensures all payloads align with a canonical EPSG code before execution, leveraging [GeoPandas User Guide](https://geopandas.org/en/stable/docs/user_guide.html) best practices for projection handling. For database-side operations, PostGIS natively handles on-the-fly transformations via `ST_Transform`, but explicit alignment at the routing layer reduces planner overhead and prevents index bypass.

Topology validation should adhere to the [OGC Simple Features Specification](https://www.ogc.org/standards/sfa), particularly regarding ring closure, self-intersection tolerance, and dimensionality constraints. The router's validation pipeline rejects payloads that fail post-repair checks, forcing upstream data quality gates to trigger rather than allowing corrupted geometries to silently degrade spatial joins or buffer operations.

Monitoring and observability must track routing decisions, validation failure rates, and backend execution latency. Instrument the `route()` and `execute()` methods with structured logging that captures `operation`, `backend`, `row_count`, `memory_bytes`, and `validation_status`. This telemetry enables platform teams to dynamically adjust thresholds, identify LLM prompt drift, and optimize connection pool sizing for peak spatial inference workloads.
