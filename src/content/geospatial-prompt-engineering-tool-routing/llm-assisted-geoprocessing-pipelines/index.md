# LLM-Assisted Geoprocessing Pipelines

LLM-Assisted Geoprocessing Pipelines represent a structural evolution in spatial data engineering, shifting from static, hard-coded ETL scripts to dynamic, intent-driven execution graphs. For AI/ML engineers, spatial data scientists, Python GIS developers, and platform teams, integrating generative models into production spatial workflows requires rigorous prompt engineering, deterministic tool routing, and explicit validation layers. This architecture operates within the broader [Geospatial Prompt Engineering & Tool Routing](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/) paradigm, where natural language intent is translated into executable spatial operations with guaranteed reproducibility and strict spatial integrity.

Unlike traditional tabular pipelines, spatial workflows introduce compounding failure modes: coordinate reference system (CRS) misalignment, topology violations, precision drift, and unbounded memory consumption during geometric operations. A production-ready LLM-assisted pipeline must enforce schema boundaries at ingestion, route operations deterministically based on computational complexity, and map spatial exceptions to a standardized error taxonomy before execution reaches the data layer.

<figure class="diagram">
<svg viewBox="0 0 860 300" role="img" aria-labelledby="lap-t lap-d" xmlns="http://www.w3.org/2000/svg">
  <title id="lap-t">LLM-assisted geoprocessing pipeline</title>
  <desc id="lap-d">Prompt spec, step planning, tool invocation, geometry validation, and pipeline output run in sequence to deliver LLM-orchestrated geoprocessing with tool routing, topology checks, and QA.</desc>
  <defs>
    <marker id="lap-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
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
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#lap-arrow)">
    <line x1="166" y1="95" x2="181" y2="95"/>
    <line x1="334" y1="95" x2="349" y2="95"/>
    <line x1="502" y1="95" x2="517" y2="95"/>
    <line x1="670" y1="95" x2="685" y2="95"/>
  </g>
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#lap-arrow)">
    <line x1="90" y1="137" x2="90" y2="202"/>
    <line x1="258" y1="137" x2="258" y2="202"/>
    <line x1="426" y1="137" x2="426" y2="202"/>
    <line x1="594" y1="137" x2="594" y2="202"/>
    <line x1="762" y1="137" x2="762" y2="202"/>
  </g>
  <rect x="15" y="205" width="822" height="58" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/>
  <g fill="#1f2937" font-size="13" text-anchor="middle">
    <text x="90" y="90"><tspan x="90" dy="0">Prompt</tspan><tspan x="90" dy="16">spec</tspan></text>
    <text x="258" y="90"><tspan x="258" dy="0">Step</tspan><tspan x="258" dy="16">planning</tspan></text>
    <text x="426" y="90"><tspan x="426" dy="0">Tool</tspan><tspan x="426" dy="16">invocation</tspan></text>
    <text x="594" y="90"><tspan x="594" dy="0">Geometry</tspan><tspan x="594" dy="16">validation</tspan></text>
    <text x="762" y="90"><tspan x="762" dy="0">Pipeline</tspan><tspan x="762" dy="16">output</tspan></text>
  </g>
  <text x="426" y="240" fill="#1f2937" font-size="15" font-weight="600" text-anchor="middle">LLM-orchestrated geoprocessing</text>
  <text x="426" y="287" fill="#5b6471" font-size="12" text-anchor="middle">Downstream: tool routing · topology checks · QA</text>
</svg>
</figure>

## Step 1: Constrained Prompt Design & Schema Validation

Open-ended LLM generation introduces unacceptable variance in production environments. The routing layer must reject free-form text and instead parse strictly validated JSON payloads. By leveraging schema validators, the orchestrator prompt defines allowable spatial operations, target backends, parameter constraints, and explicit CRS requirements. This eliminates hallucinated function names, prevents silent CRS mismatches, and guarantees that downstream executors receive deterministic, type-safe instructions.

The following Pydantic schema enforces spatial constraints at the prompt boundary. It validates CRS compatibility, restricts operations to a known enum, and requires tolerance thresholds for geometric operations.

```python
from enum import Enum
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, ValidationError
from pyproj import CRS

class SpatialOperation(str, Enum):
    BUFFER = "buffer"
    INTERSECT = "intersect"
    UNION = "union"
    SPATIAL_JOIN = "spatial_join"
    CLIP = "clip"

class GeoprocessingPayload(BaseModel):
    operation: SpatialOperation
    backend: Literal["geopandas", "postgis"]
    source_table: str
    target_table: Optional[str] = None
    input_crs: str = Field(..., description="EPSG code (e.g., 'EPSG:4326')")
    output_crs: str
    tolerance_meters: float = Field(ge=0.0, le=1000.0)
    parameters: dict = Field(default_factory=dict)

    @field_validator("input_crs", "output_crs")
    @classmethod
    def validate_crs(cls, v: str) -> str:
        try:
            crs = CRS.from_user_input(v)
            if not crs.is_geographic and not crs.is_projected:
                raise ValueError(f"Unsupported CRS type: {v}")
            return crs.to_string()
        except Exception as e:
            raise ValueError(f"Invalid CRS definition: {e}")

    @field_validator("backend")
    @classmethod
    def enforce_backend_constraints(cls, v: str, info) -> str:
        op = info.data.get("operation")
        if op in (SpatialOperation.SPATIAL_JOIN, SpatialOperation.UNION) and v == "geopandas":
            raise ValueError("Heavy topological operations must route to PostGIS backend")
        return v
```

When integrating this schema into an LLM prompt, the system prompt must explicitly forbid free-text outputs and mandate JSON-only responses matching the schema. Validation failures should immediately halt execution and return a structured error code rather than attempting fallback execution on malformed instructions.

## Step 2: Deterministic Tool Routing & Execution

Once validated, the payload routes to the appropriate spatial backend based on computational complexity, data volume, and memory constraints. Lightweight transformations, attribute joins, and simple buffers route to in-memory workflows. Heavy spatial predicates, indexed spatial joins, and large-scale aggregations route to PostGIS.

The execution layer must enforce strict connection pooling, query timeouts, and explicit `EXPLAIN ANALYZE` hooks for database operations. For in-memory execution, GeoDataFrames must be instantiated with pre-validated CRS alignment, memory caps, and explicit `.copy()` semantics to prevent reference mutation.

```python
import geopandas as gpd
import psycopg2
from contextlib import contextmanager
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SpatialRouter:
    def __init__(self, db_config: Dict[str, str], memory_limit_mb: int = 2048):
        self.db_config = db_config
        self.memory_limit_mb = memory_limit_mb

    @contextmanager
    def _get_db_connection(self):
        conn = psycopg2.connect(**self.db_config, connect_timeout=10)
        try:
            yield conn
        finally:
            conn.close()

    def execute(self, payload: GeoprocessingPayload) -> Dict[str, Any]:
        if payload.backend == "postgis":
            return self._execute_postgis(payload)
        return self._execute_geopandas(payload)

    def _execute_geopandas(self, payload: GeoprocessingPayload) -> Dict[str, Any]:
        logger.info(f"Routing to in-memory backend: {payload.operation}")
        gdf = gpd.read_file(payload.source_table)

        # Enforce CRS
        if gdf.crs is None or gdf.crs.to_string() != payload.input_crs:
            gdf = gdf.to_crs(payload.input_crs)

        gdf = gdf.copy()

        if payload.operation == SpatialOperation.BUFFER:
            # Buffer in meters requires a projected CRS; reproject if geographic
            if gdf.crs.is_geographic:
                gdf = gdf.to_crs("EPSG:3857")
            gdf["geometry"] = gdf.geometry.buffer(payload.tolerance_meters)
            gdf = gdf.to_crs(payload.output_crs)

        return {"status": "success", "row_count": len(gdf), "crs": gdf.crs.to_string()}

    def _execute_postgis(self, payload: GeoprocessingPayload) -> Dict[str, Any]:
        logger.info(f"Routing to PostGIS backend: {payload.operation}")
        with self._get_db_connection() as conn:
            with conn.cursor() as cur:
                # Use EXPLAIN ANALYZE for performance auditing in non-production
                query = """
                    SELECT ST_Transform(
                        ST_Intersection(a.geom, b.geom),
                        %s
                    ) AS geom
                    FROM %s a
                    JOIN %s b ON ST_Intersects(a.geom, b.geom)
                    WHERE ST_IsValid(a.geom) AND ST_IsValid(b.geom);
                """
                # Note: table names cannot be parameterized safely with %s in psycopg2;
                # use psycopg2.sql.Identifier for production table name injection.
                from psycopg2 import sql
                safe_query = sql.SQL("""
                    SELECT ST_Transform(
                        ST_Intersection(a.geom, b.geom),
                        %(output_srid)s
                    ) AS geom
                    FROM {src} a
                    JOIN {tgt} b ON ST_Intersects(a.geom, b.geom)
                    WHERE ST_IsValid(a.geom) AND ST_IsValid(b.geom)
                """).format(
                    src=sql.Identifier(payload.source_table),
                    tgt=sql.Identifier(payload.target_table or payload.source_table)
                )
                cur.execute(safe_query, {"output_srid": int(payload.output_crs.split(":")[-1])})
                conn.commit()
        return {"status": "success"}
```

This routing pattern ensures that computationally expensive operations never saturate application memory. When constructing parameterized PostGIS queries from natural language, the system leverages [Prompt-to-Spatial-SQL Generation](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/) patterns to maintain index utilization and prevent query plan degradation.

## Step 3: Explicit Validation & Error Mapping

Spatial pipelines fail differently than traditional tabular pipelines. Geometry validity, precision loss, and topology violations are the primary failure vectors. A production pipeline must validate outputs before committing them to storage, map spatial exceptions to a standardized taxonomy, and optionally trigger LLM-assisted topology correction routines.

The following validation module enforces geometry integrity, applies tolerance-based snapping, and maps errors to actionable codes. It integrates seamlessly with the execution layer and provides hooks for automated topology rule enforcement.

```python
from shapely.validation import make_valid
from shapely.geometry.base import BaseGeometry
import traceback

class SpatialErrorTaxonomy:
    INVALID_GEOMETRY = "ERR_GEO_001"
    TOPOLOGY_VIOLATION = "ERR_TOPO_002"
    CRS_MISMATCH = "ERR_CRS_003"
    EXECUTION_TIMEOUT = "ERR_EXEC_004"

def validate_and_map_output(gdf: gpd.GeoDataFrame, payload: GeoprocessingPayload) -> Dict[str, Any]:
    try:
        # Enforce CRS alignment post-execution
        if gdf.crs is None or gdf.crs.to_string() != payload.output_crs:
            raise ValueError(SpatialErrorTaxonomy.CRS_MISMATCH)

        # Geometry validation & repair
        invalid_mask = ~gdf.geometry.is_valid
        if invalid_mask.any():
            logger.warning(f"Repairing {invalid_mask.sum()} invalid geometries")
            gdf = gdf.copy()
            gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].apply(make_valid)

        # Confirm all geometries are now valid
        if not gdf.geometry.is_valid.all():
            raise ValueError(SpatialErrorTaxonomy.TOPOLOGY_VIOLATION)

        return {"status": "valid", "data": gdf}

    except Exception as e:
        error_code = str(e) if str(e).startswith("ERR_") else SpatialErrorTaxonomy.INVALID_GEOMETRY
        logger.error(f"Spatial validation failed: {error_code} | {traceback.format_exc()}")
        return {"status": "failed", "error_code": error_code, "trace": traceback.format_exc()}
```

When topology violations persist after automated repair, the pipeline can route the failure context to an LLM for rule interpretation and constraint relaxation. This pattern is detailed in [Topology Rule Enforcement via LLMs](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/), which covers how generative models can dynamically adjust tolerance thresholds or suggest alternative spatial predicates without compromising data integrity.

## Step 4: Production Deployment & Observability

Deploying LLM-assisted geoprocessing pipelines requires explicit observability and execution guarantees. Spatial operations are inherently non-deterministic in runtime due to data skew, index fragmentation, and CRS transformation overhead. Platform teams should implement:

1. **Async/Sync Workflow Segregation**: Route synchronous requests to lightweight in-memory operations, while offloading batch PostGIS jobs to message queues (e.g., Celery, RabbitMQ) with explicit timeout guards.
2. **Structured Spatial Logging**: Capture CRS transformations, geometry repair counts, and execution plans in JSON-formatted logs for downstream auditing.
3. **Metric Collection**: Expose Prometheus metrics for `spatial_operation_duration_seconds`, `geometry_repair_rate`, and `backend_routing_distribution`.
4. **Connection Pooling & Query Timeouts**: Use `SQLAlchemy` or `psycopg2` connection pools with `statement_timeout` set at the database level to prevent runaway spatial joins.

By combining schema-enforced prompt boundaries, deterministic backend routing, and explicit topology validation, LLM-Assisted Geoprocessing Pipelines deliver reproducible, production-grade spatial data engineering. This architecture bridges the gap between natural language intent and geospatial execution, ensuring that AI-driven workflows maintain the rigor required for enterprise mapping, environmental modeling, and infrastructure planning.
