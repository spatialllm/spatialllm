---
layout: overview.njk
order: 2
icon: routing
---

# Geospatial Prompt Engineering & Tool Routing

Production-grade spatial AI agents fail when prompt ambiguity collides with rigid geometric constraints. Unlike text-only workflows, geospatial reasoning requires explicit coordinate reference system (CRS) handling, topology validation, and deterministic execution routing. This guide establishes a validation-first architecture for Spatial LLM and AI Agent Workflows, prioritizing schema-grounded prompting, resilient tool dispatch, and observable pipeline integration.

<figure class="diagram">
<svg viewBox="0 0 860 300" role="img" aria-labelledby="gpe-t gpe-d" xmlns="http://www.w3.org/2000/svg">
  <title id="gpe-t">Prompt engineering &amp; tool routing pipeline</title>
  <desc id="gpe-d">Prompt parsing, intent detection, tool selection, parameter binding, and execution with validation run in sequence to produce routed geoprocessing tool calls that drive SQL generation, pipelines, and error mapping.</desc>
  <defs>
    <marker id="gpe-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
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
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#gpe-arrow)">
    <line x1="166" y1="95" x2="181" y2="95"/>
    <line x1="334" y1="95" x2="349" y2="95"/>
    <line x1="502" y1="95" x2="517" y2="95"/>
    <line x1="670" y1="95" x2="685" y2="95"/>
  </g>
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#gpe-arrow)">
    <line x1="90" y1="137" x2="90" y2="202"/>
    <line x1="258" y1="137" x2="258" y2="202"/>
    <line x1="426" y1="137" x2="426" y2="202"/>
    <line x1="594" y1="137" x2="594" y2="202"/>
    <line x1="762" y1="137" x2="762" y2="202"/>
  </g>
  <rect x="15" y="205" width="822" height="58" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/>
  <g fill="#1f2937" font-size="13" text-anchor="middle">
    <text x="90" y="90"><tspan x="90" dy="0">Prompt</tspan><tspan x="90" dy="16">parsing</tspan></text>
    <text x="258" y="90"><tspan x="258" dy="0">Intent</tspan><tspan x="258" dy="16">detection</tspan></text>
    <text x="426" y="90"><tspan x="426" dy="0">Tool</tspan><tspan x="426" dy="16">selection</tspan></text>
    <text x="594" y="90"><tspan x="594" dy="0">Param</tspan><tspan x="594" dy="16">binding</tspan></text>
    <text x="762" y="90"><tspan x="762" dy="0">Execution</tspan><tspan x="762" dy="16">&amp; validate</tspan></text>
  </g>
  <text x="426" y="240" fill="#1f2937" font-size="15" font-weight="600" text-anchor="middle">Routed geoprocessing tool calls</text>
  <text x="426" y="287" fill="#5b6471" font-size="12" text-anchor="middle">Downstream: SQL generation · pipelines · error mapping</text>
</svg>
</figure>

## Validation-First Architecture for Spatial Agents

Geospatial operations are unforgiving of implicit assumptions. A spatial join across mismatched projections, a buffer operation on unprojected coordinates, or a topology violation in a cadastral dataset will silently corrupt downstream analytics. Validation-first architecture mandates pre-flight checks before any LLM invocation or spatial execution.

The core pattern separates intent parsing from geometric execution. Prompts are normalized into structured spatial contracts containing explicit CRS identifiers, bounding box constraints, and expected output geometries. A validation gate verifies schema compatibility, checks topology prerequisites, and routes the request to the appropriate execution backend. This decoupling prevents hallucinated SQL, invalid WKT/WKB payloads, and unbounded compute costs.

```python
import logging
from typing import Dict, Any
from pyproj import CRS, Transformer
from shapely.geometry import shape, box, mapping
from shapely.validation import make_valid
from shapely.ops import transform

logger = logging.getLogger(__name__)

class SpatialValidationError(Exception):
    """Custom exception for spatial validation failures."""
    pass

class SpatialValidationGate:
    def __init__(self, target_crs: str = "EPSG:4326"):
        try:
            self.target_crs = CRS.from_string(target_crs)
        except Exception as e:
            raise ValueError(f"Invalid target CRS '{target_crs}': {e}")

    def normalize_crs(self, geometry: Any, source_crs_str: str) -> Any:
        """Safely transform geometry to target CRS with error handling."""
        try:
            source_crs = CRS.from_string(source_crs_str)
            if source_crs.equals(self.target_crs):
                return geometry
            transformer = Transformer.from_crs(source_crs, self.target_crs, always_xy=True)
            return transform(transformer.transform, geometry)
        except Exception as e:
            raise SpatialValidationError(f"CRS normalization failed: {e}")

    def validate_and_normalize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-flight validation for spatial agent payloads."""
        try:
            if "geometry" not in payload:
                raise SpatialValidationError("Missing 'geometry' field in spatial payload")

            geom = shape(payload["geometry"])
            if not geom.is_valid:
                logger.warning("Invalid geometry detected. Applying make_valid()")
                geom = make_valid(geom)

            if "crs" in payload and payload["crs"]:
                geom = self.normalize_crs(geom, payload["crs"])
                payload["crs"] = self.target_crs.to_string()
            else:
                payload["crs"] = self.target_crs.to_string()

            payload["geometry"] = mapping(geom)
            payload["bbox"] = list(geom.bounds)
            return payload

        except SpatialValidationError as e:
            logger.error(f"Validation gate rejected payload: {e}")
            raise
        except Exception as e:
            logger.critical(f"Unexpected validation failure: {e}")
            raise SpatialValidationError(f"Payload processing failed: {e}")
```

## Spatial Prompt Structuring & Schema Grounding

Effective spatial prompt engineering requires explicit grounding in database schemas, spatial indexes, and geometric predicates. LLMs must be constrained to operate within known spatial functions (`ST_Intersects`, `ST_DWithin`, `ST_Buffer`) rather than inventing syntactically plausible but semantically invalid operations. Prompt templates should enforce parameterized spatial filters, explicit join conditions, and bounded output formats.

Schema grounding transforms natural language requests into typed spatial contracts. By leveraging JSON Schema or Pydantic models, you force the LLM to populate only recognized fields, eliminating ambiguous phrasing like "nearby" or "close to" without quantifiable thresholds. The [Prompt-to-Spatial-SQL Generation](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/) pattern demonstrates how to map these grounded prompts directly to parameterized PostGIS queries.

```python
import json
from pydantic import BaseModel, Field, ValidationError
from typing import Literal, Optional, Dict, Any
from pyproj import CRS

class SpatialFilterContract(BaseModel):
    operation: Literal["buffer", "intersect", "within_distance", "spatial_join"]
    target_crs: str = Field(default="EPSG:4326", description="Target coordinate reference system")
    geometry: dict = Field(..., description="GeoJSON geometry object")
    distance_meters: Optional[float] = Field(None, ge=0, description="Distance threshold in meters")
    output_format: Literal["geojson", "wkt", "bbox"] = "geojson"

def parse_spatial_prompt(llm_response_json: str) -> Dict[str, Any]:
    """Parse LLM output into a validated spatial contract."""
    try:
        contract = SpatialFilterContract.model_validate_json(llm_response_json)
        # Validate CRS format explicitly
        CRS.from_string(contract.target_crs)
        return contract.model_dump()
    except ValidationError as e:
        raise ValueError(f"LLM output failed schema validation: {e.errors()}")
    except Exception as e:
        raise ValueError(f"Failed to parse spatial contract: {e}")
```

## Deterministic Tool Routing & Execution Dispatch

Once a spatial contract passes validation and schema grounding, the agent must route execution to the optimal backend. Routing decisions depend on dataset volume, geometric complexity, latency SLAs, and available compute resources. In-memory operations via [GeoPandas & PostGIS Tool Routing](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/geopandas-postgis-tool-routing/) require distinct dispatch strategies: lightweight filtering and small-batch transformations route to local GeoDataFrames, while heavy spatial joins, raster processing, or multi-GB vector operations delegate to PostGIS or cloud-native engines.

Deterministic routing also requires explicit handling of synchronous versus asynchronous execution paths. Interactive queries demand sub-second responses, while batch analytics tolerate queued execution. The [Async vs Sync Geoprocessing Workflows](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/) framework outlines how to implement circuit breakers, timeout guards, and worker pool allocation based on payload metadata.

```python
import asyncio
from enum import Enum
from typing import Callable

class ExecutionMode(Enum):
    SYNC = "sync"
    ASYNC = "async"

class SpatialRouter:
    def __init__(self, max_sync_rows: int = 5000, default_crs: str = "EPSG:4326"):
        self.max_sync_rows = max_sync_rows
        self.default_crs = default_crs

    def route_execution(self, contract: Dict[str, Any], estimated_rows: int) -> ExecutionMode:
        """Determine execution mode based on payload size and complexity."""
        if estimated_rows > self.max_sync_rows or contract.get("operation") == "spatial_join":
            return ExecutionMode.ASYNC
        return ExecutionMode.SYNC

    async def dispatch(self, contract: Dict[str, Any], executor: Callable) -> Dict[str, Any]:
        """Execute spatial operation with strict error handling and CRS normalization."""
        try:
            mode = self.route_execution(contract, estimated_rows=contract.get("row_count", 0))

            if mode == ExecutionMode.ASYNC:
                result = await asyncio.to_thread(executor, contract)
            else:
                result = executor(contract)

            return {"status": "success", "mode": mode.value, "result": result}

        except Exception as e:
            return {"status": "error", "code": "EXECUTION_FAILED", "message": str(e)}
```

## Topology Enforcement & Geometric Integrity

Geospatial AI systems frequently generate or consume geometries that violate spatial topology rules. Overlapping polygons, self-intersecting linestrings, and sliver geometries emerge from imprecise digitization, aggressive snapping, or LLM-generated coordinate sequences. Without explicit enforcement, these artifacts propagate through analytical pipelines, producing false positives in spatial relationships and corrupting area/perimeter calculations.

Topology validation must occur both at ingestion and post-processing stages. Implementing [Topology Rule Enforcement via LLMs](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/) enables agents to self-correct geometric outputs by applying rule-based constraints (e.g., `ST_IsValid`, `ST_SnapToGrid`, `ST_MakeValid`) before returning results to downstream consumers. The OGC Simple Features specification defines the mathematical foundation for these predicates, and adherence to [OGC Simple Features](https://www.ogc.org/standard/sfa/) ensures cross-platform interoperability.

```python
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import explain_validity
from shapely.validation import make_valid
from typing import Any

def enforce_topology(geom: Any, tolerance: float = 1e-6) -> Any:
    """Apply topology rules and return a valid geometry or raise."""
    if not geom.is_valid:
        reason = explain_validity(geom)
        logger.warning(f"Topology violation detected: {reason}")
        geom = make_valid(geom)

    # Simplify to eliminate floating-point slivers
    if tolerance > 0:
        geom = geom.simplify(tolerance, preserve_topology=True)

    if not geom.is_valid:
        raise SpatialValidationError("Geometry remains invalid after topology enforcement")

    return geom
```

## Error Mapping & Observable Pipeline Integration

Spatial AI pipelines require structured error mapping to distinguish between transient network failures, CRS mismatches, invalid geometries, and LLM hallucinations. Unstructured stack traces provide zero actionable signal for automated retry logic or circuit breakers. By implementing standardized error taxonomies, agents can gracefully degrade, fallback to cached results, or request human-in-the-loop clarification.

Integrating [Error Mapping for Spatial API Calls](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/) ensures that every spatial operation emits telemetry aligned with OpenTelemetry standards. Metrics should track CRS conversion latency, topology validation failure rates, and tool routing distribution. When combined with [LLM-Assisted Geoprocessing Pipelines](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/), these observability hooks enable continuous prompt refinement, automatic schema updates, and predictive scaling based on spatial workload patterns.

```python
import traceback
from enum import Enum

class SpatialErrorType(Enum):
    CRS_MISMATCH = "CRS_MISMATCH"
    TOPOLOGY_VIOLATION = "TOPOLOGY_VIOLATION"
    LLM_HALLUCINATION = "LLM_HALLUCINATION"
    EXECUTION_TIMEOUT = "EXECUTION_TIMEOUT"
    UNKNOWN = "UNKNOWN"

def map_spatial_error(exception: Exception) -> Dict[str, Any]:
    """Map raw exceptions to structured spatial error payloads."""
    error_type = SpatialErrorType.UNKNOWN
    message = str(exception)

    if "CRS" in message.upper() or "projection" in message.lower():
        error_type = SpatialErrorType.CRS_MISMATCH
    elif "invalid" in message.lower() and ("geometry" in message.lower() or "topology" in message.lower()):
        error_type = SpatialErrorType.TOPOLOGY_VIOLATION
    elif "timeout" in message.lower() or "cancelled" in message.lower():
        error_type = SpatialErrorType.EXECUTION_TIMEOUT
    elif "hallucinat" in message.lower() or "schema" in message.lower():
        error_type = SpatialErrorType.LLM_HALLUCINATION

    return {
        "error_type": error_type.value,
        "message": message,
        "traceback": traceback.format_exc(),
        "retryable": error_type in (SpatialErrorType.EXECUTION_TIMEOUT, SpatialErrorType.UNKNOWN)
    }
```

## Function-Calling Schemas for Spatial Tools

Free-text tool arguments are where spatial agents leak invalid state: a model emits a distance with no unit, a geometry with no SRID, or an operation the backend never implemented. Constraining the model to a typed function-calling contract — enumerated SRIDs, bounded distances, closed sets of operations — turns "hope the model behaves" into a schema the dispatcher can reject before any query runs.

```python
import logging
from pydantic import BaseModel, Field, ValidationError
from typing import Literal

logger = logging.getLogger(__name__)

class WithinDistanceCall(BaseModel):
    """Typed schema for a single spatial tool call the LLM may emit."""
    op: Literal["within_distance"]
    srid: Literal[4326, 3857, 26910]
    meters: float = Field(gt=0, le=50_000)

def validate_tool_call(raw: str) -> WithinDistanceCall | None:
    """Reject malformed spatial tool calls; return None so the caller can fall back."""
    try:
        return WithinDistanceCall.model_validate_json(raw)
    except ValidationError as exc:
        logger.warning("Rejected spatial tool call: %s", exc.errors())
        return None  # deterministic fallback: caller re-prompts or uses a safe default
```

Designing these contracts so the model can only produce index-aware, executable calls is covered in [Spatial Function-Calling Schemas](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/).

## Multi-Step Spatial Agent Orchestration

Real spatial tasks rarely resolve in one call — they plan, invoke geoprocessing tools, inspect the intermediate geometry, and continue. Without checkpointed state and per-step validation, a single bad step corrupts everything downstream silently. Orchestration must validate each intermediate result and be able to roll back to the last known-good checkpoint rather than compounding an error across the chain.

```python
import logging
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

def run_step(geom: BaseGeometry, tool, checkpoint: BaseGeometry) -> BaseGeometry:
    """Run one geoprocessing step; roll back to the checkpoint on invalid output."""
    try:
        result = tool(geom)
        if result.is_empty or not result.is_valid:
            raise ValueError("step produced empty/invalid geometry")
        return result
    except Exception as exc:  # noqa: BLE001
        logger.error("Step failed (%s); rolling back to checkpoint", exc)
        return checkpoint  # deterministic fallback preserves chain integrity
```

Checkpointing, resumption, and rollback across chained tools are detailed in [Multi-Step Spatial Agent Orchestration](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/).

## Production Readiness Checklist

Deploying geospatial prompt engineering & tool routing at scale requires disciplined adherence to validation, routing, and observability standards. Before promoting spatial agents to production, verify the following:

- **CRS Normalization:** All payloads are explicitly transformed to a canonical CRS before execution.
- **Schema Grounding:** LLM outputs are constrained by Pydantic/JSON Schema with strict spatial type validation.
- **Topology Enforcement:** Invalid geometries trigger `make_valid()` or are rejected with actionable error codes.
- **Deterministic Routing:** Payload size and operation type dictate sync/async dispatch with circuit breakers.
- **Structured Telemetry:** Errors map to standardized taxonomies; metrics track validation latency and routing distribution.
- **Fallback Mechanisms:** Failed LLM generations trigger deterministic spatial fallbacks (e.g., cached PostGIS queries or rule-based heuristics).

By treating spatial reasoning as a constrained, validation-first discipline rather than an open-ended generative task, engineering teams can build AI agents that scale reliably across cadastral, environmental, and logistics domains.
