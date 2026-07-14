# Mapping Spatial API Errors to User-Friendly Prompts

In LLM-assisted geospatial pipelines, the translation layer between raw spatial API failures and actionable user prompts is frequently the primary bottleneck for autonomous agent reliability. When a PostGIS topology validator rejects a self-intersecting polygon, a vector tile service returns a silent CRS mismatch, or a GeoJSON parser encounters invalid ring orientation, the default stack trace provides zero semantic value to an end-user and actively degrades the reasoning context of a spatial agent. **Mapping Spatial API Errors to User-Friendly Prompts** requires a deterministic interception architecture that parses spatial-specific error codes, normalizes them into structured JSON payloads, and routes them to constrained prompt templates. This capability sits at the operational core of modern [Geospatial Prompt Engineering & Tool Routing](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/) systems, where failure recovery must be as deterministic as the initial tool invocation.

## The Bottleneck in Autonomous Geospatial Agents

Consider a production scenario where an LLM agent generates a spatial SQL query for a multi-step buffer-and-intersect operation across a distributed PostGIS cluster. The underlying API returns a `22023` (invalid parameter value) or a `ST_IsValid` topology exception. Without explicit error mapping, the agent defaults to a blind retry loop, exhausting connection pools and triggering rate-limit cascades. The root cause is rarely a malformed query; it is typically an unhandled geometry state: non-planar coordinates, invalid ring orientation, or a missing spatial index on a dynamically materialized CTE. Rapid pipeline resolution demands an error interceptor that extracts the exact spatial violation, validates the geometry state against `shapely` or `GEOS` diagnostics, and generates a targeted prompt that instructs the agent to apply a specific repair function (e.g., `ST_MakeValid`, `ST_SnapToGrid`, or explicit CRS transformation) before the next routing step.

## Deterministic Interceptor Architecture

The mapping layer must operate synchronously within the async execution graph to prevent state drift. We implement a strict schema-driven interceptor that catches exceptions from `psycopg2`, `GeoPandas`, or RESTful spatial endpoints, normalizes them into a canonical error taxonomy, and outputs a routing-ready prompt payload. The architecture enforces three non-negotiable constraints:
1. All geometry diagnostics must be validated against `shapely.validation` and coordinate bounds.
2. Error codes must map to explicit LLM instructions rather than open-ended suggestions.
3. Fallback routing must trigger when topology repair exceeds a deterministic threshold.

## Implementation: Schema-Driven Error Mapping & Validation

The following production-ready interceptor demonstrates explicit error handling, rigorous coordinate validation, and clear next steps for pipeline integration. It uses Pydantic for payload validation, Shapely for geometry diagnostics, and PyProj for CRS normalization.

```python
import logging
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ValidationError, field_validator
from shapely.validation import explain_validity
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger("spatial_error_mapper")

class SeverityLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class SpatialErrorPayload(BaseModel):
    error_code: str
    severity: SeverityLevel
    raw_exception: Optional[str] = None
    geometry_diagnostics: Optional[str] = None
    crs_status: str
    recommended_repair: str
    next_steps: List[str]
    retry_allowed: bool = Field(default=True)

    @field_validator("crs_status")
    @classmethod
    def validate_crs_format(cls, v: str) -> str:
        allowed = ["valid_epsg", "unknown", "mismatch", "missing"]
        if v not in allowed:
            raise ValueError(f"Invalid CRS status. Must be one of {allowed}")
        return v

class SpatialErrorInterceptor:
    def __init__(self, max_repair_attempts: int = 3):
        self.max_repair_attempts = max_repair_attempts

    def validate_coordinates(self, geom: BaseGeometry) -> Dict[str, Any]:
        """Explicit coordinate validation: bounds and validity checks."""
        diagnostics = {}
        try:
            validity_msg = explain_validity(geom)
            diagnostics["is_valid"] = validity_msg == "Valid Geometry"
            diagnostics["shapely_diagnostic"] = validity_msg

            # Coordinate bounds validation (WGS84 limits)
            minx, miny, maxx, maxy = geom.bounds
            if minx < -180 or maxx > 180 or miny < -90 or maxy > 90:
                diagnostics["bounds_violation"] = True
                diagnostics["bounds"] = (minx, miny, maxx, maxy)
            else:
                diagnostics["bounds_violation"] = False

        except Exception as e:
            logger.error(f"Coordinate validation failed: {e}")
            diagnostics["is_valid"] = False
            diagnostics["shapely_diagnostic"] = str(e)

        return diagnostics

    def map_error_to_prompt(
        self,
        exception: Exception,
        geom: Optional[BaseGeometry] = None,
        attempt: int = 0
    ) -> SpatialErrorPayload:
        """Maps raw spatial exceptions to structured LLM routing payloads."""
        error_str = str(exception)
        crs_status = "unknown"
        recommended_repair = "review_query_syntax"
        next_steps = []
        severity = SeverityLevel.WARNING
        retry_allowed = True

        # Explicit error handling for PostGIS/GEOS topology violations
        if "TopologyException" in error_str or "22023" in error_str:
            severity = SeverityLevel.CRITICAL
            recommended_repair = "ST_MakeValid"
            next_steps = [
                "Apply ST_MakeValid() to the target geometry before re-execution.",
                "If ST_MakeValid fails, run ST_SnapToGrid(geom, 1e-6) to resolve floating-point drift.",
                "Validate output with ST_IsValid() and route to human-in-the-loop if invalid."
            ]
        elif "CRS mismatch" in error_str or "transform" in error_str.lower():
            crs_status = "mismatch"
            recommended_repair = "CRS_Transformation"
            next_steps = [
                "Identify source EPSG code and target CRS using pyproj.Transformer.",
                "Apply gdf.to_crs(epsg=4326) or ST_Transform() before spatial joins.",
                "Re-run bounding box intersection to verify alignment."
            ]
        elif "Rate limit" in error_str or "429" in error_str:
            severity = SeverityLevel.INFO
            recommended_repair = "Exponential_Backoff"
            next_steps = [
                "Pause execution for 2^attempt seconds.",
                "Reduce batch size to 50% of original payload.",
                "Resume pipeline with cached successful geometries."
            ]
        else:
            severity = SeverityLevel.CRITICAL
            recommended_repair = "Manual_Review"
            next_steps = [
                "Inspect raw SQL/GeoJSON payload for malformed coordinates.",
                "Check spatial index existence on target tables.",
                "Escalate to platform engineering if retry threshold exceeded."
            ]

        # Coordinate validation integration
        geom_diagnostics = None
        if geom:
            geom_diagnostics = self.validate_coordinates(geom)
            if not geom_diagnostics.get("is_valid", False):
                crs_status = "unknown"
                next_steps.insert(0, f"Geometry invalid: {geom_diagnostics.get('shapely_diagnostic', 'Unknown')}")
                if attempt >= self.max_repair_attempts:
                    retry_allowed = False
                    next_steps.append("MAX_REPAIR_ATTEMPTS_EXCEEDED: Route to fallback LLM agent.")

        return SpatialErrorPayload(
            error_code=type(exception).__name__,
            severity=severity,
            raw_exception=error_str,
            geometry_diagnostics=str(geom_diagnostics) if geom_diagnostics else None,
            crs_status=crs_status,
            recommended_repair=recommended_repair,
            next_steps=next_steps,
            retry_allowed=retry_allowed
        )
```

## Taxonomy Routing: From Stack Traces to LLM Instructions

The interceptor above transforms opaque exceptions into deterministic routing signals. When integrated with an LLM orchestrator, the `SpatialErrorPayload` directly injects into the system prompt, constraining the agent's reasoning space. For example, a `TopologyException` triggers a prompt template that forces the agent to generate `ST_MakeValid` or `ST_SnapToGrid` rather than guessing at query syntax. This approach aligns with established [Error Mapping for Spatial API Calls](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/) patterns, where failure states become explicit tool-routing triggers rather than terminal exceptions.

Key routing rules for spatial agents:
- **Invalid Ring Orientation:** Map to `ST_ForcePolygonCCW()` or `shapely.ops.orient()`.
- **Self-Intersection/Sliver Polygons:** Map to `shapely.make_valid()` or `ST_MakeValid()`.
- **CRS Misalignment:** Map to `pyproj.Transformer.from_crs()` with explicit axis-order enforcement (`always_xy=True`).
- **Index Scan Failures:** Map to `CREATE INDEX CONCURRENTLY` or `ANALYZE` commands before retry.

## Async Pipeline Integration

Spatial geoprocessing workflows frequently mix synchronous validation with asynchronous execution. To prevent state drift, the error mapper must intercept before the async event loop processes the next tool call.

```python
import asyncio
from typing import Callable, Awaitable, Any

async def execute_with_spatial_interception(
    spatial_task: Callable[..., Awaitable[Any]],
    interceptor: SpatialErrorInterceptor,
    max_retries: int = 3
) -> Any:
    """Wraps async spatial execution with deterministic error mapping and routing."""
    attempt = 0
    while attempt <= max_retries:
        try:
            result = await spatial_task()
            return result
        except Exception as e:
            attempt += 1
            logger.warning(f"Spatial execution failed (attempt {attempt}/{max_retries}): {e}")

            payload = interceptor.map_error_to_prompt(e, attempt=attempt)

            if not payload.retry_allowed:
                logger.critical("Pipeline halted: Fallback routing triggered.")
                raise RuntimeError(f"Spatial pipeline exhausted retries. Payload: {payload.model_dump_json()}")

            logger.info(f"Routing repair prompt: {payload.recommended_repair}")
            logger.info(f"Next steps: {payload.next_steps}")

            # Exponential backoff before retry
            await asyncio.sleep(2 ** attempt)

            # In production, pass payload.next_steps to LLM system prompt here
            # to constrain the next tool invocation to the recommended repair.
```

### Pipeline Integration Checklist
1. **Wrap all spatial tool calls** with `execute_with_spatial_interception` to guarantee synchronous validation before async continuation.
2. **Inject `payload.next_steps`** into the LLM's `system` or `tool_choice` context to prevent hallucinated SQL or invalid GeoJSON generation.
3. **Enforce coordinate validation** at ingestion boundaries using `validate_coordinates()` to catch out-of-bounds or non-planar geometries before they enter the PostGIS cluster.
4. **Log structured payloads** to observability platforms (Datadog, OpenTelemetry) for topology violation trend analysis and automated rule tuning.
5. **Reference authoritative spatial standards** when designing validation thresholds: [PostGIS Geometry Validity Rules](https://postgis.net/docs/using_postgis_dbmanagement.html) and [Shapely Validation Documentation](https://shapely.readthedocs.io/en/stable/manual.html#validation.make_valid).

## Conclusion

Mapping Spatial API Errors to User-Friendly Prompts is not a cosmetic UX improvement; it is a foundational safety mechanism for autonomous geospatial AI. By intercepting raw exceptions, validating coordinate states deterministically, and routing failures to constrained prompt templates, platform teams eliminate blind retry loops and enforce topology rule compliance at scale. When paired with explicit error handling, rigorous coordinate validation, and clear next steps for pipeline integration, this architecture transforms spatial API failures from pipeline terminators into deterministic routing signals. The result is a resilient, self-healing geoprocessing stack capable of operating reliably across distributed PostGIS clusters, vector tile services, and async LLM orchestration layers.
