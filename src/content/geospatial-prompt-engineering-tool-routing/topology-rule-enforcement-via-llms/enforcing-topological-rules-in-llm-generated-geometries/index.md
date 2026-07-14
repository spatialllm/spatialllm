# Enforcing Topological Rules in LLM-Generated Geometries

Large language models excel at syntactic pattern matching but operate without intrinsic spatial reasoning. When tasked with coordinate synthesis, polygon construction, or spatial SQL generation, they frequently produce geometrically invalid outputs. **Enforcing Topological Rules in LLM-Generated Geometries** requires a deterministic validation layer that intercepts probabilistic outputs before they propagate to downstream analytics, rendering pipelines, or spatial databases. This article details a production-grade validation architecture, focusing on failure modes, root causes, and a reproducible routing workflow that guarantees OGC Simple Features compliance.

## Failure Modes and Root Causes

LLMs treat coordinate arrays as floating-point tokens rather than topological primitives. The autoregressive decoder optimizes for structural plausibility (valid JSON/GeoJSON syntax) rather than spatial validity. In production pipelines, this manifests as three primary failure modes:

1. **Self-Intersecting Rings**: Models frequently generate bowtie polygons or overlapping edges when predicting boundary coordinates. The underlying attention mechanism lacks awareness of planar graph constraints, causing edges to cross without explicit node insertion.
2. **Precision-Induced Slivers**: Floating-point rounding during token decoding introduces sub-millimeter gaps or overlaps. When geometries are snapped to a grid or transformed across CRS boundaries, these micro-artifacts violate `ST_IsValid` checks and corrupt spatial joins.
3. **Unclosed or Degenerate Rings**: Missing terminal coordinate duplication or collinear vertex sequences produce degenerate geometries that pass JSON schema validation but fail spatial index insertion.

The root cause is architectural: sequential token generation is inherently unconstrained by geometric topology. Without explicit validation gates, invalid geometries propagate silently until they trigger PostGIS `GEOS` exceptions or cause silent topology corruption in analytical workflows.

## Validation Pipeline Architecture

A robust pipeline treats LLM output as untrusted input. Validation must be synchronous and blocking at the ingestion boundary. Asynchronous re-generation or tool routing should only trigger after deterministic validation fails. This separation of concerns aligns with established [Topology Rule Enforcement via LLMs](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/) patterns, where the model acts as a coordinator rather than a geometry engine.

The validation layer should implement:
- Strict schema validation via `pydantic` to enforce coordinate array structure
- OGC compliance checks using `shapely.validation.is_valid_reason()` for precise error mapping
- Deterministic repair routing (`make_valid`, precision snapping)
- Circuit breakers to prevent infinite retry loops on structurally invalid prompts

## Coordinate Validation & Explicit Error Handling

The following implementation demonstrates a production-ready validation gate. It enforces coordinate bounds, ring closure, and explicit error routing before any geometry reaches a spatial index or database.

```python
import json
import logging
from typing import List, Tuple, Dict, Any
from pydantic import BaseModel, field_validator, ValidationError
from shapely.geometry import shape
from shapely.validation import is_valid_reason
import shapely

logger = logging.getLogger("geo_validation")

class CoordinateBoundsError(Exception): pass
class TopologyValidationError(Exception): pass

class LLMGeoJSONPolygon(BaseModel):
    """Validates a GeoJSON Polygon coordinate array (list of rings)."""
    type: str
    coordinates: List[List[Tuple[float, float]]]

    @field_validator('coordinates')
    @classmethod
    def validate_coordinate_structure(cls, v: List[List[Tuple[float, float]]]) -> List[List[Tuple[float, float]]]:
        for ring_idx, ring in enumerate(v):
            if len(ring) < 4:
                raise ValueError(f"Ring {ring_idx} has <4 coordinates (minimum for closed polygon)")
            if ring[0] != ring[-1]:
                raise ValueError(f"Ring {ring_idx} is unclosed: first/last coordinates mismatch")
            for x, y in ring:
                if not (-180.0 <= x <= 180.0 and -90.0 <= y <= 90.0):
                    raise CoordinateBoundsError(f"Coordinate ({x}, {y}) exceeds WGS84 bounds")
        return v

def validate_and_route_llm_geometry(raw_output: str) -> Dict[str, Any]:
    """
    Synchronous validation gate for LLM-generated polygon geometries.
    Returns structured routing instructions for downstream pipelines.
    """
    try:
        parsed = json.loads(raw_output)
        feature = LLMGeoJSONPolygon(**parsed)
        geom = shape(feature.model_dump())

        # Explicit topology check
        validity_reason = is_valid_reason(geom)
        if validity_reason != "Valid Geometry":
            raise TopologyValidationError(validity_reason)

        return {
            "status": "valid",
            "geometry": geom,
            "routing": "direct_ingest"
        }
    except ValidationError as e:
        logger.error(f"Schema/Coordinate Validation Failed: {e}")
        return {"status": "invalid", "error_type": "schema_or_coord", "details": str(e), "routing": "prompt_retry"}
    except TopologyValidationError as e:
        logger.warning(f"Topology Violation: {e}")
        try:
            # Deterministic repair routing
            repaired = shapely.make_valid(geom)
            # Apply precision snapping to eliminate micro-slivers
            repaired = shapely.set_precision(repaired, grid_size=1e-6)
            if repaired.is_valid and not repaired.is_empty:
                return {"status": "repaired", "geometry": repaired, "routing": "repair_ingest"}
            raise TopologyValidationError("Repair produced invalid or empty geometry")
        except Exception as repair_err:
            logger.critical(f"Repair Failed: {repair_err}")
            return {"status": "failed", "error_type": "topology", "details": str(repair_err), "routing": "circuit_break"}
    except Exception as e:
        return {"status": "failed", "error_type": "unknown", "details": str(e), "routing": "circuit_break"}
```

This implementation guarantees that malformed tokens never reach the spatial engine. The `routing` key dictates downstream behavior: `direct_ingest` for valid outputs, `repair_ingest` for successfully corrected geometries, `prompt_retry` for structural failures, and `circuit_break` for unrecoverable topology violations.

## Deterministic Repair & Routing Workflow

When validation fails, the pipeline must avoid blind regeneration. Instead, it should route to deterministic repair functions or structured prompt refinement. The repair sequence follows a strict hierarchy:

1. **Precision Snapping**: `shapely.set_precision(geom, grid_size)` aligns vertices to a fixed grid to eliminate floating-point slivers.
2. **make_valid**: Resolves self-intersections by reconstructing rings according to GEOS topology rules.
3. **Node Insertion**: For complex overlaps, explicit planar graph reconstruction is required.

If repair succeeds, the geometry is tagged with a `repaired` flag and routed to a staging table for QA review. If repair fails, the pipeline triggers a structured prompt refinement loop. This approach mirrors best practices in [Geospatial Prompt Engineering & Tool Routing](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/), where the LLM receives explicit error context rather than raw failure signals.

```python
from sqlalchemy import text

def route_to_spatial_engine(result: Dict[str, Any], db_engine) -> None:
    routing_action = result.get("routing")
    if routing_action == "direct_ingest":
        with db_engine.connect() as conn:
            conn.execute(
                text("INSERT INTO spatial_data (geom) VALUES (ST_SetSRID(ST_GeomFromWKB(:wkb), 4326))"),
                {"wkb": result["geometry"].wkb}
            )
            conn.commit()
    elif routing_action == "repair_ingest":
        with db_engine.connect() as conn:
            conn.execute(
                text("INSERT INTO spatial_data (geom, validation_status) VALUES (ST_SetSRID(ST_GeomFromWKB(:wkb), 4326), 'repaired')"),
                {"wkb": result["geometry"].wkb}
            )
            conn.commit()
    elif routing_action == "prompt_retry":
        # Trigger async prompt re-generation with explicit error mapping
        trigger_prompt_refinement(error=result["details"], context="topology_violation")
    elif routing_action == "circuit_break":
        logger.critical("Circuit breaker activated: halting geometry ingestion for this batch.")
        raise RuntimeError("Unrecoverable geometry validation failure")
```

## Pipeline Integration & Next Steps

### 1. Synchronous Ingestion Boundary
Place the validation gate immediately after LLM response parsing. Do not defer validation to batch jobs. Synchronous blocking ensures that invalid geometries never enter message queues or data lakes.

### 2. Error Mapping for Spatial API Calls
Map GEOS/Shapely validation strings to standardized error codes:
- `"Self-intersection[0 0]"` → `ERR_TOPOLOGY_SELF_INTERSECT`
- `"Duplicate coordinate"` → `ERR_TOPOLOGY_DUPLICATE_VERTEX`
- `"Ring not closed"` → `ERR_SCHEMA_UNCLOSED_RING`

This mapping enables automated prompt correction and reduces manual debugging overhead.

### 3. Async vs Sync Geoprocessing Workflows
Use synchronous validation for real-time API responses and UI rendering. Route batch processing to asynchronous workers that apply `ST_IsValid` and `ST_MakeValid` at the database level, using the LLM validation layer as a pre-filter. Refer to official [PostGIS documentation on spatial validity](https://postgis.net/docs/ST_IsValid.html) for database-side enforcement patterns.

### 4. Monitoring & Circuit Breakers
Track validation failure rates per prompt template. Implement exponential backoff and circuit breakers when failure rates exceed 15%. Log `is_valid_reason()` outputs to a time-series database to identify recurring LLM hallucination patterns.

### Clear Next Steps for Platform Teams
1. **Deploy Validation Middleware**: Wrap all LLM geometry endpoints with the synchronous `validate_and_route_llm_geometry` function.
2. **Standardize Error Codes**: Implement the error mapping schema across all spatial API consumers.
3. **Integrate with Prompt Routing**: Connect validation failures to a structured prompt refinement service that appends explicit topology constraints to subsequent generations.
4. **Audit Repair Success Rates**: Monitor the ratio of `direct_ingest` vs `repair_ingest` to evaluate LLM spatial reasoning improvements over time.
5. **Enforce OGC Compliance at Scale**: Validate all outputs against the [OGC Simple Features Specification](https://www.ogc.org/standard/sfa/) before committing to production spatial indexes.

## Conclusion

Enforcing Topological Rules in LLM-Generated Geometries is not optional; it is a foundational requirement for safe, production-grade spatial AI. By treating LLM outputs as probabilistic approximations and wrapping them in deterministic validation, repair, and routing layers, platform teams can prevent silent topology corruption, eliminate cascading spatial SQL failures, and maintain strict OGC compliance. The architecture outlined here provides a reproducible, debuggable pathway from raw token generation to trusted spatial data.
