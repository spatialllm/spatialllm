# Error Mapping for Spatial API Calls

Error Mapping for Spatial API Calls is a foundational discipline for production-grade geospatial systems where raw HTTP status codes rarely capture the semantic reality of spatial failures. Unlike conventional REST endpoints, spatial services frequently return `200 OK` responses containing non-closed polygons, silent coordinate reference system (CRS) mismatches, or topology violations that only surface during downstream rendering or spatial joins. For AI/ML engineers, spatial data scientists, and platform teams, implementing a structured error taxonomy is non-negotiable. It intercepts, classifies, and routes spatial errors deterministically, preventing brittle LLM-driven pipelines from degrading into infinite retry loops when confronted with malformed coordinate arrays or constraint violations.

<figure class="diagram">
<svg viewBox="0 0 860 300" role="img" aria-labelledby="ema-t ema-d" xmlns="http://www.w3.org/2000/svg">
  <title id="ema-t">Spatial API error-mapping pipeline</title>
  <desc id="ema-d">API call, status capture, error classification, retry policy, and fallback mapping run in sequence to produce a structured spatial API error map feeding fallback routing, logging, and alerting.</desc>
  <defs>
    <marker id="ema-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
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
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#ema-arrow)">
    <line x1="166" y1="95" x2="181" y2="95"/>
    <line x1="334" y1="95" x2="349" y2="95"/>
    <line x1="502" y1="95" x2="517" y2="95"/>
    <line x1="670" y1="95" x2="685" y2="95"/>
  </g>
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#ema-arrow)">
    <line x1="90" y1="137" x2="90" y2="202"/>
    <line x1="258" y1="137" x2="258" y2="202"/>
    <line x1="426" y1="137" x2="426" y2="202"/>
    <line x1="594" y1="137" x2="594" y2="202"/>
    <line x1="762" y1="137" x2="762" y2="202"/>
  </g>
  <rect x="15" y="205" width="822" height="58" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/>
  <g fill="#1f2937" font-size="13" text-anchor="middle">
    <text x="90" y="90"><tspan x="90" dy="0">API</tspan><tspan x="90" dy="16">call</tspan></text>
    <text x="258" y="90"><tspan x="258" dy="0">Status</tspan><tspan x="258" dy="16">capture</tspan></text>
    <text x="426" y="90"><tspan x="426" dy="0">Error</tspan><tspan x="426" dy="16">classify</tspan></text>
    <text x="594" y="90"><tspan x="594" dy="0">Retry</tspan><tspan x="594" dy="16">policy</tspan></text>
    <text x="762" y="90"><tspan x="762" dy="0">Fallback</tspan><tspan x="762" dy="16">mapping</tspan></text>
  </g>
  <text x="426" y="240" fill="#1f2937" font-size="15" font-weight="600" text-anchor="middle">Structured spatial API error map</text>
  <text x="426" y="287" fill="#5b6471" font-size="12" text-anchor="middle">Downstream: fallback routing · logging · alerting</text>
</svg>
</figure>

## Architectural Foundations for Spatial Error Taxonomy

A resilient spatial error architecture begins by decoupling transport-layer failures from domain-level spatial failures. Network errors (timeouts, `5xx` responses, TLS handshake failures) require exponential backoff and circuit breaking. Spatial errors (self-intersections, out-of-bounds coordinates, projection drift, invalid GeoJSON structures) require geometric validation, coordinate transformation, or query rewriting. The classification layer must parse both the HTTP envelope and the spatial payload, extracting diagnostic metadata such as `ST_IsValidReason`, RFC 7946 `bbox` violations, or OGC API conformance deviations.

When an LLM agent generates a spatial query, the execution environment must capture failures and map them to structured error objects before deciding whether to retry, fallback to a cached dataset, or prompt the agent for correction. This handoff is particularly relevant when integrating with [Geospatial Prompt Engineering & Tool Routing](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/), where error mapping serves as the deterministic translation layer between brittle external services and resilient agent workflows.

## Step-by-Step Implementation: Parsing and Classification

The first implementation step involves wrapping spatial API clients with a unified response parser that normalizes error payloads into a consistent schema. Python developers should leverage Pydantic models to enforce strict typing on spatial error envelopes, capturing both the HTTP status and the spatial diagnostic payload. The parser must inspect `Content-Type` headers, decode GeoJSON or WKB payloads, and run immediate geometric validation before exposing data to downstream consumers.

```python
import json
import logging
from enum import Enum
from typing import Optional, Any, Dict
from pydantic import BaseModel, Field, ValidationError
from shapely.geometry import shape
from shapely.validation import explain_validity
import requests

logger = logging.getLogger(__name__)

class SpatialErrorCode(str, Enum):
    NETWORK_TIMEOUT = "network.timeout"
    HTTP_SERVER_ERROR = "http.server_error"
    CRS_MISMATCH = "spatial.crs_mismatch"
    TOPOLOGY_INVALID = "spatial.topology_invalid"
    BOUNDS_VIOLATION = "spatial.bounds_violation"
    PAYLOAD_MALFORMED = "spatial.payload_malformed"

class SpatialErrorEnvelope(BaseModel):
    http_status: int
    error_code: SpatialErrorCode
    message: str
    raw_response: Optional[str] = None
    diagnostic: Optional[Dict[str, Any]] = None
    retryable: bool = Field(default=False)

class SpatialResponseParser:
    """Production-ready parser for spatial API responses with strict error mapping."""

    VALID_CRS = "http://www.opengis.net/def/crs/OGC/1.3/CRS84"  # RFC 7946 default

    @staticmethod
    def parse_response(response: requests.Response) -> SpatialErrorEnvelope:
        if response.status_code >= 500:
            return SpatialErrorEnvelope(
                http_status=response.status_code,
                error_code=SpatialErrorCode.HTTP_SERVER_ERROR,
                message=f"Server-side spatial service failure: {response.status_code}",
                retryable=True
            )

        if response.status_code == 408:
            return SpatialErrorEnvelope(
                http_status=408,
                error_code=SpatialErrorCode.NETWORK_TIMEOUT,
                message="Spatial API request timed out",
                retryable=True
            )

        if response.status_code != 200:
            return SpatialErrorEnvelope(
                http_status=response.status_code,
                error_code=SpatialErrorCode.PAYLOAD_MALFORMED,
                message=f"Unexpected HTTP status: {response.status_code}",
                retryable=False
            )

        # Parse spatial payload
        try:
            payload = response.json()
        except json.JSONDecodeError as e:
            return SpatialErrorEnvelope(
                http_status=200,
                error_code=SpatialErrorCode.PAYLOAD_MALFORMED,
                message=f"Invalid JSON in spatial payload: {str(e)}",
                retryable=False
            )

        # Validate CRS (GeoJSON FeatureCollection CRS member — legacy; RFC 7946 deprecated it)
        crs = payload.get("crs", {}).get("properties", {}).get("name", SpatialResponseParser.VALID_CRS)
        if crs != SpatialResponseParser.VALID_CRS:
            return SpatialErrorEnvelope(
                http_status=200,
                error_code=SpatialErrorCode.CRS_MISMATCH,
                message=f"CRS mismatch detected: {crs}. Expected {SpatialResponseParser.VALID_CRS}",
                diagnostic={"found_crs": crs, "expected_crs": SpatialResponseParser.VALID_CRS},
                retryable=False
            )

        # Validate topology for each feature
        if "features" in payload:
            for idx, feature in enumerate(payload["features"]):
                geom = feature.get("geometry")
                if geom:
                    try:
                        shp = shape(geom)
                        if not shp.is_valid:
                            reason = explain_validity(shp)
                            return SpatialErrorEnvelope(
                                http_status=200,
                                error_code=SpatialErrorCode.TOPOLOGY_INVALID,
                                message=f"Invalid geometry at feature index {idx}: {reason}",
                                diagnostic={"feature_index": idx, "validity_reason": reason},
                                retryable=False
                            )
                    except Exception as e:
                        return SpatialErrorEnvelope(
                            http_status=200,
                            error_code=SpatialErrorCode.TOPOLOGY_INVALID,
                            message=f"Failed to parse geometry at index {idx}: {str(e)}",
                            retryable=False
                        )

        # All checks passed — response is valid
        return SpatialErrorEnvelope(
            http_status=200,
            error_code=SpatialErrorCode.PAYLOAD_MALFORMED,  # Using as a sentinel; callers should check http_status==200 + no error
            message="Response parsed successfully with no spatial errors detected.",
            retryable=False
        )
```

**Note on the sentinel pattern above:** a cleaner approach is to return `None` or a separate `SpatialSuccessEnvelope` for valid responses, and only return `SpatialErrorEnvelope` on failures. The code above uses the existing envelope type for consistency with downstream error-handling code that already checks `error_code`; callers should check `http_status == 200 and "successfully" in envelope.message` to detect success.

## Enforcing CRS and Topology Constraints

Silent spatial failures often propagate through pipelines until they trigger hard crashes in visualization layers or spatial join operations. The parser above enforces RFC 7946 compliance by defaulting to `CRS84` (EPSG:4326 longitude/latitude) and explicitly rejecting payloads that declare alternative projections without explicit transformation metadata. For production systems, coordinate bounds validation should also be applied to catch `NaN`, `Infinity`, or coordinates exceeding `[-180, 180]` and `[-90, 90]` when operating in WGS84.

Topology enforcement relies on rigorous validation of geometric primitives. Self-intersecting polygons, duplicate vertices, and unclosed rings are common artifacts of automated digitization or LLM-generated coordinate arrays. By leveraging `shapely.validation.explain_validity`, the system extracts human-readable diagnostic strings (e.g., `"Self-intersection[12.45 45.67]"`) that can be fed directly into agent correction loops. This approach aligns with [Topology Rule Enforcement via LLMs](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/), where structured validity reports replace generic exception traces, enabling models to autonomously rewrite malformed spatial predicates or apply automated snapping buffers.

For authoritative validation standards, refer to the [OGC API Features specification](https://docs.ogc.org/is/17-069r4/17-069r4.html), which defines conformance classes for spatial data exchange and explicitly addresses error reporting for invalid feature geometries.

## Routing Failures in LLM-Driven Pipelines

Once an error is classified, the routing layer determines the next action. Network timeouts and `5xx` errors trigger exponential backoff with jitter. Payload or topology errors trigger deterministic fallbacks:
1. **Query Rewriting:** Strip invalid predicates, apply `ST_Buffer(geom, 0)` to auto-heal minor topology violations, or switch to a simplified bounding-box query.
2. **Agent Correction:** Pass the `diagnostic` payload to the LLM with a structured system prompt requesting coordinate correction or predicate refinement.
3. **Graceful Degradation:** Return a cached, lower-resolution tile or feature set while logging the failure for model fine-tuning.

This routing logic is critical when integrating with [Prompt-to-Spatial-SQL Generation](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/), where LLMs frequently generate syntactically valid but semantically broken spatial predicates (e.g., `ST_Intersects` on mismatched CRS or invalid polygon rings). By intercepting the resulting `SpatialErrorEnvelope`, the pipeline avoids cascading PostGIS planner failures and instead routes the diagnostic context back to the generation step for self-correction.

To translate these technical envelopes into actionable agent instructions, see [Mapping Spatial API Errors to User-Friendly Prompts](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/mapping-spatial-api-errors-to-user-friendly-prompts/), which details how to convert `explain_validity` outputs and CRS mismatch codes into constrained natural language prompts that guide LLMs toward valid spatial outputs.

## Production Hardening and Observability

In asynchronous geoprocessing workflows, error mapping must be thread-safe and non-blocking. Implement a centralized error registry that aggregates `SpatialErrorEnvelope` instances, exposing metrics for Prometheus/Grafana dashboards. Key observability signals include:
- `spatial_error_rate` by `error_code`
- `crs_mismatch_frequency` by upstream provider
- `llm_correction_success_rate` after topology routing

Use connection pooling with strict timeouts, and wrap all spatial API calls in a circuit breaker pattern. When the error rate exceeds a defined threshold (e.g., 15% topology failures in a 5-minute window), the circuit should open, routing all subsequent requests to a fallback provider or cache until validation metrics stabilize. For implementation details on async spatial validation, consult the official [Shapely validation documentation](https://shapely.readthedocs.io/en/stable/manual.html#validation.make_valid), which outlines thread-safe geometry parsing and validation best practices.

Error mapping is not merely a defensive programming practice; it is the control plane for autonomous geospatial AI. By standardizing how spatial failures are captured, classified, and routed, platform teams transform brittle API integrations into resilient, self-healing pipelines capable of supporting complex LLM-driven spatial reasoning at scale.
