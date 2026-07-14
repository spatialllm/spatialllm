# Spatial Function-Calling Schemas

Spatial function-calling schemas are the typed contracts that let a language model invoke geoprocessing operations without ever hand-writing a geometry, a coordinate, or a raw SQL string. Situated inside the [Geospatial Prompt Engineering & Tool Routing](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/) discipline, this guide addresses one specific failure mode: an unconstrained model emitting a "spatial" argument that is syntactically plausible but semantically catastrophic. It teaches you to design schemas that make the model physically unable to emit an invalid geometry type, an unknown SRID, or an unbounded distance, so every accepted tool call is already inside the safe envelope before a single database row is touched.

The core insight is that the schema, not the prompt, is where correctness lives. A well-designed function definition constrains the model's output space with enums, numeric bounds, and pattern constraints; a matching server-side dispatcher then re-validates every argument and rejects anything outside the contract with a deterministic fallback. The prompt persuades; the schema enforces. When those two layers agree, tool calls become auditable, replayable, and safe to route straight into a geoprocessing backend.

<figure class="diagram">
<svg viewBox="0 0 860 300" role="img" aria-labelledby="sfcs-t sfcs-d" xmlns="http://www.w3.org/2000/svg">
  <title id="sfcs-t">Spatial function-calling schema flow</title>
  <desc id="sfcs-d">Schema definition, model tool call, argument decode, bounds and SRID validation, and safe dispatch run in sequence so only typed, bounded spatial arguments reach execution.</desc>
  <defs>
    <marker id="sfcs-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
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
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#sfcs-arrow)">
    <line x1="166" y1="95" x2="181" y2="95"/>
    <line x1="334" y1="95" x2="349" y2="95"/>
    <line x1="502" y1="95" x2="517" y2="95"/>
    <line x1="670" y1="95" x2="685" y2="95"/>
  </g>
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#sfcs-arrow)">
    <line x1="90" y1="137" x2="90" y2="202"/>
    <line x1="258" y1="137" x2="258" y2="202"/>
    <line x1="426" y1="137" x2="426" y2="202"/>
    <line x1="594" y1="137" x2="594" y2="202"/>
    <line x1="762" y1="137" x2="762" y2="202"/>
  </g>
  <rect x="15" y="205" width="822" height="58" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/>
  <g fill="#1f2937" font-size="13" text-anchor="middle">
    <text x="90" y="90"><tspan x="90" dy="0">Schema</tspan><tspan x="90" dy="16">definition</tspan></text>
    <text x="258" y="90"><tspan x="258" dy="0">Model</tspan><tspan x="258" dy="16">tool call</tspan></text>
    <text x="426" y="90"><tspan x="426" dy="0">Argument</tspan><tspan x="426" dy="16">decode</tspan></text>
    <text x="594" y="90"><tspan x="594" dy="0">Bounds &amp;</tspan><tspan x="594" dy="16">SRID check</tspan></text>
    <text x="762" y="90"><tspan x="762" dy="0">Safe</tspan><tspan x="762" dy="16">dispatch</tspan></text>
  </g>
  <text x="426" y="240" fill="#1f2937" font-size="15" font-weight="600" text-anchor="middle">Typed, bounded spatial arguments only</text>
  <text x="426" y="287" fill="#5b6471" font-size="12" text-anchor="middle">Reject path: deterministic fallback · structured error</text>
</svg>
</figure>

## Foundational Principles

Three constraints are non-negotiable when designing a spatial tool schema.

**Enumerate the SRID; never accept a free integer.** Coordinate reference systems are not interchangeable, and a model that emits `SRID 4327` (a typo for 4326) will silently corrupt every downstream distance. Restrict the SRID field to an enum of the CRS codes your data actually stores, and reject anything else at decode time rather than reprojecting on a guess.

**Bound every scalar with units in the field name.** A `distance` argument means nothing; a `distance_meters` argument bounded to `[0, 50000]` means everything. Metric operations must declare their unit in the schema and carry explicit minimum and maximum bounds, so the model cannot request a 40,000 km buffer that would materialize a planet-sized polygon.

**Constrain geometry to a discriminated type, not a string.** Accepting free WKT invites malformed rings, self-intersections, and injection. Instead, define geometry as a tagged object — a `type` enum plus a bounded coordinate array — so the schema itself rejects a `MultiPolygon` where a `Point` was required.

## Step-by-Step Implementation Pipeline

### 1. Define the tool as a bounded, self-validating schema

Start from the function the model may call and encode every constraint as a validator. The schema below defines a proximity-search tool whose SRID is an enum, whose radius is bounded and unit-named, and whose search point is a discriminated geometry with coordinate-range checks. This is the object the model receives as its tool definition and the object your server re-validates.

```python
from enum import IntEnum
from typing import Literal, Tuple
from pydantic import BaseModel, Field, field_validator, ValidationError

class AllowedSRID(IntEnum):
    WGS84 = 4326
    WEB_MERCATOR = 3857
    UTM33N = 32633

class PointArg(BaseModel):
    type: Literal["Point"]
    # [lon, lat] in WGS84 degrees; hard-bounded to the valid globe
    coordinates: Tuple[float, float]

    @field_validator("coordinates")
    @classmethod
    def within_globe(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        lon, lat = v
        if not (-180.0 <= lon <= 180.0) or not (-90.0 <= lat <= 90.0):
            raise ValueError(f"coordinate out of range: {v}")
        return v

class ProximitySearchCall(BaseModel):
    center: PointArg
    radius_meters: float = Field(ge=0.0, le=50_000.0)
    srid: AllowedSRID
    limit: int = Field(ge=1, le=500)

    @field_validator("radius_meters")
    @classmethod
    def finite_radius(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("radius_meters must be finite")
        return v
```

Because `AllowedSRID` is an enum and `radius_meters` carries bounds, a model that hallucinates SRID 9999 or an infinite radius produces a `ValidationError` at parse time — no reprojection guess, no unbounded query. When the model must instead assemble a full query, hand off to [Prompt-to-Spatial-SQL Generation](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/) rather than widening this schema.

### 2. Dispatch only after deterministic re-validation

The model's tool call arrives as raw JSON. The dispatcher must decode it against the schema, and on any failure return a structured rejection with a deterministic fallback — never a partial execution. The dispatcher owns the reject decision; the executor never sees an unvalidated argument.

```python
import logging
from typing import Any, Callable, Dict

logger = logging.getLogger("spatial_dispatch")

class RejectedCall(Exception):
    def __init__(self, code: str, detail: str):
        self.code, self.detail = code, detail
        super().__init__(f"{code}: {detail}")

def dispatch_spatial_tool(
    raw_args: Dict[str, Any],
    executor: Callable[[ProximitySearchCall], Dict[str, Any]],
    fallback: Callable[[str], Dict[str, Any]],
) -> Dict[str, Any]:
    try:
        call = ProximitySearchCall.model_validate(raw_args)
    except ValidationError as exc:
        # Deterministic reject: the model must repair and retry, or we fall back.
        logger.warning("rejected tool call: %s", exc.errors())
        return fallback("SCHEMA_REJECT")
    try:
        return executor(call)
    except Exception as exc:  # backend/runtime failure, not a schema failure
        logger.error("executor failed for validated call: %s", exc)
        return fallback("EXECUTOR_ERROR")
```

The two `except` blocks separate two distinct failure classes: a schema reject means the *arguments* were unsafe, while an executor error means a *validated* call still failed at runtime. Both converge on a deterministic fallback, but only the first should trigger a model self-correction loop. Mapping those runtime errors back into prompts the model can act on is covered in [Error Mapping for Spatial API Calls](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/).

### 3. Pin the schema to the backend it drives

A schema is only as safe as its match to the executor. The `srid` enum must be a subset of the SRIDs actually registered in your `spatial_ref_sys` table, and the geometry type must match the column the query targets. The concrete mapping from these typed fields to index-aware PostGIS predicates — including the mandatory `&&` bounding-box pre-filter — is worked end to end in [Designing JSON Schemas for PostGIS Tool Calls](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/designing-json-schemas-for-postgis-tool-calls/).

## Failure Modes & Root Causes

**Silent SRID coercion.** The most damaging failure is a permissive schema that accepts any integer SRID and reprojects on assumption. The root cause is treating the SRID as data rather than as a closed enum. Fix: reject unknown SRIDs; never guess.

**Unbounded scalar amplification.** A model asked to "find everything nearby" may emit a radius of `1e9`. Without an upper bound, the executor attempts a full-table scan disguised as a proximity query. The root cause is an unbounded numeric field; the fix is a hard `le` ceiling plus a finiteness check for `NaN`/`inf`.

**Geometry-type confusion.** When geometry is a plain string, the model can supply a `LineString` where a `Polygon` was required, and the error surfaces deep inside the backend. The root cause is an untyped geometry field; the fix is a discriminated union keyed on `type`.

**Schema drift from executor.** The schema permits SRID 3857 but the target column is stored in 4326, so validation passes and the query returns wrong results. The root cause is desynchronized contracts; the fix is deriving the enum from the live database catalog at startup.

## Production Validation Protocols

1. Assert every SRID field is an enum whose members are all present in `spatial_ref_sys` — fail startup otherwise.
2. Assert every metric scalar declares its unit in the field name and carries both `ge` and `le` bounds; a CI lint rejects unbounded floats in any tool schema.
3. Fuzz the dispatcher with malformed argument dictionaries (wrong types, out-of-range coordinates, `inf` radii) and assert every case returns the deterministic fallback and never reaches the executor.
4. Assert the executor is never invoked with a non-`ProximitySearchCall` object — enforce via a type guard and a unit test that patches the executor to record its argument type.
5. Log every rejected call with its structured error codes and track the schema-reject rate as an observability KPI; a rising rate signals prompt or schema drift.

## Related

- Up to the area overview: [Geospatial Prompt Engineering & Tool Routing](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/)
- [Designing JSON Schemas for PostGIS Tool Calls](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/designing-json-schemas-for-postgis-tool-calls/)
- [Prompt-to-Spatial-SQL Generation](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/)
- [Error Mapping for Spatial API Calls](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/)
- [Multi-Step Spatial Agent Orchestration](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/)
