# Designing JSON Schemas for PostGIS Tool Calls

Designing JSON schemas for PostGIS tool calls means writing the typed contract that stands between a language model's intent and a live spatial database, so that operations like `ST_DWithin` and `ST_Intersects` receive only validated, bounded parameters and always compile to an index-aware query. This guide sits under [Spatial Function-Calling Schemas](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/) and covers the execution stage where a validated tool call is rendered into parameterized SQL that a query planner can actually accelerate.

The stakes are concrete. A schema that maps loosely to PostGIS produces queries that either scan the whole table because they skipped the `&&` bounding-box pre-filter, or crash because a geometry column was addressed with the wrong SRID. A schema that maps tightly produces a parameterized, GiST-index-friendly query every time, with an unsafe or unbounded call rejected before it ever reaches the connection.

## When to Use This Approach

Use a per-operation JSON schema plus a server-side builder when the model must trigger a fixed vocabulary of spatial predicates against known tables. It is the right tool when correctness and index usage matter more than expressive breadth. Reach for free SQL generation only when the query shape is genuinely open-ended.

| Situation | Schema-bound tool call | Free SQL generation |
| --- | --- | --- |
| Fixed set of predicates (`ST_DWithin`, `ST_Intersects`) | Preferred — typed, bounded | Overkill, unsafe |
| Guaranteed `&&` index pre-filter | Enforced by builder | Depends on model discipline |
| Arbitrary ad-hoc analytics | Too rigid | Preferred, with guards |
| Auditable, replayable calls | Native | Requires SQL parsing |

For the open-ended path and its injection defenses, see [Generating Valid PostGIS Queries from Natural Language](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/generating-valid-postgis-queries-from-natural-language/).

## Implementation

The schema below models an `ST_DWithin` proximity join. Every field is constrained: the table and geometry column are drawn from an allowlist, the SRID is an enum, and the distance is bounded and unit-named. The builder then renders a parameterized query that *always* emits the `&&` bounding-box pre-filter before the `ST_DWithin` predicate, guaranteeing the GiST index is consulted. If any invariant fails, the call is rejected and a deterministic fallback query (empty result, logged) is returned instead of an unsafe execution.

```python
import logging
from enum import IntEnum
from typing import Dict, Any, Tuple
from pydantic import BaseModel, Field, field_validator, ValidationError
from psycopg2 import sql

logger = logging.getLogger("postgis_toolcall")

# Allowlist derived from the live catalog at startup; never trust model-supplied names.
REGISTERED_LAYERS: Dict[str, str] = {
    "parcels": "geom",
    "hydrants": "geom",
    "flood_zones": "geom",
}

class LayerSRID(IntEnum):
    WGS84 = 4326
    UTM33N = 32633

class DWithinToolCall(BaseModel):
    left_layer: str
    right_layer: str
    srid: LayerSRID
    distance_meters: float = Field(ge=0.0, le=25_000.0)
    max_rows: int = Field(ge=1, le=1000)

    @field_validator("left_layer", "right_layer")
    @classmethod
    def known_layer(cls, v: str) -> str:
        if v not in REGISTERED_LAYERS:
            raise ValueError(f"unknown layer: {v}")
        return v

    @field_validator("distance_meters")
    @classmethod
    def finite(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("distance_meters must be finite")
        return v

def build_dwithin_query(call: DWithinToolCall) -> sql.Composed:
    """Render a parameterized, index-aware ST_DWithin join.

    The && bbox pre-filter precedes ST_DWithin so the GiST index is used;
    geometries are cast to a metric CRS for a true metre distance.
    """
    lgeom = REGISTERED_LAYERS[call.left_layer]
    rgeom = REGISTERED_LAYERS[call.right_layer]
    return sql.SQL(
        """
        SELECT a.id AS left_id, b.id AS right_id
        FROM {lt} a
        JOIN {rt} b
          ON a.{lg} && ST_Expand(b.{rg}, %(dist)s)
         AND ST_DWithin(
               a.{lg}::geography, b.{rg}::geography, %(dist)s
             )
        WHERE ST_SRID(a.{lg}) = %(srid)s
          AND ST_SRID(b.{rg}) = %(srid)s
        LIMIT %(lim)s
        """
    ).format(
        lt=sql.Identifier(call.left_layer),
        rt=sql.Identifier(call.right_layer),
        lg=sql.Identifier(lgeom),
        rg=sql.Identifier(rgeom),
    )

def run_dwithin_toolcall(raw_args: Dict[str, Any], conn) -> Dict[str, Any]:
    try:
        call = DWithinToolCall.model_validate(raw_args)
    except ValidationError as exc:
        logger.warning("rejected ST_DWithin call: %s", exc.errors())
        return {"status": "rejected", "reason": "SCHEMA_REJECT", "rows": []}

    query = build_dwithin_query(call)
    params = {
        "dist": call.distance_meters,
        "srid": int(call.srid),
        "lim": call.max_rows,
    }
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        return {"status": "ok", "rows": rows}
    except Exception as exc:
        conn.rollback()
        logger.error("ST_DWithin execution failed: %s", exc)
        # Deterministic fallback: empty, logged, never a partial write.
        return {"status": "fallback", "reason": "EXEC_ERROR", "rows": []}
```

The `&& ST_Expand(...)` clause is the load-bearing detail: it gives the planner a bounding-box join condition that the GiST index can satisfy before the exact `ST_DWithin` distance test runs on the surviving candidates. The `::geography` cast makes `distance_meters` a true metric distance regardless of the stored projection. Because layer names come from `REGISTERED_LAYERS` and reach SQL only through `sql.Identifier`, the model can never inject a table name or column. This composes cleanly into a larger plan; see [Multi-Step Spatial Agent Orchestration](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/) for chaining several such calls.

## Validation & Testing

- **Reject unsafe calls.** Assert that `run_dwithin_toolcall({"left_layer": "'; DROP TABLE parcels; --", ...})` returns `status == "rejected"` and never builds a query, proving the allowlist blocks injection at the schema boundary.
- **Prove the index pre-filter is present.** Render `build_dwithin_query` and assert the generated SQL string contains both `&&` and `ST_DWithin`, with the `&&` occurring first — a regression guard against silently dropping the bounding-box pre-filter.
- **Bounds enforcement.** Assert `distance_meters=1e12` and `distance_meters=float("inf")` both raise `ValidationError`, confirming the query planner is never handed an unbounded radius.
- **Plan check in CI.** Against a seeded fixture database, run `EXPLAIN` on the built query and assert the plan references an `Index Scan` (or `Index Cond`) on the geometry column rather than a `Seq Scan`.

## Gotchas & Edge Cases

**Casting to geography changes what the index matches.** `ST_DWithin(a::geography, b::geography, d)` uses spherical distance, but the `&&` operator compares planar bounding boxes in the stored SRID. Keep the `&&` term on the *geometry* columns (via `ST_Expand`) so the index still applies; only the exact predicate uses geography.

**SRID enum wider than the data.** If the schema allows SRID 4326 but a row is stored as 32633, the `ST_SRID` guard filters it out silently and results look empty. Derive `LayerSRID` from the actual `ST_SRID` values in each layer at startup, not from a hand-typed list.

**`ST_Expand` in degrees vs metres.** When the geometry column is stored in a geographic CRS (degrees), `ST_Expand(geom, 25000)` expands by 25000 *degrees*, which is meaningless. Either store the pre-filter geometry in a projected CRS or expand using a degree-equivalent computed from latitude before building the query.

**Unbounded `LIMIT`.** Omitting `max_rows` lets a broad join stream millions of pairs. Keep `max_rows` required and bounded so a permissive proximity call cannot become an accidental cross join.

## Related

- Up to the section overview: [Spatial Function-Calling Schemas](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/)
- [Generating Valid PostGIS Queries from Natural Language](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/generating-valid-postgis-queries-from-natural-language/)
- [Multi-Step Spatial Agent Orchestration](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/)
