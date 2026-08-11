---
title: Designing JSON Schemas for PostGIS Tool Calls
description: Write tool schemas that make a wrong spatial call hard to express — typed geometry, units in names, closed enumerations and bounded numbers.
slug: designing-json-schemas-for-postgis-tool-calls
type: howto
breadcrumb: Designing Tool Schemas
datePublished: 2025-04-02
dateModified: 2026-08-11
---

# Designing JSON Schemas for PostGIS Tool Calls

Designing JSON schemas for PostGIS tool calls means writing the typed contract that stands between a language model's intent and a live spatial database, so that operations like `ST_DWithin` and `ST_Intersects` receive only validated, bounded parameters and always compile to an index-aware query. This guide sits under [Spatial Function-Calling Schemas](/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/) and covers the execution stage where a validated tool call is rendered into parameterized SQL that a query planner can actually accelerate.

The stakes are concrete. A schema that maps loosely to PostGIS produces queries that either scan the whole table because they skipped the `&&` bounding-box pre-filter, or crash because a geometry column was addressed with the wrong SRID. A schema that maps tightly produces a parameterized, GiST-index-friendly query every time, with an unsafe or unbounded call rejected before it ever reaches the connection.

## When to Use This Approach

Use a per-operation JSON schema plus a server-side builder when the model must trigger a fixed vocabulary of spatial predicates against known tables. It is the right tool when correctness and index usage matter more than expressive breadth. Reach for free SQL generation only when the query shape is genuinely open-ended.

| Situation | Schema-bound tool call | Free SQL generation |
| --- | --- | --- |
| Fixed set of predicates (`ST_DWithin`, `ST_Intersects`) | Preferred — typed, bounded | Overkill, unsafe |
| Guaranteed `&&` index pre-filter | Enforced by builder | Depends on model discipline |
| Arbitrary ad-hoc analytics | Too rigid | Preferred, with guards |
| Auditable, replayable calls | Native | Requires SQL parsing |

For the open-ended path and its injection defenses, see [Generating Valid PostGIS Queries from Natural Language](/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/generating-valid-postgis-queries-from-natural-language/).

<figure class="diagram">
<svg viewBox="16 38 748 188" role="img" aria-labelledby="djs-two-t djs-two-d" xmlns="http://www.w3.org/2000/svg"><title id="djs-two-t">A constrained schema against a permissive one</title><desc id="djs-two-d">Enumerated values, bounded numbers and required fields turn a whole class of wrong call into an impossible one; a schema of loose strings turns it into a runtime error.</desc><rect x="16" y="38" width="748" height="188" fill="#ffffff"/><rect x="30" y="52" width="340" height="160" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="410" y="52" width="340" height="160" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="200" y="84">constrained schema</text><text x="580" y="84">permissive schema</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="200" y="114">enumerated where finite</text><text x="200" y="140">bounded where numeric</text><text x="200" y="166">wrong calls cannot form</text><text x="580" y="114">free strings everywhere</text><text x="580" y="140">unbounded numbers</text><text x="580" y="166">wrong calls fail at runtime</text></g></svg>
<figcaption><b>The schema is the cheapest validator you will ever write.</b> Every constraint expressed in it is a check that never needs code, a test, or a message.</figcaption>
</figure>

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

The `&& ST_Expand(...)` clause is the load-bearing detail: it gives the planner a bounding-box join condition that the GiST index can satisfy before the exact `ST_DWithin` distance test runs on the surviving candidates. The `::geography` cast makes `distance_meters` a true metric distance regardless of the stored projection. Because layer names come from `REGISTERED_LAYERS` and reach SQL only through `sql.Identifier`, the model can never inject a table name or column. This composes cleanly into a larger plan; see [Multi-Step Spatial Agent Orchestration](/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/) for chaining several such calls.

<figure class="diagram">
<svg viewBox="16 38 728 212" role="img" aria-labelledby="djs-name-t djs-name-d" xmlns="http://www.w3.org/2000/svg"><title id="djs-name-t">Names carry meaning the description cannot</title><desc id="djs-name-d">A parameter called distance_metres cannot be filled with degrees by accident in the way that one called distance can, and the name is read at generation time where the description often is not.</desc><rect x="16" y="38" width="728" height="212" fill="#ffffff"/><rect x="30" y="52" width="700" height="46" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="50" y="80" fill="#1f2937" font-size="12.5">distance_metres: the unit is in the name</text><rect x="30" y="108" width="600" height="46" rx="6" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="50" y="136" fill="#1f2937" font-size="12.5">distance, unit described in prose: read sometimes</text><rect x="30" y="164" width="380" height="46" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="50" y="192" fill="#1f2937" font-size="12.5">d: correct only by luck</text><text x="390" y="238" fill="#1f2937" font-size="13" text-anchor="middle">The parameter name is the part of the schema the model attends to most reliably</text></svg>
<figcaption><b>Naming is the highest-leverage thing in the schema.</b> It costs nothing, it is read every time, and it removes ambiguity that no amount of description reliably fixes.</figcaption>
</figure>

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

<figure class="diagram">
<svg viewBox="16 32 748 214" role="img" aria-labelledby="djs-err-t djs-err-d" xmlns="http://www.w3.org/2000/svg"><title id="djs-err-t">What a validation failure should return</title><desc id="djs-err-d">A rejection that names the field, the constraint and an acceptable value lets the next attempt succeed; one that says the arguments were invalid produces the same call again.</desc><rect x="16" y="32" width="748" height="214" fill="#ffffff"/><rect x="30" y="46" width="360" height="86" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="52" y="76" fill="#1f2937" font-size="13" font-weight="600">the field</text><text x="52" y="102" fill="#5b6471" font-size="12">which one is wrong</text><text x="52" y="122" fill="#5b6471" font-size="12">not the whole object</text><rect x="410" y="46" width="340" height="86" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="432" y="76" fill="#1f2937" font-size="13" font-weight="600">the constraint</text><text x="432" y="102" fill="#5b6471" font-size="12">what it violated</text><text x="432" y="122" fill="#5b6471" font-size="12">in the schema's terms</text><rect x="30" y="146" width="360" height="86" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="52" y="176" fill="#1f2937" font-size="13" font-weight="600">an acceptable value</text><text x="52" y="202" fill="#5b6471" font-size="12">one that would work</text><text x="52" y="222" fill="#5b6471" font-size="12">so the retry is informed</text><rect x="410" y="146" width="340" height="86" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="432" y="176" fill="#1f2937" font-size="13" font-weight="600">invalid arguments</text><text x="432" y="202" fill="#5b6471" font-size="12">names nothing</text><text x="432" y="222" fill="#5b6471" font-size="12">the same call comes back</text></svg>
<figcaption><b>Three of these make the retry succeed.</b> The fourth guarantees a second identical attempt, which is where most wasted tool-calling budget goes.</figcaption>
</figure>

## Operating This Step Over Time

Schemas accumulate optional parameters. Each is added for one caller and then applies forever, and a tool with fourteen optional fields is one the model fills badly because it cannot tell which matter. Counting how often each parameter is actually supplied turns that into evidence — anything below a few percent is a candidate for removal or for a separate tool.

Descriptions drift out of step with behaviour more readily than names do, because behaviour changes in code and descriptions change in a schema file nobody is touching. Any change to a default or a bound should be a change to the description in the same commit, and reviewing that is worth a checklist item.

Watch the validation failure rate per tool. A tool whose calls fail schema validation regularly is not being misused; it is badly specified, and the fix is almost always a tighter type or a clearer name rather than a longer description.

## Frequently Asked Questions

<details class="faq-item"><summary><span>How specific should descriptions be?</span></summary><p>One sentence on what the parameter means, one on the unit or format, and nothing about implementation. Descriptions that explain how the tool works internally consume attention without changing the call, and descriptions that omit units produce calls that are confidently in the wrong ones. Anything longer than two sentences is usually a sign the parameter should be split or enumerated.</p></details>

<details class="faq-item"><summary><span>Should defaults live in the schema or the code?</span></summary><p>In the schema where the model benefits from seeing them, and applied in code so the behaviour is identical when the field is omitted. Stating a default the code does not apply is worse than stating none, so the two have to be generated from one source or reviewed together. A default that is visible and honoured also reduces how often the field is supplied at all.</p></details>

<details class="faq-item"><summary><span>What about parameters that depend on each other?</span></summary><p>Express what the schema can — required-if relations where the format supports them — and check the rest in code with a message in the same shape as a schema rejection. The model does not need to know whether a rule came from the schema or from a subsequent check; it needs to know which field to change. Keeping the two failure shapes identical is what makes that work.</p></details>

<details class="faq-item"><summary><span>How many tools is too many?</span></summary><p>Enough that the model starts choosing wrongly between similar ones, which in practice arrives somewhere around a dozen for closely related operations. The fix is rarely fewer capabilities; it is fewer tools with more parameters, or a routing layer that presents a subset relevant to the current request. Two tools whose descriptions differ only in a clause will be confused indefinitely.</p></details>

## Related

- Up to the section overview: [Spatial Function-Calling Schemas](/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/)
- [Generating Valid PostGIS Queries from Natural Language](/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/generating-valid-postgis-queries-from-natural-language/)
- [Multi-Step Spatial Agent Orchestration](/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/)
