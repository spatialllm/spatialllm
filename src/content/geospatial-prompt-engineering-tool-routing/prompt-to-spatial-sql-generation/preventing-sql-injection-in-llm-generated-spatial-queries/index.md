# Preventing SQL Injection in LLM-Generated Spatial Queries

The moment an LLM's output is concatenated into a SQL string, a prompt-injected geometry literal or a crafted place name becomes an executable `DROP TABLE`. Spatial queries make this worse: WKT strings, function names, and SRIDs all look like content the model should produce, so the injection surface is unusually wide. This guide replaces string-building with a parameterized, allowlisted query builder running under a read-only role, as part of [prompt-to-spatial-SQL generation](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/).

The rule is absolute: model output never becomes SQL syntax. It supplies *values* — bound as parameters — and *choices* from a fixed set the server controls. The model may pick `ST_Intersects` from an allowlist and provide a WKT string as a bound parameter; it may never emit the function call or the geometry as raw text spliced into a statement.

## When to Use This Approach

Use the safe builder for every query whose shape or values originate from a model, an end user, or any untrusted upstream. The only queries exempt are fully static statements with no interpolated input — and those are rare once an agent is involved.

| Pattern | Injection risk | Notes |
|---|---|---|
| f-string / concat of model text | Critical | Never do this, even "just for the geometry" |
| Parameter binding only | Low | Values safe; but function/column names can't be bound |
| Binding + allowlist + read-only role (this page) | Minimal | Values bound, identifiers whitelisted, blast radius capped |

Parameter binding alone is necessary but not sufficient, because you cannot bind an identifier — a table, column, or function name must be validated against an allowlist instead. Layering a read-only database role underneath caps the damage even if a gap slips through. This builder is the enforcement layer beneath [generating valid PostGIS queries from natural language](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/generating-valid-postgis-queries-from-natural-language/), and it pairs with the typed contracts in [spatial function-calling schemas](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/).

## Implementation

The builder below accepts a structured spatial request (predicate, layer, geometry WKT, limit), validates every identifier against an allowlist, binds all values as parameters, and emits a query that always leads with a `&&` bbox pre-filter before the exact `ST_*` predicate. Anything outside the allowlist is rejected with a deterministic empty result.

```python
import logging
import re
from dataclasses import dataclass

log = logging.getLogger("safe_spatial_sql")

# Server-controlled allowlists. The model may only *choose* from these.
ALLOWED_PREDICATES = {
    "intersects": "ST_Intersects",
    "within": "ST_Within",
    "dwithin": "ST_DWithin",
}
ALLOWED_LAYERS = {
    "parcels": ("parcels", "geom", 3857),
    "buildings": ("buildings", "geom", 3857),
    "flood_zones": ("flood_zones", "geom", 3857),
}
_WKT_RE = re.compile(
    r"^(POINT|LINESTRING|POLYGON|MULTIPOLYGON|MULTILINESTRING|MULTIPOINT)\s*[ZM]?\s*\(",
    re.IGNORECASE,
)


@dataclass
class SpatialRequest:
    predicate: str
    layer: str
    wkt: str
    limit: int = 100
    distance_m: float | None = None


class UnsafeRequest(Exception):
    pass


def _reject_reason(req: SpatialRequest) -> str | None:
    if req.predicate not in ALLOWED_PREDICATES:
        return f"predicate '{req.predicate}' not allowlisted"
    if req.layer not in ALLOWED_LAYERS:
        return f"layer '{req.layer}' not allowlisted"
    if not isinstance(req.wkt, str) or not _WKT_RE.match(req.wkt.strip()):
        return "wkt does not match an allowed geometry type"
    if not (1 <= int(req.limit) <= 1000):
        return "limit out of bounds"
    if req.predicate == "dwithin" and not (
        isinstance(req.distance_m, (int, float)) and 0 < req.distance_m <= 50000
    ):
        return "dwithin requires a bounded distance_m"
    return None


def build_query(req: SpatialRequest):
    """Return (sql, params) or raise UnsafeRequest. No model text ever enters `sql`."""
    reason = _reject_reason(req)
    if reason:
        raise UnsafeRequest(reason)

    fn = ALLOWED_PREDICATES[req.predicate]           # from allowlist, not user text
    table, geom_col, srid = ALLOWED_LAYERS[req.layer]  # identifiers are trusted constants

    if req.predicate == "dwithin":
        # Bbox pre-filter (&&) on an expanded envelope, then the exact ST_DWithin.
        sql = (
            f"SELECT id FROM {table} "
            f"WHERE {geom_col} && ST_Expand(ST_GeomFromText($1, $2), $3) "
            f"  AND ST_DWithin({geom_col}, ST_GeomFromText($1, $2), $3) "
            f"LIMIT $4"
        )
        params = [req.wkt, srid, float(req.distance_m), int(req.limit)]
    else:
        # && bbox pre-filter uses the GiST index before the exact predicate.
        sql = (
            f"SELECT id FROM {table} "
            f"WHERE {geom_col} && ST_GeomFromText($1, $2) "
            f"  AND {fn}({geom_col}, ST_GeomFromText($1, $2)) "
            f"LIMIT $3"
        )
        params = [req.wkt, srid, int(req.limit)]
    return sql, params


async def run_request(pool, req: SpatialRequest):
    try:
        sql, params = build_query(req)
    except UnsafeRequest as exc:
        log.warning("rejected unsafe spatial request: %s", exc)
        return []  # deterministic fallback: no rows, no execution
    try:
        # Connection MUST use a role granted only SELECT on the allowed layers.
        async with pool.acquire() as conn:
            return await conn.fetch(sql, *params)
    except Exception:
        log.exception("query execution failed")
        return []
```

Three defenses stack here. Binding places the WKT and SRID as parameters, so no geometry text is ever parsed as SQL. The allowlist maps model-chosen keys to trusted table, column, and function constants, so identifiers cannot be injected. And the pool must connect as a role with only `SELECT` on the named layers, so even a hypothetical bypass cannot write or read outside scope. The WKT regex is a shape gate, not the security boundary — the binding is.

## Validation & Testing

- **No raw model text reaches SQL.** Assert `build_query` raises `UnsafeRequest` for `predicate="intersects; DROP TABLE parcels"` and that the returned `sql` for a valid request contains no substring of `req.wkt`.
- **Identifier allowlist holds.** Assert `layer="parcels; --"` and `predicate="ST_Union"` both reject, and that only the three known layers and predicates ever produce SQL.
- **Bbox pre-filter is present.** Assert every generated statement contains `&&` occurring before the `ST_Intersects`/`ST_Within`/`ST_DWithin` call, so the index path is never skipped.

## Gotchas & Edge Cases

- **Bindable values vs identifiers.** Newcomers try to bind a table or function name and hit a syntax error, then "fix" it by concatenating — reopening the hole. Identifiers must come from the allowlist; only values are ever bound.
- **`ST_GeomFromText` on hostile WKT.** A malformed or enormous WKT can still error or burn CPU inside the parser. Keep the regex shape check, cap WKT length before binding, and let the execution `try/except` shed the rest.
- **Read-only role that is not actually read-only.** A role with `SELECT` but also default `CREATE`/`TEMP` privileges on the schema is not safe. Revoke everything but `SELECT` on the specific layers and verify with an integration test that an `INSERT` from that role fails.
- **Second-order injection via stored values.** WKT persisted now and interpolated into a query later re-creates the risk. Bind on read too; never trust "our own" stored strings as syntax.

## Related

- Up to the section: [Prompt-to-Spatial-SQL Generation](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/)
- [Generating Valid PostGIS Queries from Natural Language](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/generating-valid-postgis-queries-from-natural-language/)
- [Spatial Function-Calling Schemas](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/)
- [Designing JSON Schemas for PostGIS Tool Calls](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/designing-json-schemas-for-postgis-tool-calls/)
