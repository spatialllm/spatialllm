---
title: Preventing SQL Injection in LLM-Generated Spatial Queries
description: Remove injection structurally by never letting generated text become SQL — specifications, bound parameters, allow-lists and a read-only role.
slug: preventing-sql-injection-in-llm-generated-spatial-queries
type: howto
breadcrumb: Preventing SQL Injection
datePublished: 2025-04-17
dateModified: 2026-08-11
---

# Preventing SQL Injection in LLM-Generated Spatial Queries

The moment an LLM's output is concatenated into a SQL string, a prompt-injected geometry literal or a crafted place name becomes an executable `DROP TABLE`. Spatial queries make this worse: WKT strings, function names, and SRIDs all look like content the model should produce, so the injection surface is unusually wide. This guide replaces string-building with a parameterized, allowlisted query builder running under a read-only role, as part of [prompt-to-spatial-SQL generation](/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/).

The rule is absolute: model output never becomes SQL syntax. It supplies *values* — bound as parameters — and *choices* from a fixed set the server controls. The model may pick `ST_Intersects` from an allowlist and provide a WKT string as a bound parameter; it may never emit the function call or the geometry as raw text spliced into a statement.

## When to Use This Approach

Use the safe builder for every query whose shape or values originate from a model, an end user, or any untrusted upstream. The only queries exempt are fully static statements with no interpolated input — and those are rare once an agent is involved.

| Pattern | Injection risk | Notes |
|---|---|---|
| f-string / concat of model text | Critical | Never do this, even "just for the geometry" |
| Parameter binding only | Low | Values safe; but function/column names can't be bound |
| Binding + allowlist + read-only role (this page) | Minimal | Values bound, identifiers whitelisted, blast radius capped |

Parameter binding alone is necessary but not sufficient, because you cannot bind an identifier — a table, column, or function name must be validated against an allowlist instead. Layering a read-only database role underneath caps the damage even if a gap slips through. This builder is the enforcement layer beneath [generating valid PostGIS queries from natural language](/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/generating-valid-postgis-queries-from-natural-language/), and it pairs with the typed contracts in [spatial function-calling schemas](/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/).

<figure class="diagram">
<svg viewBox="16 38 748 188" role="img" aria-labelledby="psi-two-t psi-two-d" xmlns="http://www.w3.org/2000/svg"><title id="psi-two-t">Where the untrusted text actually is</title><desc id="psi-two-d">The generated statement is untrusted because a user's words shaped it, so the boundary that matters sits between generation and execution rather than between the user and the model.</desc><rect x="16" y="38" width="748" height="188" fill="#ffffff"/><rect x="30" y="52" width="340" height="160" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="410" y="52" width="340" height="160" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="200" y="84">validate before execution</text><text x="580" y="84">trust the generator</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="200" y="114">the statement is the input</text><text x="200" y="140">structure is checked</text><text x="200" y="166">the boundary is enforceable</text><text x="580" y="114">a prompt is the only control</text><text x="580" y="140">instructions can be talked around</text><text x="580" y="166">no boundary exists</text></g></svg>
<figcaption><b>Prompt instructions are not a security boundary.</b> They are a preference expressed to a system whose whole purpose is to be persuaded by text.</figcaption>
</figure>

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

<figure class="diagram">
<svg viewBox="16 32 748 214" role="img" aria-labelledby="psi-layer-t psi-layer-d" xmlns="http://www.w3.org/2000/svg"><title id="psi-layer-t">Four layers, each of which must hold alone</title><desc id="psi-layer-d">Parameterised values, a parsed statement, an allow list and a least-privilege connection — no one of them is sufficient and each catches what the others miss.</desc><rect x="16" y="32" width="748" height="214" fill="#ffffff"/><rect x="30" y="46" width="360" height="86" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="52" y="76" fill="#1f2937" font-size="13" font-weight="600">parameters, not concatenation</text><text x="52" y="102" fill="#5b6471" font-size="12">values never become syntax</text><text x="52" y="122" fill="#5b6471" font-size="12">the classic defence</text><rect x="410" y="46" width="340" height="86" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="432" y="76" fill="#1f2937" font-size="13" font-weight="600">parse, do not match</text><text x="432" y="102" fill="#5b6471" font-size="12">structure over text</text><text x="432" y="122" fill="#5b6471" font-size="12">comments cannot hide</text><rect x="30" y="146" width="360" height="86" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="52" y="176" fill="#1f2937" font-size="13" font-weight="600">allow list</text><text x="52" y="202" fill="#5b6471" font-size="12">known-good only</text><text x="52" y="222" fill="#5b6471" font-size="12">new is unknown, not allowed</text><rect x="410" y="146" width="340" height="86" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="432" y="176" fill="#1f2937" font-size="13" font-weight="600">least privilege</text><text x="432" y="202" fill="#5b6471" font-size="12">read only, scoped</text><text x="432" y="222" fill="#5b6471" font-size="12">the failure is contained</text></svg>
<figcaption><b>Design each layer as if the others are absent.</b> A layer that only works when the ones above it hold is decoration rather than defence.</figcaption>
</figure>

## Validation & Testing

- **No raw model text reaches SQL.** Assert `build_query` raises `UnsafeRequest` for `predicate="intersects; DROP TABLE parcels"` and that the returned `sql` for a valid request contains no substring of `req.wkt`.
- **Identifier allowlist holds.** Assert `layer="parcels; --"` and `predicate="ST_Union"` both reject, and that only the three known layers and predicates ever produce SQL.
- **Bbox pre-filter is present.** Assert every generated statement contains `&&` occurring before the `ST_Intersects`/`ST_Within`/`ST_DWithin` call, so the index path is never skipped.

## Gotchas & Edge Cases

- **Bindable values vs identifiers.** Newcomers try to bind a table or function name and hit a syntax error, then "fix" it by concatenating — reopening the hole. Identifiers must come from the allowlist; only values are ever bound.
- **`ST_GeomFromText` on hostile WKT.** A malformed or enormous WKT can still error or burn CPU inside the parser. Keep the regex shape check, cap WKT length before binding, and let the execution `try/except` shed the rest.
- **Read-only role that is not actually read-only.** A role with `SELECT` but also default `CREATE`/`TEMP` privileges on the schema is not safe. Revoke everything but `SELECT` on the specific layers and verify with an integration test that an `INSERT` from that role fails.
- **Second-order injection via stored values.** WKT persisted now and interpolated into a query later re-creates the risk. Bind on read too; never trust "our own" stored strings as syntax.

<figure class="diagram">
<svg viewBox="16 38 728 272" role="img" aria-labelledby="psi-fail-t psi-fail-d" xmlns="http://www.w3.org/2000/svg"><title id="psi-fail-t">What a bypass looks like at each layer</title><desc id="psi-fail-d">Each layer fails in a characteristic way, and knowing the shape of each failure is what makes the layering worth its cost.</desc><rect x="16" y="38" width="728" height="272" fill="#ffffff"/><rect x="30" y="52" width="700" height="46" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="50" y="80" fill="#1f2937" font-size="12.5">concatenated value: the value becomes syntax</text><rect x="30" y="108" width="640" height="46" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="50" y="136" fill="#1f2937" font-size="12.5">text match: a comment or unusual spacing hides the payload</text><rect x="30" y="164" width="560" height="46" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="50" y="192" fill="#1f2937" font-size="12.5">deny list: a function nobody thought of is permitted</text><rect x="30" y="220" width="700" height="46" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="50" y="248" fill="#1f2937" font-size="12.5">all four in place: the bypass has to defeat every layer</text><text x="390" y="294" fill="#1f2937" font-size="13" text-anchor="middle">The point of layering is that no single oversight is sufficient</text></svg>
<figcaption><b>The deny-list failure is the quiet one.</b> It works perfectly against everything its author knew about, which is exactly what makes its coverage so hard to reason about later.</figcaption>
</figure>

## Operating This Step Over Time

Allow lists erode under delivery pressure. Each addition is individually reasonable and the aggregate slowly becomes a deny list with extra steps. Reviewing the list as a whole a few times a year, rather than one entry at a time as they are proposed, is the only review that sees the shape it has actually taken.

Log rejections with the statement and the reason, and treat a rising rejection rate as a design signal rather than an attack. Almost all of it is legitimate capability the list does not cover; the small remainder is the part worth investigating, and it is only findable because the rest is counted.

Re-verify the connection's privileges after any infrastructure change. A connection that gained write access during a migration keeps it indefinitely, and nothing in the application will ever notice — the check that catches it is an assertion at startup rather than a periodic audit.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Is a read-only connection sufficient on its own?</span></summary><p>It bounds the damage and it does not prevent it. A read-only connection still permits reading tables the caller should not see and still permits statements expensive enough to take the database down for everyone. It is the right last layer precisely because it cannot be talked around, and it is a poor first one.</p></details>

<details class="faq-item"><summary><span>Should the model ever see raw user text in the prompt?</span></summary><p>It has to — that is the request. The mistake is letting that text reach the statement unparameterised. Treat the user's words as data that shapes structure the validator will check, and treat any literal value they supply as a parameter, and the model's exposure to raw text stops being a path to the database.</p></details>

<details class="faq-item"><summary><span>What about statements that stack multiple commands?</span></summary><p>Reject them at the parse step, unconditionally. A single request has one statement; anything with a second is either a bug in the generator or an attempt, and neither deserves execution. This is one of the cheapest checks available and it closes the whole class where a benign read is followed by something else.</p></details>

<details class="faq-item"><summary><span>How should a rejection be reported to the user?</span></summary><p>As an inability to answer that question, with no detail about the check. Naming the allow list or the rejected function tells an attacker exactly what to try next and tells an ordinary user nothing they can use. Log the specifics against a correlation identifier and give the reader a short sentence and a way to rephrase.</p></details>

<details class="faq-item"><summary><span>Do these checks need to run on every request?</span></summary><p>Yes, because the statement is different every time — this is not a configuration that can be validated once at deployment. The whole set costs a parse and a handful of set lookups, which is negligible beside the query it guards, so there is no version of this worth sampling or caching by request shape.</p></details>

## Related

- Up to the section: [Prompt-to-Spatial-SQL Generation](/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/)
- [Generating Valid PostGIS Queries from Natural Language](/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/generating-valid-postgis-queries-from-natural-language/)
- [Spatial Function-Calling Schemas](/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/)
- [Designing JSON Schemas for PostGIS Tool Calls](/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/designing-json-schemas-for-postgis-tool-calls/)
