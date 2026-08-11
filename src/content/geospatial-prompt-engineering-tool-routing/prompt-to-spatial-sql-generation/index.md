---
title: Prompt-to-Spatial-SQL Generation
description: Turn a natural-language question into PostGIS a database will accept — allow-listed functions, bound parameters, index-aware predicates, and a plan check before execution.
slug: prompt-to-spatial-sql-generation
type: topic
breadcrumb: Prompt to Spatial SQL
datePublished: 2025-04-15
dateModified: 2026-08-11
---

# Prompt-to-Spatial-SQL Generation

Generated SQL is the most powerful tool a spatial agent has and the one with the largest blast radius. A query that is syntactically valid, semantically wrong and unbounded in cost will run happily for four minutes and return an answer nobody can check. This topic is about constraining generation so that everything reaching the database is expressible, safe, index-aware and cheap enough to run.

It belongs to [geospatial prompt engineering and tool routing](/geospatial-prompt-engineering-tool-routing/) and assumes the positions in a question have already been resolved by [geocoding and place-name resolution](/spatial-llm-architecture-core-concepts/geocoding-and-place-name-resolution/) — a generated query containing a model-recalled coordinate is a fast way to produce a confident answer about the wrong place.

<figure class="diagram">
<svg viewBox="10 74 780 154" role="img" aria-labelledby="pss-gate-t pss-gate-d" xmlns="http://www.w3.org/2000/svg"><title id="pss-gate-t">Four gates between a question and an executed query</title><desc id="pss-gate-d">A generated query passes a function allow-list, a parameter binding step, an index-awareness check and a plan cost check before it is permitted to run.</desc><rect x="10" y="74" width="780" height="154" fill="#ffffff"/><rect x="24" y="88" width="140" height="70" rx="8" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="184" y="88" width="140" height="70" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="344" y="88" width="140" height="70" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="504" y="88" width="140" height="70" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="664" y="88" width="112" height="70" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="12.5" text-anchor="middle" font-weight="600"><text x="94" y="116">generated</text><text x="254" y="116">allow-list</text><text x="414" y="116">bind</text><text x="574" y="116">plan check</text><text x="720" y="116">run</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="94" y="138">untrusted text</text><text x="254" y="138">known functions</text><text x="414" y="138">no interpolation</text><text x="574" y="138">index used</text><text x="720" y="138">bounded</text></g><g stroke="#5b6471" stroke-width="2" marker-end="url(#pss-gate-a)"><line x1="166" y1="123" x2="180" y2="123"/><line x1="326" y1="123" x2="340" y2="123"/><line x1="486" y1="123" x2="500" y2="123"/><line x1="646" y1="123" x2="660" y2="123"/></g><defs><marker id="pss-gate-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/></marker></defs><text x="400" y="210" fill="#1f2937" font-size="13" text-anchor="middle">Any gate may reject; none of them rewrites the query to make it pass</text></svg>
<figcaption><b>Rejecting beats repairing.</b> A gate that silently fixes a query produces something the model did not write and nobody reviewed, and the difference between that and the intended query is exactly what nobody will notice.</figcaption>
</figure>

## Foundational Principles

**Generate a query shape, not free text.** The model should emit the parts of a query — predicate, filters, ordering, limit — into a template the application assembles, rather than a string the application executes. That single change removes injection, makes the allow-list enforceable, and turns validation from parsing into checking fields.

**Every value is bound, never interpolated.** Geometry, extents, dates and identifiers arrive as parameters. A query built by string formatting is unsafe even when the model is well-behaved, because the values frequently come from documents rather than from the model.

**Index-awareness is part of correctness.** A spatial predicate written without a bounding-box operator may or may not use the index depending on the planner's mood, and the difference is between forty milliseconds and forty seconds. Emit the index-aware form always; check the plan before running.

## Step-by-Step Implementation Pipeline

### 1. Define the query shapes the agent may produce

A small set of shapes covers the overwhelming majority of spatial questions, and constraining generation to them is what makes everything downstream tractable.

```python
import logging
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

log = logging.getLogger("spatial_sql")

Shape = Literal["features_in_region", "nearest_to", "aggregate_in_region", "attribute_lookup"]

ALLOWED_PREDICATES = {"intersects", "within", "contains", "dwithin"}
ALLOWED_AGGREGATES = {"count", "sum_area", "min", "max", "median"}


@dataclass(frozen=True)
class QuerySpec:
    shape: Shape
    table: str
    predicate: Optional[str] = None
    columns: tuple[str, ...] = ()
    filters: tuple[tuple[str, str, object], ...] = ()   # (column, op, value)
    aggregate: Optional[str] = None
    order_by_distance: bool = False
    limit: int = 100
```

### 2. Validate the specification against the schema

Validation happens against the real schema rather than against a list, so a renamed column produces a rejection rather than a query that fails at execution — or worse, one that silently matches nothing.

```python
class SpecRejected(ValueError):
    """The generated specification cannot be turned into a safe query."""


def validate(spec: QuerySpec, schema: dict[str, set[str]],
             max_limit: int = 500) -> QuerySpec:
    """Check every field against the schema and the allow-lists. Never repairs."""
    if spec.table not in schema:
        raise SpecRejected(f"unknown table {spec.table!r}")
    columns = schema[spec.table]
    for col in spec.columns:
        if col not in columns:
            raise SpecRejected(f"unknown column {col!r} on {spec.table}")
    for col, op, _value in spec.filters:
        if col not in columns:
            raise SpecRejected(f"unknown filter column {col!r}")
        if op not in {"=", "<", "<=", ">", ">=", "in", "ilike"}:
            raise SpecRejected(f"unsupported operator {op!r}")
    if spec.predicate is not None and spec.predicate not in ALLOWED_PREDICATES:
        raise SpecRejected(f"predicate {spec.predicate!r} is not permitted")
    if spec.aggregate is not None and spec.aggregate not in ALLOWED_AGGREGATES:
        raise SpecRejected(f"aggregate {spec.aggregate!r} is not permitted")
    if not 1 <= spec.limit <= max_limit:
        raise SpecRejected(f"limit {spec.limit} outside 1..{max_limit}")
    return spec
```

Refusing to repair is the discipline that makes this reviewable. A validator that silently clamps a limit of ten thousand to five hundred has produced a different query from the one the model proposed, and the agent will never learn that its plan was unreasonable. The allow-list approach is developed in [constraining generated SQL to an allow-listed function set](/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/constraining-generated-sql-to-an-allow-listed-function-set/).

### 3. Assemble the query from the specification

The application owns the SQL text. The model never sees it, never writes it, and cannot influence it beyond the fields the specification exposes.

```sql
-- features_in_region: index-aware by construction.
SELECT {columns}
FROM   {table}
WHERE  geom && ST_GeomFromEWKB(:region)          -- bounding box first, uses the index
  AND  ST_Intersects(geom, ST_GeomFromEWKB(:region))
  {extra_filters}
LIMIT  :limit;
```

```python
PREDICATE_SQL = {
    "intersects": "ST_Intersects(geom, ST_GeomFromEWKB(:region))",
    "within":     "ST_Within(geom, ST_GeomFromEWKB(:region))",
    "contains":   "ST_Contains(geom, ST_GeomFromEWKB(:region))",
    "dwithin":    "ST_DWithin(geom::geography, ST_GeomFromEWKB(:region)::geography, :radius_m)",
}


def build(spec: QuerySpec) -> tuple[str, dict]:
    """Return SQL text and bound parameters. Values never enter the string."""
    cols = ", ".join(spec.columns) if spec.columns else "*"
    clauses, params = ["geom && ST_GeomFromEWKB(:region)"], {"limit": spec.limit}
    if spec.predicate:
        clauses.append(PREDICATE_SQL[spec.predicate])
    for i, (col, op, value) in enumerate(spec.filters):
        key = f"f{i}"
        clauses.append(f"{col} {op} :{key}")
        params[key] = value
    sql = (f"SELECT {cols} FROM {spec.table} "
           f"WHERE {' AND '.join(clauses)} LIMIT :limit")
    return sql, params
```

Putting the bounding-box operator first, unconditionally, is what makes every generated query index-aware regardless of which predicate the model chose. The composition and its failure modes are covered in [generating valid PostGIS queries from natural language](/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/generating-valid-postgis-queries-from-natural-language/).

<figure class="diagram">
<svg viewBox="16 38 728 208" role="img" aria-labelledby="pss-idx-t pss-idx-d" xmlns="http://www.w3.org/2000/svg"><title id="pss-idx-t">Predicate forms and whether the spatial index is used</title><desc id="pss-idx-d">A bounding-box operator followed by an exact predicate always uses the index; the exact predicate alone usually does; a function applied to the indexed column never does.</desc><rect x="16" y="38" width="728" height="208" fill="#ffffff"/><rect x="30" y="52" width="700" height="52" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="52" y="84" fill="#1f2937" font-size="12.5">geom &amp;&amp; region AND ST_Intersects(geom, region) — index used, always</text><rect x="30" y="116" width="700" height="52" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="52" y="148" fill="#1f2937" font-size="12.5">ST_Intersects(geom, region) alone — index used, usually, plan permitting</text><rect x="30" y="180" width="700" height="52" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="52" y="212" fill="#1f2937" font-size="12.5">ST_Transform(geom, 3857) &amp;&amp; region — index never used, full scan every time</text></svg>
<figcaption><b>The third line is the one a model writes when asked to work in metres.</b> It is correct, it returns the right rows, and it reads the whole table to do it — which is invisible until the table is large.</figcaption>
</figure>

### 4. Check the plan before running

A validated, index-aware query can still be ruinous on a large table. Asking the planner for an estimate before execution is cheap and turns an unbounded risk into a decision.

```python
def plan_is_acceptable(conn, sql: str, params: dict,
                       max_cost: float = 500_000.0) -> tuple[bool, str]:
    """Ask the planner what this will cost. Reject rather than discover at runtime."""
    try:
        with conn.cursor() as cur:
            cur.execute(f"EXPLAIN (FORMAT JSON) {sql}", params)
            plan = cur.fetchone()[0][0]["Plan"]
    except Exception as exc:                       # a plan failure is a rejection, not a crash
        return False, f"could not plan the query: {exc}"
    cost = float(plan.get("Total Cost", 0.0))
    node = plan.get("Node Type", "")
    if "Seq Scan" in str(plan) and cost > max_cost / 10:
        return False, f"plan uses a sequential scan at cost {cost:.0f}"
    if cost > max_cost:
        return False, f"estimated cost {cost:.0f} exceeds the limit {max_cost:.0f}"
    return True, f"{node}, estimated cost {cost:.0f}"
```

### 5. Bound the execution as well as the plan

Estimates are estimates. A statement timeout is the backstop that turns a mis-estimated query into a bounded failure rather than a saturated database.

```sql
-- Per-transaction, not per-session: it must not leak to other work.
SET LOCAL statement_timeout = '5s';
SET LOCAL lock_timeout = '1s';
```

### 6. Return results the agent can reason about

Rows are not an answer. Returning the row count, the truncation flag and the query shape alongside the data is what lets an agent say "the first hundred of four thousand" instead of implying completeness.

```python
def to_result(rows: Sequence[dict], spec: QuerySpec, truncated: bool) -> dict:
    return {
        "rows": rows,
        "count": len(rows),
        "truncated": truncated,
        "shape": spec.shape,
        "note": ("results were limited; ask a narrower question for a complete set"
                 if truncated else ""),
    }
```

### 7. Keep the schema the model sees small and current

A model given a hundred-table schema will produce specifications referencing tables that exist and are irrelevant. Expose the handful of tables and columns that answer real questions, with one-line descriptions, and regenerate that description from the real schema so it cannot drift.

```python
def schema_for_prompt(schema: dict[str, dict], exposed: set[str]) -> str:
    """A compact description of only the tables an agent may query."""
    lines = []
    for table in sorted(exposed & set(schema)):
        cols = ", ".join(sorted(schema[table]["columns"]))
        lines.append(f"{table}({cols}) — {schema[table]['description']}")
    missing = exposed - set(schema)
    if missing:
        log.warning("exposed tables missing from the schema: %s", sorted(missing))
    return "\n".join(lines)
```

### 8. Log the specification, not the SQL

The specification is what the model produced and what a reviewer needs; the SQL is a deterministic function of it. Logging the spec makes a disputed answer reproducible and makes it possible to replay a question against a changed schema.

```python
def audit(spec: QuerySpec, verdict: str, cost_note: str, rows: int) -> dict:
    return {"shape": spec.shape, "table": spec.table, "predicate": spec.predicate,
            "filters": [(c, o) for c, o, _ in spec.filters], "limit": spec.limit,
            "verdict": verdict, "plan": cost_note, "rows": rows}
```

Note that the filter values are omitted from the audit record. They frequently contain user content, and the column and operator are what a reviewer needs to understand the query's shape.

### 9. Handle the aggregate shapes separately

Aggregates are where a generated query is most likely to be both expensive and misleading. A count over a region is cheap when the region is small and ruinous when it is not, and the same specification produces both depending on a parameter the model chose.

```python
AGGREGATE_SQL = {
    "count":     "count(*)",
    "sum_area":  "sum(ST_Area(geom::geography))",
    "min":       "min({col})",
    "max":       "max({col})",
    "median":    "percentile_cont(0.5) WITHIN GROUP (ORDER BY {col})",
}


def build_aggregate(spec: QuerySpec, region_area_km2: float,
                    max_region_km2: float = 5_000.0) -> tuple[str, dict]:
    """Aggregates require a bounded region — an unbounded count is not a query."""
    if spec.aggregate is None:
        raise SpecRejected("aggregate shape without an aggregate function")
    if region_area_km2 > max_region_km2:
        raise SpecRejected(
            f"region of {region_area_km2:.0f} km² exceeds the {max_region_km2:.0f} km² "
            "limit for aggregates; narrow the area or ask for a pre-computed statistic")
    body = AGGREGATE_SQL[spec.aggregate]
    if "{col}" in body:
        if not spec.columns:
            raise SpecRejected(f"aggregate {spec.aggregate!r} needs a column")
        body = body.format(col=spec.columns[0])
    sql = (f"SELECT {body} AS value FROM {spec.table} "
           "WHERE geom && ST_GeomFromEWKB(:region) "
           "AND ST_Intersects(geom, ST_GeomFromEWKB(:region))")
    return sql, {}
```

The region-area limit is a policy rather than a technical constraint, and stating it in the rejection is what makes it actionable. "Narrow the area or ask for a pre-computed statistic" tells the agent exactly which two moves are available, and the second of them is usually the right one for a genuinely national question.

Aggregates also deserve their own truncation semantics, which is to say none: an aggregate is either computed over the whole region or it is refused. A count limited to five hundred rows is not a count, and returning one with a truncation flag invites exactly the misreading the flag was meant to prevent.

### 10. Give the model examples of accepted specifications, not of SQL

Few-shot examples shape what a model produces more than instructions do, and examples of SQL teach it to write SQL. A handful of accepted specifications — one per shape, with realistic tables and filters — is both smaller in the context and better aligned with what the layer will accept.

```python
def example_specs() -> list[dict]:
    """Compact, schema-accurate examples: one per shape, regenerated from the schema."""
    return [
        {"shape": "features_in_region", "table": "parcels",
         "predicate": "intersects", "columns": ["parcel_ref", "use_class"], "limit": 50},
        {"shape": "nearest_to", "table": "stations",
         "predicate": "dwithin", "order_by_distance": True, "limit": 5},
        {"shape": "aggregate_in_region", "table": "buildings",
         "aggregate": "count", "predicate": "within"},
    ]
```

Regenerating the examples from the live schema matters for the same reason the description does. An example naming a column that no longer exists teaches the model to produce specifications that will be rejected, and the rejection rate rises for reasons nobody connects to a migration weeks earlier.

## Operating This Stage Over Time

Schemas change, and a generated-query layer is unusually sensitive to it. A renamed column turns every specification referencing it into a rejection, which is the correct behaviour and presents to users as the agent suddenly being unable to answer a question it handled yesterday. Regenerating the prompt's schema description from the live schema on every deploy is what keeps the two in step; a hand-maintained description drifts within weeks.

The allow-lists drift in the other direction. Each new capability adds a predicate or an aggregate, and after a year the list admits most of the function library — at which point it is documentation rather than a constraint. Review it periodically against what is actually used: a permitted function that has never appeared in an accepted specification is a permission with no beneficiary.

Cost limits need the same treatment for the opposite reason. A limit set against a small table will reject reasonable queries once the table is large, and the symptom is a rising rejection rate rather than an error. Track rejections by reason — unknown column, forbidden predicate, plan too expensive — because those three point at completely different fixes.

Finally, watch the truncation rate. A question that consistently returns truncated results is a question the shape vocabulary does not serve well, usually because it wanted an aggregate and got a sample. That is a signal to add a shape rather than to raise a limit.

## Failure Modes & Root Causes

**The full-table scan.** A correct query reads everything because a function wrapped the indexed column. Root cause: transforming the column rather than the parameter. Mitigation: emit the index-aware form from the template; check the plan.

**The silently narrowed answer.** A limit truncates results and the answer implies completeness. Root cause: rows returned without a truncation flag. Mitigation: return the flag and the count, and let the agent say so.

**The query that matches nothing.** A filter references a column that was renamed, or a value in the wrong units, and the result is an empty set that reads as "there is nothing there". Root cause: validation against a stale schema description. Mitigation: validate against the live schema; regenerate the prompt description from it.

**The expensive aggregate.** A count over a national table with no region filter runs for minutes. Root cause: a shape that permits an unbounded aggregate. Mitigation: require a region on every shape, and reject a specification without one.

## Production Validation Protocols

1. **No-interpolation assertion.** Assert every generated query is executed with bound parameters and that the SQL text contains no value from the specification.
2. **Index-plan test.** For each shape, assert the plan uses a spatial index against a representative table; a plan regression is silent and costly.
3. **Allow-list enforcement test.** Assert a specification naming a forbidden predicate is rejected, and that the rejection names the predicate.
4. **Truncation reporting test.** Assert that a result at the limit carries the truncation flag and the note.
5. **Schema-drift check.** Assert every exposed table and column in the prompt description exists in the live schema, on every deploy.
6. **Rejection-reason breakdown.** Publish rejections by reason and alert on a step change in any one; the three reasons have three different fixes.

<figure class="diagram">
<svg viewBox="16 38 728 178" role="img" aria-labelledby="pss-spec-t pss-spec-d" xmlns="http://www.w3.org/2000/svg"><title id="pss-spec-t">Generating a specification against generating SQL text</title><desc id="pss-spec-d">A model emitting structured fields can be validated field by field, while a model emitting SQL text must be parsed, and any parser gap becomes an execution risk.</desc><rect x="16" y="38" width="728" height="178" fill="#ffffff"/><rect x="30" y="52" width="330" height="150" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="195" y="84" fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600">SQL text</text><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="195" y="114">must be parsed to be checked</text><text x="195" y="140">every parser gap is a risk</text><text x="195" y="166">injection is possible</text><text x="195" y="190">review means reading SQL</text></g><rect x="400" y="52" width="330" height="150" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="565" y="84" fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600">specification</text><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="565" y="114">checked field by field</text><text x="565" y="140">no parser needed</text><text x="565" y="166">injection is structurally absent</text><text x="565" y="190">review means reading fields</text></g></svg>
<figcaption><b>The difference is not defence in depth; it is a different problem.</b> Validating SQL text means writing a parser and trusting it; validating a specification means checking a handful of fields against a schema, which is a task with an end.</figcaption>
</figure>

## Frequently Asked Questions

<details class="faq-item"><summary><span>Is a query shape vocabulary too restrictive for real questions?</span></summary><p>Less than it appears. Four or five shapes cover the overwhelming majority of spatial questions, and the ones they do not cover are usually questions that should not be answered by a single generated query anyway — multi-step analyses belong to the orchestration layer, where each step is itself a shape. When a genuinely new shape is needed, adding one is a small, reviewable change; the alternative is a system where any query is possible and none is checkable.</p></details>

<details class="faq-item"><summary><span>Should the model ever see the generated SQL?</span></summary><p>Only for explanation, and only after execution. Showing it beforehand invites the model to edit it, which reintroduces every problem the specification approach removes. Showing it afterwards, alongside the results, can help the model explain what it did — but the explanation should be generated from the specification, which is more readable and cannot contain anything the query did not.</p></details>

<details class="faq-item"><summary><span>How should read-only access be enforced?</span></summary><p>At the database, with a role that has no write privileges on anything, rather than by inspecting the generated text. Permissions are the only enforcement that cannot be reasoned around, and they also protect against the paths that do not go through this layer at all. The allow-list is a usability and cost control; the role is the security boundary.</p></details>

<details class="faq-item"><summary><span>What about questions that need a join?</span></summary><p>Add them as shapes with the joins pre-written, rather than letting a specification express arbitrary joins. Spatial joins are the queries most likely to be catastrophically expensive, and a small number of named, plan-checked join shapes covers the real cases while keeping the cost bounded. An agent that needs an unanticipated join is telling you about a missing shape.</p></details>

<details class="faq-item"><summary><span>Does this replace the need for a statement timeout?</span></summary><p>No — the plan check and the timeout catch different things. A plan estimate can be wrong, particularly on skewed data or stale statistics, and the timeout is what bounds the damage when it is. Set both, keep the timeout local to the transaction so it cannot leak, and treat a timeout as a signal that the cost model needs attention rather than as a routine outcome.</p></details>

## Related

- Up to the section overview: [Geospatial Prompt Engineering and Tool Routing](/geospatial-prompt-engineering-tool-routing/)
- Technique: [Generating Valid PostGIS Queries from Natural Language](/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/generating-valid-postgis-queries-from-natural-language/)
- Technique: [Preventing SQL Injection in LLM-Generated Spatial Queries](/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/preventing-sql-injection-in-llm-generated-spatial-queries/)
- Technique: [Constraining Generated SQL to an Allow-Listed Function Set](/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/constraining-generated-sql-to-an-allow-listed-function-set/)
- Peer topic: [Spatial Function-Calling Schemas](/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/)
- Peer topic: [GeoPandas and PostGIS Tool Routing](/geospatial-prompt-engineering-tool-routing/geopandas-postgis-tool-routing/)
