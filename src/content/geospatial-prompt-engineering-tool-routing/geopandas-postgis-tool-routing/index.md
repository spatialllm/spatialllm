---
title: GeoPandas and PostGIS Tool Routing
description: Decide per operation whether geometry work belongs in the database or in the process, using data volume, index availability and where the result has to go next.
slug: geopandas-postgis-tool-routing
type: topic
breadcrumb: GeoPandas and PostGIS Routing
datePublished: 2025-03-18
dateModified: 2026-08-11
---

# GeoPandas and PostGIS Tool Routing

An agent with both a spatial database and an in-process geometry library has two ways to do almost everything, and the choice is not a matter of taste. Sending a million-row overlay into a Python process is a memory incident; round-tripping a four-feature buffer through the database is three network hops for work that takes microseconds. This topic is about routing each operation to the side that should do it.

It belongs to [geospatial prompt engineering and tool routing](/geospatial-prompt-engineering-tool-routing/) and depends on the query construction described in [prompt-to-spatial-SQL generation](/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/) for everything that lands on the database side.

<figure class="diagram">
<svg viewBox="16 38 748 188" role="img" aria-labelledby="gpr-two-t gpr-two-d" xmlns="http://www.w3.org/2000/svg"><title id="gpr-two-t">What each side is good at</title><desc id="gpr-two-d">The database wins on volume, indexes and filtering; the process wins on iteration, custom logic and small data already in memory, and the routing decision turns on which of those dominates.</desc><rect x="16" y="38" width="748" height="188" fill="#ffffff"/><rect x="30" y="52" width="340" height="160" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="410" y="52" width="340" height="160" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="200" y="84">database</text><text x="580" y="84">in process</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="200" y="114">indexes, and volume</text><text x="200" y="140">filters before it moves data</text><text x="200" y="166">one round trip</text><text x="200" y="192">wins when data is large</text><text x="580" y="114">iteration and custom logic</text><text x="580" y="140">no serialisation cost</text><text x="580" y="166">easy to inspect and test</text><text x="580" y="192">wins when data is already here</text></g></svg>
<figcaption><b>The deciding factor is where the data already is.</b> Everything else — index availability, operation complexity, result size — modifies a decision that mostly follows from whether the rows have to move.</figcaption>
</figure>

## Foundational Principles

**Move the computation to the data, not the data to the computation.** A filter, a join or an aggregate over many rows belongs where the rows live. Fetching a million geometries to count them is the canonical mistake, and it is made by code that reads perfectly reasonably.

**Small data in memory stays in memory.** Once a few hundred features have been fetched, doing three more operations on them in process is faster and simpler than three more round trips, even when the database would be faster per operation.

**The routing decision is per operation, not per pipeline.** A chain that filters in the database, iterates in the process and aggregates back in the database is normal and correct, and forcing the whole chain to one side is what produces both of the failures above.

## Step-by-Step Implementation Pipeline

### 1. Estimate the row count before deciding anything

Every routing decision starts from how many rows the operation touches, and that number is available cheaply from statistics or from a bounded count.

```python
import logging
from dataclasses import dataclass
from typing import Literal, Optional

log = logging.getLogger("spatial_routing")

Side = Literal["database", "process"]

FETCH_CEILING = 50_000            # rows above which fetching is a memory risk
ROUND_TRIP_FLOOR = 200            # rows below which a round trip is not worth it


@dataclass(frozen=True)
class Estimate:
    rows: int
    exact: bool
    basis: str


def estimate_rows(conn, table: str, region_wkb: bytes, cap: int = 100_000) -> Estimate:
    """A bounded count: exact when small, capped when not, never a full scan."""
    sql = ("SELECT count(*) FROM (SELECT 1 FROM {table} "
           "WHERE geom && ST_GeomFromEWKB(%s) LIMIT %s) t").format(table=table)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (region_wkb, cap))
            rows = int(cur.fetchone()[0])
    except Exception as exc:                        # a failed estimate routes conservatively
        log.warning("row estimate failed for %s: %s", table, exc)
        return Estimate(cap, False, f"estimate failed: {exc}")
    return Estimate(rows, rows < cap, "bounded count over the region")
```

Capping the count is what keeps the estimate from becoming the expensive operation. A count that stops at a hundred thousand tells you everything the routing decision needs, and an uncapped one over a national table is exactly the query the routing was supposed to prevent.

### 2. Route on volume first

Volume dominates every other consideration, and the two thresholds are the whole of the common case.

```python
def route(est: Estimate, operation: str, already_in_memory: bool,
          index_available: bool) -> tuple[Side, str]:
    """Choose a side and say why. Volume decides most cases."""
    if already_in_memory and est.rows <= FETCH_CEILING:
        return "process", "the data is already here"
    if est.rows > FETCH_CEILING:
        return "database", f"{est.rows}+ rows is too many to fetch"
    if est.rows < ROUND_TRIP_FLOOR and already_in_memory:
        return "process", "too few rows to justify a round trip"
    if not index_available and est.rows > ROUND_TRIP_FLOOR:
        log.info("no spatial index on this table; the database has no advantage")
        return "process", "no index available to exploit"
    return "database", "the database can filter before moving data"
```

The index check earns its place. A database's advantage on a filtered operation comes almost entirely from the index, and against an unindexed table it is doing the same linear scan the process would do — with serialisation on top. The full decision surface is laid out in [the GeoPandas and PostGIS decision matrix](/geospatial-prompt-engineering-tool-routing/geopandas-postgis-tool-routing/geopandas-vs-postgis-decision-matrix/).

### 3. Push filters down, always

Whatever side the operation runs on, the filter runs in the database. This is the single highest-value rule in the topic, because it changes the volume that every subsequent decision is made against.

```sql
-- The filter is index-aware and runs before anything is fetched.
SELECT id, ST_AsBinary(geom) AS geom, use_class
FROM   parcels
WHERE  geom && ST_GeomFromEWKB(:region)
  AND  ST_Intersects(geom, ST_GeomFromEWKB(:region))
  AND  use_class = ANY(:classes)
LIMIT  :cap;
```

Fetching a region's worth of rows and filtering by attribute in the process is a common and expensive shape: the attribute filter is free in the database and costs a full transfer in the application.

### 4. Choose the transfer format deliberately

When rows do move, the encoding matters. Binary geometry is compact and fast to parse; text is human-readable and roughly twice the size; a structured object form is larger still and easier to inspect.

```python
def fetch_geometries(conn, sql: str, params: dict, cap: int = FETCH_CEILING):
    """Fetch as binary, stop at the cap, and report truncation rather than hiding it."""
    with conn.cursor() as cur:
        cur.execute(sql, {**params, "cap": cap + 1})
        rows = cur.fetchall()
    truncated = len(rows) > cap
    if truncated:
        log.info("fetch truncated at %d rows; consider routing this to the database", cap)
        rows = rows[:cap]
    return rows, truncated
```

Fetching one row beyond the cap is what makes truncation detectable. Without it, a result of exactly the cap size is ambiguous between "that is all there was" and "there was more", and the two mean completely different things for the answer.

<figure class="diagram">
<svg viewBox="66 9 622 237" role="img" aria-labelledby="gpr-cost-t gpr-cost-d" xmlns="http://www.w3.org/2000/svg"><title id="gpr-cost-t">Where time goes on each side as row count grows</title><desc id="gpr-cost-d">In-process work is dominated by transfer as rows grow, while database work stays flat because the rows never move — with the crossover well below the point most pipelines assume.</desc><rect x="66" y="9" width="622" height="237" fill="#ffffff"/><text x="390" y="34" fill="#5b6471" font-size="13" text-anchor="middle">One overlay, four row counts, two routes</text><g fill="#e4f5ec" stroke="#12805c" stroke-width="2"><rect x="80" y="146" width="54" height="28" rx="4"/><rect x="240" y="140" width="54" height="34" rx="4"/><rect x="400" y="134" width="54" height="40" rx="4"/><rect x="560" y="128" width="54" height="46" rx="4"/></g><g fill="#fdeaee" stroke="#b3324f" stroke-width="2"><rect x="140" y="152" width="54" height="22" rx="4"/><rect x="300" y="130" width="54" height="44" rx="4"/><rect x="460" y="92" width="54" height="82" rx="4"/><rect x="620" y="52" width="54" height="122" rx="4"/></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="137" y="196">200</text><text x="297" y="196">5 000</text><text x="457" y="196">50 000</text><text x="617" y="196">500 000</text></g><text x="390" y="228" fill="#1f2937" font-size="12.5" text-anchor="middle">Left bar of each pair: database. Right bar: fetch and process.</text></svg>
<figcaption><b>The crossover is earlier than it feels.</b> At two hundred rows the process wins comfortably; by fifty thousand the transfer dominates everything the process saves, and the curve keeps climbing.</figcaption>
</figure>

### 5. Keep results where the next step needs them

An operation's output should land on the side that will consume it. A database-side overlay whose result is immediately fetched for one small transformation has paid the transfer anyway; a process-side computation whose result is written back for a join has paid it twice.

```python
def plan_chain(steps: list[dict], est: Estimate) -> list[tuple[str, Side, str]]:
    """Assign a side per step, preferring to stay put rather than alternate."""
    plan, current: Optional[Side] = [], None
    for step in steps:
        side, why = route(est, step["op"], already_in_memory=(current == "process"),
                          index_available=step.get("indexed", True))
        if current is not None and side != current:
            why += "; crossing sides costs a transfer"
        plan.append((step["op"], side, why))
        current = side
    return plan
```

Counting the crossings is a useful discipline. A five-step chain that alternates sides four times has paid four transfers, and reordering the steps so that all the database work happens together is frequently possible and always cheaper.

### 6. Use the process for what the database cannot express

Custom logic — a scoring function, an iterative snap, a rule that depends on external state — belongs in the process regardless of volume, and the correct response to large volume is to reduce the rows first rather than to move the logic.

```python
def reduce_then_iterate(conn, sql: str, params: dict, predicate, cap: int = FETCH_CEILING):
    """Filter and aggregate in the database; iterate over what survives."""
    rows, truncated = fetch_geometries(conn, sql, params, cap)
    if truncated:
        raise TooManyRows(
            f"more than {cap} rows survive the filter; narrow the region or "
            "express the rule as a query")
    return [r for r in rows if predicate(r)]
```

Raising rather than processing a truncated set is the important behaviour. A custom rule applied to the first fifty thousand of two hundred thousand rows produces an answer that is silently about a quarter of the data.

### 7. Keep both sides consistent about geometry

The two sides can disagree — about validity handling, about precision, about what an empty geometry means — and a chain that crosses sides can produce a result neither side would have produced alone.

```python
def normalise_for_process(geom):
    """Apply the same validity and precision policy the database side uses."""
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
    return round_geometry(geom, PRECISION_PLACES)
```

### 8. Record which side served each operation

The routing distribution is the signal that tells you whether the thresholds are right. A system routing everything to one side has thresholds that are not discriminating, which is worth knowing before someone concludes the other side is unnecessary.

```python
def observe(op: str, side: Side, rows: int, elapsed_s: float, metrics) -> None:
    metrics.increment("spatial.route", tags={"op": op, "side": side})
    metrics.timing("spatial.route.seconds", elapsed_s, tags={"op": op, "side": side})
    metrics.gauge("spatial.route.rows", rows, tags={"op": op, "side": side})
```

### 9. Give the agent one tool, not two

The routing decision belongs in the application, not in the model. An agent offered separate database and in-process tools will choose between them on grounds that have nothing to do with row counts — phrasing, recency, whichever appeared first in the schema — and the choice will be wrong in exactly the cases where it matters.

```python
def spatial_operation(op: str, params: dict, conn) -> dict:
    """One tool. The agent asks for an operation; the router decides where it runs."""
    est = estimate_rows(conn, params["table"], params["region"])
    side, why = route(est, op, already_in_memory=params.get("in_memory", False),
                      index_available=has_spatial_index(conn, params["table"]))
    started = time.monotonic()
    result = (run_in_database(op, params, conn) if side == "database"
              else run_in_process(op, params, conn))
    observe(op, side, est.rows, time.monotonic() - started, metrics)
    return {"result": result, "side": side, "rows_considered": est.rows, "routing": why}
```

Returning the side and the row count with the result is what keeps the routing debuggable without exposing it as a choice. The agent can mention that an answer considered forty thousand features; it cannot decide to consider them in the wrong place.

This also removes an entire category of prompt engineering. A tool schema describing two backends invites the model to reason about backends, which it cannot do usefully and which consumes context that should describe the operation.

### 10. Handle the case where neither side can

Some operations are too large for the process and too expensive for the database within the turn's budget, and the honest response is neither a fetch nor a query but a job — or a narrower question.

```python
def route_or_defer(est: Estimate, op: str, budget_s: float,
                   db_rate: float, enqueue) -> tuple[str, str]:
    """When both sides exceed the budget, defer rather than starting either."""
    projected_s = est.rows * db_rate
    if projected_s <= budget_s:
        return "database", ""
    if est.rows > FETCH_CEILING:
        job = enqueue(op=op, rows=est.rows)
        return "queued", f"{est.rows} rows projected at {projected_s:.0f}s; started job {job}"
    return "process", "over the database budget but small enough to fetch"
```

The middle branch is the one worth having explicitly. Without it a large operation runs on whichever side was chosen by the volume rule, consumes the whole budget, and fails — where deferring produces a handle and a wait the user can decide about.

## Operating This Stage Over Time

Thresholds drift with the data. A fetch ceiling that was comfortable against a corpus of two million rows is a memory risk against twenty million, and nothing in the code notices — the same operation simply fetches more. Deriving the ceiling from available memory rather than from a constant, and reviewing it when the corpus grows, is what keeps the routing honest.

Index availability changes too, and usually silently. An index dropped during a migration and not recreated turns every database-side operation on that table into a scan, and the symptom is a latency regression that looks like a database problem rather than a missing index. Asserting index presence for the tables the router assumes are indexed is a cheap check that catches it immediately.

The third drift is in the operations themselves. New capabilities add steps to chains, and a chain that once ran entirely in the database acquires a process-side step in the middle, doubling its transfers. Counting side crossings per chain and reviewing the ones that grew is a more useful signal than per-operation latency, because the cost is in the pattern rather than in any one step.

Finally, watch for a routing decision that has become folklore. "We always do overlays in the database" starts as a measured conclusion and survives long after the data that justified it has changed shape. Re-measuring the crossover once a year takes an afternoon and occasionally moves it by an order of magnitude.

## Failure Modes & Root Causes

**The full fetch.** A million geometries are pulled into memory to be counted or filtered. Root cause: an operation expressed in the application that the database could have done. Mitigation: estimate rows first; push filters down unconditionally.

**The chatty chain.** Six operations on a hundred features produce six round trips. Root cause: routing per operation without considering where the data already is. Mitigation: the in-memory check, and counting side crossings.

**The silent truncation.** A fetch caps at fifty thousand rows and the answer describes a quarter of the data. Root cause: a cap with no truncation signal. Mitigation: fetch one extra row and report.

**The disagreeing sides.** A geometry repaired differently in the process than in the database produces a result that neither side would have produced. Root cause: two validity policies. Mitigation: one policy, applied on both sides.

## Production Validation Protocols

1. **Filter push-down assertion.** Assert that no code path fetches rows and then applies an attribute filter that the query could have expressed.
2. **Truncation detection test.** Assert a fetch at exactly the cap is reported as truncated, using a fixture sized to the boundary.
3. **Index presence check.** Assert every table the router treats as indexed has a spatial index, on every deploy.
4. **Crossing count.** Publish the number of side crossings per chain and alert when a chain's count increases.
5. **Consistency test.** Run a representative operation on both sides against the same input and assert the results agree within the precision policy.
6. **Routing distribution.** Publish the split by operation and alert on a collapse to one side, which usually means a threshold has stopped discriminating.

<figure class="diagram">
<svg viewBox="16 24 692 201" role="img" aria-labelledby="gpr-chain-t gpr-chain-d" xmlns="http://www.w3.org/2000/svg"><title id="gpr-chain-t">A chain that alternates sides against one that does not</title><desc id="gpr-chain-d">Reordering the same five steps so the database work happens together removes three transfers without changing the result.</desc><rect x="16" y="24" width="692" height="201" fill="#ffffff"/><text x="30" y="62" fill="#b3324f" font-size="13" font-weight="600">alternating</text><rect x="170" y="38" width="100" height="42" rx="5" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="276" y="38" width="100" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="382" y="38" width="100" height="42" rx="5" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="488" y="38" width="100" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="594" y="38" width="100" height="42" rx="5" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="430" y="104" fill="#5b6471" font-size="12" text-anchor="middle">four transfers</text><text x="30" y="166" fill="#12805c" font-size="13" font-weight="600">grouped</text><rect x="170" y="142" width="100" height="42" rx="5" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="276" y="142" width="100" height="42" rx="5" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="382" y="142" width="100" height="42" rx="5" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="488" y="142" width="100" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="594" y="142" width="100" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="430" y="208" fill="#5b6471" font-size="12" text-anchor="middle">one transfer</text></svg>
<figcaption><b>Same steps, same result, a quarter of the transfers.</b> Chains acquire alternation gradually as steps are added, and nothing about any individual step looks wrong — which is why the count is worth watching rather than the steps.</figcaption>
</figure>

## Frequently Asked Questions

<details class="faq-item"><summary><span>Where should the fetch ceiling sit?</span></summary><p>Derived from available memory and typical geometry size rather than chosen as a round number. A ceiling of fifty thousand is reasonable for small parcels and far too high for detailed boundaries with thousands of vertices each. Computing it from a measured average geometry size and a memory allowance makes it adapt to the data, and makes the constant meaningful rather than traditional.</p></details>

<details class="faq-item"><summary><span>Should routing consider which side has spare capacity?</span></summary><p>In principle yes, in practice rarely worth it. Load-aware routing is appealing and adds a dependency on metrics that may themselves be stale, and the volume-based decision is right often enough that overriding it on load mostly moves work to the side that is worse at it. Where the database is genuinely saturated, the answer is usually a queue rather than a route change.</p></details>

<details class="faq-item"><summary><span>What about operations the database supports but implements differently?</span></summary><p>Prefer the database and verify the difference once. Buffer, simplify and overlay have subtly different implementations in different engines — different join styles, different precision models — and the differences are usually irrelevant and occasionally not. A consistency test against a representative fixture tells you which case you are in, and it needs running once per library upgrade rather than continuously.</p></details>

<details class="faq-item"><summary><span>How does this interact with the async decision?</span></summary><p>They are independent and compose. Routing decides which side does the work; the async decision decides whether the user waits for it. A large database-side overlay can still be a queued job, and a small process-side operation is always inline. Keeping the two decisions separate avoids the tangle where a routing rule secretly encodes a latency assumption.</p></details>

<details class="faq-item"><summary><span>Is it worth supporting both sides at all?</span></summary><p>For anything beyond a small system, yes. A database-only pipeline cannot express custom logic without pushing it into SQL where it is hard to test, and a process-only pipeline cannot handle volume. The cost of supporting both is the consistency work described above, which is real and bounded; the cost of supporting neither properly is a system that hits a wall in one direction or the other.</p></details>

## Related

- Up to the section overview: [Geospatial Prompt Engineering and Tool Routing](/geospatial-prompt-engineering-tool-routing/)
- Technique: [Routing LLM Calls to GeoPandas vs PostGIS Backends](/geospatial-prompt-engineering-tool-routing/geopandas-postgis-tool-routing/routing-llm-calls-to-geopandas-vs-postgis-backends/)
- Comparison: [GeoPandas vs PostGIS Decision Matrix](/geospatial-prompt-engineering-tool-routing/geopandas-postgis-tool-routing/geopandas-vs-postgis-decision-matrix/)
- Peer topic: [Prompt-to-Spatial-SQL Generation](/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/)
- Peer topic: [LLM-Assisted Geoprocessing Pipelines](/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/)
- Related topic: [Cost and Latency Budgets for Spatial Agents](/geospatial-prompt-engineering-tool-routing/cost-and-latency-budgets-for-spatial-agents/)
