---
title: GeoPandas vs PostGIS Decision Matrix
description: A concrete matrix for sending spatial work to the database or the process — inputs, thresholds, the split-the-work case, and how to measure the crossover on your own data.
slug: geopandas-vs-postgis-decision-matrix
type: howto
breadcrumb: Decision Matrix
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# GeoPandas vs PostGIS Decision Matrix

Most routing arguments between an in-process library and a spatial database are settled by preference, and preference is wrong by an order of magnitude often enough to matter. A matrix built from row counts, index availability and the shape of the operation settles them from evidence instead. This guide gives that matrix and the measurement behind it — the concrete half of [GeoPandas and PostGIS tool routing](/geospatial-prompt-engineering-tool-routing/geopandas-postgis-tool-routing/).

## When to Use This Approach

Any system with both backends available needs an explicit rule, because the default in the absence of one is whichever the author is more comfortable with.

| Situation | Route to | Why |
|-----------|----------|-----|
| Millions of rows, indexed geometry column | PostGIS | Transfer cost dominates |
| Thousands of rows, already in memory | GeoPandas | Round trip dominates |
| Aggregate over a large table | PostGIS | The answer is small |
| Iterative or awkward to express in SQL | GeoPandas | Expression cost dominates |
| Large table, complex per-feature work | Both | Filter in the database, finish in process |

<figure class="diagram">
<svg viewBox="16 38 748 212" role="img" aria-labelledby="gvp-two-t gvp-two-d" xmlns="http://www.w3.org/2000/svg"><title id="gvp-two-t">What decides the routing in each direction</title><desc id="gvp-two-d">The database wins when moving the data would cost more than computing on it there, and the in-process library wins when the round trip and expression cost exceed the computation.</desc><rect x="16" y="38" width="748" height="212" fill="#ffffff"/><rect x="30" y="52" width="340" height="160" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="410" y="52" width="340" height="160" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="200" y="84">send it to the database</text><text x="580" y="84">keep it in the process</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="200" y="114">the data is large</text><text x="200" y="140">an index already exists</text><text x="200" y="166">the answer is much smaller</text><text x="580" y="114">the data is small</text><text x="580" y="140">it is already loaded</text><text x="580" y="166">the operation is awkward in SQL</text></g><text x="390" y="234" fill="#1f2937" font-size="13" text-anchor="middle">Neither is the default — routing everything one way breaks at one end or the other</text></svg>
<figcaption><b>Both failure modes are common.</b> A system that routes everything to the database ends up doing arithmetic on four rows across a network; one that routes everything in process eventually tries to load a continent.</figcaption>
</figure>

## Implementation

The matrix takes three inputs and returns a route. Row count comes from table statistics; index availability comes from the catalogue; operation shape is a property of the plan step you already have.

```python
def route(step, stats) -> str:
    rows = stats.row_estimate(step.source)
    indexed = stats.has_spatial_index(step.source, step.geom_column)

    if step.produces_aggregate and rows > SMALL:
        return "postgis"                       # the answer is tiny, the input is not
    if rows > LARGE and indexed:
        return "postgis"
    if rows > LARGE and not indexed:
        return "postgis_with_warning"          # still better than transferring it
    if step.op in AWKWARD_IN_SQL:
        return "geopandas"
    if rows <= SMALL:
        return "geopandas"
    return "split"
```

The `split` route is the one that earns the most and is written the least. It applies a spatial predicate in the database to reduce the population, then finishes the awkward part in process on what survives.

```python
def run_split(step, conn):
    narrowed = read_postgis(
        "SELECT * FROM {t} WHERE ST_Intersects({g}, ST_GeomFromEWKB(%s))".format(
            t=step.source, g=step.geom_column),
        conn, params=(step.region.wkb,), geom_col=step.geom_column)
    return awkward_operation(narrowed)         # hundreds of rows, not millions
```

The thresholds `SMALL` and `LARGE` are the only numbers in this that must be measured rather than chosen. Everything else is structural and transfers between deployments; those two do not.

<figure class="diagram">
<svg viewBox="13 38 753 220" role="img" aria-labelledby="gvp-band-t gvp-band-d" xmlns="http://www.w3.org/2000/svg"><title id="gvp-band-t">Where the crossover sits and why it moves</title><desc id="gvp-band-d">Small inputs favour the in-process path, large ones favour the database, and the band between them is wide enough that the choice matters little — but its position depends on hardware and data.</desc><rect x="13" y="38" width="753" height="220" fill="#ffffff"/><rect x="30" y="52" width="300" height="46" rx="6" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="50" y="80" fill="#1f2937" font-size="12.5">small: the round trip dominates — in process</text><rect x="30" y="108" width="480" height="46" rx="6" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="50" y="136" fill="#1f2937" font-size="12.5">the band between them: the two are within noise, either is fine</text><rect x="30" y="164" width="700" height="46" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="50" y="192" fill="#1f2937" font-size="12.5">large: transfer dominates — the database, decisively</text><text x="390" y="240" fill="#1f2937" font-size="13" text-anchor="middle">The band is wide, which is why a roughly-right threshold works and a borrowed one does not</text></svg>
<figcaption><b>The middle band is forgiving and its edges are not.</b> Getting the threshold approximately right is enough; getting it wrong by an order of magnitude — which is what borrowing a number from elsewhere does — is not.</figcaption>
</figure>

## Validation & Testing

Measure the crossover once, properly, with a benchmark that runs both paths over the same inputs at several sizes. The result is two numbers and a graph nobody needs to look at twice.

```python
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000]

def measure_crossover(conn, op):
    for n in SIZES:
        pg = time_it(lambda: run_in_postgis(op, conn, limit=n))
        gp = time_it(lambda: run_in_geopandas(op, conn, limit=n))
        print(f"{n:>9,}  postgis {pg:6.2f}s  geopandas {gp:6.2f}s")
```

Then test the router itself against fixtures rather than against the database: given a step and a stats object, assert the route. That keeps the routing logic testable without a database and makes the threshold change a one-line diff with visible consequences.

## Gotchas & Edge Cases

**Row count as a proxy for cost.** A thousand coastline polygons with tens of thousands of vertices each cost far more than a thousand building footprints. Where a table is known to hold complex geometry, a per-table multiplier is crude but maintainable; modelling vertex counts properly is better and rarely survives contact with a deadline.

**An index that exists but is not used.** A spatial index on a column the query transforms — a function applied to the geometry before comparison — is not used, and the planner will say so. Checking for the index in the catalogue is necessary and not sufficient; checking the plan is what confirms it.

**Memory as a hard boundary.** The in-process path fails absolutely rather than slowly when the frame does not fit, and the failure arrives after the transfer has already been paid for. Any route to the process should have a row ceiling above which it is simply not attempted, independent of the cost comparison.

**Different answers from the two backends.** Buffer results, validity definitions and floating-point handling differ slightly between GEOS versions and between the library and the server. For most questions this is noise; for a comparison against a previously stored result it is a discrepancy somebody will report, and recording which backend produced each answer is what makes it explicable.

**The split route not being considered.** Because the matrix reads as a binary choice, the split case tends to be skipped even when it is the best answer. Making it an explicit return value rather than an implementation detail is most of what gets it used.

<figure class="diagram">
<svg viewBox="16 52 748 194" role="img" aria-labelledby="gvp-split-t gvp-split-d" xmlns="http://www.w3.org/2000/svg"><title id="gvp-split-t">The split route in three stages</title><desc id="gvp-split-d">A spatial predicate in the database narrows millions of rows to hundreds, those are transferred, and the awkward computation runs in process where it is easy to express.</desc><rect x="16" y="52" width="748" height="194" fill="#ffffff"/><defs><marker id="gvp-split-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/></marker></defs><rect x="30" y="66" width="212" height="120" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="284" y="66" width="212" height="120" rx="8" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="538" y="66" width="212" height="120" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="136" y="100">filter in the database</text><text x="390" y="100">transfer what survives</text><text x="644" y="100">finish in process</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="136" y="130">indexed predicate</text><text x="136" y="156">millions become hundreds</text><text x="390" y="130">a small result set</text><text x="390" y="156">the round trip is paid once</text><text x="644" y="130">the awkward part</text><text x="644" y="156">easy to express, small input</text></g><g stroke="#5b6471" stroke-width="2" marker-end="url(#gvp-split-a)"><line x1="244" y1="126" x2="280" y2="126"/><line x1="498" y1="126" x2="534" y2="126"/></g><text x="390" y="228" fill="#1f2937" font-size="13" text-anchor="middle">The mistake this avoids: transferring everything in order to discard nearly all of it</text></svg>
<figcaption><b>This is the shape most real work should take.</b> Neither backend does the whole job well, and the combination does each part where it is cheap — which is why the split route deserves to be a first-class outcome rather than an afterthought.</figcaption>
</figure>

## Recording What the Router Decided

Every routing decision should be recorded with the estimate that produced it, the backend chosen and the measured duration. Three fields, one row, and they answer nearly every question anyone will later ask about the routing.

The immediate use is drift detection. Comparing the estimate against the measurement across a few thousand runs shows whether the thresholds still match the data, and systematic underestimation on one side is the clearest evidence that they do not. Without the record the only signal is a complaint about slowness, which arrives long after the drift began and points at the query rather than at the routing.

The second use is explaining a surprising answer. When two runs of the same question produced different durations, or slightly different results, the recorded backend is usually the explanation — and reconstructing it after the fact from logs that do not contain it is close to impossible. The record also makes the fallback path auditable: a request that ran on the second-choice backend because the first was unavailable is a different event from one routed there deliberately, and only the record distinguishes them.

## Operating This Step Over Time

Re-measure after any material change to the data volume, the hardware or the database version. None of those raise an error when they invalidate a threshold; the system simply starts making worse decisions, at a rate proportional to how far things have moved.

Watch for operations that always route the same way regardless of input. That usually means the estimator has no useful signal for them, which is acceptable — but it should then be an explicit constant with a comment rather than a decision that appears to be dynamic and is not.

The third habit is checking that the split route is still being taken. It is the outcome most easily lost to a well-meaning simplification, because a router reduced to a binary choice still works and still returns reasonable answers — it just stops doing the thing that made the expensive cases fast. A counter on each route makes that regression visible in a way that no test of correctness will, since nothing about the answer changes when the split disappears.

Indexes deserve their own periodic check. An index dropped during a migration, or one made unusable by a change to how the column is queried, silently moves a whole class of work from the fast path to the slow one while the routing decision continues to assume it exists. Asserting index presence at startup, and comparing the catalogue against what the router expects, catches it in seconds rather than in a support conversation.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should the router be allowed to change its mind mid-plan?</span></summary><p>Between steps, yes; within a step, no. Each step is routed on its own inputs, and a step that reduces a million rows to two hundred should hand those two hundred to whichever backend suits the next operation. Abandoning a step part-way to try the other backend is a different thing, and it belongs in the fallback path rather than the routing one.</p></details>

<details class="faq-item"><summary><span>What if the database is under load?</span></summary><p>Then the cost comparison has an extra term, and the honest response is usually to shed rather than to reroute. Sending large work to the in-process path because the database is busy converts a database problem into a memory problem, and the process has far less headroom. Backpressure at the door is the better control.</p></details>

<details class="faq-item"><summary><span>Does this apply to raster work as well?</span></summary><p>The same shape, with different constants. Raster operations have a much steeper transfer cost and a much larger benefit from tiling, which tends to push the crossover far toward the database. The matrix structure transfers; the thresholds must be measured separately, because a raster threshold copied from a vector one will be wrong in the expensive direction.</p></details>

<details class="faq-item"><summary><span>Should the model know which backend is in use?</span></summary><p>No, and it should not be given the choice either. The backend is an implementation detail with respect to the question being answered, and exposing it invites the model to express preferences based on nothing. Record it in the trace, where the people debugging need it.</p></details>

## Related

- Up to the parent topic: [GeoPandas and PostGIS Tool Routing](/geospatial-prompt-engineering-tool-routing/geopandas-postgis-tool-routing/)
- [Routing LLM Calls to GeoPandas vs PostGIS Backends](/geospatial-prompt-engineering-tool-routing/geopandas-postgis-tool-routing/routing-llm-calls-to-geopandas-vs-postgis-backends/)
- Related topic: [Cost and Latency Budgets for Spatial Agents](/geospatial-prompt-engineering-tool-routing/cost-and-latency-budgets-for-spatial-agents/)
- Related topic: [Async vs Sync Geoprocessing Workflows](/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/)
