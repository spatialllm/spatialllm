---
title: Classifying PostGIS Errors for Agent Recovery
description: Turn PostGIS error codes and GEOS exception text into four recovery classes, so an agent retries what can recover, re-plans what it caused, and stops on the rest.
slug: classifying-postgis-errors-for-agent-recovery
type: howto
breadcrumb: Classifying PostGIS Errors
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Classifying PostGIS Errors for Agent Recovery

An agent that receives a database error and decides what to do by reading its text will decide differently for the same failure phrased two ways. Classification fixes that: every error becomes one of four classes before anything else happens, and the class — not the message — drives the response. This guide covers the mapping for PostGIS specifically, and it is the concrete half of [error mapping for spatial API calls](/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/).

## When to Use This Approach

Any agent that calls PostGIS more than trivially needs this, because the error surface is genuinely varied: SQLSTATE codes from the server, GEOS exception strings from the geometry engine, connection failures from the driver, and planner refusals that arrive as ordinary errors.

| Source | Example | Class |
|--------|---------|-------|
| Connection lost, statement timeout | `57014`, `08006` | Transient |
| Undefined function, no index | `42883`, planner cost refusal | Capability |
| Invalid geometry, SRID mismatch | `XX000` with GEOS text, `22023` | Input |
| Insufficient privilege, syntax error | `42501`, `42601` | Fatal |

<figure class="diagram">
<svg viewBox="16 38 748 212" role="img" aria-labelledby="cpe-two-t cpe-two-d" xmlns="http://www.w3.org/2000/svg"><title id="cpe-two-t">Classifying on codes against matching on message text</title><desc id="cpe-two-d">SQLSTATE codes are stable across versions and locales while message text changes with both, so a classifier built on text silently stops working after an upgrade.</desc><rect x="16" y="38" width="748" height="212" fill="#ffffff"/><rect x="30" y="52" width="340" height="160" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="410" y="52" width="340" height="160" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="200" y="84">classify on the code</text><text x="580" y="84">match on the message</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="200" y="114">stable across versions</text><text x="200" y="140">unaffected by locale</text><text x="200" y="166">a new code is unknown, not wrong</text><text x="580" y="114">wording changes on upgrade</text><text x="580" y="140">translated in some deployments</text><text x="580" y="166">a near miss is misclassified</text></g><text x="390" y="234" fill="#1f2937" font-size="13" text-anchor="middle">A text classifier fails silently — it keeps returning a class, just the wrong one</text></svg>
<figcaption><b>The silent failure is the problem.</b> A code-based classifier that meets something new returns unknown and can be made to escalate; a text-based one returns whichever pattern happened to match and the agent proceeds confidently.</figcaption>
</figure>

## Implementation

Start from SQLSTATE, which the driver exposes directly and which is stable. The unavoidable exception is `XX000`, the internal-error code under which PostGIS reports most GEOS failures, and which therefore has to be refined by looking at the message. That refinement is contained to one code rather than being the whole strategy.

```python
TRANSIENT = {"08000", "08003", "08006", "08007", "40001", "40P01", "53300", "57014"}
CAPABILITY = {"42883", "42P01", "0A000", "53400"}
FATAL = {"42501", "42601", "28000", "3D000"}

GEOS_INPUT = (
    "self-intersection", "ring self-intersection", "too few points",
    "invalid number of points", "non-closed ring", "geometry is invalid",
)


def classify(exc) -> str:
    code = getattr(exc, "sqlstate", None) or getattr(exc, "pgcode", None)
    if code in TRANSIENT:
        return "transient"
    if code in CAPABILITY:
        return "capability"
    if code in FATAL:
        return "fatal"
    if code in ("XX000", "22023"):
        text = str(exc).lower()
        if any(marker in text for marker in GEOS_INPUT):
            return "input"
        if "srid" in text or "mixed srid" in text:
            return "input"
    return "unknown"
```

The `unknown` return is the important part. An unrecognised error must not be quietly assumed transient, because that is the assumption that produces three retries against something that will never work. Treat unknown as fatal for control-flow purposes and log it prominently — the log entry is how the table grows.

```python
def handle(exc, step, budget):
    cls = classify(exc)
    if cls == "unknown":
        log.error("unclassified database error", extra={"sqlstate": exc.sqlstate, "step": step.name})
        cls = "fatal"
    return RECOVERY[cls](step, budget)
```

<figure class="diagram">
<svg viewBox="13 32 751 224" role="img" aria-labelledby="cpe-four-t cpe-four-d" xmlns="http://www.w3.org/2000/svg"><title id="cpe-four-t">What each class does next</title><desc id="cpe-four-d">Transient errors retry once with jitter, capability errors route elsewhere, input errors return to the planner for correction, and fatal errors stop the chain and report.</desc><rect x="13" y="32" width="751" height="224" fill="#ffffff"/><g><rect x="30" y="46" width="172" height="150" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="216" y="46" width="172" height="150" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="402" y="46" width="172" height="150" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="588" y="46" width="162" height="150" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/></g><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="116" y="80">transient</text><text x="302" y="80">capability</text><text x="488" y="80">input</text><text x="669" y="80">fatal</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="116" y="112">timeout, deadlock</text><text x="116" y="138">retry once, jittered</text><text x="116" y="170">count toward the circuit</text><text x="302" y="112">no function, no index</text><text x="302" y="138">route to the other backend</text><text x="302" y="170">or degrade the answer</text><text x="488" y="112">invalid geometry, SRID</text><text x="488" y="138">return to the planner</text><text x="488" y="170">the model can correct it</text><text x="669" y="112">privilege, syntax</text><text x="669" y="138">stop and report</text><text x="669" y="170">this is a defect</text></g><text x="390" y="238" fill="#1f2937" font-size="13" text-anchor="middle">Only the input class involves the model — the other three are decided in code</text></svg>
<figcaption><b>One class in four goes back to the model.</b> Invalid geometry and reference-system mismatches are the failures a re-plan can genuinely fix; the rest are routing and infrastructure decisions the model has no information about.</figcaption>
</figure>

## Validation & Testing

Test the classifier against real errors rather than constructed ones. The reliable way to do that is to provoke each class against a scratch database — a deliberately invalid polygon, a query on a dropped table, a statement timeout set to one millisecond — and assert the class rather than the message.

```python
@pytest.mark.parametrize("sql,expected", [
    ("SELECT ST_Area(ST_GeomFromText('POLYGON((0 0,1 1,1 0,0 1,0 0))'))", "input"),
    ("SELECT ST_NotARealFunction(1)", "capability"),
    ("SELECT * FROM table_that_does_not_exist", "capability"),
    ("SET statement_timeout='1ms'; SELECT pg_sleep(1)", "transient"),
])
def test_classification(scratch_db, sql, expected):
    with pytest.raises(Exception) as caught:
        scratch_db.execute(sql)
    assert classify(caught.value) == expected
```

The other test worth having asserts that the unknown path is reachable and behaves safely: feed the classifier an exception with a code that is in none of the sets and confirm it returns `unknown` rather than falling through to a default.

## Gotchas & Edge Cases

**`XX000` covering everything.** PostGIS reports a large and varied set of geometry problems under one internal-error code, which is why the message refinement exists at all. Keep that refinement narrow and let anything unmatched fall to `unknown`, rather than assuming an unrecognised `XX000` is an input problem.

**Statement timeouts classified as transient forever.** A timeout is transient the first time and a capability problem the third, because a query that consistently exceeds its limit is not going to stop. Tracking timeouts per query shape and reclassifying repeat offenders is what stops the retry budget disappearing into one bad statement.

**Serialisation failures at high concurrency.** Code `40001` is genuinely transient and genuinely common under load, and retrying it is correct. Retrying it without jitter under load is how a contention problem becomes an outage.

**Connection errors that are actually capacity.** A pool exhaustion error looks like a connection failure and is really backpressure. Classifying it transient produces retries that make the exhaustion worse; treating it as capability, with a wait, is closer to the truth.

**Errors raised after partial work.** A failure part-way through a multi-statement operation leaves state behind unless the whole thing was in a transaction. The classification says what to do next; it says nothing about cleaning up, and that has to be handled separately.

<figure class="diagram">
<svg viewBox="16 38 735 212" role="img" aria-labelledby="cpe-unk-t cpe-unk-d" xmlns="http://www.w3.org/2000/svg"><title id="cpe-unk-t">How the unknown class should behave</title><desc id="cpe-unk-d">An unrecognised error treated as transient produces wasted retries, treated as fatal it stops safely and produces a log entry that grows the table.</desc><rect x="16" y="38" width="735" height="212" fill="#ffffff"/><rect x="30" y="52" width="700" height="46" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="50" y="80" fill="#1f2937" font-size="12.5">unknown treated as fatal: stops safely, logs loudly, the table grows next week</text><rect x="30" y="108" width="600" height="46" rx="6" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="50" y="136" fill="#1f2937" font-size="12.5">unknown treated as capability: degrades quietly, the log entry is missed</text><rect x="30" y="164" width="520" height="46" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="50" y="192" fill="#1f2937" font-size="12.5">unknown treated as transient: three retries, then the same failure</text><text x="390" y="238" fill="#1f2937" font-size="13" text-anchor="middle">The default for an unrecognised error is the most consequential line in the classifier</text></svg>
<figcaption><b>Defaults decide what happens to everything you did not anticipate.</b> Since the unanticipated cases are by definition the ones you know least about, the default should be the response that spends the least and complains the loudest.</figcaption>
</figure>

## Keeping the Table Honest

A classification table is a set of claims about how a system behaves, and claims decay. The way to keep it honest is to make every entry traceable to the observation that justified it — a code added because it was seen in production, with a link to the incident, rather than one copied from a reference because it looked relevant.

That discipline pays off in two places. When a code stops appearing entirely, the entry can be retired instead of being carried forward indefinitely on the strength of a documentation page. And when a code behaves differently from what the table says, the original observation gives a starting point for working out what changed — a driver version, a pooler, a server setting — rather than leaving the discrepancy as an unexplained oddity.

The table should also be small enough to read. A classifier with two hundred entries is one nobody reviews, and most of those entries will be codes seen once in a synthetic test. Four sets of a dozen or so, covering what actually happens, plus a well-behaved unknown path, handles real traffic better than exhaustive coverage of codes that never arrive.

Finally, keep the sets in one place rather than distributed across the call sites that use them. A classification duplicated into three modules diverges within a release, and the divergence is invisible until two parts of the system respond differently to the same failure.

## Operating This Step Over Time

Review the unclassified log weekly at first and monthly once it settles. Every entry is either a code that belongs in a set or a genuine defect, and both are worth ten minutes. A classifier that has produced no unknowns in six months is either complete or no longer being exercised, and it is worth knowing which.

Track recovery outcomes per class, not just counts. The number that matters for the transient set is how often a retry actually succeeded: a code that recovers less than half the time is misclassified, whatever the documentation says about it. That measurement is the only real evidence for whether the sets are right.

Re-check the sets after a major PostgreSQL or PostGIS upgrade. Codes are stable, and the mapping from a given failure to a given code occasionally is not — particularly around the internal-error refinements, which depend on message text that upgrades are free to change.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Why not use the driver's exception hierarchy?</span></summary><p>Because it groups by SQL semantics rather than by what an agent should do. A driver may raise the same exception class for a missing table and a missing column, which are both capability problems, and a different one for a deadlock and a timeout, which are both transient. The four recovery classes cut across the hierarchy, so the mapping has to be explicit — though the hierarchy is a reasonable source for the initial sets.</p></details>

<details class="faq-item"><summary><span>Should the model see the original error text?</span></summary><p>For the input class, a cleaned version helps — knowing that a ring self-intersects is genuinely useful for re-planning. For the other three it should see the class and nothing more, because there is no correction to make and the raw text will invite it to invent one. This is the same reason the reader sees a derived message rather than the original.</p></details>

<details class="faq-item"><summary><span>What about errors from GeoPandas rather than PostGIS?</span></summary><p>Map them into the same four classes, so the recovery logic is shared. The sources differ — exception types rather than SQLSTATE codes — and the classes do not, which means a plan that falls back from one backend to the other keeps consistent behaviour. Two classifiers feeding one recovery table is the right shape here.</p></details>

<details class="faq-item"><summary><span>How does classification interact with the circuit breaker?</span></summary><p>Only transient and capability failures should move the circuit's counter. Input errors say something about the request and nothing about the service, and counting them will open a circuit because a user asked for something malformed — which then denies the service to everyone else for the cool-off period.</p></details>

## Related

- Up to the parent topic: [Error Mapping for Spatial API Calls](/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/)
- [Retry and Circuit Breaker Patterns for Spatial Services](/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/retry-and-circuit-breaker-patterns-for-spatial-services/)
- [Mapping Spatial API Errors to User-Friendly Prompts](/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/mapping-spatial-api-errors-to-user-friendly-prompts/)
- Related topic: [Topology Rule Enforcement via LLMs](/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/)
