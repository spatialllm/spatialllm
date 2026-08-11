---
title: Retry and Circuit-Breaker Patterns for Spatial Services
description: Retry only what can succeed, back off with jitter, and open a circuit when a spatial dependency keeps failing so every request stops paying for it.
slug: retry-and-circuit-breaker-patterns-for-spatial-services
type: howto
breadcrumb: Retry and Circuit Breakers
datePublished: 2025-03-13
dateModified: 2026-08-11
---

# Retry and Circuit-Breaker Patterns for Spatial Services

Blindly retrying a failed spatial call is worse than not retrying at all when the failure is a self-intersecting polygon: the geometry is invalid on attempt one and still invalid on attempt five, so every retry burns a connection and hides the real error. This guide separates transient spatial faults from permanent ones and wraps them in a retry-with-jitter plus a circuit breaker that trips on the topology-failure rate. It belongs to [error mapping for spatial API calls](/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/) and governs the resilience layer between an agent and its geoprocessing backend.

The decisive move is classification. A statement timeout, a dropped socket, or a pool-exhaustion error is *transient* — the same input may succeed moments later, so backoff-and-retry is correct. A GEOS topology exception, an invalid WKT, or a CRS mismatch is *permanent* for that input — retrying only repeats the fault. Conflating the two produces retry storms that amplify load exactly when the service is already unhealthy.

## When to Use This Approach

Use a retry decorator on any call that crosses a network or contends for a pooled resource. Add a circuit breaker when repeated failures signal a systemic problem — a wedged replica, a corrupt input batch from an upstream model — where continuing to call is actively harmful. Skip retries entirely for pure validation errors: surface them immediately so the caller can re-prompt or repair.

| Error class | Example | Strategy |
|---|---|---|
| Transient | Statement timeout, connection reset, pool exhausted | Backoff + jitter, bounded attempts |
| Permanent | GEOS topology error, invalid WKT, CRS mismatch | Fail fast, no retry, return mapped error |
| Systemic | Sustained topology-failure rate, replica down | Open circuit, shed to fallback |

The breaker specifically watches the *rate* of permanent topology failures, not just transient ones. A sudden spike of `TopologyException` across many inputs usually means an upstream generator started emitting garbage geometries; opening the circuit stops the flood and gives operators a clean signal. For turning the resulting errors into readable guidance, see [mapping spatial API errors to user-friendly prompts](/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/mapping-spatial-api-errors-to-user-friendly-prompts/).

<figure class="diagram">
<svg viewBox="16 32 748 214" role="img" aria-labelledby="rcb-class-t rcb-class-d" xmlns="http://www.w3.org/2000/svg"><title id="rcb-class-t">What each failure class deserves</title><desc id="rcb-class-d">Only transient failures benefit from a retry; capability, input and fatal failures retry identically forever and consume budget that could have been spent on an alternative route.</desc><rect x="16" y="32" width="748" height="214" fill="#ffffff"/><rect x="30" y="46" width="360" height="86" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="52" y="76" fill="#1f2937" font-size="13" font-weight="600">transient</text><text x="52" y="102" fill="#5b6471" font-size="12">a timeout, a blip, a rate limit</text><text x="52" y="122" fill="#5b6471" font-size="12">one retry with jitter</text><rect x="410" y="46" width="340" height="86" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="432" y="76" fill="#1f2937" font-size="13" font-weight="600">capability</text><text x="432" y="102" fill="#5b6471" font-size="12">this route cannot serve this input</text><text x="432" y="122" fill="#5b6471" font-size="12">no retry — degrade</text><rect x="30" y="146" width="360" height="86" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="52" y="176" fill="#1f2937" font-size="13" font-weight="600">input</text><text x="52" y="202" fill="#5b6471" font-size="12">the request itself is wrong</text><text x="52" y="222" fill="#5b6471" font-size="12">no retry — correct it</text><rect x="410" y="146" width="340" height="86" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="432" y="176" fill="#1f2937" font-size="13" font-weight="600">fatal</text><text x="432" y="202" fill="#5b6471" font-size="12">not permitted, not configured</text><text x="432" y="222" fill="#5b6471" font-size="12">no retry — stop</text></svg>
<figcaption><b>Three of these four must not be retried.</b> A retry policy that does not classify spends three attempts confirming a failure it had enough information to recognise on the first.</figcaption>
</figure>

## Implementation

The decorator below classifies exceptions, retries only transient ones with exponential backoff and full jitter, and feeds a circuit breaker whose window trips on either a global failure ratio or a topology-error surge. When the circuit is open, calls return a deterministic fallback without touching the backend.

```python
import asyncio
import logging
import random
import time
from collections import deque
from functools import wraps

log = logging.getLogger("spatial_resilience")

# Substring markers that identify a permanent, non-retryable spatial fault.
_PERMANENT = ("topologyexception", "invalid wkt", "geos", "crs mismatch",
              "non-noded intersection", "ring self-intersection")


def classify(exc: Exception) -> str:
    msg = str(exc).lower()
    if any(tok in msg for tok in _PERMANENT):
        return "permanent"
    if isinstance(exc, (asyncio.TimeoutError, ConnectionError, OSError)):
        return "transient"
    # Default to transient for unknown DB errors but cap attempts tightly.
    return "transient"


class CircuitOpen(Exception):
    """Raised when the breaker is open and calls are shed."""


class Breaker:
    def __init__(self, window: int = 50, fail_ratio: float = 0.5,
                 topo_trip: int = 10, cooldown_s: float = 15.0):
        self._events: deque[tuple[float, str]] = deque(maxlen=window)
        self._fail_ratio = fail_ratio
        self._topo_trip = topo_trip
        self._cooldown_s = cooldown_s
        self._opened_at: float | None = None

    def allow(self) -> bool:
        if self._opened_at is None:
            return True
        if time.monotonic() - self._opened_at >= self._cooldown_s:
            self._opened_at = None      # half-open: let one probe through
            self._events.clear()
            return True
        return False

    def record(self, outcome: str) -> None:
        self._events.append((time.monotonic(), outcome))
        fails = sum(1 for _, o in self._events if o != "ok")
        topo = sum(1 for _, o in self._events if o == "permanent")
        if self._events.maxlen and len(self._events) >= self._events.maxlen:
            if fails / len(self._events) >= self._fail_ratio or topo >= self._topo_trip:
                self._opened_at = time.monotonic()
                log.error("circuit opened: fails=%d topo=%d", fails, topo)


def resilient(breaker: Breaker, *, attempts: int = 4, base: float = 0.2,
              cap: float = 4.0, fallback=None):
    def decorate(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            if not breaker.allow():
                log.warning("circuit open; shedding %s", fn.__name__)
                return fallback
            last: Exception | None = None
            for attempt in range(attempts):
                try:
                    result = await fn(*args, **kwargs)
                    breaker.record("ok")
                    return result
                except Exception as exc:  # noqa: BLE001 - classified below
                    kind = classify(exc)
                    breaker.record(kind)
                    last = exc
                    if kind == "permanent":
                        log.info("permanent spatial fault; no retry: %s", exc)
                        return fallback
                    sleep = min(cap, base * 2 ** attempt) * random.random()
                    log.warning("transient (%d/%d), backoff %.2fs: %s",
                                attempt + 1, attempts, sleep, exc)
                    await asyncio.sleep(sleep)
            log.error("exhausted retries: %s", last)
            return fallback
        return wrapper
    return decorate


_breaker = Breaker()


@resilient(_breaker, fallback=[])
async def nearby_hydrants(pool, wkt: str, radius_m: float):
    async with pool.acquire() as conn:
        # && bbox pre-filter uses the GiST index before the exact ST_DWithin test.
        return await conn.fetch(
            """
            SELECT h.id
            FROM hydrants h
            WHERE h.geom && ST_Expand(ST_GeomFromText($1, 3857), $2)
              AND ST_DWithin(h.geom, ST_GeomFromText($1, 3857), $2)
            """,
            wkt, radius_m,
        )
```

Full jitter (`base * 2**attempt * random()`) is deliberate: it spreads retries across the backoff window so a fleet of clients does not re-collide in synchronized waves. The fallback (`[]`) lets the caller degrade to a cached or empty answer, consistent with [implementing fallback routing for failed spatial queries](/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/implementing-fallback-routing-for-failed-spatial-queries/).

<figure class="diagram">
<svg viewBox="16 38 728 162" role="img" aria-labelledby="rcb-jitter-t rcb-jitter-d" xmlns="http://www.w3.org/2000/svg"><title id="rcb-jitter-t">Synchronised retries against jittered ones</title><desc id="rcb-jitter-d">Clients that retry on the same schedule arrive together and keep a recovering service down; adding randomness spreads the load across the recovery window.</desc><rect x="16" y="38" width="728" height="162" fill="#ffffff"/><rect x="30" y="52" width="700" height="46" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="50" y="80" fill="#1f2937" font-size="12.5">no jitter: every client retries at t+1, t+2, t+4 — arriving together</text><rect x="30" y="108" width="700" height="46" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="50" y="136" fill="#1f2937" font-size="12.5">full jitter: retries spread across the window — the service recovers</text><text x="390" y="182" fill="#1f2937" font-size="13" text-anchor="middle">A recovering dependency is knocked over by the retries of the clients waiting for it</text></svg>
<figcaption><b>Jitter is not a refinement.</b> Without it, a dependency that comes back up receives every waiting client simultaneously and goes down again, which is indistinguishable from it never having recovered.</figcaption>
</figure>

## Validation & Testing

- **Permanent faults are not retried.** Stub `fn` to raise `Exception("TopologyException: side location conflict")` and assert it is called exactly once and returns the fallback — no backoff sleeps occur.
- **Transient faults back off then succeed.** Have the stub fail with `asyncio.TimeoutError` twice then return a value; assert three total calls and that recorded sleeps are non-decreasing in expectation across attempts.
- **Breaker opens on topology surge.** Feed the breaker `topo_trip` permanent outcomes within one window and assert `allow()` returns `False`, then advance a monotonic clock past `cooldown_s` and assert it half-opens and permits a single probe.

## Gotchas & Edge Cases

- **String matching is brittle.** Classifying by message substring breaks across driver and GEOS versions. Prefer matching on `SQLSTATE`/exception subclasses where available and treat the substring list as a fallback heuristic, logging any unclassified error for review.
- **Retrying non-idempotent writes.** Backoff on an `INSERT` that partially committed can duplicate rows. Only decorate idempotent reads or writes guarded by `ON CONFLICT`; never blind-retry a bare mutation.
- **Half-open thundering herd.** When the breaker half-opens, allowing many probes at once can immediately re-trip it. Admit exactly one probe (as above) and reopen on its failure before letting general traffic resume.
- **Global breaker hides per-shard health.** One breaker across many databases trips on the busiest shard and starves the healthy ones. Key a breaker per backend endpoint so isolation is local.

<figure class="diagram">
<svg viewBox="0 56 780 184" role="img" aria-labelledby="rcb-states-t rcb-states-d" xmlns="http://www.w3.org/2000/svg"><title id="rcb-states-t">The three circuit states and what moves between them</title><desc id="rcb-states-d">A closed circuit passes traffic and counts failures, an open circuit skips the route entirely, and a half-open circuit lets one request through to test whether the dependency has recovered.</desc><rect x="0" y="56" width="780" height="184" fill="#ffffff"/><rect x="30" y="70" width="220" height="110" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="280" y="70" width="220" height="110" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="530" y="70" width="220" height="110" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="140" y="102">closed</text><text x="390" y="102">open</text><text x="640" y="102">half open</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="140" y="132">traffic passes</text><text x="140" y="158">failures counted</text><text x="390" y="132">route skipped entirely</text><text x="390" y="158">no request pays for it</text><text x="640" y="132">one request tests it</text><text x="640" y="158">success closes, failure reopens</text></g><text x="390" y="222" fill="#1f2937" font-size="13" text-anchor="middle">Only transient and capability failures move the counter — input errors say nothing about the service</text></svg>
<figcaption><b>The half-open state is what makes recovery cheap.</b> Without it the circuit either stays open too long or reopens under a flood of simultaneous probes, and both look like the dependency never came back.</figcaption>
</figure>

## Operating This Step Over Time

A circuit that has never opened is telling you its threshold is too high, and one that opens weekly is telling you about a dependency. Both are findings, and neither is visible unless openings are counted and reviewed — a circuit is one of the few controls whose non-firing is as informative as its firing.

Cool-off periods drift toward the long end after every incident, because a longer cool-off makes a recurring problem quieter. The cost is that a brief blip now takes the full cool-off to recover from, and users experience that as the outage rather than the blip. Reviewing the cool-off against observed recovery times, rather than against how the last incident felt, keeps it proportionate.

Watch for retries that survive a classification change. An error that was transient becomes permanent when a service is retired, and a retry policy that still treats it as worth attempting spends three attempts on every affected request indefinitely. Reviewing recovery outcomes — how often each class actually recovered after a retry — is the check that catches it.

## Frequently Asked Questions

<details class="faq-item"><summary><span>How many retries are worth attempting?</span></summary><p>One, in almost every interactive case. A second retry succeeds where the first failed only when the failure was very brief, and it costs a full attempt's latency out of a budget that has already lost one. In batch contexts where nobody is waiting, two or three with exponential backoff are reasonable — the constraint there is total throughput rather than a user's patience.</p></details>

<details class="faq-item"><summary><span>Should the circuit be per dependency or per operation?</span></summary><p>Per dependency, usually, because that is the thing that fails. A per-operation circuit will keep sending traffic to a dead service through the operations that have not yet accumulated failures, and a per-dependency one stops all of it at once. The exception is a dependency where one operation is genuinely much more fragile than the others, which is worth its own circuit rather than dragging the healthy operations down with it.</p></details>

<details class="faq-item"><summary><span>What should a request see when the circuit is open?</span></summary><p>The same thing it would see from a capability failure — a degradation to the next route, or a refusal naming the unavailable dependency. An open circuit is not an error to be reported to the user as such; it is a routing fact, and the user's experience should be the fallback rather than a message about internal state.</p></details>

<details class="faq-item"><summary><span>Does a circuit breaker replace a timeout?</span></summary><p>No — the timeout bounds one request and the circuit bounds the aggregate. A dependency that is slow rather than down will pass every timeout individually while consuming the entire budget of every request that touches it, and only the circuit stops that. Set both, and count a timeout as a failure for the circuit's purposes.</p></details>

## Related

- Up to the section: [Error Mapping for Spatial API Calls](/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/)
- [Mapping Spatial API Errors to User-Friendly Prompts](/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/mapping-spatial-api-errors-to-user-friendly-prompts/)
- [Backpressure and Rate Limiting for Spatial API Calls](/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/backpressure-and-rate-limiting-for-spatial-api-calls/)
- [Implementing Fallback Routing for Failed Spatial Queries](/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/implementing-fallback-routing-for-failed-spatial-queries/)
