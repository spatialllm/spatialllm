---
title: Backpressure and Rate Limiting for Spatial API Calls
description: Keep a spatial API from being overwhelmed — concurrency limits, token buckets, queue depth signals and a rejection that tells the caller when to come back.
slug: backpressure-and-rate-limiting-for-spatial-api-calls
type: howto
breadcrumb: Backpressure and Rate Limiting
datePublished: 2025-03-05
dateModified: 2026-08-11
---

# Backpressure and Rate Limiting for Spatial API Calls

A single unbounded burst of LLM-triggered spatial calls can drain a PostGIS connection pool in seconds, turning a healthy geoprocessing service into a queue of timed-out transactions. This guide shows how to place explicit backpressure — a token bucket plus a bounded semaphore — in front of expensive spatial operations so an agent sheds load gracefully instead of collapsing. It sits inside [async vs sync geoprocessing workflows](/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/) and targets the admission-control stage that guards every downstream `ST_*` call.

The core problem is fan-out amplification. An agent that decomposes one user question into a dozen buffer, intersect, and union calls will happily dispatch all of them at once. Each call may acquire a database connection, hold it while GEOS grinds through a heavy polygon, and only release it on commit. Without an admission gate, concurrency is bounded by nothing but the agent's imagination, and the pool — a scarce, fixed resource — becomes the failure point.

## When to Use This Approach

Reach for token-bucket backpressure when calls are individually expensive and the protected resource has a hard concurrency ceiling (a pool `max_size`, a licensed geocoder quota, a tile server's rate limit). Reach for a plain semaphore when you only need to cap *simultaneous* work. In practice you want both: the semaphore caps concurrency, the bucket caps sustained rate, and a timeout caps latency.

| Control | Bounds | Best for | Failure signal |
|---|---|---|---|
| Semaphore | Concurrent in-flight calls | Pool / worker protection | Acquire blocks |
| Token bucket | Calls per unit time | Quota / rate-limited APIs | No token available |
| Timeout + shed | Tail latency | User-facing agents | Deadline exceeded |

If your workload is cheap and read-only against an indexed table, a semaphore alone is enough. If calls hit an external quota, the bucket is mandatory — a semaphore lets a low-latency endpoint exceed a per-second cap. When latency budgets matter, always pair admission with a deadline so a saturated system returns a deterministic shed response rather than queueing forever. For the execution model these controls wrap, see [handling async spatial processing in Python workflows](/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/handling-async-spatial-processing-in-python-workflows/).

<figure class="diagram">
<svg viewBox="16 38 748 188" role="img" aria-labelledby="bp-two-t bp-two-d" xmlns="http://www.w3.org/2000/svg"><title id="bp-two-t">Shedding load at the door against absorbing it</title><desc id="bp-two-d">A service that accepts everything degrades for every caller at once, while one that rejects at a stated limit keeps the accepted work fast and tells the rest when to return.</desc><rect x="16" y="38" width="748" height="188" fill="#ffffff"/><rect x="30" y="52" width="340" height="160" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="410" y="52" width="340" height="160" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="200" y="84">reject at the limit</text><text x="580" y="84">accept everything</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="200" y="114">accepted work stays fast</text><text x="200" y="140">callers get a retry time</text><text x="200" y="166">the limit is visible</text><text x="580" y="114">latency grows for everyone</text><text x="580" y="140">no signal to back off</text><text x="580" y="166">failure looks like an outage</text></g></svg>
<figcaption><b>Rejection is a service, not a failure.</b> A caller told to come back in four seconds can plan; one whose request is quietly queued behind two thousand others cannot, and neither can the operator.</figcaption>
</figure>

## Implementation

The limiter below combines a refill-on-demand token bucket with an `asyncio.Semaphore` sized to the pool. Every guarded call has an admission deadline; if it cannot acquire a token and a slot in time, it sheds load and returns a deterministic fallback instead of touching the database.

```python
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

log = logging.getLogger("spatial_backpressure")


class LoadShed(Exception):
    """Raised when a call cannot be admitted within its deadline."""


@dataclass
class TokenBucket:
    rate: float          # tokens added per second
    capacity: float      # max burst size
    _tokens: float = 0.0
    _last: float = 0.0

    def __post_init__(self) -> None:
        self._tokens = self.capacity
        self._last = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last = now

    def try_take(self) -> bool:
        self._refill()
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


class SpatialAdmission:
    def __init__(self, pool_size: int, rate: float, burst: float):
        self._slots = asyncio.Semaphore(pool_size)
        self._bucket = TokenBucket(rate=rate, capacity=burst)

    async def run(
        self,
        op: Callable[[], Awaitable[Any]],
        *,
        deadline_s: float,
        fallback: Any,
    ) -> Any:
        start = time.monotonic()
        # 1. Rate gate: poll the bucket until a token frees up or the deadline passes.
        while not self._bucket.try_take():
            if time.monotonic() - start > deadline_s:
                log.warning("shed: no token within %.2fs", deadline_s)
                return fallback
            await asyncio.sleep(0.02)

        # 2. Concurrency gate: bounded wait for a pool slot.
        remaining = deadline_s - (time.monotonic() - start)
        try:
            await asyncio.wait_for(self._slots.acquire(), timeout=max(remaining, 0.0))
        except asyncio.TimeoutError:
            log.warning("shed: no pool slot within deadline")
            return fallback

        # 3. Execute under the slot with a residual deadline.
        try:
            residual = deadline_s - (time.monotonic() - start)
            return await asyncio.wait_for(op(), timeout=max(residual, 0.01))
        except asyncio.TimeoutError:
            log.error("spatial op exceeded deadline; returning fallback")
            return fallback
        except Exception:
            log.exception("spatial op failed; returning fallback")
            return fallback
        finally:
            self._slots.release()


async def demo(pool, admission: SpatialAdmission, wkt: str):
    async def op():
        async with pool.acquire() as conn:
            # Bbox pre-filter (&&) narrows candidates via the GiST index before ST_Intersects.
            return await conn.fetch(
                """
                SELECT p.id
                FROM parcels p
                WHERE p.geom && ST_GeomFromText($1, 4326)
                  AND ST_Intersects(p.geom, ST_GeomFromText($1, 4326))
                """,
                wkt,
            )

    return await admission.run(op, deadline_s=1.5, fallback=[])
```

The `fallback` is deterministic: an empty result the caller can treat as "no admitted answer" and route to a cached tier or a user-facing "system busy" message. Because shedding happens *before* `pool.acquire()`, an overloaded system never deepens its own backlog.

<figure class="diagram">
<svg viewBox="16 38 748 188" role="img" aria-labelledby="bp-shape-t bp-shape-d" xmlns="http://www.w3.org/2000/svg"><title id="bp-shape-t">Concurrency limit against token bucket</title><desc id="bp-shape-d">A concurrency limit bounds how much work is in flight and suits expensive operations; a token bucket bounds the arrival rate and suits cheap, bursty ones.</desc><rect x="16" y="38" width="748" height="188" fill="#ffffff"/><rect x="30" y="52" width="340" height="160" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="410" y="52" width="340" height="160" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="200" y="84">concurrency limit</text><text x="580" y="84">token bucket</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="200" y="114">bounds work in flight</text><text x="200" y="140">protects memory and CPU</text><text x="200" y="166">right for geoprocessing</text><text x="580" y="114">bounds the arrival rate</text><text x="580" y="140">absorbs short bursts</text><text x="580" y="166">right for cheap lookups</text></g></svg>
<figcaption><b>They bound different things.</b> A token bucket will happily admit fifty simultaneous overlays if they arrive slowly enough, and a concurrency limit will happily admit a sustained flood of cheap calls.</figcaption>
</figure>

## Validation & Testing

- **Concurrency ceiling holds.** Launch 200 `admission.run` calls against a pool of size 8 with a slow stub `op` and assert the observed peak of concurrent `op` entries never exceeds 8 (increment a counter on entry, decrement on exit, track the max).
- **Sustained rate is capped.** With `rate=10, burst=10`, drive 100 calls and assert wall-clock elapsed is at least `(100 - burst) / rate` seconds — proof the bucket throttled the tail rather than admitting instantly.
- **Shed is deterministic under saturation.** Set `deadline_s` below the stub op latency and assert every over-limit call returns exactly the `fallback` value and that zero of them invoked `pool.acquire()` (patch the pool and assert the mock's call count equals only the admitted count).

## Gotchas & Edge Cases

- **Monotonic clock only.** Refill uses `time.monotonic()`, never `time.time()`. An NTP step or leap adjustment on wall-clock time can hand out a burst of phantom tokens or freeze refills; the monotonic clock is immune.
- **Slot leak on cancellation.** If the task is cancelled between `acquire()` and the `finally`, the slot must still release. Keep the acquire and the `try/finally` in the same coroutine frame — never split them across `await` boundaries where cancellation can slip in.
- **Deadline smaller than sleep granularity.** A `deadline_s` near the 0.02 s poll interval can shed calls that a token would have covered milliseconds later. Set deadlines at least an order of magnitude above the poll step, or lower the sleep for latency-critical paths.
- **Bucket sized larger than the pool.** A generous burst lets many tokened calls stampede the semaphore, so they simply queue on `acquire()`. Keep `burst` close to `pool_size` so the two gates agree on the true concurrency limit.

<figure class="diagram">
<svg viewBox="16 38 728 212" role="img" aria-labelledby="bp-signal-t bp-signal-d" xmlns="http://www.w3.org/2000/svg"><title id="bp-signal-t">What a rejection should carry</title><desc id="bp-signal-d">A useful rejection states the limit that was hit, the current depth and a retry time, so the caller can back off intelligently rather than immediately trying again.</desc><rect x="16" y="38" width="728" height="212" fill="#ffffff"/><rect x="30" y="52" width="700" height="46" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="50" y="80" fill="#1f2937" font-size="12.5">limit reached: 64 concurrent operations</text><rect x="30" y="108" width="620" height="46" rx="6" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="50" y="136" fill="#1f2937" font-size="12.5">current depth: 71 — retry after about 4 seconds</text><rect x="30" y="164" width="520" height="46" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="50" y="192" fill="#1f2937" font-size="12.5">bare 503 with no detail — the caller retries at once</text><text x="390" y="238" fill="#1f2937" font-size="13" text-anchor="middle">A rejection with no retry hint produces a retry storm from well-behaved clients</text></svg>
<figcaption><b>The retry hint is the whole point.</b> Without it every polite client becomes an impolite one, because immediate retry is the only strategy available to a caller that has been told nothing.</figcaption>
</figure>

## Operating This Step Over Time

Limits set against one machine size become wrong the moment the deployment changes, and nothing in the code notices — the same limit simply admits too little or too much. Deriving the concurrency limit from available memory and the measured cost of a typical operation, rather than from a constant, makes it move with the deployment instead of against it.

The signal worth watching is the rejection rate alongside the queue depth, not either alone. A short queue with a high rejection rate means the limit is too tight; a deep queue with none means it is too loose and the degradation is happening inside rather than at the door. Publishing both together is what makes the difference legible.

Watch also for callers that ignore the retry hint. A client retrying immediately after a rejection converts a load-shedding mechanism into an amplifier, and the fix is usually in the client rather than in the limit — which is only discoverable if the rejection responses are counted per caller.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should the limit be global or per caller?</span></summary><p>Both, with the per-caller limit lower. A global limit protects the service and a per-caller limit stops one client consuming the whole allowance, which is the failure that makes a shared service feel unreliable to everyone except the client causing it. The per-caller limit can be generous — a fraction of the global rather than a small fixed number — and still prevents the pathological case.</p></details>

<details class="faq-item"><summary><span>What should the retry hint be based on?</span></summary><p>The observed completion rate and the current depth, not a constant. Telling a caller to retry in four seconds when the queue will take forty produces four failed retries instead of one useful wait. Computing it from depth divided by throughput is arithmetic the service already has the inputs for, and it makes the hint honest enough to be worth obeying.</p></details>

<details class="faq-item"><summary><span>How does this interact with the agent's own budget?</span></summary><p>A rejection with a retry time longer than the agent's remaining budget is effectively an outage for that turn, and the agent should degrade rather than wait. Classifying a long retry hint as a capability failure rather than a transient one — which is what the error mapping does — is what makes that happen automatically.</p></details>

<details class="faq-item"><summary><span>Is queueing better than rejecting?</span></summary><p>Only up to a bounded depth. A bounded queue absorbs bursts, which is genuinely useful, and an unbounded one converts a load problem into a latency problem that grows without limit and without error. The queue is part of the admission decision rather than an alternative to it: admit while the projected wait is worth waiting for, reject with a hint after that.</p></details>

## Related

- Up to the section: [Async vs Sync Geoprocessing Workflows](/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/)
- [Handling Async Spatial Processing in Python Workflows](/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/handling-async-spatial-processing-in-python-workflows/)
- [Retry and Circuit-Breaker Patterns for Spatial Services](/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/retry-and-circuit-breaker-patterns-for-spatial-services/)
- [Mapping Spatial API Errors to User-Friendly Prompts](/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/mapping-spatial-api-errors-to-user-friendly-prompts/)
