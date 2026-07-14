# Backpressure and Rate Limiting for Spatial API Calls

A single unbounded burst of LLM-triggered spatial calls can drain a PostGIS connection pool in seconds, turning a healthy geoprocessing service into a queue of timed-out transactions. This guide shows how to place explicit backpressure — a token bucket plus a bounded semaphore — in front of expensive spatial operations so an agent sheds load gracefully instead of collapsing. It sits inside [async vs sync geoprocessing workflows](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/) and targets the admission-control stage that guards every downstream `ST_*` call.

The core problem is fan-out amplification. An agent that decomposes one user question into a dozen buffer, intersect, and union calls will happily dispatch all of them at once. Each call may acquire a database connection, hold it while GEOS grinds through a heavy polygon, and only release it on commit. Without an admission gate, concurrency is bounded by nothing but the agent's imagination, and the pool — a scarce, fixed resource — becomes the failure point.

## When to Use This Approach

Reach for token-bucket backpressure when calls are individually expensive and the protected resource has a hard concurrency ceiling (a pool `max_size`, a licensed geocoder quota, a tile server's rate limit). Reach for a plain semaphore when you only need to cap *simultaneous* work. In practice you want both: the semaphore caps concurrency, the bucket caps sustained rate, and a timeout caps latency.

| Control | Bounds | Best for | Failure signal |
|---|---|---|---|
| Semaphore | Concurrent in-flight calls | Pool / worker protection | Acquire blocks |
| Token bucket | Calls per unit time | Quota / rate-limited APIs | No token available |
| Timeout + shed | Tail latency | User-facing agents | Deadline exceeded |

If your workload is cheap and read-only against an indexed table, a semaphore alone is enough. If calls hit an external quota, the bucket is mandatory — a semaphore lets a low-latency endpoint exceed a per-second cap. When latency budgets matter, always pair admission with a deadline so a saturated system returns a deterministic shed response rather than queueing forever. For the execution model these controls wrap, see [handling async spatial processing in Python workflows](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/handling-async-spatial-processing-in-python-workflows/).

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

## Validation & Testing

- **Concurrency ceiling holds.** Launch 200 `admission.run` calls against a pool of size 8 with a slow stub `op` and assert the observed peak of concurrent `op` entries never exceeds 8 (increment a counter on entry, decrement on exit, track the max).
- **Sustained rate is capped.** With `rate=10, burst=10`, drive 100 calls and assert wall-clock elapsed is at least `(100 - burst) / rate` seconds — proof the bucket throttled the tail rather than admitting instantly.
- **Shed is deterministic under saturation.** Set `deadline_s` below the stub op latency and assert every over-limit call returns exactly the `fallback` value and that zero of them invoked `pool.acquire()` (patch the pool and assert the mock's call count equals only the admitted count).

## Gotchas & Edge Cases

- **Monotonic clock only.** Refill uses `time.monotonic()`, never `time.time()`. An NTP step or leap adjustment on wall-clock time can hand out a burst of phantom tokens or freeze refills; the monotonic clock is immune.
- **Slot leak on cancellation.** If the task is cancelled between `acquire()` and the `finally`, the slot must still release. Keep the acquire and the `try/finally` in the same coroutine frame — never split them across `await` boundaries where cancellation can slip in.
- **Deadline smaller than sleep granularity.** A `deadline_s` near the 0.02 s poll interval can shed calls that a token would have covered milliseconds later. Set deadlines at least an order of magnitude above the poll step, or lower the sleep for latency-critical paths.
- **Bucket sized larger than the pool.** A generous burst lets many tokened calls stampede the semaphore, so they simply queue on `acquire()`. Keep `burst` close to `pool_size` so the two gates agree on the true concurrency limit.

## Related

- Up to the section: [Async vs Sync Geoprocessing Workflows](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/)
- [Handling Async Spatial Processing in Python Workflows](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/handling-async-spatial-processing-in-python-workflows/)
- [Retry and Circuit-Breaker Patterns for Spatial Services](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/retry-and-circuit-breaker-patterns-for-spatial-services/)
- [Mapping Spatial API Errors to User-Friendly Prompts](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/mapping-spatial-api-errors-to-user-friendly-prompts/)
