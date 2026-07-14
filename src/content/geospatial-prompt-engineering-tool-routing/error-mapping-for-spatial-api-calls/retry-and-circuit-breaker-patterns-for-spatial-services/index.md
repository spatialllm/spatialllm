# Retry and Circuit-Breaker Patterns for Spatial Services

Blindly retrying a failed spatial call is worse than not retrying at all when the failure is a self-intersecting polygon: the geometry is invalid on attempt one and still invalid on attempt five, so every retry burns a connection and hides the real error. This guide separates transient spatial faults from permanent ones and wraps them in a retry-with-jitter plus a circuit breaker that trips on the topology-failure rate. It belongs to [error mapping for spatial API calls](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/) and governs the resilience layer between an agent and its geoprocessing backend.

The decisive move is classification. A statement timeout, a dropped socket, or a pool-exhaustion error is *transient* — the same input may succeed moments later, so backoff-and-retry is correct. A GEOS topology exception, an invalid WKT, or a CRS mismatch is *permanent* for that input — retrying only repeats the fault. Conflating the two produces retry storms that amplify load exactly when the service is already unhealthy.

## When to Use This Approach

Use a retry decorator on any call that crosses a network or contends for a pooled resource. Add a circuit breaker when repeated failures signal a systemic problem — a wedged replica, a corrupt input batch from an upstream model — where continuing to call is actively harmful. Skip retries entirely for pure validation errors: surface them immediately so the caller can re-prompt or repair.

| Error class | Example | Strategy |
|---|---|---|
| Transient | Statement timeout, connection reset, pool exhausted | Backoff + jitter, bounded attempts |
| Permanent | GEOS topology error, invalid WKT, CRS mismatch | Fail fast, no retry, return mapped error |
| Systemic | Sustained topology-failure rate, replica down | Open circuit, shed to fallback |

The breaker specifically watches the *rate* of permanent topology failures, not just transient ones. A sudden spike of `TopologyException` across many inputs usually means an upstream generator started emitting garbage geometries; opening the circuit stops the flood and gives operators a clean signal. For turning the resulting errors into readable guidance, see [mapping spatial API errors to user-friendly prompts](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/mapping-spatial-api-errors-to-user-friendly-prompts/).

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

Full jitter (`base * 2**attempt * random()`) is deliberate: it spreads retries across the backoff window so a fleet of clients does not re-collide in synchronized waves. The fallback (`[]`) lets the caller degrade to a cached or empty answer, consistent with [implementing fallback routing for failed spatial queries](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/implementing-fallback-routing-for-failed-spatial-queries/).

## Validation & Testing

- **Permanent faults are not retried.** Stub `fn` to raise `Exception("TopologyException: side location conflict")` and assert it is called exactly once and returns the fallback — no backoff sleeps occur.
- **Transient faults back off then succeed.** Have the stub fail with `asyncio.TimeoutError` twice then return a value; assert three total calls and that recorded sleeps are non-decreasing in expectation across attempts.
- **Breaker opens on topology surge.** Feed the breaker `topo_trip` permanent outcomes within one window and assert `allow()` returns `False`, then advance a monotonic clock past `cooldown_s` and assert it half-opens and permits a single probe.

## Gotchas & Edge Cases

- **String matching is brittle.** Classifying by message substring breaks across driver and GEOS versions. Prefer matching on `SQLSTATE`/exception subclasses where available and treat the substring list as a fallback heuristic, logging any unclassified error for review.
- **Retrying non-idempotent writes.** Backoff on an `INSERT` that partially committed can duplicate rows. Only decorate idempotent reads or writes guarded by `ON CONFLICT`; never blind-retry a bare mutation.
- **Half-open thundering herd.** When the breaker half-opens, allowing many probes at once can immediately re-trip it. Admit exactly one probe (as above) and reopen on its failure before letting general traffic resume.
- **Global breaker hides per-shard health.** One breaker across many databases trips on the busiest shard and starves the healthy ones. Key a breaker per backend endpoint so isolation is local.

## Related

- Up to the section: [Error Mapping for Spatial API Calls](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/)
- [Mapping Spatial API Errors to User-Friendly Prompts](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/mapping-spatial-api-errors-to-user-friendly-prompts/)
- [Backpressure and Rate Limiting for Spatial API Calls](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/backpressure-and-rate-limiting-for-spatial-api-calls/)
- [Implementing Fallback Routing for Failed Spatial Queries](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/implementing-fallback-routing-for-failed-spatial-queries/)
