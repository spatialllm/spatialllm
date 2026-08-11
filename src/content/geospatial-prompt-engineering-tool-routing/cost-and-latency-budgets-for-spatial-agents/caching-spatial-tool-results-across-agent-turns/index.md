---
title: Caching Spatial Tool Results Across Agent Turns
description: Key spatial caches so they actually hit — rounded extents, version stamps, sorted parameters — and get conversational consistency as well as the cost saving.
slug: caching-spatial-tool-results-across-agent-turns
type: howto
breadcrumb: Caching Tool Results
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Caching Spatial Tool Results Across Agent Turns

Within one conversation the same place resolves, the same region filters and the same statistic is computed repeatedly, and none of it changes between turns. Caching that is the largest cost saving available in a spatial agent, and it delivers something more valuable as a side effect: the same question asked twice gets the same answer. This guide covers the keys, which is where caches succeed or fail — the practical half of [cost and latency budgets for spatial agents](/geospatial-prompt-engineering-tool-routing/cost-and-latency-budgets-for-spatial-agents/).

## When to Use This Approach

Cache anything deterministic that costs more than the cache lookup. In a spatial agent that is nearly every tool result, with one important exception.

| Result | Cache? | Key on |
|--------|--------|--------|
| Place resolution | Yes, aggressively | Name, corpus, gazetteer version |
| Region query result | Yes | Rounded extent, filters, index version |
| Raster statistic | Yes | Shape identity, product version |
| Topology check | Yes | Feature versions, tolerance |
| Anything with an external effect | No | — |

The last row is absolute. A cached result for a step that notified something, wrote something back or triggered a downstream process turns a repeated call into a silently skipped effect, which is a much worse failure than the cost it saved.

<figure class="diagram">
<svg viewBox="16 38 748 208" role="img" aria-labelledby="cst-key-t cst-key-d" xmlns="http://www.w3.org/2000/svg"><title id="cst-key-t">Three cache keys, one of which works</title><desc id="cst-key-d">A key with full-precision coordinates never hits twice, one containing a timestamp never hits at all, and one with rounded values and a version stamp hits reliably.</desc><rect x="16" y="38" width="748" height="208" fill="#ffffff"/><rect x="30" y="52" width="720" height="52" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="50" y="84" fill="#1f2937" font-size="12.5">region:v7:-3.200,55.940,-3.180,55.960:use=residential — rounded, versioned, sorted</text><rect x="30" y="116" width="720" height="52" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="50" y="148" fill="#1f2937" font-size="12.5">region:-3.19999831,55.94000112,… — full precision, misses every time</text><rect x="30" y="180" width="720" height="52" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="50" y="212" fill="#1f2937" font-size="12.5">region:1723380142:… — a timestamp in the key, so it cannot hit at all</text></svg>
<figcaption><b>Two of these three consume storage and return nothing.</b> Neither reports an error, both look like working caches from the outside, and only a hit-rate metric distinguishes them from the one that works.</figcaption>
</figure>

## Implementation

The key builder is the whole design. It rounds, sorts, and includes every version that could change the answer.

```python
import hashlib
import json
import logging
from typing import Any, Callable, Mapping, Optional

log = logging.getLogger("spatial_cache")

EXTENT_PLACES = 3          # ~100 m: far finer than the difference between two answers


def cache_key(kind: str, versions: Mapping[str, str], **parts: Any) -> str:
    """Deterministic across processes: rounded extents, sorted keys, versions included."""
    normalised: dict[str, Any] = {}
    for name, value in parts.items():
        if name.endswith("_bbox") and value is not None:
            normalised[name] = [round(float(v), EXTENT_PLACES) for v in value]
        elif isinstance(value, (set, frozenset)):
            normalised[name] = sorted(value)
        elif isinstance(value, str):
            normalised[name] = value.strip().lower()
        else:
            normalised[name] = value
    payload = json.dumps({"kind": kind, "versions": dict(sorted(versions.items())),
                          "parts": normalised},
                         sort_keys=True, separators=(",", ":"), default=str)
    return f"{kind}:{hashlib.sha256(payload.encode()).hexdigest()[:20]}"
```

Rounding the extent is what turns a per-request miss into a per-neighbourhood hit. Three decimal places is roughly a hundred metres, which is far finer than the difference between two answers about the same area and coarse enough that two requests about the same place share a key.

Including versions in the key rather than relying on expiry is what makes a data correction propagate immediately. When the index is rebuilt, every key changes, and the stale entries expire on their own schedule without ever being served.

```python
def cached(kind: str, cache, compute: Callable[[], Any], ttl_s: int,
           versions: Mapping[str, str], **parts: Any) -> tuple[Any, bool]:
    """Read through the cache. A cache outage degrades performance, never correctness."""
    key = cache_key(kind, versions, **parts)
    try:
        hit = cache.get(key)
        if hit is not None:
            return hit, True
    except Exception as exc:
        log.warning("cache read failed for %s: %s", key, exc)

    value = compute()
    try:
        if _is_cacheable(value):
            cache.set(key, value, ttl=ttl_s)
    except Exception as exc:
        log.warning("cache write failed for %s: %s", key, exc)
    return value, False


def _is_cacheable(value: Any) -> bool:
    """Never cache a failure, an empty result, or a flagged low-confidence answer."""
    if value is None:
        return False
    if isinstance(value, dict):
        if value.get("error") or value.get("low_confidence"):
            return False
        if value.get("truncated"):
            return False              # a truncated result depends on a limit that may change
    return True
```

Refusing to cache failures and low-confidence results is the rule that prevents a transient outage from freezing a bad answer in place for the whole time-to-live. It is also what stops a flagged fallback from being served as though it were a determination on every subsequent turn.

<figure class="diagram">
<svg viewBox="16 24 664 210" role="img" aria-labelledby="cst-consist-t cst-consist-d" xmlns="http://www.w3.org/2000/svg"><title id="cst-consist-t">Consistency as the second benefit of caching</title><desc id="cst-consist-d">Without a cache the same name can resolve differently on two turns as the index changes; with one it resolves identically for the whole conversation.</desc><rect x="16" y="24" width="664" height="210" fill="#ffffff"/><text x="30" y="62" fill="#b3324f" font-size="13" font-weight="600">no cache</text><rect x="180" y="38" width="240" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="426" y="38" width="240" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="12" text-anchor="middle"><text x="300" y="64">turn 1: the west station</text><text x="546" y="64">turn 4: the east station</text></g><text x="30" y="152" fill="#12805c" font-size="13" font-weight="600">cached</text><rect x="180" y="128" width="240" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="426" y="128" width="240" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="12" text-anchor="middle"><text x="300" y="154">turn 1: the west station</text><text x="546" y="154">turn 4: the west station</text></g><text x="390" y="216" fill="#1f2937" font-size="13" text-anchor="middle">Users notice the inconsistency long before they notice the latency</text></svg>
<figcaption><b>The saving is the smaller benefit.</b> A conversation where "the station" means two different stations is one users stop trusting, and a per-conversation cache removes that possibility entirely.</figcaption>
</figure>

## Validation & Testing

```python
def test_rounded_extents_share_a_key():
    a = cache_key("region", {"index": "v7"}, region_bbox=[-3.19999, 55.94001, -3.18, 55.96])
    b = cache_key("region", {"index": "v7"}, region_bbox=[-3.20001, 55.93999, -3.18, 55.96])
    assert a == b


def test_version_change_invalidates():
    a = cache_key("region", {"index": "v7"}, region_bbox=[-3.2, 55.94, -3.18, 55.96])
    b = cache_key("region", {"index": "v8"}, region_bbox=[-3.2, 55.94, -3.18, 55.96])
    assert a != b


def test_keys_are_stable_across_processes():
    key = cache_key("place", {"gazetteer": "2026-06"}, name="Kirkby Lonsdale")
    assert key == EXPECTED_KEY_FROM_A_PREVIOUS_RUN


def test_failures_are_not_cached(fake_cache):
    value, hit = cached("place", fake_cache, lambda: {"error": "gazetteer down"},
                        ttl_s=60, versions={"gazetteer": "2026-06"}, name="x")
    assert not hit and fake_cache.writes == 0


def test_cache_outage_does_not_fail_the_call(broken_cache):
    value, hit = cached("place", broken_cache, lambda: {"place_id": "p1"},
                        ttl_s=60, versions={}, name="x")
    assert value["place_id"] == "p1" and not hit
```

The third test is the one that catches the subtlest bug. A key built from a Python dictionary's iteration order, or from a hash that varies between processes, is stable within one worker and different in another — so the cache appears to work in development and hits almost never in production.

Write the key builder once and use it everywhere. Two call sites that construct keys slightly differently produce two caches for the same data, each with half the hit rate, and the symptom is a hit rate that looks merely disappointing rather than obviously broken.

## Gotchas & Edge Cases

**Rounding that is too coarse.** Two decimal places is roughly a kilometre, which merges genuinely different regions into one key and serves an answer about the wrong area. Three places is a reasonable default; derive it from how precisely regions actually differ in your workload.

**A version omitted from the key.** A cache keyed on the region and the filters but not the index version serves pre-rebuild answers after a rebuild. Every input that can change the answer belongs in the key, including ones that feel like infrastructure.

**Caching a truncated result.** A result limited to fifty rows is a function of the limit as well as the query, and serving it for a request that asked for two hundred is silently wrong. Either include the limit in the key or refuse to cache truncated results.

<figure class="diagram">
<svg viewBox="26 42 708 148" role="img" aria-labelledby="cst-ttl-t cst-ttl-d" xmlns="http://www.w3.org/2000/svg"><title id="cst-ttl-t">Time-to-live derived from how fast the data changes</title><desc id="cst-ttl-d">A gazetteer entry is stable for months, a coverage statistic for days and a live reading for seconds, so one time-to-live across all three is wrong for at least two of them.</desc><rect x="26" y="42" width="708" height="148" fill="#ffffff"/><rect x="40" y="56" width="220" height="120" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="276" y="56" width="220" height="120" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="512" y="56" width="208" height="120" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="150" y="88">place record</text><text x="386" y="88">coverage statistic</text><text x="616" y="88">live reading</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="150" y="118">stable for months</text><text x="150" y="144">cache for weeks</text><text x="386" y="118">changes with ingests</text><text x="386" y="144">cache for hours</text><text x="616" y="118">changes constantly</text><text x="616" y="144">do not cache</text></g></svg>
<figcaption><b>One time-to-live cannot serve three change rates.</b> Choosing per kind takes a minute and removes both the stale-answer risk at one end and the pointless miss rate at the other.</figcaption>
</figure>

**Per-conversation and shared caches conflated.** A place resolution is shared safely across conversations; a session extent is not. Namespace them, or one user's context leaks into another's resolution.

**Time-to-live chosen by habit.** An hour is a number, not a policy. Derive it from how quickly the underlying data changes: a gazetteer entry is stable for months, a coverage statistic for days, and a live sensor reading for seconds.

**A key that includes the user.** Namespacing everything per user feels safe and destroys the shared benefit: a place resolution is the same for everyone, and keying it per user turns a high-hit-rate global cache into thousands of cold ones.

**Cache hit rate measured in aggregate.** One cache collapsing to zero disappears into the average of the others. Track the rate per kind, and alert on a step change in any of them.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should the cache be per conversation or global?</span></summary><p>Both, with different keys. Place resolutions, region queries and raster statistics depend only on the data and belong in a global cache where every conversation benefits. Anything derived from conversational state — the session extent, a pinned choice, a user correction — belongs in a per-conversation store and must never share a namespace with the global one.</p></details>

<details class="faq-item"><summary><span>What hit rate should be expected?</span></summary><p>For place resolutions within a conversation, very high — most conversations mention the same handful of places repeatedly. For region queries, moderate, because regions vary with the question. A rate below about a fifth for place resolutions almost always means the key includes something that varies, and the first thing to check is whether the name is being normalised before it is hashed.</p></details>

<details class="faq-item"><summary><span>How should a cached result be surfaced to the user?</span></summary><p>Silently for anything whose age does not matter, and with a stated age for anything where it does. A cached place resolution needs no mention; a cached coverage statistic from three weeks ago does, because the answer could reasonably have changed. Carrying the age in the cached value, rather than deriving it from a time-to-live, is what makes that possible.</p></details>

<details class="faq-item"><summary><span>Is it worth caching negative results?</span></summary><p>For genuine absences, yes — a gazetteer lookup that found nothing is a stable fact and re-running it costs the same as running it. For failures, no, and the distinction is exactly the one the cacheability check draws. Caching "no such place" is useful; caching "the gazetteer was unavailable" freezes an outage into the answer for the whole time-to-live.</p></details>

<details class="faq-item"><summary><span>Does caching change what the agent should be told?</span></summary><p>Only through the age. The agent does not need to know that a result was cached — that is an implementation detail — but it does need the age of anything where staleness could matter, because that is what turns a confident statement into a dated one. Returning the age unconditionally and letting the answer layer decide whether to mention it keeps the decision in one place.</p></details>

## Related

- Up to the parent topic: [Cost and Latency Budgets for Spatial Agents](/geospatial-prompt-engineering-tool-routing/cost-and-latency-budgets-for-spatial-agents/)
- [Estimating Token Cost of a Geoprocessing Plan](/geospatial-prompt-engineering-tool-routing/cost-and-latency-budgets-for-spatial-agents/estimating-token-cost-of-a-geoprocessing-plan/)
- Related topic: [Geocoding and Place-Name Resolution](/spatial-llm-architecture-core-concepts/geocoding-and-place-name-resolution/)
- Related topic: [Fallback Routing for Geospatial Queries](/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/)
