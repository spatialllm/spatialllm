---
title: Implementing Fallback Routing for Failed Spatial Queries
description: Build the router that walks a degradation ladder within one deadline, classifies failures into retry, degrade and stop, and returns an answer that states what it lost.
slug: implementing-fallback-routing-for-failed-spatial-queries
type: howto
breadcrumb: Implementing Fallback Routing
datePublished: 2025-02-12
dateModified: 2026-08-11
---

# Implementing Fallback Routing for Failed Spatial Queries

Most fallback logic is written as nested exception handlers, and it accumulates: a retry here, a cached read there, a bare `except` that swallowed something once. This guide replaces that with a router built from a declared ladder, so the degradation path is data you can review rather than control flow you have to trace — the working implementation of [fallback routing for geospatial queries](/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/).

## When to Use This Approach

Use a declared ladder when there is more than one way to answer and they differ in cost or fidelity. A single-route operation needs a timeout and an error message, not a router.

| Situation | Ladder | Bottom rung |
|-----------|--------|-------------|
| Geometry query with a cache | Exact, simplified, cached | Refusal |
| Frame resolution | Index lookup, local subset, default | Flagged fallback |
| Raster statistic | Full resolution, coarser product, none | Refusal with the reason |
| Boundary membership | Exact only | Refusal — no degradation is acceptable |
| Descriptive lookup | Exact, cached, stale-cached | Answer with an age |

The fourth row is the one to configure first. A ladder that degrades every intent equally will eventually answer a regulatory containment question from a simplified geometry, and that answer is a coin flip presented as a fact.

<figure class="diagram">
<svg viewBox="16 38 748 188" role="img" aria-labelledby="ifr-decl-t ifr-decl-d" xmlns="http://www.w3.org/2000/svg"><title id="ifr-decl-t">A declared ladder against nested exception handlers</title><desc id="ifr-decl-d">A ladder expressed as data is reviewable in one place, while the same behaviour spread through nested exception handlers has to be traced through the code to be understood.</desc><rect x="16" y="38" width="748" height="188" fill="#ffffff"/><rect x="30" y="52" width="340" height="160" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="200" y="84" fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600">nested handlers</text><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="200" y="114">order implied by nesting</text><text x="200" y="140">claims implied by which branch</text><text x="200" y="166">a bare except somewhere</text><text x="200" y="192">reviewed by tracing</text></g><rect x="410" y="52" width="340" height="160" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="580" y="84" fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600">declared ladder</text><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="580" y="114">order is a list</text><text x="580" y="140">each rung states its claim</text><text x="580" y="166">one place to add a route</text><text x="580" y="192">reviewed by reading</text></g></svg>
<figcaption><b>Same behaviour, different reviewability.</b> The nested form is not wrong; it is simply impossible to answer "what happens when the geometry engine is down" without reading every handler, which is why it drifts.</figcaption>
</figure>

## Implementation

The router takes a ladder, a deadline and a classifier, and returns an outcome that always says which rung produced it.

```python
import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

log = logging.getLogger("fallback_router")


class Transient(Exception):
    """Worth one retry on the same rung."""


class Degradable(Exception):
    """This rung cannot serve this input; move down."""


class Fatal(Exception):
    """Stop — neither retrying nor degrading can help."""


@dataclass(frozen=True)
class Rung:
    name: str
    run: Callable[[dict, float], object]      # (request, seconds_left) -> value
    claim: str                                 # what an answer from here may assert
    typical_s: float


@dataclass(frozen=True)
class Outcome:
    value: Optional[object]
    rung: str
    claim: str
    note: str
    elapsed_s: float
    attempts: int


def classify(exc: Exception) -> str:
    if isinstance(exc, Fatal):
        return "stop"
    if isinstance(exc, Transient):
        return "retry"
    if isinstance(exc, Degradable):
        return "degrade"
    log.warning("unclassified %s — treating as degradable", type(exc).__name__)
    return "degrade"


def route(request: dict, ladder: Sequence[Rung], budget_s: float) -> Outcome:
    """Walk the ladder inside one budget. Always returns an Outcome."""
    if not ladder:
        return Outcome(None, "misconfigured", "none", "no rungs configured", 0.0, 0)
    if budget_s <= 0:
        return Outcome(None, "misconfigured", "none", f"non-positive budget {budget_s}", 0.0, 0)

    started = time.monotonic()
    attempts = 0

    def left() -> float:
        return budget_s - (time.monotonic() - started)

    for rung in ladder:
        remaining = left()
        if remaining <= 0:
            log.info("budget spent before %s", rung.name)
            break
        if remaining < rung.typical_s * 0.5:
            log.info("skipping %s: %.2fs left, typically %.2fs",
                     rung.name, remaining, rung.typical_s)
            continue
        for attempt in (1, 2):
            attempts += 1
            try:
                value = rung.run(request, left())
                return Outcome(value, rung.name, rung.claim, "",
                               time.monotonic() - started, attempts)
            except Exception as exc:
                action = classify(exc)
                log.info("%s attempt %d failed (%s): %s", rung.name, attempt, action, exc)
                if action == "stop":
                    return Outcome(None, rung.name, "none", f"stopped: {exc}",
                                   time.monotonic() - started, attempts)
                if action == "degrade" or attempt == 2 or left() <= 0:
                    break

    return Outcome(None, "refusal", "none",
                   "no route could answer within the time available",
                   time.monotonic() - started, attempts)
```

The two misconfiguration guards look trivial and earn their place. An empty ladder is the natural consequence of trimming for an intent that has no exact rung configured, and without the guard it presents as a refusal — sending an investigation after missing data when the real problem is a missing configuration entry.

Trimming the ladder per intent is what enforces the non-degradable cases, and it belongs before routing rather than inside it.

```python
NON_DEGRADABLE = {"contains", "regulatory", "boundary"}


def ladder_for(intent: str, ladder: Sequence[Rung]) -> Sequence[Rung]:
    """Trim rungs whose claim cannot support this intent."""
    if intent in NON_DEGRADABLE:
        trimmed = [r for r in ladder if r.claim == "exact"]
        if not trimmed:
            log.warning("intent %r admits no rung: no exact route configured", intent)
        return trimmed
    return ladder
```

<figure class="diagram">
<svg viewBox="16 38 748 178" role="img" aria-labelledby="ifr-class-t ifr-class-d" xmlns="http://www.w3.org/2000/svg"><title id="ifr-class-t">Three failure classes and what each one does to the walk</title><desc id="ifr-class-d">A transient failure retries once on the same rung, a degradable failure moves down immediately, and a fatal failure stops the walk entirely.</desc><rect x="16" y="38" width="748" height="178" fill="#ffffff"/><rect x="30" y="52" width="230" height="150" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="275" y="52" width="230" height="150" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="520" y="52" width="230" height="150" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="145" y="84">transient</text><text x="390" y="84">degradable</text><text x="635" y="84">fatal</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="145" y="114">a timeout, a blip</text><text x="145" y="140">one retry here</text><text x="145" y="168">then move down</text><text x="390" y="114">bad input for this rung</text><text x="390" y="140">retrying is pointless</text><text x="390" y="168">move down at once</text><text x="635" y="114">not authorised, not valid</text><text x="635" y="140">no rung can help</text><text x="635" y="168">stop and say why</text></g></svg>
<figcaption><b>Classification is what stops a router wasting a budget.</b> Retrying a malformed geometry three times on three rungs is nine identical failures and one exhausted deadline, and it is what an unclassified handler does by default.</figcaption>
</figure>

## Validation & Testing

```python
def test_ladder_reaches_the_cache_when_engines_fail():
    ladder = [_failing("exact", Degradable), _failing("simplified", Degradable),
              _returning("cached", "cached-value", claim="cached")]
    out = route({}, ladder, budget_s=2.0)
    assert out.value == "cached-value" and out.claim == "cached"


def test_fatal_stops_the_walk():
    ladder = [_failing("exact", Fatal), _returning("cached", "should-not-run")]
    out = route({}, ladder, budget_s=2.0)
    assert out.value is None and "stopped" in out.note


def test_transient_retries_once_then_degrades():
    ladder = [_failing("exact", Transient), _returning("cached", "v")]
    out = route({}, ladder, budget_s=2.0)
    assert out.value == "v" and out.attempts == 3          # two on the first rung, one on the second


def test_non_degradable_intent_refuses_rather_than_degrading():
    ladder = ladder_for("contains", [_failing("exact", Degradable),
                                     _returning("cached", "v", claim="cached")])
    out = route({}, ladder, budget_s=2.0)
    assert out.value is None and out.rung == "refusal"


def test_empty_ladder_reports_misconfiguration():
    out = route({}, [], budget_s=1.0)
    assert out.rung == "misconfigured"
```

The last two are the tests that matter over time. The non-degradable case is a policy that will be quietly eroded by anyone adding a convenient cached rung, and the misconfiguration case is the one that turns a silent refusal into a legible error.

Build the fixtures as small helpers that fail in a stated way rather than as mocks of real dependencies. The router's behaviour depends only on the exception class it sees, so testing it against a real geometry engine tests the engine and leaves the interesting cases — fatal, transient, misconfigured — unexercised.

## Gotchas & Edge Cases

**A bare exception handler above the router.** It converts every classified outcome back into an opaque failure. The router already returns rather than raising for expected conditions; anything it does raise is a bug worth surfacing.

**Retry counts that multiply across rungs.** Two attempts on each of four rungs is eight calls to systems that are probably all unhealthy at once. One retry, on the first rung only, is usually the right budget.

**Claims that drift from behaviour.** A rung labelled "simplified" whose tolerance was later increased still reports the same claim string. Assert the relationship — a simplified rung must report the tolerance it actually applied.

**Fallback rungs that are never exercised.** They only run during incidents, which is the worst time to discover a bug. Fail the rungs above them in continuous integration and assert both the value and the claim.

<figure class="diagram">
<svg viewBox="26 36 662 200" role="img" aria-labelledby="ifr-untested-t ifr-untested-d" xmlns="http://www.w3.org/2000/svg"><title id="ifr-untested-t">Rungs exercised only by incidents</title><desc id="ifr-untested-d">The top rung runs constantly and is well tested by traffic; the lower rungs run only when something is already broken, which is the worst time to discover a defect in them.</desc><rect x="26" y="36" width="662" height="200" fill="#ffffff"/><rect x="40" y="50" width="620" height="40" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="40" y="100" width="150" height="40" rx="5" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="40" y="150" width="60" height="40" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="12"><text x="60" y="76">exact — 93% of traffic</text><text x="60" y="126">simplified — 6%</text><text x="120" y="176">cached — 1%, and only during incidents</text></g><text x="380" y="218" fill="#1f2937" font-size="13" text-anchor="middle">Fail the rungs above it in continuous integration, or it is untested code</text></svg>
<figcaption><b>Traffic tests the top rung and nothing else.</b> The rungs that matter most when things go wrong are the ones with the least exposure to normal operation, which inverts the usual relationship between usage and confidence.</figcaption>
</figure>

**Ladders that grow.** Every rung is a code path, an outcome to explain and a claim to keep honest. Three or four covers nearly every real degradation; eight is usually two ladders that should be separate.

**The refusal treated as an error.** A refusal is a successful outcome of the router — it means no route could answer honestly. Counting it as an error rate makes a well-behaved system look broken and hides the actual error rate underneath it.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Where should the ladder be defined?</span></summary><p>In configuration adjacent to the intent classification, so the two are read together. A ladder buried in the function that uses it is invisible to anyone asking what happens when a dependency fails, and that question is asked most often by people who are not in that file. Keeping it as a list of named rungs with claims also makes review possible for people who do not read the implementation.</p></details>

<details class="faq-item"><summary><span>Should the router emit metrics itself?</span></summary><p>Yes, and the distribution across rungs is the single most useful signal it produces. A system serving ninety per cent from the top rung is healthy; the same system serving forty per cent from cache has a degradation that no error rate will show, because every request succeeded. Emit the rung name as a dimension and alert on a shift rather than on a threshold.</p></details>

<details class="faq-item"><summary><span>How should the claim reach the user?</span></summary><p>As a sentence opener the answer layer is required to use, not as a field the model may ignore. "From a cached result about four hours old, the nearest depot is…" reads naturally and carries the caveat where the reader will see it. A structured field alongside the answer is easy to implement and reliably dropped by the time the text is composed.</p></details>

<details class="faq-item"><summary><span>What about partial results from a failed rung?</span></summary><p>Discard them unless the rung was explicitly designed to return partials. A geometry query that timed out halfway has computed something, and that something is a subset of unknown shape — using it produces an answer that is confidently incomplete in a way nothing downstream can detect. If partial results are valuable, make them a rung of their own with a claim that says so.</p></details>

<details class="faq-item"><summary><span>Should the router know about the agent, or only about routes?</span></summary><p>Only about routes. A router that inspects the question and decides which rungs apply has absorbed the intent classifier, and the two then change together for unrelated reasons. Keep the trimming outside — classify the intent, trim the ladder, pass the result in — so the router remains a small piece of machinery that can be tested with fixtures rather than with questions.</p></details>

## Related

- Up to the parent topic: [Fallback Routing for Geospatial Queries](/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/)
- [Deadline Propagation and Timeout Budgets](/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/deadline-propagation-and-timeout-budgets/)
- Related topic: [Error Mapping for Spatial API Calls](/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/)
- Related technique: [Retry and Circuit Breaker Patterns for Spatial Services](/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/retry-and-circuit-breaker-patterns-for-spatial-services/)
