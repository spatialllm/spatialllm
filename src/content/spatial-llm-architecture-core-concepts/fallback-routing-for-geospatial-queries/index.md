---
title: Fallback Routing for Geospatial Queries
description: Design the degradation path a spatial query takes when its primary route fails — ordered alternatives, deadline propagation, and an answer that says what it lost.
slug: fallback-routing-for-geospatial-queries
type: topic
breadcrumb: Fallback Routing
datePublished: 2025-02-11
dateModified: 2026-08-11
---

# Fallback Routing for Geospatial Queries

A spatial query has more ways to fail than a text one. The geometry engine can time out on a pathological polygon, the tile service can be unreachable, the index can be mid-rebuild, and the frame resolution can come back unresolved — each of which leaves the agent holding a question it can still partly answer. Fallback routing is the design of that partial answer: what to try next, how long is left to try it in, and what the user is told about the difference.

This topic belongs to [spatial LLM architecture and core concepts](/spatial-llm-architecture-core-concepts/) and is the counterpart to the validation gates elsewhere in it. Those gates decide what is allowed in; this one decides what happens when something allowed in cannot be processed. It is closely coupled to [error mapping for spatial API calls](/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/), which classifies the failures this router acts on.

<figure class="diagram">
<svg viewBox="16 28 768 242" role="img" aria-labelledby="fbr-ladder-t fbr-ladder-d" xmlns="http://www.w3.org/2000/svg"><title id="fbr-ladder-t">A degradation ladder for one spatial question</title><desc id="fbr-ladder-d">Exact geometry, a simplified computation, a cached previous answer and a stated refusal, each a rung with less precision than the one above and each reported to the user.</desc><rect x="16" y="28" width="768" height="242" fill="#ffffff"/><rect x="30" y="42" width="740" height="46" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="30" y="98" width="560" height="46" rx="6" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="30" y="154" width="380" height="46" rx="6" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="30" y="210" width="200" height="46" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="12.5"><text x="50" y="70">exact geometry from the primary engine</text><text x="50" y="126">simplified geometry, stated tolerance</text><text x="50" y="182">cached answer, stated age</text><text x="50" y="238">refusal, stated reason</text></g><g fill="#5b6471" font-size="12"><text x="610" y="126">precision lost</text><text x="430" y="182">currency lost</text><text x="250" y="238">nothing claimed</text></g></svg>
<figcaption><b>Each rung is a smaller claim, not a worse implementation.</b> The design work is choosing what to give up in what order, and making each surrender visible in the answer rather than absorbing it silently.</figcaption>
</figure>

## Foundational Principles

**Every fallback is a smaller claim.** Degrading to a simplified geometry or a cached result changes what the answer is entitled to assert. If the answer text does not change with the rung, the fallback has quietly converted a precision loss into a correctness claim.

**The deadline is shared, not per attempt.** A router that gives each rung its own timeout will spend four times the intended budget on a slow failure. The deadline belongs to the request and is decremented as it is spent, which is the mechanism developed in [deadline propagation and timeout budgets](/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/deadline-propagation-and-timeout-budgets/).

**A refusal is a rung, not a failure of the router.** The bottom of every ladder is an honest statement that the question could not be answered with the data available. A router with no bottom rung will keep degrading until it produces something, and the something will be indefensible.

## Step-by-Step Implementation Pipeline

### 1. Enumerate the routes before writing any of them

A ladder is a list of named routes with a precision cost and a stated claim. Writing it as data rather than as nested exception handlers keeps it reviewable and makes the ordering an explicit decision.

```python
import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

log = logging.getLogger("fallback_routing")


@dataclass(frozen=True)
class Route:
    name: str
    run: Callable[[dict, float], object]     # (request, seconds_left) -> result
    claim: str                               # what an answer from this route may assert
    typical_ms: float                        # used for admission, not for timeout


@dataclass(frozen=True)
class Outcome:
    value: Optional[object]
    route: str
    claim: str
    note: str
    elapsed_ms: float
```

### 2. Classify failures into retry, degrade and stop

Not every failure should advance the ladder. A transient timeout may deserve one retry on the same rung; a malformed geometry will fail identically on every attempt and should degrade immediately; an authorisation failure should stop the whole request.

```python
class Transient(Exception):
    """Worth one retry on the same route."""


class Degradable(Exception):
    """This route cannot serve this input; try the next rung."""


class Fatal(Exception):
    """Stop: retrying or degrading cannot help."""


def classify(exc: Exception) -> str:
    if isinstance(exc, Fatal):
        return "stop"
    if isinstance(exc, Transient):
        return "retry"
    if isinstance(exc, Degradable):
        return "degrade"
    log.warning("unclassified failure %s — treating as degradable", type(exc).__name__)
    return "degrade"                                  # conservative default
```

Defaulting an unclassified failure to degradation rather than to stopping is a judgement about which mistake is cheaper. Degrading on something that was really fatal costs one wasted attempt; stopping on something that was really transient costs an answer the system could have given.

### 3. Run the ladder against a shared deadline

The router walks the routes in order, giving each whatever time remains, and stops as soon as one succeeds or the budget is exhausted.

```python
def route(request: dict, routes: Sequence[Route], budget_ms: float) -> Outcome:
    """Walk the ladder within one shared deadline. Always returns an Outcome."""
    started = time.monotonic()

    def remaining_ms() -> float:
        return budget_ms - (time.monotonic() - started) * 1000.0

    for step in routes:
        left = remaining_ms()
        if left <= 0:
            log.info("deadline exhausted before %s", step.name)
            break
        if left < step.typical_ms * 0.5:              # admission: do not start what cannot finish
            log.info("skipping %s — %0.0f ms left, typically needs %0.0f",
                     step.name, left, step.typical_ms)
            continue
        for attempt in (1, 2):
            try:
                value = step.run(request, left / 1000.0)
                return Outcome(value, step.name, step.claim, "",
                               (time.monotonic() - started) * 1000.0)
            except Exception as exc:
                action = classify(exc)
                log.info("%s failed on attempt %d (%s): %s", step.name, attempt, action, exc)
                if action == "stop":
                    return Outcome(None, step.name, "none", f"stopped: {exc}",
                                   (time.monotonic() - started) * 1000.0)
                if action == "degrade" or attempt == 2 or remaining_ms() <= 0:
                    break
    return Outcome(None, "refusal", "none",
                   "no route could answer within the deadline",
                   (time.monotonic() - started) * 1000.0)
```

The admission check is the part most implementations omit and most need. Starting a route that typically takes eight hundred milliseconds with two hundred left guarantees a timeout, consumes the remainder of the budget, and leaves nothing for the cheaper rung that would have succeeded.

<figure class="diagram">
<svg viewBox="16 22 720 208" role="img" aria-labelledby="fbr-budget-t fbr-budget-d" xmlns="http://www.w3.org/2000/svg"><title id="fbr-budget-t">Per-attempt timeouts against a shared deadline</title><desc id="fbr-budget-d">With per-attempt timeouts three failing routes consume three full timeouts; with a shared deadline the same three attempts fit inside one budget and leave room for the cheap final rung.</desc><rect x="16" y="22" width="720" height="208" fill="#ffffff"/><text x="30" y="60" fill="#b3324f" font-size="13" font-weight="600">per attempt</text><rect x="170" y="36" width="180" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="356" y="36" width="180" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="542" y="36" width="180" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="11.5" text-anchor="middle"><text x="260" y="62">route 1 — 2 s</text><text x="446" y="62">route 2 — 2 s</text><text x="632" y="62">route 3 — 2 s</text></g><text x="400" y="104" fill="#5b6471" font-size="12" text-anchor="middle">six seconds spent; the user gave up at three</text><text x="30" y="164" fill="#12805c" font-size="13" font-weight="600">shared deadline</text><rect x="170" y="140" width="150" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="326" y="140" width="120" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="452" y="140" width="90" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="548" y="140" width="70" height="42" rx="5" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><g fill="#1f2937" font-size="11.5" text-anchor="middle"><text x="245" y="166">1.5 s</text><text x="386" y="166">1.2 s</text><text x="497" y="166">0.9 s</text><text x="583" y="166">cache</text></g><text x="400" y="212" fill="#1f2937" font-size="13" text-anchor="middle">Same routes, one budget, and the cheap rung still gets its turn</text></svg>
<figcaption><b>The last rung is the one that gets starved.</b> Per-attempt timeouts spend the budget on the routes least likely to succeed, so the cached answer that would have satisfied the user never runs.</figcaption>
</figure>

### 4. Make the claim change with the rung

The route's claim string is what the answer layer is permitted to assert. Wiring it through is what turns the ladder from an availability mechanism into an honesty mechanism.

```python
CLAIM_PREFIX = {
    "exact": "",
    "simplified": "Using a simplified boundary (tolerance {tol} m), ",
    "cached": "From a cached result {age} old, ",
    "extent": "Working from the bounding extent only, ",
}


def answer_prefix(outcome: Outcome, **context) -> str:
    """The sentence opener the answer must carry, given the route that served it."""
    template = CLAIM_PREFIX.get(outcome.claim)
    if template is None:
        log.warning("route %s has an unknown claim %r", outcome.route, outcome.claim)
        return "Working from partial data, "
    return template.format(**context)
```

### 5. Give the cache a defined staleness policy

A cached answer is the most useful fallback and the easiest to misuse. It needs an age, a maximum age beyond which it is not offered, and a rule about which requests may be served from it at all.

```python
MAX_CACHE_AGE_S = {"describe": 7 * 24 * 3600, "measure": 3600, "contain": 0}


def cache_allowed(intent: str, age_s: float) -> tuple[bool, str]:
    """Whether a cached result may serve this intent at this age."""
    limit = MAX_CACHE_AGE_S.get(intent)
    if limit is None:
        return False, f"no cache policy for intent {intent!r}"
    if limit == 0:
        return False, "this intent is never served from cache"
    if age_s > limit:
        return False, f"cached result is {age_s / 3600:.1f} h old, limit {limit / 3600:.1f} h"
    return True, ""
```

Setting the containment limit to zero is a policy choice worth naming: boundary membership changes rarely but consequentially, and a stale containment answer is the one most likely to be relied upon by somebody making a decision.

### 6. Record which rung served every request

The distribution across rungs is the single most useful operational signal this component produces. A system serving ninety per cent from the primary route is healthy; the same system serving forty per cent from cache has a problem nobody has reported yet.

```python
def observe(outcome: Outcome, metrics) -> None:
    metrics.increment("spatial.route.served", tags={"route": outcome.route})
    metrics.timing("spatial.route.latency_ms", outcome.elapsed_ms,
                   tags={"route": outcome.route})
    if outcome.route == "refusal":
        metrics.increment("spatial.route.refused")
```

### 7. Test the ladder by breaking each rung deliberately

A fallback path that has never run is a fallback path that does not work. Exercise each rung in continuous integration by failing the ones above it, and assert both the value and the claim.

```python
def test_ladder_reaches_cache_when_engines_fail(routes, cached_value):
    broken = [r for r in routes if r.name in {"exact", "simplified"}]
    outcome = route({"q": "x"}, _fail(broken) + _rest(routes), budget_ms=3000)
    assert outcome.value == cached_value
    assert outcome.claim == "cached"
```

### 8. Decide when a partial answer is worse than none

Some questions do not degrade. If a user is asking whether a site falls inside a regulated boundary, a simplified geometry can flip the answer, and offering it with a caveat is not a kindness. Mark those intents and let the ladder stop early rather than reaching for a rung that cannot support the claim.

```python
NON_DEGRADABLE = {"contain", "regulatory", "boundary"}


def ladder_for(intent: str, routes: Sequence[Route]) -> Sequence[Route]:
    """Trim the ladder for intents where an approximate answer is not acceptable."""
    if intent in NON_DEGRADABLE:
        return [r for r in routes if r.claim == "exact"]
    return routes
```

### 9. Separate the router's failures from the routes' failures

The router itself can fail — a misconfigured ladder, a claim template with a missing placeholder, a clock that moves backwards — and those failures look identical to a route failing unless they are distinguished. Wrapping the walk so the router's own exceptions are labelled separately keeps an operational investigation pointed at the right component.

```python
def route_guarded(request: dict, routes, budget_ms: float) -> Outcome:
    """Never let a router bug present as a data or dependency failure."""
    if not routes:
        return Outcome(None, "misconfigured", "none",
                       "no routes configured for this intent", 0.0)
    if budget_ms <= 0:
        return Outcome(None, "misconfigured", "none",
                       f"non-positive budget {budget_ms}", 0.0)
    try:
        return route(request, routes, budget_ms)
    except Exception as exc:                          # a bug in the router, not in a route
        log.exception("router failure")
        return Outcome(None, "router_error", "none", f"router failure: {exc}", 0.0)
```

The two misconfiguration checks are worth keeping even though they look trivial. An empty ladder is the natural consequence of trimming for a non-degradable intent that has no exact route configured, and without the check it presents as a refusal — sending an investigation after missing data when the real problem is a missing configuration entry.

## Operating This Stage Over Time

Fallback routers decay in a characteristic way: the lower rungs stop working and nobody notices, because they only run when something else has already failed. The routing distribution is the instrument that catches this — if the cache rung has served nothing in three months, either the primary route has been perfect or the cache rung is broken, and only a deliberate test distinguishes them.

The second slow failure is claim drift. A route's claim string is set when the route is written and the route's behaviour changes afterwards: a "simplified" route acquires a coarser tolerance, a "cached" route starts serving longer-lived entries. Because the claim is a string and the behaviour is code, nothing forces them to agree. Assert the relationship — a simplified route must report the tolerance it actually used — rather than trusting the label.

The third is budget creep in the opposite direction from context budgets. Deadlines get raised, one incident at a time, because a longer deadline makes a specific failure go away. Each raise is defensible and the cumulative effect is a system that takes eight seconds to tell a user it cannot answer. Review the budget as a whole periodically, against what a user will actually wait for, rather than only when something times out.

Finally, keep the ladder short. Every rung is a code path that must be tested, an outcome that must be explained, and a claim that must be kept truthful. Three or four rungs covers nearly every real degradation; a ladder with eight is usually two ladders that should be separate routers for separate intents.

## Failure Modes & Root Causes

**The silent downgrade.** An answer is served from a simplified geometry and reads exactly like an exact one. Root cause: the claim is not wired into the answer. Mitigation: the prefix in step 4, asserted in tests.

**The budget cascade.** Three routes each take their full timeout and the user waits nine seconds for a refusal. Root cause: per-attempt timeouts. Mitigation: one deadline, decremented, with admission control.

**The stale containment.** A cached membership answer is served after the boundary changed. Root cause: one cache policy for every intent. Mitigation: per-intent maximum ages, with zero for boundary questions.

**The untested rung.** The cache fallback has a bug that only appears when it runs, which is only during an incident. Root cause: fallbacks exercised only by real failures. Mitigation: deliberate rung tests in continuous integration.

## Production Validation Protocols

1. **Deadline assertion.** Assert that total elapsed time never exceeds the request budget by more than a small margin, under a fixture where every route is slow.
2. **Claim-truthfulness test.** Assert that an answer served by a degraded route carries the corresponding prefix, and that the stated tolerance matches the one applied.
3. **Rung coverage.** Assert every rung is reachable by a test that fails the rungs above it.
4. **Non-degradable enforcement.** Assert that a boundary-membership intent either answers exactly or refuses, and never returns a simplified result.
5. **Routing distribution indicator.** Publish the share of requests served by each rung and alert on a shift; this is the earliest signal of an upstream degradation.
6. **Refusal-rate indicator.** Track refusals separately from errors — a rising refusal rate with a flat error rate usually means a data gap rather than an outage.

<figure class="diagram">
<svg viewBox="16 24 704 206" role="img" aria-labelledby="fbr-dist-t fbr-dist-d" xmlns="http://www.w3.org/2000/svg"><title id="fbr-dist-t">Routing distribution as a health signal</title><desc id="fbr-dist-d">A healthy week serves almost everything from the primary route, while a degraded week shifts a large share to simplified and cached routes without any error rate rising.</desc><rect x="16" y="24" width="704" height="206" fill="#ffffff"/><text x="30" y="62" fill="#12805c" font-size="13" font-weight="600">healthy</text><rect x="170" y="38" width="470" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="646" y="38" width="60" height="42" rx="5" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="405" y="64" fill="#1f2937" font-size="12" text-anchor="middle">exact — 93%</text><text x="30" y="152" fill="#c46a3d" font-size="13" font-weight="600">degraded</text><rect x="170" y="128" width="230" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="406" y="128" width="180" height="42" rx="5" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="592" y="128" width="114" height="42" rx="5" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="12" text-anchor="middle"><text x="285" y="154">exact — 45%</text><text x="496" y="154">simplified — 35%</text><text x="649" y="154">cached</text></g><text x="380" y="212" fill="#1f2937" font-size="13" text-anchor="middle">No errors were raised in either week — the difference is only visible here</text></svg>
<figcaption><b>A degradation with no error rate.</b> Every request succeeded, every answer was caveated, and the system was quietly serving approximations to half its users — which no availability metric reports.</figcaption>
</figure>

Two of these six are worth wiring into alerting rather than into a report. The routing distribution catches degradations that produce no errors at all, and the refusal rate distinguishes a data gap from an outage — a pair of signals that between them cover most of what goes wrong here without anyone having to notice a slow change in answer quality.

## Frequently Asked Questions

<details class="faq-item"><summary><span>How many retries should a transient failure get?</span></summary><p>One, in most cases, and only when the remaining budget can absorb it. A second retry on the same route rarely succeeds where the first failed and consumes budget that a lower rung could have used productively. Where the failure is genuinely a load spike rather than a fault, a retry with a short jittered delay is worth it — but the delay comes out of the same deadline, and admission control should decide whether it fits.</p></details>

<details class="faq-item"><summary><span>Should the fallback ladder be the same for every question?</span></summary><p>No, and the non-degradable set is the clearest example of why. Descriptive questions tolerate a great deal of degradation; measurements tolerate some; boundary membership tolerates almost none. One ladder per intent class, selected before routing begins, keeps each of those judgements explicit rather than encoding them in a chain of conditionals inside the router.</p></details>

<details class="faq-item"><summary><span>What should be cached — the answer or the intermediate result?</span></summary><p>The intermediate result, generally, because it is reusable across questions and its staleness is easier to reason about. Caching a geometry query's result serves many phrasings of the same question; caching the final answer serves exactly one and inherits whatever the model happened to say. The exception is an expensive multi-step computation whose intermediate results are large, where caching the conclusion is the only affordable option.</p></details>

<details class="faq-item"><summary><span>How does this interact with the agent's own retry logic?</span></summary><p>Badly, if both exist and neither knows about the other. An agent that retries a tool call which internally ran a four-rung ladder has multiplied the work by the number of retries. Make the router's outcome explicit enough that the agent can distinguish "this failed transiently" from "this was answered by a lower rung", and let the agent retry only the first.</p></details>

<details class="faq-item"><summary><span>Is a refusal really better than a rough answer?</span></summary><p>When the question was about a boundary or a regulated threshold, yes, unambiguously — a rough answer to "is this inside the zone" is a coin flip presented as a fact. For most other questions a caveated approximation is more useful than nothing, which is why the ladder exists at all. The judgement is per intent, and the value of writing it down is that it stops being made afresh, differently, in each new code path.</p></details>

## Related

- Up to the section overview: [Spatial LLM Architecture and Core Concepts](/spatial-llm-architecture-core-concepts/)
- Technique: [Implementing Fallback Routing for Failed Spatial Queries](/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/implementing-fallback-routing-for-failed-spatial-queries/)
- Technique: [Deadline Propagation and Timeout Budgets](/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/deadline-propagation-and-timeout-budgets/)
- Related topic: [Error Mapping for Spatial API Calls](/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/)
- Related topic: [Cost and Latency Budgets for Spatial Agents](/geospatial-prompt-engineering-tool-routing/cost-and-latency-budgets-for-spatial-agents/)
- Peer topic: [Evaluation and Benchmarking for Spatial LLMs](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/)
