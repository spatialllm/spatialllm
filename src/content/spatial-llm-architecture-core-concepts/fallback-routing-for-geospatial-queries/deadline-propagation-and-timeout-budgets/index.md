---
title: Deadline Propagation and Timeout Budgets
description: Give a request one deadline, spend it down across every hop, and refuse to start work that cannot finish — so a slow failure costs one budget rather than several.
slug: deadline-propagation-and-timeout-budgets
type: howto
breadcrumb: Deadlines and Timeouts
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Deadline Propagation and Timeout Budgets

Per-call timeouts are the default everywhere and they compose badly: three hops with a two-second timeout each produce a six-second failure for a user who left after three. A deadline belongs to the request, travels with it, and is spent down — which turns a chain of independent guesses into one budget with admission control. This guide implements that, as the timing mechanism behind [fallback routing for geospatial queries](/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/).

## When to Use This Approach

Propagate a deadline whenever a request crosses more than one boundary, which for a spatial agent is essentially always: resolution, retrieval, geometry, and possibly a fallback ladder inside each.

| Pattern | Effect on a slow failure | Use |
|---------|--------------------------|-----|
| Per-call timeout | Costs the sum of every timeout | Only for a single-hop request |
| Shared deadline | Costs one budget, total | The default |
| Deadline plus admission | Costs less than one budget | Where cheap fallbacks exist |
| No timeout | Costs whatever the slowest dependency does | Never |

The third row is what makes a fallback ladder work. Without admission control the expensive first rung consumes the whole budget before failing, and the cheap cached rung that would have satisfied the user never runs.

<figure class="diagram">
<svg viewBox="16 24 736 174" role="img" aria-labelledby="dpt-comp-t dpt-comp-d" xmlns="http://www.w3.org/2000/svg"><title id="dpt-comp-t">How per-call timeouts compose</title><desc id="dpt-comp-d">Three hops with two-second timeouts produce a six-second failure, while a shared three-second deadline spent down across the same hops fails within the time the user was willing to wait.</desc><rect x="16" y="24" width="736" height="174" fill="#ffffff"/><text x="30" y="62" fill="#b3324f" font-size="13" font-weight="600">per call</text><rect x="160" y="38" width="180" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="346" y="38" width="180" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="532" y="38" width="180" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="11.5" text-anchor="middle"><text x="250" y="64">resolve — 2 s</text><text x="436" y="64">retrieve — 2 s</text><text x="622" y="64">geometry — 2 s</text></g><rect x="160" y="96" width="270" height="6" rx="3" fill="#c46a3d"/><text x="446" y="106" fill="#c46a3d" font-size="12">the user gave up here</text><text x="30" y="166" fill="#12805c" font-size="13" font-weight="600">shared</text><rect x="160" y="142" width="120" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="286" y="142" width="100" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="392" y="142" width="38" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="11.5" text-anchor="middle"><text x="220" y="168">1.4 s</text><text x="336" y="168">1.1 s</text></g><text x="470" y="168" fill="#5b6471" font-size="12">answer or honest refusal, inside 3 s</text></svg>
<figcaption><b>The same hops, the same failures, half the wall clock.</b> Nothing about the dependencies changed; the difference is entirely in whether the time was owned by the request or by each call independently.</figcaption>
</figure>

## Implementation

The deadline is a small object created once per request, passed down, and consulted before every call.

```python
import logging
import time
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("deadline")


class DeadlineExceeded(TimeoutError):
    """The request budget is spent; nothing further may be started."""


@dataclass(frozen=True)
class Deadline:
    started_monotonic: float
    budget_s: float
    label: str = "request"

    @classmethod
    def start(cls, budget_s: float, label: str = "request") -> "Deadline":
        if budget_s <= 0:
            raise ValueError("deadline budget must be positive")
        return cls(time.monotonic(), float(budget_s), label)

    def remaining_s(self) -> float:
        return self.budget_s - (time.monotonic() - self.started_monotonic)

    def expired(self) -> bool:
        return self.remaining_s() <= 0

    def check(self) -> None:
        if self.expired():
            raise DeadlineExceeded(f"{self.label} budget of {self.budget_s:.1f}s is spent")

    def sub(self, fraction: float, label: str) -> "Deadline":
        """A child deadline for one stage — never longer than what remains."""
        left = self.remaining_s()
        if left <= 0:
            raise DeadlineExceeded(f"{self.label} budget is spent before {label}")
        return Deadline(time.monotonic(), min(left, max(0.0, self.budget_s * fraction)), label)
```

Using a monotonic clock rather than wall time is not a detail: a wall clock can move backwards under a time synchronisation and produce a deadline that never expires or expires immediately, and the failure is rare enough to be baffling when it happens.

Admission control is the second half, and the part most implementations omit.

```python
def admit(deadline: Deadline, typical_s: float, name: str,
          safety: float = 0.5) -> bool:
    """Should this call be started at all, given what is left?"""
    left = deadline.remaining_s()
    if left <= 0:
        log.info("skipping %s: budget already spent", name)
        return False
    if left < typical_s * safety:
        log.info("skipping %s: %.2fs left, typically needs %.2fs", name, left, typical_s)
        return False
    return True


def call_with_deadline(fn, deadline: Deadline, typical_s: float, name: str, *args):
    """Run a dependency with whatever time remains, or decline to start it."""
    if not admit(deadline, typical_s, name):
        raise DeadlineExceeded(f"no time left to attempt {name}")
    timeout = max(0.05, deadline.remaining_s())
    started = time.monotonic()
    try:
        return fn(*args, timeout=timeout)
    finally:
        log.debug("%s took %.3fs of %.3fs remaining", name,
                  time.monotonic() - started, timeout)
```

The safety factor is what makes admission useful rather than merely conservative. Requiring the full typical duration would decline calls that frequently finish faster; requiring half of it declines only the attempts that are very unlikely to complete, and the time saved goes to a cheaper rung that can.

Passing the remaining budget as the call's own timeout closes the loop. A dependency given a fixed timeout can outlive the request deadline, which is how a cancelled request keeps consuming a connection for another ninety seconds.

<figure class="diagram">
<svg viewBox="16 24 737 216" role="img" aria-labelledby="dpt-admit-t dpt-admit-d" xmlns="http://www.w3.org/2000/svg"><title id="dpt-admit-t">Admission control preserving budget for a cheap fallback</title><desc id="dpt-admit-d">Without admission the expensive rung consumes the remaining budget and fails; with it, the attempt is declined and the cached rung runs inside the time that was left.</desc><rect x="16" y="24" width="737" height="216" fill="#ffffff"/><text x="30" y="62" fill="#b3324f" font-size="13" font-weight="600">no admission</text><rect x="190" y="38" width="180" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="376" y="38" width="260" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="11.5" text-anchor="middle"><text x="280" y="64">first rung fails</text><text x="506" y="64">second rung: starts, times out</text></g><text x="650" y="64" fill="#5b6471" font-size="12">nothing left</text><text x="30" y="166" fill="#12805c" font-size="13" font-weight="600">with admission</text><rect x="190" y="142" width="180" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="376" y="142" width="90" height="42" rx="5" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="472" y="142" width="150" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="11.5" text-anchor="middle"><text x="280" y="168">first rung fails</text><text x="421" y="168">declined</text><text x="547" y="168">cache answers</text></g><text x="390" y="222" fill="#1f2937" font-size="13" text-anchor="middle">Declining an attempt that cannot finish is what leaves room for one that can</text></svg>
<figcaption><b>The declined attempt is the point.</b> Starting it would have been free of charge only if it succeeded; because it could not, the whole remaining budget went to producing a timeout instead of an answer.</figcaption>
</figure>

## Validation & Testing

```python
def test_remaining_decreases_and_expires():
    d = Deadline.start(0.05)
    assert d.remaining_s() > 0
    time.sleep(0.06)
    assert d.expired()


def test_sub_deadline_never_exceeds_the_parent():
    parent = Deadline.start(1.0)
    child = parent.sub(0.9, "retrieval")
    assert child.budget_s <= parent.remaining_s() + 1e-6


def test_admission_declines_what_cannot_finish():
    d = Deadline.start(0.2)
    time.sleep(0.15)
    assert not admit(d, typical_s=1.0, name="geometry")


def test_expired_parent_refuses_to_create_a_child():
    d = Deadline.start(0.01)
    time.sleep(0.02)
    try:
        d.sub(0.5, "retrieval")
    except DeadlineExceeded:
        return
    raise AssertionError("an expired deadline must not yield a child budget")


def test_dependency_receives_the_remaining_time():
    seen = {}
    def fake(_arg, timeout=None):
        seen["timeout"] = timeout
        return "ok"
    d = Deadline.start(1.0)
    call_with_deadline(fake, d, 0.1, "fake", "arg")
    assert 0 < seen["timeout"] <= 1.0
```

The last test is the one that catches the most common regression. Passing a constant timeout to a dependency is the default in most client libraries, and forgetting to override it means the deadline governs the orchestrator while every actual network call ignores it.

## Gotchas & Edge Cases

**Wall-clock arithmetic.** A clock adjustment makes a wall-time deadline expire immediately or never. Use a monotonic source, and be aware that it does not survive process boundaries — deadlines crossing a service boundary must travel as a remaining duration, not as an absolute timestamp.

**Deadlines passed as absolute times between machines.** Clock skew between hosts turns a shared absolute deadline into a different budget on each. Send the remaining milliseconds and let the receiver start its own clock.

**A sub-deadline that outlives its parent.** Fractional child budgets are convenient and must be clamped to what remains, or a stage granted "half the budget" late in a request gets more time than the request has left.

<figure class="diagram">
<svg viewBox="26 36 708 158" role="img" aria-labelledby="dpt-hop-t dpt-hop-d" xmlns="http://www.w3.org/2000/svg"><title id="dpt-hop-t">A deadline crossing a service boundary</title><desc id="dpt-hop-d">An absolute timestamp is misread when clocks differ between hosts, while a remaining duration is interpreted correctly by whichever machine receives it.</desc><rect x="26" y="36" width="708" height="158" fill="#ffffff"/><rect x="40" y="50" width="300" height="60" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="190" y="76" fill="#1f2937" font-size="12.5" text-anchor="middle">absolute timestamp sent</text><text x="190" y="98" fill="#5b6471" font-size="12" text-anchor="middle">clock skew changes the budget</text><rect x="420" y="50" width="300" height="60" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="570" y="76" fill="#1f2937" font-size="12.5" text-anchor="middle">remaining duration sent</text><text x="570" y="98" fill="#5b6471" font-size="12" text-anchor="middle">receiver starts its own clock</text><text x="380" y="176" fill="#1f2937" font-size="13" text-anchor="middle">Skew of a few seconds turns a three-second budget into none or into forever</text></svg>
<figcaption><b>Durations travel; instants do not.</b> The receiving host has no way to know whether a timestamp two seconds in the past means the budget is spent or that its clock is fast.</figcaption>
</figure>

**Retries inside a deadline that ignore it.** A client library retrying three times with its own backoff will happily spend the entire budget without consulting it. Disable library-level retries and do retry explicitly, checking the deadline between attempts.

**Cleanup work counted against the budget.** Writing a log line or emitting a metric after the deadline expires is correct and should not raise. Check the deadline before starting work, not in `finally` blocks.

**Typical durations that drift.** Admission control depends on knowing what a call usually costs, and that number changes. Derive it from observed latency percentiles rather than hard-coding it, and refresh it periodically.

## Frequently Asked Questions

<details class="faq-item"><summary><span>What should the request budget be?</span></summary><p>Whatever a user will actually wait for, which for an interactive agent is a few seconds and for a batch job may be minutes. Deriving it from dependency latencies is backwards: the budget is a product decision, and the dependencies then have to fit inside it or be replaced by cheaper rungs. A budget set by adding up what the current implementation happens to take is not a budget, it is a description.</p></details>

<details class="faq-item"><summary><span>Should every stage get a fractional sub-deadline?</span></summary><p>Only where a stage must be prevented from consuming everything — typically the first, expensive stage of a ladder. Elsewhere, passing the parent deadline directly is simpler and lets a fast stage return its unused time to the ones after it. Fractional splits allocate optimistically and, when an early stage finishes quickly, leave later stages artificially constrained.</p></details>

<details class="faq-item"><summary><span>How does this interact with server-side cancellation?</span></summary><p>It should trigger it. Passing the remaining budget as the call's timeout lets the client abandon the call, but the server may keep working unless it is told; where the protocol supports a deadline header or cancellation token, propagate it. Otherwise an abandoned request continues to consume database connections and geometry workers on behalf of a user who has already been answered.</p></details>

<details class="faq-item"><summary><span>What should be reported when the deadline is the reason for a refusal?</span></summary><p>The fact and the stage, not the internal numbers. "This took longer than the time available; the boundary check did not complete" is actionable — the user can retry or narrow their question. Reporting the budget and the elapsed milliseconds exposes implementation detail while answering none of the questions a reader has.</p></details>

<details class="faq-item"><summary><span>Should the deadline cover work after the answer is produced?</span></summary><p>No. Logging, metrics, cache writes and audit records happen after the user has been served and must not be cancelled by an expired deadline — nor should they extend it. Structure the request so the deadline governs everything up to producing the answer, and treat post-answer work as a separate concern with its own, generous limits. Mixing them produces the worst outcome available: an answer computed successfully and then lost because the audit write ran out of time.</p></details>

<details class="faq-item"><summary><span>How should a budget be divided when stages run concurrently?</span></summary><p>They share the same deadline rather than dividing it, since concurrent stages are spending wall-clock time together rather than in sequence. What does need attention is the aggregate: three concurrent calls each given the full remaining budget can all still be running when it expires, so the coordinating code must cancel the stragglers rather than waiting for them. A deadline that is checked only before starting work leaves that gap wide open.</p></details>

## Related

- Up to the parent topic: [Fallback Routing for Geospatial Queries](/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/)
- [Implementing Fallback Routing for Failed Spatial Queries](/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/implementing-fallback-routing-for-failed-spatial-queries/)
- Related topic: [Cost and Latency Budgets for Spatial Agents](/geospatial-prompt-engineering-tool-routing/cost-and-latency-budgets-for-spatial-agents/)
- Related topic: [Error Mapping for Spatial API Calls](/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/)
