---
title: Error Mapping for Spatial API Calls
description: Classify what a spatial service returns into retry, degrade, fix-the-input and stop, then turn each class into an action and a sentence a user can act on.
slug: error-mapping-for-spatial-api-calls
type: topic
breadcrumb: Error Mapping
datePublished: 2025-03-11
dateModified: 2026-08-11
---

# Error Mapping for Spatial API Calls

Spatial services fail in more ways than most, and the failures look alike from the outside. A geometry engine returning an error about a self-intersecting ring, a tile service returning a gateway timeout, and a database refusing a query on a missing index are three completely different situations, and an agent that treats them identically will retry the unretryable, degrade the unfixable, and tell the user nothing useful about any of them.

This topic belongs to [geospatial prompt engineering and tool routing](/geospatial-prompt-engineering-tool-routing/) and supplies the classification that [fallback routing for geospatial queries](/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/) acts on. Without it a router degrades on everything, which is safe and wasteful; with it the router does the cheapest thing that can work.

<figure class="diagram">
<svg viewBox="16 32 748 214" role="img" aria-labelledby="ems-four-t ems-four-d" xmlns="http://www.w3.org/2000/svg"><title id="ems-four-t">Four error classes and the action each one implies</title><desc id="ems-four-d">Transient failures retry, capability failures degrade, input failures need the input corrected, and fatal failures stop — four classes with four different responses and four different messages.</desc><rect x="16" y="32" width="748" height="214" fill="#ffffff"/><rect x="30" y="46" width="360" height="86" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="410" y="46" width="340" height="86" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="30" y="146" width="360" height="86" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="410" y="146" width="340" height="86" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13" font-weight="600"><text x="52" y="76">transient</text><text x="432" y="76">capability</text><text x="52" y="176">input</text><text x="432" y="176">fatal</text></g><g fill="#5b6471" font-size="12"><text x="52" y="102">timeout, rate limit, blip</text><text x="52" y="122">retry once, then degrade</text><text x="432" y="102">this route cannot serve this</text><text x="432" y="122">degrade immediately</text><text x="52" y="202">bad geometry, wrong units</text><text x="52" y="222">fix the input and re-ask</text><text x="432" y="202">not authorised, not permitted</text><text x="432" y="222">stop and say so</text></g></svg>
<figcaption><b>Only the top-left is worth retrying.</b> The other three retry identically forever, and a system that cannot tell them apart spends its budget confirming failures it already had enough information to classify.</figcaption>
</figure>

## Foundational Principles

**Classify at the boundary, not at the call site.** Every spatial dependency gets a small adapter that turns its own error vocabulary into the four shared classes. Without that, classification logic spreads through the codebase and diverges.

**An input error is a different conversation.** A self-intersecting polygon or a coordinate in the wrong units is not a service failure; it is a correctable problem, and the useful response tells the user or the agent exactly what to correct.

**The message the user sees is derived, not raw.** Service error text is written for operators and frequently contains internals. Map it to a sentence about what happened and what can be done, and keep the original in the log where it belongs.

## Step-by-Step Implementation Pipeline

### 1. Define the classes once

Four classes cover the space, and adding a fifth is almost always a sign that one of the four is being used for two things.

```python
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

log = logging.getLogger("spatial_errors")


class Kind(str, Enum):
    TRANSIENT = "transient"        # retry once, then degrade
    CAPABILITY = "capability"      # this route cannot serve this input
    INPUT = "input"                # the request is wrong and can be corrected
    FATAL = "fatal"                # stop; nothing here can help


@dataclass(frozen=True)
class Classified:
    kind: Kind
    detail: str                    # for the log
    user_message: str              # for the answer
    field: Optional[str] = None    # which input, when kind is INPUT
```

### 2. Write one adapter per dependency

Each adapter maps its own service's vocabulary. The mapping is explicit, small, and the only place that service's error strings appear.

```python
POSTGIS_CODES = {
    "XX000": Kind.CAPABILITY,      # internal geometry error, often unsupported input
    "22023": Kind.INPUT,           # invalid parameter value
    "22P02": Kind.INPUT,           # invalid text representation
    "57014": Kind.TRANSIENT,       # statement timeout
    "53300": Kind.TRANSIENT,       # too many connections
    "42501": Kind.FATAL,           # insufficient privilege
    "42P01": Kind.FATAL,           # undefined table — a deployment problem
}


def classify_postgis(exc) -> Classified:
    """Map a database error to a shared class. Unknown codes degrade, never retry."""
    code = getattr(exc, "pgcode", None)
    kind = POSTGIS_CODES.get(code)
    if kind is None:
        log.warning("unmapped database code %r: %s", code, exc)
        kind = Kind.CAPABILITY                       # conservative: try another route
    if kind is Kind.INPUT:
        return Classified(kind, str(exc),
                          "The request contained a value the database could not use.",
                          field=_guess_field(str(exc)))
    if kind is Kind.TRANSIENT:
        return Classified(kind, str(exc),
                          "The database was busy; this can usually be retried.")
    if kind is Kind.FATAL:
        return Classified(kind, str(exc),
                          "This query is not permitted with the current access.")
    return Classified(kind, str(exc),
                      "The database could not process this geometry.")
```

Defaulting an unmapped code to capability rather than transient is the safer choice: degrading costs one alternative route, while retrying an unretryable error costs the whole budget and still fails. The database-specific mapping is developed in [classifying PostGIS errors for agent recovery](/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/classifying-postgis-errors-for-agent-recovery/).

### 3. Map HTTP services the same way

Tile services, gazetteers and catalogs speak status codes, and the mapping is mostly obvious with two exceptions worth naming.

```python
HTTP_KINDS = {
    400: Kind.INPUT, 404: Kind.CAPABILITY, 409: Kind.INPUT,
    401: Kind.FATAL, 403: Kind.FATAL,
    408: Kind.TRANSIENT, 429: Kind.TRANSIENT,
    500: Kind.TRANSIENT, 502: Kind.TRANSIENT, 503: Kind.TRANSIENT, 504: Kind.TRANSIENT,
}


def classify_http(status: int, body: str, retry_after: Optional[float]) -> Classified:
    kind = HTTP_KINDS.get(status, Kind.CAPABILITY)
    if status == 429 and retry_after and retry_after > 30:
        # A long backoff is effectively an outage for an interactive turn.
        return Classified(Kind.CAPABILITY, body[:200],
                          "The service is rate-limited for longer than this request can wait.")
    if status == 404:
        return Classified(kind, body[:200],
                          "That resource is not available from this service.")
    return Classified(kind, body[:200], _http_sentence(kind))
```

A 404 is a capability failure rather than an input error because the request was well-formed and the service simply does not have it — which means another route might. And a rate limit with a long retry window is functionally an outage for an interactive turn, so treating it as transient wastes the wait.

### 4. Turn geometry-engine failures into input errors where possible

Geometry libraries report problems in terms of rings and nodes, and those messages map cleanly onto correctable input conditions. Doing that mapping is what turns an opaque failure into a repair the pipeline can attempt.

```python
GEOMETRY_PATTERNS = (
    ("self-intersection", "The geometry crosses itself and was repaired before use."),
    ("ring not closed",   "A boundary ring was not closed."),
    ("too few points",    "A shape had too few points to be a polygon."),
    ("non-noded",         "Two boundaries cross without a shared node."),
)


def classify_geometry(exc) -> Classified:
    text = str(exc).lower()
    for needle, message in GEOMETRY_PATTERNS:
        if needle in text:
            return Classified(Kind.INPUT, str(exc), message, field="geometry")
    return Classified(Kind.CAPABILITY, str(exc),
                      "The geometry could not be processed by this operation.")
```

<figure class="diagram">
<svg viewBox="16 24 737 160" role="img" aria-labelledby="ems-cost-t ems-cost-d" xmlns="http://www.w3.org/2000/svg"><title id="ems-cost-t">What misclassification costs</title><desc id="ems-cost-d">Treating a capability failure as transient spends the whole budget on retries; treating a transient failure as fatal gives up on a request that would have succeeded on the second attempt.</desc><rect x="16" y="24" width="737" height="160" fill="#ffffff"/><text x="30" y="62" fill="#b3324f" font-size="13" font-weight="600">capability as transient</text><rect x="270" y="38" width="150" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="426" y="38" width="150" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="582" y="38" width="150" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="501" y="64" fill="#1f2937" font-size="12" text-anchor="middle">three retries, budget gone</text><text x="30" y="152" fill="#b3324f" font-size="13" font-weight="600">transient as fatal</text><rect x="270" y="128" width="150" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="426" y="128" width="306" height="42" rx="5" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><text x="345" y="154" fill="#1f2937" font-size="12" text-anchor="middle">one blip</text><text x="579" y="154" fill="#1f2937" font-size="12" text-anchor="middle">gave up on a request that would have worked</text></svg>
<figcaption><b>Both errors are expensive and only one is visible.</b> Over-retrying shows up as latency and cost; giving up too early shows up as a lower success rate that looks like a service problem.</figcaption>
</figure>

### 5. Attach the action, not just the class

The class implies an action, and making that explicit at the boundary means the router does not have to know about services at all.

```python
ACTIONS = {
    Kind.TRANSIENT:  ("retry_once", "then degrade"),
    Kind.CAPABILITY: ("degrade", "try the next route"),
    Kind.INPUT:      ("fix_input", "correct and re-ask"),
    Kind.FATAL:      ("stop", "no route can help"),
}


def to_action(c: Classified) -> dict:
    action, note = ACTIONS[c.kind]
    return {"action": action, "note": note, "user_message": c.user_message,
            "field": c.field, "detail": c.detail}
```

### 6. Give the agent a correctable error, not a failure

An input error is the one class where the agent can do something itself. Returning the field and a specific correction turns a dead end into a second attempt that usually succeeds. The phrasing of those messages matters more than it looks and is covered in [mapping spatial API errors to user-friendly prompts](/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/mapping-spatial-api-errors-to-user-friendly-prompts/).

```python
def correction_hint(c: Classified) -> Optional[str]:
    """A specific instruction the agent can act on, or None."""
    if c.kind is not Kind.INPUT:
        return None
    if c.field == "geometry":
        return "Repair or simplify the geometry and try again."
    if c.field:
        return f"The value supplied for {c.field} was not accepted; correct it and re-ask."
    return "One of the supplied values was not accepted."
```

### 7. Break the circuit when a route keeps failing

Repeated capability or transient failures from one dependency mean the route should be skipped entirely for a while, rather than retried by every request in turn.

```python
@dataclass
class Circuit:
    failures: int = 0
    opened_at: float = 0.0

    def record(self, c: Classified, now: float, threshold: int = 5,
               cool_off_s: float = 30.0) -> bool:
        """Return True when the route should be skipped."""
        if c.kind in {Kind.TRANSIENT, Kind.CAPABILITY}:
            self.failures += 1
            if self.failures >= threshold and not self.opened_at:
                self.opened_at = now
                log.warning("circuit opened after %d failures", self.failures)
        else:
            self.failures = 0
        if self.opened_at and now - self.opened_at > cool_off_s:
            self.opened_at, self.failures = 0.0, 0     # half-open: let one through
        return bool(self.opened_at)
```

Resetting on an input or fatal error is deliberate: those say something about the request rather than about the service, and counting them toward a circuit would open it on a stream of malformed requests while the dependency was perfectly healthy. The full pattern is in [retry and circuit breaker patterns for spatial services](/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/retry-and-circuit-breaker-patterns-for-spatial-services/).

### 8. Report by class, not by count

An error rate aggregated across classes is unactionable. Broken down, it points directly at the fix: rising input errors mean upstream data has changed, rising transient errors mean a dependency is unhealthy, rising capability errors mean the routing is sending work somewhere it does not belong.

```python
def error_report(counts: dict[Kind, int]) -> dict:
    total = sum(counts.values()) or 1
    return {k.value: {"count": n, "share": round(n / total, 4)} for k, n in counts.items()}
```

### 9. Correlate errors across a request

A single question can produce failures from three dependencies, and reading them separately makes each look like an isolated incident. Attaching a correlation identifier at the start of the turn and stamping it on every classification is what turns three log lines into one story.

```python
@dataclass(frozen=True)
class ErrorEvent:
    correlation_id: str
    dependency: str
    classified: Classified
    at: float


def summarise_turn(events: list[ErrorEvent]) -> str:
    """One line describing everything that failed during a turn."""
    if not events:
        return ""
    by_kind: dict[Kind, list[str]] = {}
    for event in events:
        by_kind.setdefault(event.classified.kind, []).append(event.dependency)
    parts = [f"{kind.value}: {', '.join(sorted(set(deps)))}"
             for kind, deps in sorted(by_kind.items(), key=lambda kv: kv[0].value)]
    return "; ".join(parts)
```

The pattern this reveals most often is a cascade: one dependency times out, the fallback route hits a second that is also degraded, and the third produces an input error because the request was reshaped on the way down. Read individually those look like three unrelated problems; read together they are one, and the fix is at the top.

### 10. Decide what an error does to the conversation

Classification determines what the system does; a separate decision determines what the user is told and whether the turn continues. A transient failure that the retry resolved should be invisible. A capability failure that degraded should be mentioned in one clause. Only an input error or a fatal one deserves to be the subject of the response.

```python
def conversational_weight(c: Classified, recovered: bool) -> str:
    """How prominent this failure should be in what the user reads."""
    if recovered and c.kind is Kind.TRANSIENT:
        return "silent"                       # it worked; do not narrate the retry
    if recovered and c.kind is Kind.CAPABILITY:
        return "clause"                       # "using a cached result, ..."
    if c.kind is Kind.INPUT:
        return "subject"                      # the answer is about the correction
    return "subject"
```

Getting this wrong in the safe direction is still wrong. An agent that reports every retry and every degradation produces answers cluttered with operational detail nobody asked for, and users learn to skip the first paragraph — which is where the caveat that actually mattered will eventually appear.

## Operating This Stage Over Time

The mapping tables are the part that rots. Services add status codes, databases add error conditions, and geometry libraries reword their messages between versions — and every unmapped case falls through to the conservative default, which works and hides the drift. Logging unmapped codes distinctly, and reviewing that log, is what keeps the tables current; without it the default gradually becomes the main path.

Message quality drifts too, in a way that is invisible from inside. Sentences written for one audience get reused for another, and a message that made sense to an engineer reads as gibberish to the person actually receiving it. Reading a sample of user-facing error messages every few months is unglamorous and catches more than any test.

Watch the class mix rather than the error rate. A system whose total error rate is flat while input errors rise and transient errors fall has had an upstream data change, not an improvement, and the aggregate conceals it entirely. The same applies to circuit openings: a circuit that opens weekly is telling you about a dependency, and one that has never opened is telling you the threshold is too high.

Finally, resist adding classes. The pressure comes from cases that feel special — a rate limit that might be short, a timeout that might be the query's fault — and each new class multiplies the routing logic that has to handle it. Those cases are better expressed as a class plus a field than as a fifth class.

## Failure Modes & Root Causes

**The retry storm.** A dependency fails, every request retries three times, and the retries keep it failing. Root cause: unclassified errors treated as transient. Mitigation: conservative default to capability, plus a circuit breaker.

**The opaque message.** A user is shown a database error containing a table name and a hint about a cast. Root cause: raw service text passed through. Mitigation: derived messages at the boundary, with the original kept in the log.

**The unfixable input error.** The agent is told the request was wrong but not which part. Root cause: no field on the classification. Mitigation: carry the field and a specific correction.

**The circuit that never opens.** A degraded dependency is retried by every request for hours. Root cause: a threshold set high enough never to trigger, or failures reset by unrelated errors. Mitigation: count only transient and capability failures, and review the threshold against observed incidents.

## Production Validation Protocols

1. **Adapter coverage test.** Assert every documented error code for each dependency maps to a class; a new code should fail the test rather than reach the default silently.
2. **Default-path visibility.** Assert unmapped codes are logged distinctly, and publish the count so drift is measurable.
3. **Message review gate.** Assert no user-facing message contains a table name, a stack frame or a code; a fixture of raw errors is enough to check it.
4. **Circuit behaviour test.** Assert the circuit opens after the threshold, skips while open, and half-opens after the cool-off.
5. **Class-mix indicator.** Publish the share of each class and alert on a shift; the mix is more informative than the total.
6. **Correction round trip.** Assert an input error produces a hint that, applied to the request, produces a valid one — this is the test that keeps hints from becoming decorative.

<figure class="diagram">
<svg viewBox="16 24 704 210" role="img" aria-labelledby="ems-mix-t ems-mix-d" xmlns="http://www.w3.org/2000/svg"><title id="ems-mix-t">A flat error rate hiding a change in the mix</title><desc id="ems-mix-d">Two weeks with the same total error rate: in the second, input errors have risen sharply and transient errors fallen, which points at an upstream data change rather than a service problem.</desc><rect x="16" y="24" width="704" height="210" fill="#ffffff"/><text x="30" y="62" fill="#5b6471" font-size="12.5">week 1</text><rect x="140" y="38" width="90" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="236" y="38" width="300" height="42" rx="5" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="542" y="38" width="120" height="42" rx="5" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><g fill="#1f2937" font-size="11.5" text-anchor="middle"><text x="185" y="64">input</text><text x="386" y="64">transient</text><text x="602" y="64">capability</text></g><text x="30" y="152" fill="#5b6471" font-size="12.5">week 2</text><rect x="140" y="128" width="300" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="446" y="128" width="90" height="42" rx="5" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="542" y="128" width="120" height="42" rx="5" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><g fill="#1f2937" font-size="11.5" text-anchor="middle"><text x="290" y="154">input</text><text x="491" y="154">transient</text><text x="602" y="154">capability</text></g><text x="380" y="216" fill="#1f2937" font-size="13" text-anchor="middle">Same total, entirely different cause — and the aggregate reports no change at all</text></svg>
<figcaption><b>The total is the least informative number available.</b> Every useful conclusion here comes from the mix, and an error dashboard that reports one line has thrown away the part that identifies the problem.</figcaption>
</figure>

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should the raw error ever reach the user?</span></summary><p>Only behind an explicit request for detail, and never by default. Raw text leaks internals, is written for the wrong audience, and occasionally contains data from other requests. Keeping it in the log with a correlation identifier gives support everything they need while the user sees a sentence about what happened and what to do.</p></details>

<details class="faq-item"><summary><span>How should a timeout be classified when the query was simply too big?</span></summary><p>As capability rather than transient, when you can tell — a statement timeout on a query the planner estimated as expensive is not a blip, and retrying it will time out identically. Where the estimate was low and the timeout was a surprise, transient is the right first guess with one retry. The plan cost is the signal that distinguishes them, which is one more reason to check it before running.</p></details>

<details class="faq-item"><summary><span>Is a per-dependency adapter worth the boilerplate?</span></summary><p>Yes, and it is less boilerplate than it looks: each adapter is a table and a function. The alternative is classification logic at every call site, which diverges immediately and makes a change to the policy — say, treating long rate limits as capability — into an audit of the whole codebase rather than an edit in four files.</p></details>

<details class="faq-item"><summary><span>What about errors that arrive as successful responses?</span></summary><p>They are common in spatial services and need the same treatment: a body containing an error object, an empty result where one was guaranteed, a status of 200 with a failure flag. Classify from the body, not the status, wherever the service does this, and treat an unexpected empty result as capability rather than as a valid answer of "nothing here".</p></details>

<details class="faq-item"><summary><span>How does this relate to the agent's own error handling?</span></summary><p>The agent should see the class and the action, never the exception. Its job is to decide what to tell the user and whether to try something else, and a well-classified error gives it both without requiring it to know that a particular database returns a particular code. Where the two overlap badly is retries: if both the adapter and the agent retry, the request costs the product of the two counts.</p></details>

## Related

- Up to the section overview: [Geospatial Prompt Engineering and Tool Routing](/geospatial-prompt-engineering-tool-routing/)
- Technique: [Mapping Spatial API Errors to User-Friendly Prompts](/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/mapping-spatial-api-errors-to-user-friendly-prompts/)
- Technique: [Retry and Circuit Breaker Patterns for Spatial Services](/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/retry-and-circuit-breaker-patterns-for-spatial-services/)
- Technique: [Classifying PostGIS Errors for Agent Recovery](/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/classifying-postgis-errors-for-agent-recovery/)
- Related topic: [Fallback Routing for Geospatial Queries](/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/)
- Peer topic: [Async vs Sync Geoprocessing Workflows](/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/)
