---
title: Cost and Latency Budgets for Spatial Agents
description: Give an agent a token and time budget it must plan within, estimate a plan's cost before running it, and cache the results that make the next turn cheap.
slug: cost-and-latency-budgets-for-spatial-agents
type: topic
breadcrumb: Cost and Latency Budgets
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Cost and Latency Budgets for Spatial Agents

Spatial agents are expensive in a way text agents are not. A single question can trigger a gazetteer lookup, a catalog search, a filtered vector query, several geometry operations and a raster read, each with its own latency and each contributing tokens to a context that has to be re-sent on every turn. Without a budget the agent will spend whatever the slowest path costs, and the first sign is a bill or a timeout rather than a design decision.

This topic belongs to [geospatial prompt engineering and tool routing](/geospatial-prompt-engineering-tool-routing/) and is the constraint the routing decisions elsewhere in it operate under. It shares its deadline mechanics with [fallback routing for geospatial queries](/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/), and its token accounting with [context-window optimization for maps](/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/).

<figure class="diagram">
<svg viewBox="26 9 748 221" role="img" aria-labelledby="clb-spend-t clb-spend-d" xmlns="http://www.w3.org/2000/svg"><title id="clb-spend-t">Where a spatial turn actually spends</title><desc id="clb-spend-d">Tool latency and re-sent context dominate a spatial turn, while the model's own generation is a small share — which inverts the usual assumption about where optimisation effort belongs.</desc><rect x="26" y="9" width="748" height="221" fill="#ffffff"/><text x="400" y="34" fill="#5b6471" font-size="13" text-anchor="middle">One question, four tool calls, three turns of accumulated context</text><rect x="40" y="58" width="300" height="44" rx="5" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="346" y="58" width="260" height="44" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="612" y="58" width="148" height="44" rx="5" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="12" text-anchor="middle"><text x="190" y="86">tool latency</text><text x="476" y="86">re-sent context</text><text x="686" y="86">generation</text></g><rect x="40" y="124" width="720" height="44" rx="5" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><text x="400" y="152" fill="#1f2937" font-size="12.5" text-anchor="middle">the budget covers all three, and only one of them is under the model&#8217;s control</text><text x="400" y="212" fill="#1f2937" font-size="13" text-anchor="middle">Optimising generation is the smallest of the three levers available</text></svg>
<figcaption><b>The model is the cheap part.</b> A spatial turn is mostly waiting for tools and re-sending geometry, which is why budgets here are about plans and caches rather than about prompt length.</figcaption>
</figure>

## Foundational Principles

**The budget is a constraint on the plan, not a limit on the run.** Checking cost as it accrues catches an overrun after the money is spent. Estimating a plan before executing it lets the agent choose a cheaper plan, which is the only intervention that actually saves anything.

**Tokens and time are separate budgets with separate failure modes.** A plan can be cheap and slow, or fast and expensive, and collapsing them into one number hides which constraint bound. Track both, report both, and let the routing decision see both.

**Cached results are the largest available saving.** Within one conversation the same place resolves, the same region filters and the same geometry loads repeatedly, and none of it changes. A cache keyed correctly removes more cost than any prompt optimisation, and it makes answers consistent between turns as a side effect.

## Step-by-Step Implementation Pipeline

### 1. Give the request an explicit budget

Both numbers come from the product rather than from the implementation: what a user will wait for, and what a turn may cost. Deriving them from what the current pipeline happens to spend produces a description rather than a budget.

```python
import logging
from dataclasses import dataclass

log = logging.getLogger("agent_budget")


@dataclass(frozen=True)
class Budget:
    tokens: int
    seconds: float
    label: str = "turn"

    def scaled(self, fraction: float, label: str) -> "Budget":
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        return Budget(int(self.tokens * fraction), self.seconds * fraction, label)


@dataclass
class Spend:
    tokens: int = 0
    seconds: float = 0.0

    def add(self, tokens: int, seconds: float) -> None:
        self.tokens += max(0, tokens)
        self.seconds += max(0.0, seconds)

    def within(self, budget: Budget) -> bool:
        return self.tokens <= budget.tokens and self.seconds <= budget.seconds
```

### 2. Estimate a plan before running it

Every tool has a typical cost in both currencies, measured from production rather than guessed. Summing them over a proposed plan gives an estimate accurate enough to reject the expensive plans, which is all it needs to do.

```python
from typing import Mapping, Sequence

TOOL_COST = {                                   # measured, refreshed periodically
    "resolve_place":     {"tokens": 120,  "seconds": 0.15},
    "catalog_search":    {"tokens": 400,  "seconds": 0.30},
    "vector_query":      {"tokens": 900,  "seconds": 0.45},
    "geometry_op":       {"tokens": 250,  "seconds": 0.60},
    "raster_summary":    {"tokens": 180,  "seconds": 1.20},
}


def estimate(plan: Sequence[str], costs: Mapping[str, dict] = TOOL_COST) -> Spend:
    """Sum the known cost of a plan. Unknown steps are charged a pessimistic default."""
    spend = Spend()
    for step in plan:
        cost = costs.get(step)
        if cost is None:
            log.info("no cost model for %r — charging the pessimistic default", step)
            cost = {"tokens": 800, "seconds": 1.0}
        spend.add(cost["tokens"], cost["seconds"])
    return spend
```

Charging an unknown step the pessimistic default rather than zero is what keeps a new tool from silently escaping the budget. A step with no cost model is exactly the step most likely to be expensive, because nobody has measured it yet.

### 3. Reject or trim a plan that does not fit

An over-budget plan is not an error; it is a plan that needs to be smaller. The agent should be told which constraint bound and by how much, so the next proposal is a reduction rather than a repeat.

```python
@dataclass(frozen=True)
class PlanVerdict:
    admissible: bool
    reason: str
    over_tokens: int
    over_seconds: float


def admit(plan: Sequence[str], budget: Budget) -> PlanVerdict:
    """Decide whether a plan fits, and say precisely how it does not."""
    spend = estimate(plan)
    over_tokens = max(0, spend.tokens - budget.tokens)
    over_seconds = max(0.0, spend.seconds - budget.seconds)
    if not over_tokens and not over_seconds:
        return PlanVerdict(True, "", 0, 0.0)
    parts = []
    if over_tokens:
        parts.append(f"{over_tokens} tokens over")
    if over_seconds:
        parts.append(f"{over_seconds:.1f}s over")
    return PlanVerdict(False, "; ".join(parts), over_tokens, round(over_seconds, 2))
```

### 4. Make the cheap plan discoverable

Rejecting a plan is only useful if a cheaper one exists and the agent can find it. The practical mechanism is a set of substitutions — a coarser raster product, a cached region instead of a fresh query, a name lookup instead of a catalog search — offered alongside the rejection.

```python
SUBSTITUTIONS = {
    "raster_summary": ("cached_raster_summary", "a stored statistic for this shape"),
    "catalog_search": ("known_collection", "the collection used on the previous turn"),
    "vector_query":   ("cached_vector_query", "the same region, already retrieved"),
}


def cheaper_plan(plan: Sequence[str], budget: Budget) -> tuple[list[str], list[str]]:
    """Substitute the most expensive steps until the plan fits, listing what changed."""
    trimmed, notes = list(plan), []
    for _ in range(len(plan)):
        if admit(trimmed, budget).admissible:
            break
        costly = max(range(len(trimmed)),
                     key=lambda i: TOOL_COST.get(trimmed[i], {"seconds": 1.0})["seconds"])
        replacement = SUBSTITUTIONS.get(trimmed[costly])
        if replacement is None:
            break                                 # nothing cheaper: the plan must shrink
        trimmed[costly] = replacement[0]
        notes.append(f"{plan[costly]} replaced by {replacement[1]}")
    return trimmed, notes
```

Recording the substitutions is what lets the answer carry its caveats. A plan that used a cached statistic instead of a fresh one has produced an answer with an age, and that age belongs in the sentence.

<figure class="diagram">
<svg viewBox="16 24 720 206" role="img" aria-labelledby="clb-plan-t clb-plan-d" xmlns="http://www.w3.org/2000/svg"><title id="clb-plan-t">Estimating before running against measuring while running</title><desc id="clb-plan-d">A plan estimated in advance can be trimmed before anything is spent, while cost measured during execution only reports an overrun after the money has gone.</desc><rect x="16" y="24" width="720" height="206" fill="#ffffff"/><text x="30" y="62" fill="#b3324f" font-size="13" font-weight="600">measure while running</text><rect x="260" y="38" width="150" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="416" y="38" width="150" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="572" y="38" width="150" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="490" y="64" fill="#1f2937" font-size="12" text-anchor="middle">spent, spent, over budget</text><text x="30" y="152" fill="#12805c" font-size="13" font-weight="600">estimate first</text><rect x="260" y="128" width="150" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="416" y="128" width="150" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="572" y="128" width="90" height="42" rx="5" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="447" y="196" fill="#5b6471" font-size="12" text-anchor="middle">the plan is trimmed to fit before it runs</text><text x="390" y="212" fill="#1f2937" font-size="13" text-anchor="middle">The only intervention that saves anything happens before the first call</text></svg>
<figcaption><b>Measurement during execution is reporting, not control.</b> By the time the third tool call has overrun the budget, the money is spent and the latency has already been experienced by the user.</figcaption>
</figure>

### 5. Cache the results that repeat

Within a conversation the same lookups recur constantly, and caching them is worth more than any other optimisation here. The keys matter: a place resolution keyed on the name and corpus, a region query keyed on the rounded extent, a raster statistic keyed on the shape and the product version.

```python
def cache_key(kind: str, **parts) -> str:
    """Deterministic keys — rounded extents, sorted parts, version stamps included."""
    rendered = ",".join(f"{k}={parts[k]}" for k in sorted(parts))
    return f"{kind}:{rendered}"


def cached(kind: str, cache, compute, ttl_s: int, **parts):
    """Read through a cache that never fails a request when it is unavailable."""
    key = cache_key(kind, **parts)
    try:
        hit = cache.get(key)
        if hit is not None:
            return hit, True
    except Exception as exc:
        log.warning("cache read failed for %s: %s", key, exc)
    value = compute()
    try:
        cache.set(key, value, ttl=ttl_s)
    except Exception as exc:
        log.warning("cache write failed for %s: %s", key, exc)
    return value, False
```

The consistency benefit is as valuable as the cost saving. A cached place resolution guarantees that "the station" means the same station on turn four as on turn one, which users notice far more than they notice latency. The keying strategy and its pitfalls are developed in [caching spatial tool results across agent turns](/geospatial-prompt-engineering-tool-routing/cost-and-latency-budgets-for-spatial-agents/caching-spatial-tool-results-across-agent-turns/).

### 6. Charge the context, not just the calls

The largest token cost in a multi-turn spatial conversation is usually the context being re-sent, and it grows with every tool result added to the history. A budget that counts only the tokens a tool returns will be wrong by the number of turns.

```python
def turn_token_cost(history_tokens: int, new_tool_tokens: int,
                    generation_tokens: int) -> int:
    """Everything the provider will bill for this turn."""
    return history_tokens + new_tool_tokens + generation_tokens


def prune_history(history: list[dict], keep_tokens: int, count_tokens) -> list[dict]:
    """Keep the most recent and the load-bearing; summarise the rest."""
    kept, used = [], 0
    for entry in reversed(history):
        cost = count_tokens(entry["text"])
        if used + cost > keep_tokens and not entry.get("pinned"):
            continue
        kept.append(entry)
        used += cost
    return list(reversed(kept))
```

Pinning matters more than recency for spatial conversations. The resolved place, the chosen collection and the region are load-bearing for every subsequent turn, while an intermediate geometry result from three turns ago usually is not.

### 7. Estimate the plan's cost in the prompt, not only in the code

An agent that cannot see the budget cannot plan within it. Putting the remaining budget and the tool cost table into the context — briefly — changes the plans a model proposes, and it is much cheaper than rejecting plans repeatedly. The estimation approach is set out in [estimating token cost of a geoprocessing plan](/geospatial-prompt-engineering-tool-routing/cost-and-latency-budgets-for-spatial-agents/estimating-token-cost-of-a-geoprocessing-plan/).

```python
def budget_hint(budget: Budget, spend: Spend) -> str:
    """A short, honest statement of what is left, for the model's context."""
    return (f"Budget remaining: {max(0, budget.tokens - spend.tokens)} tokens, "
            f"{max(0.0, budget.seconds - spend.seconds):.1f}s. "
            "Prefer cached results and fewer tool calls when the budget is tight.")
```

### 8. Report the spend with the answer

The spend belongs in the response metadata, not only in a metric. It is what lets a caller decide whether to retry with a larger budget, and what lets an operator see which question shapes are expensive without instrumenting each one separately.

```python
def to_response_meta(budget: Budget, spend: Spend, cache_hits: int,
                     substitutions: list[str]) -> dict:
    return {
        "tokens_used": spend.tokens,
        "seconds_used": round(spend.seconds, 2),
        "within_budget": spend.within(budget),
        "cache_hits": cache_hits,
        "substitutions": substitutions,
    }
```

## Operating This Stage Over Time

Cost models go stale in one direction: everything gets more expensive as data grows. A vector query measured at 450 milliseconds against a two-million-chunk corpus will not hold at twenty million, and a plan admitted on the old figures will overrun without anything having changed in the agent. Refresh the cost table from observed latencies on a schedule, and treat a large drift as a signal to look at the tool rather than at the table.

The second drift is in the cache hit rate, and it moves for reasons that have nothing to do with caching. A change in how regions are computed will change the rounded extents used as keys, and a hit rate that falls from seventy per cent to five overnight is almost always a key-shape change rather than a traffic change. Track the rate per cache kind, not in aggregate, or the collapse of one cache disappears into the average of the others.

The third is budget creep, and it happens one incident at a time. Each individual increase is justified by a specific slow case, and the cumulative effect is a system that takes eight seconds and costs four times what it did. Review the budget as a whole periodically against what a user will actually wait for, rather than only when something times out.

Finally, be careful about optimising the wrong thing. Generation is the most visible cost and usually the smallest; the two larger ones — tool latency and re-sent context — are less obvious and far more responsive to effort. Measuring the split once, on real traffic, redirects most teams' optimisation work immediately.

## Failure Modes & Root Causes

**The unbounded plan.** An agent proposes eleven tool calls and executes all of them. Root cause: no admission step. Mitigation: estimate and admit before running, with substitutions offered on rejection.

**The invisible context bill.** Token cost grows every turn as tool results accumulate, and nobody notices until the conversation hits a window limit. Root cause: counting only new tokens. Mitigation: charge the history, prune with pinning.

**The cache that never hits.** Keys include a timestamp, an unrounded extent or an unsorted parameter list, so every request misses. Root cause: keys built from whatever was to hand. Mitigation: deterministic key construction, tested.

**The stale cost table.** Plans are admitted on figures measured a year ago against a corpus a tenth of the size. Root cause: a table treated as configuration rather than as a measurement. Mitigation: scheduled refresh from observed latencies, with a drift alert.

## Production Validation Protocols

1. **Admission assertion.** Assert no plan runs without passing admission; a bypass path is the first thing to appear under deadline pressure.
2. **Estimate accuracy check.** Compare estimated against actual spend on a sample of turns and alert when the median error exceeds 30%; a cost model that is systematically wrong is worse than none.
3. **Key determinism test.** Assert that the same logical request produces the same cache key across processes, including after a restart.
4. **Hit-rate indicator per kind.** Publish cache hit rates separately for place, region and raster caches, and alert on a step change.
5. **Budget-overrun rate.** Track the share of turns that exceed budget despite admission; a rising rate means the cost model has drifted.
6. **Pinned-context test.** Assert the resolved place and chosen collection survive history pruning; losing them is what makes a long conversation start contradicting itself.

<figure class="diagram">
<svg viewBox="16 38 728 208" role="img" aria-labelledby="clb-cache-t clb-cache-d" xmlns="http://www.w3.org/2000/svg"><title id="clb-cache-t">Cache keys that hit and keys that cannot</title><desc id="clb-cache-d">A key built from a rounded extent and a version stamp is stable across requests, while one containing a timestamp or an unrounded extent misses every time.</desc><rect x="16" y="38" width="728" height="208" fill="#ffffff"/><rect x="30" y="52" width="700" height="52" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="50" y="84" fill="#1f2937" font-size="12.5">region:v3:-3.200,55.940,-3.180,55.960 — rounded, versioned, sorted: hits</text><rect x="30" y="116" width="700" height="52" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="50" y="148" fill="#1f2937" font-size="12.5">region:-3.19999831,55.94000112,… — full precision: never hits twice</text><rect x="30" y="180" width="700" height="52" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="50" y="212" fill="#1f2937" font-size="12.5">region:1723380142:… — a timestamp in the key: misses by construction</text></svg>
<figcaption><b>Two of these three caches are decoration.</b> Neither reports an error, both consume storage, and the hit rate is the only thing that distinguishes them from the one that works.</figcaption>
</figure>

Of these six, the estimate-accuracy check is the one that keeps the whole mechanism honest. Admission is only as good as the numbers it admits against, and a cost model that has drifted produces a gate that passes everything — which looks exactly like a gate that is working, since nothing is being rejected.

The pinned-context test is the second worth building early. Losing the resolved place from a pruned history is the failure that makes a long conversation start contradicting itself, and it is invisible in every cost metric because the pruning did exactly what it was asked to do.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should the model see the exact budget numbers?</span></summary><p>The remaining amounts, yes, and briefly. A model told it has nine hundred tokens and two seconds left proposes materially different plans from one told nothing, and the hint costs about thirty tokens. What is not worth including is the full cost table for every tool — it is long, it changes, and the model does not need per-tool figures to prefer fewer calls.</p></details>

<details class="faq-item"><summary><span>How accurate does the cost model need to be?</span></summary><p>Within a factor of two is enough to reject the plans worth rejecting. The estimate is a gate, not a forecast: it exists to catch the eleven-call plan, not to predict a four-call plan to the millisecond. Precision beyond that is effort spent on a number whose main job is comparison against a budget that was itself chosen by judgement.</p></details>

<details class="faq-item"><summary><span>What should happen when even the cheapest plan does not fit?</span></summary><p>Refuse with the reason and the shortfall, and offer the larger budget as an option where the caller can grant one. That is a much better outcome than running the plan anyway and timing out halfway, which spends the whole budget and produces nothing. For interactive use it is also the moment to ask whether the question can be narrowed, since a smaller region usually makes the plan fit.</p></details>

<details class="faq-item"><summary><span>Is caching safe when the underlying data changes?</span></summary><p>With a version stamp in the key, yes, and without one, no. The stamp is what makes a data correction invalidate the affected entries automatically rather than serving a superseded answer until the time-to-live expires. Choose the time-to-live from how quickly the data actually changes rather than from a round number, and set it to zero for anything where a stale answer would be consequential.</p></details>

<details class="faq-item"><summary><span>How does this interact with the fallback ladder?</span></summary><p>The budget sets the deadline the ladder walks within, and the ladder's admission control is the same mechanism applied per rung. Keeping them consistent matters: a ladder whose rungs have their own timeouts, inside an agent with a turn budget, will overrun the budget by the number of rungs. One deadline, spent down, is the arrangement that composes.</p></details>

## Related

- Up to the section overview: [Geospatial Prompt Engineering and Tool Routing](/geospatial-prompt-engineering-tool-routing/)
- Technique: [Estimating Token Cost of a Geoprocessing Plan](/geospatial-prompt-engineering-tool-routing/cost-and-latency-budgets-for-spatial-agents/estimating-token-cost-of-a-geoprocessing-plan/)
- Technique: [Caching Spatial Tool Results Across Agent Turns](/geospatial-prompt-engineering-tool-routing/cost-and-latency-budgets-for-spatial-agents/caching-spatial-tool-results-across-agent-turns/)
- Peer topic: [Multi-Step Spatial Agent Orchestration](/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/)
- Related topic: [Fallback Routing for Geospatial Queries](/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/)
- Related topic: [Context-Window Optimization for Maps](/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/)
