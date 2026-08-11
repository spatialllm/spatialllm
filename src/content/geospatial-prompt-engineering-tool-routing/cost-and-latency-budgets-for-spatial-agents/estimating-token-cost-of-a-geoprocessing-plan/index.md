---
title: Estimating Token Cost of a Geoprocessing Plan
description: Cost a proposed plan before running it — per-tool measurements, the re-sent context nobody counts, and a rejection that tells the agent how to make the plan smaller.
slug: estimating-token-cost-of-a-geoprocessing-plan
type: howto
breadcrumb: Estimating Plan Cost
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Estimating Token Cost of a Geoprocessing Plan

Cost measured while a plan runs is a report; cost estimated before it runs is a control. This guide builds the estimate, including the part that is almost always omitted — the context re-sent on every turn — and turns an over-budget plan into a specific instruction for making it smaller, as the admission step of [cost and latency budgets for spatial agents](/geospatial-prompt-engineering-tool-routing/cost-and-latency-budgets-for-spatial-agents/).

## When to Use This Approach

Estimate before every multi-step plan. A single tool call does not need it; a plan of four or more calls does, because that is where the cost becomes non-obvious and the cheaper alternative becomes available.

| Cost component | Usually counted? | Share of a spatial turn |
|----------------|------------------|-------------------------|
| Tool result tokens | Yes | Moderate |
| Re-sent conversation context | Rarely | Often the largest |
| Generation | Yes | Small |
| Tool latency | Sometimes | Dominates wall clock |
| Retries and fallbacks | Almost never | Spiky |

The second row is the one that changes conclusions. A five-turn conversation re-sends everything four times, so a tool result added on turn one is billed five times, and a plan judged only on its own output tokens is understated by a factor that grows with the conversation.

<figure class="diagram">
<svg viewBox="16 24 722 210" role="img" aria-labelledby="etc-resend-t etc-resend-d" xmlns="http://www.w3.org/2000/svg"><title id="etc-resend-t">A tool result billed once against billed five times</title><desc id="etc-resend-d">A result added to the conversation on turn one is re-sent on every subsequent turn, so its true cost is the result size multiplied by the number of remaining turns.</desc><rect x="16" y="24" width="722" height="210" fill="#ffffff"/><text x="30" y="62" fill="#5b6471" font-size="12.5">counted</text><rect x="150" y="38" width="110" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="205" y="64" fill="#1f2937" font-size="12" text-anchor="middle">900 tokens</text><text x="30" y="152" fill="#5b6471" font-size="12.5">billed</text><rect x="150" y="128" width="110" height="42" rx="5" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="266" y="128" width="110" height="42" rx="5" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="382" y="128" width="110" height="42" rx="5" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="498" y="128" width="110" height="42" rx="5" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="614" y="128" width="110" height="42" rx="5" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="437" y="196" fill="#5b6471" font-size="12" text-anchor="middle">4 500 tokens billed across five turns</text><text x="390" y="216" fill="#1f2937" font-size="13" text-anchor="middle">A large tool result early in a conversation is the most expensive thing in it</text></svg>
<figcaption><b>Position in the conversation is part of the cost.</b> The same result costs five times more on turn one than on turn five, which is an argument for fetching detail late and summaries early.</figcaption>
</figure>

## Implementation

The estimator sums per-tool costs, adds the projected re-send, and returns a breakdown rather than a total.

```python
import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

log = logging.getLogger("plan_cost")

PESSIMISTIC = {"tokens": 800, "seconds": 1.0}


@dataclass(frozen=True)
class Breakdown:
    tool_tokens: int
    resend_tokens: int
    generation_tokens: int
    seconds: float
    unknown_steps: tuple[str, ...]

    @property
    def total_tokens(self) -> int:
        return self.tool_tokens + self.resend_tokens + self.generation_tokens


def estimate_plan(plan: Sequence[str], costs: Mapping[str, dict],
                  history_tokens: int, expected_remaining_turns: int,
                  generation_tokens: int = 500) -> Breakdown:
    """Cost a plan including the context it will cause to be re-sent."""
    if expected_remaining_turns < 1:
        raise ValueError("expected_remaining_turns must be at least 1")

    tool_tokens, seconds, unknown = 0, 0.0, []
    for step in plan:
        cost = costs.get(step)
        if cost is None:
            unknown.append(step)
            cost = PESSIMISTIC
        tool_tokens += int(cost["tokens"])
        seconds += float(cost["seconds"])

    # Everything this plan adds is re-sent on each remaining turn after this one.
    resend = (history_tokens + tool_tokens) * max(0, expected_remaining_turns - 1)
    if unknown:
        log.info("no cost model for %s — charged the pessimistic default", ", ".join(unknown))
    return Breakdown(tool_tokens, resend, generation_tokens, round(seconds, 2),
                     tuple(unknown))
```

Charging unknown steps a pessimistic default rather than zero is what keeps a newly added tool from escaping the budget. The step with no cost model is, by construction, the one nobody has measured, and assuming it is free is the assumption most likely to be wrong.

The rejection needs to be specific enough to act on, which means naming the expensive step rather than reporting a total.

```python
def admit_or_explain(plan: Sequence[str], costs: Mapping[str, dict],
                     budget_tokens: int, budget_seconds: float,
                     history_tokens: int, turns: int) -> tuple[bool, str]:
    """Admit the plan, or say which step to remove and by how much it is over."""
    b = estimate_plan(plan, costs, history_tokens, turns)
    over_tokens = b.total_tokens - budget_tokens
    over_seconds = b.seconds - budget_seconds
    if over_tokens <= 0 and over_seconds <= 0:
        return True, ""

    worst = max(plan, key=lambda s: costs.get(s, PESSIMISTIC)["tokens"])
    parts = []
    if over_tokens > 0:
        parts.append(f"{over_tokens} tokens over (of which {b.resend_tokens} is re-sent context)")
    if over_seconds > 0:
        parts.append(f"{over_seconds:.1f}s over")
    return False, (f"{'; '.join(parts)}. The largest step is {worst!r}; "
                   "removing or substituting it would bring the plan closest to budget.")
```

Naming the re-send share in the rejection is what teaches the agent the non-obvious lesson. A plan that is over budget mostly because of accumulated history needs a shorter conversation rather than a shorter plan, and those are completely different corrections.

<figure class="diagram">
<svg viewBox="16 24 764 210" role="img" aria-labelledby="etc-break-t etc-break-d" xmlns="http://www.w3.org/2000/svg"><title id="etc-break-t">A breakdown that changes the correction</title><desc id="etc-break-d">Two plans equally over budget: one dominated by its own tool results, the other by re-sent history, and each needs a different change to fit.</desc><rect x="16" y="24" width="764" height="210" fill="#ffffff"/><text x="30" y="62" fill="#5b6471" font-size="12.5">plan A</text><rect x="140" y="38" width="380" height="42" rx="5" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="526" y="38" width="130" height="42" rx="5" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="11.5" text-anchor="middle"><text x="330" y="64">tool results</text><text x="591" y="64">re-sent</text></g><text x="680" y="64" fill="#5b6471" font-size="12">remove a step</text><text x="30" y="152" fill="#5b6471" font-size="12.5">plan B</text><rect x="140" y="128" width="130" height="42" rx="5" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="276" y="128" width="380" height="42" rx="5" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="11.5" text-anchor="middle"><text x="205" y="154">tools</text><text x="466" y="154">re-sent history</text></g><text x="690" y="154" fill="#5b6471" font-size="12">prune history</text><text x="390" y="216" fill="#1f2937" font-size="13" text-anchor="middle">The same overrun, two different fixes — a total cannot distinguish them</text></svg>
<figcaption><b>The breakdown is the actionable part.</b> Reporting one number tells the agent to try a smaller plan, which is the right correction half the time and wasted effort the other half.</figcaption>
</figure>

## Validation & Testing

```python
COSTS = {"resolve_place": {"tokens": 120, "seconds": 0.15},
         "vector_query": {"tokens": 900, "seconds": 0.45}}


def test_resend_dominates_a_long_conversation():
    short = estimate_plan(["vector_query"], COSTS, history_tokens=2000, expected_remaining_turns=1)
    long = estimate_plan(["vector_query"], COSTS, history_tokens=2000, expected_remaining_turns=6)
    assert long.total_tokens > short.total_tokens * 3


def test_unknown_step_is_charged_and_named():
    b = estimate_plan(["mystery_tool"], COSTS, 0, 1)
    assert b.tool_tokens == PESSIMISTIC["tokens"] and b.unknown_steps == ("mystery_tool",)


def test_rejection_names_the_largest_step():
    ok, why = admit_or_explain(["resolve_place", "vector_query"], COSTS,
                               budget_tokens=100, budget_seconds=10, history_tokens=0, turns=1)
    assert not ok and "vector_query" in why


def test_zero_turns_is_rejected():
    try:
        estimate_plan(["resolve_place"], COSTS, 0, 0)
    except ValueError:
        return
    raise AssertionError("a plan with no remaining turns is a programming error")
```

The first test encodes the property this whole guide exists for. It fails immediately if someone simplifies the estimator to count only tool tokens, which is the simplification everyone reaches for because it is the only part that is obviously attributable to the plan.

Compare estimates against actuals on a sample of real turns and publish the median error. An estimator that is systematically low by a factor of two is worse than none, because it admits plans that then overrun and erodes trust in the whole mechanism.

Keep the cost table beside the tool definitions rather than in a separate configuration file. A tool added without a cost entry is the case the pessimistic default exists for, and having the two adjacent makes the omission visible in review rather than at runtime.

## Gotchas & Edge Cases

**Costs measured on a small corpus.** A vector query that returns nine hundred tokens against two million chunks returns more against twenty million. Refresh the table from observed usage rather than from a one-off measurement.

**Retries excluded from the estimate.** A plan with a step that fails half the time costs 1.5 times its nominal estimate. Where a step's failure rate is known and material, fold it into the cost rather than pretending retries are free.

**Generation treated as negligible.** It is small relative to context on a spatial turn and is not zero, and a plan that produces a long structured answer generates considerably more than one that returns a sentence. Estimate it from the answer shape rather than using a constant everywhere.

<figure class="diagram">
<svg viewBox="16 24 684 206" role="img" aria-labelledby="etc-retry-t etc-retry-d" xmlns="http://www.w3.org/2000/svg"><title id="etc-retry-t">A step with a known failure rate costs more than its nominal price</title><desc id="etc-retry-d">A step that fails half the time and is retried once costs one and a half times its nominal estimate, which a plan estimate that ignores retries understates.</desc><rect x="16" y="24" width="684" height="206" fill="#ffffff"/><text x="30" y="62" fill="#5b6471" font-size="12.5">nominal</text><rect x="170" y="38" width="220" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="280" y="64" fill="#1f2937" font-size="12" text-anchor="middle">900 tokens</text><text x="30" y="152" fill="#5b6471" font-size="12.5">observed</text><rect x="170" y="128" width="220" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="396" y="128" width="110" height="42" rx="5" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="280" y="154" fill="#1f2937" font-size="12" text-anchor="middle">900</text><text x="451" y="154" fill="#1f2937" font-size="12" text-anchor="middle">+450 retries</text><text x="380" y="212" fill="#1f2937" font-size="13" text-anchor="middle">Fold a known failure rate into the cost rather than treating retries as free</text></svg>
<figcaption><b>Retries are budgeted or they are unbudgeted.</b> A step known to fail half the time has a real cost of one and a half attempts, and an estimator that prices it at one will admit plans that reliably overrun.</figcaption>
</figure>

**Turn count guessed pessimistically.** Assuming ten remaining turns makes almost every plan over budget. Estimate it from observed conversation lengths, and default low — an underestimate produces a plan that fits and a slightly larger bill, while an overestimate blocks work that would have been fine.

**Latency and tokens conflated into one score.** A single blended cost hides which budget bound, and the two have different remedies — a slow plan may be queued while an expensive one must shrink. Report them separately, name which one bound, and let the caller decide between trimming the plan and deferring it to the queue.

**Estimates cached against a stale table.** Caching plan estimates saves nothing worth having and guarantees the cache outlives a cost-table refresh. Estimate fresh; it is arithmetic over a handful of entries.

## Frequently Asked Questions

<details class="faq-item"><summary><span>How accurate does the estimate need to be?</span></summary><p>Within a factor of two, which is easily achieved and is enough to reject the plans worth rejecting. The estimate is a gate rather than a forecast: it exists to catch the eleven-step plan and the one that adds twelve thousand tokens to a conversation, not to predict a three-step plan to the token. Effort spent past that accuracy is effort not spent on the cost table, which is where the real error lives.</p></details>

<details class="faq-item"><summary><span>Where should the expected turn count come from?</span></summary><p>The observed distribution of conversation lengths for your workload, taken at a low percentile rather than the mean. Conversations have a long tail, and estimating against the tail makes every plan look unaffordable. Using the median remaining turns, recomputed occasionally, is a reasonable default that adapts as usage changes.</p></details>

<details class="faq-item"><summary><span>Should the estimate include the cost of the estimate?</span></summary><p>No — it is arithmetic over a small table and costs nothing measurable. What is worth including is the cost of any model call used to produce the plan in the first place, which is real and is frequently forgotten because it happens before the plan exists. Attribute it to the turn rather than to the plan.</p></details>

<details class="faq-item"><summary><span>What should happen to a plan that is over budget on time but not tokens?</span></summary><p>Route it to the queue rather than trimming it. A plan that costs little and takes long is exactly the shape the async path exists for, and shrinking it to fit an interactive budget throws away work the user wanted. Reporting which of the two budgets bound is what makes that routing decision possible.</p></details>

<details class="faq-item"><summary><span>Should the estimate be shown to the user?</span></summary><p>Not usually, and the exception is worth having: when a plan is rejected and the user can grant a larger budget, telling them roughly what it would cost turns an unexplained refusal into a choice. Keep it approximate — &#8220;this would take about a minute&#8221; rather than a token count — since the underlying numbers are internal and the precision would be misleading anyway.</p></details>

## Related

- Up to the parent topic: [Cost and Latency Budgets for Spatial Agents](/geospatial-prompt-engineering-tool-routing/cost-and-latency-budgets-for-spatial-agents/)
- [Caching Spatial Tool Results Across Agent Turns](/geospatial-prompt-engineering-tool-routing/cost-and-latency-budgets-for-spatial-agents/caching-spatial-tool-results-across-agent-turns/)
- Related topic: [Context-Window Optimization for Maps](/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/)
- Related topic: [Async vs Sync Geoprocessing Workflows](/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/)
