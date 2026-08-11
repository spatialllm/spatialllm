---
title: Choosing Between Queued and Inline Geoprocessing
description: Decide before running whether an operation answers in the conversation or becomes a job with a handle — from an estimate, a budget and a stated threshold rather than by waiting.
slug: choosing-between-queued-and-inline-geoprocessing
type: howto
breadcrumb: Queued or Inline
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Choosing Between Queued and Inline Geoprocessing

The decision that determines whether a spatial agent feels responsive is made before any work starts: does this operation answer inside the conversation, or does it become a job the reader collects later? Getting it wrong in one direction produces a four-minute silence; getting it wrong in the other produces a handle for something that would have finished in two seconds. This guide covers making that call from an estimate — the practical half of [async and synchronous geoprocessing workflows](/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/).

## When to Use This Approach

Any system where operation cost varies by more than an order of magnitude needs an explicit boundary. If every operation takes roughly the same time, pick one path and stop reading.

| Operation | Typical cost | Path |
|-----------|--------------|------|
| Place resolution, attribute lookup | Milliseconds | Inline, always |
| Region query over an indexed column | Under a second | Inline |
| Buffer and intersect at city scale | Seconds | Inline, with a budget check |
| Overlay across a national dataset | Minutes | Queued |
| Anything the reader explicitly batched | Unbounded | Queued |

<figure class="diagram">
<svg viewBox="16 38 748 212" role="img" aria-labelledby="cbq-two-t cbq-two-d" xmlns="http://www.w3.org/2000/svg"><title id="cbq-two-t">Deciding from an estimate against discovering by waiting</title><desc id="cbq-two-d">An estimate taken before execution lets the path be chosen and the reader told what to expect, while waiting and then giving up spends the budget before learning anything.</desc><rect x="16" y="38" width="748" height="212" fill="#ffffff"/><rect x="30" y="52" width="340" height="160" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="410" y="52" width="340" height="160" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="200" y="84">decide from an estimate</text><text x="580" y="84">discover by waiting</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="200" y="114">the path is chosen up front</text><text x="200" y="140">the reader is told what to expect</text><text x="200" y="166">a timeout is a bug, not a design</text><text x="580" y="114">the budget is spent first</text><text x="580" y="140">the reader waits, then fails</text><text x="580" y="166">the timeout is the mechanism</text></g><text x="390" y="234" fill="#1f2937" font-size="13" text-anchor="middle">A timeout tells you what a query was not — an estimate tells you what it is</text></svg>
<figcaption><b>The timeout is the expensive way to learn.</b> By the time it fires the budget is gone, the reader has waited the full interval, and the only information gained is that the operation took longer than a number somebody picked.</figcaption>
</figure>

## Implementation

The estimator needs three inputs and produces one number. Row counts come from table statistics, which are free. A per-operation cost coefficient comes from measurements you have already taken. The output count is projected from the operation's own semantics — a buffer preserves the count, an intersection reduces it, a dissolve collapses it.

```python
COEFFICIENTS = {           # measured seconds per thousand features, on this deployment
    "buffer": 0.04,
    "intersect": 0.31,
    "dissolve": 0.55,
    "nearest": 0.12,
}


def estimate_seconds(plan: list[Step], stats: Stats) -> float:
    total, rows = 0.0, None
    for step in plan:
        rows = stats.row_estimate(step) if rows is None else project(step, rows)
        total += COEFFICIENTS.get(step.op, 0.20) * (rows / 1000)
    return total
```

The threshold is not a constant either. It is whatever remains of the interactive budget after the model's own turn, which means an operation that would run inline at the start of a conversation may not at the end of a long one.

```python
def choose_path(plan, stats, budget: Budget) -> str:
    projected = estimate_seconds(plan, stats)
    headroom = budget.remaining_seconds() - RESPONSE_RESERVE
    if projected > headroom:
        return "queued"
    if projected > headroom * 0.6:      # close enough that a bad estimate hurts
        return "queued"
    return "inline"
```

The margin in that second branch matters more than it looks. Estimates are wrong, and they are wrong asymmetrically: an operation that overruns an inline budget produces a timeout, while one that is queued unnecessarily produces a handle for something that finishes immediately. The second outcome is much cheaper, so the threshold should sit below the true boundary rather than at it.

<figure class="diagram">
<svg viewBox="5 38 770 220" role="img" aria-labelledby="cbq-band-t cbq-band-d" xmlns="http://www.w3.org/2000/svg"><title id="cbq-band-t">Where the threshold should sit relative to the budget</title><desc id="cbq-band-d">Operations well inside the budget run inline, those near the boundary are queued because estimate error is more costly there, and those beyond it are always queued.</desc><rect x="5" y="38" width="770" height="220" fill="#ffffff"/><rect x="30" y="52" width="700" height="46" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="50" y="80" fill="#1f2937" font-size="12.5">well inside the budget: run inline, the estimate can be wrong and it still fits</text><rect x="30" y="108" width="620" height="46" rx="6" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="50" y="136" fill="#1f2937" font-size="12.5">near the boundary: queue it — a wrong estimate here costs a timeout</text><rect x="30" y="164" width="700" height="46" rx="6" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="50" y="192" fill="#1f2937" font-size="12.5">beyond the budget: queue it, and say so before the reader starts waiting</text><text x="390" y="240" fill="#1f2937" font-size="13" text-anchor="middle">Queuing something that would have been fast costs a handle; the reverse costs the whole turn</text></svg>
<figcaption><b>The two errors are not symmetric.</b> Because queuing something fast is nearly free and running something slow inline is not, the correct threshold sits deliberately below the point where the two paths actually cross.</figcaption>
</figure>

## Validation & Testing

Test the estimator against reality rather than against itself. Record the projection alongside the measured duration for every operation and compare the two over a few thousand runs; what matters is not the mean error but the tail, because it is the underestimates that produce timeouts.

```python
def test_estimates_do_not_underestimate_badly():
    records = load_execution_log(days=7)
    ratios = [r.actual / r.projected for r in records if r.projected > 0]
    ratios.sort()
    p95 = ratios[int(len(ratios) * 0.95)]
    assert p95 < 2.5, f"95th percentile underestimate factor {p95:.1f} — margin too thin"
```

Test the boundary itself with two fixtures: an operation just under the threshold, which must run inline and return a result, and one just over, which must return a handle. Both should be checked for the thing that is easy to get wrong — that the queued path calls exactly the same implementation with exactly the same defaults as the inline one.

## Gotchas & Edge Cases

**A queued path with different defaults.** The most confusing bug this design can produce is an answer that differs depending on how long the operation was estimated to take. It happens when the worker is a separate entry point that has drifted — a different tolerance, a different projection, a different feature limit. Both paths must call one function.

**Stale coefficients.** Measured on one deployment, kept after a database upgrade or a hardware change, they produce systematically wrong routing that nothing reports. Re-measure after any infrastructure change, and treat a shift in the projection-to-actual ratio as the signal.

**Estimates that ignore the data distribution.** A row count says nothing about geometry complexity, and a thousand-vertex coastline costs far more than a thousand rectangles. Where a dataset is known to be complex, a per-table multiplier is cruder than modelling vertex counts and considerably more likely to be maintained.

**Queuing without telling the reader.** A handle that arrives with no indication of expected duration is worse than a wait, because the reader has no basis for deciding whether to stay. The estimate that drove the decision is exactly the number to report.

**The reader who wanted a rough answer.** Some questions do not need the expensive path at all. Offering a sampled or simplified result inline, with the exact one queued, resolves more requests than either path alone — and the estimate is what makes the offer possible.

<figure class="diagram">
<svg viewBox="16 38 748 212" role="img" aria-labelledby="cbq-tell-t cbq-tell-d" xmlns="http://www.w3.org/2000/svg"><title id="cbq-tell-t">What the reader is told in each case</title><desc id="cbq-tell-d">An inline result arrives directly, a queued job reports its estimate and a handle, and an expensive request can be offered a sampled answer immediately with the exact one following.</desc><rect x="16" y="38" width="748" height="212" fill="#ffffff"/><g><rect x="30" y="52" width="228" height="150" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="276" y="52" width="228" height="150" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="522" y="52" width="228" height="150" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/></g><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="144" y="84">inline</text><text x="390" y="84">queued</text><text x="636" y="84">offered both</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="144" y="116">the answer, directly</text><text x="144" y="142">no mention of paths</text><text x="144" y="168">nothing to collect</text><text x="390" y="116">an estimate, in words</text><text x="390" y="142">a handle to collect</text><text x="390" y="168">a mention on the next turn</text><text x="636" y="116">a sampled answer now</text><text x="636" y="142">the exact one queued</text><text x="636" y="168">most readers stop here</text></g><text x="390" y="234" fill="#1f2937" font-size="13" text-anchor="middle">The third column resolves more requests than either of the first two</text></svg>
<figcaption><b>The estimate is what makes the third option possible.</b> Without a projection there is no basis for offering a cheaper approximation, so every expensive request becomes a wait regardless of whether the reader needed precision.</figcaption>
</figure>

## Recording the Decision

Every routing decision should leave a record containing the projection, the headroom it was compared against, the path chosen and, once available, the measured duration. Those four fields are enough to answer every question anyone will later ask about the boundary, and they cost a row.

The record is what makes the threshold tunable from evidence. Without it, the only signal is complaints — a reader who waited too long, or one puzzled by a handle for something trivial — and complaints arrive filtered through whoever happened to notice. With it, the distribution of projected against actual durations is visible directly, and the margin can be set to whatever underestimate rate is actually tolerable rather than to a round number.

It also makes a specific class of bug findable. When a reader reports that the same question behaved differently on two occasions, the decision record shows immediately whether the difference was in the estimate, in the available headroom, or in the data — three quite different problems that are indistinguishable from the outside. Keeping the record for a few weeks is sufficient; this is diagnostic data rather than an audit trail.

## Operating This Step Over Time

The number worth tracking is the rate at which each path is chosen, alongside how often an inline operation exceeded its budget. A rising inline-overrun rate means the coefficients have drifted or the data has grown, and both are fixed by re-measuring rather than by widening the budget.

Watch also for requests that are always queued. A category of question that never answers inline is usually one where the data model, not the estimate, is the problem — a missing index or a table that should have been aggregated — and the queue is absorbing it just well enough that nobody investigates.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should the reader be able to force one path?</span></summary><p>Forcing the queue, yes; forcing inline, no. A reader who knows they are starting something large and wants to walk away is expressing a real preference the estimator cannot see. A reader forcing an expensive operation into the request path is asking for a timeout, and the honest response is the estimate and the queued path anyway.</p></details>

<details class="faq-item"><summary><span>What if there is no estimate for a new operation?</span></summary><p>Queue it until there is one. A default coefficient is a guess that will be wrong in whichever direction is least convenient, and the first few dozen executions of a new operation are exactly the measurements needed to set it properly. Queuing by default makes the unknown case safe rather than surprising.</p></details>

<details class="faq-item"><summary><span>How does this interact with caching?</span></summary><p>Check the cache before estimating, because a hit makes the whole decision moot. The more interesting case is a partial hit, where some steps of a plan are cached and the estimate should only cover the rest — which is a good reason for the estimator to work step by step rather than over the plan as a whole.</p></details>

<details class="faq-item"><summary><span>Does the model need to know which path was taken?</span></summary><p>Only enough to phrase the response. Told that a result is available it composes an answer; told that a job was started with an estimated duration it says so. What it should not receive is the estimate as a number to reason about, because it will convert it into a promise the system has not made.</p></details>

<details class="faq-item"><summary><span>Should a queued job ever be promoted back to inline?</span></summary><p>Not usefully. By the time the handle has been issued the reader has already been told to expect a wait, and delivering the result immediately afterwards is a pleasant surprise rather than a design goal. What is worth doing is checking the result store before the next turn, so that a job which finished quickly gets mentioned in the very next exchange rather than waiting to be asked about.</p></details>

## Related

- Up to the parent topic: [Async vs Sync Geoprocessing Workflows](/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/)
- [Handling Async Spatial Processing in Python Workflows](/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/handling-async-spatial-processing-in-python-workflows/)
- [Backpressure and Rate Limiting for Spatial API Calls](/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/backpressure-and-rate-limiting-for-spatial-api-calls/)
- Related topic: [Cost and Latency Budgets for Spatial Agents](/geospatial-prompt-engineering-tool-routing/cost-and-latency-budgets-for-spatial-agents/)
