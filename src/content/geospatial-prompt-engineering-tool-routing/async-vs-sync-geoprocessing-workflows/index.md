---
title: Async vs Sync Geoprocessing Workflows
description: Decide which spatial operations answer inline and which become jobs, then build the queue, the backpressure and the progress reporting that make the async path usable.
slug: async-vs-sync-geoprocessing-workflows
type: topic
breadcrumb: Async and Sync Workflows
datePublished: 2025-03-04
dateModified: 2026-08-11
---

# Async vs Sync Geoprocessing Workflows

Some spatial operations finish in forty milliseconds and some take four minutes, and the same agent has to handle both without making the fast ones slow or the slow ones invisible. The decision of which is which is not a performance detail — it changes the shape of the conversation, because an operation that becomes a job needs a handle, a progress story and a way to deliver its result to a user who has moved on.

This topic belongs to [geospatial prompt engineering and tool routing](/geospatial-prompt-engineering-tool-routing/) and operates under the constraints set by [cost and latency budgets for spatial agents](/geospatial-prompt-engineering-tool-routing/cost-and-latency-budgets-for-spatial-agents/). Its failure modes overlap with [fallback routing for geospatial queries](/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/), which decides what to do when neither path can answer.

<figure class="diagram">
<svg viewBox="16 42 748 188" role="img" aria-labelledby="avs-split-t avs-split-d" xmlns="http://www.w3.org/2000/svg"><title id="avs-split-t">Where the boundary between inline and queued sits</title><desc id="avs-split-d">Operations under the interactive budget answer inline; operations above it become jobs with a handle, and the middle band is the one that needs an explicit policy rather than a guess.</desc><rect x="16" y="42" width="748" height="188" fill="#ffffff"/><rect x="30" y="56" width="250" height="60" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="296" y="56" width="200" height="60" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="512" y="56" width="238" height="60" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="155" y="84">under 2 s</text><text x="396" y="84">2 s to 30 s</text><text x="631" y="84">over 30 s</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="155" y="106">inline, always</text><text x="396" y="106">policy decides</text><text x="631" y="106">queued, always</text></g><rect x="296" y="140" width="200" height="76" rx="8" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><text x="396" y="166" fill="#1f2937" font-size="12.5" text-anchor="middle">the band that needs a rule</text><text x="396" y="188" fill="#5b6471" font-size="12" text-anchor="middle">estimated size, user context,</text><text x="396" y="208" fill="#5b6471" font-size="12" text-anchor="middle">whether the answer is needed now</text></svg>
<figcaption><b>The middle band is where the design lives.</b> Everything under a couple of seconds is obviously inline and everything over half a minute is obviously a job; the interesting decisions are the operations whose cost depends on the input.</figcaption>
</figure>

## Foundational Principles

**The decision is made from an estimate, not from experience.** Whether an operation is inline or queued has to be decided before it runs, from the size of its inputs, because deciding afterwards means the user has already waited.

**A queued job needs a handle the conversation can carry.** An operation that becomes a job produces an identifier, and every subsequent turn must be able to ask about it. Without that the job is fire-and-forget and its result has nowhere to go.

**Backpressure is part of correctness.** A queue with no admission limit accepts work faster than it completes it, and the visible symptom is not an error but a completion time that grows until it is unbounded. Reject at the door and say so.

## Step-by-Step Implementation Pipeline

### 1. Estimate the operation's cost from its inputs

The estimate does not need to be accurate; it needs to sort operations into the right bucket. Feature counts, vertex counts and raster extents are the inputs that predict cost, and they are all cheap to obtain.

```python
import logging
from dataclasses import dataclass
from typing import Literal, Optional

log = logging.getLogger("geoprocessing_mode")

Mode = Literal["inline", "queued"]

INLINE_BUDGET_S = 2.0
ALWAYS_QUEUE_S = 30.0


@dataclass(frozen=True)
class Estimate:
    seconds: float
    basis: str


def estimate_cost(op: str, feature_count: int, vertex_count: int,
                  raster_cells: int = 0) -> Estimate:
    """A rough per-operation cost model, measured rather than assumed."""
    rates = {                                    # seconds per unit, from production timings
        "buffer":    (2.0e-4, 1.0e-6, 0.0),
        "overlay":   (8.0e-4, 4.0e-6, 0.0),
        "dissolve":  (1.5e-3, 6.0e-6, 0.0),
        "zonal":     (1.0e-4, 5.0e-7, 3.0e-8),
    }
    per_feature, per_vertex, per_cell = rates.get(op, (1.0e-3, 5.0e-6, 1.0e-7))
    seconds = (feature_count * per_feature + vertex_count * per_vertex
               + raster_cells * per_cell)
    return Estimate(round(seconds, 3),
                    f"{op}: {feature_count} features, {vertex_count} vertices")
```

### 2. Choose the mode from the estimate and the context

The estimate sorts the obvious cases; the middle band needs the request's own context — whether a user is waiting, how much of their budget is left, and whether the answer is needed to continue.

```python
def choose_mode(est: Estimate, remaining_s: float, interactive: bool) -> tuple[Mode, str]:
    """Decide inline or queued, and say why."""
    if est.seconds >= ALWAYS_QUEUE_S:
        return "queued", f"estimated {est.seconds:.1f}s exceeds the inline ceiling"
    if est.seconds <= INLINE_BUDGET_S and est.seconds < remaining_s:
        return "inline", ""
    if not interactive:
        return "inline", "batch context: waiting is acceptable"
    if est.seconds >= remaining_s:
        return "queued", (f"estimated {est.seconds:.1f}s exceeds the "
                          f"{remaining_s:.1f}s remaining in this turn")
    return "queued", "estimated cost is above the interactive budget"
```

Returning the reason alongside the mode is what lets the agent explain itself. "This will take about forty seconds, so I have started it as a job" is a far better turn than a silent pause followed by a result.

### 3. Admit work into the queue, or refuse at the door

A queue that accepts everything degrades invisibly. Admission control against depth and estimated completion time turns that into an explicit rejection with a number attached.

```python
@dataclass(frozen=True)
class Admission:
    accepted: bool
    reason: str
    estimated_wait_s: float


def admit(queue_depth: int, in_flight_s: float, est: Estimate,
          max_depth: int = 200, max_wait_s: float = 600.0) -> Admission:
    """Accept work only when it can complete in a time worth waiting for."""
    if queue_depth >= max_depth:
        return Admission(False, f"queue is full ({queue_depth} jobs)", in_flight_s)
    wait = in_flight_s + est.seconds
    if wait > max_wait_s:
        return Admission(False,
                         f"estimated completion in {wait / 60:.0f} min exceeds the limit",
                         wait)
    return Admission(True, "", wait)
```

Reporting the estimated wait even on rejection is what makes the refusal actionable: a user told "about twenty minutes" can decide to narrow the region, and one told "the queue is full" cannot. The rate-limiting mechanics are developed in [backpressure and rate limiting for spatial API calls](/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/backpressure-and-rate-limiting-for-spatial-api-calls/).

<figure class="diagram">
<svg viewBox="16 24 764 160" role="img" aria-labelledby="avs-queue-t avs-queue-d" xmlns="http://www.w3.org/2000/svg"><title id="avs-queue-t">A queue without admission control against one with it</title><desc id="avs-queue-d">Accepting every job produces a queue whose completion time grows without bound and without error; rejecting at the door keeps waits predictable and makes the constraint visible.</desc><rect x="16" y="24" width="764" height="160" fill="#ffffff"/><text x="30" y="62" fill="#b3324f" font-size="13" font-weight="600">no admission</text><rect x="200" y="38" width="90" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="296" y="38" width="140" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="442" y="38" width="290" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="11.5" text-anchor="middle"><text x="245" y="64">2 min</text><text x="366" y="64">9 min</text><text x="587" y="64">41 min, still accepting</text></g><text x="30" y="152" fill="#12805c" font-size="13" font-weight="600">with admission</text><rect x="200" y="128" width="140" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="346" y="128" width="150" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="502" y="128" width="120" height="42" rx="5" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><g fill="#1f2937" font-size="11.5" text-anchor="middle"><text x="270" y="154">6 min</text><text x="421" y="154">9 min</text><text x="562" y="154">rejected</text></g><text x="650" y="154" fill="#5b6471" font-size="12">with a stated wait</text></svg>
<figcaption><b>Unbounded queues fail silently.</b> Nothing errors as the wait grows; jobs simply take longer, users stop trusting the feature, and the metric that would have shown it is the one nobody added.</figcaption>
</figure>

### 4. Give every job a handle and a status

The handle is what the conversation carries. It needs to be stable, short enough to appear in a sentence, and resolvable to a status without the original context.

```python
@dataclass(frozen=True)
class Job:
    job_id: str
    op: str
    status: Literal["queued", "running", "done", "failed", "cancelled"]
    submitted_at: float
    estimated_s: float
    result_ref: Optional[str] = None
    error: Optional[str] = None


def status_sentence(job: Job, now: float) -> str:
    """What the agent tells the user about this job, in one line."""
    waited = now - job.submitted_at
    if job.status == "queued":
        return (f"Job {job.job_id} is queued; estimated {job.estimated_s:.0f}s of work, "
                f"waiting {waited:.0f}s so far.")
    if job.status == "running":
        return f"Job {job.job_id} is running, {waited:.0f}s elapsed."
    if job.status == "done":
        return f"Job {job.job_id} finished; the result is available."
    if job.status == "cancelled":
        return f"Job {job.job_id} was cancelled."
    return f"Job {job.job_id} failed: {job.error or 'no reason recorded'}."
```

### 5. Make the inline path cancellable too

An inline operation that exceeds its estimate is the worst case: the user is waiting and nothing is watching. Running it under the turn's deadline, with cancellation, converts that into a bounded failure that can be re-offered as a job.

```python
def run_inline(op_callable, est: Estimate, deadline_s: float):
    """Run inline under the deadline; on overrun, offer the queued path instead."""
    if est.seconds > deadline_s:
        raise WouldExceedDeadline(
            f"estimated {est.seconds:.1f}s against {deadline_s:.1f}s available")
    try:
        return op_callable(timeout=deadline_s)
    except TimeoutError as exc:
        log.info("inline operation exceeded its estimate: %s", exc)
        raise WouldExceedDeadline("the operation ran longer than estimated") from exc
```

### 6. Deliver results to a conversation that has moved on

A job finishing after the user has asked three other questions needs somewhere to go. The workable pattern is a result reference the agent can mention on the next turn, plus a pinned entry in the conversation state so the handle survives history pruning.

```python
def pending_jobs_hint(jobs: list[Job], now: float) -> str:
    """A short line for the model's context so it can mention finished work."""
    done = [j for j in jobs if j.status == "done"]
    running = [j for j in jobs if j.status in {"queued", "running"}]
    parts = []
    if done:
        parts.append(f"{len(done)} job(s) finished: " + ", ".join(j.job_id for j in done))
    if running:
        parts.append(f"{len(running)} still running")
    return "; ".join(parts)
```

### 7. Retry jobs, not inline calls

An inline operation that fails has a user waiting and should degrade rather than retry. A queued job has nobody waiting and can retry with backoff, provided the failure is transient and the retry is bounded.

```python
def should_retry(job: Job, attempts: int, error_class: str,
                 max_attempts: int = 3) -> tuple[bool, float]:
    """Retry queued work on transient failures only, with bounded backoff."""
    if job.status != "failed" or attempts >= max_attempts:
        return False, 0.0
    if error_class not in {"transient", "resource"}:
        return False, 0.0
    return True, min(60.0, 2.0 ** attempts)
```

### 8. Report queue health as a first-class metric

Depth, wait time and rejection rate together describe whether the async path is working. Depth alone does not: a short queue that rejects half its arrivals is in worse shape than a long one that accepts everything and completes it.

```python
def queue_health(depth: int, p95_wait_s: float, rejected: int, accepted: int) -> dict:
    total = rejected + accepted
    return {
        "depth": depth,
        "p95_wait_s": round(p95_wait_s, 1),
        "rejection_rate": round(rejected / total, 4) if total else 0.0,
        "healthy": depth < 150 and p95_wait_s < 300 and (not total or rejected / total < 0.05),
    }
```

### 9. Keep the two paths behaviourally identical

The most confusing bug in a dual-path system is an operation that produces one answer inline and a different one through the queue. It happens easily: the worker runs with different defaults, reads a different snapshot, or applies a simplification the inline path does not. The fix is structural — one implementation, invoked two ways.

```python
def execute(op: str, params: dict, snapshot_id: str) -> dict:
    """The single implementation. Both paths call this; neither has its own defaults."""
    return OPERATIONS[op](**params, snapshot=snapshot_id)


def run_any(op: str, params: dict, mode: Mode, snapshot_id: str, enqueue, deadline_s: float):
    """Dispatch by mode without duplicating the operation itself."""
    if mode == "inline":
        return execute(op, params, snapshot_id)
    return enqueue(op=op, params=params, snapshot=snapshot_id)
```

Passing the snapshot identifier through both paths is the detail that matters most. A queued job that runs twenty minutes later against a corpus that has since been rebuilt produces an answer to a slightly different question, and the user has no way to tell — the handle they were given makes it look like the same request.

### 10. Decide what a cancelled job leaves behind

Cancellation is easy to offer and easy to implement badly. A job cancelled mid-write can leave partial output that a later read treats as complete, which is a worse outcome than not offering cancellation at all.

```python
def cancel(job: Job, store) -> Job:
    """Mark cancelled and remove partial output; never leave a half-written result."""
    if job.status in {"done", "failed", "cancelled"}:
        return job
    try:
        if job.result_ref:
            store.delete(job.result_ref)          # partial output is not a result
    except Exception as exc:
        log.warning("could not remove partial output for %s: %s", job.job_id, exc)
    return Job(job.job_id, job.op, "cancelled", job.submitted_at,
               job.estimated_s, None, None)
```

Writing results under a temporary reference and promoting them atomically on completion is the arrangement that makes this safe. Without it, cancellation and failure both leave debris that looks like output, and the read path has to distinguish them by inspecting content — which it will do incorrectly at least once.

## Operating This Stage Over Time

The cost model is the part that ages. Operation timings are measured against a corpus and a machine, and both change; a buffer that took two seconds against a million features will take longer against ten million, and every mode decision made from the old figures will send work down the inline path that no longer belongs there. Refresh the rates from observed durations on a schedule, and alert when the median estimate error exceeds a third.

The second drift is in what users ask for. A system whose questions were originally local acquires regional ones as people discover it works, and the share of operations landing in the queued band grows without anything in the code changing. Track the inline/queued split as a ratio rather than as counts, because the ratio moving is the signal that the workload has changed shape.

The third is a quiet one: jobs that nobody collects. Results delivered to a conversation that ended are storage with no consumer, and they accumulate. Expire them on a schedule tied to how long a conversation plausibly lasts, and count the expirations — a high rate means the delivery path is not working rather than that users are careless.

Finally, keep the two paths behaviourally identical where it matters. An operation that produces one answer inline and a subtly different one through the queue — because the job runs with different defaults, or against a different snapshot — is a bug that only appears at the boundary, and the boundary moves whenever the cost model is refreshed.

## Failure Modes & Root Causes

**The inline operation that was not.** An operation estimated at one second takes ninety because the input was larger than the estimate assumed. Root cause: a cost model that ignores the dominant input. Mitigation: include vertex and cell counts, not just feature counts, and cancel on the deadline.

**The unbounded queue.** Completion times grow until the feature is unusable, with no error at any point. Root cause: no admission control. Mitigation: reject on depth and estimated wait, with the wait reported.

**The orphaned job.** Work completes and the result has nowhere to go because the conversation ended or the handle was pruned. Root cause: handles not pinned in conversation state. Mitigation: pin them, and expire results deliberately.

**The retry storm.** A failing dependency causes every queued job to retry simultaneously, which keeps it failing. Root cause: unbounded retries with no jitter. Mitigation: bounded attempts, exponential backoff, and classification so only transient failures retry.

## Production Validation Protocols

1. **Mode-decision test.** Assert that an input above the queue threshold is never run inline, using a fixture sized to trigger it.
2. **Admission-under-load test.** Assert the queue rejects with a stated wait once depth or projected wait exceeds the limits.
3. **Handle-survival test.** Assert a job handle survives history pruning and is resolvable after the conversation has moved on.
4. **Deadline-cancellation test.** Assert an inline operation that overruns is cancelled and re-offered as a job rather than allowed to run.
5. **Estimate-accuracy indicator.** Compare estimated against actual durations and alert when the median error exceeds a third.
6. **Queue-health publication.** Publish depth, tail wait and rejection rate together; any one alone is misleading.

<figure class="diagram">
<svg viewBox="16 38 728 160" role="img" aria-labelledby="avs-life-t avs-life-d" xmlns="http://www.w3.org/2000/svg"><title id="avs-life-t">The life of a queued job as the conversation continues</title><desc id="avs-life-d">A job is submitted on one turn, runs while the conversation moves on, and its completion is surfaced on a later turn through a pinned handle that survived history pruning.</desc><rect x="16" y="38" width="728" height="160" fill="#ffffff"/><rect x="30" y="52" width="160" height="60" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="206" y="52" width="160" height="60" rx="8" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="382" y="52" width="160" height="60" rx="8" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="558" y="52" width="172" height="60" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="12.5" text-anchor="middle" font-weight="600"><text x="110" y="80">turn 1: submit</text><text x="286" y="80">turn 2</text><text x="462" y="80">turn 3</text><text x="644" y="80">turn 4: deliver</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="110" y="102">handle pinned</text><text x="286" y="102">other questions</text><text x="462" y="102">job running</text><text x="644" y="102">result mentioned</text></g><rect x="30" y="140" width="700" height="44" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="380" y="168" fill="#1f2937" font-size="12.5" text-anchor="middle">the handle survives history pruning because it is pinned, not because it is recent</text></svg>
<figcaption><b>Recency is the wrong retention rule here.</b> A job handle from three turns ago is more load-bearing than the intermediate result from the last one, and a pruner that keeps the newest entries will drop exactly the wrong thing.</figcaption>
</figure>

## Frequently Asked Questions

<details class="faq-item"><summary><span>Where should the inline threshold sit?</span></summary><p>Below what a user will wait for without feedback, which is around two seconds for a conversational interface. The number matters less than the fact that it is chosen deliberately and applied consistently: a system where some operations block for eight seconds and others queue at three is one whose behaviour users cannot predict, and unpredictability is more frustrating than slowness.</p></details>

<details class="faq-item"><summary><span>Should the agent poll for job completion?</span></summary><p>Not within a turn. Polling ties up the turn's budget waiting for something that will not finish, which is precisely why the work was queued. Check completed jobs at the start of each turn instead, from stored state, and mention anything that finished. That costs nothing and produces the natural conversational behaviour of "while we were talking, that finished".</p></details>

<details class="faq-item"><summary><span>How should partial progress be reported?</span></summary><p>By stage rather than by percentage, where the stages are meaningful. "Clipping the raster" and "computing zonal statistics" tell a user something; "43%" tells them the implementation has a loop. Percentages also invite the model to extrapolate a completion time, which will be wrong in a way the user remembers.</p></details>

<details class="faq-item"><summary><span>Can the same operation be inline for one user and queued for another?</span></summary><p>Yes, and it should be — that is what the estimate is for. A buffer over twelve features and one over four hundred thousand are the same operation and completely different jobs. What must not vary is the rule: two users with the same input should get the same mode, or the behaviour becomes impossible to explain or to test.</p></details>

<details class="faq-item"><summary><span>What happens to a job when the conversation ends?</span></summary><p>It completes and its result expires. Cancelling on disconnect sounds tidier and is usually wrong, because the user may return, and because cancellation mid-write can leave partial state. Let it finish, store the result against the conversation with a time-to-live, and count expirations — a high rate means the delivery path is broken rather than that users are abandoning work.</p></details>

## Related

- Up to the section overview: [Geospatial Prompt Engineering and Tool Routing](/geospatial-prompt-engineering-tool-routing/)
- Technique: [Handling Async Spatial Processing in Python Workflows](/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/handling-async-spatial-processing-in-python-workflows/)
- Technique: [Backpressure and Rate Limiting for Spatial API Calls](/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/backpressure-and-rate-limiting-for-spatial-api-calls/)
- Technique: [Choosing Between Queued and Inline Geoprocessing](/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/choosing-between-queued-and-inline-geoprocessing/)
- Peer topic: [Cost and Latency Budgets for Spatial Agents](/geospatial-prompt-engineering-tool-routing/cost-and-latency-budgets-for-spatial-agents/)
- Peer topic: [Multi-Step Spatial Agent Orchestration](/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/)
