---
title: Multi-Step Spatial Agent Orchestration
description: Run spatial work that spans turns — checkpointed state, idempotent steps, recovery that resumes rather than replays, and a record of what each step actually did.
slug: multi-step-spatial-agent-orchestration
type: topic
breadcrumb: Agent Orchestration
datePublished: 2025-04-08
dateModified: 2026-08-11
---

# Multi-Step Spatial Agent Orchestration

A spatial analysis that takes four minutes and six operations cannot live inside a turn. It needs state that survives, steps that can be re-run without doubling their effect, and a recovery path that resumes from the last good checkpoint rather than starting again — because starting again is how a four-minute analysis becomes a twelve-minute one.

This topic belongs to [geospatial prompt engineering and tool routing](/geospatial-prompt-engineering-tool-routing/) and takes over where [LLM-assisted geoprocessing pipelines](/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/) stops: that topic plans and validates a chain within a turn, this one runs chains that outlive it.

<figure class="diagram">
<svg viewBox="16 24 772 216" role="img" aria-labelledby="mso-resume-t mso-resume-d" xmlns="http://www.w3.org/2000/svg"><title id="mso-resume-t">Replaying against resuming after a failure at step four</title><desc id="mso-resume-d">A chain that fails at its fourth step can either re-run everything from the start or resume from the last checkpoint, and the difference is the cost of the three steps that already succeeded.</desc><rect x="16" y="24" width="772" height="216" fill="#ffffff"/><text x="30" y="62" fill="#b3324f" font-size="13" font-weight="600">replay</text><g fill="#fdeaee" stroke="#b3324f" stroke-width="2"><rect x="150" y="38" width="120" height="42" rx="5"/><rect x="276" y="38" width="120" height="42" rx="5"/><rect x="402" y="38" width="120" height="42" rx="5"/><rect x="528" y="38" width="120" height="42" rx="5"/><rect x="654" y="38" width="120" height="42" rx="5"/></g><text x="462" y="64" fill="#1f2937" font-size="12" text-anchor="middle">all five steps again</text><text x="30" y="152" fill="#12805c" font-size="13" font-weight="600">resume</text><g fill="#eef2f7" stroke="#5b6471" stroke-width="2"><rect x="150" y="128" width="120" height="42" rx="5"/><rect x="276" y="128" width="120" height="42" rx="5"/><rect x="402" y="128" width="120" height="42" rx="5"/></g><g fill="#e4f5ec" stroke="#12805c" stroke-width="2"><rect x="528" y="128" width="120" height="42" rx="5"/><rect x="654" y="128" width="120" height="42" rx="5"/></g><text x="336" y="154" fill="#1f2937" font-size="12" text-anchor="middle">from checkpoint</text><text x="700" y="154" fill="#1f2937" font-size="12" text-anchor="middle">re-run</text><text x="400" y="222" fill="#1f2937" font-size="13" text-anchor="middle">Resuming needs step outputs to be addressable and steps to be safe to re-run</text></svg>
<figcaption><b>Resuming is not free — it is bought with checkpoints.</b> A chain whose intermediates are discarded has no choice but to replay, which is why the storage decision is made when the chain is designed rather than after it first fails.</figcaption>
</figure>

## Foundational Principles

**Every step is idempotent.** Running a step twice must produce the same state as running it once. Without that, resuming after an ambiguous failure — did the write land? — is unsafe, and the only safe recovery is a full replay.

**State is external and addressable.** Step outputs live somewhere with a name, not in a process's memory. A chain whose intermediates exist only in the worker cannot survive the worker.

**A failed step names itself.** Recovery starts from knowing which step failed and why, and that requires the run record to be written as the chain progresses rather than at its end.

## Step-by-Step Implementation Pipeline

### 1. Give the run an identity and a record

The run record is the durable object. It carries the plan, the status of each step, and the references to what each step produced.

```python
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

log = logging.getLogger("spatial_orchestration")

Status = Literal["pending", "running", "done", "failed", "skipped"]


@dataclass
class StepRecord:
    name: str
    status: Status = "pending"
    output_ref: Optional[str] = None
    rows: Optional[int] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


@dataclass
class Run:
    run_id: str
    plan_digest: str
    steps: list[StepRecord] = field(default_factory=list)
    created_at: float = 0.0

    def first_incomplete(self) -> Optional[StepRecord]:
        return next((s for s in self.steps if s.status != "done"), None)


def plan_digest(plan) -> str:
    """A stable digest so a resumed run can prove it is the same plan."""
    payload = json.dumps([{ "name": s.name, "op": s.op, "inputs": list(s.inputs),
                            "params": s.params} for s in plan.steps],
                         sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
```

The digest is what makes resuming safe. A run resumed against an edited plan would restore outputs produced by different steps, and comparing digests turns that into a rejection rather than a subtly wrong result.

### 2. Make step outputs addressable and content-keyed

An output reference derived from the step's inputs and parameters gives idempotency almost for free: re-running a step with the same inputs writes to the same place, so a duplicate run overwrites rather than duplicating.

```python
def output_ref(run_id: str, step, input_refs: Sequence[str]) -> str:
    """Deterministic reference: same step, same inputs, same place."""
    key = json.dumps({"op": step.op, "params": step.params, "inputs": sorted(input_refs)},
                     sort_keys=True, separators=(",", ":"), default=str)
    return f"{run_id}/{step.name}/{hashlib.sha256(key.encode()).hexdigest()[:12]}"
```

### 3. Write outputs atomically

A step interrupted mid-write leaves partial output that a resumed run will treat as complete. Writing under a temporary name and promoting on success removes that entire class of problem.

```python
def write_atomic(store, ref: str, payload) -> None:
    """Write to a temporary reference, then promote. Never leave a partial result."""
    staging = f"{ref}.staging"
    store.put(staging, payload)
    try:
        store.promote(staging, ref)                  # atomic rename or equivalent
    except Exception:
        store.delete(staging)
        raise
```

### 4. Run the chain from the first incomplete step

Resuming is then a matter of finding the first step that is not done and continuing, with the earlier outputs loaded by reference rather than recomputed.

```python
def execute_run(run: Run, plan, store, execute, validate) -> Run:
    """Resume from the first incomplete step. Idempotent per step."""
    if run.plan_digest != plan_digest(plan):
        raise ValueError("the plan has changed since this run started; start a new run")

    outputs = {s.name: s.output_ref for s in run.steps if s.status == "done"}
    for record, step in zip(run.steps, plan.steps):
        if record.status == "done":
            continue
        record.status, record.started_at = "running", time.monotonic()
        try:
            inputs = [store.get(outputs[r]) if r in outputs else store.get_source(r)
                      for r in step.inputs]
            result = execute(step, inputs)
            check = validate(result, step.name)
            if not check.ok:
                record.status, record.error = "failed", check.reason
                return run
            ref = output_ref(run.run_id, step, [outputs.get(r, r) for r in step.inputs])
            write_atomic(store, ref, result)
            record.output_ref, record.rows = ref, len(result)
            record.status, record.finished_at = "done", time.monotonic()
            outputs[step.name] = ref
        except Exception as exc:                     # a step failure is recorded, not raised
            log.warning("step %s failed: %s", step.name, exc)
            record.status, record.error = "failed", str(exc)
            return run
    return run
```

Returning the run rather than raising is what makes the failure inspectable. The caller has a record naming the failed step, its error and every output that succeeded, which is exactly what a recovery decision needs. The recovery patterns themselves are covered in [recovering a failed step without replaying the chain](/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/recovering-a-failed-step-without-replaying-the-chain/).

<figure class="diagram">
<svg viewBox="16 24 764 160" role="img" aria-labelledby="mso-atomic-t mso-atomic-d" xmlns="http://www.w3.org/2000/svg"><title id="mso-atomic-t">Partial output against an atomic promote</title><desc id="mso-atomic-d">A step interrupted mid-write leaves a partial file that a resumed run treats as a completed output; writing to staging and promoting on success means an interrupted step leaves nothing.</desc><rect x="16" y="24" width="764" height="160" fill="#ffffff"/><text x="30" y="62" fill="#b3324f" font-size="13" font-weight="600">direct write</text><rect x="190" y="38" width="230" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="426" y="38" width="120" height="42" rx="5" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><text x="305" y="64" fill="#1f2937" font-size="12" text-anchor="middle">interrupted at 60%</text><text x="600" y="64" fill="#5b6471" font-size="12">the resume reads it as complete</text><text x="30" y="152" fill="#12805c" font-size="13" font-weight="600">staged</text><rect x="190" y="128" width="230" height="42" rx="5" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="426" y="128" width="150" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="305" y="154" fill="#1f2937" font-size="12" text-anchor="middle">staging file</text><text x="501" y="154" fill="#1f2937" font-size="12" text-anchor="middle">promoted</text><text x="640" y="154" fill="#5b6471" font-size="12">or nothing at all</text></svg>
<figcaption><b>The partial file is the bug.</b> It exists, it parses, it has a plausible row count, and it is a subset of unknown shape — which the resumed chain then treats as the step's real output.</figcaption>
</figure>

### 5. Distinguish a failed step from a failed run

A step that failed transiently can be retried in place; one that failed on its input needs the plan changed; one that failed on authorisation ends the run. Recording the class alongside the error is what lets recovery be automatic where it can be.

```python
def recovery_for(record: StepRecord, error_class: str, attempts: int) -> str:
    """What to do about a failed step."""
    if error_class == "transient" and attempts < 3:
        return "retry_step"
    if error_class == "input":
        return "amend_plan"
    if error_class == "capability":
        return "reroute_step"
    return "end_run"
```

### 6. Checkpoint the run record, not just the outputs

Outputs without a record are unreachable, and a record written only at the end is lost precisely when it is needed. Persist the record after every step transition.

```python
def persist(run: Run, store) -> None:
    """Write the run record after every transition; it is the map to everything else."""
    try:
        write_atomic(store, f"{run.run_id}/run.json", run)
    except Exception as exc:                        # a record we cannot write is a run we cannot resume
        log.error("could not persist run %s: %s", run.run_id, exc)
        raise
```

Raising here, unlike everywhere else in this topic, is correct. A run whose record cannot be written is one that will produce outputs nobody can find, and continuing to compute them is worse than stopping.

### 7. Expire runs deliberately

Intermediate outputs are large and most are never read again. Expiry policy belongs with the run, keyed on how long a conversation plausibly lasts, and the expiry itself should be recorded so a later resume attempt gets an explanation rather than a missing file.

```python
def expire(run: Run, store, max_age_s: float, now: float) -> bool:
    """Remove intermediates but keep the record, so a resume attempt is explicable."""
    if now - run.created_at < max_age_s:
        return False
    for record in run.steps:
        if record.output_ref:
            store.delete(record.output_ref)
            record.output_ref, record.status = None, "skipped"
    persist(run, store)
    return True
```

### 8. Report progress in terms a user recognises

A run's status is a sequence of named steps, and the useful report names the current step rather than a percentage. The state-checkpointing mechanics that make this reliable are covered in [chaining geoprocessing tools with state checkpoints](/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/chaining-geoprocessing-tools-with-state-checkpoints/).

```python
def progress_sentence(run: Run) -> str:
    done = sum(1 for s in run.steps if s.status == "done")
    current = run.first_incomplete()
    if current is None:
        return f"All {len(run.steps)} steps finished."
    if current.status == "failed":
        return f"Stopped at '{current.name}': {current.error}"
    return f"Step {done + 1} of {len(run.steps)}: {current.name}."
```

### 9. Decide what a step may do besides compute

Most orchestrated steps are pure — they read inputs and write an output — and a few are not. A step that notifies an external system, writes to a shared table or triggers a downstream process has an effect that re-running will repeat, and idempotency by output reference does nothing for it.

```python
EFFECTFUL_OPS = {"publish", "notify", "write_back"}


def guard_effects(step, record: StepRecord) -> None:
    """An effectful step may only run once; its completion is the lock."""
    if step.op not in EFFECTFUL_OPS:
        return
    if record.status == "done":
        raise AlreadyPerformed(
            f"step {step.name!r} has already run and has an external effect; "
            "resuming must not repeat it")
    if record.status == "failed" and record.error and "after_effect" in record.error:
        raise AmbiguousEffect(
            f"step {step.name!r} failed after its external effect may have occurred; "
            "resolve manually before resuming")
```

The second branch is the uncomfortable one and the reason effectful steps are worth avoiding. A notification that failed *after* sending cannot be distinguished from one that failed before, and no amount of orchestration resolves that — only a human, or an idempotency key carried into the external system, can. Where possible, push effects to the end of the chain so that at most one step is in this category.

### 10. Keep the run record small enough to read

A run record that grows with the data becomes unreadable and expensive to persist after every transition. It should carry references and counts, never payloads, and its size should be bounded regardless of how much data the run processed.

```python
MAX_ERROR_CHARS = 400


def compact(record: StepRecord) -> StepRecord:
    """Bound the record's size so persisting it stays cheap."""
    if record.error and len(record.error) > MAX_ERROR_CHARS:
        record.error = record.error[:MAX_ERROR_CHARS] + "… (truncated; see logs)"
    return record


def run_summary(run: Run) -> dict:
    """What a caller needs, without the internals."""
    return {
        "run_id": run.run_id,
        "steps": [{"name": s.name, "status": s.status, "rows": s.rows} for s in run.steps],
        "failed_at": next((s.name for s in run.steps if s.status == "failed"), None),
    }
```

Truncating the error rather than dropping it keeps the record useful while bounding its size. Errors from spatial libraries can run to thousands of characters — a geometry engine will happily include the offending coordinates — and a record persisted after every transition cannot afford to carry them.

The row counts, by contrast, are worth every byte. They are the diagnostic that explains a surprising result faster than anything else in the system, and they are the reason a completed run is worth keeping after its intermediates have expired.

## Operating This Stage Over Time

Orchestrated runs accumulate storage faster than anyone expects, because every step of every run writes an intermediate and most are read once. Expiry needs to be a scheduled job rather than a good intention, and the metric to watch is total intermediate storage rather than run count — a small number of large runs dominates.

Plans change while runs are in flight. A deploy that alters a step's parameters means every resumable run now has a stale digest and will refuse to resume, which is correct and surprising. Draining in-flight runs before a plan change, or accepting that they restart, is a decision worth making explicitly rather than discovering.

The recovery classification drifts with the dependencies. An error that was transient becomes permanent when a service is retired, and a retry loop that used to succeed now consumes three attempts before giving up. Reviewing recovery outcomes — how often each class actually recovered — is the check that keeps the classification honest.

Finally, watch the resume rate. A system where most runs complete on the first attempt has orchestration that is working; one where most runs resume at least once has a reliability problem that the orchestration is successfully hiding, and hiding it indefinitely is not the goal.

## Failure Modes & Root Causes

**The replayed chain.** A failure at step five re-runs steps one to four. Root cause: intermediates not persisted or not addressable. Mitigation: content-keyed output references and a durable run record.

**The partial output.** A resumed run consumes a file written half-way. Root cause: direct writes. Mitigation: staging plus atomic promotion.

**The doubled effect.** A step re-run appends rather than replaces, so a resumed run doubles its output. Root cause: non-idempotent steps. Mitigation: deterministic output references and overwrite semantics.

**The unresumable run.** Outputs exist but nothing knows where. Root cause: a run record written only at completion. Mitigation: persist after every transition; treat a failed persist as fatal.

## Production Validation Protocols

1. **Idempotency test.** Run a step twice and assert the resulting state is identical, including row counts and references.
2. **Resume-after-kill test.** Kill a worker mid-chain and assert the run resumes from the last completed step rather than the beginning.
3. **Partial-write test.** Interrupt a write and assert the resumed run does not see a partial output.
4. **Digest-mismatch test.** Alter the plan and assert a resume attempt is refused with a clear reason.
5. **Storage indicator.** Publish total intermediate storage and the age distribution of runs; expiry problems appear here first.
6. **Resume-rate indicator.** Track the share of runs that resumed at least once; a rise means a dependency is degrading.

<figure class="diagram">
<svg viewBox="16 38 728 192" role="img" aria-labelledby="mso-class-t mso-class-d" xmlns="http://www.w3.org/2000/svg"><title id="mso-class-t">Recovery action by failure class</title><desc id="mso-class-d">A transient failure retries the same step, an input failure needs the plan amended, a capability failure reroutes the step elsewhere, and anything else ends the run.</desc><rect x="16" y="38" width="728" height="192" fill="#ffffff"/><rect x="30" y="52" width="170" height="120" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="216" y="52" width="170" height="120" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="402" y="52" width="170" height="120" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="588" y="52" width="142" height="120" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="115" y="84">transient</text><text x="301" y="84">input</text><text x="487" y="84">capability</text><text x="659" y="84">fatal</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="115" y="114">retry this step</text><text x="115" y="138">up to three times</text><text x="301" y="114">amend the plan</text><text x="301" y="138">the step is wrong</text><text x="487" y="114">reroute the step</text><text x="487" y="138">another backend</text><text x="659" y="114">end the run</text><text x="659" y="138">and say why</text></g><text x="380" y="212" fill="#1f2937" font-size="13" text-anchor="middle">Three of the four recover without discarding the work already done</text></svg>
<figcaption><b>Only the last class loses the run.</b> The value of classification here is not tidiness — it is that three of four failures can be handled without repeating steps that already succeeded.</figcaption>
</figure>

The resume-after-kill test is the one that has to exist before the orchestration is trusted. Every other property here can be reasoned about; whether a run actually survives losing its worker is an empirical question, and the answer is frequently no for reasons that only appear under a real interruption — a connection held open, a lock not released, a record written after the output rather than before.

Run it as part of continuous integration rather than as an occasional exercise. Killing a worker mid-chain is easy to automate, the assertion is a single comparison of step statuses, and the failure it catches is the one that turns a four-minute analysis into an unrecoverable one at the worst possible moment.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Where should intermediate outputs be stored?</span></summary><p>Wherever they can be addressed by reference and written atomically — object storage and a database both work, and a worker's local disk does not, because it disappears with the worker. The deciding property is not speed but durability across the failure you are trying to survive: if a run must survive a worker restart, its state cannot live inside one.</p></details>

<details class="faq-item"><summary><span>Should a resumed run re-validate the steps that already completed?</span></summary><p>Not their contents, but their existence. Re-reading and re-checking a large intermediate costs nearly as much as recomputing it, which defeats the purpose. Verifying that the reference resolves and the row count matches the record is cheap and catches the cases that matter — an expired output, a deleted file, a truncated write that got promoted anyway.</p></details>

<details class="faq-item"><summary><span>How does this interact with the async job queue?</span></summary><p>An orchestrated run is usually a queued job, and the two layers compose: the queue decides when the run gets a worker, the orchestration decides where it resumes. Keeping them separate matters because a queue retry and an orchestration resume are different operations — the first re-dispatches the run, the second continues it, and conflating them produces a run that restarts every time it is re-dispatched.</p></details>

<details class="faq-item"><summary><span>What should the agent tell the user during a long run?</span></summary><p>The current step by name, and nothing about time remaining. Named steps are meaningful — "computing the overlay" tells a user something about scale — while an estimated completion time will be wrong and remembered. If a run has been going unusually long, say that rather than predicting; the honest statement is more useful than a confident number.</p></details>

<details class="faq-item"><summary><span>Is orchestration worth it for a three-step chain?</span></summary><p>Not usually. Three steps that complete inside a turn need validation between them, which the pipeline topic covers, and adding durable state and resume logic to something that finishes in two seconds is machinery with no beneficiary. The threshold is roughly whether a failure would cost enough recomputation to be worth avoiding, which in practice means chains measured in minutes rather than seconds.</p></details>

## Related

- Up to the section overview: [Geospatial Prompt Engineering and Tool Routing](/geospatial-prompt-engineering-tool-routing/)
- Technique: [Chaining Geoprocessing Tools with State Checkpoints](/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/chaining-geoprocessing-tools-with-state-checkpoints/)
- Technique: [Recovering a Failed Step Without Replaying the Chain](/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/recovering-a-failed-step-without-replaying-the-chain/)
- Peer topic: [LLM-Assisted Geoprocessing Pipelines](/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/)
- Peer topic: [Async vs Sync Geoprocessing Workflows](/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/)
- Related topic: [Cost and Latency Budgets for Spatial Agents](/geospatial-prompt-engineering-tool-routing/cost-and-latency-budgets-for-spatial-agents/)
