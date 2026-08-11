---
title: Recovering a Failed Step Without Replaying the Chain
description: Resume a broken geoprocessing chain at the step that failed — fingerprinted checkpoints, a stale-tail rule, and recovery that depends on the failure class rather than a retry count.
slug: recovering-a-failed-step-without-replaying-the-chain
type: howto
breadcrumb: Recovering a Failed Step
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Recovering a Failed Step Without Replaying the Chain

A chain of seven operations that fails at step six should cost one step to recover, not six. That it usually costs six is a consequence of two missing things: state that records what each step produced, and a rule for deciding whether that state is still true. This guide covers both, and it is the recovery half of [multi-step spatial agent orchestration](/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/).

## When to Use This Approach

Recovery machinery earns its place when steps are expensive and chains are deep. Below that, replaying is simpler and costs less than the state it would avoid.

| Chain | Step cost | Approach |
|-------|-----------|----------|
| Two or three cheap steps | Under a second | Replay — checkpointing costs more |
| Deep chain, one expensive step | Mixed | Checkpoint the expensive step only |
| Deep chain, several expensive steps | Seconds to minutes | Checkpoint each expensive output |
| Any chain over shared reference data | Any | Checkpoint, and share by fingerprint |

<figure class="diagram">
<svg viewBox="16 38 748 212" role="img" aria-labelledby="rfs-two-t rfs-two-d" xmlns="http://www.w3.org/2000/svg"><title id="rfs-two-t">Resuming against replaying after a failure at step six</title><desc id="rfs-two-d">Resuming from the last checkpoint re-runs only the failed step, while replaying repeats every earlier step that had already succeeded.</desc><rect x="16" y="38" width="748" height="212" fill="#ffffff"/><rect x="30" y="52" width="340" height="160" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="410" y="52" width="340" height="160" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="200" y="84">resume at step six</text><text x="580" y="84">replay from step one</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="200" y="114">one step re-runs</text><text x="200" y="140">cost matches the fault</text><text x="200" y="166">earlier state is inspectable</text><text x="580" y="114">five successful steps repeat</text><text x="580" y="140">cost matches the depth</text><text x="580" y="166">nothing intermediate survives</text></g><text x="390" y="234" fill="#1f2937" font-size="13" text-anchor="middle">Failures cluster late in a chain, which is exactly where replay is most expensive</text></svg>
<figcaption><b>The asymmetry gets worse with depth.</b> Early steps rarely fail because their inputs are the source data; late steps fail on everything the earlier ones produced, so the expensive recovery is also the likely one.</figcaption>
</figure>

## Implementation

A checkpoint stores a reference to the output rather than the output itself, alongside enough context to decide later whether it is still valid.

```python
@dataclass(frozen=True)
class Checkpoint:
    step_name: str
    output_ref: str            # a table name or object key, not the data
    fingerprint: str           # over inputs, parameters and data versions
    created_at: str


def fingerprint(step, upstream: list[Checkpoint], versions: dict) -> str:
    payload = {
        "op": step.op,
        "params": dict(sorted(step.params.items())),
        "inputs": [c.fingerprint for c in upstream],
        "versions": {k: versions[k] for k in sorted(step.source_layers)},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
```

Two properties of that fingerprint do the real work. Sorting the parameters means a plan that lists them in a different order still matches. Including the upstream fingerprints means a change anywhere earlier in the chain propagates forward automatically — which is what makes the stale-tail rule fall out for free rather than needing to be implemented.

```python
def resume_point(plan, stored: dict[str, Checkpoint], versions) -> int:
    upstream: list[Checkpoint] = []
    for i, step in enumerate(plan.steps):
        want = fingerprint(step, upstream, versions)
        have = stored.get(step.name)
        if have is None or have.fingerprint != want:
            return i               # everything from here on is stale
        upstream.append(have)
    return len(plan.steps)         # nothing to do
```

Recovery then depends on why the step failed, not on how many times it has been tried. A transient failure resumes at the same step; an input failure goes back to the planner, which may change a parameter and thereby invalidate the tail; a capability failure reroutes; a fatal one stops with the earlier checkpoints intact.

<figure class="diagram">
<svg viewBox="13 32 753 224" role="img" aria-labelledby="rfs-class-t rfs-class-d" xmlns="http://www.w3.org/2000/svg"><title id="rfs-class-t">What recovery looks like for each failure class</title><desc id="rfs-class-d">Transient failures re-run the same step, capability failures reroute it, input failures return to the planner and invalidate the tail, and fatal failures stop while preserving earlier checkpoints.</desc><rect x="13" y="32" width="753" height="224" fill="#ffffff"/><g><rect x="30" y="46" width="172" height="150" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="216" y="46" width="172" height="150" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="402" y="46" width="172" height="150" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="588" y="46" width="162" height="150" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/></g><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="116" y="80">transient</text><text x="302" y="80">capability</text><text x="488" y="80">input</text><text x="669" y="80">fatal</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="116" y="112">re-run this step</text><text x="116" y="138">checkpoints all valid</text><text x="116" y="170">nothing else changes</text><text x="302" y="112">reroute this step</text><text x="302" y="138">checkpoints all valid</text><text x="302" y="170">a different backend</text><text x="488" y="112">back to the planner</text><text x="488" y="138">a parameter changes</text><text x="488" y="170">the tail invalidates</text><text x="669" y="112">stop here</text><text x="669" y="138">keep the checkpoints</text><text x="669" y="170">a person decides next</text></g><text x="390" y="238" fill="#1f2937" font-size="13" text-anchor="middle">Only the input class invalidates anything — the other three resume against unchanged state</text></svg>
<figcaption><b>Three of four failures leave the checkpoints intact.</b> That is what makes recovery cheap: most failures do not say anything about the work already done, so discarding it is a decision nobody actually made.</figcaption>
</figure>

## Validation & Testing

The test that matters most asserts that a resumed chain produces the same result as an uninterrupted one. Everything else in this design is an optimisation of that property.

```python
def test_resume_matches_clean_run(chain, fixture):
    expected = run(chain, fixture)

    with fail_at_step(chain, index=4, kind="transient"):
        with pytest.raises(StepFailed):
            run(chain, fixture)
    recovered = run(chain, fixture)          # picks up from checkpoints

    assert recovered == expected
```

The second test asserts the stale-tail rule: change a parameter on an early step and confirm that every later checkpoint is discarded, not just the one directly affected. This is the property most likely to be broken by a well-meaning change to the fingerprint.

## Gotchas & Edge Cases

**Fingerprints that omit data versions.** A checkpoint keyed only on operation and parameters will be reused after the underlying layer has been updated, producing an answer that is fast, confident and out of date. The version of every source layer belongs in the fingerprint.

**Resuming after a partial write.** A step that failed while writing leaves a table that exists and is incomplete. Writing to a staging name and renaming on success makes a failure leave nothing, which is the only version of this that is safe to resume against.

**Checkpoints that outlive their usefulness.** Storage grows quietly because each checkpoint is small and nothing removes them. A time-to-live matched to how long a conversation can plausibly resume is sufficient, and expiry is safe by construction since the worst case is recomputation.

**Sharing across users without scoping.** A fingerprint over inputs and parameters alone will happily serve one caller's result to another. Where the data is not readable by everyone, whatever scopes it has to appear in the fingerprint.

**Recovery that hides a systematic failure.** A step that fails on most runs and recovers each time is a defect being absorbed by the machinery. Counting recoveries per step, rather than only counting final outcomes, is what keeps it visible.

<figure class="diagram">
<svg viewBox="12 58 755 188" role="img" aria-labelledby="rfs-stale-t rfs-stale-d" xmlns="http://www.w3.org/2000/svg"><title id="rfs-stale-t">Why a change invalidates the whole tail</title><desc id="rfs-stale-d">A parameter change at step two changes its fingerprint, which is an input to step three's fingerprint, which is an input to step four's — so every later checkpoint is discarded automatically.</desc><rect x="12" y="58" width="755" height="188" fill="#ffffff"/><defs><marker id="rfs-stale-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/></marker></defs><rect x="26" y="72" width="160" height="106" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="212" y="72" width="160" height="106" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="398" y="72" width="160" height="106" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="584" y="72" width="160" height="106" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="106" y="104">step one</text><text x="292" y="104">step two</text><text x="478" y="104">step three</text><text x="664" y="104">step four</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="106" y="134">unchanged</text><text x="106" y="158">reuse it</text><text x="292" y="134">a parameter changed</text><text x="292" y="158">recompute</text><text x="478" y="134">its input changed</text><text x="478" y="158">recompute</text><text x="664" y="134">its input changed</text><text x="664" y="158">recompute</text></g><g stroke="#5b6471" stroke-width="2" marker-end="url(#rfs-stale-a)"><line x1="188" y1="125" x2="208" y2="125"/><line x1="374" y1="125" x2="394" y2="125"/><line x1="560" y1="125" x2="580" y2="125"/></g><text x="390" y="228" fill="#1f2937" font-size="13" text-anchor="middle">Nothing implements this rule — it falls out of including upstream fingerprints in each one</text></svg>
<figcaption><b>The propagation is structural, not procedural.</b> Because each fingerprint contains the ones before it, invalidation travels forward automatically — which is why this design has no separate invalidation logic to get wrong.</figcaption>
</figure>

## Telling the Reader What Happened

A recovered chain should read as one answer, not as an incident report. The reader asked a question; that the fourth step was retried against a different backend is not part of the answer, and mentioning it converts a successful recovery into a reason for doubt.

There are two exceptions worth honouring. The first is when recovery changed something the reader would care about — a fallback that produced a coarser result, a step that used a cached input rather than a fresh one. That belongs in the answer as a clause, because it changes what the number means. The second is duration: if recovery took long enough that the reader noticed, saying that a step was retried is more reassuring than silence about a delay they experienced.

Everything else belongs in the trace, keyed by the same identifier the answer carries. That is what turns a support conversation from a reconstruction into a lookup, and it costs one field in the response. The distinction to hold onto is between what changed the answer and what merely changed how the answer was obtained — the first is the reader's business and the second is yours.

## Operating This Step Over Time

Track the ratio of resumed runs to clean ones, and the depth at which resumption typically happens. A chain that resumes constantly at the same step has a defect there; one that never resumes is paying for checkpoints it does not use, and the checkpointing can be narrowed to the expensive steps.

Watch checkpoint reuse across conversations as well. A high cross-conversation hit rate on shared reference data is the strongest argument for keeping the machinery; a rate near zero suggests the fingerprint includes something conversation-specific that does not belong in it.

The fingerprint deserves a review whenever the plan format changes. Adding a field to a step that is not included in the fingerprint creates checkpoints that match when they should not, which is the one failure in this design that produces wrong answers rather than merely slow ones. The safe habit is to derive the fingerprint from the whole step object with an explicit exclusion list, so a new field is included by default and omitting it is a deliberate act somebody has to write down.

Storage layout is worth revisiting too. Checkpoints written beside the source data resume cleanly from any worker; ones written to local disk resume only from the machine that created them, which is fine on a single host and becomes an intermittent failure the moment a second one appears. That transition tends to happen without anyone reconsidering where intermediate results live.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should the reader be asked before a chain resumes?</span></summary><p>Only when recovery will change the answer rather than just the route. A retry against the same backend needs no confirmation; falling back to a coarser method, or resuming against a cached input that is a day old, does — because those change what the number means. Tying the confirmation to whether the result semantics change, rather than to whether recovery occurred, keeps the interruptions meaningful.</p></details>

<details class="faq-item"><summary><span>What if the chain is long enough that the conversation has moved on?</span></summary><p>Complete it and store the result against the conversation rather than cancelling. The reader may return, cancellation mid-write is the case that leaves debris, and a finished result mentioned on the next turn is natural behaviour. Where they do not return, the expiry handles it and the count of expirations tells you whether the delivery path is working.</p></details>

<details class="faq-item"><summary><span>Can checkpoints be used to answer a follow-up question?</span></summary><p>Frequently, and it is one of the better arguments for keeping them. A follow-up that changes only the final aggregation reuses everything before it, which the fingerprint handles without any special case. This is also why checkpoints should be keyed on the computation rather than on the conversation turn that produced them.</p></details>

<details class="faq-item"><summary><span>How does this differ from an ordinary result cache?</span></summary><p>Mostly in lifetime and intent. A cache exists to make repeated work cheap and can be discarded at will; checkpoints exist to make recovery cheap and are meaningful only in the context of one plan. They can share a backing store and a fingerprint scheme, and confusing their expiry policies produces either a cache that never hits or checkpoints that vanish mid-chain.</p></details>

## Related

- Up to the parent topic: [Multi-Step Spatial Agent Orchestration](/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/)
- [Chaining Geoprocessing Tools with State Checkpoints](/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/chaining-geoprocessing-tools-with-state-checkpoints/)
- Related topic: [Error Mapping for Spatial API Calls](/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/)
- Related topic: [Cost and Latency Budgets for Spatial Agents](/geospatial-prompt-engineering-tool-routing/cost-and-latency-budgets-for-spatial-agents/)
