---
title: LLM-Assisted Geoprocessing Pipelines
description: Turn a natural-language analysis request into a validated chain of geoprocessing steps, with every step's output checked before the next one consumes it.
slug: llm-assisted-geoprocessing-pipelines
type: topic
breadcrumb: Geoprocessing Pipelines
datePublished: 2025-03-25
dateModified: 2026-08-11
---

# LLM-Assisted Geoprocessing Pipelines

"Which residential parcels are within 200 metres of a watercourse and outside the flood zone" is one sentence and four geoprocessing operations, each of which can succeed while producing something the next step cannot use. Planning that chain, and validating between its steps, is what separates an analysis you can defend from a sequence of plausible operations ending in a number.

This topic belongs to [geospatial prompt engineering and tool routing](/geospatial-prompt-engineering-tool-routing/) and sits above [GeoPandas and PostGIS tool routing](/geospatial-prompt-engineering-tool-routing/geopandas-postgis-tool-routing/), which decides where each planned step runs. Chains long enough to need checkpointing belong to [multi-step spatial agent orchestration](/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/).

<figure class="diagram">
<svg viewBox="6 42 786 180" role="img" aria-labelledby="lgp-chain-t lgp-chain-d" xmlns="http://www.w3.org/2000/svg"><title id="lgp-chain-t">One sentence decomposed into four validated steps</title><desc id="lgp-chain-d">Selecting residential parcels, buffering watercourses, intersecting the two and subtracting the flood zone, with a validation gate between each pair of steps.</desc><rect x="6" y="42" width="786" height="180" fill="#ffffff"/><g fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"><rect x="24" y="56" width="170" height="72" rx="8"/><rect x="214" y="56" width="170" height="72" rx="8"/><rect x="404" y="56" width="170" height="72" rx="8"/><rect x="594" y="56" width="180" height="72" rx="8"/></g><g fill="#1f2937" font-size="12.5" text-anchor="middle" font-weight="600"><text x="109" y="84">select</text><text x="299" y="84">buffer</text><text x="489" y="84">intersect</text><text x="684" y="84">difference</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="109" y="108">residential parcels</text><text x="299" y="108">watercourses, 200 m</text><text x="489" y="108">the two results</text><text x="684" y="108">minus the flood zone</text></g><g stroke="#5b6471" stroke-width="2" marker-end="url(#lgp-chain-a)"><line x1="196" y1="92" x2="210" y2="92"/><line x1="386" y1="92" x2="400" y2="92"/><line x1="576" y1="92" x2="590" y2="92"/></g><defs><marker id="lgp-chain-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/></marker></defs><rect x="24" y="156" width="750" height="52" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="399" y="188" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">a validation gate between every pair: non-empty, valid, in the expected frame, plausibly sized</text></svg>
<figcaption><b>The gates are where the analysis becomes defensible.</b> Each operation can succeed and produce an empty or degenerate result, and without a check the next step consumes it happily and the chain ends in a confident zero.</figcaption>
</figure>

## Foundational Principles

**Decompose before executing.** The whole chain is planned, reviewed and estimated before the first operation runs. A pipeline assembled step by step as the model reacts to each result cannot be costed, cannot be checkpointed, and cannot be explained afterwards.

**Every intermediate result is validated.** Non-empty, valid, in the expected frame, and of a plausible size. Those four checks catch nearly every silent failure, and they are cheap relative to the operations they guard.

**An empty intermediate is a stop, not a value.** A buffer that produced nothing, an intersection that found nothing — these mean the analysis has already failed, and continuing produces an answer about nothing that reads like an answer about something.

## Step-by-Step Implementation Pipeline

### 1. Decompose the request into a typed plan

The plan is a list of typed steps with named inputs and outputs, produced once and reviewable as data.

```python
import logging
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

log = logging.getLogger("geoprocessing_plan")

Op = Literal["select", "buffer", "intersect", "difference", "dissolve", "aggregate"]


@dataclass(frozen=True)
class Step:
    name: str
    op: Op
    inputs: tuple[str, ...]              # names of earlier steps or source layers
    params: dict


@dataclass(frozen=True)
class Plan:
    steps: tuple[Step, ...]
    result: str                          # the step whose output is the answer


class PlanRejected(ValueError):
    """The plan cannot be executed as written."""
```

### 2. Check the plan before running any of it

Most bad plans are detectable statically: a step referencing an output that does not exist, an operation applied to the wrong geometry type, a chain whose result is not produced by any step.

```python
GEOMETRY_REQUIREMENTS = {
    "buffer":     {"Point", "LineString", "Polygon", "MultiPolygon"},
    "intersect":  {"Polygon", "MultiPolygon"},
    "difference": {"Polygon", "MultiPolygon"},
    "dissolve":   {"Polygon", "MultiPolygon"},
}


def check_plan(plan: Plan, sources: dict[str, str]) -> Plan:
    """Static checks: references resolve, types match, the result exists."""
    produced: dict[str, str] = dict(sources)          # name -> geometry type
    for step in plan.steps:
        for ref in step.inputs:
            if ref not in produced:
                raise PlanRejected(f"step {step.name!r} references unknown input {ref!r}")
        allowed = GEOMETRY_REQUIREMENTS.get(step.op)
        if allowed:
            for ref in step.inputs:
                if produced[ref] not in allowed:
                    raise PlanRejected(
                        f"{step.op} cannot take {produced[ref]} from {ref!r}")
        produced[step.name] = _output_type(step.op, [produced[r] for r in step.inputs])
    if plan.result not in produced:
        raise PlanRejected(f"the plan's result {plan.result!r} is not produced by any step")
    return plan
```

Type-checking the chain statically catches the most common planning error — buffering a polygon and then intersecting with a line, or dissolving something that is not an area — before any expensive operation runs. The decomposition itself is developed in [decomposing natural language into geoprocessing steps](/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/decomposing-natural-language-into-geoprocessing-steps/).

### 3. Estimate the whole chain, not each step

A chain's cost is dominated by the step with the largest intermediate, which is frequently not the step that looks expensive. Estimating the whole plan lets the agent trade a cheaper decomposition before anything runs.

```python
def estimate_chain(plan: Plan, sizes: dict[str, int]) -> tuple[int, str]:
    """Track the largest intermediate; that is what the chain will cost."""
    current = dict(sizes)
    peak, peak_step = 0, ""
    for step in plan.steps:
        inputs = [current.get(r, 0) for r in step.inputs]
        if step.op == "buffer":
            out = inputs[0]
        elif step.op in {"intersect", "difference"}:
            out = int(max(inputs) * 1.2)              # overlays can add vertices and parts
        elif step.op == "dissolve":
            out = max(1, inputs[0] // 10)
        else:
            out = inputs[0] if inputs else 0
        current[step.name] = out
        if out > peak:
            peak, peak_step = out, step.name
    return peak, peak_step
```

### 4. Validate between every pair of steps

The four checks are the same each time, and running them uniformly is what makes the chain's failure modes predictable.

```python
@dataclass(frozen=True)
class Check:
    ok: bool
    reason: str


def validate_intermediate(result, expected_epsg: int, input_count: int,
                          step_name: str) -> Check:
    """Four checks that catch nearly every silent intermediate failure."""
    if result is None or len(result) == 0:
        return Check(False, f"{step_name} produced no features")
    invalid = sum(1 for g in result.geometry if g is not None and not g.is_valid)
    if invalid:
        return Check(False, f"{step_name} produced {invalid} invalid geometries")
    epsg = getattr(result.crs, "to_epsg", lambda: None)()
    if epsg != expected_epsg:
        return Check(False, f"{step_name} output is in EPSG:{epsg}, expected {expected_epsg}")
    if input_count and len(result) > input_count * 50:
        return Check(False,
                     f"{step_name} produced {len(result)} features from {input_count} "
                     "inputs — check for an unintended cross join")
    return Check(True, "")
```

The size sanity check is the one that catches the most expensive mistake. An overlay against the wrong layer produces a cross product, and the result is a valid, non-empty, correctly framed dataset of two million slivers that the next step will happily consume. The full set of checks is developed in [validating intermediate geoprocessing outputs](/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/validating-intermediate-geoprocessing-outputs/).

<figure class="diagram">
<svg viewBox="16 36 705 192" role="img" aria-labelledby="lgp-empty-t lgp-empty-d" xmlns="http://www.w3.org/2000/svg"><title id="lgp-empty-t">An empty intermediate propagating to a confident zero</title><desc id="lgp-empty-d">A buffer that produced nothing feeds an intersection that produces nothing, which feeds a count that returns zero — a well-formed answer to a question the chain never actually asked.</desc><rect x="16" y="36" width="705" height="192" fill="#ffffff"/><rect x="30" y="50" width="150" height="56" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="196" y="50" width="150" height="56" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="362" y="50" width="150" height="56" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="528" y="50" width="150" height="56" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="12" text-anchor="middle"><text x="105" y="82">840 parcels</text><text x="271" y="82">0 buffers</text><text x="437" y="82">0 features</text><text x="603" y="82">count: 0</text></g><text x="390" y="140" fill="#5b6471" font-size="12.5" text-anchor="middle">every step succeeded; the answer is a confident zero about nothing</text><rect x="196" y="164" width="150" height="50" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="271" y="194" fill="#1f2937" font-size="12" text-anchor="middle">gate stops here</text><text x="366" y="194" fill="#1f2937" font-size="12.5">reason: the watercourse layer was empty here</text></svg>
<figcaption><b>Zero is the most dangerous valid answer.</b> It is well-formed, it is what the arithmetic produced, and it is indistinguishable from a genuine finding unless something checked the step that first produced nothing.</figcaption>
</figure>

### 5. Stop on a failed check, with the step named

A failed intermediate is a stop, and the message needs the step name and the reason — that is what turns a dead chain into a correctable plan.

```python
def run_chain(plan: Plan, execute, expected_epsg: int) -> dict:
    """Execute with a gate after every step. Stops at the first failure."""
    results: dict[str, object] = {}
    for step in plan.steps:
        inputs = [results.get(r) for r in step.inputs]
        try:
            out = execute(step, inputs)
        except Exception as exc:
            return {"ok": False, "failed_at": step.name, "reason": f"execution failed: {exc}"}
        check = validate_intermediate(out, expected_epsg,
                                      sum(len(i) for i in inputs if i is not None),
                                      step.name)
        if not check.ok:
            log.info("chain stopped at %s: %s", step.name, check.reason)
            return {"ok": False, "failed_at": step.name, "reason": check.reason}
        results[step.name] = out
    return {"ok": True, "result": results[plan.result], "steps": list(results)}
```

### 6. Keep the plan and the result together

An analysis result without its plan cannot be defended or repeated. Returning both, with the parameters that produced each step, is what makes a number a finding rather than an assertion.

```python
def to_finding(plan: Plan, result, notes: Sequence[str]) -> dict:
    return {
        "value": summarise(result),
        "method": [{"step": s.name, "op": s.op, "params": s.params} for s in plan.steps],
        "notes": list(notes),
    }
```

### 7. Prefer a plan the database can run in one pass

A chain of four operations executed as four round trips through a process is slower and more fragile than the same chain expressed as one query where the database can do it. Detecting that case is worth the effort for common shapes.

```python
FUSIBLE = ({"select", "buffer", "intersect"}, {"select", "intersect", "difference"})


def can_fuse(plan: Plan) -> bool:
    """Can this chain be expressed as a single query?"""
    ops = {s.op for s in plan.steps}
    return any(ops <= fusible for fusible in FUSIBLE) and len(plan.steps) <= 4
```

Fusing is an optimisation and not a requirement, and it should never change the result. Where a fused query and a stepwise chain disagree, the stepwise chain is the reference — it is the one whose intermediates were checked. The specific pattern of buffer, overlay and dissolve chains is covered in [planning buffer, overlay and dissolve chains from a prompt](/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/planning-buffer-overlay-dissolve-chains-from-a-prompt/).

### 8. Record what each step actually produced

Row counts per step are the cheapest diagnostic in the whole topic. A chain that ended in zero is explained instantly by the step where the count first became zero, and that number costs nothing to record.

```python
def step_counts(results: dict[str, object]) -> dict[str, int]:
    return {name: (len(value) if value is not None else 0)
            for name, value in results.items()}
```

### 9. Resolve source layers against the catalog at plan time

A plan referencing a layer that does not exist should fail while it is still a plan, not three steps into execution. Resolving names against the live catalog during the static check is what makes that happen, and it also catches the more dangerous case of a name that resolves to something other than what was meant.

```python
def resolve_sources(plan: Plan, catalog) -> dict[str, str]:
    """Map every source name to a real layer and its geometry type, or reject."""
    sources: dict[str, str] = {}
    referenced = {ref for step in plan.steps for ref in step.inputs}
    step_names = {step.name for step in plan.steps}
    for ref in sorted(referenced - step_names):
        record = catalog.get(ref)
        if record is None:
            raise PlanRejected(
                f"source layer {ref!r} does not exist; available layers are "
                f"{', '.join(sorted(catalog)[:8])}")
        if record.get("deprecated"):
            log.info("plan uses deprecated layer %r (successor: %s)",
                     ref, record.get("successor"))
        sources[ref] = record["geometry_type"]
    return sources
```

Listing the available layers in the rejection is what makes it recoverable in one turn. A model told that a layer does not exist will invent another plausible name; told which names exist, it picks one.

### 10. Keep the plan reproducible across data versions

A finding produced against last month's data cannot be compared with one produced today unless both record which version they ran against. Stamping the source versions onto the plan makes a re-run either identical or explicitly different.

```python
def stamp_versions(plan: Plan, catalog) -> dict:
    """Record the version of every source the plan touches."""
    referenced = {ref for step in plan.steps for ref in step.inputs}
    step_names = {step.name for step in plan.steps}
    return {ref: catalog[ref].get("version", "unversioned")
            for ref in sorted(referenced - step_names) if ref in catalog}
```

The unversioned default is deliberately visible. A source with no version is one whose findings cannot be reproduced, and seeing that string in a stored result is more useful than an absent field nobody notices — it is a prompt to fix the source rather than the analysis.

## Operating This Stage Over Time

Plans acquire steps. A chain that started as three operations grows a filter, then a reprojection, then a cleanup, and each addition is individually justified. The cost is not just latency: every additional step is another place an intermediate can be empty, and a seven-step chain fails more often than a three-step one for reasons that have nothing to do with the data. Reviewing chain length distribution occasionally, and asking whether the longest ones could be expressed differently, is worth more than optimising any single step.

Source layers change beneath a plan. A layer renamed or re-partitioned upstream turns a working plan into a static check failure, which is the good case, or into a plan that runs against a different layer with the same name, which is not. Validating source names against the live catalog at plan time — not at execution time — turns the second case into the first.

The size sanity threshold needs attention as data grows. A multiplier that flagged cross joins against a corpus of thousands will flag legitimate results against a corpus of millions, and once it produces false positives it gets raised until it no longer fires. Derive it from the operation rather than from a constant: an intersection legitimately produces more features than its inputs, a dissolve produces fewer, and a buffer produces the same number.

Finally, keep the reference implementation stepwise even after fusing is introduced. The fused path is faster and unvalidated between steps, and the day the two disagree you will want the one whose intermediates were checked to be the one you trust.

## Failure Modes & Root Causes

**The confident zero.** Every step succeeds, the answer is zero, and no step produced anything after the second. Root cause: no gate on empty intermediates. Mitigation: stop at the first empty result, naming the step.

**The cross join.** An overlay against the wrong layer produces millions of slivers, and the chain continues. Root cause: no size sanity check. Mitigation: bound the output size relative to the inputs, per operation.

**The frame slip.** One step returns geometry in a different frame and the next overlays it against the original, producing an empty or nonsensical result. Root cause: frames checked at ingestion but not between steps. Mitigation: check the frame on every intermediate.

**The unrepeatable finding.** A number is produced and nobody can reconstruct how. Root cause: the plan discarded after execution. Mitigation: return the plan with the result, always.

## Production Validation Protocols

1. **Static plan check.** Assert every plan passes reference resolution and geometry-type checking before execution; a plan that fails should never reach a database.
2. **Empty-intermediate test.** Assert a chain whose second step produces nothing stops there, with that step named.
3. **Cross-join guard test.** Assert an overlay producing far more features than its inputs is stopped, using a fixture that triggers it.
4. **Frame-consistency test.** Assert an intermediate in an unexpected frame stops the chain rather than being reprojected silently.
5. **Fusion equivalence test.** Where fusing is enabled, assert the fused and stepwise paths produce identical results on a fixture set.
6. **Step-count publication.** Publish per-step row counts with every finding; they are the diagnostic that explains a surprising answer.

<figure class="diagram">
<svg viewBox="16 38 748 192" role="img" aria-labelledby="lgp-gates-t lgp-gates-d" xmlns="http://www.w3.org/2000/svg"><title id="lgp-gates-t">The four checks applied at every gate</title><desc id="lgp-gates-d">Non-empty, valid geometry, expected frame and plausible size — four cheap checks that between them catch nearly every silent intermediate failure.</desc><rect x="16" y="38" width="748" height="192" fill="#ffffff"/><rect x="30" y="52" width="170" height="120" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="216" y="52" width="170" height="120" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="402" y="52" width="170" height="120" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="588" y="52" width="162" height="120" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="115" y="84">non-empty</text><text x="301" y="84">valid</text><text x="487" y="84">right frame</text><text x="669" y="84">plausible size</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="115" y="112">catches the</text><text x="115" y="134">confident zero</text><text x="301" y="112">catches repairs</text><text x="301" y="134">the next step needs</text><text x="487" y="112">catches a slip</text><text x="487" y="134">between steps</text><text x="669" y="112">catches a</text><text x="669" y="134">cross join</text></g><text x="390" y="212" fill="#1f2937" font-size="13" text-anchor="middle">Four comparisons per step, and between them nearly every silent failure</text></svg>
<figcaption><b>Cheap checks, expensive failures.</b> Each of these is a length, a boolean or an integer comparison, and each one guards against a failure that otherwise reaches an answer intact.</figcaption>
</figure>

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should the model see intermediate results?</span></summary><p>Counts and validity, not geometry. A model told that the buffer step produced 1,240 features can reason about whether that is plausible and can explain the finding; a model handed the geometry spends context on data it cannot use and occasionally starts reasoning about individual features. The exception is a short chain over a handful of features, where the geometry is small and genuinely informative.</p></details>

<details class="faq-item"><summary><span>How long should a chain be allowed to get?</span></summary><p>Four or five steps covers most real analyses, and a plan longer than that is usually two analyses or one that should be expressed differently. Longer chains are not wrong, but each step multiplies the ways the whole can fail and lengthens the explanation the answer needs. A cap with an override, plus a log of overrides, keeps the pressure in the right direction.</p></details>

<details class="faq-item"><summary><span>What should happen when a step succeeds but produces something implausible?</span></summary><p>Stop, and say what was implausible. A dissolve that produced more features than it consumed, an intersection larger than either input, a buffer that changed the feature count — these are all valid outputs and all signals that the plan does not mean what it appears to. Continuing produces an answer nobody can defend, and the check costs one comparison.</p></details>

<details class="faq-item"><summary><span>Can the plan be repaired automatically when a check fails?</span></summary><p>Sometimes, and it should be offered rather than applied. A frame mismatch has an obvious repair, and applying it silently means the chain now contains a step nobody planned. Returning the failure with a suggested amendment lets the agent re-plan explicitly, which keeps the plan and the execution in agreement — which is the property that makes the finding defensible.</p></details>

<details class="faq-item"><summary><span>How does this differ from orchestration?</span></summary><p>Scope and durability. A pipeline is one analysis planned up front and executed within a turn; orchestration handles chains that span turns, checkpoint state and survive failures partway through. The boundary is roughly whether the work fits in an interactive budget — see <a href="/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/">multi-step spatial agent orchestration</a> for what happens when it does not.</p></details>

## Related

- Up to the section overview: [Geospatial Prompt Engineering and Tool Routing](/geospatial-prompt-engineering-tool-routing/)
- Technique: [Decomposing Natural Language into Geoprocessing Steps](/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/decomposing-natural-language-into-geoprocessing-steps/)
- Technique: [Validating Intermediate Geoprocessing Outputs](/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/validating-intermediate-geoprocessing-outputs/)
- Technique: [Planning Buffer, Overlay and Dissolve Chains from a Prompt](/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/planning-buffer-overlay-dissolve-chains-from-a-prompt/)
- Peer topic: [Multi-Step Spatial Agent Orchestration](/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/)
- Peer topic: [Topology Rule Enforcement via LLMs](/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/)
