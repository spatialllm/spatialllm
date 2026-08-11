---
title: Topology Rule Enforcement via LLMs
description: State the topological rules a dataset must satisfy, check generated geometry against them, and repair or reject with a tolerance chosen from the data's own accuracy.
slug: topology-rule-enforcement-via-llms
type: topic
breadcrumb: Topology Rule Enforcement
datePublished: 2025-04-22
dateModified: 2026-08-11
---

# Topology Rule Enforcement via LLMs

Geometry produced or edited by a model is individually plausible and collectively wrong. Parcels overlap by half a metre, a boundary that should follow a river diverges by three, and a set of zones that must tile an area leaves slivers between them. None of those is a validity failure — every shape is a legal polygon — and none of them is visible without rules that say what the collection is supposed to satisfy.

This topic belongs to [geospatial prompt engineering and tool routing](/geospatial-prompt-engineering-tool-routing/) and complements [spatial reasoning and relation inference](/spatial-llm-architecture-core-concepts/spatial-reasoning-and-relation-inference/): that topic verifies claims a model makes about relations, this one enforces relations a dataset must have.

<figure class="diagram">
<svg viewBox="46 46 704 203" role="img" aria-labelledby="tre-valid-t tre-valid-d" xmlns="http://www.w3.org/2000/svg"><title id="tre-valid-t">Valid geometry that breaks a topological rule</title><desc id="tre-valid-d">Two parcels that overlap slightly and two that leave a sliver between them are both made of individually valid polygons, and only a rule about the collection detects either.</desc><rect x="46" y="46" width="704" height="203" fill="#ffffff"/><rect x="60" y="60" width="150" height="110" rx="4" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="196" y="60" width="150" height="110" rx="4" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2" fill-opacity="0.7"/><text x="203" y="200" fill="#1f2937" font-size="12.5" text-anchor="middle">overlap: both polygons valid</text><rect x="440" y="60" width="140" height="110" rx="4" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="596" y="60" width="140" height="110" rx="4" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="588" y="200" fill="#1f2937" font-size="12.5" text-anchor="middle">sliver: both polygons valid</text><text x="390" y="232" fill="#5b6471" font-size="12" text-anchor="middle">Validity is a property of one shape; topology is a property of the collection</text></svg>
<figcaption><b>Every shape here passes a validity check.</b> The failure is in the relationship between them, which no per-geometry test can see and which a rule about the collection detects immediately.</figcaption>
</figure>

## Foundational Principles

**Rules are declared, not implied.** A dataset either must tile without gaps, or must not overlap, or must share boundaries with a reference layer — and which of those applies is a statement about the data that belongs in configuration rather than in whoever last edited it.

**Tolerance comes from the data's accuracy.** Exact topological equality is unachievable in data captured by different surveys, and a rule with no tolerance rejects everything. The tolerance is the capture accuracy of the less accurate source, and it is stated with every result.

**Repair is offered, never silent.** A snap that closes a sliver moves vertices, and moving vertices in a regulated dataset is a decision somebody has to own. Detect, propose, and let the repair be applied deliberately.

## Step-by-Step Implementation Pipeline

### 1. Declare the rules a dataset must satisfy

Four rules cover most real requirements, and expressing them as data makes them reviewable by the people who understand the dataset rather than only by the people who wrote the checker.

```python
import logging
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

log = logging.getLogger("topology_rules")

RuleKind = Literal["no_overlap", "no_gaps", "covered_by", "boundary_shared"]


@dataclass(frozen=True)
class Rule:
    kind: RuleKind
    layer: str
    reference: Optional[str] = None      # for covered_by and boundary_shared
    tolerance_m: float = 0.0
    severity: Literal["error", "warning"] = "error"


@dataclass(frozen=True)
class Violation:
    rule: Rule
    feature_ids: tuple[str, ...]
    measure_m: float                     # how far the rule is broken, in metres
    detail: str
```

Recording the measure rather than a boolean is what makes a violation triageable. A twelve-centimetre overlap between parcels captured to a metre is noise; a four-metre overlap is an error, and both fail the same rule.

### 2. Choose the tolerance from the sources

The tolerance is not a tuning parameter. It comes from the stated accuracy of the data, and where two layers differ it comes from the worse of them.

```python
def tolerance_for(layer_accuracy_m: float, reference_accuracy_m: Optional[float] = None,
                  floor_m: float = 0.01) -> float:
    """Tolerance is the worse of the two capture accuracies, never less than the floor."""
    candidates = [a for a in (layer_accuracy_m, reference_accuracy_m) if a is not None]
    if not candidates:
        log.info("no stated accuracy for this layer; using the floor tolerance")
        return floor_m
    return max(floor_m, max(candidates))
```

Deriving it this way makes every result defensible. "Adjacent within the stated accuracy of both sources" is a claim you can put in front of a surveyor; "adjacent within half a metre because that made the test pass" is not. Choosing a snapping tolerance that preserves topology rather than destroying it is developed in [choosing a snapping tolerance that preserves topology](/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/choosing-a-snapping-tolerance-that-preserves-topology/).

### 3. Check overlap and gaps as area measures

Both rules reduce to an area computed in a metric projection, which makes them comparable to each other and to a tolerance expressed in metres.

```python
def check_no_overlap(features, to_metric, tolerance_m: float) -> list[Violation]:
    """Any pairwise intersection with area beyond the tolerance is a violation."""
    violations = []
    projected = [(f["id"], to_metric(f["geom"])) for f in features]
    index = build_index(projected)                       # a spatial index, not a nested loop
    for fid, geom in projected:
        for other_id, other in index.query(geom.bounds):
            if other_id <= fid:
                continue                                 # each pair once
            try:
                shared = geom.intersection(other)
            except Exception as exc:
                log.warning("overlay failed for %s/%s: %s", fid, other_id, exc)
                continue
            if shared.is_empty or shared.area <= tolerance_m ** 2:
                continue
            violations.append(Violation(
                Rule("no_overlap", "", tolerance_m=tolerance_m),
                (fid, other_id), round(shared.area ** 0.5, 3),
                f"overlap of {shared.area:.2f} m²"))
    return violations
```

Using a spatial index rather than a nested loop is the difference between a check that runs on a thousand features and one that runs on a million. The naive form is quadratic and looks perfectly reasonable until the dataset grows.

### 4. Detect slivers as gaps between neighbours

A gap rule asks whether a set of polygons tiles a region, and the useful output is the gaps themselves rather than a pass or fail.

```python
def check_no_gaps(features, boundary, to_metric, tolerance_m: float) -> list[Violation]:
    """Everything inside the boundary must be covered; report the uncovered parts."""
    from shapely.ops import unary_union
    covered = unary_union([to_metric(f["geom"]) for f in features])
    gaps = to_metric(boundary).difference(covered)
    if gaps.is_empty:
        return []
    parts = list(getattr(gaps, "geoms", [gaps]))
    violations = []
    for part in parts:
        if part.area <= tolerance_m ** 2:
            continue                                     # within capture accuracy
        violations.append(Violation(
            Rule("no_gaps", "", tolerance_m=tolerance_m), (),
            round(part.area ** 0.5, 3), f"uncovered area of {part.area:.2f} m²"))
    return violations
```

<figure class="diagram">
<svg viewBox="56 9 600 224" role="img" aria-labelledby="tre-tol-t tre-tol-d" xmlns="http://www.w3.org/2000/svg"><title id="tre-tol-t">Tolerance separating capture noise from real violations</title><desc id="tre-tol-d">Overlaps below the capture accuracy of the sources are noise and pass; overlaps above it are genuine violations, and the boundary between them is the stated accuracy rather than a chosen number.</desc><rect x="56" y="9" width="600" height="224" fill="#ffffff"/><text x="390" y="34" fill="#5b6471" font-size="13" text-anchor="middle">Overlap size against the stated capture accuracy of one metre</text><g fill="#e4f5ec" stroke="#12805c" stroke-width="2"><rect x="70" y="96" width="70" height="52" rx="5"/><rect x="156" y="96" width="70" height="52" rx="5"/><rect x="242" y="96" width="70" height="52" rx="5"/></g><g fill="#fdeaee" stroke="#b3324f" stroke-width="2"><rect x="400" y="96" width="70" height="52" rx="5"/><rect x="486" y="96" width="70" height="52" rx="5"/><rect x="572" y="96" width="70" height="52" rx="5"/></g><rect x="340" y="76" width="6" height="92" rx="3" fill="#c46a3d"/><text x="343" y="192" fill="#c46a3d" font-size="12" text-anchor="middle">1 m</text><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="190" y="216">capture noise: pass</text><text x="520" y="216">real violations: report</text></g></svg>
<figcaption><b>The line is set by the data, not by the checker.</b> A tolerance chosen to make a particular dataset pass is a tolerance that will accept a genuine error in the next one, which is why deriving it from stated accuracy is worth the small effort.</figcaption>
</figure>

### 5. Propose a repair rather than applying one

For each violation there is usually an obvious repair, and proposing it — with the vertex movement it implies — is what lets a person or a policy decide.

```python
@dataclass(frozen=True)
class Repair:
    violation: Violation
    action: Literal["snap", "trim", "fill", "none"]
    max_movement_m: float
    note: str


def propose(violation: Violation, tolerance_m: float) -> Repair:
    """Suggest a repair and state how far it would move geometry."""
    if violation.rule.kind == "no_overlap":
        if violation.measure_m <= tolerance_m * 2:
            return Repair(violation, "snap", violation.measure_m,
                          "snap the shared boundary; movement within twice the tolerance")
        return Repair(violation, "none", 0.0,
                      "overlap is too large to be a capture artefact; investigate the source")
    if violation.rule.kind == "no_gaps":
        return Repair(violation, "fill", violation.measure_m,
                      "assign the gap to the neighbour with the longest shared boundary")
    return Repair(violation, "none", 0.0, "no automatic repair for this rule")
```

Refusing to propose a repair for a large overlap is the important behaviour. A four-metre overlap between parcels is not a snapping problem; it is two sources disagreeing about where a boundary is, and snapping it away destroys the evidence of that disagreement. The snapping and noding mechanics are covered in [snapping and noding LLM-generated geometries](/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/snapping-and-noding-llm-generated-geometries/).

### 6. Apply repairs as a batch, with a record

When repairs are applied, they are applied together, recorded, and re-checked — because snapping one boundary frequently creates a violation somewhere else.

```python
def apply_repairs(features, repairs: Sequence[Repair], apply, recheck) -> dict:
    """Apply, re-check, and report what changed. Repairs can create new violations."""
    applied = [r for r in repairs if r.action != "none"]
    if not applied:
        return {"applied": 0, "remaining": len(repairs), "new_violations": []}
    updated = apply(features, applied)
    after = recheck(updated)
    log.info("applied %d repair(s); %d violation(s) remain", len(applied), len(after))
    return {"applied": len(applied), "updated": updated,
            "remaining": len(after), "new_violations": after,
            "max_movement_m": max(r.max_movement_m for r in applied)}
```

### 7. Enforce rules on generated geometry before it is stored

Geometry a model produced or edited goes through the same rules as anything else, and the check runs before the write rather than after. A violation found after storage is a data-quality task; found before, it is a rejected tool call.

```python
def gate_generated(features, rules: Sequence[Rule], check_all) -> tuple[bool, list[Violation]]:
    """Generated geometry must satisfy the same rules as everything else."""
    violations = [v for v in check_all(features, rules) if v.rule.severity == "error"]
    if violations:
        worst = max(violations, key=lambda v: v.measure_m)
        log.info("rejecting generated geometry: %s (worst %.2f m)",
                 worst.detail, worst.measure_m)
    return (not violations), violations
```

### 8. Report violations in a form a person can work through

A list of a thousand violations is unusable; the same list grouped by rule, sorted by measure and capped is a work queue. Reporting the distribution alongside the top offenders is what lets someone judge whether the dataset has a systematic problem or a handful of bad features.

```python
def violation_report(violations: Sequence[Violation], top: int = 20) -> dict:
    by_rule: dict[str, list[Violation]] = {}
    for v in violations:
        by_rule.setdefault(v.rule.kind, []).append(v)
    return {
        kind: {
            "count": len(items),
            "max_m": round(max(i.measure_m for i in items), 3),
            "median_m": round(sorted(i.measure_m for i in items)[len(items) // 2], 3),
            "worst": [i.feature_ids for i in
                      sorted(items, key=lambda i: -i.measure_m)[:top]],
        }
        for kind, items in sorted(by_rule.items())
    }
```

### 9. Keep a rule set per dataset, not per system

Rules belong to a dataset, and a system serving several datasets needs several rule sets. A single global set produces either rules that do not apply — a no-gaps rule against a point layer — or rules so weak they check nothing.

```python
RULE_SETS = {
    "parcels": (
        Rule("no_overlap", "parcels", tolerance_m=0.5),
        Rule("covered_by", "parcels", reference="districts", tolerance_m=0.5),
    ),
    "zoning": (
        Rule("no_overlap", "zoning", tolerance_m=1.0),
        Rule("no_gaps", "zoning", reference="district_boundary", tolerance_m=1.0),
    ),
}


def rules_for(layer: str) -> Sequence[Rule]:
    rules = RULE_SETS.get(layer)
    if not rules:
        log.info("no topology rules declared for layer %r", layer)
        return ()
    return rules
```

The log line on a missing rule set matters more than it looks. A layer with no rules is not necessarily wrong — many layers genuinely have no topological requirements — but a layer that acquired rules and then lost them in a configuration edit looks identical, and the message is the only thing that distinguishes them.

### 10. Decide who owns a violation

A violation is a finding about data, and data has an owner. Routing violations to whoever maintains the source, with the feature identifiers and the measure, is what turns a report into a fix; routing them all to whoever built the checker turns it into a backlog.

```python
def route_violations(violations: Sequence[Violation], owners: dict[str, str]) -> dict:
    """Group violations by the owner of the layer they concern."""
    routed: dict[str, list[Violation]] = {}
    for v in violations:
        owner = owners.get(v.rule.layer, "unassigned")
        routed.setdefault(owner, []).append(v)
    if "unassigned" in routed:
        log.warning("%d violation(s) concern layers with no recorded owner",
                    len(routed["unassigned"]))
    return routed
```

An unassigned bucket that keeps growing is itself the finding. It means a layer entered the system without anyone taking responsibility for its quality, and the topology check has surfaced that rather than the geometry problem it was looking for.

## Operating This Stage Over Time

Rules accumulate and then stop being read. A dataset acquires a rule for each problem someone once had, and after two years the configuration contains rules that no longer apply to data that has since been replaced. Reviewing the rule set against the violations it actually produces — a rule that has never fired in a year is either unnecessary or broken — keeps it meaningful.

Tolerances drift in the dangerous direction. Each time a check produces too many violations, the tolerance is raised slightly, and each raise is justified by that day's data. Deriving tolerance from stated accuracy rather than from a constant removes most of this pressure, and recording the derivation makes a manual override visible when it happens.

The check's cost grows quadratically with feature count unless the index is doing its job. A check that ran in seconds against a hundred thousand features and takes an hour against a million usually means the index has stopped being used — a bounds computation moved inside the loop, or the index is rebuilt per query. Tracking check duration against feature count makes that obvious rather than mysterious.

Finally, watch the ratio of proposed to applied repairs. A system where every proposal is applied has a repair step that is effectively automatic, which may be intended and should be a decision; one where none are applied has a proposal step nobody trusts, which is worth understanding before it is removed.

## Failure Modes & Root Causes

**The silent snap.** Vertices move to satisfy a rule and nobody records it, so the stored geometry differs from the source with no trace. Root cause: repair applied inside the check. Mitigation: propose and apply as separate steps, with the movement recorded.

**The tolerance that swallowed a real error.** A raised tolerance passes a four-metre disagreement as capture noise. Root cause: tolerance treated as a tuning parameter. Mitigation: derive it from stated accuracy; require an explicit override with a reason.

**The quadratic check.** A rule that ran fine on a small dataset takes hours on a large one. Root cause: pairwise comparison without a spatial index. Mitigation: index-backed candidate selection, with duration tracked against feature count.

**The repair that broke something else.** Snapping one boundary opened a gap against a third feature. Root cause: repairs applied without re-checking. Mitigation: batch, apply, re-check, and report new violations.

## Production Validation Protocols

1. **Rule-coverage assertion.** Assert every layer that should have rules has at least one, and that every rule names a layer that exists.
2. **Tolerance-provenance test.** Assert every tolerance is derived from a stated accuracy or carries an explicit override with a reason.
3. **Index-use test.** Assert the overlap check uses a spatial index, using a fixture large enough that a quadratic implementation would time out.
4. **Repair-record test.** Assert every applied repair records its maximum vertex movement.
5. **Re-check assertion.** Assert repairs are followed by a re-check and that new violations are reported rather than swallowed.
6. **Generated-geometry gate.** Assert model-produced geometry is checked before storage, with a fixture that violates a rule.

<figure class="diagram">
<svg viewBox="46 46 699 172" role="img" aria-labelledby="tre-cascade-t tre-cascade-d" xmlns="http://www.w3.org/2000/svg"><title id="tre-cascade-t">A repair creating a new violation</title><desc id="tre-cascade-d">Snapping two parcels together to close an overlap moves a shared vertex, which opens a gap against a third parcel that was previously compliant.</desc><rect x="46" y="46" width="699" height="172" fill="#ffffff"/><rect x="60" y="60" width="110" height="110" rx="4" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="170" y="60" width="110" height="110" rx="4" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="280" y="60" width="70" height="110" rx="4" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="205" y="200" fill="#1f2937" font-size="12.5" text-anchor="middle">before: one overlap</text><rect x="420" y="60" width="110" height="110" rx="4" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="530" y="60" width="110" height="110" rx="4" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="660" y="60" width="70" height="110" rx="4" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="580" y="200" fill="#1f2937" font-size="12.5" text-anchor="middle">after: the overlap closed, a gap opened</text></svg>
<figcaption><b>Repairs are not local.</b> A shared vertex participates in several relationships, and moving it to satisfy one of them changes the others — which is why a repair pass is followed by a full re-check rather than a spot verification.</figcaption>
</figure>

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should a model ever be asked to repair topology?</span></summary><p>To describe the problem, yes; to move the vertices, no. Snapping and noding are deterministic geometric operations with well-defined semantics, and a model performing them by editing coordinates will produce output that is plausible and subtly wrong in ways no reader can check. Where a model helps is in explaining a violation and in choosing between proposed repairs, which is a judgement rather than a computation.</p></details>

<details class="faq-item"><summary><span>What tolerance should apply when accuracy is not stated?</span></summary><p>A small floor, and a note that the tolerance is unfounded. Inventing a plausible accuracy figure is worse than using a conservative floor, because it produces results that look derived. Where the missing figure matters — a regulated boundary dataset, say — the honest answer is that the check cannot be run defensibly until the accuracy is established.</p></details>

<details class="faq-item"><summary><span>How should rules interact with the validity checks at ingestion?</span></summary><p>Sequentially and separately. Validity is a per-geometry property checked at the ingestion gate; topology is a collection property checked after a dataset is assembled. Running them together conflates two very different failure modes, and a report that mixes "this polygon self-intersects" with "these two parcels overlap" is harder to act on than two reports.</p></details>

<details class="faq-item"><summary><span>Can rules be enforced incrementally as features are edited?</span></summary><p>Yes for the local rules — overlap and shared boundary can be checked against a feature's neighbours cheaply — and no for the global ones, since a gap rule is a statement about a whole region. The practical arrangement is incremental checks on edit for what is affordable, plus a full pass on a schedule, with the full pass being the authority.</p></details>

<details class="faq-item"><summary><span>What should happen when a dataset has too many violations to fix?</span></summary><p>Report the distribution and stop, rather than proposing thousands of repairs. A dataset with fifty thousand violations has a systematic problem — a frame mismatch, a wrong reference layer, a bad import — and the fix is upstream. The median and maximum measures are what identify it: violations clustered near the tolerance are capture noise, and violations clustered at ten metres are a misalignment.</p></details>

## Related

- Up to the section overview: [Geospatial Prompt Engineering and Tool Routing](/geospatial-prompt-engineering-tool-routing/)
- Technique: [Enforcing Topological Rules in LLM-Generated Geometries](/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/enforcing-topological-rules-in-llm-generated-geometries/)
- Technique: [Snapping and Noding LLM-Generated Geometries](/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/snapping-and-noding-llm-generated-geometries/)
- Technique: [Choosing a Snapping Tolerance That Preserves Topology](/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/choosing-a-snapping-tolerance-that-preserves-topology/)
- Related topic: [Spatial Reasoning and Relation Inference](/spatial-llm-architecture-core-concepts/spatial-reasoning-and-relation-inference/)
- Peer topic: [LLM-Assisted Geoprocessing Pipelines](/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/)
