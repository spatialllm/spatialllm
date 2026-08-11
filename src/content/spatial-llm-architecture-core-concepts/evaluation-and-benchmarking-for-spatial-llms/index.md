---
title: Evaluation and Benchmarking for Spatial LLMs
description: Measure a spatial agent on what actually fails — geometry accuracy, hallucinated coordinates, relation correctness and refusal quality — with thresholds you can defend.
slug: evaluation-and-benchmarking-for-spatial-llms
type: topic
breadcrumb: Evaluation and Benchmarking
datePublished: 2025-02-04
dateModified: 2026-08-11
---

# Evaluation and Benchmarking for Spatial LLMs

General language benchmarks say nothing useful about whether a system puts things in the right place. A spatial agent can score well on every text metric available and still put a building in the wrong county, because the failure is geometric and the metric was lexical. This topic builds the evaluation that measures what actually breaks: where the geometry landed, whether the coordinates were real, whether the relations hold, and whether the system refused when it should have.

It belongs to [spatial LLM architecture and core concepts](/spatial-llm-architecture-core-concepts/) and is the gate on everything else here — a normalization rule, a tokenization policy or a routing change is only an improvement if something measures it. Its measurements depend on the verification machinery in [spatial reasoning and relation inference](/spatial-llm-architecture-core-concepts/spatial-reasoning-and-relation-inference/), which supplies the ground truth for relation scoring.

<figure class="diagram">
<svg viewBox="16 32 748 228" role="img" aria-labelledby="eval-fam-t eval-fam-d" xmlns="http://www.w3.org/2000/svg"><title id="eval-fam-t">Four failure families and the metric each one needs</title><desc id="eval-fam-d">Geometry placement, coordinate plausibility, relation correctness and refusal quality each fail independently and each needs its own measurement; a single aggregate score hides all four.</desc><rect x="16" y="32" width="748" height="228" fill="#ffffff"/><rect x="30" y="46" width="360" height="94" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="410" y="46" width="340" height="94" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="30" y="156" width="360" height="94" rx="8" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><rect x="410" y="156" width="340" height="94" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="13.5" font-weight="600"><text x="52" y="76">placement</text><text x="432" y="76">plausibility</text><text x="52" y="186">relations</text><text x="432" y="186">refusal</text></g><g fill="#5b6471" font-size="12"><text x="52" y="102">did the shape land on the truth?</text><text x="52" y="124">measured by overlap ratio</text><text x="432" y="102">is the coordinate real at all?</text><text x="432" y="124">measured by screening rate</text><text x="52" y="212">do the stated relations hold?</text><text x="52" y="234">measured against real predicates</text><text x="432" y="212">did it decline when it should?</text><text x="432" y="234">measured on unanswerable cases</text></g></svg>
<figcaption><b>Four independent axes.</b> A system can improve on placement while regressing on refusal, and a single headline number will report the average as progress — which is why the report has four columns and no total.</figcaption>
</figure>

## Foundational Principles

**Every score needs a defined degenerate case.** Geometry evaluation is full of inputs that have no natural score: disjoint shapes, empty unions, unparseable output. Decide what each one scores before running anything, and make the decision visible in the code rather than emerging from an exception handler.

**Parse failures are results, not exclusions.** Dropping output the harness could not read flatters the model exactly where it is weakest. Score an unparseable answer as a failure and report the parse rate separately, so two systems with identical scores over their parseable outputs are still distinguishable.

**A refusal can be correct.** An agent that declines a question it lacks the data to answer has behaved well, and an evaluation that scores every non-answer as a miss will reward a system that guesses. Build unanswerable cases into the set deliberately and score refusal on them as a success.

## Step-by-Step Implementation Pipeline

### 1. Build a case set that spans the failure families

The set is the asset; the metrics are derived from it. Aim for coverage of each family rather than volume, and label each case with the family it exercises so the report can be broken down.

```python
import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("spatial_eval")


@dataclass(frozen=True)
class Case:
    case_id: str
    family: str                # placement | plausibility | relations | refusal
    question: str
    truth_geometry: Optional[object] = None
    truth_relations: tuple[tuple[str, str, str], ...] = ()
    answerable: bool = True
    region: Optional[str] = None
```

Include the region on every case. Spatial systems fail unevenly across geography — a model that handles one country well may be hopeless in another — and an aggregate score that mixes regions will report a competent system as mediocre and a regionally broken one as fine.

### 2. Score placement with a robust overlap ratio

Placement is measured by the overlap between predicted and true footprints, computed in an equal-area projection and defined on every degenerate input. The implementation, including repair and clamping, is in [measuring spatial overlap for model-generated geometries](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/measuring-spatial-iou-for-llm-generated-geometries/).

```python
def score_placement(predicted, truth, spatial_iou) -> tuple[float, str]:
    """Overlap ratio with an explicit note for every degenerate case."""
    if predicted is None:
        return 0.0, "no geometry produced"
    try:
        value = spatial_iou(predicted, truth)
    except Exception as exc:                       # the scorer itself must not abort a sweep
        log.warning("scorer failed: %s", exc)
        return 0.0, f"scorer error: {exc}"
    if value == 0.0:
        return 0.0, "disjoint or unusable"
    return value, ""
```

### 3. Screen for implausible coordinates before scoring anything

A coordinate in the wrong hemisphere scores zero on overlap and tells you nothing about how wrong it was. Screening separates "produced a bad shape" from "produced a fictional place", which are different problems with different fixes. The screening rules are in [detecting hallucinated coordinates in model output](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/detecting-hallucinated-coordinates-in-llm-output/).

```python
def screen_plausibility(geom, expected_region_bbox) -> tuple[bool, str]:
    """Cheap checks that catch fiction before expensive geometry runs."""
    if geom is None or geom.is_empty:
        return False, "no geometry"
    x, y = geom.representative_point().x, geom.representative_point().y
    if not (-180.0 <= x <= 180.0 and -90.0 <= y <= 90.0):
        return False, "outside the coordinate domain"
    if abs(x) < 1e-9 and abs(y) < 1e-9:
        return False, "null island"
    w, s, e, n = expected_region_bbox
    if not (w - 1 <= x <= e + 1 and s - 1 <= y <= n + 1):
        return False, "outside the expected region"
    return True, ""
```

The null-island check earns its place. A coordinate pair of exactly zero is almost always a parsing failure or a missing value rendered as a number, and it is common enough to be worth naming rather than letting it score as an ordinary miss thousands of kilometres away.

### 4. Score relations against real predicates

Relation scoring compares the relations an answer asserted against the relations that actually hold, computed over the truth geometry. Precision and recall are both meaningful here and measure different failures: precision falls when the system asserts relations that do not hold, recall falls when it misses ones that do.

```python
def score_relations(asserted: set, truth: set) -> dict:
    """Precision and recall over relation triples, with the disagreements listed."""
    if not truth and not asserted:
        return {"precision": 1.0, "recall": 1.0, "false": [], "missed": []}
    false_positives = sorted(asserted - truth)
    missed = sorted(truth - asserted)
    correct = len(asserted & truth)
    precision = correct / len(asserted) if asserted else 1.0
    recall = correct / len(truth) if truth else 1.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "false": false_positives, "missed": missed}
```

Returning the actual disagreements rather than only the numbers is what makes this metric usable. A recall of 0.6 is a number; a list of the four relations that were missed is a bug report.

<figure class="diagram">
<svg viewBox="76 6 688 240" role="img" aria-labelledby="eval-conf-t eval-conf-d" xmlns="http://www.w3.org/2000/svg"><title id="eval-conf-t">Answer and refusal against answerability</title><desc id="eval-conf-d">A two-by-two of whether the case was answerable against whether the system answered, showing that a refusal on an unanswerable case is a success and an answer on one is the most damaging outcome.</desc><rect x="76" y="6" width="688" height="240" fill="#ffffff"/><rect x="180" y="46" width="280" height="88" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="470" y="46" width="280" height="88" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="180" y="144" width="280" height="88" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="470" y="144" width="280" height="88" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="320" y="80">answered correctly</text><text x="610" y="80">refused unnecessarily</text><text x="320" y="178">answered anyway</text><text x="610" y="178">refused correctly</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="320" y="106">the intended outcome</text><text x="610" y="106">costs usefulness, not trust</text><text x="320" y="204">the outcome that destroys trust</text><text x="610" y="204">a success, and usually unscored</text></g><text x="90" y="86" fill="#5b6471" font-size="12">answerable</text><text x="90" y="184" fill="#5b6471" font-size="12">not answerable</text><text x="320" y="30" fill="#5b6471" font-size="12" text-anchor="middle">answered</text><text x="610" y="30" fill="#5b6471" font-size="12" text-anchor="middle">refused</text></svg>
<figcaption><b>The two green cells are both wins.</b> Most evaluation harnesses score only the top-left and treat the bottom-right as a failure, which rewards a system that answers everything — including the questions it has no data for.</figcaption>
</figure>

### 5. Score refusal on the cases that cannot be answered

Refusal is scored on a separate axis with its own truth: a case marked unanswerable expects a decline, and a case marked answerable expects an attempt. Scoring them together with placement produces a metric that improves when the system becomes more reckless.

```python
def score_refusal(answered: bool, answerable: bool) -> tuple[float, str]:
    """Refusing an unanswerable case is a success, not a miss."""
    if answerable and answered:
        return 1.0, ""
    if answerable and not answered:
        return 0.0, "declined a question it had the data for"
    if not answerable and not answered:
        return 1.0, "correctly declined"
    return 0.0, "answered a question it had no basis for"
```

The last branch deserves the heaviest weight in any report a reader will act on. An unnecessary refusal costs usefulness and is visible to the user, who can rephrase; an unfounded answer costs trust and is invisible, because it looks exactly like a good one.

### 6. Report per family and per region, never as one number

The report is a table, not a scalar. A single blended score can improve while two of its components regress, and the blend weights encode a judgement nobody wrote down.

```python
from collections import defaultdict


def summarise(results: list[dict]) -> dict:
    """Break results down by family and region; no aggregate is produced."""
    buckets = defaultdict(list)
    for r in results:
        buckets[(r["family"], r.get("region", "all"))].append(r)
    report = {}
    for (family, region), rows in sorted(buckets.items()):
        scores = [r["score"] for r in rows]
        scores.sort()
        report[f"{family}/{region}"] = {
            "n": len(rows),
            "median": scores[len(scores) // 2],
            "p10": scores[max(0, int(0.1 * len(scores)) - 1)],
            "parse_rate": round(sum(r["parsed"] for r in rows) / len(rows), 4),
        }
    return report
```

Reporting the tenth percentile alongside the median is deliberate. Spatial quality is bimodal — most answers are close and a few are in another country — and a mean sits between two peaks describing neither. The percentile is what a release gate should read.

### 7. Fix the harness before fixing the model

An evaluation harness has bugs, and its bugs look exactly like model regressions. Before acting on a score change, verify that the truth data has not moved, the library versions have not changed, and the scoring code produces identical results on a frozen fixture.

```python
FROZEN_EXPECTED = {"case-001": 0.8734, "case-014": 0.0, "case-102": 1.0}


def harness_self_check(score_case) -> None:
    """Fail loudly if the harness itself has changed behaviour."""
    for case_id, expected in FROZEN_EXPECTED.items():
        got = round(score_case(case_id), 4)
        if abs(got - expected) > 1e-4:
            raise AssertionError(
                f"harness drift on {case_id}: expected {expected}, got {got}; "
                "check library versions and truth data before blaming the model")
```

### 8. Set thresholds from the distribution, not from a round number

A release gate needs a number, and the number should come from observed behaviour rather than from taste. The procedure and its pitfalls are in [setting release thresholds for spatial agents](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/setting-release-thresholds-for-spatial-agents/).

```python
def propose_threshold(historical: list[float], allowed_regression: float = 0.02) -> float:
    """A gate just below current behaviour, so noise does not block a release."""
    if len(historical) < 5:
        raise ValueError("not enough history to propose a threshold")
    ordered = sorted(historical)
    baseline = ordered[len(ordered) // 2]
    return round(max(0.0, baseline - allowed_regression), 4)
```

### 9. Wire the gate into the release, with an override that is recorded

A gate nobody can override will be removed the first time it blocks an urgent fix. A gate whose override is silent is not a gate. The workable arrangement is an override that requires a reason and appears in the release record.

```python
def release_gate(report: dict, thresholds: dict, override_reason: str | None) -> bool:
    failures = [k for k, v in report.items()
                if k in thresholds and v["p10"] < thresholds[k]]
    if not failures:
        return True
    if override_reason:
        log.warning("release gate overridden for %s: %s", ", ".join(failures), override_reason)
        return True
    log.error("release blocked by %s", ", ".join(failures))
    return False
```

## Operating This Stage Over Time

Evaluation sets rot in a specific way: they stop being hard. Cases that were failing get fixed, the set fills with cases the system passes, and the score rises toward one while real-world quality plateaus. The remedy is to add cases from production failures continuously and to retire nothing — a case that has passed for a year costs a second to run and is the thing that will catch a regression when someone rewrites the tokenizer.

Truth data is the other durable problem. Geometry truth ages: parcels are resurveyed, boundaries are corrected, and a case labelled three years ago may now disagree with reality rather than with the model. Version the truth alongside the cases, record which version a score was measured against, and treat a truth correction as an event that invalidates historical comparisons rather than one that silently changes them.

Library versions belong in the same category. A geometry engine upgrade can move overlap scores by a percent or two through changes in repair behaviour, which is indistinguishable from a model regression unless the versions are pinned and recorded. The self-check in step 6 exists precisely to make that distinguishable, and it should run before every sweep rather than occasionally.

Finally, resist the pull toward a single headline number. It will be asked for, repeatedly, and the honest response is a small table — four families, a median and a tenth percentile each. The reason is not pedantry: the families fail independently and are fixed by different work, and a blended score is a decision about their relative importance that is better made explicitly, in a release conversation, than implicitly, in a weighting constant.

A last note on culture rather than code. Evaluation only changes behaviour when its output is read by the people making decisions, which means the report has to be short enough to read and stable enough to compare. Four families, two statistics each, a parse rate and a region breakdown fits on a screen; a hundred-metric dashboard does not get read and therefore does not gate anything, however carefully it was built.

## Failure Modes & Root Causes

**The flattering filter.** Unparseable outputs are excluded and the score rises. Root cause: treating a parse failure as a harness problem rather than a result. Mitigation: score it zero, report the parse rate separately.

**The guessing reward.** Every refusal counts as a miss, so the highest-scoring system is the one that never declines. Root cause: no unanswerable cases in the set. Mitigation: build them in and score refusal as success on them.

**The regional average.** A system that works in one region and fails in another reports as uniformly mediocre. Root cause: aggregation across regions. Mitigation: region on every case, breakdown in every report.

**The harness regression.** A library upgrade moves scores and the team spends a week investigating the model. Root cause: unpinned dependencies and no self-check. Mitigation: frozen fixtures with expected values, checked before every sweep.

## Production Validation Protocols

1. **Harness self-check first.** Run the frozen fixture check before any sweep; a drift there invalidates everything downstream.
2. **Parse-rate reporting.** Publish the share of outputs that parsed, per family; a score without it is not comparable across runs.
3. **Refusal scoring.** Assert the set contains unanswerable cases and that refusal on them scores as success.
4. **Region breakdown.** Assert every case carries a region and every report is broken down by it.
5. **Percentile gate.** Gate releases on the tenth percentile rather than the mean, and record the threshold with the release.
6. **Override audit.** Assert that any gate override carries a recorded reason, and review overrides periodically — a gate overridden every release is a gate that needs rethinking, not repeating.

<figure class="diagram">
<svg viewBox="56 7 648 237" role="img" aria-labelledby="eval-dist-t eval-dist-d" xmlns="http://www.w3.org/2000/svg"><title id="eval-dist-t">Why the mean is the wrong statistic for spatial scores</title><desc id="eval-dist-d">A bimodal distribution of overlap scores with a cluster near one and a cluster near zero; the mean falls in an empty region between them while the tenth percentile tracks the failing tail.</desc><rect x="56" y="7" width="648" height="237" fill="#ffffff"/><text x="380" y="32" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">Distribution of overlap scores across one evaluation sweep</text><g fill="#fdeaee" stroke="#b3324f" stroke-width="2"><rect x="70" y="96" width="60" height="76" rx="4"/><rect x="140" y="140" width="60" height="32" rx="4"/></g><g fill="#eef2f7" stroke="#5b6471" stroke-width="2"><rect x="210" y="160" width="60" height="12" rx="3"/><rect x="280" y="162" width="60" height="10" rx="3"/><rect x="350" y="160" width="60" height="12" rx="3"/></g><g fill="#e4f5ec" stroke="#12805c" stroke-width="2"><rect x="420" y="136" width="60" height="36" rx="4"/><rect x="490" y="104" width="60" height="68" rx="4"/><rect x="560" y="72" width="60" height="100" rx="4"/><rect x="630" y="88" width="60" height="84" rx="4"/></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="100" y="194">0.0</text><text x="310" y="194">0.5</text><text x="660" y="194">1.0</text></g><text x="310" y="226" fill="#b3324f" font-size="12.5" text-anchor="middle">the mean lands here, where almost nothing is</text></svg>
<figcaption><b>Two populations, not one.</b> Answers are usually close or badly wrong, with little in between, so the average describes a case that rarely occurs — while the tenth percentile sits inside the failing cluster and moves when it does.</figcaption>
</figure>

## Frequently Asked Questions

<details class="faq-item"><summary><span>How many cases does a useful set need?</span></summary><p>Fewer than expected for detecting large regressions and many more for detecting small ones. Fifty well-chosen cases spanning four families will catch anything that breaks a subsystem; distinguishing a two-point improvement needs several hundred, because the tenth percentile of a fifty-case set moves by a whole case. Start at fifty, add every production failure, and let the set grow into the precision you need rather than trying to construct it up front.</p></details>

<details class="faq-item"><summary><span>Should a model be used as a judge for spatial answers?</span></summary><p>For prose quality, sometimes; for geometry, no. Geometric correctness is computable exactly, and substituting a judgement for a computation trades a reliable measurement for an unreliable one. Where a judge genuinely helps is in scoring whether an answer's hedging matched its evidence — a linguistic property that no predicate captures and that matters a great deal for trust.</p></details>

<details class="faq-item"><summary><span>What should an evaluation do about non-determinism?</span></summary><p>Measure it. Run a subset of cases several times and report the spread, because a system whose answers vary between runs has a quality property that a single sweep cannot see. If the spread is large relative to the differences you are trying to detect, lower the sampling temperature for evaluation runs and say that you did — but do not pretend the deployed system is deterministic if it is not.</p></details>

<details class="faq-item"><summary><span>How does this relate to retrieval evaluation?</span></summary><p>They measure different things and should stay separate. Retrieval quality and answer quality fail independently, and a combined score cannot tell you which one moved — a system whose retrieval degraded and whose reasoning improved can score flat. Keep a labelled retrieval set of its own, as described in <a href="/geospatial-rag-pipelines/">geospatial RAG pipelines</a>, and read the two reports side by side.</p></details>

<details class="faq-item"><summary><span>Is it worth evaluating latency and cost alongside quality?</span></summary><p>Yes, in the same sweep, because they trade against each other and a quality improvement bought with a tenfold cost increase is a decision someone should make consciously. Record tokens consumed and wall-clock time per case, report them beside the quality columns, and see <a href="/geospatial-prompt-engineering-tool-routing/cost-and-latency-budgets-for-spatial-agents/">cost and latency budgets for spatial agents</a> for how to act on them.</p></details>

## Related

- Up to the section overview: [Spatial LLM Architecture and Core Concepts](/spatial-llm-architecture-core-concepts/)
- Technique: [Measuring Spatial Overlap for Model-Generated Geometries](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/measuring-spatial-iou-for-llm-generated-geometries/)
- Technique: [Detecting Hallucinated Coordinates in LLM Output](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/detecting-hallucinated-coordinates-in-llm-output/)
- Technique: [Building Regression Test Harnesses for Spatial Agents](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/building-regression-test-harnesses-for-spatial-agents/)
- Technique: [Setting Release Thresholds for Spatial Agents](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/setting-release-thresholds-for-spatial-agents/)
- Peer topic: [Spatial Reasoning and Relation Inference](/spatial-llm-architecture-core-concepts/spatial-reasoning-and-relation-inference/)
