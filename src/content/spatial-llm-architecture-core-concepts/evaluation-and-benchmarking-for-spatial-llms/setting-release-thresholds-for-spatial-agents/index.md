---
title: Setting Release Thresholds for Spatial Agents
description: Derive gate values from observed behaviour rather than round numbers, gate on a percentile rather than a mean, and make overrides visible instead of inevitable.
slug: setting-release-thresholds-for-spatial-agents
type: howto
breadcrumb: Setting Release Thresholds
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Setting Release Thresholds for Spatial Agents

A release gate needs a number, and the number is usually chosen by someone saying "let's require ninety per cent". That figure has no relationship to the system's actual behaviour, so it either blocks every release or none, and within two months it has been removed. This guide derives thresholds from measured history instead, as the decision layer over [evaluation and benchmarking for spatial LLMs](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/).

## When to Use This Approach

Set thresholds once you have enough history to see the noise floor — five or more sweeps of a stable system. Before that, publish the numbers without gating on them; a gate calibrated against two data points measures the two data points.

| Situation | Gate on | Why |
|-----------|---------|-----|
| Stable system, several sweeps of history | Tenth percentile, just below baseline | Catches regressions without blocking noise |
| New system, little history | Nothing — publish only | Any threshold is arbitrary |
| A metric that is bimodal | A percentile, never the mean | The mean sits between the two modes |
| A safety-relevant metric | An absolute floor, plus the percentile | Some behaviour is unacceptable at any baseline |
| A metric that has never failed | Review whether it measures anything | A gate that cannot fire is documentation |

The fourth row is the exception to the whole approach. Most thresholds should be relative to observed behaviour, and a small number — an unfounded-answer rate, a refusal-path failure — deserve an absolute floor that does not move when the baseline drifts.

<figure class="diagram">
<svg viewBox="33 9 713 221" role="img" aria-labelledby="srt-noise-t srt-noise-d" xmlns="http://www.w3.org/2000/svg"><title id="srt-noise-t">Setting a gate relative to the observed noise floor</title><desc id="srt-noise-d">Several sweeps of a stable system vary within a band; a gate set just below that band catches a genuine regression while allowing normal variation through.</desc><rect x="33" y="9" width="713" height="221" fill="#ffffff"/><text x="390" y="34" fill="#5b6471" font-size="13" text-anchor="middle">Six sweeps of an unchanged system, then one regression</text><g fill="#e4f5ec" stroke="#12805c" stroke-width="2"><rect x="70" y="70" width="60" height="90" rx="4"/><rect x="150" y="76" width="60" height="84" rx="4"/><rect x="230" y="66" width="60" height="94" rx="4"/><rect x="310" y="74" width="60" height="86" rx="4"/><rect x="390" y="68" width="60" height="92" rx="4"/><rect x="470" y="78" width="60" height="82" rx="4"/></g><g fill="#fdeaee" stroke="#b3324f" stroke-width="2"><rect x="570" y="118" width="60" height="42" rx="4"/></g><rect x="50" y="158" width="600" height="4" rx="2" fill="#c46a3d"/><text x="672" y="166" fill="#c46a3d" font-size="12">gate</text><text x="390" y="212" fill="#1f2937" font-size="13" text-anchor="middle">The gate sits below the noise band and above anything that would be a real regression</text></svg>
<figcaption><b>The band, not the target, sets the gate.</b> A threshold above the observed variation fires on every release; one far below it never fires at all. Both are removed within a quarter.</figcaption>
</figure>

## Implementation

The proposal function reads history, checks that the history is stable enough to derive from, and returns a threshold with the evidence behind it.

```python
import logging
import statistics
from dataclasses import dataclass
from typing import Sequence

log = logging.getLogger("release_thresholds")

MIN_HISTORY = 5


@dataclass(frozen=True)
class Threshold:
    metric: str
    value: float
    baseline: float
    spread: float
    basis: str


def propose(metric: str, history: Sequence[float],
            allowed_regression: float = 0.02,
            absolute_floor: float | None = None) -> Threshold | None:
    """Derive a gate from observed behaviour. Returns None when history is too thin."""
    values = [v for v in history if v is not None]
    if len(values) < MIN_HISTORY:
        log.info("%s: only %d sweep(s) of history — publishing without a gate",
                 metric, len(values))
        return None

    baseline = statistics.median(values)
    spread = statistics.pstdev(values) if len(values) > 1 else 0.0
    # Sit below both the baseline-minus-allowance and the observed noise band.
    candidate = min(baseline - allowed_regression, baseline - 2 * spread)
    if absolute_floor is not None:
        candidate = max(candidate, absolute_floor)
    value = round(max(0.0, candidate), 4)

    basis = (f"median {baseline:.4f} over {len(values)} sweep(s), "
             f"spread {spread:.4f}, allowance {allowed_regression}")
    if absolute_floor is not None and value == absolute_floor:
        basis += f"; raised to the absolute floor {absolute_floor}"
    return Threshold(metric, value, baseline, spread, basis)
```

Taking the minimum of two candidates is what makes the gate robust. An allowance alone ignores how noisy the metric is, so a metric that varies by five points between identical sweeps will trip a two-point allowance constantly; two standard deviations alone can produce an absurdly low gate for a very stable metric. The minimum respects both.

The gate itself needs an override path, and the override needs to be recorded rather than merely possible.

```python
@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: tuple[str, ...]
    overridden: bool
    reason: str


def evaluate(report: dict, thresholds: dict[str, Threshold],
             override_reason: str | None = None) -> GateResult:
    """Apply thresholds to a report. An override is allowed and always recorded."""
    failures = tuple(
        f"{name}: {report[name]['p10']:.4f} below {t.value:.4f}"
        for name, t in thresholds.items()
        if name in report and report[name]["p10"] < t.value
    )
    if not failures:
        return GateResult(True, (), False, "")
    if override_reason:
        log.warning("release gate overridden (%s): %s", override_reason, "; ".join(failures))
        return GateResult(True, failures, True, override_reason)
    log.error("release blocked: %s", "; ".join(failures))
    return GateResult(False, failures, False, "")
```

Allowing the override is not a weakness. A gate with no override will be deleted the first time it blocks an urgent security fix, and its absence afterwards is permanent; an override that requires a reason and appears in the release record keeps the gate alive and turns each bypass into a decision somebody signed.

<figure class="diagram">
<svg viewBox="16 24 678 202" role="img" aria-labelledby="srt-mean-t srt-mean-d" xmlns="http://www.w3.org/2000/svg"><title id="srt-mean-t">Why the gate reads a percentile rather than a mean</title><desc id="srt-mean-d">A bimodal score distribution moves its failing tail while its mean barely changes, so a mean-based gate misses a regression that a tenth-percentile gate catches immediately.</desc><rect x="16" y="24" width="678" height="202" fill="#ffffff"/><text x="30" y="60" fill="#5b6471" font-size="12.5">before</text><rect x="140" y="38" width="90" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="240" y="38" width="440" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="185" y="64" fill="#1f2937" font-size="11.5" text-anchor="middle">10%</text><text x="460" y="64" fill="#1f2937" font-size="11.5" text-anchor="middle">90% near 1.0</text><text x="30" y="146" fill="#5b6471" font-size="12.5">after</text><rect x="140" y="124" width="200" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="350" y="124" width="330" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="240" y="150" fill="#1f2937" font-size="11.5" text-anchor="middle">22%</text><text x="515" y="150" fill="#1f2937" font-size="11.5" text-anchor="middle">78% near 1.0</text><text x="390" y="208" fill="#1f2937" font-size="13" text-anchor="middle">The mean moved by 0.11; the tenth percentile moved from 0.9 to 0.0</text></svg>
<figcaption><b>The failing tail doubled and the mean barely noticed.</b> Spatial scores are bimodal — answers are close or badly wrong — so the statistic that tracks the failing population is the one a gate should read.</figcaption>
</figure>

## Validation & Testing

```python
def test_thin_history_yields_no_gate():
    assert propose("placement", [0.9, 0.91, 0.89]) is None


def test_noisy_metric_gets_a_lower_gate_than_a_stable_one():
    stable = propose("a", [0.90, 0.90, 0.91, 0.90, 0.90])
    noisy = propose("b", [0.90, 0.82, 0.95, 0.86, 0.93])
    assert noisy.value < stable.value


def test_absolute_floor_wins_when_it_is_higher():
    t = propose("unfounded_answers", [0.2] * 6, absolute_floor=0.5)
    assert t.value == 0.5 and "absolute floor" in t.basis


def test_override_is_recorded_not_silent(caplog):
    result = evaluate(FAILING_REPORT, THRESHOLDS, override_reason="hotfix for outage")
    assert result.passed and result.overridden and result.failures
    assert any("overridden" in r.message for r in caplog.records)
```

The fourth test is the one that keeps the override honest. An override implementation that returns `passed=True` and discards the failures is indistinguishable, from the caller's perspective, from a genuine pass — and a release record that cannot show which releases bypassed the gate cannot answer the only question anyone asks about gates afterwards.

Store the derived thresholds as data rather than recomputing them on every run. A gate whose value is recalculated from recent history at evaluation time will drift downward automatically as behaviour degrades, which is the opposite of what a gate is for.

## Gotchas & Edge Cases

**A threshold derived from a period that included a regression.** The baseline absorbs the bad sweeps and the gate is set below the degraded behaviour, permanently. Exclude known-bad periods explicitly, and record which sweeps contributed.

**Gating on a metric with no history because it is new.** New metrics should publish for several sweeps before gating; a gate on a metric nobody has watched will fire on its own noise and be blamed on the release.

**One allowance applied to every metric.** Two points of regression is generous for a stable metric and meaningless for a noisy one. Derive per metric, from that metric's own spread.

<figure class="diagram">
<svg viewBox="56 46 628 194" role="img" aria-labelledby="srt-drift-t srt-drift-d" xmlns="http://www.w3.org/2000/svg"><title id="srt-drift-t">A gate that only ever moves downward</title><desc id="srt-drift-d">Each recalibration after an override lowers the threshold slightly, so over a year the gate falls below anything that could realistically fail and stops protecting anything.</desc><rect x="56" y="46" width="628" height="194" fill="#ffffff"/><g fill="#e4f5ec" stroke="#12805c" stroke-width="2"><rect x="70" y="60" width="80" height="26" rx="4"/></g><g fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"><rect x="200" y="82" width="80" height="26" rx="4"/><rect x="330" y="104" width="80" height="26" rx="4"/></g><g fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"><rect x="460" y="130" width="80" height="26" rx="4"/></g><g fill="#fdeaee" stroke="#b3324f" stroke-width="2"><rect x="590" y="158" width="80" height="26" rx="4"/></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="110" y="106">Q1</text><text x="240" y="128">Q2</text><text x="370" y="150">Q3</text><text x="500" y="176">Q4</text><text x="630" y="204">Q5</text></g><text x="380" y="228" fill="#1f2937" font-size="13" text-anchor="middle">Nothing raised it, so by the fifth quarter it cannot fire</text></svg>
<figcaption><b>A ratchet in one direction is not a gate.</b> Every individual lowering was justified by a specific incident; the cumulative effect was never decided by anyone, which is what makes it worth reviewing on a schedule.</figcaption>
</figure>

**Thresholds that only ever go down.** Each override or recalibration lowers the bar slightly and nothing raises it, so after a year the gate is below anything that could fail. Recalibrate upward too when the baseline improves, and record when you last did.

**A gate on an aggregate rather than a family.** A blended score can pass while one family regresses badly. Gate per family and per region, which is also what makes a failure actionable.

**Overrides that become routine.** A gate overridden every release is not protecting anything; it is generating a ritual. Review the override log periodically and either fix the underlying regression or retire the gate deliberately.

## Frequently Asked Questions

<details class="faq-item"><summary><span>How much history is enough?</span></summary><p>Five sweeps is the practical minimum for a spread estimate that means anything, and ten is comfortable. What matters more than the count is that the sweeps come from a period when nothing changed — five sweeps spanning two model upgrades measure the upgrades, not the noise. If the system changes faster than history accumulates, that is itself worth knowing, and the honest response is to publish without gating until it settles.</p></details>

<details class="faq-item"><summary><span>Should the gate block or warn?</span></summary><p>Block, with the override available. A warning gate is read for two weeks and ignored thereafter, which is worse than no gate because it creates the impression of protection. The override is what makes blocking tolerable, and requiring a written reason is what keeps the override from becoming reflexive.</p></details>

<details class="faq-item"><summary><span>What about metrics where higher is worse?</span></summary><p>Invert them at the source rather than special-casing the threshold logic. A rate of unfounded answers is naturally "lower is better", and carrying two conventions through a gate implementation is exactly the kind of thing that produces a gate wired backwards — which passes every release and is discovered months later.</p></details>

<details class="faq-item"><summary><span>Should thresholds differ per region?</span></summary><p>Yes, where the system genuinely performs differently across regions and both are in scope. A single threshold set from a mixed report is dominated by the region with the most cases, and it will let a regression in a smaller region through. Per-region thresholds are more work to maintain and are the only way the smaller region gets protected at all.</p></details>

<details class="faq-item"><summary><span>Who should own the thresholds?</span></summary><p>Whoever owns the consequence of the metric being wrong, which is rarely the person who wrote the harness. A placement threshold belongs to whoever answers for a wrong footprint; a refusal threshold belongs to whoever answers for an unfounded claim. Naming an owner per threshold sounds bureaucratic and is the thing that keeps a gate from being quietly lowered by whoever is closest to the release.</p></details>

<details class="faq-item"><summary><span>How should a threshold change be recorded?</span></summary><p>As a change to a versioned file, with the evidence in the commit: the sweeps it was derived from, the spread, and why the previous value no longer applies. A threshold that lives in a dashboard configuration can be edited by anyone in seconds and leaves no trace, which is how gates drift downward without anybody deciding to lower them.</p></details>

## Related

- Up to the parent topic: [Evaluation and Benchmarking for Spatial LLMs](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/)
- [Building Regression Test Harnesses for Spatial Agents](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/building-regression-test-harnesses-for-spatial-agents/)
- [Measuring Spatial IoU for LLM-Generated Geometries](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/measuring-spatial-iou-for-llm-generated-geometries/)
- Related technique: [Tuning Fusion Weights for Toponym-Heavy Queries](/geospatial-rag-pipelines/hybrid-spatial-keyword-retrieval/tuning-fusion-weights-for-toponym-heavy-queries/)
