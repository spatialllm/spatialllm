---
title: Validating Intermediate Geoprocessing Outputs
description: Check every intermediate for emptiness, validity, frame and plausible size, so a chain stops at the step that failed rather than at its confident zero.
slug: validating-intermediate-geoprocessing-outputs
type: howto
breadcrumb: Validating Intermediates
datePublished: 2025-03-27
dateModified: 2026-08-11
---

# Validating Intermediate Geoprocessing Outputs

When a geoprocessing chain runs buffer, then clip, then dissolve, a single invalid geometry produced in step one silently poisons every step after it — and the only symptom is a wrong answer with no error. This guide adds assertions *between* steps — geometry validity, CRS agreement, and non-empty, plausibly-sized results — so a broken intermediate halts the chain at its source instead of propagating. It is a companion technique within [LLM-assisted geoprocessing pipelines](/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/).

The failure this prevents is error propagation. A dissolve over a self-intersecting buffer output may not throw; it may return a subtly wrong multipolygon that the next stage happily consumes. By the time a human notices, the provenance is lost. A between-step gate converts a late, mysterious wrong answer into an early, precise failure with the exact step, input, and reason attached.

## When to Use This Approach

Add inter-step validation to any chain of two or more geoprocessing operations, especially when the steps or their arguments came from a model rather than a fixed script. The check is cheap relative to the operations it guards, and the diagnostic value is highest exactly where a model can plan a plausible-but-wrong sequence.

| Check point | Catches | Cost |
|---|---|---|
| Input only | Bad user/model input | Cheapest, misses mid-chain corruption |
| Between every step (this page) | Propagated invalidity, CRS drift, empty results | Low, high diagnostic value |
| Final output only | Nothing about *where* it broke | Cheap, poor provenance |

Validate between every step when correctness matters more than a few milliseconds per stage. If a chain is short and every op is known-safe, an input-plus-output check may suffice. The between-step gate pairs naturally with a typed plan from [decomposing natural language into geoprocessing steps](/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/decomposing-natural-language-into-geoprocessing-steps/), where each step is already named and its expected output known. For CRS handling specifically, lean on [coordinate reference system normalization](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/).

<figure class="diagram">
<svg viewBox="16 38 748 188" role="img" aria-labelledby="vig-two-t vig-two-d" xmlns="http://www.w3.org/2000/svg"><title id="vig-two-t">Checking each step against checking only the answer</title><desc id="vig-two-d">A check after every step names the operation that broke; a check only at the end reports that something in a chain of six is wrong.</desc><rect x="16" y="38" width="748" height="188" fill="#ffffff"/><rect x="30" y="52" width="340" height="160" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="410" y="52" width="340" height="160" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="200" y="84">check every step</text><text x="580" y="84">check only the end</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="200" y="114">the failing step is named</text><text x="200" y="140">cost stops accumulating</text><text x="200" y="166">the fix is local</text><text x="580" y="114">something in the chain failed</text><text x="580" y="140">the whole chain was paid for</text><text x="580" y="166">the fix is a search</text></g></svg>
<figcaption><b>The difference is diagnostic, not just economic.</b> Knowing which of six operations produced an empty result is most of the debugging, and the end-only check throws that information away.</figcaption>
</figure>

## Implementation

The validator below runs after each step. It confirms the result is a non-empty GeoDataFrame, every geometry is valid, the CRS matches the declared working CRS, and the total area sits within an order-of-magnitude sanity band relative to the previous step. Any violation halts the chain and returns a deterministic fallback carrying the last known-good state.

```python
import logging
from dataclasses import dataclass
from typing import Callable, Optional
import geopandas as gpd

log = logging.getLogger("interstep_validator")


class StepViolation(Exception):
    def __init__(self, step: str, reason: str):
        super().__init__(f"[{step}] {reason}")
        self.step, self.reason = step, reason


@dataclass
class GateResult:
    ok: bool
    gdf: Optional[gpd.GeoDataFrame]
    halted_at: Optional[str] = None
    reason: Optional[str] = None


def check_step(
    gdf: gpd.GeoDataFrame,
    step_name: str,
    working_crs: str,
    prev_area: Optional[float],
    area_ratio_limit: float = 100.0,
) -> float:
    """Assert one intermediate output; return its total area for the next comparison."""
    if gdf is None or len(gdf) == 0:
        raise StepViolation(step_name, "empty result set")
    if gdf.crs is None:
        raise StepViolation(step_name, "missing CRS")
    if gdf.crs.to_string() != working_crs:
        raise StepViolation(step_name, f"CRS {gdf.crs.to_string()} != {working_crs}")

    invalid = gdf[~gdf.geometry.is_valid]
    if len(invalid) > 0:
        raise StepViolation(step_name, f"{len(invalid)} invalid geometries")
    if gdf.geometry.is_empty.any():
        raise StepViolation(step_name, "contains empty geometries")

    area = float(gdf.geometry.area.sum())
    if prev_area is not None and prev_area > 0:
        ratio = max(area, 1e-9) / prev_area
        if ratio > area_ratio_limit or ratio < 1.0 / area_ratio_limit:
            raise StepViolation(
                step_name, f"area changed {ratio:.1f}x vs prior step (limit {area_ratio_limit}x)"
            )
    return area


def run_validated_chain(
    seed: gpd.GeoDataFrame,
    steps: list[tuple[str, Callable[[gpd.GeoDataFrame], gpd.GeoDataFrame]]],
    working_crs: str = "EPSG:3857",
) -> GateResult:
    """Execute steps in order, gating each output; halt to last-good state on violation."""
    current = seed
    last_good = seed
    prev_area: Optional[float] = None
    try:
        prev_area = check_step(current, "seed", working_crs, None)
    except StepViolation as v:
        log.error("seed failed validation: %s", v)
        return GateResult(ok=False, gdf=None, halted_at="seed", reason=v.reason)

    for name, fn in steps:
        try:
            current = fn(last_good)
            prev_area = check_step(current, name, working_crs, prev_area)
            last_good = current
        except StepViolation as v:
            log.error("halting chain: %s", v)
            # Deterministic fallback: return the last validated state, not the corrupt one.
            return GateResult(ok=False, gdf=last_good, halted_at=v.step, reason=v.reason)
        except Exception as exc:  # operation itself threw
            log.exception("step '%s' raised", name)
            return GateResult(ok=False, gdf=last_good, halted_at=name, reason=str(exc))

    return GateResult(ok=True, gdf=last_good)
```

The fallback returns `last_good` — the most recent validated GeoDataFrame — so a caller always receives a coherent geometry set plus the exact step and reason that stopped the chain. That payload is what an orchestrator feeds back for a re-plan or surfaces to an operator.

<figure class="diagram">
<svg viewBox="16 32 748 214" role="img" aria-labelledby="vig-what-t vig-what-d" xmlns="http://www.w3.org/2000/svg"><title id="vig-what-t">The four things worth checking after every step</title><desc id="vig-what-d">Emptiness, count plausibility, geometry validity and reference-system agreement catch nearly every failure that would otherwise surface as a wrong answer.</desc><rect x="16" y="32" width="748" height="214" fill="#ffffff"/><rect x="30" y="46" width="360" height="86" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="52" y="76" fill="#1f2937" font-size="13" font-weight="600">is it empty</text><text x="52" y="102" fill="#5b6471" font-size="12">the most common silent failure</text><text x="52" y="122" fill="#5b6471" font-size="12">and a valid result</text><rect x="410" y="46" width="340" height="86" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="432" y="76" fill="#1f2937" font-size="13" font-weight="600">is the count plausible</text><text x="432" y="102" fill="#5b6471" font-size="12">against the input count</text><text x="432" y="122" fill="#5b6471" font-size="12">orders of magnitude matter</text><rect x="30" y="146" width="360" height="86" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="52" y="176" fill="#1f2937" font-size="13" font-weight="600">are the geometries valid</text><text x="52" y="202" fill="#5b6471" font-size="12">self-intersection propagates</text><text x="52" y="222" fill="#5b6471" font-size="12">and gets worse</text><rect x="410" y="146" width="340" height="86" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="432" y="176" fill="#1f2937" font-size="13" font-weight="600">do the systems agree</text><text x="432" y="202" fill="#5b6471" font-size="12">a mismatch is silent</text><text x="432" y="222" fill="#5b6471" font-size="12">and quietly wrong</text></svg>
<figcaption><b>All four are cheap and all four are silent when they fail.</b> That combination is what makes them worth checking every time rather than when something looks wrong.</figcaption>
</figure>

## Validation & Testing

- **Invalid geometry halts immediately.** Insert a step that returns a bowtie (self-intersecting) polygon and assert `run_validated_chain` returns `ok=False` with `halted_at` equal to that step's name.
- **CRS drift is caught.** Have a step reproject to `EPSG:4326` while `working_crs` is `EPSG:3857`; assert the gate halts with a CRS reason and that `gdf` equals the pre-drift `last_good`.
- **Area sanity band.** Feed a step whose output area is 1000x the prior step and assert it halts on the ratio limit, while a 2x change passes — proving the band flags implausible blow-ups without over-triggering.

## Gotchas & Edge Cases

- **Legitimately empty results.** Some queries correctly return nothing (no schools in the flood zone). Treating every empty set as a violation causes false halts; let the caller declare per-step whether empty is a valid terminal state.
- **Area check across mixed geometry types.** Point and line layers have zero area, so the ratio band is meaningless for them. Gate the area comparison on polygonal geometry types and fall back to feature-count sanity for others.
- **Validity in a geographic CRS.** `is_valid` and `area` computed in `EPSG:4326` are misleading — area is in square degrees and validity ignores the sphere. Run these checks in the projected `working_crs`, consistent with [normalizing mixed CRS data before LLM ingestion](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/normalizing-mixed-crs-data-before-llm-ingestion/).
- **Repairing instead of halting.** Silently `make_valid`-ing every intermediate hides the upstream bug that produced the invalidity. Halt and report first; repair only as an explicit, logged policy the caller opts into.

<figure class="diagram">
<svg viewBox="16 38 728 212" role="img" aria-labelledby="vig-band-t vig-band-d" xmlns="http://www.w3.org/2000/svg"><title id="vig-band-t">Plausible against implausible counts</title><desc id="vig-band-d">A step whose output count is wildly out of proportion to its input has usually done something other than what was intended, even when it raises no error.</desc><rect x="16" y="38" width="728" height="212" fill="#ffffff"/><rect x="30" y="52" width="700" height="46" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="50" y="80" fill="#1f2937" font-size="12.5">input 12,400 → output 11,900: plausible</text><rect x="30" y="108" width="620" height="46" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="50" y="136" fill="#1f2937" font-size="12.5">input 12,400 → output 12,400,000: a join went wrong</text><rect x="30" y="164" width="520" height="46" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="50" y="192" fill="#1f2937" font-size="12.5">input 12,400 → output 0: a filter or a projection is wrong</text><text x="390" y="238" fill="#1f2937" font-size="13" text-anchor="middle">Neither extreme raises an error — both produce a confident, wrong answer</text></svg>
<figcaption><b>The count is the cheapest signal available.</b> It costs nothing to record, it catches the two most common pipeline failures, and it needs no knowledge of what the data means.</figcaption>
</figure>

## Operating This Step Over Time

Checks accumulate and slow the pipeline down, one reasonable addition at a time. The way to keep that in hand is to measure what each check costs against how often it has ever fired: a check that has never fired in a year and costs a tenth of the step it guards is a candidate for removal or for sampling rather than a permanent tax.

Thresholds are the other maintenance. A count-plausibility band set against last year's data will fire constantly once the data has grown, and the response is usually to widen it until it stops firing — which removes the check without removing its cost. Deriving the band from the input count rather than from an absolute makes it move with the data.

Record which check caught each failure, not just that a failure occurred. Over a few months that turns into a ranking of which checks earn their place, which is the only real evidence for what to keep.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should a failed check stop the pipeline?</span></summary><p>It should stop that branch and report which check failed on which step. Continuing past a failed validity check means every later step operates on geometry known to be broken, and the eventual error will name the last step rather than the first. Where a plan has independent branches, the others can continue — the point is not to discard work, it is not to build on a known-bad result.</p></details>

<details class="faq-item"><summary><span>Are empty results always failures?</span></summary><p>No, and that is exactly why they need to be flagged rather than raised. "No hospitals within two kilometres" is a real and useful answer; an empty result caused by a projection mismatch looks identical. Recording the emptiness with the step that produced it lets the difference be worked out, whereas silently passing it along guarantees it will not be.</p></details>

<details class="faq-item"><summary><span>How expensive should the checks be?</span></summary><p>Bounded by the step they guard — a check that costs a meaningful fraction of the operation is worth sampling rather than running on every feature. Count, emptiness and reference-system checks are effectively free. Full validity checking on a large result is not, and a sample of a few hundred features catches systematic breakage while missing only the isolated case.</p></details>

<details class="faq-item"><summary><span>Should the model see the check results?</span></summary><p>It should see the classified outcome, not the raw check output. Telling a model that step three produced an empty result lets it consider whether the threshold was wrong; handing it a validation report invites it to reason about details it cannot verify. The summary is what changes the next decision.</p></details>

<details class="faq-item"><summary><span>What should a check record when it passes?</span></summary><p>The values it measured, not the fact that it passed. A row that says the step went from twelve thousand features to eleven thousand nine hundred is useful weeks later when someone asks why an answer changed; a row that says the check passed is useful to nobody. The measurements are also what allow thresholds to be tuned from evidence rather than from the last complaint.</p></details>

## Related

- Up to the section: [LLM-Assisted Geoprocessing Pipelines](/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/)
- [Decomposing Natural Language into Geoprocessing Steps](/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/decomposing-natural-language-into-geoprocessing-steps/)
- [Evaluation and Benchmarking for Spatial LLMs](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/)
- [Building Regression Test Harnesses for Spatial Agents](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/building-regression-test-harnesses-for-spatial-agents/)
