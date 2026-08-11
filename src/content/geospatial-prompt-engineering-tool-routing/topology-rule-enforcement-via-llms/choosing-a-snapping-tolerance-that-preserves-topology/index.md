---
title: Choosing a Snapping Tolerance That Preserves Topology
description: Derive a snapping tolerance from source precision and the smallest feature that must survive, then verify it against feature counts rather than tuning until warnings stop.
slug: choosing-a-snapping-tolerance-that-preserves-topology
type: howto
breadcrumb: Snapping Tolerance
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Choosing a Snapping Tolerance That Preserves Topology

A snapping tolerance is a claim about how far apart two things can be and still be the same thing. Chosen too small it leaves the gaps it was meant to close; chosen too large it deletes features and merges neighbours that were genuinely distinct. Both failures are silent, which is why the value deserves derivation rather than adjustment. This guide covers deriving it, and it is the quantitative half of [topology rule enforcement via LLMs](/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/).

## When to Use This Approach

A tolerance is needed wherever geometry from different sources has to line up, and it is needed nowhere else.

| Situation | Tolerance | Note |
|-----------|-----------|------|
| One source, consistent survey | None | Snapping can only cause harm |
| Two sources, same nominal precision | From source precision | Small — close rounding gaps |
| Two sources, different precisions | From the coarser one | Bounded by the finest real feature |
| Generated geometry meeting real data | From the target data | The generated side has no precision |

<figure class="diagram">
<svg viewBox="16 38 735 220" role="img" aria-labelledby="cst-band-t cst-band-d" xmlns="http://www.w3.org/2000/svg"><title id="cst-band-t">The window a usable tolerance sits in</title><desc id="cst-band-d">A tolerance must be larger than the positional noise in the source data and smaller than the narrowest feature that has to survive, and the gap between those two bounds is usually wide.</desc><rect x="16" y="38" width="735" height="220" fill="#ffffff"/><rect x="30" y="52" width="240" height="46" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="50" y="80" fill="#1f2937" font-size="12.5">below the noise: gaps stay open</text><rect x="30" y="108" width="560" height="46" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="50" y="136" fill="#1f2937" font-size="12.5">the usable window: above the noise, below the narrowest real feature</text><rect x="30" y="164" width="700" height="46" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="50" y="192" fill="#1f2937" font-size="12.5">above the narrowest feature: real geometry is deleted, silently</text><text x="390" y="240" fill="#1f2937" font-size="13" text-anchor="middle">The window is usually wide — which is why a derived value works and a tuned one drifts</text></svg>
<figcaption><b>The choice is not delicate, and it is not arbitrary either.</b> Both bounds come from properties of the data you can measure, and anything inside them behaves acceptably; the trouble comes entirely from values chosen without reference to either.</figcaption>
</figure>

## Implementation

Both bounds are measurable. The lower one comes from the coordinate precision actually present in the source — not what the metadata claims, which is frequently optimistic.

```python
def observed_precision(gdf, sample=5000) -> float:
    """Smallest consistent coordinate increment actually present in the data."""
    coords = np.vstack([np.asarray(g.exterior.coords) for g in gdf.geometry.head(sample)])
    deltas = np.diff(np.unique(np.round(coords[:, 0], 9)))
    deltas = deltas[deltas > 0]
    return float(np.percentile(deltas, 1)) if len(deltas) else 0.0
```

The upper bound comes from the narrowest feature that must survive, which is a question about the data's purpose rather than its numbers — but it can be approximated well from the distribution of feature widths.

```python
def narrowest_meaningful(gdf, quantile=0.01) -> float:
    """Approximate width of the narrowest features worth keeping."""
    widths = gdf.geometry.apply(lambda g: 2 * g.area / g.length if g.length else 0.0)
    return float(widths[widths > 0].quantile(quantile))


def derive_tolerance(gdf) -> float:
    low = max(observed_precision(gdf) * 2, 1e-6)
    high = narrowest_meaningful(gdf) / 4
    if high <= low:
        raise ValueError("no safe tolerance: source noise exceeds the finest real feature")
    return math.sqrt(low * high)          # geometric mean, comfortably inside the window
```

That exception matters. When the noise in a source exceeds the size of its smallest real features, no tolerance is safe, and the correct response is to refuse rather than to pick something in the middle and hope. It is a genuine data problem and snapping will not fix it.

<figure class="diagram">
<svg viewBox="16 38 748 212" role="img" aria-labelledby="cst-two-t cst-two-d" xmlns="http://www.w3.org/2000/svg"><title id="cst-two-t">Deriving the tolerance against tuning it</title><desc id="cst-two-d">A derived tolerance moves with the data and can be justified, while one tuned until warnings stop tends toward whatever value is quietest rather than whatever value is correct.</desc><rect x="16" y="38" width="748" height="212" fill="#ffffff"/><rect x="30" y="52" width="340" height="160" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="410" y="52" width="340" height="160" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="200" y="84">derived from the data</text><text x="580" y="84">tuned until it is quiet</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="200" y="114">moves when the data does</text><text x="200" y="140">the bounds are recorded</text><text x="200" y="166">a bad value raises an error</text><text x="580" y="114">fixed at whatever silenced it</text><text x="580" y="140">no record of why</text><text x="580" y="166">drifts upward over time</text></g><text x="390" y="234" fill="#1f2937" font-size="13" text-anchor="middle">Warnings get quieter as the tolerance grows — which is exactly the wrong incentive</text></svg>
<figcaption><b>Tuning has a built-in direction.</b> Every increase removes complaints and every decrease adds them, so a value adjusted by feedback alone climbs until it is deleting things nobody has noticed yet.</figcaption>
</figure>

## Validation & Testing

Verify by counting, before and after. A snapping operation that changed the feature count changed the data, and whether that was intended is a question somebody should answer explicitly.

```python
def verify_snap(before, after, tolerance) -> dict:
    report = {
        "tolerance": tolerance,
        "features_before": len(before),
        "features_after": len(after),
        "features_lost": len(before) - len(after),
        "area_delta_ratio": abs(after.area.sum() - before.area.sum()) / before.area.sum(),
        "still_invalid": int((~after.is_valid).sum()),
    }
    assert report["features_lost"] == 0, report
    assert report["area_delta_ratio"] < 1e-4, report
    return report
```

Run that over a representative extract at several candidate tolerances. The derived value should sit comfortably inside the range where nothing is lost, rather than at its edge — if it sits at the edge, one of the two bounds was measured on unrepresentative data.

## Gotchas & Edge Cases

**Tolerance in the wrong units.** A value derived in a projected system and applied to geographic coordinates is wrong by roughly five orders of magnitude, and it will either do nothing or destroy everything. The tolerance and the reference system belong to each other and should be stored together.

**A single tolerance across mixed data.** Building footprints and coastlines have entirely different noise floors and feature sizes. One value that suits both is usually a value that suits neither, and per-layer derivation costs a function call.

**Snapping before validity.** Snapping invalid geometry produces differently invalid geometry, sometimes spectacularly. Validate first, repair what is unambiguous, and snap after — the order is not interchangeable.

**Cascading movement.** Each snap moves vertices, which can bring other vertices within tolerance of each other. A single pass is predictable; iterating to a fixed point is not, and can walk a boundary a surprising distance from where it started.

**Losing the record.** A dataset that has been snapped and does not say so, at what tolerance, is one whose boundaries cannot be reconciled with the source. The tolerance belongs in the output metadata, not only in the code that applied it.

<figure class="diagram">
<svg viewBox="10 52 750 194" role="img" aria-labelledby="cst-order-t cst-order-d" xmlns="http://www.w3.org/2000/svg"><title id="cst-order-t">The order the operations have to run in</title><desc id="cst-order-d">Validate first, repair unambiguous problems, then snap at the derived tolerance, then verify counts and areas — each stage depending on the one before it.</desc><rect x="10" y="52" width="750" height="194" fill="#ffffff"/><defs><marker id="cst-order-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/></marker></defs><rect x="26" y="66" width="160" height="118" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="212" y="66" width="160" height="118" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="398" y="66" width="160" height="118" rx="8" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><rect x="584" y="66" width="160" height="118" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="106" y="100">validate</text><text x="292" y="100">repair</text><text x="478" y="100">snap</text><text x="664" y="100">verify</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="106" y="130">find what is broken</text><text x="106" y="154">before moving anything</text><text x="292" y="130">unambiguous fixes only</text><text x="292" y="154">closures, duplicates</text><text x="478" y="130">at the derived value</text><text x="478" y="154">one pass, recorded</text><text x="664" y="130">counts and areas</text><text x="664" y="154">nothing lost or merged</text></g><g stroke="#5b6471" stroke-width="2" marker-end="url(#cst-order-a)"><line x1="188" y1="125" x2="208" y2="125"/><line x1="374" y1="125" x2="394" y2="125"/><line x1="560" y1="125" x2="580" y2="125"/></g><text x="390" y="228" fill="#1f2937" font-size="13" text-anchor="middle">Snapping first makes the validation report describe geometry that no longer exists</text></svg>
<figcaption><b>Verification is the stage most often skipped.</b> It is also the only one that would have caught the tolerance being wrong, since every other stage completes successfully at any value you give it.</figcaption>
</figure>

## What to Record Alongside the Result

A snapped dataset should carry four facts: the tolerance applied, the reference system it was measured in, the counts before and after, and the version of the source it was derived from. Together they make the operation reproducible and its effect auditable, and separately none of them does.

The reference system is the field most often omitted and the one whose absence causes the worst confusion, because a tolerance without units is a number that looks meaningful and cannot be checked. The counts are what let somebody later confirm that nothing was lost without re-running the whole operation. And the source version is what explains a tolerance that no longer suits the data — the derivation was correct for the input it saw, and the input has moved.

Recording the derivation inputs as well as the result is worth the extra two fields. When a tolerance later looks wrong, the useful question is whether the noise floor or the feature-size distribution changed, and that is answerable in seconds with the recorded bounds and not at all without them.

## Operating This Step Over Time

Re-derive whenever a source is updated rather than carrying the value forward. A resurvey changes the noise floor, a new supplier changes it more, and neither raises anything — the tolerance simply stops matching the data it was computed for.

Track features lost per run as a first-class number. It should be zero; a run where it is not is either a data problem or a tolerance problem, and both are worth stopping for. A slow rise across runs is the specific signature of a tolerance that is now too large for data that has become finer.

The derivation code itself needs occasional attention, because both measurements make assumptions about the data that stop holding quietly. The precision estimate reads a sample of coordinates and will mislead if the sample is drawn from a subset with different characteristics — an extract sorted by identifier often is. The feature-width approximation treats every polygon as roughly rectangular, which is fine for parcels and poor for river corridors. Neither assumption fails loudly; both are worth re-checking when a new kind of data arrives rather than when a result looks strange.

It is also worth keeping one worked example per dataset — the measured bounds, the derived value, and the verification report — somewhere a person can read. When a tolerance is questioned months later, that record answers the question in a minute, and its absence turns the same question into an afternoon of re-deriving numbers that were already known.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Can the model choose the tolerance?</span></summary><p>No, and it should not be asked to. The value depends on measurable properties of specific datasets that a model has no access to, so its answer is a plausible-sounding number with no relationship to the data. Where the model helps is upstream: describing what the user considers a meaningful feature, which is one of the two bounds and the only one that is a judgement rather than a measurement.</p></details>

<details class="faq-item"><summary><span>What if two layers need different tolerances but must align?</span></summary><p>Use the smaller of the two, and expect some gaps to remain. Snapping the finer layer at the coarser layer's tolerance destroys detail that the finer survey exists to provide, and the residual gaps are honest evidence that two datasets disagree. Recording the remaining gaps is more useful than closing them with a value that damages one side.</p></details>

<details class="faq-item"><summary><span>Should snapping happen at ingest or at query time?</span></summary><p>At ingest, almost always. It is expensive, it is deterministic, and doing it once means every later query sees consistent geometry rather than each one paying for its own alignment. The exception is a comparison between datasets that are not normally used together, where the alignment is specific to that question and does not belong in either source.</p></details>

<details class="faq-item"><summary><span>How does this relate to simplification?</span></summary><p>They are different operations with similar-looking parameters, and confusing them is common. Snapping moves vertices to close gaps between features; simplification removes vertices to reduce detail within one. A simplification tolerance applied as a snapping tolerance will usually be far too large, because the two are answering different questions about the same data.</p></details>

## Related

- Up to the parent topic: [Topology Rule Enforcement via LLMs](/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/)
- [Snapping and Noding LLM-Generated Geometries](/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/snapping-and-noding-llm-generated-geometries/)
- [Enforcing Topological Rules in LLM-Generated Geometries](/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/enforcing-topological-rules-in-llm-generated-geometries/)
- Related topic: [LLM-Assisted Geoprocessing Pipelines](/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/)
