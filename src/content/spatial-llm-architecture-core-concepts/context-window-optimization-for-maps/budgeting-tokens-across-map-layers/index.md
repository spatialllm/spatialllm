---
title: Budgeting Tokens Across Map Layers
description: Allocate a geometry budget between layers by relevance rather than density, floor every requested layer, and report what each layer contributed and what it omitted.
slug: budgeting-tokens-across-map-layers
type: howto
breadcrumb: Budgeting Across Layers
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Budgeting Tokens Across Map Layers

A map view has layers, and they have wildly different densities: four hundred buildings, thirty road segments, two zoning polygons. A single feature limit hands the whole budget to the densest layer, which is almost never the one holding the answer. This guide allocates the budget deliberately, as the layer-aware half of [context-window optimization for maps](/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/).

## When to Use This Approach

Allocate per layer whenever more than one layer is in scope. For a single-layer view a global limit is equivalent and simpler.

| Question | Layer weighting | Reason |
|----------|-----------------|--------|
| "What is at this address" | Buildings high, zoning low | The answer is a specific feature |
| "Can I build here" | Zoning and constraints high | Buildings are context, not answer |
| "What is this area like" | Even, favouring variety | Character comes from the mix |
| "Where does the boundary run" | One layer, high detail | Everything else is a distraction |
| Unclassified | Even, with floors | Fails least badly |

The fourth row is worth separating out because it inverts the usual advice. When a question is about one layer's geometry, spending budget on other layers actively harms the answer by consuming detail the subject needed.

<figure class="diagram">
<svg viewBox="26 9 672 239" role="img" aria-labelledby="btl-dense-t btl-dense-d" xmlns="http://www.w3.org/2000/svg"><title id="btl-dense-t">Density against relevance across three layers</title><desc id="btl-dense-d">The densest layer contributes the most features under a global limit while contributing the least to the answer, and the two-feature zoning layer that holds the answer is crowded out entirely.</desc><rect x="26" y="9" width="672" height="239" fill="#ffffff"/><text x="390" y="34" fill="#5b6471" font-size="13" text-anchor="middle">One view, three layers, and where the answer actually is</text><rect x="200" y="56" width="470" height="42" rx="5" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="200" y="108" width="120" height="42" rx="5" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="200" y="160" width="40" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="12.5"><text x="40" y="82">buildings</text><text x="40" y="134">roads</text><text x="40" y="186">zoning</text></g><g fill="#1f2937" font-size="12" text-anchor="middle"><text x="435" y="82">400 features</text><text x="260" y="134">30</text><text x="220" y="186">2</text></g><text x="330" y="186" fill="#12805c" font-size="12">the answer is in here</text><text x="390" y="230" fill="#1f2937" font-size="13" text-anchor="middle">A global limit of thirty features shows thirty buildings and nothing else</text></svg>
<figcaption><b>Density and relevance are unrelated.</b> The layer with the most features usually has the least to say per feature, and a limit expressed in features rather than in layers systematically prefers it.</figcaption>
</figure>

## Implementation

The allocator distributes the geometry budget by weight, guarantees every requested layer a floor, and returns the allocation so it can be logged and explained.

```python
import logging
from dataclasses import dataclass
from typing import Mapping

log = logging.getLogger("layer_budget")

FLOOR_DIVISOR = 4              # every layer gets at least budget / (layers * 4)


@dataclass(frozen=True)
class Allocation:
    per_layer: dict[str, int]
    floor: int
    note: str


def allocate(available: Mapping[str, int], weights: Mapping[str, float],
             geometry_budget: int) -> Allocation:
    """Split a geometry budget across layers by weight, with a floor for each."""
    layers = [name for name, count in available.items() if count > 0]
    if not layers:
        return Allocation({}, 0, "no layer has any feature in view")
    if geometry_budget <= 0:
        raise ValueError("geometry budget must be positive")

    floor = max(1, geometry_budget // (len(layers) * FLOOR_DIVISOR))
    total_weight = sum(max(0.0, weights.get(name, 0.0)) for name in layers)
    if total_weight <= 0:
        even = geometry_budget // len(layers)
        log.info("no usable weights supplied — allocating evenly")
        return Allocation({name: even for name in layers}, floor,
                          "even allocation: no weights supplied")

    raw = {name: max(floor, int(geometry_budget * max(0.0, weights.get(name, 0.0)) / total_weight))
           for name in layers}

    # Floors can push the total over budget; trim proportionally from the layers above floor.
    overshoot = sum(raw.values()) - geometry_budget
    while overshoot > 0:
        trimmable = [n for n in layers if raw[n] > floor]
        if not trimmable:
            break                                  # every layer is at its floor: accept and log
        for name in sorted(trimmable, key=lambda n: -raw[n]):
            if overshoot <= 0:
                break
            take = min(raw[name] - floor, max(1, overshoot // len(trimmable)))
            raw[name] -= take
            overshoot -= take

    note = "" if overshoot <= 0 else f"floors exceed the budget by {overshoot} tokens"
    if note:
        log.info("layer floors exceed the geometry budget by %d", overshoot)
    return Allocation(raw, floor, note)
```

The floor is the load-bearing part. Without it a weighting that reflects the question — buildings at 0.7, zoning at 0.05 — gives the zoning layer a handful of tokens and effectively excludes it, which is precisely the failure the per-layer approach exists to prevent. With it, a low-weighted layer still contributes one or two features, and one zoning polygon is frequently the whole answer.

Accepting an overshoot when every layer sits at its floor is also deliberate. That condition means the budget is too small for the number of layers requested, and the honest response is to report it rather than to silently drop a layer.

<figure class="diagram">
<svg viewBox="16 24 734 206" role="img" aria-labelledby="btl-alloc-t btl-alloc-d" xmlns="http://www.w3.org/2000/svg"><title id="btl-alloc-t">Weighted allocation with and without a floor</title><desc id="btl-alloc-d">Pure weighting starves a low-weighted layer to nothing, while the same weighting with a floor leaves each layer a small guaranteed share that is often enough to carry the answer.</desc><rect x="16" y="24" width="734" height="206" fill="#ffffff"/><text x="30" y="62" fill="#b3324f" font-size="13" font-weight="600">weights only</text><rect x="200" y="38" width="440" height="42" rx="5" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="646" y="38" width="90" height="42" rx="5" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><g fill="#1f2937" font-size="12" text-anchor="middle"><text x="420" y="64">buildings</text><text x="691" y="64">roads</text></g><text x="30" y="152" fill="#12805c" font-size="13" font-weight="600">with a floor</text><rect x="200" y="128" width="330" height="42" rx="5" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="536" y="128" width="110" height="42" rx="5" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="652" y="128" width="84" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="12" text-anchor="middle"><text x="365" y="154">buildings</text><text x="591" y="154">roads</text><text x="694" y="154">zoning</text></g><text x="390" y="212" fill="#1f2937" font-size="13" text-anchor="middle">The zoning layer had two polygons and one of them was the answer</text></svg>
<figcaption><b>A small guaranteed share is worth a large weighted one.</b> Layers with few features tend to have high information per feature, so the floor recovers most of what pure weighting throws away for a small cost to the dominant layer.</figcaption>
</figure>

## Validation & Testing

```python
def test_every_layer_with_features_gets_at_least_the_floor():
    a = allocate({"buildings": 400, "roads": 30, "zoning": 2},
                 {"buildings": 0.7, "roads": 0.25, "zoning": 0.05}, 8000)
    assert all(v >= a.floor for v in a.per_layer.values())


def test_empty_layers_are_excluded_from_the_split():
    a = allocate({"buildings": 400, "hydrology": 0}, {"buildings": 1.0}, 4000)
    assert "hydrology" not in a.per_layer


def test_total_does_not_exceed_the_budget_when_it_can_be_met():
    a = allocate({"a": 10, "b": 10, "c": 10}, {"a": 0.8, "b": 0.15, "c": 0.05}, 3000)
    assert sum(a.per_layer.values()) <= 3000


def test_impossible_budget_is_reported_not_hidden():
    a = allocate({f"l{i}": 5 for i in range(12)}, {}, 24)
    assert a.note and "exceed" in a.note


def test_missing_weights_fall_back_to_even():
    a = allocate({"a": 5, "b": 5}, {}, 1000)
    assert a.per_layer["a"] == a.per_layer["b"]
```

The fourth test is the one worth keeping through refactors. It asserts that an over-constrained allocation is visible rather than resolved by quietly dropping layers, which is what every straightforward implementation does when the arithmetic stops working.

Run these against the real weight table rather than fixtures of it. The weights are configuration that changes as question classes are added, and a test that supplies its own weights keeps passing long after the production table has acquired a class with no entry — which silently falls through to an even split.

## Gotchas & Edge Cases

**Weights derived from layer density.** Weighting by how many features a layer has re-creates the problem the allocator exists to solve. Weights come from the question class, never from the data.

**A floor set too high.** With a dozen layers and a small budget, floors alone can consume everything and the weighting stops mattering. Scale the floor with the layer count, as above, and report when it binds.

**Layers with features that are individually enormous.** A single national boundary can exceed a whole layer's allocation. Reduce within the layer first — the tokenizer's reduction ladder handles this — and only then treat the layer as over budget.

<figure class="diagram">
<svg viewBox="16 24 710 206" role="img" aria-labelledby="btl-unspent-t btl-unspent-d" xmlns="http://www.w3.org/2000/svg"><title id="btl-unspent-t">Redistributing an allocation a layer cannot spend</title><desc id="btl-unspent-d">A layer with two small polygons leaves most of its allocation unused; a second pass returns the surplus to the layers that can absorb it, recovering budget that would otherwise be wasted.</desc><rect x="16" y="24" width="710" height="206" fill="#ffffff"/><text x="30" y="62" fill="#5b6471" font-size="12.5">first pass</text><rect x="170" y="38" width="260" height="42" rx="5" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="436" y="38" width="130" height="42" rx="5" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="572" y="38" width="140" height="42" rx="5" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="11.5" text-anchor="middle"><text x="300" y="64">buildings</text><text x="501" y="64">roads</text><text x="642" y="64">zoning: unspent</text></g><text x="30" y="152" fill="#12805c" font-size="12.5">redistributed</text><rect x="170" y="128" width="350" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="526" y="128" width="150" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="682" y="128" width="30" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="11.5" text-anchor="middle"><text x="345" y="154">buildings</text><text x="601" y="154">roads</text></g><text x="380" y="212" fill="#1f2937" font-size="13" text-anchor="middle">The surplus goes to the layers that can use it, in weight order</text></svg>
<figcaption><b>Unspent budget is a real loss.</b> A layer with two polygons cannot absorb a large share, and leaving those tokens idle costs exactly as much as allocating them badly would have.</figcaption>
</figure>

**Allocation computed before the question is classified.** The weights depend on the question class, so an allocator called too early in the pipeline gets defaults every time. Classify first, allocate second.

**Empty layers counted in the split.** A layer with no features in view should not receive a floor; giving it one takes tokens from layers that could use them. Filter before allocating, as the implementation does.

**A weight table with no entry for a class.** The lookup falls through to an even split, which is a reasonable default and is not what anyone intended for that class. Log the fall-through and treat a recurring one as a missing table row.

**Allocation not reported.** The per-layer split is the single most useful line in a prompt-assembly log, because it explains at a glance why an answer talked about buildings and not zoning. Log it with the question class beside it.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Where should layer weights come from?</span></summary><p>A small table keyed on question class, hand-written and reviewed rather than learned. There are rarely more than half a dozen classes worth distinguishing, the weights are easy to reason about, and a table is inspectable in a way a fitted model is not. Learn them only if you have a labelled set large enough to fit them and the discipline to refit when the question mix changes.</p></details>

<details class="faq-item"><summary><span>Should the allocation be in tokens or in features?</span></summary><p>Tokens, because features vary enormously in cost — a building footprint and a coastline differ by two orders of magnitude. Allocating features gives a layer of large geometries an unbounded share of the budget while appearing fair. Convert to features inside each layer, using measured token costs, once the token allocation is known.</p></details>

<details class="faq-item"><summary><span>What happens when a layer cannot use its allocation?</span></summary><p>Return the surplus and redistribute. A layer with two small polygons will not spend a large allocation, and leaving the tokens unused wastes budget that the dense layers would happily absorb. A second pass that redistributes unspent allocation in weight order is a few lines and recovers a useful fraction on most requests.</p></details>

<details class="faq-item"><summary><span>Does this interact with retrieved prose?</span></summary><p>Only through the split between prose and geometry made one level up. Once that split is fixed, this allocator divides the geometry half and never touches the prose. Keeping the two decisions separate is what makes both explainable: one answers "how much of the window is geometry", the other answers "which layers get it".</p></details>

<details class="faq-item"><summary><span>How should a layer that was omitted entirely be reported?</span></summary><p>By name, in the prompt header, alongside the count it would have contributed. A model told that a hydrology layer exists in the view but was not included can say so when a question turns out to be about water; one that simply never saw the layer will answer as though the view contained no water features at all. This costs a handful of tokens and removes a whole class of confidently incomplete answers.</p></details>

<details class="faq-item"><summary><span>Is it worth allocating differently for follow-up questions?</span></summary><p>Yes, and it is one of the clearest wins available. A follow-up narrows the subject — the user has asked about zoning specifically — so the weighting for that turn should shift decisively toward the layer they asked about. Treating every turn as an independent request with the same weights wastes most of the budget re-establishing context the conversation already has.</p></details>

## Related

- Up to the parent topic: [Context-Window Optimization for Maps](/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/)
- [Managing Context Limits with Quadtree Indexes](/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/managing-llm-context-limits-with-quadtree-indexes/)
- Related topic: [Geometry Tokenization Strategies](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/)
- Related topic: [Cost and Latency Budgets for Spatial Agents](/geospatial-prompt-engineering-tool-routing/cost-and-latency-budgets-for-spatial-agents/)
