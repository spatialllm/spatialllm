---
title: Context-Window Optimization for Maps
description: Fit a map's worth of features into a finite context — budgeting by layer, selecting by relevance, summarising the rest, and degrading in a way the model can report.
slug: context-window-optimization-for-maps
type: topic
breadcrumb: Context-Window Optimization
datePublished: 2025-01-28
dateModified: 2026-08-11
---

# Context-Window Optimization for Maps

A map view holds thousands of features. A context window holds a few dozen, once room has been left for the question, the instructions and the answer. Optimization is the discipline of deciding which features survive that reduction, how the rest are represented, and how the model is told what it is not seeing — because a model that silently receives ten of four hundred buildings will answer as though it saw them all.

This topic belongs to [spatial LLM architecture and core concepts](/spatial-llm-architecture-core-concepts/) and consumes the output of [geometry tokenization strategies](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/), which sets the cost of each feature. It is also the stage that determines whether an answer can be trusted at all: every other control on this site prevents wrong information from entering the context, and this one prevents right information from being silently absent.

<figure class="diagram">
<svg viewBox="26 9 728 224" role="img" aria-labelledby="cwo-budget-t cwo-budget-d" xmlns="http://www.w3.org/2000/svg"><title id="cwo-budget-t">How a context window is actually spent</title><desc id="cwo-budget-d">Instructions, the question, retrieved prose, geometry and the reserved answer space divide one window, leaving far less room for features than the raw window size suggests.</desc><rect x="26" y="9" width="728" height="224" fill="#ffffff"/><text x="390" y="34" fill="#5b6471" font-size="13" text-anchor="middle">A 32k window, once everything else has taken its share</text><rect x="40" y="58" width="110" height="60" rx="6" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="156" y="58" width="70" height="60" rx="6" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="232" y="58" width="230" height="60" rx="6" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="468" y="58" width="160" height="60" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="634" y="58" width="106" height="60" rx="6" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="11.5" text-anchor="middle"><text x="95" y="93">instructions</text><text x="191" y="93">question</text><text x="347" y="93">retrieved prose</text><text x="548" y="93">geometry</text><text x="687" y="93">answer</text></g><g fill="#5b6471" font-size="11.5" text-anchor="middle"><text x="95" y="136">4k</text><text x="191" y="136">1k</text><text x="347" y="136">11k</text><text x="548" y="136">8k</text><text x="687" y="136">6k</text></g><text x="390" y="188" fill="#1f2937" font-size="13" text-anchor="middle">Eight thousand tokens of geometry is roughly ten full-precision features</text><text x="390" y="216" fill="#5b6471" font-size="12" text-anchor="middle">The map view behind the question held four hundred</text></svg>
<figcaption><b>The window is not the budget.</b> Reserving answer space and accounting for instructions first is what turns "we have 32k" into the number that actually governs selection — usually a quarter of the headline figure.</figcaption>
</figure>

## Foundational Principles

**Reserve the answer before spending anything.** A prompt that fills the window leaves the model no room to answer, and the failure is a truncated response rather than an error. Subtract the answer allowance first, then the instructions, then the question; what remains is the real budget.

**Selection is a spatial decision, not a truncation.** Taking the first thirty features from a query result orders them by whatever the database felt like. Selecting the thirty that matter — nearest, largest, most relevant to the question class — is the difference between a useful context and an arbitrary one.

**Omission must be reported.** The model has to know that it is seeing a selection, how large the selection was, and by what rule it was chosen. Without that, every answer implicitly claims completeness, and the ones about counts and extremes will be confidently wrong.

## Step-by-Step Implementation Pipeline

### 1. Compute the real budget

The budget is what remains after every fixed cost is accounted for. Compute it explicitly rather than assuming a fraction, because instruction blocks grow over time and nobody notices until a prompt overflows.

```python
import logging
from dataclasses import dataclass

log = logging.getLogger("context_budget")


@dataclass(frozen=True)
class Budget:
    window: int
    instructions: int
    question: int
    answer_reserve: int

    @property
    def available(self) -> int:
        left = self.window - self.instructions - self.question - self.answer_reserve
        if left <= 0:
            raise ValueError(
                f"no room for content: window {self.window} is consumed by "
                f"instructions {self.instructions}, question {self.question}, "
                f"reserve {self.answer_reserve}")
        return left


def split_budget(available: int, prose_share: float = 0.6) -> tuple[int, int]:
    """Divide the remaining budget between retrieved prose and geometry."""
    prose = int(available * prose_share)
    return prose, available - prose
```

Raising when the fixed costs consume the window is deliberate. That condition means a prompt template has grown past what the model can hold, which is a configuration bug, and discovering it as an exception during a build is much cheaper than discovering it as truncated answers in production.

### 2. Select features by a rule the question implies

Different question classes want different features. A question about what is nearby wants the nearest; a question about the character of an area wants the largest and most typical; a question about a specific named feature wants that one and its immediate neighbours.

```python
from typing import Callable, Sequence


def select_features(features: Sequence[dict], question_class: str, focus,
                    limit: int) -> tuple[list[dict], str]:
    """Return the chosen features and a human-readable statement of the rule."""
    if question_class == "nearby":
        chosen = sorted(features, key=lambda f: f["distance_m"])[:limit]
        return chosen, f"the {len(chosen)} nearest of {len(features)}"
    if question_class == "character":
        chosen = sorted(features, key=lambda f: -f["area_m2"])[:limit]
        return chosen, f"the {len(chosen)} largest of {len(features)}"
    if question_class == "named":
        named = [f for f in features if f.get("matched_name")]
        rest = sorted((f for f in features if not f.get("matched_name")),
                      key=lambda f: f["distance_m"])
        chosen = (named + rest)[:limit]
        return chosen, f"the named feature and its {max(0, len(chosen) - len(named))} nearest neighbours"
    chosen = sorted(features, key=lambda f: f["distance_m"])[:limit]
    log.info("unknown question class %r — defaulting to nearest", question_class)
    return chosen, f"the {len(chosen)} nearest of {len(features)} (default rule)"
```

Returning the rule as a string alongside the features is what makes step 5 possible. A selection whose rule is implicit in the sort order cannot be described to the model, and an undescribed selection is indistinguishable from a complete set.

### 3. Budget per layer, not per feature

A map view is layered — buildings, roads, zoning, hydrology — and a single global limit lets whichever layer happens to be densest crowd out the rest. Allocating per layer, in proportion to how much each matters for the question, keeps the context representative.

```python
def allocate_by_layer(layers: dict[str, int], weights: dict[str, float],
                      geometry_budget: int) -> dict[str, int]:
    """Split the geometry budget across layers, never starving a requested layer."""
    total_weight = sum(weights.get(name, 0.0) for name in layers) or 1.0
    floor = max(1, geometry_budget // (len(layers) * 4))     # every layer gets something
    allocation = {}
    for name in layers:
        share = weights.get(name, 0.0) / total_weight
        allocation[name] = max(floor, int(geometry_budget * share))
    overshoot = sum(allocation.values()) - geometry_budget
    if overshoot > 0:                                        # trim the largest proportionally
        largest = max(allocation, key=lambda n: allocation[n])
        allocation[largest] = max(floor, allocation[largest] - overshoot)
    return allocation
```

The per-layer floor is what stops a zoning layer with two polygons from being squeezed out entirely by a building layer with four hundred. The full treatment of weighting layers against a question is in [budgeting tokens across map layers](/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/budgeting-tokens-across-map-layers/).

<figure class="diagram">
<svg viewBox="16 20 698 202" role="img" aria-labelledby="cwo-layer-t cwo-layer-d" xmlns="http://www.w3.org/2000/svg"><title id="cwo-layer-t">Global limit against per-layer allocation</title><desc id="cwo-layer-d">A single global feature limit lets the densest layer consume the whole budget, while per-layer allocation with a floor keeps every requested layer represented in the context.</desc><rect x="16" y="20" width="698" height="202" fill="#ffffff"/><text x="30" y="58" fill="#b3324f" font-size="13" font-weight="600">global limit</text><rect x="200" y="34" width="500" height="46" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="450" y="62" fill="#1f2937" font-size="12" text-anchor="middle">buildings — 30 of 30 slots</text><text x="30" y="146" fill="#12805c" font-size="13" font-weight="600">per layer</text><rect x="200" y="112" width="230" height="46" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="436" y="112" width="130" height="46" rx="6" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="572" y="112" width="128" height="46" rx="6" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><g fill="#1f2937" font-size="12" text-anchor="middle"><text x="315" y="140">buildings — 14</text><text x="501" y="140">roads — 8</text><text x="636" y="140">zoning — 8</text></g><text x="390" y="204" fill="#1f2937" font-size="13" text-anchor="middle">The zoning layer had two polygons and all of the answer in them</text></svg>
<figcaption><b>Density is not importance.</b> The layer with the most features is usually the least informative per feature, and a global limit is a rule that systematically prefers it.</figcaption>
</figure>

### 4. Summarise what did not fit

Features that were excluded still carry information in aggregate: how many there were, their combined extent, their distribution by type. A short statistical summary costs a few dozen tokens and prevents an entire class of wrong answers about counts and coverage.

```python
from collections import Counter


def summarise_excluded(excluded: Sequence[dict]) -> str:
    """A compact, honest description of what was left out."""
    if not excluded:
        return ""
    kinds = Counter(f.get("kind", "unknown") for f in excluded)
    parts = ", ".join(f"{n} {k}" for k, n in kinds.most_common(4))
    if len(kinds) > 4:
        parts += ", and others"
    areas = [f.get("area_m2", 0.0) for f in excluded]
    total = sum(areas)
    return (f"{len(excluded)} further features not shown ({parts}); "
            f"combined area {total:,.0f} m².")
```

### 5. Tell the model what it is looking at

The selection rule, the counts and the summary go into the context as a header on the feature block. This is the single highest-value fifty tokens in the whole prompt.

```python
def feature_block(features: Sequence[dict], rule: str, excluded_summary: str,
                  render: Callable[[dict], str]) -> str:
    """Features plus an explicit statement of the selection and the omission."""
    lines = [f"Features shown: {rule}."]
    if excluded_summary:
        lines.append(excluded_summary)
    lines.append("")
    lines.extend(render(f) for f in features)
    return "\n".join(lines)
```

### 6. Use a hierarchical index to select, not to represent

Spatial index structures — quadtrees, grids — are excellent for deciding which features matter and poor as a substitute for the features themselves. Use them to narrow, then send real geometry for what survives. The approach and its limits are worked through in [managing context limits with quadtree indexes](/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/managing-llm-context-limits-with-quadtree-indexes/).

```python
def cells_over_budget(cells: dict[str, int], budget: int) -> list[str]:
    """Cells dense enough that their features cannot all be shown."""
    return [cell for cell, count in cells.items() if count > budget // 4]
```

### 7. Degrade the representation before dropping the feature

When a layer's allocation binds, reducing precision or simplifying is almost always better than excluding a feature entirely, because presence carries more information than detail for most questions. The exception is a question specifically about boundary detail, which is why the question class reaches this decision too.

```python
def fit_layer(features, allocation: int, tokenize, question_class: str):
    """Prefer more features at lower detail, unless the question is about detail."""
    detail_matters = question_class in {"boundary", "measure"}
    for rung in (("full",) if detail_matters else ("full", "fewer_decimals", "simplified")):
        rendered = [tokenize(f, rung) for f in features]
        if sum(r.tokens for r in rendered) <= allocation:
            return rendered, rung
    # Still over: keep the most relevant features at the coarsest rung.
    rendered = [tokenize(f, "simplified") for f in features]
    kept, used = [], 0
    for r in rendered:
        if used + r.tokens > allocation:
            break
        kept.append(r)
        used += r.tokens
    return kept, "simplified+truncated_set"
```

### 8. Verify the assembled prompt before sending it

The last step is a measurement, not an estimate. Count the assembled prompt, assert it fits with the reserve intact, and fail loudly if it does not — an over-long prompt is either silently truncated by the provider or rejected, and both are worse than a build-time error.

```python
def assert_fits(prompt: str, budget: Budget, count_tokens) -> None:
    used = count_tokens(prompt)
    ceiling = budget.window - budget.answer_reserve
    if used > ceiling:
        raise ValueError(f"assembled prompt is {used} tokens, ceiling is {ceiling}")
    log.info("prompt assembled: %d tokens, %d reserved for the answer",
             used, budget.answer_reserve)
```

## Operating This Stage Over Time

Context budgets are unusual among the controls on this site in that they get looser rather than tighter as time passes — windows grow, and the pressure that motivated careful selection fades. Three things are worth holding onto anyway.

The first is that a larger window does not remove the need for selection, it moves the failure. Filling a very large window with four hundred features produces answers that are measurably worse than a well-chosen thirty, because the model must now decide which of four hundred to attend to and the weakest ones are the likeliest to contain a confidently phrased irrelevance. Selection remains a quality control after it has stopped being a capacity control.

The second is that instruction blocks grow silently. Every new capability adds a paragraph, and a template that consumed four thousand tokens at launch will consume seven a year later without anyone deciding that. Measuring the fixed costs on every build, and failing when they cross a threshold, is what keeps that growth deliberate.

The third is that the selection rules encode assumptions about the questions, and the questions change. A rule tuned for "what is nearby" serves a user base that has started asking comparative questions badly, and the symptom is not an error but a slow rise in follow-up questions. Logging the selection rule alongside each answer makes it possible to look back and ask whether the rule that fired was the right one, which is not a question anyone can answer from the answers alone.

One practical habit closes most of the gap: log the number of features available, the number shown, and the rule, for every request. Three integers and a string, and they turn every subsequent quality investigation from speculation into a query.

One more habit is worth adopting alongside those three: keep the assembled prompt for a handful of representative requests as a build artefact, and read them occasionally. Token counts tell you whether a prompt fits; only reading one tells you whether the thirty features it selected are the thirty a person would have picked, and that judgement is not available from any metric the pipeline produces.

## Failure Modes & Root Causes

**The confident count.** Asked how many buildings are in an area, the model counts the ones it can see and reports that number. Root cause: omission not reported. Mitigation: the excluded summary in step 4, and a header stating the selection rule.

**The truncated answer.** The prompt fills the window and the response is cut off mid-sentence. Root cause: no answer reserve. Mitigation: subtract the reserve first, and assert the assembled prompt fits.

**The crowded-out layer.** A dense layer consumes the whole allocation and a two-feature layer that held the answer never appears. Root cause: a global feature limit. Mitigation: per-layer allocation with a floor.

**The arbitrary thirty.** Features are taken in whatever order the store returned them, so the selection varies between runs and correlates with nothing. Root cause: truncation used as selection. Mitigation: an explicit rule per question class, recorded and reported.

## Production Validation Protocols

1. **Reserve assertion.** Assert every assembled prompt leaves the full answer reserve; this is the check that catches template growth.
2. **Omission-header test.** Assert that any prompt whose feature set was reduced carries both the selection rule and the excluded summary.
3. **Layer-floor test.** Assert every requested layer contributes at least one feature when it has any, using a fixture with one very dense layer.
4. **Determinism test.** Assemble the same request twice and assert an identical prompt; a difference means the selection depends on store ordering.
5. **Count-question regression.** Keep a fixture where the answer is a count that exceeds the budget, and assert the model is told the true total rather than being left to count.
6. **Fixed-cost indicator.** Publish instruction and template token counts on every build so growth is visible before it binds.

<figure class="diagram">
<svg viewBox="16 32 728 214" role="img" aria-labelledby="cwo-rule-t cwo-rule-d" xmlns="http://www.w3.org/2000/svg"><title id="cwo-rule-t">Selection rules by question class</title><desc id="cwo-rule-d">Four question classes with the feature-selection rule each one implies, from nearest for proximity questions to the named feature plus neighbours for identification questions.</desc><rect x="16" y="32" width="728" height="214" fill="#ffffff"/><rect x="30" y="46" width="340" height="86" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="390" y="46" width="340" height="86" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="30" y="146" width="340" height="86" rx="8" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><rect x="390" y="146" width="340" height="86" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="13" font-weight="600"><text x="52" y="76">&#8220;what is nearby&#8221;</text><text x="412" y="76">&#8220;what is this area like&#8221;</text><text x="52" y="176">&#8220;where is the depot&#8221;</text><text x="412" y="176">&#8220;how many are there&#8221;</text></g><g fill="#5b6471" font-size="12"><text x="52" y="102">nearest first, detail low</text><text x="52" y="122">count matters less than spread</text><text x="412" y="102">largest and most typical</text><text x="412" y="122">a sample, described as one</text><text x="52" y="202">the named feature, then neighbours</text><text x="52" y="222">detail high for the subject</text><text x="412" y="202">send the total, not the features</text><text x="412" y="222">a count is not a geometry question</text></g></svg>
<figcaption><b>The bottom-right case is the one to notice.</b> Counting questions do not want features at all; they want an aggregate computed in the database, and sending thirty of four hundred features is the most expensive possible way to get that answer wrong.</figcaption>
</figure>

## Frequently Asked Questions

<details class="faq-item"><summary><span>How much should be reserved for the answer?</span></summary><p>Enough for the longest answer the system is expected to give, plus a margin — commonly a fifth of the window for conversational use and more where answers include tables or lists. Measuring is better than guessing: take the ninety-fifth percentile of observed answer lengths and add half again. Reserving too little produces truncation, which is highly visible; reserving too much simply means slightly fewer features, which is not.</p></details>

<details class="faq-item"><summary><span>Is it better to send many coarse features or few detailed ones?</span></summary><p>Many coarse ones for most questions, because presence and arrangement carry more of the answer than boundary detail does. The exception is any question that measures — distances, areas, whether a boundary follows a feature — where a simplified geometry produces a confidently wrong number. Letting the question class decide, as the fitting step does, is more reliable than picking one policy for the whole system.</p></details>

<details class="faq-item"><summary><span>Should the omission summary be prose or structured?</span></summary><p>Prose, and short. Structured data in a prompt invites the model to reason over it as though it were features, which is not what a summary is for. One sentence stating the counts, the kinds and the combined extent is read correctly and cited correctly, where a nested object of aggregates tends to reappear in the answer as though it were evidence about individual features.</p></details>

<details class="faq-item"><summary><span>What about questions that genuinely need every feature?</span></summary><p>They should not be answered from the context at all. A question requiring all four hundred buildings is a query, not a reading task, and the right move is to compute the answer in the database and send the result. The signal to watch for is a question class whose selection rule keeps hitting the budget: that is usually a question that wanted an aggregate and was handed a sample.</p></details>

<details class="faq-item"><summary><span>How should the geometry and prose shares be split?</span></summary><p>Start at roughly sixty per cent prose and adjust from measured answer quality rather than from principle. Prose usually carries more of the answer, because documents explain and geometry only locates; the exception is a workload dominated by questions about arrangement and proximity, where the balance tips. Whatever split you choose, make it a parameter rather than a constant scattered through the assembly code, so changing it is one edit and one measurement instead of an afternoon.</p></details>

<details class="faq-item"><summary><span>Does this change with a much larger context window?</span></summary><p>The capacity pressure eases and the quality pressure does not. Attention degrades over long contexts, and a window filled with marginal features measurably worsens answers compared to a well-chosen subset. Treat a larger window as room for better prose and more layers rather than as permission to stop selecting, and keep the omission reporting regardless — it is what makes the answer honest, not what makes it fit.</p></details>

## Related

- Up to the section overview: [Spatial LLM Architecture and Core Concepts](/spatial-llm-architecture-core-concepts/)
- Technique: [Managing Context Limits with Quadtree Indexes](/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/managing-llm-context-limits-with-quadtree-indexes/)
- Technique: [Budgeting Tokens Across Map Layers](/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/budgeting-tokens-across-map-layers/)
- Peer topic: [Geometry Tokenization Strategies](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/)
- Peer topic: [Vector-Raster Hybrid Processing](/spatial-llm-architecture-core-concepts/vector-raster-hybrid-processing/)
- Related topic: [Cost and Latency Budgets for Spatial Agents](/geospatial-prompt-engineering-tool-routing/cost-and-latency-budgets-for-spatial-agents/)
