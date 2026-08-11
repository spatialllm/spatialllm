---
title: Managing LLM Context Limits with Quadtree Indexes
description: Use a hierarchical spatial index to decide which features deserve context and to summarise the rest, without letting the index become a substitute for the geometry.
slug: managing-llm-context-limits-with-quadtree-indexes
type: howto
breadcrumb: Quadtree Context Limits
datePublished: 2025-01-29
dateModified: 2026-08-11
---

# Managing LLM Context Limits with Quadtree Indexes

A hierarchical spatial index answers two questions cheaply that a context budget needs constantly: where are the features concentrated, and which of them are near the thing being asked about. Using it to select is excellent; using it to replace the geometry is where teams go wrong. This guide covers both sides, as a selection technique for [context-window optimization for maps](/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/).

## When to Use This Approach

Use a hierarchical index when the view holds more features than the budget can carry and their distribution is uneven — which is almost always true of real map data.

| Situation | Index role | Watch for |
|-----------|------------|-----------|
| Dense urban view, budget-bound | Select and summarise | Cells becoming the answer |
| Sparse rural view | Little benefit; select directly | Overhead for no gain |
| Question about one named feature | Neighbour lookup only | Whole-cell summaries drowning the subject |
| Counting question | Aggregate from the index | Sending features at all |
| Boundary question | Do not use cells | The staircase is not the boundary |

The counting row is the one worth acting on immediately. A question about how many features are in an area should be answered from the index's own counts, exactly, and never by sending a sample of features and hoping the model extrapolates.

<figure class="diagram">
<svg viewBox="16 38 748 178" role="img" aria-labelledby="qt-role-t qt-role-d" xmlns="http://www.w3.org/2000/svg"><title id="qt-role-t">Two legitimate uses of the index and one that is not</title><desc id="qt-role-d">The index selects which features deserve context and supplies exact counts for aggregate questions, but its cells must not be sent in place of geometry for questions about boundaries.</desc><rect x="16" y="38" width="748" height="178" fill="#ffffff"/><rect x="30" y="52" width="230" height="150" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="275" y="52" width="230" height="150" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="520" y="52" width="230" height="150" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="145" y="84">select</text><text x="390" y="84">count</text><text x="635" y="84">substitute</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="145" y="114">which features matter</text><text x="145" y="140">cheap, no geometry sent</text><text x="145" y="168">the intended use</text><text x="390" y="114">exact aggregates</text><text x="390" y="140">answer without features</text><text x="390" y="168">also intended</text><text x="635" y="114">cells instead of shapes</text><text x="635" y="140">a staircase, not a boundary</text><text x="635" y="168">not a use at all</text></g></svg>
<figcaption><b>Two green boxes and one red.</b> The index is a selection and aggregation structure; treating its cells as a cheap encoding of the features inside them replaces the data with a grid at whatever resolution happened to be chosen.</figcaption>
</figure>

## Implementation

The index is built once per view, queried for density and neighbourhood, and used to produce both a selection and an honest summary of what was left out.

```python
import logging
from dataclasses import dataclass, field
from typing import Iterable, Sequence

log = logging.getLogger("spatial_index_budget")

MAX_DEPTH = 12


@dataclass
class Node:
    bounds: tuple[float, float, float, float]
    depth: int
    features: list = field(default_factory=list)
    children: list = field(default_factory=list)

    def count(self) -> int:
        return len(self.features) + sum(c.count() for c in self.children)


def _quadrants(b):
    w, s, e, n = b
    mx, my = (w + e) / 2, (s + n) / 2
    return ((w, s, mx, my), (mx, s, e, my), (w, my, mx, n), (mx, my, e, n))


def build(features: Sequence[dict], bounds, capacity: int = 16, depth: int = 0) -> Node:
    """Build a hierarchical index over feature representative points."""
    node = Node(bounds, depth)
    if len(features) <= capacity or depth >= MAX_DEPTH:
        node.features = list(features)
        if depth >= MAX_DEPTH and len(features) > capacity:
            log.info("depth limit reached with %d coincident features", len(features))
        return node
    buckets = [[] for _ in range(4)]
    for f in features:
        x, y = f["x"], f["y"]
        for i, (w, s, e, n) in enumerate(_quadrants(bounds)):
            if w <= x <= e and s <= y <= n:
                buckets[i].append(f)
                break
        else:                                        # numerically on a boundary
            node.features.append(f)
    node.children = [build(b, q, capacity, depth + 1)
                     for b, q in zip(buckets, _quadrants(bounds)) if b]
    return node
```

The fall-through that keeps boundary features at the parent node matters more than it looks. Floating-point comparisons occasionally place a point in no quadrant, and a build that silently drops those features produces an index whose counts disagree with the data — which then propagates into every aggregate answer.

Selection walks the index outward from the focus, taking whole nodes while the budget allows and summarising the rest.

```python
@dataclass(frozen=True)
class Selection:
    features: tuple
    summarised_nodes: tuple
    total_available: int


def select(root: Node, focus, budget_features: int) -> Selection:
    """Take the nearest features up to the budget; summarise the nodes not taken."""
    ordered = sorted(_walk(root), key=lambda n: _distance(n.bounds, focus))
    taken, summarised, used = [], [], 0
    for node in ordered:
        if used + len(node.features) <= budget_features:
            taken.extend(node.features)
            used += len(node.features)
        elif node.features:
            summarised.append(node)
    return Selection(tuple(taken), tuple(summarised), root.count())


def summarise(nodes: Iterable[Node]) -> str:
    """One honest sentence about everything not shown."""
    nodes = list(nodes)
    if not nodes:
        return ""
    total = sum(len(n.features) for n in nodes)
    return (f"{total} further features are present in the view but not shown, "
            f"spread across {len(nodes)} areas.")
```

Summarising by node rather than by feature is what keeps the summary honest and short. "Four hundred further features" is a number; "four hundred further features across nine areas" tells a reader whether the omission is one dense block or the whole view, which changes how much the shown selection can be trusted to represent it.

<figure class="diagram">
<svg viewBox="46 32 698 218" role="img" aria-labelledby="qt-dense-t qt-dense-d" xmlns="http://www.w3.org/2000/svg"><title id="qt-dense-t">Uneven density and what the index does about it</title><desc id="qt-dense-d">A view with one dense block and a sparse remainder subdivides only where features are concentrated, so the selection can take whole sparse areas and summarise the dense one.</desc><rect x="46" y="32" width="698" height="218" fill="#ffffff"/><rect x="60" y="46" width="300" height="170" rx="6" fill="#ffffff" stroke="#5b6471" stroke-width="2"/><rect x="60" y="46" width="150" height="85" rx="0" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="60" y="46" width="75" height="42" rx="0" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="60" y="46" width="37" height="21" rx="0" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="285" y="180" fill="#1f2937" font-size="12" text-anchor="middle">sparse</text><text x="210" y="238" fill="#5b6471" font-size="12" text-anchor="middle">the index subdivides only where it must</text><rect x="430" y="60" width="300" height="60" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="580" y="86" fill="#1f2937" font-size="12.5" text-anchor="middle">sparse areas: take every feature</text><text x="580" y="106" fill="#5b6471" font-size="12" text-anchor="middle">cheap and complete</text><rect x="430" y="136" width="300" height="60" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="580" y="162" fill="#1f2937" font-size="12.5" text-anchor="middle">dense block: summarise</text><text x="580" y="182" fill="#5b6471" font-size="12" text-anchor="middle">count and extent, not features</text></svg>
<figcaption><b>The structure follows the data.</b> Uniform sampling spends the same effort everywhere; a hierarchical index spends it where the features are, which is what lets a sparse half of the view be included completely for almost nothing.</figcaption>
</figure>

## Validation & Testing

```python
def test_index_count_matches_the_input():
    root = build(FEATURES, BOUNDS)
    assert root.count() == len(FEATURES)


def test_boundary_features_are_not_dropped():
    on_edge = [{"x": 0.0, "y": 0.0, "id": "edge"}] * 40
    root = build(on_edge, (-1.0, -1.0, 1.0, 1.0), capacity=4)
    assert root.count() == 40


def test_selection_never_exceeds_the_budget():
    sel = select(build(FEATURES, BOUNDS), FOCUS, budget_features=25)
    assert len(sel.features) <= 25


def test_summary_reports_the_true_remainder():
    sel = select(build(FEATURES, BOUNDS), FOCUS, budget_features=25)
    assert sel.total_available == len(FEATURES)
    assert str(len(FEATURES) - len(sel.features)) in summarise(sel.summarised_nodes)


def test_depth_limit_terminates_on_coincident_points():
    coincident = [{"x": 1.0, "y": 1.0, "id": str(i)} for i in range(500)]
    root = build(coincident, (0.0, 0.0, 2.0, 2.0), capacity=4)
    assert root.count() == 500
```

The last test is the one that prevents a stack overflow in production. Coincident points — several records at the same address, a batch geocoded to a town centroid — subdivide forever without a depth limit, and the failure arrives as a crash on one unusual view rather than as a gradual slowdown.

The count invariant deserves to run on every build rather than only in tests. It is a single comparison against the input length, it costs nothing, and it is the only check that notices when a change to the partition logic starts losing features at quadrant boundaries.

## Gotchas & Edge Cases

**Cells sent instead of features.** The index's cells are an artefact of the data's distribution, not a description of the world. Send features; use cells to decide which.

**Selection by node when nodes are uneven.** Taking whole nodes is efficient and can overshoot badly when one node holds most of the features. Check the node's size against the remaining budget before taking it, and fall back to per-feature selection inside it.

**Rebuilding the index per request.** Building over a view's features is cheap; building over a whole corpus per request is not. Build per view, or maintain a persistent index and query a subtree.

<figure class="diagram">
<svg viewBox="46 42 666 176" role="img" aria-labelledby="qt-node-t qt-node-d" xmlns="http://www.w3.org/2000/svg"><title id="qt-node-t">Whole-node selection overshooting a budget</title><desc id="qt-node-d">Taking a node whole is efficient until one node holds most of the features, at which point the selection overshoots the budget and must fall back to choosing within the node.</desc><rect x="46" y="42" width="666" height="176" fill="#ffffff"/><rect x="60" y="56" width="90" height="46" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="160" y="56" width="90" height="46" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="260" y="56" width="420" height="46" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="11.5" text-anchor="middle"><text x="105" y="84">8</text><text x="205" y="84">11</text><text x="470" y="84">a node holding 260 features</text></g><rect x="60" y="130" width="360" height="8" rx="4" fill="#c46a3d"/><text x="440" y="140" fill="#c46a3d" font-size="12">budget: 30 features</text><text x="380" y="200" fill="#1f2937" font-size="13" text-anchor="middle">Check the node&#8217;s size against the remaining budget before taking it whole</text></svg>
<figcaption><b>Whole-node selection is an optimisation with a cliff.</b> It is exactly right until the node is large, and the check that prevents the overshoot is one comparison in the selection loop.</figcaption>
</figure>

**Counts drifting from the data.** An index whose count disagrees with the source is worse than no index, because aggregates are answered from it. Assert the invariant on build, as the first test does.

**A depth limit reached silently.** Hitting the limit means the capacity assumption has failed for this data, which is worth a log line — otherwise the first symptom is an oversized leaf node that defeats selection.

**Distance measured to node bounds in degrees.** Ordering nodes by an unprojected distance biases selection at high latitude in the same way it biases reranking. Project, or accept that the ordering is approximate and say so.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Is a quadtree the right structure, or would a grid do?</span></summary><p>A fixed grid is simpler and adequate when density is roughly uniform, which real map data rarely is. The hierarchical structure earns its complexity precisely because it subdivides where features concentrate and leaves sparse regions as single large nodes, which is what makes taking a whole sparse region cheap. If your data genuinely is uniform, a grid will be faster and easier to reason about.</p></details>

<details class="faq-item"><summary><span>What capacity should a node have?</span></summary><p>Around the number of features you would be willing to include or exclude as a unit — sixteen is a reasonable default for a selection use case. Too small and the tree is deep and the selection granular to the point of being per-feature anyway; too large and whole-node decisions become coarse enough to overshoot the budget badly. Tune it against the budget rather than against a benchmark.</p></details>

<details class="faq-item"><summary><span>Can the index answer questions directly?</span></summary><p>Aggregates, yes, and it should. Counts within a region, density, extent of coverage — these are exact from the index and cost nothing, and answering them from a sample of features is both more expensive and wrong. Anything about individual features or their shapes needs the features themselves.</p></details>

<details class="faq-item"><summary><span>How does this interact with the layer allocation?</span></summary><p>It sits inside it. The layer allocator decides how many tokens each layer gets; the index decides which of that layer's features to spend them on. Building one index per layer keeps both decisions clean, and it also means a dense layer's summary describes that layer rather than the view as a whole — which is what a reader needs when the answer turns on one sparse layer.</p></details>

<details class="faq-item"><summary><span>Should the index be built over centroids or over full geometry?</span></summary><p>Over representative points for selection, which is what makes the build cheap, and with the full geometry retained on each feature for when it is chosen. Indexing full geometry means every large feature spans many nodes and the structure stops being a clean partition; indexing points loses nothing that selection needs, since the decision is about which features to include rather than about where their edges run.</p></details>

## Related

- Up to the parent topic: [Context-Window Optimization for Maps](/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/)
- [Budgeting Tokens Across Map Layers](/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/budgeting-tokens-across-map-layers/)
- Related topic: [Geometry Tokenization Strategies](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/)
- Related technique: [Filtering Retrieval by Bounding Box Before Vector Search](/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/filtering-retrieval-by-bounding-box-before-vector-search/)
