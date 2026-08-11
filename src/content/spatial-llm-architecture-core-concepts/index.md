---
layout: overview.njk
order: 1
navLabel: Architecture
icon: architecture
title: Spatial LLM Architecture and Core Concepts
description: The representation, resolution, reasoning and evaluation layers a spatial language system needs — and the failure each one prevents when it is built deliberately.
slug: spatial-llm-architecture-core-concepts
type: overview
breadcrumb: Architecture and Core Concepts
datePublished: 2025-01-07
dateModified: 2026-08-11
---

# Spatial LLM Architecture and Core Concepts

A language model reasoning about places fails differently from one reasoning about text. It does not produce nonsense; it produces fluent, confident answers that are displaced by a datum shift, measured in square degrees, or attached to a town with the same name in another country. Every one of those failures is silent — the output is well-formed, the pipeline reports success, and nothing raises. This section is the set of layers that make those failures impossible rather than unlikely.

The layers are not optional extras on top of a working system; they are what makes the system work. A geometry must arrive in a declared frame before it can be tokenized, be tokenized before it can fit a context window, be positioned by a lookup rather than by recall before any relation about it can be checked, and be measured against a case set before any of it can be claimed to have improved. What follows is each layer, the specific failure it prevents, and the standard it has to meet in production.

<figure class="diagram">
<svg viewBox="0 0 860 330" role="img" aria-labelledby="slm-arch-t slm-arch-d" xmlns="http://www.w3.org/2000/svg"><title id="slm-arch-t">The layers of a spatial language system</title><desc id="slm-arch-d">Ingestion normalises frames and repairs geometry, representation tokenizes and budgets it, resolution turns names into records, reasoning verifies relations, and evaluation gates the whole thing before release.</desc><rect x="0" y="0" width="860" height="330" fill="#ffffff"/><text x="430" y="34" fill="#5b6471" font-size="13" text-anchor="middle">Each layer depends on the one before it and prevents one specific silent failure</text><g fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"><rect x="24" y="52" width="190" height="80" rx="8"/><rect x="234" y="52" width="190" height="80" rx="8"/><rect x="444" y="52" width="190" height="80" rx="8"/><rect x="654" y="52" width="182" height="80" rx="8"/></g><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="119" y="82">normalize</text><text x="329" y="82">represent</text><text x="539" y="82">resolve</text><text x="745" y="82">reason</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="119" y="106">one declared frame</text><text x="329" y="106">tokens and budgets</text><text x="539" y="106">names to records</text><text x="745" y="106">relations, checked</text><text x="119" y="124">no silent assumption</text><text x="329" y="124">no truncated ring</text><text x="539" y="124">no recalled coordinate</text><text x="745" y="124">no composed claim</text></g><g stroke="#5b6471" stroke-width="2" marker-end="url(#slm-arch-a)"><line x1="216" y1="92" x2="230" y2="92"/><line x1="426" y1="92" x2="440" y2="92"/><line x1="636" y1="92" x2="650" y2="92"/></g><defs><marker id="slm-arch-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/></marker></defs><rect x="24" y="162" width="812" height="66" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="430" y="190" fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600">evaluation — measures every layer, gates every release</text><text x="430" y="214" fill="#5b6471" font-size="12" text-anchor="middle">placement, plausibility, relations and refusal, each on its own axis</text><rect x="24" y="252" width="812" height="60" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="430" y="278" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">fallback routing — what happens when any of it is unavailable</text><text x="430" y="300" fill="#5b6471" font-size="12" text-anchor="middle">a smaller claim, stated, rather than a silent approximation</text></svg>
<figcaption><b>Two layers run underneath rather than alongside.</b> Evaluation measures all four upper layers and gates releases on them; fallback routing decides what each one does when its dependencies are missing. Neither is a stage in the pipeline, and both are what make the pipeline trustworthy.</figcaption>
</figure>

## Normalizing the Reference Frame Before Anything Else

Two datasets describing the same street can disagree by two hundred metres and both be correct, because they were recorded against different realisations of the ground. The ingestion gate resolves that before anything downstream can be misled: every geometry arrives in one declared frame, validated, with its transformation recorded.

```python
def normalize(geom, declared_epsg, target_epsg=4326):
    """One entry point; either a fully described result or an explicit rejection."""
    source = require_frame(declared_epsg)          # refuses; never assumes
    geom, repaired = repair(geom)                  # deterministic, and recorded
    if source.is_geographic and axis_order_suspect(geom, source):
        raise CRSRejected("coordinates appear axis-swapped; correct the source")
    transformer, note = build_transformer(source, CRS.from_epsg(target_epsg))
    return Normalized(shapely_transform(transformer.transform, geom),
                      declared_epsg, target_epsg, note, repaired)
```

The refusal is the load-bearing part. Treating undeclared coordinates as geographic because they look like degrees is the most expensive habit in spatial software: rejection produces an error someone fixes in an hour, while assumption produces answers wrong by a datum shift for as long as the system runs. The full gate, including per-source overrides for the sources that declare wrongly, is in [coordinate reference system normalization](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/).

Storing in one frame does not mean measuring in it. Areas need an equal-area projection and distances need one centred on what is being measured, chosen per operation rather than baked into storage — the split argued in [choosing a canonical frame for spatial LLM pipelines](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/choosing-a-canonical-crs-for-llm-pipelines/). Calling an area accessor on geographic coordinates returns square degrees, a unit whose relationship to square metres varies with latitude, and it is the single most common measurement error in spatial code.

## Turning Geometry Into Tokens Without Losing It

A model reads tokens; a polygon is a list of high-precision numbers. How that conversion is done determines both what fraction of the context one feature consumes and how much of its meaning survives.

```python
def tokenize_within_budget(geom, budget, count_tokens, decimals=5):
    """Reduce along a defined ladder. Never truncates the token stream."""
    for rung, text in (("full", render(geom, decimals)),
                       ("fewer_decimals", render(geom, decimals - 2)),
                       ("simplified", render(simplify_safely(geom, 25.0), decimals - 2))):
        if count_tokens(text) <= budget:
            return Tokenized(text, rung)
    return Tokenized(extent_of(geom), "extent_only")     # a stated loss, not a cut
```

Three decisions do most of the work. Precision is a policy chosen from the questions rather than inherited from the exporter — five decimal places is roughly a metre, and coordinates arrive with twelve. Representation is chosen once for the corpus, because the compact, structured and cell forms differ by a factor of six in cost and completely in what they preserve. And the budget degrades along a ladder rather than truncating, because a cut coordinate list is not a coarser geometry but a parse error. See [geometry tokenization strategies](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/) and, for the arithmetic, [coordinate precision versus token cost](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/coordinate-precision-versus-token-cost/).

Vertex reduction deserves particular care because it is the most effective saving and the easiest way to produce a self-intersecting ring or a sliver between two parcels that used to share a boundary. Simplify with a topology-preserving algorithm, in a metric frame, and validate afterwards.

## Fitting a Map Into a Context Window

A map view holds thousands of features and a context window holds a few dozen once the answer allowance and the instructions are subtracted. What survives that reduction, and whether the model is told what did not, decides whether an answer about counts or coverage is trustworthy.

```python
def feature_block(features, rule, excluded_summary, render):
    """Features plus an explicit statement of the selection and the omission."""
    lines = [f"Features shown: {rule}."]
    if excluded_summary:
        lines.append(excluded_summary)
    lines.extend(render(f) for f in features)
    return "\n".join(lines)
```

That header is the highest-value fifty tokens in the prompt. Without it every answer implicitly claims completeness, and the ones about counts and extremes will be confidently wrong. Selection also has to be a spatial decision rather than a truncation — the thirty nearest, or the thirty largest, chosen by a rule the question implies and reported alongside the results. And the budget is allocated per layer rather than globally, because the densest layer is almost never the one holding the answer: see [context-window optimization for maps](/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/) and [budgeting tokens across map layers](/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/budgeting-tokens-across-map-layers/).

A larger window moves this problem rather than removing it. Attention degrades over long contexts, so a window filled with marginal features measurably worsens answers compared with a well-chosen subset; selection remains a quality control after it has stopped being a capacity control.

<figure class="diagram">
<svg viewBox="16 32 768 228" role="img" aria-labelledby="slm-fail-t slm-fail-d" xmlns="http://www.w3.org/2000/svg"><title id="slm-fail-t">Six silent failures and the layer that prevents each</title><desc id="slm-fail-d">Assumed frames, truncated geometry, unreported omission, recalled coordinates, composed relation claims and unmeasured regressions each produce fluent wrong answers, and each is prevented by one specific layer.</desc><rect x="16" y="32" width="768" height="228" fill="#ffffff"/><rect x="30" y="46" width="360" height="60" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="410" y="46" width="360" height="60" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="30" y="116" width="360" height="60" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="410" y="116" width="360" height="60" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="30" y="186" width="360" height="60" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="410" y="186" width="360" height="60" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="12.5" font-weight="600"><text x="50" y="70">assumed frame</text><text x="430" y="70">truncated geometry</text><text x="50" y="140">unreported omission</text><text x="430" y="140">recalled coordinate</text><text x="50" y="210">composed relation</text><text x="430" y="210">unmeasured regression</text></g><g fill="#5b6471" font-size="12"><text x="50" y="92">prevented by the ingestion gate</text><text x="430" y="92">prevented by the reduction ladder</text><text x="50" y="162">prevented by the selection header</text><text x="430" y="162">prevented by gazetteer grounding</text><text x="50" y="232">prevented by predicate verification</text><text x="430" y="232">prevented by the case set</text></g></svg>
<figcaption><b>None of these raises an exception.</b> That is what unites them and why the controls are structural: each produces a well-formed answer, and the only thing distinguishing it from a correct one is a check that ran before it was composed.</figcaption>
</figure>

## Turning Names Into Positions You Can Defend

Every spatial question that arrives in words has to become a position before anything can be computed. That conversion is where a system either acquires provenance or loses it, because a coordinate the model recalled looks identical to one that was looked up.

```python
def to_context(res: Resolution) -> dict:
    """The shape every downstream stage should receive — never a bare coordinate pair."""
    if res.record is None:
        return {"resolved": False, "confidence": res.confidence,
                "alternatives": [a.name for a in res.alternatives], "reason": res.rationale}
    return {"resolved": True, "place_id": res.record.place_id, "bbox": res.record.bbox,
            "confidence": res.confidence,
            "alternatives": [a.name for a in res.alternatives]}
```

Two rules make this work. A model may propose a name and never a coordinate — extraction is a language task, location is a lookup. And ambiguity is the normal case rather than an exception, so a resolver that always returns one result has not simplified anything, it has moved the decision somewhere invisible. Where candidates are within scoring noise of each other, the honest output is no selection plus the alternatives, which lets the agent ask a four-word question instead of being wrong half the time. The mechanics are in [geocoding and place-name resolution](/spatial-llm-architecture-core-concepts/geocoding-and-place-name-resolution/) and [disambiguating duplicate toponyms with spatial context](/spatial-llm-architecture-core-concepts/geocoding-and-place-name-resolution/disambiguating-duplicate-toponyms-with-spatial-context/).

A resolution also carries an extent rather than a point. "Where is this county" has an answer thousands of square kilometres across, and collapsing it to a centroid produces distance calculations wrong by tens of kilometres in a way nothing detects.

## Verifying What the Model Says About Space

"The depot is north of the river and inside the enterprise zone" is two claims and a conjunction, and a model will produce sentences like that from context supporting neither. Verification decomposes them and checks each against real geometry.

```python
def check_topological(a, b, kind) -> tuple[bool | None, str]:
    """Return (verdict, note). None means unverifiable — never a guess."""
    if a is None or b is None:
        return None, "one or both geometries unavailable"
    if kind == "contains":
        return b.contains(a), "exact predicate"
    if kind == "adjacent":
        return a.touches(b) or a.distance(b) < TOLERANCE, "within the stated tolerance"
    return None, f"no exact predicate for {kind!r}"
```

The three-valued verdict is the whole design. "Not shown to be true" and "shown to be false" mean different things to a reader, and collapsing them makes an agent sound authoritative about the limits of its own data. Direction needs a stated convention rather than a predicate — a bearing sector, applied identically from every entry point — and distance needs a metric frame and a defined endpoint, because centroid and edge distances differ by more than a factor of two for ordinary shapes. See [spatial reasoning and relation inference](/spatial-llm-architecture-core-concepts/spatial-reasoning-and-relation-inference/).

Some relations cannot be computed from the geometry most corpora hold. "Upstream of" needs a flow network; "overlooks" needs terrain and a viewshed. A registry of supported relations, with everything outside it routed to a refusal that names what would be needed, is more useful than a distance check wearing the wrong label.

## Combining Gridded and Vector Data

A model cannot read a raster; it can read a statement derived from one. Producing that statement correctly means clipping to the shape rather than its bounding box, sampling at a resolution the question can support, and rounding to a precision the sample justifies.

```python
def to_statement(name, proportions, labels, grid, coverage) -> str:
    """One sentence a model can quote, with everything a reader needs to judge it."""
    top = ", ".join(f"{labels.get(c, c)} {p:g}%" for c, p in list(proportions.items())[:4])
    date = f", captured {grid.captured_year}" if grid.captured_year else ""
    cover = "" if coverage > 0.98 else f", from {coverage * 100:.0f}% of the shape"
    return f"{name}: {top} (from {grid.pixel_size_m:g} m data{date}{cover})."
```

Three of those four components are routinely dropped, and the figure alone is the part that means least without them. A thirty-metre grid cannot describe a single building, and the honest response to a shape smaller than a few cells is a refusal naming the cell count rather than a percentage. The pipeline is set out in [vector-raster hybrid processing](/spatial-llm-architecture-core-concepts/vector-raster-hybrid-processing/), with the reading mechanics in [aligning raster tiles with vector masks](/spatial-llm-architecture-core-concepts/vector-raster-hybrid-processing/aligning-raster-tiles-with-vector-masks-for-llm-context/).

## Choosing Embeddings for a Geometry-Bearing Corpus

Retrieval over these corpora is ordinary technical-prose retrieval with an unusual metadata layer, and the most consequential decision is one number: dimensionality, which sets the memory footprint of every index and every replica.

```python
def embedding_text(chunk) -> str:
    """What gets embedded. Coordinates deliberately excluded."""
    head = [p for p in (chunk.name, chunk.feature_type,
                        ", ".join(chunk.parents[:2])) if p]
    return f"{' — '.join(head)}\n{chunk.body}" if head else chunk.body
```

Coordinates do not belong in the vector. Embedding a number places it near other numbers of similar magnitude, which has nothing to do with proximity on the ground, and it consumes dimensions meant to carry meaning. Position belongs in a spatial index where it can be filtered exactly. Dimensionality should be the smallest that holds recall on your own corpus under a real region filter — usually far below the largest available, since the curve flattens early on technical prose. See [spatial embedding models](/spatial-llm-architecture-core-concepts/spatial-embedding-models/) and [choosing vector dimensionality for spatial retrieval](/spatial-llm-architecture-core-concepts/spatial-embedding-models/choosing-vector-dimensionality-for-spatial-retrieval/).

## Degrading Deliberately When Something Is Unavailable

A spatial query has more ways to fail than a text one, and each failure leaves the agent holding a question it can still partly answer. Fallback routing is the design of that partial answer.

```python
def route(request, ladder, budget_s) -> Outcome:
    """Walk the ladder inside one shared deadline; each rung states its claim."""
    started = time.monotonic()
    for rung in ladder:
        left = budget_s - (time.monotonic() - started)
        if left <= 0 or left < rung.typical_s * 0.5:
            continue                                   # admission: do not start what cannot finish
        try:
            return Outcome(rung.run(request, left), rung.name, rung.claim)
        except Degradable:
            continue
    return Outcome(None, "refusal", "none")
```

Every fallback is a smaller claim, and the claim has to reach the answer text — a degraded result that reads exactly like an exact one has converted a precision loss into a correctness claim. The deadline is shared rather than per attempt, so three failing rungs cost one budget rather than three. And some intents do not degrade at all: a boundary-membership question answered from a simplified geometry is a coin flip presented as a fact. See [fallback routing for geospatial queries](/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/) and [deadline propagation and timeout budgets](/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/deadline-propagation-and-timeout-budgets/).

## Measuring Whether Any of It Improved

General language benchmarks say nothing about whether a system puts things in the right place. Spatial evaluation measures four independent families, and reports them separately because they are fixed by different work.

```python
def summarise(results) -> dict:
    """Break down by family and region; no aggregate is produced."""
    report = {}
    for (family, region), rows in group(results):
        scores = sorted(r["score"] for r in rows)
        report[f"{family}/{region}"] = {
            "n": len(rows),
            "median": scores[len(scores) // 2],
            "p10": scores[max(0, int(0.1 * len(scores)) - 1)],
            "parse_rate": round(sum(r["parsed"] for r in rows) / len(rows), 4)}
    return report
```

Three properties make this honest. Parse failures are results rather than exclusions, since dropping unreadable output flatters the model exactly where it is weakest. Refusal is scored as a success on cases that cannot be answered, or the highest-scoring system is the one that never declines. And the gate reads a percentile rather than a mean, because spatial scores are bimodal — answers are close or badly wrong — and the mean sits between the two modes describing neither. See [evaluation and benchmarking for spatial LLMs](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/).

<figure class="diagram">
<svg viewBox="16 32 748 214" role="img" aria-labelledby="slm-dep-t slm-dep-d" xmlns="http://www.w3.org/2000/svg"><title id="slm-dep-t">What each layer assumes about the one before it</title><desc id="slm-dep-d">Tokenization assumes a known frame, context assembly assumes parseable geometry, relation checking assumes resolved positions, and evaluation assumes all of them are recorded.</desc><rect x="16" y="32" width="748" height="214" fill="#ffffff"/><rect x="30" y="46" width="360" height="86" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="410" y="46" width="340" height="86" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="30" y="146" width="360" height="86" rx="8" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><rect x="410" y="146" width="340" height="86" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="13" font-weight="600"><text x="52" y="76">tokenization assumes</text><text x="432" y="76">context assembly assumes</text><text x="52" y="176">relation checking assumes</text><text x="432" y="176">evaluation assumes</text></g><g fill="#5b6471" font-size="12"><text x="52" y="102">a known frame — otherwise precision</text><text x="52" y="122">means nothing</text><text x="432" y="102">parseable geometry — otherwise the</text><text x="432" y="122">budget protects noise</text><text x="52" y="202">resolved positions — otherwise the</text><text x="52" y="222">predicate runs on the wrong shape</text><text x="432" y="202">everything recorded — otherwise a</text><text x="432" y="222">score change cannot be attributed</text></g></svg>
<figcaption><b>Skip a layer and the next one still runs.</b> That is the difficulty: each layer's assumption is invisible when violated, so a system missing the ingestion gate produces tokenized geometry, assembled context and verified relations — all computed correctly over coordinates that mean something else.</figcaption>
</figure>

## What Changes When the System Is Built This Way

Three things are noticeably different about a spatial system with these layers in place, and none of them is speed.

The first is that failures become visible. A pipeline without an ingestion gate does not fail — it produces answers that are wrong by a datum shift, and the first sign is a user noticing that a distance does not match their map, months later, with no record of how the geometry got there. With the gate, the same input produces a rejection on the day it arrives, attributed to a source, with a reason someone can act on. The total number of problems is identical; what changes is when they surface and whether they are attributable.

The second is that answers acquire limits. A system that resolves places through a gazetteer, verifies relations against predicates and reports what it omitted will decline questions it cannot support — and users find that far easier to work with than confident answers of unknown reliability. The refusals are not a cost of the architecture; they are its most visible product, because they are the cases where a system without it would have guessed.

The third is that changes become measurable. A prompt edit, a chunking change and a library upgrade all move the same numbers, and without a case set and a recorded environment there is no way to say which. Teams without evaluation do not stop changing things; they stop being able to tell whether the changes helped, which over a year produces a system nobody is willing to modify.

None of this requires the full apparatus on day one. The gate and the no-coordinate rule can be a hundred lines; the case set can be three frozen cases in a script. What matters is that the shape is right early, because every one of these layers is cheaper to add before the corpus exists than after.

## Where the Layers Meet Their Limits

It is worth being explicit about what this architecture does not do, since each limit is a place where teams reasonably reach for something else.

It does not make a model good at geometry. Verification catches false relation claims and does not help a model produce better ones; a system whose relation claims are usually wrong will spend most of its time hedging. The fix there is upstream — better retrieved context, precomputed facts in the prompt — rather than a stricter checker.

It does not answer questions the data cannot support. A thirty-metre raster will not describe a building, a gazetteer without small features will not resolve a field name, and no amount of engineering converts absent data into a defensible answer. The layers make that boundary explicit, which is genuinely valuable and is frequently mistaken for a shortcoming of the system rather than of its inputs.

It does not remove the need for judgement about thresholds. Every band, tolerance and weight described here is a policy: the confidence at which a measured claim is permitted, the tolerance at which two parcels count as adjacent, the cell count below which a proportion is refused. Those numbers are decisions, they belong to the people who answer for the answers, and writing them down is the whole of the discipline.

## Production Engineering Standards

1. **No geometry enters without a declared frame.** Undeclared coordinates are rejected into a queue a person reads, never assumed to be geographic. Per-source defaults are permitted, recorded with their evidence, and reviewed.
2. **Every transformation is recorded.** Source code, target code and the transformation path travel with the geometry, because datum transformations are not unique and two environments can disagree by a metre.
3. **Measurement never happens in the storage frame.** Areas use an equal-area projection, distances a projection centred on the subject, and both record which one produced the number.
4. **Geometry is reduced along a ladder, never truncated.** Precision first, then vertices, then the extent — each rung reported to the consumer so a simplified boundary is never mistaken for the real one.
5. **Every prompt states its selection rule and its omission.** The features shown, the rule that chose them, and a summary of what was left out, or answers about counts and coverage cannot be trusted.
6. **No coordinate reaches the geometry layer without a place identifier.** Positions come from lookups with provenance and confidence; a model-produced coordinate is screened before anything acts on it.
7. **Relations are computed or marked unverifiable.** Three-valued verdicts throughout, with the convention and tolerance stated, and a falsified load-bearing claim stops the turn rather than being hedged.
8. **Every answer that degraded says so.** The fallback rung's claim reaches the answer text, and boundary-membership questions refuse rather than approximate.
9. **Releases are gated on a percentile, per family and per region.** Parse failures score as results, refusals score as successes where refusal was correct, and the gate's overrides are recorded with a reason.
10. **The harness proves it did not move before the agent is blamed.** Library versions, tokenizer and case-set digest are captured with every sweep, and a self-check on frozen cases runs first.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Which layer should a team build first?</span></summary><p>The ingestion gate, without close competition. Every other layer computes over geometry, and geometry whose frame is unknown makes all of that computation meaningless in a way no later check can recover. It is also the cheapest to add early and the most painful to retrofit, because retrofitting means re-normalising a corpus whose provenance was never recorded.</p></details>

<details class="faq-item"><summary><span>Is a specialised spatial model needed for any of this?</span></summary><p>Almost never. The work here is engineering around a general model rather than replacing it: resolution is a lookup, relations are predicates, measurement is a projection, and the model's job is language. Where specialised models genuinely help is in tasks where the shape itself is the subject — matching footprints, clustering trajectories — which is a different problem from answering questions about places.</p></details>

<details class="faq-item"><summary><span>How do these layers relate to retrieval?</span></summary><p>They sit underneath it. <a href="/geospatial-rag-pipelines/">Geospatial RAG pipelines</a> assembles the context a model reasons over, and it depends on the representation and frame decisions made here — a chunk cannot carry a defensible extent unless the geometry inside it was normalised first. The boundary is worth stating plainly: this section decides what a geometry means, retrieval decides which ones the model sees.</p></details>

<details class="faq-item"><summary><span>What does an agent do with all this?</span></summary><p>It routes. Once context is assembled and positions are resolved, the remaining decisions are which tool to call, how to recover when a call fails, and what to spend — the subject of <a href="/geospatial-prompt-engineering-tool-routing/">geospatial prompt engineering and tool routing</a>. Confusing the two produces a system that retries a search when it should have called a geometry engine.</p></details>

<details class="faq-item"><summary><span>How much of this is needed for a small internal tool?</span></summary><p>The ingestion gate and the no-coordinate rule, at minimum, because those two prevent the failures that are hardest to detect and most damaging when they reach a decision. Evaluation can start as three frozen cases in a script. Fallback routing can be one cached rung. The layers scale down considerably; what does not scale down is the discipline of refusing to assume a frame and refusing to accept a recalled coordinate.</p></details>

## Related

- Section: [Geospatial RAG Pipelines](/geospatial-rag-pipelines/) — assembling context from geometry-bearing corpora
- Section: [Geospatial Prompt Engineering and Tool Routing](/geospatial-prompt-engineering-tool-routing/) — what an agent does with that context
- Topic: [Coordinate Reference System Normalization](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/)
- Topic: [Geometry Tokenization Strategies](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/)
- Topic: [Context-Window Optimization for Maps](/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/)
- Topic: [Geocoding and Place-Name Resolution](/spatial-llm-architecture-core-concepts/geocoding-and-place-name-resolution/)
- Topic: [Spatial Reasoning and Relation Inference](/spatial-llm-architecture-core-concepts/spatial-reasoning-and-relation-inference/)
- Topic: [Vector-Raster Hybrid Processing](/spatial-llm-architecture-core-concepts/vector-raster-hybrid-processing/)
- Topic: [Spatial Embedding Models](/spatial-llm-architecture-core-concepts/spatial-embedding-models/)
- Topic: [Fallback Routing for Geospatial Queries](/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/)
- Topic: [Evaluation and Benchmarking for Spatial LLMs](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/)
