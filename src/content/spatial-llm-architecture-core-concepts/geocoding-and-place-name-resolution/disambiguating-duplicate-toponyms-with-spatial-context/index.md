---
title: Disambiguating Duplicate Toponyms with Spatial Context
description: Choose between places that share a name using administrative context, conversation extent and prominence — with a margin test that refuses rather than guesses.
slug: disambiguating-duplicate-toponyms-with-spatial-context
type: howto
breadcrumb: Disambiguating Toponyms
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Disambiguating Duplicate Toponyms with Spatial Context

Duplicate place names are not an edge case. Any gazetteer covering more than one region will return several records for a substantial share of names, and a system that picks the most prominent every time will consistently answer about a city when the user meant a village. This guide implements the choice, and the refusal, as the decision stage of [geocoding and place-name resolution](/spatial-llm-architecture-core-concepts/geocoding-and-place-name-resolution/).

## When to Use This Approach

Run disambiguation whenever grounding returns more than one candidate. Skip it when the name is unique or when the user has already selected a record in this conversation.

| Signal available | Strength | Applies when |
|------------------|----------|--------------|
| A containing administrative name in the question | Strongest | "Newport in Shropshire" |
| Consistency with the conversation's area | Strong | Follow-up questions |
| A feature type implied by the question | Moderate | "the river Avon" |
| Gazetteer prominence | Weakest | Nothing else is available |
| Nothing | — | Refuse and ask |

The ordering is what makes this work. Prominence is a property of the world rather than of the question, so it can break a tie between otherwise equal candidates and must never outweigh something the user actually said.

<figure class="diagram">
<svg viewBox="26 32 728 218" role="img" aria-labelledby="ddt-order-t ddt-order-d" xmlns="http://www.w3.org/2000/svg"><title id="ddt-order-t">Disambiguation signals in priority order</title><desc id="ddt-order-d">An administrative name in the question outweighs conversational context, which outweighs an implied feature type, which outweighs gazetteer prominence — the weakest and the one most systems rely on.</desc><rect x="26" y="32" width="728" height="218" fill="#ffffff"/><rect x="40" y="46" width="700" height="42" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="40" y="98" width="540" height="42" rx="6" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="40" y="150" width="380" height="42" rx="6" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><rect x="40" y="202" width="200" height="42" rx="6" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="12.5"><text x="60" y="73">an administrative name appears in the question</text><text x="60" y="125">consistent with the area already discussed</text><text x="60" y="177">the question implies a feature type</text><text x="60" y="229">gazetteer prominence</text></g><g fill="#5b6471" font-size="12"><text x="600" y="125">weaker</text><text x="440" y="177">weaker still</text><text x="260" y="229">a tiebreaker only</text></g></svg>
<figcaption><b>The shortest bar is the one most systems use alone.</b> Prominence produces a defensible answer for the common case and a systematically wrong one for every question about a small place — which is most questions about small places.</figcaption>
</figure>

## Implementation

The scorer combines the signals, applies a margin test, and returns either a choice or an unresolved result carrying the alternatives.

```python
import logging
from dataclasses import dataclass
from typing import Optional, Sequence

log = logging.getLogger("toponym_disambiguation")

MARGIN = 0.12                      # below this gap, the candidates are indistinguishable


@dataclass(frozen=True)
class Choice:
    record: Optional[object]
    confidence: float
    alternatives: tuple
    rationale: str


def _overlaps(a, b) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def disambiguate(
    candidates: Sequence,
    context_names: frozenset[str],
    session_bbox: Optional[tuple] = None,
    implied_type: Optional[str] = None,
) -> Choice:
    """Pick a candidate or refuse. Never returns a choice it cannot justify."""
    if not candidates:
        return Choice(None, 0.0, (), "no candidates supplied")
    if len(candidates) == 1:
        return Choice(candidates[0], 0.8, (), "only one candidate")

    scored = []
    for c in candidates:
        score, reasons = 0.15 + 0.20 * getattr(c, "importance", 0.0), []
        parents = {p.lower() for p in getattr(c, "parents", ())}
        if context_names & parents:
            score += 0.45
            reasons.append("a containing region is named in the question")
        if session_bbox is not None and _overlaps(c.bbox, session_bbox):
            score += 0.25
            reasons.append("consistent with the area under discussion")
        if implied_type and getattr(c, "feature_type", None) == implied_type:
            score += 0.15
            reasons.append(f"matches the implied type {implied_type!r}")
        scored.append((round(min(1.0, score), 3), c, "; ".join(reasons)))

    # Deterministic ordering: score, then identifier, so ties never depend on input order.
    scored.sort(key=lambda t: (-t[0], getattr(t[1], "place_id", "")))
    best, runner_up = scored[0], scored[1]

    if best[0] - runner_up[0] < MARGIN:
        log.info("toponym ambiguous: %.3f against %.3f", best[0], runner_up[0])
        return Choice(None, best[0], tuple(c for _, c, _ in scored[:4]),
                      "candidates are within scoring noise of each other")

    return Choice(best[1], best[0], tuple(c for _, c, _ in scored[1:4]),
                  best[2] or "selected on prominence alone")
```

The margin test is the whole design. Without it the function always returns a candidate, and its error rate equals the frequency with which the runner-up was correct — which for prominence-only decisions on small places is close to half. With it, ambiguous cases become a question the agent can ask, and a four-word clarification from the user resolves what no amount of scoring could.

Note also that "selected on prominence alone" appears in the rationale when no other signal fired. That string is what lets a reviewer see, in a log, that a decision was made on the weakest available evidence even though it cleared the margin.

<figure class="diagram">
<svg viewBox="16 24 714 206" role="img" aria-labelledby="ddt-margin-t ddt-margin-d" xmlns="http://www.w3.org/2000/svg"><title id="ddt-margin-t">Two score distributions and what the margin test does with each</title><desc id="ddt-margin-d">A clear winner separated from the runner-up by more than the margin is chosen; two candidates within the margin are returned unresolved so the agent can ask which was meant.</desc><rect x="16" y="24" width="714" height="206" fill="#ffffff"/><text x="30" y="62" fill="#12805c" font-size="13" font-weight="600">decided</text><rect x="180" y="38" width="330" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="560" y="38" width="130" height="42" rx="5" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><g fill="#1f2937" font-size="12" text-anchor="middle"><text x="345" y="64">best 0.80</text><text x="625" y="64">next 0.35</text></g><text x="30" y="152" fill="#c46a3d" font-size="13" font-weight="600">ambiguous</text><rect x="180" y="128" width="260" height="42" rx="5" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="450" y="128" width="240" height="42" rx="5" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="12" text-anchor="middle"><text x="310" y="154">best 0.42</text><text x="570" y="154">next 0.38</text></g><text x="390" y="212" fill="#1f2937" font-size="13" text-anchor="middle">A four-hundredths gap is noise — asking costs one turn, guessing costs the answer</text></svg>
<figcaption><b>The lower row is where most damage happens.</b> A scorer with no margin returns the left-hand candidate with apparent confidence, and nothing in the output distinguishes it from the clear case above.</figcaption>
</figure>

## Validation & Testing

```python
def test_administrative_context_beats_prominence():
    big = Rec("big", importance=0.9, parents=("Gwent",), bbox=B1)
    small = Rec("small", importance=0.1, parents=("Shropshire",), bbox=B2)
    choice = disambiguate([big, small], context_names=frozenset({"shropshire"}))
    assert choice.record is small


def test_near_tie_refuses():
    a = Rec("a", importance=0.50, parents=(), bbox=B1)
    b = Rec("b", importance=0.48, parents=(), bbox=B2)
    choice = disambiguate([a, b], context_names=frozenset())
    assert choice.record is None and len(choice.alternatives) == 2


def test_session_context_resolves_a_follow_up():
    a = Rec("a", importance=0.5, parents=(), bbox=(-3.3, 55.8, -3.0, 56.1))
    b = Rec("b", importance=0.5, parents=(), bbox=(1.0, 51.0, 1.4, 51.4))
    choice = disambiguate([a, b], frozenset(), session_bbox=(-3.4, 55.7, -2.9, 56.2))
    assert choice.record is a


def test_ordering_is_deterministic():
    a, b = Rec("a", 0.5, (), B1), Rec("b", 0.5, (), B2)
    assert disambiguate([a, b], frozenset()).alternatives == \
           disambiguate([b, a], frozenset()).alternatives
```

The first test encodes the priority ordering as an executable claim, which is the only way it survives. Every future change that "improves" the scoring by weighting prominence more heavily will fail it, and that failure is the conversation worth having.

The fourth test guards something subtler than it appears. Sorting on score alone leaves ties broken by input order, and input order comes from the gazetteer, which means the same question can resolve differently on two calls that returned the same records in a different sequence. Adding the identifier to the sort key costs nothing and removes a class of irreproducible behaviour.

## Gotchas & Edge Cases

**Session context inherited too eagerly.** A conversation that has moved on to a different region will keep resolving names into the old one. Expire the session extent after a few turns without a spatial reference, or reset it when the user names a new region explicitly.

**Administrative names matched loosely.** Substring matching on region names produces false positives — a question mentioning "Newport" matches a candidate whose parent is "Newport", which is circular. Match on whole normalised names against the candidate's parent list, not on substrings of the question.

**Alternatives truncated before the right one.** Returning the top four alternatives is convenient and can exclude the correct record when a name is very common. Where the candidate count is large, say so in the rationale so the agent can offer to narrow rather than presenting four of forty as though they were all.

<figure class="diagram">
<svg viewBox="26 36 708 177" role="img" aria-labelledby="ddt-sess-t ddt-sess-d" xmlns="http://www.w3.org/2000/svg"><title id="ddt-sess-t">A session extent that has grown too large to discriminate</title><desc id="ddt-sess-d">After several questions about different regions the union of resolved extents covers most of the country, at which point every candidate overlaps it and the signal stops separating anything.</desc><rect x="26" y="36" width="708" height="177" fill="#ffffff"/><rect x="40" y="50" width="120" height="70" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="100" y="90" fill="#1f2937" font-size="12" text-anchor="middle">turn 1</text><rect x="180" y="50" width="200" height="70" rx="6" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="280" y="90" fill="#1f2937" font-size="12" text-anchor="middle">turns 1&#8211;3</text><rect x="400" y="50" width="320" height="70" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="560" y="90" fill="#1f2937" font-size="12" text-anchor="middle">turns 1&#8211;8: covers everything</text><text x="380" y="168" fill="#1f2937" font-size="13" text-anchor="middle">Every candidate now overlaps the session extent, so the signal contributes nothing</text><text x="380" y="196" fill="#5b6471" font-size="12" text-anchor="middle">Cap the extent and decay it, or it silently stops working</text></svg>
<figcaption><b>A signal that degrades to always-true is worse than one that is absent.</b> It keeps contributing its weight to every candidate, which shifts scores without discriminating and quietly widens the band where the margin test fires.</figcaption>
</figure>

**A margin tuned on one language or region.** Score distributions differ between a gazetteer that covers one country densely and one that covers the world thinly. Check the margin against the ambiguity fixture rather than assuming a constant transfers.

**Refusals that do not reach the user.** An unresolved choice that the agent silently converts into "I could not find that place" wastes the alternatives it was given. The refusal path should name the candidates: asking "did you mean the one in Shropshire or the one in Gwent" is the entire point of returning them.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should the model be allowed to pick from the alternatives?</span></summary><p>Yes, and it is often the best consumer of them, because it can see conversational cues the scorer cannot — a user's earlier mention of a county, a document they referenced. The constraint is that its choice must be intersected with the identifiers supplied: a model that returns a place not in the list has invented one, and that is exactly the failure grounding exists to prevent.</p></details>

<details class="faq-item"><summary><span>How should the session extent be maintained?</span></summary><p>As the union of the extents of places resolved so far in the conversation, capped at a reasonable size and decayed over turns. An uncapped union grows to cover a country after a handful of questions about different places, at which point it stops discriminating; decay keeps it tracking what the conversation is currently about rather than everything it has ever mentioned.</p></details>

<details class="faq-item"><summary><span>What about names that are ambiguous within one region?</span></summary><p>They are the hardest case and the one where prominence helps least, because two villages in the same county have similar prominence and share every parent. Feature type sometimes separates them; otherwise this is a genuine ambiguity and the refusal is the correct output. A user asking about one of two identically named villages in the same district expects to be asked which.</p></details>

<details class="faq-item"><summary><span>Does this belong before or after retrieval?</span></summary><p>Before, because retrieval needs the region. That creates an ordering problem for names that only appear in retrieved documents, which cannot be disambiguated before the retrieval that surfaced them — handle those in a second pass with the retrieved context as an additional signal, and give them lower confidence, since the mention was incidental rather than chosen.</p></details>

<details class="faq-item"><summary><span>How should a user&#8217;s correction be recorded?</span></summary><p>As a pinned resolution for that name in that conversation, overriding the scorer for the rest of the session. It should also be logged as an evaluation case: a user correcting a disambiguation is telling you precisely which signal was missing, and a handful of those cases is a better guide to weighting than any amount of reasoning about the scores.</p></details>

<details class="faq-item"><summary><span>Is prominence worth including at all?</span></summary><p>Yes, as the tiebreaker it is. Without it, two candidates with no distinguishing context always fall inside the margin and every such question becomes a clarification, which is exhausting for a user asking about a well-known city that happens to share its name with a hamlet. A small prominence weight resolves the genuinely lopsided cases and leaves the close ones ambiguous, which is the behaviour you want from a tiebreaker.</p></details>

## Related

- Up to the parent topic: [Geocoding and Place-Name Resolution](/spatial-llm-architecture-core-concepts/geocoding-and-place-name-resolution/)
- [Grounding Place Names Against a Gazetteer](/spatial-llm-architecture-core-concepts/geocoding-and-place-name-resolution/grounding-place-names-against-a-gazetteer/)
- Related technique: [Tuning Fusion Weights for Toponym-Heavy Queries](/geospatial-rag-pipelines/hybrid-spatial-keyword-retrieval/tuning-fusion-weights-for-toponym-heavy-queries/)
- Concept: [Spatial Reasoning and Relation Inference](/spatial-llm-architecture-core-concepts/spatial-reasoning-and-relation-inference/)
