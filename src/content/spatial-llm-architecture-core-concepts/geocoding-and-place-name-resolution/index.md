---
title: Geocoding and Place-Name Resolution
description: Turn place names in prompts into coordinates you can defend — gazetteer grounding, ambiguity handling, confidence that survives to the answer, and a refusal path.
slug: geocoding-and-place-name-resolution
type: topic
breadcrumb: Geocoding and Place Names
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Geocoding and Place-Name Resolution

Every spatial question that arrives in words has to become a position before anything can be computed from it. That conversion is the most under-engineered stage in most spatial agents: a model is asked where somewhere is, it answers with coordinates that look right, and nothing downstream can tell that the answer was recalled rather than looked up. Place-name resolution is the discipline of making that step a lookup with provenance instead of a generation with confidence.

This topic belongs to [spatial LLM architecture and core concepts](/spatial-llm-architecture-core-concepts/) and addresses one failure precisely: an agent that places a town in the wrong country because two towns share a name, or that invents a plausible coordinate for a place it has never seen. It feeds every stage that needs a focus — the region filter in [spatial context retrieval and reranking](/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/), the routing decisions in [fallback routing for geospatial queries](/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/) — and its confidence is the input those stages gate on.

<figure class="diagram">
<svg viewBox="16 38 748 188" role="img" aria-labelledby="gpn-two-t gpn-two-d" xmlns="http://www.w3.org/2000/svg"><title id="gpn-two-t">Recalled coordinates against looked-up coordinates</title><desc id="gpn-two-d">A model asked for a position produces a plausible answer with no provenance and no way to detect an error; a gazetteer lookup produces a position with an identifier, an extent and a confidence that downstream stages can check.</desc><rect x="16" y="38" width="748" height="188" fill="#ffffff"/><rect x="30" y="52" width="340" height="160" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="200" y="84" fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600">recalled by the model</text><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="200" y="114">a coordinate pair, formatted well</text><text x="200" y="138">no identifier, no source</text><text x="200" y="162">indistinguishable from correct</text><text x="200" y="188">wrong by 200 km or by nothing</text></g><rect x="410" y="52" width="340" height="160" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="580" y="84" fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600">looked up in a gazetteer</text><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="580" y="114">a coordinate pair and an extent</text><text x="580" y="138">record identifier and source</text><text x="580" y="162">confidence and alternatives</text><text x="580" y="188">checkable, and refusable</text></g></svg>
<figcaption><b>The two outputs look identical in a response body.</b> That is what makes recalled coordinates dangerous: nothing about the shape of the answer reveals which one you got, so the distinction has to be enforced at the point of generation.</figcaption>
</figure>

## Foundational Principles

**A model may propose a name, never a coordinate.** Extracting "the town of Springfield" from a question is a language task and models are good at it. Deciding which Springfield, and where it is, is a lookup against a reference dataset. Keeping that boundary sharp is the single change that most improves a spatial agent's reliability, and it is also the easiest to erode, because a model that occasionally supplies coordinates directly appears to work.

**Ambiguity is the normal case, not an exception.** Duplicate toponyms are common enough that any system handling more than one country will meet them constantly. A resolver that returns one result per name is not simpler than one that returns a ranked set; it has simply moved the decision somewhere invisible.

**A resolution carries an extent, not just a point.** "Where is this county" has an answer that is thousands of square kilometres, and collapsing it to a centroid produces distance calculations that are wrong by tens of kilometres in ways nothing detects. Return the geometry the gazetteer holds, and let the caller reduce it deliberately if they need a point.

## Step-by-Step Implementation Pipeline

### 1. Extract candidate names without resolving them

The extraction step identifies spans that look like places and classifies their type, and it should be generous: a false positive costs a lookup that returns nothing, while a false negative means a place silently never becomes a position.

```python
import logging
import re
from dataclasses import dataclass

log = logging.getLogger("place_extraction")

_TITLECASE = re.compile(r"\b(?:[A-Z][a-zÀ-ſ'’-]+)(?:[ -](?:[A-Z][a-zÀ-ſ'’-]+|of|the|upon|on|le|la))*\b")
_ADMIN_HINT = re.compile(r"\b(county|district|borough|parish|region|province|state|shire)\b", re.I)


@dataclass(frozen=True)
class Mention:
    text: str
    start: int
    end: int
    admin_hint: bool


def extract_mentions(question: str, stopnames: frozenset[str]) -> list[Mention]:
    """Find place-like spans. Generous by design: unresolvable ones cost one lookup."""
    out: list[Mention] = []
    for m in _TITLECASE.finditer(question):
        text = m.group(0).strip()
        if text.lower() in stopnames or len(text) < 3:
            continue
        window = question[max(0, m.start() - 40): m.end() + 40]
        out.append(Mention(text, m.start(), m.end(), bool(_ADMIN_HINT.search(window))))
    if not out:
        log.info("no place mention found in %r — the question may be non-spatial", question)
    return out
```

The stop-name set is not optional. Sentence-initial capitalisation makes "The", "Where" and "Flood" look like proper nouns, and a resolver that dutifully looks them up will occasionally find a real place called Flood and anchor the whole query to it.

### 2. Look up each mention against a gazetteer, returning a ranked set

The lookup returns every plausible match with its type, extent, population or importance rank, and containing administrative units. It never returns one result silently.

```python
from typing import Callable, Sequence


@dataclass(frozen=True)
class PlaceRecord:
    place_id: str
    name: str
    feature_type: str            # settlement, admin, water, transport…
    bbox: tuple[float, float, float, float]
    importance: float            # gazetteer's own prominence measure, 0..1
    parents: tuple[str, ...]     # containing administrative names, largest last


def lookup(mention: Mention, gazetteer: Callable[[str], Sequence[PlaceRecord]],
           limit: int = 10) -> list[PlaceRecord]:
    """Return candidates; an empty list is a valid, meaningful answer."""
    try:
        hits = list(gazetteer(mention.text))[:limit]
    except Exception as exc:                       # gazetteer outage must not fail the turn
        log.warning("gazetteer lookup failed for %r: %s", mention.text, exc)
        return []
    if not hits:
        log.info("no gazetteer match for %r", mention.text)
    return hits
```

### 3. Disambiguate with context, and refuse when context is absent

Choosing among candidates uses three signals: any administrative names elsewhere in the question, the extent of the conversation so far, and the gazetteer's own importance ranking as a last resort. The full treatment is in [disambiguating duplicate toponyms with spatial context](/spatial-llm-architecture-core-concepts/geocoding-and-place-name-resolution/disambiguating-duplicate-toponyms-with-spatial-context/).

```python
@dataclass(frozen=True)
class Resolution:
    record: PlaceRecord | None
    confidence: float
    alternatives: tuple[PlaceRecord, ...]
    rationale: str


def disambiguate(candidates: Sequence[PlaceRecord], context_names: frozenset[str],
                 session_bbox=None) -> Resolution:
    """Pick a candidate, or return None with the alternatives intact."""
    if not candidates:
        return Resolution(None, 0.0, (), "no gazetteer candidate")
    if len(candidates) == 1:
        return Resolution(candidates[0], 0.8, (), "unique gazetteer match")

    scored = []
    for c in candidates:
        score = 0.15 + 0.35 * c.importance
        if context_names & {p.lower() for p in c.parents}:
            score += 0.40                          # an administrative name in the question
        if session_bbox is not None and _overlaps(c.bbox, session_bbox):
            score += 0.20                          # consistent with what we were discussing
        scored.append((round(min(1.0, score), 3), c))

    scored.sort(key=lambda t: (-t[0], t[1].place_id))
    best, runner_up = scored[0], scored[1]
    if best[0] - runner_up[0] < 0.12:              # too close to call
        return Resolution(None, best[0], tuple(c for _, c in scored[:4]),
                          "ambiguous: candidates within scoring noise")
    return Resolution(best[1], best[0], tuple(c for _, c in scored[1:4]),
                      "selected on context and prominence")
```

The margin test is the part worth defending in review. Returning the top candidate regardless of margin makes the resolver look decisive and produces a systematic error rate equal to how often the second candidate was right. Returning nothing with the alternatives attached lets the agent ask, which is nearly always the better move for a question that a human could disambiguate in four words.

<figure class="diagram">
<svg viewBox="16 38 768 218" role="img" aria-labelledby="gpn-amb-t gpn-amb-d" xmlns="http://www.w3.org/2000/svg"><title id="gpn-amb-t">Three outcomes from one place-name lookup</title><desc id="gpn-amb-d">A unique match resolves with high confidence, a clear winner among several resolves with its alternatives recorded, and a near tie returns no selection so the agent can ask which place was meant.</desc><rect x="16" y="38" width="768" height="218" fill="#ffffff"/><rect x="30" y="52" width="230" height="150" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="285" y="52" width="230" height="150" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="540" y="52" width="230" height="150" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="145" y="84">unique match</text><text x="400" y="84">clear winner</text><text x="655" y="84">near tie</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="145" y="114">one gazetteer record</text><text x="145" y="140">confidence 0.8</text><text x="145" y="168">proceed normally</text><text x="400" y="114">context names a parent</text><text x="400" y="140">margin above threshold</text><text x="400" y="168">proceed, record alternatives</text><text x="655" y="114">two plausible places</text><text x="655" y="140">margin inside the noise</text><text x="655" y="168">ask which one was meant</text></g><text x="400" y="238" fill="#1f2937" font-size="13" text-anchor="middle">The third column is the one a decisive resolver silently converts into an error</text></svg>
<figcaption><b>Asking is cheap; being wrong is not.</b> A four-word clarifying question costs one turn, while a confidently wrong resolution propagates into every distance, containment and retrieval decision that follows.</figcaption>
</figure>

### 4. Ground the result against the reference dataset before returning it

A resolution is not finished until it has been checked against the dataset it will be used with. A gazetteer entry for a village is useless if the corpus covers a different country, and detecting that mismatch at resolution time produces a much better error message than detecting it as an empty result set three stages later. The mechanics are in [grounding place names against a gazetteer](/spatial-llm-architecture-core-concepts/geocoding-and-place-name-resolution/grounding-place-names-against-a-gazetteer/).

```python
def ground(res: Resolution, corpus_bbox) -> Resolution:
    """Reject a resolution that falls outside the data we can actually answer from."""
    if res.record is None:
        return res
    if not _overlaps(res.record.bbox, corpus_bbox):
        log.info("resolved %r outside the corpus extent", res.record.name)
        return Resolution(None, 0.0, res.alternatives,
                          "resolved outside the area this corpus covers")
    return res
```

### 5. Carry the resolution forward as data, not as a coordinate

Downstream stages need the identifier, the geometry, the confidence and the alternatives — not a bare latitude and longitude. A pipeline that flattens the resolution into two floats loses the ability to explain an answer, to detect an inconsistency between turns, or to offer the alternative when a user says "no, the other one".

```python
def to_context(res: Resolution) -> dict:
    """The shape every downstream stage should receive."""
    if res.record is None:
        return {"resolved": False, "confidence": res.confidence,
                "alternatives": [a.name for a in res.alternatives],
                "reason": res.rationale}
    return {
        "resolved": True,
        "place_id": res.record.place_id,
        "name": res.record.name,
        "feature_type": res.record.feature_type,
        "bbox": res.record.bbox,
        "confidence": res.confidence,
        "alternatives": [a.name for a in res.alternatives],
    }
```

### 6. Cache resolutions per conversation, and per corpus

Within one conversation a name should resolve once. Re-resolving on each turn is not just wasteful; it lets the answer to "how far is it from the station" refer to a different station than the previous turn did, which is a class of inconsistency users find genuinely alarming.

```python
def resolve_cached(name: str, session, corpus_id: str, resolver) -> Resolution:
    """Stable within a conversation, and shared across conversations on the same corpus."""
    key = (corpus_id, name.strip().lower())
    if key in session.resolutions:
        return session.resolutions[key]
    res = resolver(name)
    if res.record is not None:                     # never cache an unresolved name
        session.resolutions[key] = res
    return res
```

Refusing to cache an unresolved name matters because unresolved usually means "the user has not yet told us which one". Caching that state would make the clarification they are about to give ineffective.

### 7. Decide what a resolution is allowed to authorise

Not every use of a resolved place carries the same risk. Naming what is nearby tolerates a resolution that is roughly right; measuring a distance does not; making a containment claim about a regulated boundary tolerates almost nothing. The confidence produced above only becomes useful once it is mapped onto that spectrum, and the mapping belongs in code rather than in a prompt.

```python
RISK_FLOORS = {                       # minimum confidence to answer this way
    "describe": 0.35,                 # "there is a school nearby"
    "measure": 0.65,                  # "it is 400 m from the station"
    "contain": 0.80,                  # "the site is inside the conservation area"
    "act": 0.90,                      # anything that triggers a downstream action
}


def permitted(res: Resolution, intent: str) -> tuple[bool, str]:
    """Return whether this resolution may support this class of claim, and why not."""
    floor = RISK_FLOORS.get(intent)
    if floor is None:
        return False, f"unknown intent {intent!r} — refusing by default"
    if res.record is None:
        return False, f"place unresolved: {res.rationale}"
    if res.confidence < floor:
        return False, (f"confidence {res.confidence:.2f} below the {floor:.2f} needed "
                       f"to {intent}; alternatives: "
                       f"{', '.join(a.name for a in res.alternatives) or 'none'}")
    return True, ""
```

Defaulting to refusal for an unrecognised intent is deliberate. New intents get added by people extending the agent, and a permissive default means the first version of every new capability runs with no floor at all — which is precisely when the code is least well understood.

The floors themselves are policy and should be published with the system rather than buried. A team that decides containment claims need 0.8 has made a defensible judgement; a team whose containment claims happen to inherit whatever the describe path used has made no judgement at all, and will discover this the first time a boundary claim is challenged.

### 8. Record every resolution for replay

Place resolution is a decision with consequences, and like every such decision in this pipeline it needs to be reconstructible after the gazetteer has changed. Store the mention text, the chosen identifier, the confidence, the alternatives that were in play, and the gazetteer version — five fields that turn a disputed answer into a lookup.

```python
def audit_record(mention: Mention, res: Resolution, gazetteer_version: str) -> dict:
    return {
        "mention": mention.text,
        "place_id": res.record.place_id if res.record else None,
        "confidence": res.confidence,
        "alternatives": [a.place_id for a in res.alternatives],
        "gazetteer_version": gazetteer_version,
        "rationale": res.rationale,
    }
```

The alternatives field is the one people leave out and the one that answers the interesting question. When somebody reports that the agent talked about the wrong Newport, the useful thing to know is not only which Newport it chose but whether the right one was even a candidate — a scoring problem and a coverage problem need entirely different fixes, and without the alternatives recorded they look identical.

## Failure Modes & Root Causes

**The confident hallucination.** The model supplies coordinates directly and they are wrong by a country. Root cause: no hard boundary between naming and locating. Mitigation: never accept a coordinate from a generation step; intersect any model-proposed place with the gazetteer candidate set and discard anything outside it.

**The prominence trap.** Every ambiguous name resolves to the largest place with that name, so a question about a village consistently answers about a city. Root cause: importance used as the primary signal rather than the tiebreaker. Mitigation: context first, prominence last, and a margin test that refuses when the two disagree.

**The centroid distance error.** Distances are computed from the centroid of a large administrative area and are wrong by tens of kilometres. Root cause: reducing an extent to a point at resolution time. Mitigation: return geometry; let the consumer reduce it, and prefer edge distance where the question is about proximity.

**Session drift.** A name resolves differently on turn two than on turn one because the conversation's context changed. Root cause: no caching. Mitigation: per-conversation resolution cache, with an explicit path for the user to override.

## Production Validation Protocols

1. **No-coordinate assertion.** Assert that no coordinate reaches the geometry layer without an accompanying place identifier or an explicit user-supplied origin; this is the gate that catches recalled coordinates.
2. **Ambiguity regression set.** Maintain a fixture of known-ambiguous names with the expected outcome — resolved, or deliberately unresolved — and assert both kinds.
3. **Extent preservation test.** Assert that resolving an administrative area returns a polygon, not a point, and that its area is non-zero.
4. **Corpus-grounding test.** Assert that a name resolving outside the corpus extent returns an unresolved result with the explanatory reason rather than a usable position.
5. **Session stability test.** Resolve the same name twice within one session and assert identical identifiers.
6. **Unresolved-rate indicator.** Track the share of mentions that fail to resolve and alert on a step change; a spike usually means a gazetteer outage rather than a change in the questions.

<figure class="diagram">
<svg viewBox="26 42 708 186" role="img" aria-labelledby="gpn-sig-t gpn-sig-d" xmlns="http://www.w3.org/2000/svg"><title id="gpn-sig-t">Disambiguation signals in priority order</title><desc id="gpn-sig-d">An administrative name in the question is the strongest signal, followed by consistency with the conversation so far, followed by the gazetteer's prominence ranking, which is a tiebreaker rather than evidence.</desc><rect x="26" y="42" width="708" height="186" fill="#ffffff"/><rect x="40" y="56" width="680" height="46" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="40" y="112" width="500" height="46" rx="6" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="40" y="168" width="300" height="46" rx="6" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="60" y="84" fill="#1f2937" font-size="12.5">a containing administrative name appears in the question</text><text x="60" y="140" fill="#1f2937" font-size="12.5">consistent with the area already under discussion</text><text x="60" y="196" fill="#1f2937" font-size="12.5">the gazetteer thinks it is more prominent</text><text x="600" y="140" fill="#5b6471" font-size="12">weaker</text><text x="420" y="196" fill="#5b6471" font-size="12">weakest — a tiebreaker only</text></svg>
<figcaption><b>Prominence is the shortest bar for a reason.</b> It is a property of the world rather than of the question, so it can settle a tie between otherwise equal candidates and should never outweigh something the user actually said.</figcaption>
</figure>

Of these, the no-coordinate assertion is the one that changes how the system is built rather than merely catching a bug. Once it is in place, every path that needs a position has to go through the resolver, which means every position in the system has an identifier, a source and a confidence attached to it. That single structural constraint eliminates the entire class of failures this topic exists to address, and it is far easier to add at the start than to retrofit once a dozen call sites have learned to pass bare coordinate pairs around.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Which gazetteer should a production system use?</span></summary><p>Whichever one covers your regions with the feature types your questions ask about, and whose licence permits the use. Global gazetteers are excellent for settlements and weak on small features such as individual buildings or field parcels; national datasets are the reverse. Most production systems end up consulting two, which is fine provided identifiers are namespaced so a record from one is never confused with a record from the other.</p></details>

<details class="faq-item"><summary><span>Should the model be shown the candidate list?</span></summary><p>Yes, when the resolver has declined to choose. Handing the model three candidates with their containing regions lets it either pick using something in the conversation the resolver could not see, or ask a well-formed question naming the actual alternatives. What it must not be allowed to do is add a candidate that was not in the list, which is enforced by intersecting its answer with the identifiers you supplied.</p></details>

<details class="faq-item"><summary><span>How should historic or renamed places be handled?</span></summary><p>With an alias table that records the period each name was current, applied at query time rather than baked into the index. A document from 1950 and one from last year may legitimately use different names for the same place, and a resolver that silently maps the old name to the current record will produce answers that are geographically right and historically misleading. Return the record with a note about the name change rather than substituting quietly.</p></details>

<details class="faq-item"><summary><span>Does this belong before or after retrieval?</span></summary><p>Before, because retrieval needs the region and the region comes from the resolution. The exception is a question where the place only appears in retrieved documents rather than in the question itself, which is a different problem — there the resolution runs over the retrieved text and its confidence should be lower, since the mention was not something the user chose to say.</p></details>

## Related

- Up to the section overview: [Spatial LLM Architecture and Core Concepts](/spatial-llm-architecture-core-concepts/)
- Technique: [Grounding Place Names Against a Gazetteer](/spatial-llm-architecture-core-concepts/geocoding-and-place-name-resolution/grounding-place-names-against-a-gazetteer/)
- Technique: [Disambiguating Duplicate Toponyms with Spatial Context](/spatial-llm-architecture-core-concepts/geocoding-and-place-name-resolution/disambiguating-duplicate-toponyms-with-spatial-context/)
- Peer topic: [Spatial Reasoning and Relation Inference](/spatial-llm-architecture-core-concepts/spatial-reasoning-and-relation-inference/)
- Peer topic: [Coordinate Reference System Normalization](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/)
- Related topic: [Hybrid Spatial and Keyword Retrieval](/geospatial-rag-pipelines/hybrid-spatial-keyword-retrieval/)
