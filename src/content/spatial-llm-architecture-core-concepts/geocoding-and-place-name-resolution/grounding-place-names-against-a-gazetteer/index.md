---
title: Grounding Place Names Against a Gazetteer
description: Replace model-recalled coordinates with gazetteer lookups that return an identifier, an extent and a confidence, and reject anything the corpus cannot support.
slug: grounding-place-names-against-a-gazetteer
type: howto
breadcrumb: Grounding Against a Gazetteer
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Grounding Place Names Against a Gazetteer

The moment a place name becomes a coordinate is the moment a spatial agent either acquires provenance or loses it. This guide implements the lookup that keeps it: names in, records out, with an identifier and an extent that everything downstream can check — the first working step of [geocoding and place-name resolution](/spatial-llm-architecture-core-concepts/geocoding-and-place-name-resolution/).

## When to Use This Approach

Ground every name that will influence a spatial computation. Names that only appear in prose the model is summarising do not need grounding; names that determine a region, a distance or a containment claim always do.

| Name appears in | Ground it? | Why |
|-----------------|------------|-----|
| The user's question | Always | It sets the region for everything downstream |
| A retrieved document, cited in the answer | Usually | The answer will imply a position |
| A retrieved document, background only | No | Grounding costs a lookup and changes nothing |
| The model's own output | Always, before it is acted on | This is where invented places enter |

The final row is the one that justifies the whole exercise. A model naming a place the corpus has never mentioned is either recalling something real or inventing something plausible, and only a lookup distinguishes them.

<figure class="diagram">
<svg viewBox="10 20 760 210" role="img" aria-labelledby="gpg-flow-t gpg-flow-d" xmlns="http://www.w3.org/2000/svg"><title id="gpg-flow-t">A name becoming a grounded record</title><desc id="gpg-flow-d">A name is normalised, looked up against the gazetteer, checked against the corpus extent and returned as a record with an identifier, an extent and a confidence, or rejected with a reason.</desc><rect x="10" y="20" width="760" height="210" fill="#ffffff"/><rect x="24" y="90" width="150" height="70" rx="8" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="198" y="90" width="150" height="70" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="372" y="90" width="150" height="70" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="546" y="34" width="210" height="70" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="546" y="146" width="210" height="70" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="12.5" text-anchor="middle" font-weight="600"><text x="99" y="118">name</text><text x="273" y="118">normalise</text><text x="447" y="118">gazetteer</text><text x="651" y="62">grounded record</text><text x="651" y="174">rejected</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="99" y="140">as written</text><text x="273" y="140">case, accents, aliases</text><text x="447" y="140">candidate records</text><text x="651" y="86">id, extent, confidence</text><text x="651" y="198">reason, alternatives</text></g><g stroke="#5b6471" stroke-width="2" marker-end="url(#gpg-flow-a)"><line x1="176" y1="125" x2="194" y2="125"/><line x1="350" y1="125" x2="368" y2="125"/><line x1="524" y1="112" x2="542" y2="82"/><line x1="524" y1="138" x2="542" y2="168"/></g><defs><marker id="gpg-flow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/></marker></defs></svg>
<figcaption><b>Two exits, both structured.</b> The rejection carries as much information as the success — a reason and the alternatives considered — because a name that failed to ground is a question the agent can ask rather than an error it must swallow.</figcaption>
</figure>

## Implementation

The lookup normalises the name, queries the gazetteer, applies alias expansion, and grounds the result against the corpus extent before returning it.

```python
import logging
import unicodedata
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

log = logging.getLogger("gazetteer_grounding")


@dataclass(frozen=True)
class PlaceRecord:
    place_id: str
    name: str
    feature_type: str
    bbox: tuple[float, float, float, float]
    importance: float
    parents: tuple[str, ...]


@dataclass(frozen=True)
class Grounded:
    record: Optional[PlaceRecord]
    confidence: float
    alternatives: tuple[PlaceRecord, ...]
    reason: str


def normalise_name(raw: str) -> str:
    """Fold case and accents for lookup; the original is kept for display."""
    folded = unicodedata.normalize("NFKD", raw.strip())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return " ".join(folded.lower().split())


def ground(
    raw_name: str,
    gazetteer: Callable[[str], Sequence[PlaceRecord]],
    corpus_bbox: tuple[float, float, float, float],
    aliases: dict[str, str] | None = None,
    limit: int = 10,
) -> Grounded:
    """Look a name up and ground it against the corpus. Never raises."""
    if not raw_name or not raw_name.strip():
        return Grounded(None, 0.0, (), "empty name")

    key = normalise_name(raw_name)
    if aliases and key in aliases:
        log.info("expanding alias %r to %r", key, aliases[key])
        key = aliases[key]

    try:
        candidates = list(gazetteer(key))[:limit]
    except Exception as exc:                        # outage must not fail the turn
        log.warning("gazetteer lookup failed for %r: %s", raw_name, exc)
        return Grounded(None, 0.0, (), f"gazetteer unavailable: {exc}")

    if not candidates:
        return Grounded(None, 0.0, (), f"no gazetteer record for {raw_name!r}")

    in_corpus = [c for c in candidates if _overlaps(c.bbox, corpus_bbox)]
    if not in_corpus:
        return Grounded(None, 0.0, tuple(candidates[:3]),
                        f"{raw_name!r} resolves outside the area this corpus covers")

    if len(in_corpus) == 1:
        return Grounded(in_corpus[0], 0.8, (), "unique record within the corpus extent")

    ranked = sorted(in_corpus, key=lambda c: (-c.importance, c.place_id))
    return Grounded(None, 0.0, tuple(ranked[:4]),
                    f"{len(in_corpus)} records named {raw_name!r} within the corpus extent")


def _overlaps(a, b) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])
```

Two behaviours here are deliberate and easy to get wrong. A name with several candidates inside the corpus is *not* resolved by prominence at this stage — it is returned unresolved with the candidates attached, so the disambiguation logic can apply context the lookup does not have. And a name that resolves only outside the corpus is a distinct outcome from one that does not resolve at all, because the first tells the user their question is about somewhere this system has no data for, which is a much more useful message.

The corpus-extent check also acts as a cheap plausibility screen on model-supplied names. A model that invents a town will usually invent one the gazetteer has never heard of; where it invents one that exists elsewhere in the world, the extent check catches it.

<figure class="diagram">
<svg viewBox="16 32 748 214" role="img" aria-labelledby="gpg-out-t gpg-out-d" xmlns="http://www.w3.org/2000/svg"><title id="gpg-out-t">Four outcomes and the message each one produces</title><desc id="gpg-out-d">A unique record grounds cleanly, several records return unresolved with alternatives, a record outside the corpus produces a coverage message, and no record at all suggests the name may be invented.</desc><rect x="16" y="32" width="748" height="214" fill="#ffffff"/><rect x="30" y="46" width="360" height="86" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="410" y="46" width="340" height="86" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="30" y="146" width="360" height="86" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="410" y="146" width="340" height="86" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13" font-weight="600"><text x="52" y="76">one record</text><text x="432" y="76">several records</text><text x="52" y="176">outside the corpus</text><text x="432" y="176">no record at all</text></g><g fill="#5b6471" font-size="12"><text x="52" y="102">ground it, confidence 0.8</text><text x="52" y="122">proceed normally</text><text x="432" y="102">return unresolved, with candidates</text><text x="432" y="122">let context or the user decide</text><text x="52" y="202">&#8220;this system has no data there&#8221;</text><text x="52" y="222">a coverage answer, not a failure</text><text x="432" y="202">possibly an invented place</text><text x="432" y="222">never proceed on it</text></g></svg>
<figcaption><b>Four different messages, not one error.</b> Each outcome tells the user something specific and actionable, which is the practical difference between a system that grounds names and one that merely looks them up.</figcaption>
</figure>

## Validation & Testing

```python
CORPUS = (-8.0, 49.0, 2.0, 61.0)


def test_unique_record_grounds():
    g = ground("Kirkby Lonsdale", lambda _: [REC_KIRKBY], CORPUS)
    assert g.record is REC_KIRKBY and g.confidence >= 0.8


def test_several_records_return_unresolved_with_alternatives():
    g = ground("Newport", lambda _: [REC_NEWPORT_A, REC_NEWPORT_B], CORPUS)
    assert g.record is None and len(g.alternatives) == 2


def test_outside_corpus_is_distinct_from_not_found():
    outside = ground("Springfield", lambda _: [REC_SPRINGFIELD_US], CORPUS)
    missing = ground("Nowhereton", lambda _: [], CORPUS)
    assert "outside the area" in outside.reason
    assert "no gazetteer record" in missing.reason


def test_gazetteer_outage_degrades_rather_than_raises():
    def broken(_):
        raise ConnectionError("gazetteer down")
    g = ground("Kirkby Lonsdale", broken, CORPUS)
    assert g.record is None and "unavailable" in g.reason
```

The third test is the one that keeps the messages distinct. It is tempting to collapse both cases into "not found", and doing so replaces two useful answers with one unhelpful one — the user who asked about a place in another country gets told their place does not exist.

The fourth test is worth keeping even though it looks like a formality. A gazetteer is an external dependency on the hot path of every spatial question, and a system that raises when it is unavailable turns a degraded lookup into a completely failed conversation — which is a much larger outage than the one that actually occurred.

## Gotchas & Edge Cases

**Alias expansion applied before normalisation.** Aliases keyed on raw strings miss every variation in case and accent. Normalise first, key the alias table on normalised names, and keep the original for display.

**Gazetteer identifiers reused across sources.** Two gazetteers can both issue an identifier of "12345". Namespace identifiers by source at ingestion, or a record from one will eventually be mistaken for a record from the other.

<figure class="diagram">
<svg viewBox="26 42 708 168" role="img" aria-labelledby="gpg-ns-t gpg-ns-d" xmlns="http://www.w3.org/2000/svg"><title id="gpg-ns-t">Why gazetteer identifiers need namespacing</title><desc id="gpg-ns-d">Two gazetteers can issue the same numeric identifier for different places, so an unnamespaced identifier from one source can be silently matched against a record from the other.</desc><rect x="26" y="42" width="708" height="168" fill="#ffffff"/><rect x="40" y="56" width="300" height="60" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="420" y="56" width="300" height="60" rx="8" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><text x="190" y="92" fill="#1f2937" font-size="12.5" text-anchor="middle">source A record 12345</text><text x="570" y="92" fill="#1f2937" font-size="12.5" text-anchor="middle">source B record 12345</text><rect x="40" y="140" width="680" height="56" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="380" y="164" fill="#1f2937" font-size="12.5" text-anchor="middle">stored as &#8220;12345&#8221;: the two are now the same place</text><text x="380" y="186" fill="#5b6471" font-size="12" text-anchor="middle">stored as &#8220;a:12345&#8221; and &#8220;b:12345&#8221;: they are not</text></svg>
<figcaption><b>A collision with no symptom.</b> Nothing errors when two sources share an identifier; a citation simply points at the wrong record, and the mistake surfaces only when somebody follows the link.</figcaption>
</figure>

**Corpus extent computed from a stale index.** If the extent used for grounding comes from an index snapshot rather than from the current corpus, newly added regions will be reported as uncovered. Recompute it on index build and expose it as a value the grounding code reads rather than a constant.

**Feature type ignored.** A river and a town can share a name, and grounding a question about "the Avon" to the town produces answers about the wrong kind of thing entirely. Where the question implies a type — "along", "upstream", "the town of" — filter candidates by it before ranking.

**Very common names dominating the lookup.** A gazetteer query for a name that appears hundreds of times returns a truncated list that may exclude the right record. Where the count exceeds the limit, say so in the reason rather than silently ranking the first ten.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should grounding run on every turn of a conversation?</span></summary><p>Once per name per conversation, cached. Re-grounding is not just wasteful — it lets the same name resolve differently on turn four than on turn one, because the corpus or the gazetteer changed in between, and a user who notices that has good reason to distrust everything else. Cache successful groundings for the session and let an explicit correction from the user override them.</p></details>

<details class="faq-item"><summary><span>What confidence should a unique gazetteer match get?</span></summary><p>High but not maximal — around 0.8 in the scheme used here. A unique match is strong evidence and not proof: gazetteers have gaps, and a name that matches exactly one record may still be the wrong record if the right one is simply absent. Reserving the top of the range for cases corroborated by context keeps the band meaningful.</p></details>

<details class="faq-item"><summary><span>How should a name that only appears in retrieved text be handled?</span></summary><p>Ground it with lower confidence than a name the user typed, because the mention was not a deliberate choice by anyone. In practice that means grounding it for the purpose of linking it to a record, and declining to use it as the region for a spatial filter unless the user's own question also referenced it. The distinction matters most in long documents that mention many places in passing.</p></details>

<details class="faq-item"><summary><span>Is it worth building a local gazetteer subset?</span></summary><p>Almost always, for latency and for reliability. A subset covering your regions, refreshed on a schedule, removes an external dependency from the hot path and makes the corpus-extent check trivial because the subset and the corpus share a boundary. Keep the upstream lookup as a fallback for names the subset misses, and log those misses — they are the list of records the subset should be extended with.</p></details>

<details class="faq-item"><summary><span>What should be stored when a name grounds successfully?</span></summary><p>The namespaced identifier, the extent, the confidence, and the gazetteer version — the same quartet that makes any other resolution in this pipeline reconstructible. Storing the coordinates alone is the shortcut that removes every later possibility of checking the decision, and it is particularly costly here because gazetteer records change: a village annexed into a town keeps its name and acquires a different extent, and only the identifier and version let you see that the answer moved for a reason.</p></details>

## Related

- Up to the parent topic: [Geocoding and Place-Name Resolution](/spatial-llm-architecture-core-concepts/geocoding-and-place-name-resolution/)
- [Disambiguating Duplicate Toponyms with Spatial Context](/spatial-llm-architecture-core-concepts/geocoding-and-place-name-resolution/disambiguating-duplicate-toponyms-with-spatial-context/)
- Related topic: [Hybrid Spatial and Keyword Retrieval](/geospatial-rag-pipelines/hybrid-spatial-keyword-retrieval/)
- Concept: [Evaluation and Benchmarking for Spatial LLMs](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/)
