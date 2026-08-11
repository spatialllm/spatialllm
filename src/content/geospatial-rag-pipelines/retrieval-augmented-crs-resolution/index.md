---
title: Retrieval-Augmented CRS Resolution
description: Pin down the coordinate reference system of an ambiguous geographic mention from retrieved evidence and library validation, instead of letting a model guess it.
slug: retrieval-augmented-crs-resolution
type: topic
breadcrumb: Retrieval-Augmented CRS Resolution
datePublished: 2025-03-18
dateModified: 2026-08-11
---

# Retrieval-Augmented CRS Resolution

Ambiguous geographic mentions rarely announce their coordinate reference system. A gazetteer name, a six-figure national grid reference, or a bare easting/northing pair carries an implicit frame that a language model cannot recover from the token stream alone. Retrieval-augmented resolution fetches supporting evidence from a spatial knowledge index and pins down the correct EPSG code before any reasoning begins.

This topic sits inside the [geospatial RAG pipelines](/geospatial-rag-pipelines/) discipline and addresses one failure mode precisely: an agent that silently assumes WGS84 for coordinates recorded in a projected local grid, producing answers displaced by hundreds of metres. The retrieval layer supplies datum hints, provenance, and neighbouring geometry that make the decision explicit and auditable rather than guessed. It builds directly on [coordinate reference system normalization](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/), extending that ingestion-time gate with a retrieval step for the harder case where the frame is not declared anywhere in the input.

<figure class="diagram">
<svg viewBox="0 0 760 300" role="img" aria-labelledby="rcr-amb-t rcr-amb-d" xmlns="http://www.w3.org/2000/svg"><title id="rcr-amb-t">One coordinate pair, three defensible reference frames</title><desc id="rcr-amb-d">The same numeric pair reads as a plausible position in three different frames, landing in three different places on the ground. Only retrieved provenance distinguishes them, so magnitude alone can never settle the question.</desc><rect x="0" y="0" width="760" height="300" fill="#ffffff"/><rect x="250" y="30" width="260" height="56" rx="8" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><text x="380" y="54" fill="#1f2937" font-size="13" font-weight="600" text-anchor="middle">easting 651409, northing 313177</text><text x="380" y="74" fill="#5b6471" font-size="12" text-anchor="middle">no frame declared anywhere in the document</text><g stroke="#5b6471" stroke-width="2" marker-end="url(#rcr-amb-a)"><line x1="330" y1="88" x2="150" y2="140"/><line x1="380" y1="88" x2="380" y2="140"/><line x1="430" y1="88" x2="610" y2="140"/></g><defs><marker id="rcr-amb-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/></marker></defs><rect x="40" y="144" width="220" height="74" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="270" y="144" width="220" height="74" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="500" y="144" width="220" height="74" rx="8" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="150" y="174">EPSG:27700</text><text x="380" y="174">EPSG:32631</text><text x="610" y="174">EPSG:2154</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="150" y="198">a point in East Anglia</text><text x="380" y="198">a point off Nigeria</text><text x="610" y="198">a point in Brittany</text></g><text x="380" y="256" fill="#1f2937" font-size="13" text-anchor="middle">All three are numerically plausible — the magnitudes cannot break the tie</text><text x="380" y="280" fill="#5b6471" font-size="12" text-anchor="middle">Retrieved provenance decides; the model is never asked to</text></svg>
<figcaption><b>Why this needs retrieval at all.</b> Range checks narrow the field but never close it, because projected grids were designed to produce comfortable six-figure numbers over their own territory. Evidence about where the document came from is the only thing that separates these three answers.</figcaption>
</figure>

## Foundational Principles

**Never let the model choose the frame.** A language model will happily emit `EPSG:4326` for a British National Grid reference because both look plausible in text. Frame selection must be a deterministic function of retrieved evidence and library validation, not a generated token. The model reasons over geometry that has already been resolved. This is the same separation of duties that keeps generated SQL from touching an unvetted function list — the model proposes, a deterministic layer disposes.

**Every resolution carries a confidence.** A frame that survives a validated round-trip and agrees with retrieved provenance earns high confidence; one that falls back earns a low-confidence flag that propagates downstream. Callers must be able to gate on it — a distance query answered from a low-confidence frame is worse than a refusal, because it is wrong in a way that looks right.

**The fallback is WGS84, and it is loud.** When retrieval returns nothing usable or validation rejects every candidate, the pipeline degrades to EPSG:4326 and tags the result so no consumer mistakes a guess for a determination. Silence is the failure; a flagged fallback is the safe path.

## Step-by-Step Implementation Pipeline

### 1. Retrieve candidate records from the knowledge index

The knowledge index stores documents keyed by place name, grid-system alias, and datum authority, each carrying a candidate EPSG and a coordinate-plausibility envelope. Retrieval returns the top matches for the ambiguous mention; downstream steps never trust these blindly. The heuristics that turn raw neighbours into a scored code are detailed in [inferring CRS from retrieved spatial context](/geospatial-rag-pipelines/retrieval-augmented-crs-resolution/inferring-crs-from-retrieved-spatial-context/).

```python
from dataclasses import dataclass
from typing import Callable, Sequence
import logging

log = logging.getLogger("crs_resolution")

@dataclass(frozen=True)
class CRSCandidate:
    epsg: int
    source: str          # provenance: gazetteer, grid-alias, dataset header
    weight: float        # retrieval similarity in [0, 1]

def retrieve_crs_candidates(
    mention: str,
    index_query: Callable[[str, int], Sequence[CRSCandidate]],
    k: int = 5,
) -> list[CRSCandidate]:
    """Fetch candidate records; never raise — return [] so callers can fall back."""
    try:
        hits = list(index_query(mention, k))
    except Exception as exc:                      # index outage, timeout, bad response
        log.warning("index query failed for %r: %s", mention, exc)
        return []
    # Deduplicate by code, keeping the strongest retrieval weight.
    best: dict[int, CRSCandidate] = {}
    for c in hits:
        if c.epsg not in best or c.weight > best[c.epsg].weight:
            best[c.epsg] = c
    return sorted(best.values(), key=lambda c: c.weight, reverse=True)
```

### 2. Validate each candidate with pyproj

A retrieved code is only a hypothesis until `pyproj` confirms it constructs, is not deprecated, and places the mention's coordinates inside its area of use. Validation is where hallucinated or stale codes are rejected — a step the [normalization gate](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/) assumes has already run for declared inputs.

```python
from pyproj import CRS
from pyproj.exceptions import CRSError

def crs_is_usable(epsg: int, sample_xy: tuple[float, float] | None) -> bool:
    """True only if the code constructs, is current, and can host sample_xy."""
    try:
        crs = CRS.from_epsg(epsg)
    except CRSError as exc:
        log.info("Rejecting EPSG:%s — will not construct: %s", epsg, exc)
        return False
    if crs.is_deprecated:
        log.info("Rejecting EPSG:%s — deprecated", epsg)
        return False
    if sample_xy is not None and crs.area_of_use is not None and crs.is_geographic:
        x, y = sample_xy
        w, s, e, n = crs.area_of_use.bounds
        if not (w - 1 <= x <= e + 1 and s - 1 <= y <= n + 1):
            log.info("EPSG:%s area-of-use excludes %s", epsg, sample_xy)
            return False
    return True
```

The `is_geographic` guard on the envelope test is deliberate: an area of use is published in degrees, so comparing it against projected eastings and northings would reject every correct answer. For projected candidates the equivalent check is to transform the sample into geographic coordinates first and test that result, which is exactly what the round-trip gate in the validation protocols does.

### 3. Resolve with a deterministic fallback

The top-level `resolve_crs` orchestrates retrieval, validation, and scoring, always returning a structured result — never an exception. When no candidate survives, it degrades to `EPSG:4326` with a low-confidence flag so the reasoning layer and the [fallback routing](/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/) layer both see the uncertainty.

```python
@dataclass(frozen=True)
class ResolvedCRS:
    epsg: int
    confidence: float    # 0.0 fallback .. 1.0 validated + provenance-agreed
    low_confidence: bool
    rationale: str

def resolve_crs(mention, index_query, sample_xy=None, threshold=0.55) -> ResolvedCRS:
    candidates = retrieve_crs_candidates(mention, index_query)
    for cand in candidates:
        if crs_is_usable(cand.epsg, sample_xy):
            conf = round(min(1.0, 0.5 + cand.weight / 2), 3)
            return ResolvedCRS(cand.epsg, conf, conf < threshold,
                               f"validated from {cand.source}")
    log.warning("No usable frame for %r; falling back to EPSG:4326", mention)
    return ResolvedCRS(4326, 0.0, True, "deterministic fallback — no candidate validated")
```

### 4. Propagate the confidence into the answer, not just the log

A resolution that is only recorded in a log line has no effect on behaviour. The confidence must reach the caller as data, so that a low-confidence frame changes what the agent is allowed to say. Distance and containment answers are the ones to gate hardest, because a frame error moves them by a bounded but large amount; naming answers can often proceed with a hedge. The rules for turning that flag into agent behaviour are covered in [resolving ambiguous EPSG codes from document context](/geospatial-rag-pipelines/retrieval-augmented-crs-resolution/resolving-ambiguous-epsg-codes-from-document-context/).

```python
def guard_spatial_answer(resolved: ResolvedCRS, answer_kind: str) -> str | None:
    """Return None to proceed, or a refusal string the agent must surface verbatim."""
    if not resolved.low_confidence:
        return None
    if answer_kind in {"distance", "contains", "nearest"}:
        return ("The reference frame for these coordinates could not be established "
                f"({resolved.rationale}); a measured answer would be unreliable.")
    return None                                    # descriptive answers may proceed
```

<figure class="diagram">
<svg viewBox="16 20 768 252" role="img" aria-labelledby="rcr-gate-t rcr-gate-d" xmlns="http://www.w3.org/2000/svg"><title id="rcr-gate-t">Decision path from an ambiguous mention to a guarded answer</title><desc id="rcr-gate-d">Retrieved candidates pass through a validation gate. A surviving candidate yields a confident frame and an unguarded answer; an empty result falls back to WGS84, is flagged, and measured answers are refused rather than estimated.</desc><rect x="16" y="20" width="768" height="252" fill="#ffffff"/><rect x="30" y="112" width="150" height="66" rx="8" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><text x="105" y="140" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">candidates</text><text x="105" y="160" fill="#5b6471" font-size="12" text-anchor="middle">top k from index</text><rect x="240" y="112" width="150" height="66" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="315" y="140" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">validation gate</text><text x="315" y="160" fill="#5b6471" font-size="12" text-anchor="middle">constructs · current</text><rect x="470" y="34" width="300" height="72" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="620" y="64" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">survivor — confident frame</text><text x="620" y="86" fill="#5b6471" font-size="12" text-anchor="middle">measured answers proceed normally</text><rect x="470" y="186" width="300" height="72" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="620" y="216" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">no survivor — flagged fallback</text><text x="620" y="238" fill="#5b6471" font-size="12" text-anchor="middle">distance and containment are refused</text><g stroke="#5b6471" stroke-width="2" marker-end="url(#rcr-gate-a)"><line x1="182" y1="145" x2="236" y2="145"/><line x1="392" y1="130" x2="466" y2="82"/><line x1="392" y1="160" x2="466" y2="210"/></g><defs><marker id="rcr-gate-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/></marker></defs></svg>
<figcaption><b>The flag has to change behaviour.</b> Confidence that only reaches a log file is decoration. Here it reaches the answer path, where a fallback frame turns a measured claim into an explicit refusal — the one outcome a user can act on.</figcaption>
</figure>

### 5. Score provenance, not similarity alone

Retrieval similarity measures how much a candidate record *looks like* the mention. Provenance measures how much authority stands behind it. The two are easy to conflate and behave very differently under adversarial input: a document that repeats a place name twenty times scores high on similarity while carrying no authority at all.

A workable scoring scheme separates the two and requires both. Similarity selects the shortlist; provenance decides the confidence. Sources are ranked by how directly they assert a frame — a dataset header that declares its own projection outranks a gazetteer entry, which outranks a prose mention of a grid system, which outranks a co-occurring place name.

```python
SOURCE_AUTHORITY = {                     # how directly the source asserts a frame
    "dataset_header": 1.00,
    "grid_alias": 0.80,
    "gazetteer": 0.65,
    "cooccurrence": 0.40,
}

def score_candidate(cand: CRSCandidate, corroborating_sources: set[str]) -> float:
    """Combine retrieval similarity with source authority and independent corroboration."""
    authority = SOURCE_AUTHORITY.get(cand.source, 0.30)   # unknown source: treated as weak
    distinct = len(corroborating_sources - {cand.source})
    corroboration = min(0.20, 0.10 * distinct)            # capped: two sources, then no more
    score = 0.55 * authority + 0.30 * cand.weight + corroboration
    return round(min(1.0, score), 3)
```

The cap on corroboration is the load-bearing detail. Without it, a catalogue that was mirrored into the index four times manufactures near-certainty from a single assertion. Capping the bonus at two independent sources means a genuinely corroborated candidate still outranks a lone one, while a duplicated one gains almost nothing. Deduplicating by origin before counting matters just as much, and the two together are what keep the confidence number honest enough to gate on.

Note also what the weights imply: even a perfect similarity score with the weakest source class tops out around 0.54, which sits below the default threshold and therefore lands in the flagged band. That is intentional. Co-occurrence is evidence that something is worth investigating, never evidence that a frame has been determined.


### 6. Cache resolutions, and invalidate them on registry change

Frame resolution is expensive relative to how often the answer changes. The same six-figure grid reference in the same corpus resolves identically every time, so the natural design is a cache keyed on the mention text plus the corpus identifier. What makes this more than an optimisation is consistency: a cached resolution guarantees that the same mention resolves the same way across two queries an hour apart, which a live index cannot promise while it is being rebuilt.

```python
def cached_resolve(mention: str, corpus_id: str, cache, resolver) -> ResolvedCRS:
    """Resolve once per (mention, corpus); never let a cache miss become an error."""
    key = f"crs:{corpus_id}:{registry_version()}:{mention.strip().lower()}"
    try:
        hit = cache.get(key)
        if hit is not None:
            return ResolvedCRS(**hit)
    except Exception as exc:                       # cache outage must not fail the query
        log.warning("cache read failed for %s: %s", key, exc)
    resolved = resolver(mention)
    try:
        if not resolved.low_confidence:            # never cache a guess
            cache.set(key, resolved.__dict__, ttl=30 * 24 * 3600)
    except Exception as exc:
        log.warning("cache write failed for %s: %s", key, exc)
    return resolved
```

Two details in that key are doing real work. Including the registry version means a datum registry update invalidates every cached answer automatically rather than serving stale codes for a month. Refusing to cache a flagged fallback means a transient index outage does not freeze a bad answer into place for the whole time-to-live — the next query retries and, once the index is healthy, gets the real determination.

## Failure Modes & Root Causes

**Plausible-but-wrong code.** Retrieval surfaces a neighbouring region's grid whose coordinate ranges overlap the true one. Root cause: coordinate magnitude alone is ambiguous, as the opening figure shows. Mitigation: require provenance agreement, not just an envelope hit, before granting high confidence.

**Deprecated datum codes.** Legacy catalogues store superseded codes that still construct but shift positions by tens of metres. Root cause: stale index content. Mitigation: the `is_deprecated` gate in step 2, plus periodic re-indexing against the current registry. This one is insidious because every automated check passes — the geometry is valid, the transform succeeds, and the answer is quietly displaced by the datum shift.

**Silent WGS84 assumption.** The most damaging mode — the frame is guessed as 4326 and no flag is raised. Root cause: treating "no evidence" as "geographic". Mitigation: the fallback in step 3 is always tagged `low_confidence=True`, and step 4 makes that tag consequential.

**Confidence inflation from duplicated sources.** Five retrieved documents that all derive from one upstream catalogue look like five independent votes. Root cause: scoring on hit count rather than on distinct provenance. Mitigation: deduplicate candidates by source identity before weighing them, and cap the contribution of any single origin.

## Production Validation Protocols

1. **Assert structured returns.** `resolve_crs` must never raise for adversarial mentions; a fuzz test feeds empty strings, emoji, and malformed grid references and asserts a `ResolvedCRS` comes back every time.
2. **Round-trip gate.** For every non-fallback resolution, transform the sample coordinate to EPSG:4326 and back; reject if positional error exceeds one metre.
3. **Confidence monotonicity.** A validated candidate must score strictly above the `0.0` fallback; assert `confidence > 0` whenever `low_confidence is False`.
4. **Fallback-rate indicator.** Alert when the share of flagged fallbacks exceeds a rolling baseline — a spike signals index drift or an outage rather than a change in the documents.
5. **Refusal-path test.** Assert that a low-confidence resolution produces a refusal for distance questions; this is the assertion that catches a regression where the flag stops being read.
6. **Provenance logging.** Persist the rationale and source with each resolution for audit and replay, so a disputed answer can be reconstructed months later.

<figure class="diagram">
<svg viewBox="16 38 714 189" role="img" aria-labelledby="rcr-conf-t rcr-conf-d" xmlns="http://www.w3.org/2000/svg"><title id="rcr-conf-t">What each confidence band permits</title><desc id="rcr-conf-d">Four confidence bands from flagged fallback through weak and corroborated evidence to a validated round trip, each paired with the class of answer the agent is allowed to give.</desc><rect x="16" y="38" width="714" height="189" fill="#ffffff"/><rect x="30" y="52" width="164" height="84" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="204" y="52" width="164" height="84" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="378" y="52" width="164" height="84" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="552" y="52" width="164" height="84" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="112" y="82">0.00</text><text x="286" y="82">0.50 – 0.65</text><text x="460" y="82">0.65 – 0.85</text><text x="634" y="82">0.85 – 1.00</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="112" y="106">flagged fallback</text><text x="286" y="106">one weak source</text><text x="460" y="106">corroborated</text><text x="634" y="106">round trip verified</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="112" y="126">refuse measures</text><text x="286" y="126">hedge and cite</text><text x="460" y="126">answer with source</text><text x="634" y="126">answer plainly</text></g><text x="370" y="182" fill="#1f2937" font-size="13" text-anchor="middle">The band, not the code, decides what the agent is permitted to claim</text><text x="370" y="210" fill="#5b6471" font-size="12" text-anchor="middle">Publish the thresholds with the pipeline — they are policy, not physics</text></svg>
<figcaption><b>Bands, not booleans.</b> A single low-confidence flag forces every uncertain case into the same refusal. Graded bands let a weakly corroborated frame still answer a descriptive question while still refusing to measure with it.</figcaption>
</figure>

The round-trip gate is the one to invest in first. It is a property test rather than an example test — it holds for every coordinate in every frame, so it needs no fixture maintenance, and it fails on precisely the class of error that is otherwise invisible: a code that constructs, validates, and quietly moves the point. Run it on the actual sample coordinate rather than a synthetic one, because a frame's behaviour at the edge of its area of use is where the interesting failures live.

The fallback-rate indicator is the one to alert on. Every other gate answers "is this resolution correct"; the rate answers "is the pipeline still working", which is the question that matters at three in the morning when an upstream catalogue changes its schema and every lookup starts returning nothing. Set the threshold from a fortnight of observed traffic rather than from a round number, and page on the derivative as well as the level — a fallback rate climbing steadily from two percent to six is a more useful signal than one that has sat at five percent since launch.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Can the model be used to rank candidates, even if it does not choose them?</span></summary><p>Yes, and this is often the best use of it. Ranking a fixed, validated candidate list is a bounded task where a wrong answer is still a correct-format answer, and every option has already passed the library checks. What must not happen is the model emitting a code that was never retrieved — enforce that by intersecting whatever the model returns with the candidate set and discarding anything outside it.</p></details>

<details class="faq-item"><summary><span>How large should the knowledge index be?</span></summary><p>Smaller than instinct suggests. The registry holds thousands of codes, but any one corpus draws on a handful of grids, and indexing the full registry mostly adds near-miss candidates that make scoring harder. Start from the codes actually present in your sources, add the national grids of the regions you cover, and grow the index in response to measured fallback spikes rather than in anticipation.</p></details>

<details class="faq-item"><summary><span>What if two candidates validate equally well?</span></summary><p>Return the fallback, flagged, rather than picking one. A tie means the evidence does not distinguish them, and choosing arbitrarily converts an honest ambiguity into a confident error with a fifty percent failure rate. Record both candidates in the rationale so the reviewer can see what the tie was between.</p></details>

<details class="faq-item"><summary><span>How should a resolved frame be recorded so the decision survives review?</span></summary><p>Store four things next to the geometry: the resolved code, the confidence, the rationale string, and the identifier of the source that supplied the winning candidate. That quartet is enough to reconstruct the decision without rerunning the pipeline, which matters because the index will have changed by the time anyone disputes an answer. Storing only the code is the common shortcut and it makes every later argument unresolvable — you can see what the pipeline concluded but not why, and a resolution that cannot be defended is one that has to be redone from scratch.</p></details>

<details class="faq-item"><summary><span>Does this belong at ingestion or at query time?</span></summary><p>At ingestion whenever the corpus is stable, because resolving once and storing the result makes every later query cheaper and consistent. Query-time resolution is for mentions that arrive with the question — a user pasting a coordinate pair — where there is no ingestion moment. Many systems need both, and they should share one implementation so a mention resolves the same way regardless of when it arrived.</p></details>

## Related

- Up to the section overview: [Geospatial RAG Pipelines](/geospatial-rag-pipelines/)
- Foundational concept: [Coordinate Reference System Normalization](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/)
- Technique: [Inferring CRS from Retrieved Spatial Context](/geospatial-rag-pipelines/retrieval-augmented-crs-resolution/inferring-crs-from-retrieved-spatial-context/)
- Technique: [Resolving Ambiguous EPSG Codes from Document Context](/geospatial-rag-pipelines/retrieval-augmented-crs-resolution/resolving-ambiguous-epsg-codes-from-document-context/)
- Peer topic: [Chunk-Boundary Strategies for Spatial Corpora](/geospatial-rag-pipelines/chunk-boundary-strategies-for-spatial-corpora/)
- Related concept: [Fallback Routing for Geospatial Queries](/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/)
