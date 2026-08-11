---
title: Spatial Metadata and Catalog Indexing
description: Turn dataset catalogs into a retrieval surface an agent can search — indexing extent, time, resolution and licence so the right dataset is found before any pixel is read.
slug: spatial-metadata-and-catalog-indexing
type: topic
breadcrumb: Metadata and Catalog Indexing
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Spatial Metadata and Catalog Indexing

Before an agent can answer a question about the ground, it has to find the dataset that describes the ground. That is a catalog problem, not a retrieval problem, and treating it as retrieval is why so many spatial agents confidently answer from whatever imagery happened to embed nearby. A catalog index makes the search over datasets explicit: extent, time, resolution, licence and provenance become fields an agent can filter on, and the choice of dataset becomes a decision that can be inspected.

This topic belongs to [geospatial RAG pipelines](/geospatial-rag-pipelines/) and sits one level above the chunk-oriented work in [chunk-boundary strategies for spatial corpora](/geospatial-rag-pipelines/chunk-boundary-strategies-for-spatial-corpora/). Where that topic asks "which passage answers this", this one asks "which collection should we be reading at all" — and getting that wrong makes every downstream stage irrelevant, however well it is engineered.

<figure class="diagram">
<svg viewBox="12 32 752 226" role="img" aria-labelledby="smc-two-t smc-two-d" xmlns="http://www.w3.org/2000/svg"><title id="smc-two-t">Two searches, not one: catalog selection before content retrieval</title><desc id="smc-two-d">A question first searches a catalog of datasets by extent, time and resolution to choose a collection, and only then searches within that collection for the passages or tiles that answer it.</desc><rect x="12" y="32" width="752" height="226" fill="#ffffff"/><rect x="30" y="110" width="140" height="70" rx="8" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><text x="100" y="140" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">question</text><text x="100" y="160" fill="#5b6471" font-size="12" text-anchor="middle">place · time · topic</text><rect x="220" y="46" width="240" height="86" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="340" y="76" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">search 1 — the catalog</text><text x="340" y="98" fill="#5b6471" font-size="12" text-anchor="middle">which collections cover this</text><text x="340" y="118" fill="#5b6471" font-size="12" text-anchor="middle">extent, period and resolution?</text><rect x="220" y="158" width="240" height="86" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="340" y="188" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">search 2 — the content</text><text x="340" y="210" fill="#5b6471" font-size="12" text-anchor="middle">which items inside the chosen</text><text x="340" y="230" fill="#5b6471" font-size="12" text-anchor="middle">collection answer the question?</text><rect x="510" y="110" width="240" height="70" rx="8" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><text x="630" y="140" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">answer with provenance</text><text x="630" y="162" fill="#5b6471" font-size="12" text-anchor="middle">names the dataset it relied on</text><g stroke="#5b6471" stroke-width="2" marker-end="url(#smc-two-a)"><line x1="172" y1="132" x2="216" y2="100"/><line x1="340" y1="134" x2="340" y2="154"/><line x1="462" y1="196" x2="506" y2="160"/></g><defs><marker id="smc-two-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/></marker></defs></svg>
<figcaption><b>Two searches with different shapes.</b> The first is a structured query over a few thousand catalog records and should be nearly exact. The second is the fuzzy semantic search everyone builds. Collapsing them into one is what produces answers drawn from a dataset nobody would have chosen.</figcaption>
</figure>

## Foundational Principles

**A catalog record is structured, so search it structurally.** Extent, temporal range, resolution and licence are not prose and should not be embedded as prose. They are filters, and a catalog search that begins with them narrows thousands of collections to a handful before any similarity is computed.

**Fitness for purpose is a computable property.** "Is this dataset good enough for this question" is usually answerable from metadata alone: a 30-metre land-cover product cannot answer a question about a single building, and a survey from 2011 cannot answer a question about last year's construction. Encode those rules rather than hoping the model infers them.

**Provenance must survive to the answer.** Every claim an agent makes should be traceable to a catalog record with an identifier, a version and a licence. Without that chain, a correct answer and a lucky one are indistinguishable, and a licence violation is invisible until someone else notices it.

## Step-by-Step Implementation Pipeline

### 1. Normalise catalog records into one internal shape

Catalogs arrive in several dialects — collection manifests, service capability documents, hand-maintained spreadsheets. Normalise them once, on ingestion, into a single record type, and reject records that cannot supply the fields the later stages depend on.

```python
from dataclasses import dataclass
from datetime import date
from typing import Optional
import logging

log = logging.getLogger("catalog")

@dataclass(frozen=True)
class CatalogRecord:
    collection_id: str
    title: str
    bbox: tuple[float, float, float, float]   # west, south, east, north in EPSG:4326
    start: Optional[date]
    end: Optional[date]
    resolution_m: Optional[float]
    licence: str
    version: str

def normalise(raw: dict) -> Optional[CatalogRecord]:
    """Return a record, or None when the source cannot supply the mandatory fields."""
    try:
        bbox = tuple(float(v) for v in raw["extent"]["spatial"][:4])
        if len(bbox) != 4 or bbox[0] > bbox[2] or bbox[1] > bbox[3]:
            log.warning("rejecting %s — degenerate extent %s", raw.get("id"), bbox)
            return None
        return CatalogRecord(
            collection_id=str(raw["id"]),
            title=str(raw.get("title") or raw["id"]),
            bbox=bbox,
            start=_as_date(raw.get("extent", {}).get("temporal", [None, None])[0]),
            end=_as_date(raw.get("extent", {}).get("temporal", [None, None])[1]),
            resolution_m=_as_float(raw.get("gsd")),
            licence=str(raw.get("license") or "unknown"),
            version=str(raw.get("version") or "0"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        log.warning("rejecting malformed catalog record %r: %s", raw.get("id"), exc)
        return None
```

Rejecting a record is the right response to missing mandatory fields, and it must be loud. A catalog that silently drops a fifth of its sources looks healthy from the inside and produces the mystifying failure where an agent insists no data exists for a region that is obviously well covered.

### 2. Index the structured fields for real filtering

The catalog is small — thousands of records, not millions — which means an ordinary relational table with a spatial index is exactly the right tool. Resist the pull toward putting it in the vector store just because the rest of the pipeline lives there.

```sql
CREATE TABLE catalog_collections (
    collection_id text PRIMARY KEY,
    title         text NOT NULL,
    geom          geometry(Polygon, 4326) NOT NULL,
    period        daterange,
    resolution_m  double precision,
    licence       text NOT NULL,
    version       text NOT NULL
);

CREATE INDEX catalog_geom_idx   ON catalog_collections USING gist (geom);
CREATE INDEX catalog_period_idx ON catalog_collections USING gist (period);

-- Index-aware selection: bounding box first, exact predicate second.
SELECT collection_id, title, resolution_m, licence
FROM   catalog_collections
WHERE  geom && ST_MakeEnvelope(:w, :s, :e, :n, 4326)
  AND  ST_Intersects(geom, ST_MakeEnvelope(:w, :s, :e, :n, 4326))
  AND  period && daterange(:from_date, :to_date)
  AND  (resolution_m IS NULL OR resolution_m <= :max_resolution_m)
ORDER  BY resolution_m NULLS LAST
LIMIT  20;
```

The `NULLS LAST` ordering encodes a small policy: a dataset that does not declare its resolution is not disqualified, but it ranks below every dataset that does. That is usually right, and it is the kind of judgement that belongs in the query rather than in a comment.

### 3. Score fitness, and make the rejections explicit

Filtering answers "could this dataset apply". Fitness answers "should it". The second needs the question's requirements, and its output should include why each candidate lost, because that explanation is what a user needs when the answer is "no suitable data".

```python
@dataclass(frozen=True)
class Fitness:
    record: CatalogRecord
    score: float
    reasons: tuple[str, ...]        # why it scored as it did — kept for the answer

def assess(record: CatalogRecord, need_resolution_m: float,
           need_year: int) -> Fitness:
    reasons, score = [], 1.0
    if record.resolution_m is None:
        score *= 0.7
        reasons.append("resolution not declared")
    elif record.resolution_m > need_resolution_m:
        score *= max(0.0, need_resolution_m / record.resolution_m)
        reasons.append(f"coarser than needed ({record.resolution_m:g} m)")
    if record.end is not None and record.end.year < need_year:
        gap = need_year - record.end.year
        score *= 0.85 ** gap
        reasons.append(f"ends {gap} year(s) before the period asked about")
    if record.licence.lower() in {"unknown", "restricted"}:
        score = 0.0
        reasons.append("licence does not permit use")
    return Fitness(record, round(score, 3), tuple(reasons))
```

Note that the licence check zeroes the score rather than scaling it. Some constraints are not trade-offs: a dataset that cannot legally be used is not a slightly worse option, and expressing that as a multiplier eventually lets a very high resolution overcome it.

### 4. Present the choice, not just the result

The agent should receive a shortlist with scores and reasons, and should name the chosen collection in its answer. This is what turns dataset selection from a hidden step into a reviewable one. The field-level mechanics of turning catalog attributes into agent-visible filters are set out in [mapping catalog fields to retrieval filters](/geospatial-rag-pipelines/spatial-metadata-and-catalog-indexing/mapping-catalog-fields-to-retrieval-filters/).

```python
def shortlist(candidates: list[Fitness], k: int = 3) -> list[dict]:
    """Return a small, explained shortlist; never an unexplained single winner."""
    usable = [c for c in candidates if c.score > 0.0]
    if not usable:
        return []                                   # caller must say "no suitable dataset"
    top = sorted(usable, key=lambda c: c.score, reverse=True)[:k]
    return [{
        "collection_id": c.record.collection_id,
        "title": c.record.title,
        "version": c.record.version,
        "licence": c.record.licence,
        "score": c.score,
        "caveats": list(c.reasons),
    } for c in top]
```

### 5. Keep the catalog fresh, and record when it was refreshed

Catalogs go stale in two directions: new collections appear, and existing ones extend their temporal coverage. Both are invisible until someone asks about a recent date and is told there is no coverage. A scheduled refresh with a recorded timestamp turns that from a mystery into a monitored number.

```python
def refresh_catalog(source_iter, upsert, now) -> dict:
    """Refresh from source; count outcomes so staleness is measurable."""
    added = updated = rejected = 0
    for raw in source_iter:
        rec = normalise(raw)
        if rec is None:
            rejected += 1
            continue
        created = upsert(rec, refreshed_at=now)
        added += int(created)
        updated += int(not created)
    stats = {"added": added, "updated": updated, "rejected": rejected, "refreshed_at": now}
    log.info("catalog refresh: %s", stats)
    return stats
```

The rejection count is the number to alert on. A source that silently changes its schema shows up here as a step change in rejections, days or weeks before anyone notices the missing coverage in an answer. Indexing the individual items within a collection, once the collection has been chosen, is covered in [indexing catalog collections for agent retrieval](/geospatial-rag-pipelines/spatial-metadata-and-catalog-indexing/indexing-stac-collections-for-agent-retrieval/).

<figure class="diagram">
<svg viewBox="16 7 768 231" role="img" aria-labelledby="smc-fit-t smc-fit-d" xmlns="http://www.w3.org/2000/svg"><title id="smc-fit-t">Fitness assessment for one question against four collections</title><desc id="smc-fit-d">Four candidate collections scored against a question needing five-metre resolution and recent coverage. Two are ruled out outright by licence and resolution; two survive with recorded caveats.</desc><rect x="16" y="7" width="768" height="231" fill="#ffffff"/><text x="400" y="32" fill="#5b6471" font-size="13" text-anchor="middle">Question needs 5 m resolution, coverage through last year, open licence</text><rect x="30" y="56" width="360" height="76" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="52" y="84" fill="#1f2937" font-size="13" font-weight="600">aerial survey 2 m — score 1.00</text><text x="52" y="110" fill="#5b6471" font-size="12">covers the extent, current, open licence</text><rect x="410" y="56" width="360" height="76" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="432" y="84" fill="#1f2937" font-size="13" font-weight="600">satellite 10 m — score 0.43</text><text x="432" y="110" fill="#5b6471" font-size="12">usable but coarser than the question needs</text><rect x="30" y="148" width="360" height="76" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="52" y="176" fill="#1f2937" font-size="13" font-weight="600">land cover 30 m — score 0.10</text><text x="52" y="202" fill="#5b6471" font-size="12">far too coarse and three years out of date</text><rect x="410" y="148" width="360" height="76" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="432" y="176" fill="#1f2937" font-size="13" font-weight="600">commercial 0.5 m — score 0.00</text><text x="432" y="202" fill="#5b6471" font-size="12">best data available, licence forbids this use</text></svg>
<figcaption><b>The best dataset is not always the eligible one.</b> The half-metre imagery would answer the question perfectly and cannot be used. Encoding that as a zero rather than a penalty is what stops a sufficiently demanding question from quietly overriding a licence.</figcaption>
</figure>

### 6. Resolve the question's requirements before scoring anything

Fitness scoring needs a target resolution and a target period, and those come from the question. Extracting them is a small, well-bounded task that is easy to get wrong in a way that poisons every subsequent step: a question about "this building" that resolves to a hundred-metre requirement will happily accept land-cover data.

Three signals do most of the work. The **subject scale** — building, parcel, street, neighbourhood, region — maps directly onto a resolution requirement. The **temporal language** — "now", "last year", "in the 1990s", "since the flood" — maps onto a period. And the **verb** distinguishes measurement from description: "how far", "how much" and "how many" demand data good enough to compute with, while "what is there" tolerates coarser sources.

```python
SCALE_RESOLUTION_M = {
    "building": 1.0, "parcel": 2.0, "street": 5.0,
    "neighbourhood": 20.0, "region": 100.0,
}

def requirements(subject_scale: str, year: int, verb_class: str) -> dict:
    """Turn question features into explicit dataset requirements."""
    base = SCALE_RESOLUTION_M.get(subject_scale)
    if base is None:
        base = 20.0                                # unknown scale: assume neighbourhood
    if verb_class == "measure":
        base = base / 2.0                          # measuring needs finer data than describing
    return {"max_resolution_m": base, "need_year": year, "strict": verb_class == "measure"}
```

Halving the requirement for measurement questions is a defensible default rather than a law, and it is worth stating in the pipeline's documentation because it changes which datasets are eligible. A team that finds it too strict should loosen it deliberately, in one place, rather than discovering that different code paths disagree.

When the scale cannot be determined, assume the middle of the range rather than the extremes. Assuming the finest scale rules out every usable dataset and produces a spurious "no suitable data"; assuming the coarsest admits everything and produces a confident answer from unsuitable sources. The middle is wrong less badly in both directions, and the caveat mechanism carries the uncertainty forward.

### 7. Cache the catalog decision per question shape

Catalog selection is stable: the same region, period and scale will select the same collection until the catalog itself changes. Caching that decision, keyed on the requirement tuple plus the catalog refresh timestamp, removes a database round trip from the hot path and — more valuably — guarantees that two identical questions asked minutes apart choose the same dataset.

```python
def cached_shortlist(reqs: dict, bbox, catalog_version: str, cache, compute):
    """Stable selection per (requirements, extent, catalog version)."""
    key = f"cat:{catalog_version}:{round(bbox[0],3)},{round(bbox[1],3)}," \
          f"{round(bbox[2],3)},{round(bbox[3],3)}:{reqs['max_resolution_m']}:{reqs['need_year']}"
    try:
        hit = cache.get(key)
        if hit is not None:
            return hit
    except Exception as exc:
        log.warning("catalog cache read failed: %s", exc)
    result = compute(reqs, bbox)
    try:
        if result:                                 # never cache an empty shortlist
            cache.set(key, result, ttl=6 * 3600)
    except Exception as exc:
        log.warning("catalog cache write failed: %s", exc)
    return result
```

Rounding the extent into the key is what makes the cache useful rather than a per-query miss. Three decimal places is roughly a hundred metres, which is far finer than the difference between two collections and coarse enough that the same neighbourhood shares a key. Refusing to cache an empty shortlist means a transient catalog outage cannot freeze "no data available" into place for six hours.

## Failure Modes & Root Causes

**Silent dataset substitution.** The agent answers from whatever collection embedded nearest, which may be a different sensor, era or resolution than the question needs. Root cause: no catalog stage at all — one flat retrieval over everything. Mitigation: the two-search structure in the opening figure.

**Coverage that exists on paper only.** A collection's declared extent spans a country while its actual items cover three cities. Root cause: trusting collection-level extent without item-level verification. Mitigation: verify coverage at the item level for the specific region before committing to a collection, and record declared and observed extent separately.

**Licence drift.** A collection's terms change between ingestion and use. Root cause: licence captured once at ingestion and never refreshed. Mitigation: treat licence as a refreshed field, and re-check it at answer time for anything published externally.

**Resolution mismatch presented as certainty.** A question about a single building answered from 30-metre data, with no hedge. Root cause: fitness reasons computed and then discarded. Mitigation: carry the caveats into the answer, as step 4 does — the shortlist entry and the hedge are the same data.

## Production Validation Protocols

1. **Mandatory-field gate.** Assert every indexed record has an extent, a licence and a version; reject at ingestion rather than filtering at query time.
2. **Rejection-rate alert.** Track normalisation rejections per source and alert on a step change — the earliest signal of an upstream schema change.
3. **Extent sanity test.** Assert no indexed extent spans the whole world unless the collection genuinely is global; a default extent is a common data-entry artefact.
4. **Licence enforcement test.** Assert a restricted collection never appears in a shortlist, with a fixture that would otherwise score highest.
5. **Freshness indicator.** Publish the age of the most recent refresh and alert past a threshold; staleness is a silent failure with a loud symptom much later.
6. **Explained-choice assertion.** Assert every answer that used a dataset names its collection identifier and version; an unattributed answer fails the build.

<figure class="diagram">
<svg viewBox="16 38 728 178" role="img" aria-labelledby="smc-rec-t smc-rec-d" xmlns="http://www.w3.org/2000/svg"><title id="smc-rec-t">Fields of a normalised catalog record and what reads them</title><desc id="smc-rec-d">Six catalog fields grouped by consumer: the spatial filter reads extent, the temporal filter reads the period, fitness scoring reads resolution and licence, and the answer layer reads identifier and version.</desc><rect x="16" y="38" width="728" height="178" fill="#ffffff"/><rect x="30" y="52" width="216" height="150" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="138" y="80" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">read by the filter</text><text x="138" y="112" fill="#5b6471" font-size="12" text-anchor="middle">bbox — spatial index</text><text x="138" y="140" fill="#5b6471" font-size="12" text-anchor="middle">period — temporal index</text><text x="138" y="172" fill="#5b6471" font-size="12" text-anchor="middle">narrows thousands to tens</text><rect x="272" y="52" width="216" height="150" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="380" y="80" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">read by scoring</text><text x="380" y="112" fill="#5b6471" font-size="12" text-anchor="middle">resolution — fitness</text><text x="380" y="140" fill="#5b6471" font-size="12" text-anchor="middle">licence — eligibility</text><text x="380" y="172" fill="#5b6471" font-size="12" text-anchor="middle">ranks tens to three</text><rect x="514" y="52" width="216" height="150" rx="8" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><text x="622" y="80" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">read by the answer</text><text x="622" y="112" fill="#5b6471" font-size="12" text-anchor="middle">collection id — citation</text><text x="622" y="140" fill="#5b6471" font-size="12" text-anchor="middle">version — reproducibility</text><text x="622" y="172" fill="#5b6471" font-size="12" text-anchor="middle">makes the claim checkable</text></svg>
<figcaption><b>Every field has a consumer.</b> A catalog schema justified this way stays small. Fields nothing reads are the ones that rot, and a rotted field that looks authoritative is worse than an absent one.</figcaption>
</figure>

Of these, the explained-choice assertion is the one that changes team behaviour rather than merely catching bugs. Once every answer must name a collection and version, the catalog stops being infrastructure that someone maintains occasionally and becomes part of the answer surface, which is where it belongs. It also makes the "no suitable dataset" outcome respectable: an agent that says which four collections it considered and why each failed is giving a far more useful answer than one that quietly produces a plausible number from a source nobody would have sanctioned.

The freshness indicator deserves a specific treatment because staleness in a catalog behaves unlike staleness elsewhere. A stale document index returns slightly out-of-date passages; a stale catalog returns confident answers about a region from a collection that has since been superseded by a better one. The failure is not that the answer is wrong, but that a better answer existed and was never considered.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should catalog records be embedded as vectors at all?</span></summary><p>Their prose descriptions, yes — that is how a question about "vegetation health" finds a collection titled with a sensor name. Their structured fields, no. Embedding an extent produces a vector that is near other extents with similar numbers, which is meaningless. The workable design embeds the title and abstract for the semantic half and keeps extent, period, resolution and licence as real filters over the same table.</p></details>

<details class="faq-item"><summary><span>How do I handle collections with no declared resolution?</span></summary><p>Rank them below everything that declares one, and say so in the caveats, which is what the scoring function above does. Inferring a resolution from the data would be better but is rarely worth the pipeline complexity for a field that most well-maintained catalogs supply. What you must not do is treat a missing value as zero or as infinite; both turn an unknown into a confident claim in opposite directions.</p></details>

<details class="faq-item"><summary><span>Does the agent need to see the whole shortlist, or just the winner?</span></summary><p>The shortlist. A single winner gives the model nothing to reason about when the top choice has caveats — it cannot say "the best available data is coarser than your question needs" if it never saw the alternatives. Three entries with scores and caveats is usually enough, and it also makes the selection reviewable when the answer is disputed.</p></details>

<details class="faq-item"><summary><span>Where should item-level records live relative to collection-level ones?</span></summary><p>In a separate table with a foreign key, and usually with a different index strategy: collections are thousands of records with large extents, items are millions with small ones. Mixing them makes the spatial index serve two very different distributions badly. Query them in sequence — collection first, items within the chosen collection second — which is the same two-search shape the whole topic is built on.</p></details>

## Related

- Up to the section overview: [Geospatial RAG Pipelines](/geospatial-rag-pipelines/)
- Technique: [Indexing Catalog Collections for Agent Retrieval](/geospatial-rag-pipelines/spatial-metadata-and-catalog-indexing/indexing-stac-collections-for-agent-retrieval/)
- Technique: [Mapping Catalog Fields to Retrieval Filters](/geospatial-rag-pipelines/spatial-metadata-and-catalog-indexing/mapping-catalog-fields-to-retrieval-filters/)
- Peer topic: [Spatial Context Retrieval and Reranking](/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/)
- Peer topic: [Chunk-Boundary Strategies for Spatial Corpora](/geospatial-rag-pipelines/chunk-boundary-strategies-for-spatial-corpora/)
- Concept: [Vector-Raster Hybrid Processing](/spatial-llm-architecture-core-concepts/vector-raster-hybrid-processing/)
