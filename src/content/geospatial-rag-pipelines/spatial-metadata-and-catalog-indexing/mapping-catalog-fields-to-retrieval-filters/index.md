---
title: Mapping Catalog Fields to Retrieval Filters
description: Turn catalog attributes into a small, typed filter vocabulary an agent can use, so dataset selection is expressed as constraints rather than as free-text hope.
slug: mapping-catalog-fields-to-retrieval-filters
type: howto
breadcrumb: Fields to Filters
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Mapping Catalog Fields to Retrieval Filters

A catalog has forty fields and an agent can usefully filter on five. Choosing which five, giving them types and units the agent cannot misuse, and rejecting a filter request that would silently match nothing — that is the interface between a language model and a structured catalog, and it is where most dataset-selection bugs live. This guide builds that interface for [spatial metadata and catalog indexing](/geospatial-rag-pipelines/spatial-metadata-and-catalog-indexing/).

## When to Use This Approach

Build a typed filter vocabulary as soon as an agent is allowed to choose datasets. Before that point a hard-coded query is simpler and safer; after it, free-text filtering against arbitrary fields produces failures that look like missing data.

| Catalog field | Expose as | Why |
|---------------|-----------|-----|
| Spatial extent | Region parameter, always applied | The one filter every query needs |
| Temporal range | Period parameter with an open end | Questions are usually "since", not "between" |
| Resolution | Numeric maximum, in metres | Comparable across sources only in one unit |
| Licence class | Enumerated, not free text | Eligibility is a closed set |
| Processing level | Enumerated, optional | Meaningful within a family, not across |
| Everything else | Not exposed | Unfilterable fields belong in the answer, not the query |

The last row is the discipline that makes this work. Every exposed field is a field the agent can get wrong, and a field that rarely changes an outcome adds risk without adding capability.

<figure class="diagram">
<svg viewBox="16 28 748 198" role="img" aria-labelledby="mcf-narrow-t mcf-narrow-d" xmlns="http://www.w3.org/2000/svg"><title id="mcf-narrow-t">From forty catalog fields to five filter parameters</title><desc id="mcf-narrow-d">Most catalog fields are descriptive and belong in the answer; a small number are discriminative and become typed filter parameters the agent may set.</desc><rect x="16" y="28" width="748" height="198" fill="#ffffff"/><rect x="30" y="56" width="240" height="140" rx="8" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><text x="150" y="86" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">catalog record</text><text x="150" y="112" fill="#5b6471" font-size="12" text-anchor="middle">forty fields, mostly prose</text><text x="150" y="136" fill="#5b6471" font-size="12" text-anchor="middle">titles, abstracts, contacts,</text><text x="150" y="158" fill="#5b6471" font-size="12" text-anchor="middle">keywords, lineage, links</text><rect x="330" y="42" width="200" height="76" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="430" y="70" fill="#1f2937" font-size="12.5" text-anchor="middle">five typed filters</text><text x="430" y="94" fill="#5b6471" font-size="12" text-anchor="middle">the agent may set these</text><rect x="330" y="136" width="200" height="76" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="430" y="164" fill="#1f2937" font-size="12.5" text-anchor="middle">descriptive fields</text><text x="430" y="188" fill="#5b6471" font-size="12" text-anchor="middle">returned, never queried</text><rect x="590" y="88" width="160" height="76" rx="8" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><text x="670" y="116" fill="#1f2937" font-size="12.5" text-anchor="middle">shortlist</text><text x="670" y="140" fill="#5b6471" font-size="12" text-anchor="middle">with caveats attached</text><g stroke="#5b6471" stroke-width="2" marker-end="url(#mcf-narrow-a)"><line x1="272" y1="98" x2="326" y2="86"/><line x1="272" y1="156" x2="326" y2="168"/><line x1="532" y1="98" x2="586" y2="118"/><line x1="532" y1="166" x2="586" y2="146"/></g><defs><marker id="mcf-narrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/></marker></defs></svg>
<figcaption><b>Both halves reach the answer; only one reaches the query.</b> Descriptive fields make the shortlist explicable, which is a different job from making it selective, and conflating the two is what produces forty-parameter tool schemas.</figcaption>
</figure>

## Implementation

The filter object is validated and normalised before it touches the database, and a request that cannot be satisfied is rejected with a reason rather than executed into an empty result.

```python
import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

log = logging.getLogger("catalog_filters")

LICENCE_CLASSES = frozenset({"open", "attribution", "share-alike", "restricted"})
PROCESSING_LEVELS = frozenset({"raw", "calibrated", "analysis-ready", "derived"})

MAX_REGION_SPAN_DEG = 60.0        # beyond this the filter is not narrowing anything


class FilterError(ValueError):
    """The requested filter cannot be satisfied — surface the reason, do not query."""


@dataclass(frozen=True)
class CatalogFilter:
    west: float
    south: float
    east: float
    north: float
    since: Optional[date] = None
    max_resolution_m: Optional[float] = None
    licence_classes: frozenset[str] = frozenset({"open", "attribution"})
    processing_level: Optional[str] = None


def validate(f: CatalogFilter) -> CatalogFilter:
    """Normalise and reject impossible filters before they reach the catalog."""
    if not (f.west < f.east and f.south < f.north):
        raise FilterError("region has zero or negative extent")
    if (f.east - f.west) > MAX_REGION_SPAN_DEG or (f.north - f.south) > MAX_REGION_SPAN_DEG:
        log.info("region spans %.1f degrees — treating as unfiltered",
                 max(f.east - f.west, f.north - f.south))
    if f.max_resolution_m is not None and f.max_resolution_m <= 0:
        raise FilterError("max_resolution_m must be a positive number of metres")
    unknown = f.licence_classes - LICENCE_CLASSES
    if unknown:
        raise FilterError(f"unknown licence class(es): {sorted(unknown)}")
    if not f.licence_classes:
        raise FilterError("at least one licence class must be permitted")
    if f.processing_level is not None and f.processing_level not in PROCESSING_LEVELS:
        raise FilterError(f"unknown processing level: {f.processing_level!r}")
    if f.since is not None and f.since.year < 1800:
        raise FilterError("since date predates any plausible catalog record")
    return f


def to_sql_params(f: CatalogFilter) -> dict:
    """Bind parameters for the catalog query; never string-format a filter into SQL."""
    f = validate(f)
    return {
        "west": f.west, "south": f.south, "east": f.east, "north": f.north,
        "since": f.since,
        "max_res": f.max_resolution_m,
        "licences": tuple(sorted(f.licence_classes)),
        "level": f.processing_level,
    }
```

Raising rather than returning an empty result is the important choice. An agent that receives "no datasets match" will report there is no data; an agent that receives "max_resolution_m must be positive" can correct itself and retry. The two responses lead to completely different conversations, and only one of them is honest about what went wrong.

The query then binds those parameters into the index-aware form, with every optional filter written so a null value means "do not constrain":

```sql
SELECT collection_id, title, resolution_m, licence, version
FROM   catalog_collections
WHERE  geom && ST_MakeEnvelope(:west, :south, :east, :north, 4326)
  AND  ST_Intersects(geom, ST_MakeEnvelope(:west, :south, :east, :north, 4326))
  AND  (:since IS NULL OR upper(period) >= :since)
  AND  (:max_res IS NULL OR resolution_m IS NULL OR resolution_m <= :max_res)
  AND  licence_class = ANY(:licences)
  AND  (:level IS NULL OR processing_level = :level)
ORDER  BY resolution_m NULLS LAST
LIMIT  20;
```

Note that the resolution predicate admits records with an undeclared resolution rather than excluding them. That is a policy decision written into the query: an undeclared value is unknown, not disqualifying, and the fitness scoring downstream penalises it instead.

<figure class="diagram">
<svg viewBox="16 42 728 192" role="img" aria-labelledby="mcf-null-t mcf-null-d" xmlns="http://www.w3.org/2000/svg"><title id="mcf-null-t">Three ways to treat an undeclared field value</title><desc id="mcf-null-d">An undeclared resolution can be excluded, admitted freely, or admitted with a scoring penalty; only the third preserves both recall and honesty about the uncertainty.</desc><rect x="16" y="42" width="728" height="192" fill="#ffffff"/><rect x="30" y="56" width="220" height="120" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="270" y="56" width="220" height="120" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="510" y="56" width="220" height="120" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="140" y="86">exclude</text><text x="380" y="86">admit freely</text><text x="620" y="86">admit and penalise</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="140" y="114">loses usable datasets</text><text x="140" y="138">silently, for a missing</text><text x="140" y="158">metadata field</text><text x="380" y="114">ranks unknown data</text><text x="380" y="138">alongside verified data</text><text x="380" y="158">no signal to the reader</text><text x="620" y="114">keeps it reachable</text><text x="620" y="138">ranks it below known</text><text x="620" y="158">carries a caveat</text></g><text x="380" y="216" fill="#1f2937" font-size="13" text-anchor="middle">Missing is a third state — treating it as either extreme loses information</text></svg>
<figcaption><b>Nulls are where filter design usually goes wrong.</b> Excluding them is invisible in testing, because test fixtures tend to be complete; admitting them freely is invisible in production, because nothing marks the answer as less certain.</figcaption>
</figure>

## Validation & Testing

```python
from datetime import date


def test_impossible_filter_raises_with_a_reason():
    try:
        validate(CatalogFilter(-3.0, 55.0, -3.5, 56.0))     # east west of west
    except FilterError as exc:
        assert "extent" in str(exc)
        return
    raise AssertionError("an inverted region must be rejected")


def test_unknown_licence_class_is_rejected_not_ignored():
    try:
        validate(CatalogFilter(-3.5, 55.0, -3.0, 56.0,
                               licence_classes=frozenset({"public-domain-ish"})))
    except FilterError:
        return
    raise AssertionError("an unknown licence class must not silently pass")


def test_undeclared_resolution_survives_the_filter(conn):
    rows = run_catalog_query(conn, to_sql_params(
        CatalogFilter(-3.5, 55.0, -3.0, 56.0, max_resolution_m=5.0)))
    assert any(r["resolution_m"] is None for r in rows)


def test_defaults_exclude_restricted_licences():
    f = CatalogFilter(-3.5, 55.0, -3.0, 56.0)
    assert "restricted" not in f.licence_classes
```

The last test guards a default rather than a behaviour, which is exactly why it is worth writing: defaults are changed casually, and a default that quietly starts admitting restricted datasets is a compliance problem discovered by someone outside the team.

## Gotchas & Edge Cases

**Units left implicit.** A resolution parameter that accepts "30" without a unit will receive feet, arcseconds and metres from different callers. Name the parameter with its unit, as `max_resolution_m` does, and reject values whose magnitude is implausible for that unit.

**Enumerations that grow silently.** A licence class added upstream and not added to the permitted set is rejected as unknown, which is safe, and it will present as "no datasets available". Log unknown values distinctly from invalid ones so the difference is visible.

**Filters formatted into SQL.** Building the query as a string invites both injection and subtle type coercion bugs. Bind parameters, always, and let the driver handle the null semantics rather than emitting different SQL per combination.

<figure class="diagram">
<svg viewBox="16 38 728 158" role="img" aria-labelledby="mcf-err-t mcf-err-d" xmlns="http://www.w3.org/2000/svg"><title id="mcf-err-t">Two responses to an unsatisfiable filter</title><desc id="mcf-err-d">An empty result set leads an agent to report that no data exists, while a rejection with a reason lets it correct the parameter and retry.</desc><rect x="16" y="38" width="728" height="158" fill="#ffffff"/><rect x="30" y="52" width="330" height="130" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="195" y="82" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">empty result set</text><text x="195" y="112" fill="#5b6471" font-size="12" text-anchor="middle">agent concludes: no data exists</text><text x="195" y="140" fill="#5b6471" font-size="12" text-anchor="middle">user hears a confident absence</text><text x="195" y="166" fill="#5b6471" font-size="12" text-anchor="middle">nothing to correct</text><rect x="400" y="52" width="330" height="130" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="565" y="82" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">rejection with a reason</text><text x="565" y="112" fill="#5b6471" font-size="12" text-anchor="middle">agent sees the invalid parameter</text><text x="565" y="140" fill="#5b6471" font-size="12" text-anchor="middle">corrects and retries</text><text x="565" y="166" fill="#5b6471" font-size="12" text-anchor="middle">or explains the real constraint</text></svg>
<figcaption><b>An empty set is an answer; a rejection is a conversation.</b> The distinction costs one exception type and removes the most common way a dataset-selection layer misleads the person reading its output.</figcaption>
</figure>

**Region parameters accepted in the wrong axis order.** A caller passing latitude before longitude produces a region that is either empty or somewhere unexpected. Validate that the values fall in their respective ranges as well as being ordered, since ordering alone does not catch a swap near the equator.

**A filter vocabulary that drifts from the catalog.** Fields get renamed upstream; the filter layer keeps referring to the old name and silently constrains nothing. Assert in a schema test that every filter parameter maps to a column that exists, and run it against the real catalog rather than a fixture.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should the agent be able to set the region, or should the system derive it?</span></summary><p>Derive it by default, allow an override. The region usually follows from the question and deriving it consistently is more reliable than asking a model to construct a bounding box, which it will do plausibly and occasionally in the wrong hemisphere. Keep the override for the case where the user has stated an area explicitly, and validate an overridden region exactly as hard as a derived one.</p></details>

<details class="faq-item"><summary><span>How should a filter that returns nothing be reported?</span></summary><p>With the filter itself attached. "No datasets match" is unactionable; "no datasets covering this region, at 5 m or finer, under an open licence" tells the reader which constraint to relax and lets the agent offer to relax it. This costs one extra field in the response and turns the most common dead end into a conversation.</p></details>

<details class="faq-item"><summary><span>Is it worth exposing keyword search over catalog titles as a filter?</span></summary><p>As a ranking signal rather than a filter, yes. Titles and abstracts are exactly where semantic and lexical search belong, and constraining on them is brittle — a collection titled with a sensor name will not match a query about vegetation however well it answers it. Filter on the structured fields, rank on the prose, which is the same division this whole section is built on.</p></details>

<details class="faq-item"><summary><span>What about fields that only apply to some collection types?</span></summary><p>Expose them as optional and ignore them for collections where they do not apply, rather than partitioning the vocabulary by type. A processing-level filter that silently excludes every vector dataset because vectors have no processing level is the failure mode to avoid, and the null-tolerant predicate form above is what prevents it.</p></details>

Keep the filter vocabulary in one module that both the agent's tool schema and the database query are generated from. Two hand-maintained copies drift within a release or two, and the drift presents as a filter that the agent believes it set and the query quietly ignored — which is indistinguishable, from the outside, from a catalog that lacks the data.

<details class="faq-item"><summary><span>Should filter defaults differ between interactive and batch use?</span></summary><p>Only in how strictly they fail. The same defaults should apply in both, so a result reproduced in batch matches what a user saw interactively, but a batch job can reasonably treat a rejected filter as a hard stop while an interactive agent retries with a corrected parameter. Keep one validator and let the caller decide what to do with the exception.</p></details>

## Related

- Up to the parent topic: [Spatial Metadata and Catalog Indexing](/geospatial-rag-pipelines/spatial-metadata-and-catalog-indexing/)
- [Indexing Catalog Collections for Agent Retrieval](/geospatial-rag-pipelines/spatial-metadata-and-catalog-indexing/indexing-stac-collections-for-agent-retrieval/)
- Concept: [Spatial Function-Calling Schemas](/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/)
- Technique: [Validating Tool Arguments with GeoJSON Schema](/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/validating-tool-arguments-with-geojson-schema/)
