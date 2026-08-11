---
title: Resolving Ambiguous EPSG Codes from Document Context
description: Disambiguate a bare projection code using the document around it — zone hints, datum era, units and stated accuracy — then decide what the agent may claim from the result.
slug: resolving-ambiguous-epsg-codes-from-document-context
type: howto
breadcrumb: Ambiguous EPSG Codes
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Resolving Ambiguous EPSG Codes from Document Context

Some documents declare a frame and still leave you guessing. "UTM zone 31" names a projection and a zone but not a datum, and the codes that match it place the same point up to two hundred metres apart. "OSGB" names a grid whose historic and current realisations differ. This guide handles the case where a declaration exists but underdetermines the code, and decides what the agent may safely claim afterwards — the consequential half of [retrieval-augmented CRS resolution](/geospatial-rag-pipelines/retrieval-augmented-crs-resolution/).

## When to Use This Approach

Use it when a frame is named but not coded, or coded ambiguously, and there is surrounding text to read. It is distinct from inference over retrieved neighbours: here the evidence is in the document itself, which makes it stronger and narrower.

| Declaration found | Ambiguity | Resolve from |
|-------------------|-----------|--------------|
| Projection and zone, no datum | Several datums per zone | Survey date, agency, stated accuracy |
| Grid name with historic realisations | Which realisation | Publication era, revision note |
| Code that is deprecated | Successor mapping | Registry successor, plus a note |
| Units named but frame implied | Geographic against projected | Unit words in the same sentence |

The common thread is that the document usually says enough, in prose, to close the gap — and that prose is discarded by every pipeline that extracts the code with a regular expression and moves on.

<figure class="diagram">
<svg viewBox="0 0 780 250" role="img" aria-labelledby="rae-amb-t rae-amb-d" xmlns="http://www.w3.org/2000/svg"><title id="rae-amb-t">One declaration, three candidate codes, one document that settles it</title><desc id="rae-amb-d">A declaration naming a projection zone matches several datum realisations; a sentence in the same document naming the survey era and agency selects one of them.</desc><rect x="0" y="0" width="780" height="250" fill="#ffffff"/><rect x="250" y="28" width="280" height="52" rx="8" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><text x="390" y="60" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">&#8220;coordinates in UTM zone 31 north&#8221;</text><rect x="30" y="118" width="220" height="62" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="280" y="118" width="220" height="62" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="530" y="118" width="220" height="62" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="140" y="146">historic datum</text><text x="390" y="146">current datum</text><text x="640" y="146">regional datum</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="140" y="168">pre-1990 surveys</text><text x="390" y="168">selected by the era note</text><text x="640" y="168">agency-specific</text></g><g stroke="#5b6471" stroke-width="2" marker-end="url(#rae-amb-a)"><line x1="330" y1="82" x2="180" y2="114"/><line x1="390" y1="82" x2="390" y2="114"/><line x1="450" y1="82" x2="600" y2="114"/></g><defs><marker id="rae-amb-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/></marker></defs><text x="390" y="222" fill="#1f2937" font-size="13" text-anchor="middle">&#8220;surveyed 2021 by the national mapping agency&#8221; closes the gap</text></svg>
<figcaption><b>The sentence next to the declaration is evidence.</b> Extracting the code and discarding its surroundings throws away the only information that can distinguish realisations sitting two hundred metres apart.</figcaption>
</figure>

## Implementation

The resolver reads the declaration and a context window around it, applies era and agency hints, and returns either a single code with a rationale or an ambiguity record listing what it could not separate.

```python
import logging
import re
from dataclasses import dataclass
from typing import Optional

from pyproj import CRS
from pyproj.exceptions import CRSError

log = logging.getLogger("epsg_disambiguation")

_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_UNIT_METRIC = re.compile(r"\b(metre|meter|metres|meters|m)\b", re.IGNORECASE)
_UNIT_DEGREE = re.compile(r"\b(degree|degrees|decimal degrees|lat(itude)?)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Resolution:
    epsg: Optional[int]
    candidates: tuple[int, ...]
    confidence: float
    rationale: str


def _latest_year(context: str) -> Optional[int]:
    years = [int(m.group(0)) for m in _YEAR.finditer(context)]
    return max(years) if years else None


def _constructs(epsg: int) -> bool:
    try:
        CRS.from_epsg(epsg)
        return True
    except CRSError:
        return False


def resolve_ambiguous(
    candidates: tuple[int, ...],
    context: str,
    era_map: dict[int, tuple[int, int]],       # epsg -> (valid_from, valid_to)
    agency_map: dict[str, int],                # agency phrase -> preferred epsg
) -> Resolution:
    """Narrow a candidate set using document context. Never raises; may return None."""
    usable = tuple(c for c in candidates if _constructs(c))
    if not usable:
        return Resolution(None, (), 0.0, "no candidate code constructs")
    if len(usable) == 1:
        return Resolution(usable[0], usable, 0.8, "only one constructible candidate")

    lowered = context.lower()
    for phrase, epsg in agency_map.items():
        if phrase in lowered and epsg in usable:
            return Resolution(epsg, usable, 0.9, f"agency phrase {phrase!r} in context")

    year = _latest_year(context)
    if year is not None:
        in_era = [c for c in usable
                  if c in era_map and era_map[c][0] <= year <= era_map[c][1]]
        if len(in_era) == 1:
            return Resolution(in_era[0], usable, 0.75, f"era {year} matches one realisation")
        if in_era:
            usable = tuple(in_era)               # narrowed but not settled

    metric = bool(_UNIT_METRIC.search(context))
    degrees = bool(_UNIT_DEGREE.search(context))
    if metric != degrees:                        # exactly one unit family mentioned
        want_projected = metric
        filtered = tuple(c for c in usable
                         if CRS.from_epsg(c).is_geographic != want_projected)
        if len(filtered) == 1:
            return Resolution(filtered[0], usable, 0.7, "unit words select one candidate")
        if filtered:
            usable = filtered

    log.info("ambiguous frame unresolved among %s", usable)
    return Resolution(None, usable, 0.0,
                      f"context did not separate {len(usable)} candidates")
```

The ordering of the three hints is not arbitrary. Agency is strongest because an organisation's standard frame is a documented fact rather than an inference. Era is next because datum realisations have published validity periods. Units are weakest because a document can name metres while storing degrees, and the check only fires when exactly one unit family appears — a document mentioning both tells you nothing.

Returning `None` with a populated candidate list is the important shape. It lets the caller say "this is one of these three, and they differ by up to two hundred metres" rather than choosing arbitrarily, which is the honest answer and, for most downstream questions, a usable one.

<figure class="diagram">
<svg viewBox="16 38 728 192" role="img" aria-labelledby="rae-act-t rae-act-d" xmlns="http://www.w3.org/2000/svg"><title id="rae-act-t">What each resolution outcome permits</title><desc id="rae-act-d">Three outcomes — settled, narrowed but unresolved, and unresolvable — each mapped to the class of answer the agent may give and the disclosure it must carry.</desc><rect x="16" y="38" width="728" height="192" fill="#ffffff"/><rect x="30" y="52" width="220" height="120" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="270" y="52" width="220" height="120" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="510" y="52" width="220" height="120" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="140" y="82">settled</text><text x="380" y="82">narrowed</text><text x="620" y="82">unresolvable</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="140" y="110">one code, evidenced</text><text x="140" y="132">measure and state freely</text><text x="140" y="154">cite the evidence</text><text x="380" y="110">two or three codes</text><text x="380" y="132">measure with a stated</text><text x="380" y="154">positional uncertainty</text><text x="620" y="110">no discriminating context</text><text x="620" y="132">describe, do not measure</text><text x="620" y="154">ask for the frame</text></g><text x="380" y="212" fill="#1f2937" font-size="13" text-anchor="middle">The middle column is the useful one — most documents land there</text></svg>
<figcaption><b>Narrowed is an answer.</b> A pipeline with only "resolved" and "failed" throws away the common case, where the frame is known to within a bounded displacement that many questions can tolerate if it is stated.</figcaption>
</figure>

## Validation & Testing

```python
ERA = {4277: (1936, 2001), 27700: (1936, 2100), 4258: (1989, 2100)}
AGENCY = {"national mapping agency": 27700}


def test_agency_phrase_settles_the_choice():
    r = resolve_ambiguous((4277, 27700), "surveyed by the national mapping agency", ERA, AGENCY)
    assert r.epsg == 27700 and r.confidence >= 0.9


def test_unresolvable_returns_candidates_not_a_guess():
    r = resolve_ambiguous((4277, 27700), "no useful context here", ERA, AGENCY)
    assert r.epsg is None and set(r.candidates) == {4277, 27700}


def test_nonconstructible_candidates_are_dropped():
    r = resolve_ambiguous((999999, 27700), "context", ERA, AGENCY)
    assert r.epsg == 27700 and r.rationale.startswith("only one")
```

The second test is the one that protects the design. Every future change that makes the resolver "more decisive" will fail it, which is exactly when a human should look at whether the new decisiveness is earned.


The third test guards a subtler property: a candidate list containing a code that no longer exists in the registry should shrink rather than fail. Registries do change, and a pipeline that raises on an unknown code will stop resolving every document that mentions it, including the ones where a perfectly good alternative candidate was available all along.

Run all three against the real era and agency maps rather than fixtures of them. These maps are configuration that drifts — an agency changes its standard, a datum realisation is superseded — and a test suite that uses its own copies will keep passing after the production configuration has stopped being correct.

## Gotchas & Edge Cases

**A year in the context that is not the survey year.** Documents cite regulations, reference earlier reports, and carry publication dates unrelated to the data. Taking the maximum year is a heuristic that fails on a 2024 report about a 1975 survey; prefer a year found in the same sentence as the declaration, and fall back to the maximum only when none is nearby.

<figure class="diagram">
<svg viewBox="16 42 748 155" role="img" aria-labelledby="rae-win-t rae-win-d" xmlns="http://www.w3.org/2000/svg"><title id="rae-win-t">Context window placement around a frame declaration</title><desc id="rae-win-d">A narrow window around the declaration captures the survey era and agency; a document-wide window also captures a bibliography and an unrelated regulation date, which corrupt the heuristics.</desc><rect x="16" y="42" width="748" height="155" fill="#ffffff"/><rect x="30" y="56" width="720" height="58" rx="8" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="250" y="66" width="280" height="38" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="390" y="90" fill="#1f2937" font-size="12" text-anchor="middle">declaration plus a few hundred characters</text><text x="140" y="90" fill="#5b6471" font-size="12" text-anchor="middle">bibliography</text><text x="640" y="90" fill="#5b6471" font-size="12" text-anchor="middle">regulation dates</text><text x="390" y="152" fill="#1f2937" font-size="13" text-anchor="middle">The narrow window contains the evidence; the wide one contains distractors</text><text x="390" y="180" fill="#5b6471" font-size="12" text-anchor="middle">Prefer a structured metadata block over proximity whenever the document has one</text></svg>
<figcaption><b>Wider is not better here.</b> Every extra paragraph adds years and organisation names that the heuristics will happily treat as evidence, turning a clean unresolved result into a confident wrong one.</figcaption>
</figure>

**Deprecated codes with more than one successor.** The registry does not always offer a single replacement, and choosing among successors is the same ambiguity one level down. Return the successor set rather than the first entry, and let the same narrowing logic run over it.

**Agency phrases that appear in a bibliography.** A citation to another organisation's report is not evidence about this document's frame. Restrict the agency search to a window around the declaration rather than the whole document, and prefer phrases in a methods or metadata section.

**Confidence that survives a change of evidence.** Once a resolution is cached, a later correction to the document does not invalidate it unless the cache key includes a document version. Key on the document's content hash so an edited document resolves afresh.

**Codes that differ only in axis order.** Some pairs describe the same datum with swapped axes, and no amount of context resolves which one an exporter used. Detect that case explicitly and treat it as an axis-order question rather than a frame question — a different check with a different fix.

## Frequently Asked Questions

<details class="faq-item"><summary><span>How large should the context window around the declaration be?</span></summary><p>A few hundred characters either side, or the enclosing section if the document has structure. Wider windows pull in unrelated years and agency names and make the heuristics noisier, which shows up as confident wrong answers rather than as failures. If a document has a metadata block, prefer it over proximity — a declared frame in a structured header with its own date is stronger evidence than any prose within reach.</p></details>

<details class="faq-item"><summary><span>What positional uncertainty should a narrowed result report?</span></summary><p>The maximum displacement between the candidate frames over the document's extent, computed rather than assumed. Two realisations of the same datum can differ by centimetres in one region and by hundreds of metres in another, so a fixed figure is misleading in both directions. Transform a corner of the extent through each candidate and take the largest separation.</p></details>

<details class="faq-item"><summary><span>Should the agent mention the ambiguity in its answer?</span></summary><p>Whenever the answer is a measurement and the ambiguity exceeds the precision being reported. Saying "roughly 400 metres, though the source's datum is ambiguous by up to 200 metres" is honest and actionable; saying "412 metres" from a narrowed resolution is a false precision that the pipeline knew about and suppressed. For descriptive answers, the ambiguity usually does not need surfacing.</p></details>

<details class="faq-item"><summary><span>Can the era and agency maps be learned from the corpus?</span></summary><p>Partly, and it is worth doing for agency preferences, which are stable and observable — an organisation's documents consistently declare the same frame. Era validity should come from the registry rather than from data, since the corpus will contain documents that used a datum outside its official period and learning from them encodes the error.</p></details>

## Related

- Up to the parent topic: [Retrieval-Augmented CRS Resolution](/geospatial-rag-pipelines/retrieval-augmented-crs-resolution/)
- [Inferring CRS from Retrieved Spatial Context](/geospatial-rag-pipelines/retrieval-augmented-crs-resolution/inferring-crs-from-retrieved-spatial-context/)
- Technique: [Choosing a Canonical CRS for Spatial LLM Pipelines](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/choosing-a-canonical-crs-for-llm-pipelines/)
- Concept: [Fallback Routing for Geospatial Queries](/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/)
