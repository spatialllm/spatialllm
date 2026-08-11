---
title: WKT vs GeoJSON vs H3 for LLM Tokenization
description: Compare three geometry representations on token cost, fidelity and how reliably a model can manipulate them, and pick one for the whole corpus rather than per source.
slug: wkt-vs-geojson-vs-h3-for-llm-tokenization
type: howto
breadcrumb: Representation Comparison
datePublished: 2025-01-22
dateModified: 2026-08-11
---

# WKT vs GeoJSON vs H3 for LLM Tokenization

Three representations dominate geometry in prompts, and they differ by roughly a factor of six in cost and completely in what they preserve. Choosing between them is not a matter of taste: each one makes a different set of questions cheap and a different set impossible. This guide compares them on the axes that matter for a prompt pipeline, as part of [geometry tokenization strategies](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/).

## When to Use This Approach

Choose once, for the corpus, and convert at ingestion. Mixing representations means every parser and every prompt must handle both, and the model spends capacity distinguishing forms rather than reasoning about places.

| Property | Compact text | Structured object | Cell identifiers |
|----------|--------------|-------------------|------------------|
| Token cost for one polygon | Low | High — about 1.6× | Lowest — about 0.4× |
| Exactness | Exact | Exact | Lossy by construction |
| Model manipulation reliability | Moderate | High | High for set operations |
| Containment and adjacency | Needs a predicate | Needs a predicate | Native, cheap |
| Boundary questions | Answerable | Answerable | Not answerable |
| Human readability in a log | Good | Verbose | Opaque |

The row that decides most cases is the second-to-last. Cell identifiers answer "is this inside that" and "what is next to this" almost for free and cannot say anything about where a boundary actually runs, because the boundary has been replaced by a staircase.

<figure class="diagram">
<svg viewBox="16 38 748 178" role="img" aria-labelledby="rep-tri-t rep-tri-d" xmlns="http://www.w3.org/2000/svg"><title id="rep-tri-t">What each representation preserves and what it costs</title><desc id="rep-tri-d">Compact text is exact and cheap, structured objects are exact and expensive but easiest for a model to edit reliably, and cell identifiers are cheapest and cannot describe a boundary.</desc><rect x="16" y="38" width="748" height="178" fill="#ffffff"/><rect x="30" y="52" width="230" height="150" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="275" y="52" width="230" height="150" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="520" y="52" width="230" height="150" rx="8" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="145" y="84">compact text</text><text x="390" y="84">structured object</text><text x="635" y="84">cell identifiers</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="145" y="114">exact geometry</text><text x="145" y="140">cheapest exact form</text><text x="145" y="166">fiddly for a model to edit</text><text x="145" y="190">the usual default</text><text x="390" y="114">exact geometry</text><text x="390" y="140">1.6× the tokens</text><text x="390" y="166">model edits it reliably</text><text x="390" y="190">worth it when it must</text><text x="635" y="114">approximate region</text><text x="635" y="140">0.4× the tokens</text><text x="635" y="166">set operations are free</text><text x="635" y="190">no boundary detail</text></g></svg>
<figcaption><b>The third column is a different kind of object.</b> Cells are not a cheaper encoding of the same polygon; they are a set of regions that approximately covers it, which is why they answer membership questions well and boundary questions not at all.</figcaption>
</figure>

## Implementation

The conversion functions below are the ones a corpus needs at ingestion. Each returns the text plus a note about what was lost, so the prompt layer can state it.

```python
import logging
from dataclasses import dataclass

log = logging.getLogger("geometry_representation")


@dataclass(frozen=True)
class Rendered:
    text: str
    form: str            # compact | structured | cells
    lossy: bool
    note: str


def as_compact_text(geom, places: int) -> Rendered:
    """Exact geometry, minimal syntax. The default for most corpora."""
    def fmt(v: float) -> str:
        return f"{round(v, places):.{places}f}".rstrip("0").rstrip(".") or "0"
    rings = [", ".join(f"{fmt(x)} {fmt(y)}" for x, y in ring) for ring in _rings(geom)]
    return Rendered(f"POLYGON(({'), ('.join(rings)}))", "compact", False, "")


def as_structured(geom, places: int) -> Rendered:
    """Exact geometry in an object form a model can edit without breaking syntax."""
    import json
    coords = [[[round(x, places), round(y, places)] for x, y in ring]
              for ring in _rings(geom)]
    body = {"type": "Polygon", "coordinates": coords}
    return Rendered(json.dumps(body, separators=(",", ":")), "structured", False, "")


def as_cells(geom, resolution: int, to_cells, max_cells: int = 64) -> Rendered:
    """A cell cover of the geometry. Cheap, approximate, and explicitly labelled."""
    try:
        cells = list(to_cells(geom, resolution))
    except Exception as exc:                       # a library failure must not lose the feature
        log.warning("cell cover failed at resolution %d: %s", resolution, exc)
        return as_compact_text(geom, 5)
    while len(cells) > max_cells and resolution > 1:
        resolution -= 1
        try:
            cells = list(to_cells(geom, resolution))
        except Exception:
            break
    note = (f"approximated by {len(cells)} cells at resolution {resolution}; "
            "boundary detail is not represented")
    return Rendered(" ".join(cells), "cells", True, note)
```

Falling back to exact text when the cell library fails is the right direction: an exact representation is never wrong, only expensive, whereas dropping the feature or emitting an empty cover is both.

The coarsening loop deserves a note. A cell cover of a large or thin polygon can run to thousands of identifiers, which defeats the point of using cells at all; stepping the resolution down until the cover fits keeps the cheap representation cheap, at the cost of a coarser approximation that the note reports.

<figure class="diagram">
<svg viewBox="46 36 608 210" role="img" aria-labelledby="rep-stair-t rep-stair-d" xmlns="http://www.w3.org/2000/svg"><title id="rep-stair-t">A boundary replaced by a cell staircase</title><desc id="rep-stair-d">A curved administrative boundary covered by grid cells becomes a stepped approximation, so membership near the edge depends on cell size rather than on the real line.</desc><rect x="46" y="36" width="608" height="210" fill="#ffffff"/><rect x="60" y="50" width="290" height="150" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="205" y="130" fill="#1f2937" font-size="12.5" text-anchor="middle">the real boundary</text><rect x="430" y="50" width="70" height="50" rx="3" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><rect x="500" y="50" width="70" height="50" rx="3" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><rect x="430" y="100" width="70" height="50" rx="3" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><rect x="500" y="100" width="70" height="50" rx="3" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><rect x="570" y="100" width="70" height="50" rx="3" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><rect x="430" y="150" width="70" height="50" rx="3" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><rect x="500" y="150" width="70" height="50" rx="3" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><text x="535" y="228" fill="#1f2937" font-size="12.5" text-anchor="middle">the cell cover</text><text x="205" y="228" fill="#5b6471" font-size="12" text-anchor="middle">membership depends on cell size</text></svg>
<figcaption><b>Good enough for "which region", never for "where is the line".</b> A question about a site near a boundary is answered by the cell size rather than by the data, and no caveat makes that answer usable.</figcaption>
</figure>

## Validation & Testing

```python
def test_compact_and_structured_agree_geometrically():
    from shapely import wkt
    import json
    from shapely.geometry import shape
    a = wkt.loads(as_compact_text(POLYGON, 5).text)
    b = shape(json.loads(as_structured(POLYGON, 5).text))
    assert a.equals_exact(b, 1e-5)


def test_cell_form_is_labelled_lossy():
    r = as_cells(POLYGON, 9, to_cells)
    assert r.lossy and "boundary detail is not represented" in r.note


def test_cell_cover_coarsens_rather_than_exploding():
    r = as_cells(LARGE_THIN_POLYGON, 12, to_cells, max_cells=64)
    assert r.form == "compact" or len(r.text.split()) <= 64


def test_library_failure_falls_back_to_exact():
    def broken(_geom, _res):
        raise RuntimeError("cell library unavailable")
    r = as_cells(POLYGON, 9, broken)
    assert r.form == "compact" and not r.lossy
```

The first test is the one that catches a whole class of conversion bugs, because it compares two independent renderings of the same geometry rather than checking either against a fixture string. Fixture strings encode formatting decisions and break whenever formatting changes; a geometric comparison breaks only when the geometry does.

## Gotchas & Edge Cases

**Structured output that a model has edited.** The main reason to pay 1.6× is that a model asked to modify geometry breaks compact text far more often than it breaks an object form. If nothing in the pipeline asks the model to produce or edit geometry, that advantage is not being used and the extra cost is pure.

**Cells used for a boundary question.** The failure is silent: the model answers confidently from a staircase. Where cells are used, restrict them to membership and adjacency, and route boundary questions to exact geometry.

<figure class="diagram">
<svg viewBox="26 9 676 221" role="img" aria-labelledby="rep-edit-t rep-edit-d" xmlns="http://www.w3.org/2000/svg"><title id="rep-edit-t">Failure rate when a model edits geometry in each exact form</title><desc id="rep-edit-d">Asked to modify a polygon, a model produces unparseable compact text far more often than unparseable structured output, because nesting and delimiters are easier to get wrong without redundancy.</desc><rect x="26" y="9" width="676" height="221" fill="#ffffff"/><text x="390" y="34" fill="#5b6471" font-size="13" text-anchor="middle">Unparseable output when a model is asked to modify one polygon</text><rect x="200" y="60" width="330" height="46" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="200" y="122" width="90" height="46" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="12.5"><text x="40" y="90">compact text</text><text x="40" y="152">structured</text></g><g fill="#1f2937" font-size="12" text-anchor="middle"><text x="365" y="90">unbalanced parentheses, dropped commas</text><text x="245" y="152">rare</text></g><text x="390" y="212" fill="#1f2937" font-size="13" text-anchor="middle">The premium buys reliability only where the model actually writes geometry</text></svg>
<figcaption><b>Pay the premium for the right reason.</b> If nothing in the pipeline asks a model to emit or edit geometry, the structured form is simply more expensive; if something does, it is the difference between a working feature and an intermittent one.</figcaption>
</figure>

**Ring orientation differing between forms.** The two exact forms have different conventions in common use, and a corpus that converts between them without canonicalising will produce two token sequences for one shape. Fix orientation at ingestion.

**Compact text with excess whitespace.** Some writers emit a space after every comma and around every parenthesis, which is a measurable fraction of the tokens in a coordinate-dense string. Emit a canonical minimal form rather than whatever a library defaults to.

**Cell resolution chosen globally.** One resolution across a corpus of mixed feature sizes produces a four-cell cover for a parcel and a four-thousand-cell cover for a county. Choose resolution from the feature's extent, cap the cell count, and report both.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Can more than one representation be stored?</span></summary><p>Stored, yes; sent to the model, no. Keeping exact geometry in the database and a cell index alongside it is a normal and useful design — the cells serve filtering and the geometry serves measurement. What causes trouble is putting both in a prompt, where the model must decide which one to trust and will occasionally reason over the approximation while quoting the exact form.</p></details>

<details class="faq-item"><summary><span>Which form should be used when the model must return geometry?</span></summary><p>The structured object form, almost always. A model closing three levels of parentheses correctly in a long coordinate list is doing something it is not reliably good at, and the failure mode is unparseable output. An object form has redundancy that makes errors detectable and often repairable, and the token premium is small relative to the cost of a failed generation.</p></details>

<details class="faq-item"><summary><span>Do cell identifiers help with retrieval as well as prompts?</span></summary><p>Substantially, and that is often the better use of them. A cell identifier is an exact-match token, which means the lexical half of a hybrid retrieval system can find every document about a region with a keyword lookup rather than a spatial join. That is a real capability, and it does not require the cells to appear in any prompt.</p></details>

<details class="faq-item"><summary><span>How much does the choice matter compared to precision?</span></summary><p>Comparable, and they compound. Moving from a structured object at seven decimal places to compact text at five is roughly a threefold reduction, which is larger than either change alone. Make both decisions at the same time and measure the combined result on real features, since the interaction between representation overhead and coordinate length is not obvious from either number.</p></details>

<details class="faq-item"><summary><span>What should a corpus do about geometry types other than polygons?</span></summary><p>Apply the same choice consistently. Points are cheap in every form and the decision hardly matters; lines behave like polygon rings and inherit the same trade-offs; multipart geometries multiply the representation overhead and are where the structured form&#8217;s verbosity hurts most. If one type dominates the corpus, let it drive the choice, and convert the minority types into the same form rather than special-casing them.</p></details>

<details class="faq-item"><summary><span>How should the chosen form be recorded?</span></summary><p>In the chunk metadata, alongside the precision and any simplification tolerance. A reader — human or machine — encountering geometry in a retrieved chunk needs to know whether it is exact, and that fact belongs with the data rather than in a pipeline document. It also makes a later migration tractable: converting a corpus is straightforward when every record says what it currently is.</p></details>

One practical note on migration. Converting a corpus between exact forms is mechanical and safe; converting to or from cells is not, because the cell form cannot reconstruct what it approximated. If cells are stored as the only representation, that decision is irreversible for the affected features, which is a reason to keep exact geometry in the database even when cells are what reaches the prompt.

Whichever form is chosen, write the conversion in one place and call it from everywhere. The failure this avoids is not conversion errors but drift: two call sites that format geometry slightly differently produce a corpus with two dialects of the same representation, which is all the cost of mixing forms with none of the benefit.

## Related

- Up to the parent topic: [Geometry Tokenization Strategies](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/)
- [Coordinate Precision Versus Token Cost](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/coordinate-precision-versus-token-cost/)
- [How to Tokenize Polygon Boundaries for Transformer Models](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/how-to-tokenize-polygon-boundaries-for-transformer-models/)
- Related topic: [Hybrid Spatial and Keyword Retrieval](/geospatial-rag-pipelines/hybrid-spatial-keyword-retrieval/)
