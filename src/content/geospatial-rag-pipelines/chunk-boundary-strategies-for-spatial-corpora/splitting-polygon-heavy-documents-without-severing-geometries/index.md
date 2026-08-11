---
title: Splitting Polygon-Heavy Documents Without Severing Geometries
description: A splitter that treats coordinate lists as no-cut zones, keeps every feature whole, and isolates or generalizes the one polygon too large to fit the window.
slug: splitting-polygon-heavy-documents-without-severing-geometries
type: howto
breadcrumb: Splitting Without Severing
datePublished: 2025-03-12
dateModified: 2026-08-11
---

# Splitting Polygon-Heavy Documents Without Severing Geometries

A land-use appendix with four hundred polygons is the document that breaks retrieval pipelines. Each feature is a long coordinate list, the whole file is far past any embedding window, and the obvious fix — split every 800 tokens — produces chunks containing the second half of one ring and the first half of the next. This guide builds a splitter that cannot do that, as the ingestion stage of [chunk-boundary strategies for spatial corpora](/geospatial-rag-pipelines/chunk-boundary-strategies-for-spatial-corpora/).

## When to Use This Approach

Use it whenever geometry is inline in the text being chunked, which is the normal case for GeoJSON exports, WKT dumps, survey appendices and anything converted from a spatial database into a document. If geometry lives in a separate column and only its description is chunked, you have a simpler problem and should keep it that way.

| Document shape | Splitter | Why |
|----------------|----------|-----|
| Prose with occasional coordinates | Span-aware, paragraph boundaries | Cheap, and the geometry is rare enough to skip |
| Dense feature collection | Feature-atomic, one or few per chunk | Paragraph breaks are meaningless here |
| One very large geometry | Isolate and generalize | No boundary exists that helps |
| Mixed report with appendices | Span-aware for the body, feature-atomic for the appendix | The two halves want different rules |

The decision hinges on one measurement: what share of the document's tokens sit inside geometry spans. Below roughly a fifth, a span-aware paragraph splitter behaves well. Above a half, paragraph breaks are decoration and the feature is the only sensible unit.

<figure class="diagram">
<svg viewBox="26 9 708 205" role="img" aria-labelledby="spd-share-t spd-share-d" xmlns="http://www.w3.org/2000/svg"><title id="spd-share-t">Choosing a splitter from the share of tokens inside geometry</title><desc id="spd-share-d">Three bands of geometry density map to three splitter choices: paragraph-based with span protection at low density, feature-atomic at high density, and a mixed strategy in between.</desc><rect x="26" y="9" width="708" height="205" fill="#ffffff"/><text x="380" y="34" fill="#5b6471" font-size="13" text-anchor="middle">Measure once per document: tokens inside geometry spans, over total tokens</text><rect x="40" y="62" width="220" height="86" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="270" y="62" width="220" height="86" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="500" y="62" width="220" height="86" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="150" y="92">under 20%</text><text x="380" y="92">20% to 50%</text><text x="610" y="92">over 50%</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="150" y="118">paragraph splitter</text><text x="150" y="138">with span protection</text><text x="380" y="118">split the document</text><text x="380" y="138">body from appendix</text><text x="610" y="118">feature-atomic:</text><text x="610" y="138">one feature per chunk</text></g><text x="380" y="196" fill="#1f2937" font-size="13" text-anchor="middle">The measurement takes one pass and settles an argument that otherwise runs for weeks</text></svg>
<figcaption><b>One number decides the strategy.</b> Teams argue about chunk size and rarely measure geometry density, which is the variable that actually determines whether paragraph boundaries mean anything in a given document.</figcaption>
</figure>

## Implementation

The splitter below walks the document once, treating detected geometry spans as atomic. It accumulates features until adding the next one would exceed the budget, then closes the chunk. A single feature larger than the budget is emitted alone and flagged — never cut.

```python
import logging
import re
from dataclasses import dataclass
from typing import Callable, Iterator

log = logging.getLogger("geo_splitter")

_FEATURE = re.compile(
    r"(?:\b(?:POINT|LINESTRING|POLYGON|MULTIPOLYGON|GEOMETRYCOLLECTION)\s*\([^;]*?\))"
    r"|(?:\{[^{}]*\"type\"\s*:\s*\"(?:Polygon|MultiPolygon|Feature)\"[^{}]*\})",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class Chunk:
    text: str
    feature_count: int
    tokens: int
    oversized: bool           # a single feature exceeded the budget


def _segments(text: str) -> Iterator[tuple[str, bool]]:
    """Yield (segment, is_geometry) covering the document with no gaps or overlaps."""
    cursor = 0
    for m in _FEATURE.finditer(text):
        if m.start() > cursor:
            yield text[cursor:m.start()], False
        yield m.group(0), True
        cursor = m.end()
    if cursor < len(text):
        yield text[cursor:], False


def split_without_severing(
    text: str,
    count_tokens: Callable[[str], int],
    budget: int = 800,
) -> list[Chunk]:
    """Split a geometry-bearing document so no chunk contains a partial feature."""
    if budget < 64:
        raise ValueError("budget too small to hold any realistic feature")

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_tokens = 0
    buf_features = 0

    def flush(oversized: bool = False) -> None:
        nonlocal buf, buf_tokens, buf_features
        body = "".join(buf).strip()
        if body:
            chunks.append(Chunk(body, buf_features, buf_tokens, oversized))
        buf, buf_tokens, buf_features = [], 0, 0

    for segment, is_geometry in _segments(text):
        try:
            seg_tokens = count_tokens(segment)
        except Exception as exc:                      # tokenizer outage or bad input
            log.warning("token count failed, estimating from length: %s", exc)
            seg_tokens = max(1, len(segment) // 4)    # deterministic fallback

        # A single feature larger than the whole budget gets its own chunk, whole.
        if is_geometry and seg_tokens > budget:
            flush()
            chunks.append(Chunk(segment.strip(), 1, seg_tokens, True))
            log.info("isolated oversized feature: %d tokens (budget %d)", seg_tokens, budget)
            continue

        if buf_tokens + seg_tokens > budget and buf:
            flush()

        buf.append(segment)
        buf_tokens += seg_tokens
        buf_features += int(is_geometry)

    flush()
    return chunks
```

Three properties make this safe. The segment generator covers the document exactly once, so no text is dropped or duplicated. Geometry segments are never split, because the loop only ever appends them whole. And the oversized case is handled before the budget check, so the "flush then append" path cannot be reached with a feature that will not fit.

The token-count fallback matters more than it looks. Chunking often runs against a remote tokenizer, and a timeout mid-document would otherwise abort a build; estimating four characters per token is wrong by perhaps 15% and lets the build finish with a flag rather than a crash.

<figure class="diagram">
<svg viewBox="16 42 703 207" role="img" aria-labelledby="spd-walk-t spd-walk-d" xmlns="http://www.w3.org/2000/svg"><title id="spd-walk-t">How the buffer fills and flushes across segments</title><desc id="spd-walk-d">Segments are appended to a buffer until the next one would exceed the budget, at which point the buffer flushes as a chunk. An oversized feature bypasses the buffer entirely and is emitted alone.</desc><rect x="16" y="42" width="703" height="207" fill="#ffffff"/><rect x="30" y="56" width="110" height="56" rx="6" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="150" y="56" width="110" height="56" rx="6" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="270" y="56" width="110" height="56" rx="6" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="390" y="56" width="150" height="56" rx="6" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="550" y="56" width="110" height="56" rx="6" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><g fill="#1f2937" font-size="11.5" text-anchor="middle"><text x="85" y="90">prose 90</text><text x="205" y="90">feature 310</text><text x="325" y="90">feature 280</text><text x="465" y="90">feature 1240</text><text x="605" y="90">feature 300</text></g><rect x="30" y="150" width="350" height="52" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="390" y="150" width="150" height="52" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="550" y="150" width="110" height="52" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="11.5" text-anchor="middle"><text x="205" y="182">chunk 1 — 680 tokens</text><text x="465" y="182">isolated, flagged</text><text x="605" y="182">chunk 3</text></g><text x="400" y="232" fill="#5b6471" font-size="12" text-anchor="middle">Chunk sizes differ because feature sizes differ — that is the design, not a defect</text></svg>
<figcaption><b>The oversized feature never enters the buffer.</b> Handling it before the budget check is what removes the tempting branch where a large feature is "partially added" — the branch that, once written, is the one that severs a ring.</figcaption>
</figure>

## Validation & Testing

```python
from shapely import wkt


def test_every_chunk_parses():
    doc = load_fixture("land_use_appendix.txt")
    for chunk in split_without_severing(doc, count_tokens, budget=800):
        for m in _FEATURE.finditer(chunk.text):
            wkt.loads(m.group(0))          # raises if a ring was severed


def test_no_feature_is_lost_or_duplicated():
    doc = load_fixture("land_use_appendix.txt")
    source = len(_FEATURE.findall(doc))
    chunked = sum(len(_FEATURE.findall(c.text)) for c in split_without_severing(doc, count_tokens))
    assert chunked == source


def test_oversized_feature_is_isolated_not_cut():
    giant = "POLYGON((" + ", ".join(f"{i/1000} {i/2000}" for i in range(20000)) + "))"
    chunks = split_without_severing(giant, count_tokens, budget=800)
    assert len(chunks) == 1 and chunks[0].oversized
    wkt.loads(chunks[0].text)              # still a valid polygon
```

The first test is the one that matters and the one to run on real corpus samples rather than a fixture alone. The second catches the subtler bug where the segment generator's cursor arithmetic drifts, silently dropping the text between two adjacent features. The third is a property test in disguise: it asserts that the pipeline's stated policy — isolate, never cut — holds for the one input that most tempts an implementation to break it.

Run the first test against a rotating sample of production documents rather than only against fixtures. Fixtures encode the failures you already knew about; a sample encodes the ones your sources are about to send you. A hundred documents a night, chosen at random from the previous day's ingestion, is enough to catch a new export format within a day of it appearing.

## Gotchas & Edge Cases

**Nested braces defeat the GeoJSON pattern.** A `Feature` object containing a nested `properties` object with its own braces will not match a pattern that forbids interior braces, so the feature is treated as prose and can be cut. Where the corpus is genuine JSON, parse it rather than pattern-matching it; the regex approach is for mixed documents where a parser has nothing to hold onto.

**Coordinates split across lines.** Some exporters wrap long coordinate lists. The patterns above use `DOTALL` for exactly this reason; a pattern without it silently stops matching at the first newline and every wrapped feature becomes cuttable.

**A budget smaller than the average feature.** If most features exceed the budget, nearly every chunk is flagged oversized and the pipeline degenerates into one chunk per feature with warnings attached. That is not a splitter failure — it is the signal to generalize geometry before chunking, or to raise the budget.

**Escaped or quoted geometry inside a JSON string.** A WKT literal stored as a string value inside a larger JSON document appears twice to a naive scan — once as the quoted string and once as the geometry inside it — and merging those overlapping spans is what keeps the segment generator from emitting the same text in two chunks. Merge spans before segmenting, never after, and prefer the outermost match when two overlap.

**Attribute rows stranded after a flush.** A feature can land at the end of one chunk with its attribute block at the start of the next. Extend spans to cover adjacent attributes before splitting, as the parent topic describes, or accept that some retrieved geometry will arrive without its description.

<figure class="diagram">
<svg viewBox="16 38 708 176" role="img" aria-labelledby="spd-over-t spd-over-d" xmlns="http://www.w3.org/2000/svg"><title id="spd-over-t">What to do with a feature that will not fit</title><desc id="spd-over-d">Three responses to an oversized feature — isolate it whole, generalize it to a tolerance, or split it by part for a multipart geometry — with the cost each one carries.</desc><rect x="16" y="38" width="708" height="176" fill="#ffffff"/><rect x="30" y="52" width="216" height="106" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="262" y="52" width="216" height="106" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="494" y="52" width="216" height="106" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="138" y="80">isolate whole</text><text x="370" y="80">generalize</text><text x="602" y="80">split by part</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="138" y="106">always correct</text><text x="138" y="128">may exceed the window</text><text x="370" y="106">fits, and stays valid</text><text x="370" y="128">loses boundary detail</text><text x="602" y="106">only for multipart</text><text x="602" y="128">each part stands alone</text></g><text x="370" y="196" fill="#1f2937" font-size="13" text-anchor="middle">All three keep every ring closed — none of them is a cut</text></svg>
<figcaption><b>Three legal answers, one illegal one.</b> Cutting is absent from this figure on purpose: it is not a fourth option with different trade-offs, it is the failure the other three exist to avoid.</figcaption>
</figure>

## Frequently Asked Questions

<details class="faq-item"><summary><span>What simplification tolerance is safe for an oversized polygon?</span></summary><p>One that is small relative to the questions the corpus will answer. If retrieval is about which district a site falls in, tens of metres is invisible; if it is about a boundary dispute, even a metre is not. Choose the tolerance from the use case, apply it with a topology-preserving simplifier so the result stays valid, and record both the original vertex count and the tolerance in the chunk metadata so the loss is visible to anything reading the chunk later.</p></details>

<details class="faq-item"><summary><span>Should the splitter emit overlapping chunks for context?</span></summary><p>Only by whole features, and only when retrieval quality measurably improves. Overlap in prose chunking works because sentences are cheap to duplicate; features are not, and a duplicated feature can occupy two slots in one retrieved context window. If you do overlap, deduplicate on feature identity at retrieval time rather than trusting that the model will notice it is reading the same polygon twice.</p></details>

<details class="faq-item"><summary><span>How do I chunk a document where geometry and prose interleave tightly?</span></summary><p>Let the feature drive the boundary and carry the surrounding prose with it. A pattern that works well is to attach each feature to the paragraph immediately preceding it, which in survey and assessment documents is almost always the sentence that describes it. The result is chunks that read as a unit rather than as a geometry with a stranded caption.</p></details>

<details class="faq-item"><summary><span>Is a token budget the right constraint, or should it be characters?</span></summary><p>Tokens, because the window it protects is measured in tokens and coordinate text tokenizes very unevenly — a run of digits and decimal points produces far more tokens per character than prose does. Budgeting in characters systematically underestimates coordinate-heavy content by a factor that varies with coordinate precision, which is exactly the content this splitter exists to handle.</p></details>

## Related

- Up to the parent topic: [Chunk-Boundary Strategies for Spatial Corpora](/geospatial-rag-pipelines/chunk-boundary-strategies-for-spatial-corpora/)
- [Carrying Frame and Extent Metadata Into Every Chunk](/geospatial-rag-pipelines/chunk-boundary-strategies-for-spatial-corpora/carrying-crs-and-extent-metadata-into-every-chunk/)
- Concept: [Geometry Tokenization Strategies](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/)
- Concept: [Context-Window Optimization for Maps](/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/)
