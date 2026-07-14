# Chunk-Boundary Strategies for Spatial Corpora

Geometry-rich documents break naive chunkers. A fixed-size splitter that cuts a corpus every 800 tokens will sever a polygon mid-coordinate, orphan a feature from the CRS header that gives its numbers meaning, or divorce a geometry from the attributes that describe it. Chunk-boundary strategy for spatial corpora is the discipline of placing splits where they preserve meaning — never inside a geometry, never between a feature and its context.

This area belongs to [Geospatial RAG Pipelines](https://www.spatialllm.org/geospatial-rag-pipelines/) and targets a specific retrieval failure: a chunk that contains half a WKT polygon or a coordinate block with no reference frame retrieves and embeds as noise, then poisons downstream reasoning with truncated geometry. The strategies here keep each feature atomic and CRS-anchored, so retrieval returns spans a model can actually parse. They lean on [geometry tokenization strategies](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/) for how geometry becomes tokens and on [context-window optimization for maps](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/) for how much of it a chunk can afford to carry.

<figure class="diagram">
<svg viewBox="0 0 860 300" role="img" aria-labelledby="cbss-t cbss-d" xmlns="http://www.w3.org/2000/svg">
  <title id="cbss-t">Geometry-aware chunk boundary placement</title>
  <desc id="cbss-d">A spatial document is scanned for geometry spans, split only at safe boundaries between whole features, each chunk is stamped with the CRS header, guarded against a token budget, and emitted for indexing.</desc>
  <defs>
    <marker id="cbss-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/>
    </marker>
  </defs>
  <g fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2">
    <rect x="15" y="55" width="150" height="80" rx="8"/>
    <rect x="183" y="55" width="150" height="80" rx="8"/>
    <rect x="351" y="55" width="150" height="80" rx="8"/>
    <rect x="519" y="55" width="150" height="80" rx="8"/>
    <rect x="687" y="55" width="150" height="80" rx="8"/>
  </g>
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#cbss-arrow)">
    <line x1="166" y1="95" x2="181" y2="95"/>
    <line x1="334" y1="95" x2="349" y2="95"/>
    <line x1="502" y1="95" x2="517" y2="95"/>
    <line x1="670" y1="95" x2="685" y2="95"/>
  </g>
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#cbss-arrow)">
    <line x1="90" y1="137" x2="90" y2="202"/>
    <line x1="258" y1="137" x2="258" y2="202"/>
    <line x1="426" y1="137" x2="426" y2="202"/>
    <line x1="594" y1="137" x2="594" y2="202"/>
    <line x1="762" y1="137" x2="762" y2="202"/>
  </g>
  <rect x="15" y="205" width="822" height="58" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/>
  <g fill="#1f2937" font-size="13" text-anchor="middle">
    <text x="90" y="90"><tspan x="90" dy="0">Detect</tspan><tspan x="90" dy="16">geometry spans</tspan></text>
    <text x="258" y="90"><tspan x="258" dy="0">Split at safe</tspan><tspan x="258" dy="16">boundaries</tspan></text>
    <text x="426" y="90"><tspan x="426" dy="0">Stamp CRS</tspan><tspan x="426" dy="16">header</tspan></text>
    <text x="594" y="90"><tspan x="594" dy="0">Token</tspan><tspan x="594" dy="16">budget guard</tspan></text>
    <text x="762" y="90"><tspan x="762" dy="0">Emit</tspan><tspan x="762" dy="16">chunk</tspan></text>
  </g>
  <text x="426" y="240" fill="#1f2937" font-size="15" font-weight="600" text-anchor="middle">Whole features, CRS-anchored, within budget</text>
  <text x="426" y="287" fill="#5b6471" font-size="12" text-anchor="middle">Fallback: simplify oversized feature · flag · isolate in its own chunk</text>
</svg>
</figure>

## Foundational Principles

**A geometry is indivisible.** A WKT `POLYGON(...)` or a GeoJSON `Feature` is the atomic unit; a split may fall between features but never inside one. A half-polygon is not a smaller polygon — it is malformed text that embeds as garbage and can crash a parser downstream.

**CRS travels with every chunk.** Coordinates without a reference frame are meaningless numbers. Each emitted chunk carries the document's CRS header in its metadata so a retrieved fragment is self-describing, consistent with the anchoring enforced by [CRS normalization](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/).

**The token budget is a hard ceiling with a defined escape.** Chunks must fit the embedding model's window, but the budget can never justify severing a feature. When a single feature exceeds the budget, the strategy simplifies or isolates it — it does not cut it.

## Step-by-Step Implementation Pipeline

### 1. Detect geometry spans before choosing any boundary

The first pass locates the byte ranges occupied by WKT and GeoJSON geometries so later steps treat them as no-cut zones. Detection must be conservative: a false negative severs a geometry, so ambiguous spans are treated as geometry. The token accounting here feeds the budget logic described in [context-window optimization for maps](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/).

```python
import re
from dataclasses import dataclass

# Matches WKT primitives and GeoJSON geometry objects at a coarse span level.
_WKT = re.compile(r"\b(?:POINT|LINESTRING|POLYGON|MULTIPOLYGON|GEOMETRYCOLLECTION)\s*\([^;]*?\)",
                  re.IGNORECASE | re.DOTALL)
_GEOJSON = re.compile(r'\{[^{}]*"type"\s*:\s*"(?:Polygon|MultiPolygon|Feature)"[^{}]*\}', re.DOTALL)

@dataclass(frozen=True)
class Span:
    start: int
    end: int

def detect_geometry_spans(text: str) -> list[Span]:
    """Return sorted, merged no-cut spans; on regex failure, protect the whole doc."""
    try:
        spans = [Span(m.start(), m.end()) for m in _WKT.finditer(text)]
        spans += [Span(m.start(), m.end()) for m in _GEOJSON.finditer(text)]
    except Exception as exc:                       # pathological input, catastrophic backtracking
        return [Span(0, len(text))]                # conservative: forbid all interior cuts
    spans.sort(key=lambda s: s.start)
    merged: list[Span] = []
    for s in spans:
        if merged and s.start <= merged[-1].end:
            merged[-1] = Span(merged[-1].start, max(merged[-1].end, s.end))
        else:
            merged.append(s)
    return merged
```

### 2. Place boundaries only at safe offsets

Candidate split points are the paragraph breaks that fall outside every geometry span. The splitter walks the document accumulating a token count and cuts at the last safe offset before the budget is reached. The mechanics of keeping each feature whole are developed fully in [splitting polygon-heavy documents without severing geometries](https://www.spatialllm.org/geospatial-rag-pipelines/chunk-boundary-strategies-for-spatial-corpora/splitting-polygon-heavy-documents-without-severing-geometries/).

```python
def is_safe_offset(offset: int, spans: list[Span]) -> bool:
    """A cut is safe only if it lies strictly outside every geometry span."""
    return not any(s.start < offset < s.end for s in spans)

def safe_boundaries(text: str, spans: list[Span]) -> list[int]:
    candidates = [m.start() for m in re.finditer(r"\n\s*\n", text)]
    safe = [c for c in candidates if is_safe_offset(c, spans)]
    if not safe:                                   # no natural break survived
        safe = [len(text)]                         # emit the document whole
    return safe
```

### 3. Stamp CRS metadata and guard the budget

Each chunk is emitted with the source CRS attached and its token count verified. When a chunk containing a single indivisible feature still exceeds the budget, the guard routes it to the simplify-or-isolate fallback rather than forcing a cut.

```python
def emit_chunk(body: str, crs: str, count_tokens, budget: int) -> dict:
    tokens = count_tokens(body)
    over_budget = tokens > budget
    return {
        "text": body.strip(),
        "crs": crs,                                # self-describing chunk
        "token_count": tokens,
        "needs_fallback": over_budget,             # triggers simplify/isolate
    }
```

## Failure Modes & Root Causes

**Severed geometry.** A boundary lands inside a coordinate list, emitting a truncated ring. Root cause: chunking on token count without a no-cut map. Mitigation: the span detection in step 1 gated by `is_safe_offset`.

**Orphaned CRS.** The header declaring the reference frame lands in chunk one; the geometries land in chunk five with no frame. Root cause: treating the CRS header as ordinary prose. Mitigation: stamp CRS onto every chunk in step 3.

**Attribute divorce.** A feature's geometry and its attribute table split into different chunks, so retrieval returns coordinates with no description. Root cause: boundaries that ignore feature grouping. Mitigation: extend geometry spans to include the adjacent attribute block before choosing a boundary.

**Budget-forced cut.** An oversized feature tempts a mid-geometry split to satisfy the window. Root cause: treating the budget as absolute. Mitigation: the simplify-or-isolate fallback, never a cut.

## Production Validation Protocols

1. **Parse every geometry per chunk.** In CI, load each emitted chunk's geometries with a WKT/GeoJSON reader; any parse failure fails the build — it proves a severed feature escaped.
2. **CRS presence assertion.** Assert every chunk carries a non-empty `crs` field; a missing frame is a hard error.
3. **Budget-with-escape check.** Assert `token_count <= budget` OR `needs_fallback is True`; no chunk may quietly exceed the window without flagging.
4. **Round-trip feature count.** Sum features across all chunks and assert it equals the source count — no feature dropped, none duplicated by overlap.
5. **Reconstruction test.** Concatenating chunks in order (minus overlap) must reproduce the source geometry set byte-for-byte.

## Related

- Up to the area overview: [Geospatial RAG Pipelines](https://www.spatialllm.org/geospatial-rag-pipelines/)
- Concept: [Geometry Tokenization Strategies](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/)
- Concept: [Context-Window Optimization for Maps](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/)
- In-depth technique: [Splitting Polygon-Heavy Documents Without Severing Geometries](https://www.spatialllm.org/geospatial-rag-pipelines/chunk-boundary-strategies-for-spatial-corpora/splitting-polygon-heavy-documents-without-severing-geometries/)
