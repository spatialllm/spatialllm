# Splitting Polygon-Heavy Documents Without Severing Geometries

A cadastral export or zoning document can pack thousands of coordinates into a single feature, and a token-based splitter will happily cut through the middle of one. This guide implements a splitter that detects every geometry span, keeps each feature inside exactly one chunk, adds overlap for surrounding context, and — when a lone feature is larger than the whole budget — simplifies rather than severs it. It is the concrete splitter behind [chunk-boundary strategies for spatial corpora](https://www.spatialllm.org/geospatial-rag-pipelines/chunk-boundary-strategies-for-spatial-corpora/).

The splitter sits at the indexing stage of a retrieval pipeline, between raw document ingestion and embedding. Its contract is strict: every emitted chunk must contain only whole geometries, every geometry must appear in exactly one chunk, and no chunk may exceed the token budget without having first tried to shrink an oversized feature by decimation.

## When to Use This Approach

Use this splitter whenever a corpus contains inline geometry literals — WKT or GeoJSON — dense enough that a single feature can approach or exceed the chunk budget. For prose-only documents a standard recursive character splitter is cheaper and sufficient.

| Document profile | Recommended splitter | Why |
| --- | --- | --- |
| Prose with occasional coordinates | Standard recursive splitter | No feature approaches the budget |
| Many small features per document | Geometry-aware, feature-atomic | Prevents mid-feature cuts, keeps attributes attached |
| A few very large polygons | Geometry-aware + simplify fallback | One feature can exceed the budget alone |
| Pre-tiled feature collections | Split per feature directly | Boundaries are already feature-aligned |

The simplify fallback is what distinguishes this approach: alternatives either drop oversized features or truncate them, both of which corrupt retrieval. Decimating a polygon's vertices preserves its identity and rough footprint while fitting the window.

## Implementation

The function below detects geometry spans, greedily packs whole features into budget-bounded chunks with a configurable overlap, and decimates any single feature that cannot fit. It validates each decimated result with Shapely before accepting it and falls back to a bounding-box envelope if simplification still overflows. It pairs naturally with [how to tokenize polygon boundaries for transformer models](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/how-to-tokenize-polygon-boundaries-for-transformer-models/), which governs how the surviving vertices become tokens.

```python
from dataclasses import dataclass, field
import logging
from shapely import wkt
from shapely.errors import GEOSException

log = logging.getLogger("geometry_splitter")

@dataclass
class Chunk:
    text: str
    crs: str
    token_count: int
    simplified: bool = False
    features: list[str] = field(default_factory=list)

def split_without_severing(
    features: list[str],          # each item is one whole WKT/GeoJSON feature string
    crs: str,
    count_tokens,                 # callable: str -> int
    budget: int = 800,
    overlap: int = 1,             # trailing features repeated for context
) -> list[Chunk]:
    """Pack whole features into chunks; decimate any feature that alone exceeds budget."""
    if not features:
        return []

    prepared: list[tuple[str, int]] = []
    for feat in features:
        tokens = count_tokens(feat)
        if tokens > budget:
            feat, tokens = _shrink_feature(feat, count_tokens, budget)
        prepared.append((feat, tokens))

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_tokens = 0
    for feat, tokens in prepared:
        if buf and buf_tokens + tokens > budget:
            chunks.append(_finalize(buf, crs, count_tokens))
            buf = buf[-overlap:] if overlap else []      # carry context forward
            buf_tokens = sum(count_tokens(f) for f in buf)
        buf.append(feat)
        buf_tokens += tokens
    if buf:
        chunks.append(_finalize(buf, crs, count_tokens))
    return chunks

def _finalize(buf: list[str], crs: str, count_tokens) -> Chunk:
    body = "\n".join(buf)
    simplified = any(f.startswith("<<decimated") or f.startswith("<<bbox") for f in buf)
    return Chunk(text=body, crs=crs, token_count=count_tokens(body),
                 simplified=simplified, features=list(buf))

def _shrink_feature(feat: str, count_tokens, budget: int) -> tuple[str, int]:
    """Decimate an oversized single feature; last resort is its bounding box."""
    try:
        geom = wkt.loads(feat)
    except (GEOSException, ValueError) as exc:
        log.warning("Unparseable oversized feature, isolating verbatim: %s", exc)
        return feat, count_tokens(feat)             # isolate, never cut

    for tol in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        try:
            reduced = geom.simplify(tol, preserve_topology=True)
        except GEOSException:
            continue
        if reduced.is_empty or not reduced.is_valid:
            continue
        candidate = f"<<decimated tol={tol}>> {reduced.wkt}"
        if count_tokens(candidate) <= budget:
            log.info("Decimated feature at tolerance %s to fit budget", tol)
            return candidate, count_tokens(candidate)

    envelope = f"<<bbox fallback>> {geom.envelope.wkt}"     # deterministic last resort
    log.warning("Simplification insufficient; falling back to bounding box")
    return envelope, count_tokens(envelope)
```

## Validation & Testing

1. **No severed geometry.** For every emitted chunk, strip the decoration markers and load each feature with `shapely.wkt.loads`; any raised `GEOSException` fails CI, proving a feature was cut.
2. **Exactly-once coverage.** Assert that the count of distinct source features across all chunks (ignoring overlap repeats) equals the input feature count — none dropped, none lost to decimation.
3. **Budget respected.** Assert every chunk satisfies `token_count <= budget`; a chunk over budget means the shrink fallback failed and must halt the build.
4. **Simplify is flagged.** Assert any chunk containing a decimated or bbox feature has `simplified is True`, so downstream consumers can discount geometric precision.

## Gotchas & Edge Cases

**Topology collapse under simplification.** Aggressive `simplify` can turn a thin polygon empty or self-intersecting. Fix: the loop rejects empty or invalid results and escalates tolerance, ending at the always-valid bounding-box envelope.

**Overlap duplicating large features.** Carrying an oversized feature forward as overlap can blow the next chunk's budget. Fix: overlap should carry small trailing features only; guard the carried token sum and skip overlap when it alone nears the budget.

**Attribute-geometry separation.** If a feature's attributes are stored separately from its geometry string, packing by geometry alone can strand them. Fix: concatenate each feature's attributes into its feature string before splitting, so they move as one unit.

**Unparseable literals.** A malformed WKT string cannot be decimated. Fix: `_shrink_feature` isolates it verbatim in its own chunk rather than attempting a cut, preserving the no-sever guarantee even for bad input.

## Related

- Up to the area guide: [Chunk-Boundary Strategies for Spatial Corpora](https://www.spatialllm.org/geospatial-rag-pipelines/chunk-boundary-strategies-for-spatial-corpora/)
- Related area: [Retrieval-Augmented CRS Resolution](https://www.spatialllm.org/geospatial-rag-pipelines/retrieval-augmented-crs-resolution/)
- Prerequisite technique: [How to Tokenize Polygon Boundaries for Transformer Models](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/how-to-tokenize-polygon-boundaries-for-transformer-models/)
