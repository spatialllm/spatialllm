---
layout: overview.njk
order: 3
icon: retrieval
---

# Geospatial RAG Pipelines

Retrieval-augmented generation lets a spatial LLM answer grounded questions by pulling supporting documents at inference time, but the geographic dimension introduces a failure mode that plain-text RAG never sees: the retriever returns a passage that is semantically on-topic yet spatially wrong — coordinates in the wrong reference system, a region a thousand kilometres from the query, or a polygon severed mid-ring by a naive text splitter. That poisoned context flows straight into generation, and the model confidently emits fabricated coordinates. This area treats retrieval as a spatial correctness problem, not just a similarity problem, so that every retrieved chunk carries a verifiable CRS, a bounding box, and topology intact enough to reason over.

<figure class="diagram">
<svg viewBox="0 0 860 300" role="img" aria-labelledby="grag-t grag-d" xmlns="http://www.w3.org/2000/svg">
  <title id="grag-t">Geospatial retrieval pipeline</title>
  <desc id="grag-d">Spatial corpus indexing, bbox-aware retrieval, CRS resolution, spatial reranking, and grounded generation run in sequence so the language model receives context that is both relevant and geographically correct.</desc>
  <defs>
    <marker id="grag-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
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
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#grag-arrow)">
    <line x1="166" y1="95" x2="181" y2="95"/>
    <line x1="334" y1="95" x2="349" y2="95"/>
    <line x1="502" y1="95" x2="517" y2="95"/>
    <line x1="670" y1="95" x2="685" y2="95"/>
  </g>
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#grag-arrow)">
    <line x1="90" y1="137" x2="90" y2="202"/>
    <line x1="258" y1="137" x2="258" y2="202"/>
    <line x1="426" y1="137" x2="426" y2="202"/>
    <line x1="594" y1="137" x2="594" y2="202"/>
    <line x1="762" y1="137" x2="762" y2="202"/>
  </g>
  <rect x="15" y="205" width="822" height="58" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/>
  <g fill="#1f2937" font-size="13" text-anchor="middle">
    <text x="90" y="90"><tspan x="90" dy="0">Corpus</tspan><tspan x="90" dy="16">indexing</tspan></text>
    <text x="258" y="90"><tspan x="258" dy="0">Bbox</tspan><tspan x="258" dy="16">retrieval</tspan></text>
    <text x="426" y="90"><tspan x="426" dy="0">CRS</tspan><tspan x="426" dy="16">resolution</tspan></text>
    <text x="594" y="90"><tspan x="594" dy="0">Spatial</tspan><tspan x="594" dy="16">reranking</tspan></text>
    <text x="762" y="90"><tspan x="762" dy="0">Grounded</tspan><tspan x="762" dy="16">generation</tspan></text>
  </g>
  <text x="426" y="240" fill="#1f2937" font-size="15" font-weight="600" text-anchor="middle">Context that is relevant AND geographically correct</text>
  <text x="426" y="287" fill="#5b6471" font-size="12" text-anchor="middle">Guards against: wrong CRS · wrong region · severed polygons</text>
</svg>
</figure>

Spatial RAG sits alongside two related areas of this site. The upstream mechanics of encoding geometry — how a shape becomes a vector — belong to [Spatial LLM Architecture & Core Concepts](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/), and the downstream act of turning a grounded answer into executable spatial operations belongs to [Geospatial Prompt Engineering & Tool Routing](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/). Retrieval is the connective tissue between them: it decides which geographic facts ever reach the model.

## Choosing a store that filters by geography, not just similarity

A vector database that ranks purely on embedding distance will happily return the top-k nearest neighbours regardless of where on Earth they sit. When a query is scoped to a metropolitan boundary, unconstrained similarity search leaks documents from other continents into the context window. The store must therefore support a metadata and bounding-box pre-filter that runs *before* the approximate nearest-neighbour scan, narrowing the candidate set to the geographic envelope of interest.

```python
def store_supports_geo_prefilter(store, probe_bbox, probe_vector):
    """Verify a vector store applies a geographic pre-filter before ANN ranking.
    Returns True only if scoped results are a strict subset of unscoped ones."""
    try:
        scoped = store.search(probe_vector, k=25, bbox=probe_bbox)
        unscoped = store.search(probe_vector, k=25)
    except (AttributeError, TypeError) as exc:
        # Store lacks a bbox parameter; treat as unsupported rather than guessing.
        return False
    scoped_ids = {hit.id for hit in scoped}
    unscoped_ids = {hit.id for hit in unscoped}
    if not scoped_ids:
        return False
    return scoped_ids.issubset(unscoped_ids | scoped_ids)
```

Getting this decision right up front avoids re-indexing an entire corpus later; the trade-offs between engines are worked through in [Spatial Vector Store Selection](https://www.spatialllm.org/geospatial-rag-pipelines/spatial-vector-store-selection/).

## Resolving the reference system of retrieved context

Text passages describing coordinates rarely state their CRS explicitly. A chunk may quote eastings and northings from a national grid, latitude and longitude in WGS84, or Web Mercator metres, and all three look like plausible number pairs to a language model. If the pipeline forwards them without resolution, the model averages incompatible frames and emits coordinates that land in the ocean.

```python
def infer_axis_frame(sample_coords):
    """Heuristically flag whether retrieved coordinates are geographic degrees.
    Falls back to 'unknown' so the caller must resolve CRS explicitly."""
    try:
        xs = [float(x) for x, _ in sample_coords]
        ys = [float(y) for _, y in sample_coords]
    except (ValueError, TypeError):
        return "unknown"
    if not xs or not ys:
        return "unknown"
    if all(-180 <= x <= 180 for x in xs) and all(-90 <= y <= 90 for y in ys):
        return "geographic_degrees"
    return "projected_or_unknown"
```

Because the reference frame must be attached to every retrieved fragment, the retrieval step and the CRS step are inseparable; the resolution strategy is detailed in [Retrieval-Augmented CRS Resolution](https://www.spatialllm.org/geospatial-rag-pipelines/retrieval-augmented-crs-resolution/) and builds on the ingestion-time work in [Coordinate Reference System Normalization](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/).

## Splitting spatial documents without severing geometries

Generic recursive text splitters chunk on character counts and paragraph breaks. Applied to a document containing an embedded WKT polygon or a GeoJSON feature collection, they cut a ring in half, leaving one chunk with an unclosed boundary and another with orphaned vertices. Either fragment retrieved alone yields an invalid geometry the model cannot reason about.

```python
def chunk_preserves_geometry(chunk_text):
    """Reject a chunk that opens a geometry literal it does not close."""
    opens = chunk_text.count("POLYGON") + chunk_text.count("MULTIPOLYGON")
    balanced = chunk_text.count("(") == chunk_text.count(")")
    if opens and not balanced:
        return False  # geometry literal was severed at the chunk boundary
    return True
```

Boundary-aware splitting keeps each retrievable unit self-contained, a discipline expanded in [Chunk-Boundary Strategies for Spatial Corpora](https://www.spatialllm.org/geospatial-rag-pipelines/chunk-boundary-strategies-for-spatial-corpora/) and connected to [Geometry Tokenization Strategies](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/).

## Reranking candidates by distance and relevance together

First-stage vector recall optimises for semantic closeness; it does not know that a query about flood risk in one river basin should down-weight an otherwise similar passage about a basin on another continent. A spatial reranker fuses embedding score with a geographic proximity term so that near, relevant context outranks far, relevant context.

```python
import math

def spatial_rerank(candidates, query_lonlat, alpha=0.6):
    """Blend semantic score with inverse geographic distance.
    Returns candidates sorted by fused score; empty input yields an empty list."""
    if not candidates:
        return []
    qx, qy = query_lonlat
    ranked = []
    for c in candidates:
        try:
            dx = (c.lon - qx) * math.cos(math.radians(qy))
            dy = c.lat - qy
            dist_deg = math.hypot(dx, dy)
        except AttributeError:
            dist_deg = float("inf")  # missing geometry sinks to the bottom
        proximity = 1.0 / (1.0 + dist_deg)
        fused = alpha * c.score + (1 - alpha) * proximity
        ranked.append((fused, c))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [c for _, c in ranked]
```

Fusing the two signals is the last correctness gate before the prompt is assembled, and it is developed fully in [Spatial Context Retrieval & Reranking](https://www.spatialllm.org/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/).

## Measuring whether retrieval actually grounded the answer

None of the preceding stages is trustworthy without measurement. A grounded spatial answer must be checked against the coordinates that were actually retrieved: if the model emits a point outside the union of retrieved bounding boxes, retrieval failed to constrain it and the output should be quarantined. Treating this as a scored metric rather than a spot check turns regressions into build failures.

```python
def answer_is_grounded(answer_points, retrieved_bboxes):
    """Confirm every emitted coordinate falls inside some retrieved envelope."""
    if not retrieved_bboxes:
        return False  # nothing was retrieved; the answer cannot be grounded
    for px, py in answer_points:
        inside = any(minx <= px <= maxx and miny <= py <= maxy
                     for (minx, miny, maxx, maxy) in retrieved_bboxes)
        if not inside:
            return False
    return True
```

Systematic scoring of grounding, hallucinated coordinates, and geometric fidelity is the remit of [Evaluation & Benchmarking for Spatial LLMs](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/).

## Production Engineering Standards

1. Every retrieved chunk carries an explicit CRS tag; a chunk with an unresolved reference frame is never placed in the context window.
2. The vector store applies a bounding-box and metadata pre-filter before ANN ranking — never a post-filter that silently collapses recall.
3. Geometry literals (WKT, GeoJSON) are never split across chunk boundaries; a splitter that opens a ring must close it in the same unit.
4. Reranking fuses semantic score with a geographic proximity term; distance is a first-class ranking signal, not an afterthought.
5. Generation is gated: any coordinate the model emits outside the union of retrieved bounding boxes triggers a deterministic fallback and is not returned.
6. Retrieval latency, geographic recall, and grounding rate are emitted as observability KPIs on every request, with alert thresholds wired into CI regression harnesses.
7. Token budgets account for geometry verbosity; high-vertex polygons are downsampled or summarised before entering the prompt, never truncated mid-coordinate.
