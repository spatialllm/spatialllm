# Retrieval-Augmented CRS Resolution

Ambiguous geographic mentions rarely announce their coordinate reference system. A gazetteer name, a six-figure national grid reference, or a bare easting/northing pair carries an implicit CRS that a language model cannot recover from the token stream alone. Retrieval-augmented CRS resolution fetches supporting evidence from a spatial knowledge index and pins down the correct EPSG code before any reasoning begins.

This area sits inside the [Geospatial RAG Pipelines](https://www.spatialllm.org/geospatial-rag-pipelines/) discipline and addresses one failure mode precisely: an agent that silently assumes WGS84 for coordinates that were actually recorded in a projected local grid, producing answers displaced by hundreds of metres. The retrieval layer supplies datum hints, provenance, and neighbouring geometry that make the CRS decision explicit and auditable rather than guessed. It builds directly on [coordinate reference system normalization](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/), extending that ingestion-time gate with a retrieval step for the harder case where the CRS is not declared anywhere in the input.

<figure class="diagram">
<svg viewBox="0 0 860 300" role="img" aria-labelledby="racr-t racr-d" xmlns="http://www.w3.org/2000/svg">
  <title id="racr-t">Retrieval-augmented CRS resolution flow</title>
  <desc id="racr-d">An ambiguous mention is embedded, candidate CRS records are retrieved from a knowledge index, each candidate is validated with pyproj, the best is scored, and a confidence-tagged EPSG resolves the geometry before the LLM reasons.</desc>
  <defs>
    <marker id="racr-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
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
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#racr-arrow)">
    <line x1="166" y1="95" x2="181" y2="95"/>
    <line x1="334" y1="95" x2="349" y2="95"/>
    <line x1="502" y1="95" x2="517" y2="95"/>
    <line x1="670" y1="95" x2="685" y2="95"/>
  </g>
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#racr-arrow)">
    <line x1="90" y1="137" x2="90" y2="202"/>
    <line x1="258" y1="137" x2="258" y2="202"/>
    <line x1="426" y1="137" x2="426" y2="202"/>
    <line x1="594" y1="137" x2="594" y2="202"/>
    <line x1="762" y1="137" x2="762" y2="202"/>
  </g>
  <rect x="15" y="205" width="822" height="58" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/>
  <g fill="#1f2937" font-size="13" text-anchor="middle">
    <text x="90" y="90"><tspan x="90" dy="0">Ambiguous</tspan><tspan x="90" dy="16">mention</tspan></text>
    <text x="258" y="90"><tspan x="258" dy="0">Retrieve</tspan><tspan x="258" dy="16">candidates</tspan></text>
    <text x="426" y="90"><tspan x="426" dy="0">Validate</tspan><tspan x="426" dy="16">via pyproj</tspan></text>
    <text x="594" y="90"><tspan x="594" dy="0">Score</tspan><tspan x="594" dy="16">confidence</tspan></text>
    <text x="762" y="90"><tspan x="762" dy="0">Resolved</tspan><tspan x="762" dy="16">EPSG</tspan></text>
  </g>
  <text x="426" y="240" fill="#1f2937" font-size="15" font-weight="600" text-anchor="middle">Confidence-tagged CRS before the LLM reasons</text>
  <text x="426" y="287" fill="#5b6471" font-size="12" text-anchor="middle">Fallback: EPSG:4326 · low-confidence flag · human review queue</text>
</svg>
</figure>

## Foundational Principles

**Never let the model choose the CRS.** A language model will happily emit `EPSG:4326` for a British National Grid reference because both look plausible in text. CRS selection must be a deterministic function of retrieved evidence and `pyproj` validation, not a generated token. The model reasons over geometry that has already been resolved.

**Every resolution carries a confidence.** A CRS that survives a validated round-trip and agrees with retrieved provenance earns high confidence; one that falls back earns a low-confidence flag that propagates downstream. Callers must be able to gate on it — a distance query answered from a low-confidence CRS is worse than a refusal.

**The fallback is EPSG:4326, and it is loud.** When retrieval returns nothing usable or validation rejects every candidate, the pipeline degrades to WGS84 and tags the result so no consumer mistakes a guess for a determination. Silence is the failure; a flagged fallback is the safe path.

## Step-by-Step Implementation Pipeline

### 1. Retrieve candidate CRS records from the knowledge index

The knowledge index stores documents keyed by place name, grid-system alias, and datum authority, each carrying a candidate EPSG and a coordinate-plausibility envelope. Retrieval returns the top matches for the ambiguous mention; downstream steps never trust these blindly. The heuristics that turn raw neighbours into a scored EPSG are detailed in [inferring CRS from retrieved spatial context](https://www.spatialllm.org/geospatial-rag-pipelines/retrieval-augmented-crs-resolution/inferring-crs-from-retrieved-spatial-context/).

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
    """Fetch candidate CRS records; never raise — return [] so callers can fall back."""
    try:
        hits = list(index_query(mention, k))
    except Exception as exc:                      # index outage, timeout, bad response
        log.warning("CRS index query failed for %r: %s", mention, exc)
        return []
    # Deduplicate by EPSG, keeping the strongest retrieval weight.
    best: dict[int, CRSCandidate] = {}
    for c in hits:
        if c.epsg not in best or c.weight > best[c.epsg].weight:
            best[c.epsg] = c
    return sorted(best.values(), key=lambda c: c.weight, reverse=True)
```

### 2. Validate each candidate with pyproj

A retrieved EPSG is only a hypothesis until `pyproj` confirms it constructs, is not deprecated, and places the mention's coordinates inside its area of use. Validation is where hallucinated or stale codes are rejected — a step the [normalization gate](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/) assumes has already run for declared inputs.

```python
from pyproj import CRS
from pyproj.exceptions import CRSError

def crs_is_usable(epsg: int, sample_xy: tuple[float, float] | None) -> bool:
    """True only if the EPSG constructs, is current, and can host sample_xy."""
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

### 3. Resolve with a deterministic fallback

The top-level `resolve_crs` orchestrates retrieval, validation, and scoring, always returning a structured result — never an exception. When no candidate survives, it degrades to `EPSG:4326` with a low-confidence flag so the reasoning layer and the [fallback routing](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/fallback-routing-for-geospatial-queries/) layer both see the uncertainty.

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
    log.warning("No usable CRS for %r; falling back to EPSG:4326", mention)
    return ResolvedCRS(4326, 0.0, True, "deterministic fallback — no candidate validated")
```

## Failure Modes & Root Causes

**Plausible-but-wrong EPSG.** Retrieval surfaces a neighbouring region's grid whose coordinate ranges overlap the true one. Root cause: coordinate magnitude alone is ambiguous. Mitigation: require provenance agreement, not just an area-of-use hit, before granting high confidence.

**Deprecated datum codes.** Legacy catalogues store superseded EPSG codes that still construct but shift positions by tens of metres. Root cause: stale index content. Mitigation: the `is_deprecated` gate in step 2, plus periodic re-indexing against the current registry.

**Silent WGS84 assumption.** The most damaging mode — the CRS is guessed as 4326 and no flag is raised. Root cause: treating "no evidence" as "geographic". Mitigation: the fallback in step 3 is always tagged `low_confidence=True`.

## Production Validation Protocols

1. **Assert structured returns.** `resolve_crs` must never raise for adversarial mentions; a CI fuzz test feeds empty strings, emoji, and malformed grid refs and asserts a `ResolvedCRS` is returned every time.
2. **Round-trip gate.** For every non-fallback resolution, transform the sample coordinate to `EPSG:4326` and back; reject if positional error exceeds one metre.
3. **Confidence monotonicity.** A validated candidate must score strictly above the `0.0` fallback; assert `confidence > 0` whenever `low_confidence is False`.
4. **Fallback-rate KPI.** Alert when the share of `epsg == 4326 and low_confidence` results exceeds a rolling baseline — a spike signals index drift or an outage.
5. **Provenance logging.** Persist `rationale` and source with each resolution for audit and replay.

## Related

- Up to the area overview: [Geospatial RAG Pipelines](https://www.spatialllm.org/geospatial-rag-pipelines/)
- Foundational concept: [Coordinate Reference System Normalization](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/)
- In-depth technique: [Inferring CRS from Retrieved Spatial Context](https://www.spatialllm.org/geospatial-rag-pipelines/retrieval-augmented-crs-resolution/inferring-crs-from-retrieved-spatial-context/)
