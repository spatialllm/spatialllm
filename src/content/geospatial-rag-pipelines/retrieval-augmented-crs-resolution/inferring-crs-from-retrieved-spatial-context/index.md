# Inferring CRS from Retrieved Spatial Context

When a retrieved neighbour lacks an explicit EPSG code, the correct coordinate reference system has to be inferred from circumstantial evidence: the magnitude of its coordinates, national-grid hints in its text, and the provenance of the dataset it came from. This guide implements a scoring function that turns those signals into an `(epsg, confidence)` pair, verified by a `pyproj` round-trip, for the retrieval stage of [retrieval-augmented CRS resolution](https://www.spatialllm.org/geospatial-rag-pipelines/retrieval-augmented-crs-resolution/).

Inference runs after retrieval but before the geometry reaches the reasoning layer. Its job is narrow: convert weak, overlapping hints into a single defensible EPSG with a calibrated confidence, or decline cleanly. It never guesses in silence — an unresolved neighbour returns a low confidence that the caller can gate on rather than an unmarked assumption.

## When to Use This Approach

Reach for heuristic inference only when a declared CRS is genuinely absent. If the source carries an SRID or a WKT `AUTHORITY` node, trust it and skip inference entirely. Use inference for gazetteer snippets, scanned survey extracts, and community datasets where the CRS lives in convention rather than metadata.

| Signal available | Recommended path | Confidence ceiling |
| --- | --- | --- |
| Declared SRID / EPSG in header | Trust directly, validate only | 1.00 |
| National-grid text hint (e.g. grid letters) | Grid-alias lookup + round-trip | 0.90 |
| Coordinate magnitude only | Magnitude-range heuristic | 0.70 |
| Conflicting or no signal | Decline, low-confidence fallback | 0.00 |

Coordinate magnitude alone is the weakest signal because projected ranges overlap between neighbouring zones, so it caps confidence below the threshold that most consumers require for metric answers.

## Implementation

The inference function combines a magnitude test, a grid-alias hint, and dataset provenance into a weighted score, then confirms the chosen EPSG with a validated `Transformer` round-trip. It returns `(epsg, confidence)` and falls back to `(4326, 0.0)` if nothing survives. This complements the ingestion-time [mixed-CRS normalization](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/normalizing-mixed-crs-data-before-llm-ingestion/) routine, which handles inputs whose CRS is already known.

```python
from dataclasses import dataclass
from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError
import logging

log = logging.getLogger("crs_inference")

# Coarse plausibility envelopes (min_x, min_y, max_x, max_y) per candidate CRS.
MAGNITUDE_ENVELOPES = {
    4326: (-180.0, -90.0, 180.0, 90.0),      # geographic degrees
    27700: (0.0, 0.0, 700000.0, 1300000.0),  # British National Grid metres
    32633: (166000.0, 0.0, 834000.0, 9330000.0),  # UTM 33N metres
}

@dataclass(frozen=True)
class SpatialContext:
    sample_xy: tuple[float, float]
    grid_hint_epsg: int | None   # from national-grid letters, if parsed
    provenance_epsg: int | None  # EPSG the source dataset usually publishes in

def _roundtrip_ok(epsg: int, xy: tuple[float, float], tol_m: float = 1.0) -> bool:
    """Confirm the EPSG survives a forward+inverse transform within tolerance."""
    try:
        fwd = Transformer.from_crs(CRS.from_epsg(epsg), CRS.from_epsg(4326), always_xy=True)
        inv = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True)
    except CRSError as exc:
        log.info("EPSG:%s failed to construct: %s", epsg, exc)
        return False
    lon, lat = fwd.transform(*xy)
    x2, y2 = inv.transform(lon, lat)
    if any(v != v for v in (lon, lat, x2, y2)):   # NaN from out-of-domain input
        return False
    err = ((x2 - xy[0]) ** 2 + (y2 - xy[1]) ** 2) ** 0.5
    return err <= tol_m

def infer_crs(ctx: SpatialContext, threshold: float = 0.75) -> tuple[int, float]:
    """Return (epsg, confidence). Falls back to (4326, 0.0) when unresolved."""
    scores: dict[int, float] = {}
    x, y = ctx.sample_xy

    # 1. Magnitude plausibility — weak evidence, contributes at most 0.55.
    for epsg, (mnx, mny, mxx, mxy) in MAGNITUDE_ENVELOPES.items():
        if mnx <= x <= mxx and mny <= y <= mxy:
            scores[epsg] = scores.get(epsg, 0.0) + 0.55

    # 2. National-grid text hint — strong, adds 0.35.
    if ctx.grid_hint_epsg is not None:
        scores[ctx.grid_hint_epsg] = scores.get(ctx.grid_hint_epsg, 0.0) + 0.35

    # 3. Dataset provenance agreement — corroborating, adds 0.25.
    if ctx.provenance_epsg is not None:
        scores[ctx.provenance_epsg] = scores.get(ctx.provenance_epsg, 0.0) + 0.25

    if not scores:
        log.warning("No CRS signal for %s; falling back to EPSG:4326", ctx.sample_xy)
        return (4326, 0.0)

    # Rank, then require a validated round-trip before committing.
    for epsg, raw in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        confidence = round(min(1.0, raw), 3)
        if confidence >= threshold and _roundtrip_ok(epsg, ctx.sample_xy):
            return (epsg, confidence)

    log.info("Best signal below threshold or failed round-trip; falling back")
    return (4326, 0.0)
```

## Validation & Testing

1. **Known-CRS recovery.** Feed a British National Grid coordinate with a grid hint and assert `infer_crs` returns `(27700, c)` with `c >= 0.75`.
2. **Ambiguous magnitude declines.** A coordinate that only satisfies a magnitude envelope must return confidence at or below the threshold, never a high-confidence commitment.
3. **Round-trip enforcement.** Monkeypatch a candidate to fail the transform and assert the function skips it and reaches the `(4326, 0.0)` fallback rather than raising.

## Gotchas & Edge Cases

**Overlapping envelopes.** UTM zones and national grids share metre-scale ranges, so magnitude can match two CRSs at once. Fix: keep magnitude weight below the acceptance threshold so a corroborating hint or provenance signal is always required to commit.

**NaN from out-of-domain transforms.** `pyproj` returns `inf`/`nan` rather than raising when a coordinate is far outside a projection's domain. Fix: the explicit NaN check in `_roundtrip_ok` — never assume the transform raised.

**Provenance drift.** A dataset's habitual EPSG can change between publication years. Fix: treat provenance as corroborating weight only, never as a standalone determination, and re-check the mapping on re-index.

**Axis-order confusion.** Omitting `always_xy=True` silently swaps easting and northing, passing the round-trip while producing a mirrored position. Fix: pin `always_xy=True` on every `Transformer`, as shown.

## Related

- Up to the area guide: [Retrieval-Augmented CRS Resolution](https://www.spatialllm.org/geospatial-rag-pipelines/retrieval-augmented-crs-resolution/)
- Prerequisite: [Normalizing Mixed-CRS Data Before LLM Ingestion](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/normalizing-mixed-crs-data-before-llm-ingestion/)
- Peer technique: [Spatial Context Retrieval and Reranking](https://www.spatialllm.org/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/)
