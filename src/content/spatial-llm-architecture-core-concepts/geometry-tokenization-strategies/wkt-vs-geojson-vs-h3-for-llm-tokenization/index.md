# WKT vs GeoJSON vs H3 for LLM Tokenization

Every geometry you hand a language model is first flattened into text, and the encoding you pick — Well-Known Text, GeoJSON, or an H3 cell set — decides how many tokens it costs, how much precision survives, and how reliably the model reasons about it. Choosing wrong wastes context on punctuation the model does not need or discards the topology it does. This guide compares the three encodings head to head and gives a selector that picks the cheapest one under a token budget. It sits in the [geometry tokenization strategies](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/) area of the [spatial LLM architecture](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/) work.

There is no universally best encoding. The right answer depends on whether you need exact boundaries, area buckets, or just a locality hint — and on how much of the context window you can afford to spend on geometry versus the actual task.

## When to Use This Approach

Compare encodings whenever geometry shares the context window with instructions, retrieved passages, or tool schemas — that is, almost always. If geometry is the only payload and the window is huge, encoding choice barely matters; the moment you are budgeting tokens, it dominates.

| Dimension | WKT | GeoJSON | H3 |
| --- | --- | --- | --- |
| Token cost | Low–medium (terse) | High (verbose keys/braces) | Low (short cell ids) |
| Precision control | Exact, decimal-tunable | Exact, decimal-tunable | Quantized to resolution |
| Topology fidelity | Full (rings, holes) | Full (rings, holes) | Lossy (cell approximation) |
| Model familiarity | Moderate | High (JSON everywhere) | Low (opaque hex ids) |
| Best for | Compact exact geometry | Structured multi-attribute features | Coarse area / spatial join keys |

WKT is the terse exact option: `POLYGON((...))` with no repeated keys. GeoJSON is the most model-familiar because models have seen enormous amounts of JSON, but its braces, quotes, and `"coordinates"` keys inflate token count sharply on large geometries. H3 replaces boundaries with a set of hierarchical hex cell ids — extremely compact and ideal as spatial join keys, but it quantizes the shape to a resolution grid and cannot represent an exact boundary. Reach for H3 when "which cells does this cover" beats "what is the precise edge," and pair it with [quadtree context management](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/managing-llm-context-limits-with-quadtree-indexes/) when the area is large.

## Implementation

The selector encodes a geometry three ways, estimates token cost with a simple heuristic — no external tokenizer is imported, so the estimate stays dependency-free and deterministic — and returns the cheapest encoding that fits the budget, falling back to the most compact available option when none fit. The estimate treats runs of digits, coordinate punctuation, and words as separate tokens, which tracks real subword tokenizers closely enough for budgeting.

Under a per-request budget $B$ tokens with $R$ reserved for the prompt, the geometry must satisfy $t_{\text{geom}} \le B - R$; the selector picks $\arg\min_{e}\, t_e$ subject to that bound.

```python
import json
import re
from shapely.geometry import mapping
from shapely import wkt as shp_wkt

_TOKEN_RE = re.compile(r"\d+|[A-Za-z_]+|[^\s\w]")

def estimate_tokens(text: str) -> int:
    """Heuristic subword estimate: digit-runs, words, and single symbols.
    Deliberately NOT a real tokenizer — deterministic and dependency-free."""
    if not text:
        return 0
    return len(_TOKEN_RE.findall(text))

def encode_wkt(geom, ndigits: int = 5) -> str:
    return shp_wkt.dumps(geom, rounding_precision=ndigits)

def encode_geojson(geom, ndigits: int = 5) -> str:
    def _round(obj):
        if isinstance(obj, float):
            return round(obj, ndigits)
        if isinstance(obj, list):
            return [_round(o) for o in obj]
        if isinstance(obj, dict):
            return {k: _round(v) for k, v in obj.items()}
        return obj
    return json.dumps(_round(mapping(geom)), separators=(",", ":"))

def encode_h3(geom, resolution: int) -> str:
    """Cover the geometry with H3 cells. Falls back to WKT if h3 is unavailable
    or the geometry cannot be celled (e.g. sub-cell sliver)."""
    try:
        import h3
    except ImportError:
        return None  # caller treats None as "encoding unavailable"
    try:
        gj = mapping(geom)
        cells = h3.polygon_to_cells(h3.geo_to_h3shape(gj), resolution)
    except Exception:
        return None
    if not cells:
        return None
    return " ".join(sorted(cells))

def choose_encoding(geom, budget_tokens: int, reserved: int = 0,
                    ndigits: int = 5, h3_resolution: int = 7,
                    prefer_topology: bool = True):
    """Return (encoding_name, text, est_tokens) for the cheapest option that fits.
    Deterministic fallback: if none fit, return the smallest encoding anyway
    and flag it so the caller can truncate or route elsewhere."""
    if budget_tokens <= reserved:
        raise ValueError("reserved tokens must be below the total budget")
    if geom.is_empty:
        raise ValueError("cannot encode an empty geometry")

    limit = budget_tokens - reserved
    candidates = {
        "wkt": encode_wkt(geom, ndigits),
        "geojson": encode_geojson(geom, ndigits),
    }
    h3_text = encode_h3(geom, h3_resolution)
    if h3_text is not None and not prefer_topology:
        candidates["h3"] = h3_text  # lossy: only offer when topology isn't required

    scored = {name: (text, estimate_tokens(text)) for name, text in candidates.items()}
    fitting = {n: v for n, v in scored.items() if v[1] <= limit}

    if fitting:
        name = min(fitting, key=lambda n: fitting[n][1])
        text, cost = fitting[name]
        return name, text, cost

    # Nothing fits: return the globally smallest and let the caller decide.
    name = min(scored, key=lambda n: scored[n][1])
    text, cost = scored[name]
    return f"{name}:OVER_BUDGET", text, cost
```

Set `prefer_topology=False` only when the downstream task genuinely tolerates quantization — a spatial join or a "which region" question — because H3 discards the exact boundary. For exact work the selector chooses between WKT and GeoJSON, and WKT almost always wins on token cost. When even the cheapest exact encoding blows the budget, treat `OVER_BUDGET` as a signal to simplify the geometry first, following [polygon-boundary tokenization](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/how-to-tokenize-polygon-boundaries-for-transformer-models/).

## Validation & Testing

Pin the ranking behaviour and the budget guarantee:

```python
from shapely.geometry import Polygon

def test_wkt_cheaper_than_geojson_for_polygons():
    poly = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    assert estimate_tokens(encode_wkt(poly)) < estimate_tokens(encode_geojson(poly))

def test_selector_respects_budget():
    poly = Polygon([(i, i % 2) for i in range(200)] + [(0, 0)])
    name, text, cost = choose_encoding(poly, budget_tokens=50, reserved=0)
    assert cost > 0
    # Either it fits, or it is explicitly flagged — never a silent overflow.
    assert cost <= 50 or name.endswith("OVER_BUDGET")

def test_empty_geometry_rejected():
    import pytest
    with pytest.raises(ValueError):
        choose_encoding(Polygon(), budget_tokens=100)
```

Calibrate the heuristic once against the real tokenizer you deploy with: encode a sample of geometries, compare `estimate_tokens` to the true count, and confirm the estimate stays within a stable ratio. You are budgeting, not billing, so a consistent 10–15% offset is fine as long as it does not flip the ranking.

## Gotchas & Edge Cases

**GeoJSON whitespace.** `json.dumps` with default separators adds a space after every comma and colon, inflating the estimate by tens of percent. Always pass `separators=(",", ":")` so the comparison reflects a compact payload.

**H3 resolution is a cliff.** Bumping resolution by one multiplies the cell count by roughly seven. An area that is 20 cells at resolution 7 can be 140 at resolution 8 — H3 stops being the cheap option fast. Estimate cell count before committing.

**Precision inflation.** Serializing full float64 coordinates emits 15+ digits per number, each a token. Round to the precision your task actually needs (`ndigits`) — five decimals is roughly a metre and usually plenty.

**Model familiarity vs cost.** WKT is cheaper but some models parse GeoJSON more reliably. If accuracy drops on WKT, the token savings are false economy; benchmark task accuracy, not just token count, before standardizing.

## Related

- Up to the area overview: [Geometry Tokenization Strategies](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/)
- [How to Tokenize Polygon Boundaries for Transformer Models](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/how-to-tokenize-polygon-boundaries-for-transformer-models/)
- [Context Window Optimization for Maps](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/)
