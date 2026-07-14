# How to Tokenize Polygon Boundaries for Transformer Models

Tokenizing polygon boundaries for sequence-based architectures requires a deterministic, topology-preserving serialization pipeline. Unlike point or line geometries, polygons introduce nested ring structures, variable-length coordinate arrays, and strict topological invariants (e.g., non-self-intersection, correct winding order, hole containment). The primary engineering challenge is converting continuous, CRS-dependent spatial primitives into discrete, fixed-vocabulary sequences without introducing geometric drift, context window overflow, or invalid topology during autoregressive generation. This workflow sits at the intersection of spatial data engineering and sequence modeling, demanding rigorous validation, quantization control, and fallback routing for malformed inputs.

## 1. Coordinate Reference System Normalization & Adaptive Quantization

Raw coordinate arrays cannot be directly tokenized. Latitude/longitude values span arbitrary ranges, exhibit floating-point precision artifacts, and lack scale invariance across regions. The first mandatory step is CRS normalization, which projects all inputs into a consistent, locally metric-preserving CRS before applying deterministic quantization.

A common failure mode in production is using uniform bit-depth quantization across global extents. This causes severe precision loss near the equator and coordinate aliasing at high latitudes. The mitigation strategy employs adaptive quantization based on the geometry's bounding box and target spatial resolution.

```python
import numpy as np
from shapely.geometry import Polygon, box
from shapely.validation import make_valid, explain_validity
from shapely.ops import transform
import pyproj
from typing import Tuple, Optional

class QuantizationError(Exception):
    pass

def normalize_and_quantize(
    geom: Polygon,
    source_crs: str = "EPSG:4326",
    target_crs: str = "EPSG:3857",
    precision_bits: int = 14
) -> np.ndarray:
    """
    CRS normalization + adaptive quantization for polygon boundaries.
    Includes explicit validation, error routing, and coordinate bounds checking.
    """
    # 1. Topology & Type Validation
    if not isinstance(geom, Polygon):
        raise TypeError(f"Expected shapely.geometry.Polygon, got {type(geom).__name__}")

    if not geom.is_valid:
        reason = explain_validity(geom)
        try:
            geom = make_valid(geom)
            if not isinstance(geom, Polygon):
                raise QuantizationError(f"make_valid changed geometry type: {reason}")
        except Exception as e:
            raise QuantizationError(f"Failed to auto-repair invalid geometry: {reason}") from e

    # 2. CRS Projection using shapely.ops.transform (pyproj.transform was removed in pyproj 3.0)
    try:
        if source_crs != target_crs:
            transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
            geom = transform(transformer.transform, geom)
    except pyproj.exceptions.ProjError as e:
        raise QuantizationError(f"CRS transformation failed: {e}")

    # 3. Coordinate Validation & Degeneracy Check
    coords = np.array(geom.exterior.coords)
    if coords.shape[0] < 4:
        raise QuantizationError("Polygon must have at least 3 unique vertices (closed ring)")

    min_x, min_y, max_x, max_y = geom.bounds
    scale = max(max_x - min_x, max_y - min_y)
    if np.isclose(scale, 0.0):
        raise QuantizationError("Degenerate geometry with zero spatial extent")

    # 4. Adaptive Quantization
    max_val = 2**precision_bits - 1
    if scale > max_val:
        raise QuantizationError(
            f"Geometry scale ({scale:.2f}) exceeds quantization capacity for {precision_bits} bits"
        )

    quantization_factor = max_val / scale
    quantized = np.round((coords - [min_x, min_y]) * quantization_factor).astype(np.int32)

    # 5. Post-Quantization Bounds Validation
    if np.any(quantized < 0) or np.any(quantized > max_val):
        raise QuantizationError("Quantized coordinates exceed vocabulary bounds")

    return quantized
```

**Pipeline Integration Notes:**
- Wrap this function in a Pydantic model or FastAPI dependency to enforce strict input schemas.
- Implement a dead-letter queue (DLQ) for geometries that trigger `QuantizationError`, routing them to a fallback rasterization service.
- Log `scale` and `quantization_factor` metrics to monitor precision drift across geographic regions.

## 2. Topology-Aware Ring Serialization & Winding Order Enforcement

Transformers process flat sequences, but polygons contain hierarchical ring structures. Exterior rings must follow counter-clockwise (CCW) winding order, while interior rings (holes) must be clockwise (CW) per [OGC Simple Features](https://www.ogc.org/standards/sfa) specifications. Failing to enforce this causes ambiguity during autoregressive decoding.

```python
from shapely.geometry import LinearRing
from shapely.ops import orient
import json

def serialize_rings(geom: Polygon, max_tokens: int = 4096) -> dict:
    """
    Flattens exterior and interior rings into a token-ready dictionary.
    Enforces winding order and validates hole containment.
    """
    try:
        # Force correct winding order (CCW exterior, CW holes per OGC SFS)
        oriented_geom = orient(geom, sign=1.0)
    except Exception as e:
        raise ValueError(f"Winding order correction failed: {e}")

    # Validate hole containment
    exterior = LinearRing(oriented_geom.exterior.coords)
    for i, interior in enumerate(oriented_geom.interiors):
        hole_ring = LinearRing(interior.coords)
        # Use the polygon's interior containment check (more reliable than ring-level)
        hole_poly = Polygon(interior.coords)
        exterior_poly = Polygon(oriented_geom.exterior.coords)
        if not exterior_poly.contains(hole_poly):
            raise ValueError(f"Interior ring {i} is not fully contained within exterior ring")
        if hole_poly.exterior.crosses(exterior_poly.exterior):
            raise ValueError(f"Interior ring {i} intersects exterior boundary")

    # Flatten into sequence with control tokens
    sequence = []
    token_count = 0

    # Add exterior ring
    for x, y in oriented_geom.exterior.coords:
        sequence.extend([int(x), int(y)])
        token_count += 2
    sequence.append(-1)  # RING_SEPARATOR token
    token_count += 1

    # Add interior rings
    for interior in oriented_geom.interiors:
        for x, y in interior.coords:
            sequence.extend([int(x), int(y)])
            token_count += 2
        sequence.append(-1)
        token_count += 1

    if token_count > max_tokens:
        raise OverflowError(f"Serialized sequence ({token_count}) exceeds max_tokens ({max_tokens})")

    return {
        "sequence": sequence,
        "length": token_count,
        "num_holes": len(oriented_geom.interiors)
    }
```

**Pipeline Integration Notes:**
- Replace the magic number `-1` with a dedicated `SpecialTokens` enum mapped to your tokenizer vocabulary.
- Integrate with HuggingFace `PreTrainedTokenizerFast` to register spatial control tokens before training begins.
- Add unit tests using known pathological geometries (e.g., bowties, overlapping holes) to verify `orient` and containment checks.

## 3. Discrete Vocabulary Mapping & Context Window Optimization

Once coordinates are quantized and rings serialized, they must be mapped to a fixed vocabulary. Directly feeding raw integers into an embedding layer causes out-of-distribution (OOD) activations. Instead, use a structured embedding strategy that preserves spatial locality while respecting context window limits.

```python
import torch
from torch.nn import Embedding

class SpatialCoordinateEmbedding(torch.nn.Module):
    """
    Maps quantized (x, y) pairs to dense embeddings with explicit padding/truncation.
    """
    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_token_id = vocab_size - 1
        self.coord_embedding = Embedding(vocab_size, embed_dim, padding_idx=self.pad_token_id)
        self.register_buffer("pad_token", torch.tensor(self.pad_token_id))

    def forward(self, sequences: torch.Tensor) -> tuple:
        # 1. Context Window Validation
        if sequences.dim() != 2:
            raise ValueError("Input must be 2D tensor [batch, seq_len]")

        batch_size, seq_len = sequences.shape
        if seq_len > self.max_seq_len:
            # Truncate from the end to preserve start-of-polygon topology
            sequences = sequences[:, :self.max_seq_len]

        # 2. Coordinate Range Validation
        if torch.any(sequences < 0) or torch.any(sequences >= self.vocab_size):
            raise ValueError(f"Token IDs out of vocabulary bounds [0, {self.vocab_size-1}]")

        # 3. Embedding Lookup
        embeddings = self.coord_embedding(sequences)

        # 4. Padding Mask Generation
        mask = (sequences != self.pad_token_id).float()
        return embeddings, mask
```

**Pipeline Integration Notes:**
- Precompute a lookup table for `vocab_size` during data preprocessing to avoid runtime OOV errors.
- Implement dynamic chunking for polygons exceeding `max_seq_len`: split rings at midpoint, inject `CONTINUE` tokens, and reassemble during post-processing.
- Monitor embedding norms during training; sudden spikes indicate quantization aliasing or vocabulary misalignment.

## 4. Production Pipeline Integration & Safety Guardrails

Deploying a polygon tokenizer in production requires robust fallback routing, continuous validation, and strict isolation between training and inference data flows. Geospatial models are highly sensitive to coordinate drift, which can silently corrupt downstream spatial reasoning tasks.

### Fallback Routing for Malformed Inputs
Not all geometries can be safely serialized. Implement a tiered routing strategy:
1. **Primary Path:** Quantization → Ring Serialization → Token Mapping.
2. **Secondary Path (Repair):** If `make_valid` returns a non-Polygon type, route to a vector-to-raster converter. Raster masks bypass topological constraints and can be tokenized as flattened pixel grids.
3. **Tertiary Path (Reject):** Log to DLQ with full geometry provenance (source ID, CRS, failure reason). Never silently drop or approximate invalid boundaries.

### Pipeline Integration Checklist
- **Pre-Flight Validation:** Run `shapely.validation.explain_validity` and CRS checks in a dedicated validation microservice before ingestion.
- **Deterministic Seeding:** Fix random seeds for any stochastic repair operations to ensure reproducible tokenization across distributed workers.
- **Context Window Budgeting:** Reserve 15% of the sequence length for control tokens (`START`, `END`, `HOLE`, `PAD`). This prevents autoregressive models from generating malformed termination sequences.
- **Monitoring:** Track `quantization_error_meters` (difference between original and de-quantized coordinates) and `topology_violation_rate`. Alert if either exceeds 0.5% over a rolling 24-hour window.

### Final Next Steps for Platform Teams
1. **CI/CD Hook:** Add a `pytest` suite that validates tokenization round-trip accuracy (`original_geom ≈ dequantized_geom`) across EPSG:4326, EPSG:3857, and local UTM zones.
2. **Batch Processing:** Wrap the tokenizer in an Apache Beam or Ray Data pipeline to parallelize quantization and ring serialization across multi-node clusters.
3. **Safety Audit:** Conduct adversarial testing with synthetic geometries (e.g., extreme aspect ratios, near-zero area, overlapping vertices) to verify error routing and fallback mechanisms.
4. **Model Alignment:** Ensure your transformer's positional encoding scheme accounts for the 2D coordinate interleaving (`[x0, y0, x1, y1, ...]`). Standard 1D positional encodings degrade spatial locality; consider RoPE or ALiBi variants tuned for coordinate sequences.

For deeper architectural guidance, consult [Geometry Tokenization Strategies](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/), [PyProj documentation](https://pyproj4.github.io/pyproj/stable/), and [Shapely validation docs](https://shapely.readthedocs.io/en/stable/manual.html).

By enforcing strict coordinate validation, adaptive quantization, and explicit fallback routing, you can safely tokenize polygon boundaries without sacrificing topological integrity or overwhelming transformer context windows.
