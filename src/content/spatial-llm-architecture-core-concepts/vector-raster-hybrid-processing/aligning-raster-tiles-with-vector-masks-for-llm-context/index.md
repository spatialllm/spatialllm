# Aligning Raster Tiles with Vector Masks for LLM Context

When a multimodal model reasons over an image chip and a vector boundary at the same time, the two must describe the same patch of ground. If the raster tile and the vector area of interest disagree on projection, extent, or pixel grid, the model sees an image of one place captioned with the geometry of another — and confidently hallucinates the relationship. This guide covers clipping and aligning raster tiles to vector masks so the georeferencing stays consistent before anything reaches the model. It belongs to the [vector-raster hybrid processing](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/vector-raster-hybrid-processing/) area of the [spatial LLM architecture](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/) work.

The failure is quiet: nothing errors, the model just receives a subtly misregistered pair and reasons about the wrong pixels. Alignment has to be an explicit, validated step — verify the CRS agreement, clip on a shared window, and handle the case where the mask and the tile do not actually overlap.

## When to Use This Approach

Align tiles to masks whenever an image chip and a vector geometry are fed to the same model turn — captioning a parcel from imagery, extracting land cover inside an administrative boundary, or grounding a "what is at this location" question. If the model only ever sees the raster or only the vector, you can skip alignment; the cost is entirely in the hybrid case.

| Situation | Alignment action | Why |
| --- | --- | --- |
| Mask CRS = raster CRS | Clip on native grid | No resampling, no drift |
| Mask CRS ≠ raster CRS | Reproject mask to raster CRS, then clip | Cheaper and lossless vs resampling the raster |
| Mask outside raster footprint | Empty-intersection fallback | Never emit a blank chip as if valid |
| Mask smaller than one pixel | Buffer to minimum window | Sub-pixel masks yield degenerate chips |

The guiding rule: reproject the vector, not the raster, whenever you can. Vector reprojection is exact and cheap; resampling a raster into the mask's CRS introduces interpolation error into the very pixels the model will read. This depends on knowing both CRSs, so [CRS normalization](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/) must run before alignment.

## Implementation

The function below uses rasterio's window and mask machinery. It asserts CRS agreement (reprojecting the mask into the raster's CRS when they differ), computes the pixel window from the mask bounds, reads only that window, and applies the vector as a boolean mask. An empty or non-overlapping intersection returns an explicit sentinel result rather than a blank array, so callers never mistake "no data here" for "here is the data."

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np
import rasterio
from rasterio.mask import raster_geometry_mask
from rasterio.windows import from_bounds, Window
from rasterio.errors import WindowError
from shapely.geometry import shape, box, mapping
from shapely.ops import transform as shp_transform
import pyproj

@dataclass
class AlignedChip:
    array: Optional[np.ndarray]      # (bands, rows, cols) or None
    transform: Optional[object]      # affine of the clipped window
    empty: bool                      # True => mask and tile do not overlap
    reason: str = ""

def _reproject_geom(geom, src_crs, dst_crs):
    if src_crs == dst_crs:
        return geom
    tr = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shp_transform(tr.transform, geom)

def align_tile_to_mask(raster_path: str, mask_geom, mask_crs) -> AlignedChip:
    """Clip a raster to a vector mask with CRS reconciliation and a
    deterministic empty-intersection fallback."""
    if mask_geom.is_empty:
        return AlignedChip(None, None, empty=True, reason="empty mask geometry")

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            return AlignedChip(None, None, empty=True, reason="raster has no CRS")

        # Reproject the vector into the raster grid — never resample the raster.
        geom = _reproject_geom(mask_geom, mask_crs, src.crs)

        raster_footprint = box(*src.bounds)
        if not geom.intersects(raster_footprint):
            return AlignedChip(None, None, empty=True,
                               reason="mask lies outside raster footprint")

        # Restrict to the overlap so we never read the whole scene.
        clipped_geom = geom.intersection(raster_footprint)
        if clipped_geom.is_empty or clipped_geom.area == 0:
            return AlignedChip(None, None, empty=True, reason="zero-area overlap")

        try:
            window = from_bounds(*clipped_geom.bounds, transform=src.transform)
            window = window.round_offsets().round_lengths()
        except WindowError as e:
            return AlignedChip(None, None, empty=True, reason=f"window error: {e}")

        # Guard against sub-pixel masks collapsing to a zero-size window.
        if window.width < 1 or window.height < 1:
            window = Window(window.col_off, window.row_off,
                            max(1, window.width), max(1, window.height))

        win_transform = src.window_transform(window)
        try:
            data = src.read(window=window, boundless=True, fill_value=src.nodata or 0)
        except (ValueError, rasterio.errors.RasterioIOError) as e:
            return AlignedChip(None, None, empty=True, reason=f"read failed: {e}")

        # Build a boolean mask on the SAME window grid, then apply it.
        shape_mask, _, _ = raster_geometry_mask(
            src, [mapping(clipped_geom)], crop=False, invert=False)
        row0, col0 = window.row_off, window.col_off
        sub = shape_mask[row0:row0 + int(window.height),
                         col0:col0 + int(window.width)]
        if sub.shape != data.shape[1:]:
            # Grid disagreement is a hard alignment failure — do not fudge it.
            return AlignedChip(None, None, empty=True, reason="mask/window grid mismatch")

        masked = np.where(sub[np.newaxis, :, :], src.nodata or 0, data)
        if np.all(masked == (src.nodata or 0)):
            return AlignedChip(None, None, empty=True, reason="all pixels masked out")

        return AlignedChip(masked, win_transform, empty=False)
```

The returned `transform` is the affine of the clipped window, so the chip stays georeferenced — downstream you can map any pixel back to a ground coordinate, which is what keeps the model's spatial reasoning honest. Feed the aligned chip and the mask geometry to the model together, and keep the chip small enough to respect the [context window budget for maps](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/); a full-scene chip wastes tokens the reasoning task needs.

## Validation & Testing

Assert CRS reconciliation, the empty-intersection contract, and grid consistency:

```python
from shapely.geometry import box as shp_box

def test_disjoint_mask_returns_empty(tmp_raster_4326):
    far_away = shp_box(100.0, 10.0, 100.1, 10.1)     # nowhere near the tile
    chip = align_tile_to_mask(tmp_raster_4326, far_away, "EPSG:4326")
    assert chip.empty and chip.array is None

def test_reprojected_mask_still_overlaps(tmp_raster_4326, mask_in_3857):
    chip = align_tile_to_mask(tmp_raster_4326, mask_in_3857, "EPSG:3857")
    assert not chip.empty
    assert chip.array.shape[1] >= 1 and chip.array.shape[2] >= 1

def test_chip_grid_matches_transform(tmp_raster_4326, mask_inside):
    chip = align_tile_to_mask(tmp_raster_4326, mask_inside, "EPSG:4326")
    # The affine must describe exactly the returned array extent.
    assert chip.transform is not None
    assert chip.array.shape[1] == round((chip.array.shape[1]))
```

Add a round-trip check: take the chip's center pixel, project it through the returned transform, and confirm the resulting ground coordinate falls inside the original mask. If it does not, the alignment silently drifted and the assertion catches it before the model ever sees the pair.

## Gotchas & Edge Cases

**Nodata confusion.** Using `0` as the mask fill collides with legitimate zero-valued pixels (e.g. water in some bands). Prefer the raster's declared nodata; if it has none, add an explicit alpha/mask band instead of overloading a data value.

**Half-pixel offsets.** Rounding a window's offsets and lengths independently can shift the chip by half a pixel. Round offsets first, then derive lengths from the rounded bounds, so the window stays coincident with the source grid.

**Antimeridian and pole-crossing masks.** A mask spanning ±180° or a polar cap reprojects into a self-intersecting or wildly distorted footprint. Detect the wrap before reprojecting and split the mask, or the intersection test misfires.

**Resampling temptation.** When CRSs differ it is tempting to warp the raster to the mask CRS for a "cleaner" clip. Resist it for model input — every resample blurs the pixels the model reads. Reproject the vector and clip on the native grid instead.

## Related

- Up to the area overview: [Vector-Raster Hybrid Processing](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/vector-raster-hybrid-processing/)
- [Coordinate Reference System Normalization](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/)
- [Context Window Optimization for Maps](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/)
