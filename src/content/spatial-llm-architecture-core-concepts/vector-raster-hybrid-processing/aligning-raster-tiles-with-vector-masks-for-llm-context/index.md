---
title: Aligning Raster Tiles with Vector Masks for LLM Context
description: Read the right pixels for a shape that spans several tiles — matching grids, rasterizing the mask, and refusing to resample categorical data to force a fit.
slug: aligning-raster-tiles-with-vector-masks-for-llm-context
type: howto
breadcrumb: Aligning Tiles and Masks
datePublished: 2025-02-26
dateModified: 2026-08-11
---

# Aligning Raster Tiles with Vector Masks for LLM Context

A shape rarely sits inside one tile, and the tiles it spans do not always share a grid. Getting the alignment right is what separates a statistic that describes the shape from one that describes a rectangle near it, and the failure produces no error at any point. This guide handles the alignment and the reading, as the mechanical core of [vector-raster hybrid processing](/spatial-llm-architecture-core-concepts/vector-raster-hybrid-processing/).

## When to Use This Approach

Use it whenever a mask spans more than one file, or whenever the mask and the raster arrive in different frames — which together cover almost every real request.

| Situation | Handling | Cost |
|-----------|----------|------|
| One tile, same frame | Read the window, apply the mask | Trivial |
| Several tiles, aligned grids | Read each, concatenate values | Cheap |
| Several tiles, unaligned grids | Report per tile, or resample deliberately | Real |
| Mask in a different frame | Reproject the mask, never the raster | Cheap |
| Categorical data needing resampling | Nearest neighbour, and say so | Lossy |

The third row is where judgement is required. Unaligned grids mean the pixels are not comparable, and pretending otherwise by resampling one to the other is a decision with consequences that should be recorded rather than absorbed.

<figure class="diagram">
<svg viewBox="16 24 764 210" role="img" aria-labelledby="art-align-t art-align-d" xmlns="http://www.w3.org/2000/svg"><title id="art-align-t">Aligned and offset grids across a tile boundary</title><desc id="art-align-d">Two tiles whose grid origins differ by half a cell cannot have their pixels concatenated, because a value from one does not describe the same ground as a value from the other.</desc><rect x="16" y="24" width="764" height="210" fill="#ffffff"/><text x="30" y="60" fill="#12805c" font-size="13" font-weight="600">aligned</text><rect x="180" y="38" width="60" height="42" rx="3" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="242" y="38" width="60" height="42" rx="3" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="304" y="38" width="60" height="42" rx="3" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="366" y="38" width="60" height="42" rx="3" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="470" y="64" fill="#5b6471" font-size="12">values are directly comparable</text><text x="30" y="150" fill="#b3324f" font-size="13" font-weight="600">offset</text><rect x="180" y="128" width="60" height="42" rx="3" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="242" y="128" width="60" height="42" rx="3" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="335" y="128" width="60" height="42" rx="3" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="397" y="128" width="60" height="42" rx="3" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="500" y="154" fill="#5b6471" font-size="12">half a cell apart — not the same ground</text><text x="390" y="216" fill="#1f2937" font-size="13" text-anchor="middle">Test the origins before concatenating anything</text></svg>
<figcaption><b>Half a cell is enough to break it.</b> Concatenated values from offset grids produce a statistic whose denominator is right and whose numerator counts the wrong ground, with nothing to indicate that anything happened.</figcaption>
</figure>

## Implementation

The reader checks alignment, rasterizes the mask per tile, and accumulates values with their coverage.

```python
import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

log = logging.getLogger("tile_alignment")


@dataclass(frozen=True)
class TileGrid:
    epsg: int
    pixel_size_m: float
    origin: tuple[float, float]
    nodata: float | None


class Unalignable(ValueError):
    """The tiles cannot be combined without a decision the caller must make."""


def grids_align(a: TileGrid, b: TileGrid, tolerance: float = 1e-3) -> bool:
    """Same frame, same pixel size, origins offset by a whole number of cells."""
    if a.epsg != b.epsg:
        return False
    if abs(a.pixel_size_m - b.pixel_size_m) > 1e-6 * a.pixel_size_m:
        return False
    dx = (a.origin[0] - b.origin[0]) / a.pixel_size_m
    dy = (a.origin[1] - b.origin[1]) / a.pixel_size_m
    return abs(dx - round(dx)) < tolerance and abs(dy - round(dy)) < tolerance


def read_masked(datasets: Sequence, mask, mask_epsg: int,
                rasterize) -> tuple[np.ndarray, float]:
    """Read every pixel of the mask across the tiles. Raises if grids disagree."""
    if not datasets:
        raise Unalignable("no tiles cover the mask")

    grids = [TileGrid(d.crs.to_epsg(), float(d.res[0]), (d.transform.c, d.transform.f),
                      d.nodata) for d in datasets]
    reference = grids[0]
    for other in grids[1:]:
        if not grids_align(reference, other):
            raise Unalignable(
                f"tiles do not share a grid ({reference.epsg}/{reference.pixel_size_m:g} m "
                f"against {other.epsg}/{other.pixel_size_m:g} m) — resample deliberately "
                "or report per tile")

    projected = mask if mask_epsg == reference.epsg else _reproject(mask, mask_epsg,
                                                                   reference.epsg)
    collected, inside_total = [], 0
    for dataset, grid in zip(datasets, grids):
        try:
            window = dataset.window(*projected.bounds)
            block = dataset.read(1, window=window, masked=True)
            inside = rasterize(projected, dataset, window)      # True where the mask covers
        except Exception as exc:
            log.warning("tile read failed, skipping it: %s", exc)
            continue
        values = block[inside]
        inside_total += int(inside.sum())
        usable = values.compressed() if hasattr(values, "compressed") else values
        if grid.nodata is not None:
            usable = usable[usable != grid.nodata]
        collected.append(usable)

    if not collected or inside_total == 0:
        raise Unalignable("the mask selected no pixels in any tile")
    stacked = np.concatenate(collected)
    return stacked, float(stacked.size) / float(inside_total)
```

Raising rather than resampling when grids disagree is the decision this whole guide turns on. Resampling is sometimes the right answer and is never one a reader should discover by accident, so the function refuses and names the alternatives; the caller decides, and records the decision.

Skipping a tile that fails to read, rather than aborting, is the opposite instinct and is also right: a mask spanning six tiles where one is corrupt still has five tiles of real data, and the coverage fraction reports the loss. What must not happen is the loss being invisible, which is why coverage is a return value rather than a log line.

<figure class="diagram">
<svg viewBox="106 36 658 190" role="img" aria-labelledby="art-span-t art-span-d" xmlns="http://www.w3.org/2000/svg"><title id="art-span-t">A mask spanning four tiles, one of them unreadable</title><desc id="art-span-d">Three tiles contribute pixels and one fails, so the statistic is computed from three quarters of the shape and the coverage fraction reports exactly that.</desc><rect x="106" y="36" width="658" height="190" fill="#ffffff"/><rect x="120" y="50" width="140" height="80" rx="4" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="262" y="50" width="140" height="80" rx="4" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="120" y="132" width="140" height="80" rx="4" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="262" y="132" width="140" height="80" rx="4" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="332" y="178" fill="#1f2937" font-size="12" text-anchor="middle">unreadable</text><rect x="450" y="60" width="300" height="60" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="600" y="86" fill="#1f2937" font-size="12.5" text-anchor="middle">three tiles contribute pixels</text><text x="600" y="106" fill="#5b6471" font-size="12" text-anchor="middle">the statistic is still computed</text><rect x="450" y="136" width="300" height="60" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="600" y="162" fill="#1f2937" font-size="12.5" text-anchor="middle">coverage: 74% of the shape</text><text x="600" y="182" fill="#5b6471" font-size="12" text-anchor="middle">reported, not swallowed</text></svg>
<figcaption><b>Partial data with a stated fraction beats no data.</b> The failure mode to avoid is the third option — a statistic computed from three quarters of a shape and presented as though it described all of it.</figcaption>
</figure>

## Validation & Testing

```python
def test_offset_grids_are_refused():
    a = TileGrid(27700, 10.0, (0.0, 0.0), None)
    b = TileGrid(27700, 10.0, (5.0, 0.0), None)      # half a cell
    assert not grids_align(a, b)


def test_whole_cell_offsets_are_accepted():
    a = TileGrid(27700, 10.0, (0.0, 0.0), None)
    b = TileGrid(27700, 10.0, (30.0, -20.0), None)
    assert grids_align(a, b)


def test_mixed_frames_raise_rather_than_reprojecting_the_raster():
    try:
        read_masked([TILE_27700, TILE_4326], MASK, 27700, rasterize)
    except Unalignable as exc:
        assert "do not share a grid" in str(exc)
        return
    raise AssertionError("mixed frames must be refused, not silently resampled")


def test_unreadable_tile_reduces_coverage_without_aborting():
    values, coverage = read_masked([GOOD, GOOD, BROKEN], MASK, 27700, rasterize)
    assert values.size > 0 and coverage < 1.0


def test_empty_selection_raises():
    try:
        read_masked([TILE_ELSEWHERE], MASK, 27700, rasterize)
    except Unalignable:
        return
    raise AssertionError("a mask selecting no pixels must not return an empty statistic")
```

The third test is the one that will be argued about, and it is worth keeping. Resampling on the caller's behalf is convenient, it works, and it silently changes categorical values into interpolated ones that correspond to no class — which is a corruption rather than a loss.

Build the fixtures from real tile metadata rather than from constructed grids. Origin values in production data carry the floating-point residue of whatever produced them, and a tolerance that looks generous against clean numbers can be too tight against real ones — which presents as a refusal to combine tiles that every other tool combines happily.

## Gotchas & Edge Cases

**Reprojecting the raster to match the mask.** Resamples every pixel and, for categorical data, invents values. Move the vector; it is cheap and lossless.

**Window computed from the mask's bounds in the wrong frame.** Produces a window somewhere else entirely and usually reads nothing, which presents as "no data for this shape". Reproject the mask before computing the window, not after.

**Coverage computed against the window rather than the mask.** The window is a rectangle and the mask is not, so dividing by the window's pixel count understates coverage badly for irregular shapes. Divide by the pixels inside the mask.

**Tiles with different nodata values.** Concatenating values from tiles whose nodata conventions differ leaves one tile's nodata counted as data. Read each tile's own nodata from its metadata rather than assuming a shared constant.

<figure class="diagram">
<svg viewBox="11 36 723 197" role="img" aria-labelledby="art-win-t art-win-d" xmlns="http://www.w3.org/2000/svg"><title id="art-win-t">Coverage measured against the window rather than the mask</title><desc id="art-win-d">Dividing usable pixels by the rectangular read window rather than by the pixels inside the mask understates coverage badly for an irregular shape.</desc><rect x="11" y="36" width="723" height="197" fill="#ffffff"/><rect x="60" y="50" width="280" height="140" rx="4" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="60" y="50" width="140" height="140" rx="3" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="200" y="130" width="140" height="60" rx="3" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="200" y="216" fill="#5b6471" font-size="12" text-anchor="middle">green: the mask · amber: the rest of the window</text><rect x="420" y="60" width="300" height="54" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="570" y="92" fill="#1f2937" font-size="12.5" text-anchor="middle">divide by the window: 58% coverage</text><rect x="420" y="130" width="300" height="54" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="570" y="162" fill="#1f2937" font-size="12.5" text-anchor="middle">divide by the mask: 96% coverage</text></svg>
<figcaption><b>The denominator decides the message.</b> One of these reads as a serious data gap and the other as a complete measurement, and only the second describes the shape the question was about.</figcaption>
</figure>

**Overlapping tiles double-counting.** Many tile schemes overlap by a few pixels at the edges, so a mask spanning a boundary counts those pixels twice. Deduplicate by position, or clip each tile's window to its exclusive extent.

**A mask spanning tiles from two product versions.** They can share a grid and disagree about what a class code means, so concatenation produces proportions that are arithmetically clean and semantically mixed. Check the product version alongside the grid, and treat a mismatch exactly as you would treat a grid mismatch — as a refusal rather than as something to average over.

**Alignment tolerance too tight.** Origins stored as floating-point values accumulate tiny errors, and an exact comparison rejects grids that are aligned in every practical sense. A thousandth of a cell is a workable tolerance and a whole cell is not.

## Frequently Asked Questions

<details class="faq-item"><summary><span>When is resampling actually the right answer?</span></summary><p>When the analysis genuinely needs a single combined array and the data is continuous — elevation, temperature, reflectance — where interpolation produces meaningful values. For categorical data it is only acceptable with nearest-neighbour selection, which preserves class validity at the cost of shifting boundaries by up to half a cell. Either way it belongs in a documented preparation step that produces a new dataset, not inside a request.</p></details>

<details class="faq-item"><summary><span>Should statistics be computed per tile and combined, or over concatenated values?</span></summary><p>Over concatenated values when the grids align, because combining per-tile proportions requires weighting by pixel count and is easy to get subtly wrong. When the grids do not align, per-tile reporting is the honest fallback: three statements with their own extents say more than one statement whose provenance is a mixture.</p></details>

<details class="faq-item"><summary><span>How should very large masks be handled?</span></summary><p>By reading in blocks and accumulating counts rather than values. A national-scale mask over fine imagery will not fit in memory as an array, and the summary only needs counts per class, so a streaming accumulator over windowed reads gives the same answer at constant memory. The coverage fraction accumulates the same way.</p></details>

<details class="faq-item"><summary><span>What if only some tiles have the class scheme you expect?</span></summary><p>Treat it as an alignment failure of a different kind and refuse. Tiles from different product versions can share a grid and disagree about what code seven means, and concatenating them produces proportions that are arithmetically clean and semantically meaningless. Check the scheme identifier alongside the grid, and fail loudly when it varies.</p></details>

<details class="faq-item"><summary><span>Where should the alignment check live?</span></summary><p>In the reader, before any pixel is fetched, so a misaligned set costs one comparison rather than a full read followed by a discovery. Putting it in the caller means every caller has to remember, and one of them will not — usually the batch job that processes a thousand shapes overnight and produces a thousand quietly wrong statistics.</p></details>

## Related

- Up to the parent topic: [Vector-Raster Hybrid Processing](/spatial-llm-architecture-core-concepts/vector-raster-hybrid-processing/)
- [Summarising Raster Statistics for Model Prompts](/spatial-llm-architecture-core-concepts/vector-raster-hybrid-processing/summarizing-raster-statistics-for-llm-prompts/)
- Related technique: [Indexing Catalog Collections for Agent Retrieval](/geospatial-rag-pipelines/spatial-metadata-and-catalog-indexing/indexing-stac-collections-for-agent-retrieval/)
- Related topic: [Coordinate Reference System Normalization](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/)
