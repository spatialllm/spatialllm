---
title: Vector-Raster Hybrid Processing
description: Combine gridded and vector data for model context — aligning tiles to masks, summarising pixels into statements, and keeping resolution honest in the answer.
slug: vector-raster-hybrid-processing
type: topic
breadcrumb: Vector-Raster Processing
datePublished: 2025-02-25
dateModified: 2026-08-11
---

# Vector-Raster Hybrid Processing

A model cannot read a raster. What it can read is a statement derived from one — "seventy per cent of this parcel is impermeable surface, from thirty-metre land-cover data captured in 2023" — and the whole of this topic is about producing statements like that correctly. The pixels have to be clipped to the right shape, sampled at a defensible resolution, summarised without inventing precision, and labelled with what they came from.

This topic belongs to [spatial LLM architecture and core concepts](/spatial-llm-architecture-core-concepts/) and pairs with [context-window optimization for maps](/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/), which decides how much room the resulting statements get. Which raster to use in the first place is a catalog question, answered by [spatial metadata and catalog indexing](/geospatial-rag-pipelines/spatial-metadata-and-catalog-indexing/).

<figure class="diagram">
<svg viewBox="7 56 763 188" role="img" aria-labelledby="vrh-path-t vrh-path-d" xmlns="http://www.w3.org/2000/svg"><title id="vrh-path-t">From pixels to a statement a model can use</title><desc id="vrh-path-d">A raster is aligned to the vector mask, clipped, summarised into class proportions, and rendered as a sentence carrying the resolution, the date and the share of the shape actually covered.</desc><rect x="7" y="56" width="763" height="188" fill="#ffffff"/><rect x="24" y="70" width="150" height="76" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="196" y="70" width="150" height="76" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="368" y="70" width="150" height="76" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="540" y="70" width="216" height="76" rx="8" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="99" y="100">align</text><text x="271" y="100">clip to mask</text><text x="443" y="100">summarise</text><text x="648" y="100">state it</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="99" y="124">same frame, same grid</text><text x="271" y="124">exact shape, not its box</text><text x="443" y="124">proportions, not pixels</text><text x="648" y="124">with resolution and date</text></g><g stroke="#5b6471" stroke-width="2" marker-end="url(#vrh-path-a)"><line x1="176" y1="108" x2="192" y2="108"/><line x1="348" y1="108" x2="364" y2="108"/><line x1="520" y1="108" x2="536" y2="108"/></g><defs><marker id="vrh-path-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/></marker></defs><rect x="24" y="184" width="732" height="46" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="390" y="212" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">Every stage can silently invent precision the pixels never had</text></svg>
<figcaption><b>Four stages, four ways to overstate.</b> Misalignment shifts the answer, clipping to the bounding box inflates it, summarising to two decimals implies accuracy the grid cannot support, and omitting the resolution lets a reader assume any of it was measured finely.</figcaption>
</figure>

## Foundational Principles

**Alignment precedes everything.** A raster and a vector mask in different frames, or on grids offset by half a cell, produce a plausible number that is systematically wrong. Reproject the mask to the raster's frame rather than the reverse, because resampling pixels degrades them and reprojecting a polygon does not.

**Clip to the shape, never to its extent.** A parcel's bounding box can be twice its area for an elongated or L-shaped feature, and the pixels in the difference belong to somewhere else. The bounding box is a pre-filter for reading, not a substitute for the mask.

**Report the resolution with the number.** "Sixty-eight per cent impermeable" from thirty-metre data means something different from the same figure at one metre, and only one of the two supports a statement about a single building. The resolution travels with the statistic or the statistic is not usable.

## Step-by-Step Implementation Pipeline

### 1. Read the raster's grid definition before anything else

Every later decision depends on the transform, the frame and the nodata value. Read them once, and fail early if any is missing rather than discovering it as a wrong number later.

```python
import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("vector_raster")


@dataclass(frozen=True)
class Grid:
    epsg: int
    pixel_size_m: float
    nodata: Optional[float]
    captured_year: Optional[int]


class RasterUnusable(ValueError):
    """The raster cannot be used for a defensible statistic."""


def read_grid(dataset) -> Grid:
    """Extract the grid definition, refusing anything that cannot be interpreted."""
    epsg = dataset.crs.to_epsg() if dataset.crs else None
    if epsg is None:
        raise RasterUnusable("raster has no declared reference frame")
    if dataset.res[0] <= 0 or dataset.res[1] <= 0:
        raise RasterUnusable(f"implausible pixel size {dataset.res}")
    if abs(dataset.res[0] - dataset.res[1]) > 1e-6 * dataset.res[0]:
        log.info("non-square pixels %s — area statistics will use cell area", dataset.res)
    return Grid(epsg, float(dataset.res[0]), dataset.nodata,
                dataset.tags().get("captured_year"))
```

### 2. Reproject the mask into the raster's frame

Moving the vector is cheap and lossless; moving the raster resamples every pixel and changes the values being counted. The direction of this transformation is the single most consequential decision in the whole pipeline.

```python
from shapely.ops import transform as shapely_transform
from pyproj import CRS, Transformer


def mask_in_raster_frame(mask, mask_epsg: int, grid: Grid):
    """Bring the vector to the pixels, never the pixels to the vector."""
    if mask_epsg == grid.epsg:
        return mask
    tf = Transformer.from_crs(CRS.from_epsg(mask_epsg), CRS.from_epsg(grid.epsg),
                              always_xy=True)
    try:
        return shapely_transform(tf.transform, mask)
    except Exception as exc:
        raise RasterUnusable(f"mask could not be reprojected to EPSG:{grid.epsg}: {exc}") from exc
```

### 3. Check that the shape is big enough for the grid

A thirty-metre grid cannot say anything about a shape smaller than a few cells. Computing a percentage from four pixels is arithmetic, not measurement, and the honest response is to decline rather than to round.

```python
MIN_CELLS_FOR_PROPORTION = 12


def cells_covered(mask, grid: Grid) -> float:
    """Approximate cell count for the mask at this resolution."""
    return mask.area / (grid.pixel_size_m ** 2)


def resolution_adequate(mask, grid: Grid) -> tuple[bool, str]:
    n = cells_covered(mask, grid)
    if n < MIN_CELLS_FOR_PROPORTION:
        return False, (f"the shape covers about {n:.0f} cells at "
                       f"{grid.pixel_size_m:g} m; too few for a proportion")
    return True, ""
```

The threshold is a judgement rather than a law, and stating it in the refusal is what makes it reviewable. A reader told "this parcel is four pixels across in the available data" understands the limitation immediately; one told "68%" does not.

<figure class="diagram">
<svg viewBox="8 36 756 210" role="img" aria-labelledby="vrh-clip-t vrh-clip-d" xmlns="http://www.w3.org/2000/svg"><title id="vrh-clip-t">Clipping to the bounding box against clipping to the shape</title><desc id="vrh-clip-d">An L-shaped parcel occupies little more than half of its bounding box, so statistics computed over the box include a large area belonging to neighbouring features.</desc><rect x="8" y="36" width="756" height="210" fill="#ffffff"/><rect x="60" y="50" width="280" height="150" rx="4" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="60" y="50" width="120" height="150" rx="3" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="180" y="140" width="160" height="60" rx="3" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="200" y="228" fill="#1f2937" font-size="12.5" text-anchor="middle">green: the parcel · amber: the rest of its box</text><rect x="430" y="60" width="320" height="60" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="590" y="86" fill="#1f2937" font-size="12.5" text-anchor="middle">box statistics: 41% of pixels</text><text x="590" y="108" fill="#5b6471" font-size="12" text-anchor="middle">belong to other parcels</text><rect x="430" y="140" width="320" height="60" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="590" y="166" fill="#1f2937" font-size="12.5" text-anchor="middle">mask statistics: only this parcel</text><text x="590" y="188" fill="#5b6471" font-size="12" text-anchor="middle">the number the question asked for</text></svg>
<figcaption><b>The box is a reading optimisation, not an analysis unit.</b> Reading the box from storage is efficient and correct; counting its pixels is a different question from the one that was asked, and the difference grows with how irregular the shape is.</figcaption>
</figure>

### 4. Sample the pixels within the mask

The read uses the bounding box for efficiency and the mask for correctness, with nodata excluded from the denominator rather than counted as a class.

```python
import numpy as np


def sample(dataset, mask, grid: Grid) -> tuple[np.ndarray, float]:
    """Return the values inside the mask and the share of the shape that had data."""
    try:
        window = dataset.window(*mask.bounds)
        block = dataset.read(1, window=window, masked=True)
        shape_mask = rasterize_mask(mask, dataset, window)     # True inside the polygon
    except Exception as exc:
        raise RasterUnusable(f"could not read the raster window: {exc}") from exc

    inside = block[shape_mask]
    if inside.size == 0:
        raise RasterUnusable("the mask selected no pixels")
    valid = inside.compressed() if hasattr(inside, "compressed") else inside
    if grid.nodata is not None:
        valid = valid[valid != grid.nodata]
    coverage = float(valid.size) / float(inside.size)
    return valid, coverage
```

Returning the coverage fraction alongside the values is what prevents the most common overstatement in this pipeline: a statistic computed over the third of a shape that had data, reported as though it described the whole.

### 5. Summarise into proportions, at a precision the grid supports

Class proportions are the useful output for categorical rasters and simple statistics for continuous ones. Both should be rounded to a precision the cell count can support — reporting a percentage to one decimal from forty cells implies a resolution that does not exist.

```python
from collections import Counter


def class_proportions(values: np.ndarray, cell_count: int) -> dict[int, float]:
    """Proportions rounded to a precision the sample size supports."""
    if values.size == 0:
        return {}
    digits = 0 if cell_count < 100 else (1 if cell_count < 10_000 else 2)
    counts = Counter(int(v) for v in values)
    total = sum(counts.values())
    return {cls: round(100.0 * n / total, digits) for cls, n in counts.most_common()}
```

The precision rule is worth stating explicitly because rounding is where invented accuracy usually enters. Forty cells cannot distinguish 67.4% from 67.9%, and printing either one invites a reader to treat the difference as meaningful. The mechanics of turning these into prose are in [summarising raster statistics for model prompts](/spatial-llm-architecture-core-concepts/vector-raster-hybrid-processing/summarizing-raster-statistics-for-llm-prompts/).

### 6. Render the statement with its provenance

The output is a sentence, and the sentence carries the resolution, the capture date, the coverage and the class labels. Omitting any of them produces a statement that reads as more certain than it is.

```python
def to_statement(name: str, proportions: dict[int, float], labels: dict[int, str],
                 grid: Grid, coverage: float) -> str:
    """One sentence a model can quote, with everything a reader needs to judge it."""
    if not proportions:
        return f"{name}: no usable raster values within the shape."
    top = ", ".join(f"{labels.get(cls, f'class {cls}')} {pct:g}%"
                    for cls, pct in list(proportions.items())[:4])
    date = f", captured {grid.captured_year}" if grid.captured_year else ""
    cover = "" if coverage > 0.98 else f", from {coverage * 100:.0f}% of the shape"
    return f"{name}: {top} (from {grid.pixel_size_m:g} m data{date}{cover})."
```

### 7. Align tiles when the shape spans more than one

A shape crossing a tile boundary needs values from several files, and those files may not share a grid origin even when they share a frame. Reading them independently and concatenating the values is correct only if the grids align; where they do not, the honest options are to resample to a common grid deliberately or to report per tile. The alignment mechanics are covered in [aligning raster tiles with vector masks](/spatial-llm-architecture-core-concepts/vector-raster-hybrid-processing/aligning-raster-tiles-with-vector-masks-for-llm-context/).

```python
def grids_align(a: Grid, b: Grid, origin_a, origin_b) -> bool:
    """Same frame, same pixel size, and origins offset by a whole number of cells."""
    if a.epsg != b.epsg or abs(a.pixel_size_m - b.pixel_size_m) > 1e-6:
        return False
    dx = (origin_a[0] - origin_b[0]) / a.pixel_size_m
    dy = (origin_a[1] - origin_b[1]) / a.pixel_size_m
    return abs(dx - round(dx)) < 1e-3 and abs(dy - round(dy)) < 1e-3
```

### 8. Refuse rather than resample when the question needs precision

Resampling is sometimes necessary and always a loss, and for categorical data it is worse than a loss — averaging class codes produces values that correspond to no class at all. When a question needs a statistic that the available grid cannot support, the correct output is the refusal, with the reason.

```python
def statistic_or_refusal(mask, grid: Grid, dataset, name: str, labels) -> str:
    ok, why = resolution_adequate(mask, grid)
    if not ok:
        return f"{name}: cannot be measured from this data — {why}."
    try:
        values, coverage = sample(dataset, mask, grid)
    except RasterUnusable as exc:
        return f"{name}: cannot be measured from this data — {exc}."
    props = class_proportions(values, int(cells_covered(mask, grid)))
    return to_statement(name, props, labels, grid, coverage)
```

## Operating This Stage Over Time

Raster pipelines drift for reasons that have nothing to do with the code. The commonest is a source that changes its class scheme: a land-cover product reissues with an extra category, the numeric codes shift, and every proportion the pipeline produces is now labelled wrongly while remaining arithmetically correct. Pin the class scheme version alongside the data and assert that the codes present in a raster are all in the expected set — an unexpected code is a loud failure and a silent mislabelling otherwise.

The second is resolution creep in the opposite direction from what you would expect. Sources improve, a ten-metre product replaces a thirty-metre one, and the code that refused to measure small parcels starts succeeding. That is good, and it changes the meaning of historical answers: a statement produced last year for the same parcel said something different because it was measured differently. Recording the resolution in the statement, as step 6 does, is what makes those two answers comparable rather than contradictory.

The third is coverage. Cloud gaps, sensor outages and processing failures leave holes that vary over time, so the same query can return a confident number one month and a low-coverage caveat the next. Track the coverage fraction as a monitored value rather than only reporting it: a fall across many queries usually means a data problem upstream rather than a change in the shapes being asked about.

A useful discipline throughout is to treat every number this stage produces as a quotation from a dataset rather than as a measurement of the world. That framing makes the provenance fields feel necessary rather than decorative, and it is the framing a careful reader will apply anyway.

## Failure Modes & Root Causes

**The bounding-box statistic.** A proportion computed over the shape's extent rather than the shape, inflated by whatever surrounds it. Root cause: using the read window as the analysis unit. Mitigation: rasterize the mask and index by it, as step 4 does.

**The resampled category.** Class codes averaged during a reprojection, producing values corresponding to no real class. Root cause: reprojecting the raster instead of the mask, with a continuous resampling method. Mitigation: move the vector; where a raster must be resampled, use nearest-neighbour for categorical data and say so.

**The confident percentage from four pixels.** A statistic that is arithmetically valid and meaningless. Root cause: no minimum sample check. Mitigation: the cell-count gate, with the refusal carrying the cell count.

**The vanished nodata.** Nodata values counted as a class, producing a large proportion of something that was simply not measured. Root cause: reading without a mask or with the wrong nodata value. Mitigation: exclude explicitly and report the coverage fraction.

## Production Validation Protocols

1. **Mask-not-box assertion.** For an L-shaped fixture, assert the mask statistic differs from the box statistic; if they match, the mask is not being applied.
2. **Frame assertion.** Assert the mask and raster frames match at the point of sampling, and that the mask was the thing reprojected.
3. **Coverage reporting.** Assert every emitted statement includes the coverage fraction whenever it falls below the threshold.
4. **Class-code invariant.** Assert every code encountered is in the expected scheme; an unknown code fails the build rather than being labelled as unknown.
5. **Precision rule test.** Assert the number of decimal places in an emitted proportion matches the cell count that produced it.
6. **Refusal path test.** Assert a shape smaller than the minimum cell count produces a refusal naming the cell count, not a number.

<figure class="diagram">
<svg viewBox="0 36 760 194" role="img" aria-labelledby="vrh-res-t vrh-res-d" xmlns="http://www.w3.org/2000/svg"><title id="vrh-res-t">What each grid resolution can honestly answer</title><desc id="vrh-res-d">Four resolutions from sub-metre to a hundred metres, each paired with the smallest feature it can describe and the questions that become unanswerable below that size.</desc><rect x="0" y="36" width="760" height="194" fill="#ffffff"/><rect x="30" y="50" width="164" height="120" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="204" y="50" width="164" height="120" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="378" y="50" width="164" height="120" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="552" y="50" width="164" height="120" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="112" y="80">0.25 m</text><text x="286" y="80">2 m</text><text x="460" y="80">10 m</text><text x="634" y="80">30 m</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="112" y="108">roof detail</text><text x="112" y="132">a building</text><text x="286" y="108">a garden</text><text x="286" y="132">a small parcel</text><text x="460" y="108">a field</text><text x="460" y="132">a city block</text><text x="634" y="108">a district</text><text x="634" y="132">not a parcel</text></g><text x="380" y="212" fill="#1f2937" font-size="13" text-anchor="middle">A question about a building answered from the rightmost column is not a coarse answer — it is not an answer</text></svg>
<figcaption><b>Resolution is a hard boundary, not a quality dial.</b> Below a few cells the statistic stops describing the shape at all, which is why the gate refuses rather than caveats.</figcaption>
</figure>

Of these, the mask-not-box assertion is the one that catches the most damaging bug for the least effort. It needs one irregular fixture shape and two lines, and it fails loudly the moment somebody optimises the read path by dropping the rasterized mask — an optimisation that looks harmless, speeds the query up noticeably, and changes every number the stage produces.

The class-code invariant deserves the same treatment for a different reason: it is the only check that notices when an upstream product reissues with a changed scheme. Everything else keeps working, the proportions still sum to a hundred, and only the labels are wrong.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should partially covered pixels at the edge be counted?</span></summary><p>It depends on the shape's size relative to the cell. For a shape hundreds of cells across, edge treatment changes the answer by a fraction of a per cent and any consistent rule is fine. For a shape a dozen cells across it dominates, which is another reason for the minimum-cell gate. Where it matters, area-weighted counting — including each edge cell in proportion to its overlap — is more defensible than an all-or-nothing rule, and worth the extra work only in that regime.</p></details>

<details class="faq-item"><summary><span>Is it ever right to reproject the raster instead of the mask?</span></summary><p>When many masks will be evaluated against the same raster and the reprojection can be done once, offline, with a documented resampling method. That is a data-preparation decision rather than a query-time one, and it should produce a new stored dataset with its own provenance rather than happening invisibly inside a request. Reprojecting at query time, per request, is the pattern to avoid.</p></details>

<details class="faq-item"><summary><span>How should continuous rasters be summarised differently from categorical ones?</span></summary><p>With distribution statistics rather than proportions — a median and a range say more about elevation or temperature than a mean, and far more than a single value. Report the statistic that survives outliers, because a raster clipped to a shape frequently includes a few cells of something else at the edges, and a mean is exactly the statistic those cells move most.</p></details>

<details class="faq-item"><summary><span>What if two rasters disagree about the same place?</span></summary><p>Report both, with their dates and resolutions, rather than picking one or averaging them. Disagreement between a 2019 ten-metre product and a 2023 thirty-metre one is information — it may be change over time, or it may be a resolution artefact — and collapsing it into one number destroys the only clue a reader has. If one must be chosen, the catalog fitness scoring is where that choice belongs, with its reasons recorded.</p></details>

<details class="faq-item"><summary><span>Do these statements belong in the prompt or in the answer?</span></summary><p>In the prompt, as facts, and quoted into the answer with their provenance intact. Computing them after generation, to check something the model already said, is much harder — the model will have phrased the claim in a way that does not map cleanly onto a statistic. Precomputing the handful of statistics the question implies, and letting the model quote them, keeps the numbers correct and the sentence natural.</p></details>

## Related

- Up to the section overview: [Spatial LLM Architecture and Core Concepts](/spatial-llm-architecture-core-concepts/)
- Technique: [Aligning Raster Tiles with Vector Masks](/spatial-llm-architecture-core-concepts/vector-raster-hybrid-processing/aligning-raster-tiles-with-vector-masks-for-llm-context/)
- Technique: [Summarising Raster Statistics for Model Prompts](/spatial-llm-architecture-core-concepts/vector-raster-hybrid-processing/summarizing-raster-statistics-for-llm-prompts/)
- Peer topic: [Context-Window Optimization for Maps](/spatial-llm-architecture-core-concepts/context-window-optimization-for-maps/)
- Related topic: [Spatial Metadata and Catalog Indexing](/geospatial-rag-pipelines/spatial-metadata-and-catalog-indexing/)
- Related topic: [Evaluation and Benchmarking for Spatial LLMs](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/)
