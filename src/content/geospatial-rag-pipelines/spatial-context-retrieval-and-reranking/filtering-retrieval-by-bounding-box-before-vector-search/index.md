---
title: Filtering Retrieval by Bounding Box Before Vector Search
description: Narrow the candidate population with an index-aware extent predicate before any vector is compared, so selective spatial queries stay fast and keep their recall.
slug: filtering-retrieval-by-bounding-box-before-vector-search
type: howto
breadcrumb: Bounding-Box Pre-Filter
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Filtering Retrieval by Bounding Box Before Vector Search

The cheapest way to make a spatial retrieval pipeline fast is to stop searching most of the corpus. A bounding-box predicate backed by a spatial index removes 99% of candidates for the price of one index lookup, and every expensive stage downstream then operates on a set small enough to afford. This guide covers writing that filter so the planner actually uses it, and so recall survives the narrowing — the first stage of [spatial context retrieval and reranking](/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/).

## When to Use This Approach

Use a pre-filter whenever the query has a region and the corpus spans more than it. Skip it only when the corpus is already confined to the query region, in which case the filter costs a little and removes nothing.

| Corpus against query region | Filter | Effect |
|-----------------------------|--------|--------|
| National corpus, one town | Essential | Removes 99%+ before any comparison |
| Regional corpus, one site | Valuable | Removes most; keeps recall high |
| Site corpus, that site | Skip | Nothing to remove; adds a predicate for nothing |
| Corpus with mixed geometry coverage | Essential, with a fallback lane | Otherwise ungeoreferenced chunks vanish |

The last row is the one that catches teams out. A filter on extent silently excludes every chunk whose extent is missing, and in a corpus where a fifth of chunks lack geometry that is a fifth of the knowledge disappearing from every spatial query.

<figure class="diagram">
<svg viewBox="16 9 754 215" role="img" aria-labelledby="fbb-order-t fbb-order-d" xmlns="http://www.w3.org/2000/svg"><title id="fbb-order-t">Cost of filtering before against after the vector comparison</title><desc id="fbb-order-d">Filtering first compares vectors for a few thousand candidates; filtering afterwards compares vectors for the whole corpus and then discards nearly all of them, at a cost hundreds of times higher.</desc><rect x="16" y="9" width="754" height="215" fill="#ffffff"/><text x="390" y="34" fill="#5b6471" font-size="13" text-anchor="middle">Same result set, two orders of magnitude apart in work done</text><rect x="30" y="60" width="200" height="60" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="260" y="60" width="200" height="60" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="490" y="60" width="260" height="60" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="12.5" text-anchor="middle"><text x="130" y="88">index lookup</text><text x="360" y="88">4 100 candidates</text><text x="620" y="88">4 100 vector comparisons</text></g><text x="130" y="110" fill="#5b6471" font-size="12" text-anchor="middle">filter first</text><rect x="30" y="150" width="200" height="60" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="260" y="150" width="200" height="60" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="490" y="150" width="260" height="60" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="12.5" text-anchor="middle"><text x="130" y="178">full scan</text><text x="360" y="178">2 000 000 candidates</text><text x="620" y="178">2 000 000 comparisons, then discard</text></g><text x="130" y="200" fill="#5b6471" font-size="12" text-anchor="middle">filter after</text></svg>
<figcaption><b>Order is the whole optimisation.</b> Both paths return the same documents. The difference is whether the expensive comparison runs on the region or on the world, and no amount of index tuning recovers the second case.</figcaption>
</figure>

## Implementation

The predicate must be written so the spatial index is used: the bounding-box operator first, the exact predicate second, and no function wrapped around the indexed column.

```sql
-- Chunks carry a true geometry and a GiST index over it.
CREATE INDEX IF NOT EXISTS spatial_chunks_geom_idx ON spatial_chunks USING gist (geom);

WITH region AS (
    SELECT ST_MakeEnvelope(:west, :south, :east, :north, 4326) AS g
)
SELECT c.chunk_id,
       c.body,
       1 - (c.embedding <=> :qvec) AS semantic
FROM   spatial_chunks c, region r
WHERE  c.geom && r.g                       -- index-aware: bounding-box pre-filter
  AND  ST_Intersects(c.geom, r.g)          -- exact predicate on what survives
ORDER  BY c.embedding <=> :qvec
LIMIT  :k;
```

Two habits break this. Wrapping the indexed column in a function — `ST_Transform(c.geom, 3857) && …` — makes the index unusable, because the index holds untransformed values; transform the *query* geometry into the column's frame instead. And writing `ST_DWithin` or `ST_Intersects` without the `&&` operator leaves the planner to infer the index opportunity, which it often does and sometimes does not, with a hundred-fold latency difference between the two outcomes.

The application-side wrapper adds the fallback lane for chunks without geometry and enforces a floor on the region so a degenerate input cannot silently match nothing.

```python
import logging
from dataclasses import dataclass
from typing import Optional, Sequence

log = logging.getLogger("bbox_prefilter")

MIN_SPAN_DEG = 1e-4        # ~11 m: below this, a "region" is a rounding artefact


@dataclass(frozen=True)
class Region:
    west: float
    south: float
    east: float
    north: float

    def expanded(self, min_span: float = MIN_SPAN_DEG) -> "Region":
        """Grow a degenerate region to a usable minimum rather than matching nothing."""
        dx = max(0.0, min_span - (self.east - self.west)) / 2
        dy = max(0.0, min_span - (self.north - self.south)) / 2
        return Region(self.west - dx, self.south - dy, self.east + dx, self.north + dy)

    def is_sane(self) -> bool:
        return (self.west < self.east and self.south < self.north
                and -180 <= self.west and self.east <= 180
                and -90 <= self.south and self.north <= 90)


def retrieve(conn, qvec, region: Optional[Region], k: int = 200,
             include_ungeoreferenced: bool = True) -> Sequence[dict]:
    """Pre-filtered retrieval with an explicit lane for chunks that have no geometry."""
    if region is None:
        log.info("no region supplied — falling back to unfiltered vector search")
        return _vector_only(conn, qvec, k)

    region = region.expanded()
    if not region.is_sane():
        log.warning("nonsensical region %s — falling back to unfiltered search", region)
        return _vector_only(conn, qvec, k)

    rows = _filtered(conn, qvec, region, k)
    if include_ungeoreferenced:
        # Separate lane: chunks with no extent can still be relevant, but they are
        # kept apart so they never displace a candidate known to be in the region.
        rows = list(rows) + list(_no_geometry(conn, qvec, max(1, k // 10)))
    if not rows:
        log.info("region returned nothing — degrading to unfiltered search")
        return _vector_only(conn, qvec, k)
    return rows
```

The final degradation is a judgement call worth making explicitly. Returning nothing is sometimes the honest answer — the corpus genuinely says nothing about this place — but far more often it means the region was wrong, and an unfiltered result set with a clear "outside your area" marker is more useful to a reader than an empty page.

<figure class="diagram">
<svg viewBox="16 38 728 208" role="img" aria-labelledby="fbb-idx-t fbb-idx-d" xmlns="http://www.w3.org/2000/svg"><title id="fbb-idx-t">Query shapes that keep or lose the spatial index</title><desc id="fbb-idx-d">Three predicate forms: bounding-box operator followed by an exact predicate uses the index; an exact predicate alone may or may not; a function applied to the indexed column never does.</desc><rect x="16" y="38" width="728" height="208" fill="#ffffff"/><rect x="30" y="52" width="700" height="52" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="52" y="84" fill="#1f2937" font-size="12.5">geom &amp;&amp; envelope AND ST_Intersects(geom, envelope) — index used, always</text><rect x="30" y="116" width="700" height="52" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="52" y="148" fill="#1f2937" font-size="12.5">ST_Intersects(geom, envelope) alone — index used, usually, depending on the plan</text><rect x="30" y="180" width="700" height="52" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="52" y="212" fill="#1f2937" font-size="12.5">ST_Transform(geom, 3857) &amp;&amp; envelope — index never used, full scan every time</text></svg>
<figcaption><b>The third line is the one that ships.</b> It looks correct, it returns correct results, and it reads a whole table to do it — which is invisible until the corpus grows past the point where a scan is affordable.</figcaption>
</figure>

## Validation & Testing

```python
def test_plan_uses_the_spatial_index(conn):
    plan = explain(conn, REPRESENTATIVE_QUERY)
    assert "Index Scan" in plan or "Bitmap Index Scan" in plan
    assert "Seq Scan on spatial_chunks" not in plan


def test_degenerate_region_is_expanded_not_empty(conn):
    point_region = Region(-3.19, 55.95, -3.19, 55.95)   # zero span
    rows = retrieve(conn, QVEC, point_region, k=10)
    assert rows, "a zero-span region must be grown, not matched against nothing"


def test_ungeoreferenced_chunks_reachable(conn):
    rows = retrieve(conn, QVEC, REGION, k=100, include_ungeoreferenced=True)
    assert any(r["geom"] is None for r in rows)


def test_recall_holds_under_a_tight_filter(conn):
    tight = Region(-3.20, 55.94, -3.18, 55.96)
    got = {r["chunk_id"] for r in retrieve(conn, QVEC, tight, k=50)}
    assert BRUTE_FORCE_TRUTH_TIGHT <= got
```

The plan assertion is the one that earns its keep over years. Query plans change when statistics change, when a version is upgraded, and when someone adds an innocuous predicate, and a plan regression produces no error and no wrong answer — only a page that takes forty seconds to load.

## Gotchas & Edge Cases

**Antimeridian-crossing regions.** An envelope from 179 to −179 degrees is interpreted as spanning almost the whole world rather than the two degrees intended. Split such a region into two envelopes and union the results, and detect the case by testing whether the west value exceeds the east.

**Filtering on a stale extent.** If chunk extents were copied from a document header rather than computed from geometry, the filter is testing a claim rather than a fact. Recompute extents at index time; a filter is only as good as the metadata it reads.

<figure class="diagram">
<svg viewBox="26 42 715 201" role="img" aria-labelledby="fbb-anti-t fbb-anti-d" xmlns="http://www.w3.org/2000/svg"><title id="fbb-anti-t">A region crossing the antimeridian, split into two envelopes</title><desc id="fbb-anti-d">A region from 179 to minus 179 degrees is read as spanning the globe. Splitting it at the antimeridian into two envelopes and unioning the results restores the intended two-degree span.</desc><rect x="26" y="42" width="715" height="201" fill="#ffffff"/><rect x="40" y="56" width="680" height="54" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="380" y="88" fill="#1f2937" font-size="12.5" text-anchor="middle">one envelope, west 179 to east minus 179 — read as 358 degrees wide</text><rect x="40" y="140" width="150" height="54" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="570" y="140" width="150" height="54" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="115" y="172" fill="#1f2937" font-size="12" text-anchor="middle">179 to 180</text><text x="645" y="172" fill="#1f2937" font-size="12" text-anchor="middle">minus 180 to minus 179</text><text x="380" y="172" fill="#1f2937" font-size="12.5" text-anchor="middle">two envelopes, unioned — two degrees, as intended</text><text x="380" y="226" fill="#5b6471" font-size="12" text-anchor="middle">Detect the case by testing whether the west value exceeds the east value</text></svg>
<figcaption><b>A one-line test prevents a whole-corpus match.</b> Nothing errors when a region wraps; the filter simply stops filtering, and the latency regression looks like a capacity problem rather than a correctness one.</figcaption>
</figure>

**A region derived from a zoom level.** Map clients happily produce a region of zero area at high zoom, or one spanning the world at low zoom. Floor the span, as the code does, and cap it too — an unbounded region turns the pre-filter into a no-op while still paying its cost.

**Over-fetching to compensate.** When recall drops under a tight filter, the tempting fix is to raise the candidate limit. That works, at a cost proportional to the raise, and it treats the symptom: the real problem is usually a store that post-filters rather than pre-filters, which no limit fully fixes.

**Geometry stored in a projected frame.** If the column is projected and the query region is geographic, the operator compares incompatible numbers and silently matches nothing. Transform the query region into the column's frame — never the column into the region's.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should the filter use a bounding box or the true region geometry?</span></summary><p>Both, in that order. The bounding box is what the index can use; the true geometry is what makes the answer exact. Running only the box admits candidates in the corners of the rectangle that fall outside an irregular region, which for something like a catchment or a district can be a substantial fraction. Running only the exact predicate risks losing the index. The two-line form gives you both properties.</p></details>

<details class="faq-item"><summary><span>How much should the region be buffered, if at all?</span></summary><p>Enough to cover the uncertainty in the focus, and no more. If the focus came from a geocoder with a hundred-metre error, a hundred-metre buffer is honest; if it came from a surveyed parcel boundary, none is needed. Buffering "just to be safe" widens the population and dilutes the ranking, and it is a poor substitute for scoring proximity properly in the reranking stage.</p></details>

<details class="faq-item"><summary><span>Can the pre-filter and the vector search run in separate systems?</span></summary><p>They can, and it costs you a round trip and a consistency problem. The pattern is to fetch identifiers from the spatial system and pass them as a filter to the vector system, which works until the list is large enough that passing it becomes the bottleneck. It is workable for regional queries returning thousands of identifiers and painful for anything broader — one of the reasons to prefer a store that holds both.</p></details>

<details class="faq-item"><summary><span>What should happen when the region is very large?</span></summary><p>Treat it as absent. A region covering most of the corpus removes nothing while adding a predicate, and the honest handling is to skip the filter and let the semantic search do the work. Set a threshold on the fraction of the corpus extent the region covers, and log when it fires, since a run of very large regions usually means the region-sizing logic upstream has stopped working.</p></details>

## Related

- Up to the parent topic: [Spatial Context Retrieval and Reranking](/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/)
- [Reranking Spatial Results by Distance and Relevance](/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/reranking-spatial-results-by-distance-and-relevance/)
- Related topic: [Spatial Vector Store Selection](/geospatial-rag-pipelines/spatial-vector-store-selection/)
- Technique: [Carrying Frame and Extent Metadata Into Every Chunk](/geospatial-rag-pipelines/chunk-boundary-strategies-for-spatial-corpora/carrying-crs-and-extent-metadata-into-every-chunk/)
