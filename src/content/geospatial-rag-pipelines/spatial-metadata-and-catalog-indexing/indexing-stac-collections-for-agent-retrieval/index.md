---
title: Indexing Catalog Collections for Agent Retrieval
description: Ingest collection manifests and their items into a two-level index an agent can query — collection extent and period first, item-level coverage second.
slug: indexing-stac-collections-for-agent-retrieval
type: howto
breadcrumb: Indexing Catalog Collections
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Indexing Catalog Collections for Agent Retrieval

A collection manifest says a dataset covers a country between 2017 and now. Its items say which square kilometres were actually captured on which days, and the gap between those two statements is where agents produce confident answers about places nobody photographed. This guide builds the two-level index that closes it, implementing the catalog stage of [spatial metadata and catalog indexing](/geospatial-rag-pipelines/spatial-metadata-and-catalog-indexing/).

## When to Use This Approach

Index at both levels whenever a collection's coverage is uneven, which is nearly always for anything captured rather than modelled. A single-level index is adequate only for products that genuinely blanket their declared extent.

| Collection type | Coverage | Index |
|-----------------|----------|-------|
| Modelled national product | Complete by construction | Collection level only |
| Satellite archive | Dense but cloud-gapped | Both levels |
| Aerial survey programme | Patchy, campaign-based | Both levels, item extent essential |
| Ground survey records | Sparse points | Item level, collection extent is decoration |
| Derived analytic layer | Follows its input | Both, and record the lineage |

The distinction that matters is whether "covers" means "is defined over" or "was observed across". A modelled layer is defined everywhere in its extent; an archive was observed wherever the sensor happened to look.

<figure class="diagram">
<svg viewBox="46 32 718 198" role="img" aria-labelledby="isc-gap-t isc-gap-d" xmlns="http://www.w3.org/2000/svg"><title id="isc-gap-t">Declared collection extent against observed item coverage</title><desc id="isc-gap-d">A collection declares a rectangular national extent while its items cover three separated areas, leaving most of the declared extent unobserved and any query there unanswerable.</desc><rect x="46" y="32" width="718" height="198" fill="#ffffff"/><rect x="60" y="46" width="330" height="170" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="86" y="72" width="80" height="46" rx="4" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="200" y="130" width="80" height="46" rx="4" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="290" y="66" width="80" height="46" rx="4" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="225" y="204" fill="#1f2937" font-size="12.5" text-anchor="middle">declared extent, three observed areas</text><rect x="430" y="46" width="320" height="76" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="590" y="74" fill="#1f2937" font-size="12.5" text-anchor="middle">collection-level index only</text><text x="590" y="98" fill="#5b6471" font-size="12" text-anchor="middle">every point in the rectangle looks covered</text><rect x="430" y="140" width="320" height="76" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="590" y="168" fill="#1f2937" font-size="12.5" text-anchor="middle">item-level index as well</text><text x="590" y="192" fill="#5b6471" font-size="12" text-anchor="middle">coverage is a fact, not a claim</text></svg>
<figcaption><b>The declared extent is a bounding rectangle around observations, not a statement about them.</b> Treating it as coverage is what lets an agent answer confidently about a region the sensor never saw.</figcaption>
</figure>

## Implementation

The ingester walks collections, normalises each into the catalog record shape, then pages through items and stores their footprints in a separate table keyed back to the collection.

```python
import logging
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Iterator, Optional

log = logging.getLogger("catalog_ingest")

MAX_ITEM_PAGES = 500          # a guard against an endless paging loop


@dataclass(frozen=True)
class Item:
    item_id: str
    collection_id: str
    bbox: tuple[float, float, float, float]
    captured: Optional[date]
    usable_fraction: Optional[float]     # e.g. cloud-free share, when the source reports it


def _paged_items(client, collection_id: str, page_size: int = 500) -> Iterator[dict]:
    """Yield raw item records, stopping on error rather than aborting the whole ingest."""
    token, pages = None, 0
    while pages < MAX_ITEM_PAGES:
        try:
            batch, token = client.items(collection_id, limit=page_size, token=token)
        except Exception as exc:
            log.warning("item paging failed for %s after %d page(s): %s",
                        collection_id, pages, exc)
            return                                  # partial coverage beats no ingest
        if not batch:
            return
        yield from batch
        pages += 1
        if not token:
            return
    log.warning("item paging for %s hit the page guard at %d", collection_id, MAX_ITEM_PAGES)


def normalise_item(raw: dict, collection_id: str) -> Optional[Item]:
    """Return an item, or None when it cannot supply a usable footprint."""
    try:
        bbox = tuple(float(v) for v in raw["bbox"][:4])
        if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
            log.warning("item %s has a degenerate footprint", raw.get("id"))
            return None
        return Item(
            item_id=str(raw["id"]),
            collection_id=collection_id,
            bbox=bbox,
            captured=_as_date(raw.get("properties", {}).get("datetime")),
            usable_fraction=_usable_fraction(raw.get("properties", {})),
        )
    except (KeyError, TypeError, ValueError) as exc:
        log.warning("rejecting malformed item %r: %s", raw.get("id"), exc)
        return None


def ingest_collection(client, collection_id: str, upsert_item) -> dict:
    """Ingest one collection's items; report counts so partial ingests are visible."""
    stored = rejected = 0
    for raw in _paged_items(client, collection_id):
        item = normalise_item(raw, collection_id)
        if item is None:
            rejected += 1
            continue
        upsert_item(item)
        stored += 1
    stats = {"collection": collection_id, "stored": stored, "rejected": rejected}
    log.info("catalog item ingest: %s", stats)
    return stats
```

Two decisions here are about failing well. Paging stops on error rather than raising, because a collection with fifty thousand items and a flaky endpoint should contribute the forty thousand it managed rather than nothing — provided the partial state is recorded, which the counts do. And the page guard exists because a paging token that never advances is a real failure mode of paginated APIs, and without a bound it becomes an infinite loop in a nightly job.

The item table wants a spatial index and a date index, and the queries against it are the same index-aware shape used everywhere in this section:

```sql
CREATE TABLE catalog_items (
    item_id       text PRIMARY KEY,
    collection_id text NOT NULL REFERENCES catalog_collections(collection_id),
    geom          geometry(Polygon, 4326) NOT NULL,
    captured      date,
    usable_frac   double precision
);

CREATE INDEX catalog_items_geom_idx ON catalog_items USING gist (geom);
CREATE INDEX catalog_items_when_idx ON catalog_items (collection_id, captured DESC);

-- Does this collection actually observe this place, recently enough, usably enough?
SELECT count(*) AS observations,
       max(captured) AS most_recent
FROM   catalog_items
WHERE  collection_id = :collection
  AND  geom && :region                     -- index-aware pre-filter
  AND  ST_Intersects(geom, :region)
  AND  captured >= :since
  AND  coalesce(usable_frac, 1.0) >= :min_usable;
```

<figure class="diagram">
<svg viewBox="16 38 748 178" role="img" aria-labelledby="isc-two-t isc-two-d" xmlns="http://www.w3.org/2000/svg"><title id="isc-two-t">Two tables, two indexes, two query shapes</title><desc id="isc-two-d">Collections are few with large extents and are queried to shortlist; items are many with small extents and are queried to confirm coverage within the chosen collection.</desc><rect x="16" y="38" width="748" height="178" fill="#ffffff"/><rect x="30" y="52" width="340" height="150" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="410" y="52" width="340" height="150" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="200" y="82">collections</text><text x="580" y="82">items</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="200" y="112">thousands of rows</text><text x="200" y="136">large, overlapping extents</text><text x="200" y="160">queried to shortlist</text><text x="200" y="184">answers: which dataset</text><text x="580" y="112">millions of rows</text><text x="580" y="136">small, tiled extents</text><text x="580" y="160">queried within one collection</text><text x="580" y="184">answers: was it observed</text></g></svg>
<figcaption><b>Two very different spatial distributions.</b> Mixing them in one table gives the index a bimodal workload it serves badly for both; separating them lets each index specialise, at the cost of one extra join nobody misses.</figcaption>
</figure>

## Validation & Testing

```python
def test_items_fall_inside_their_collection_extent(conn):
    stray = conn.execute("""
        SELECT i.item_id FROM catalog_items i
        JOIN catalog_collections c USING (collection_id)
        WHERE NOT ST_Within(i.geom, ST_Expand(c.geom, 0.01))
        LIMIT 5""").fetchall()
    assert not stray, f"items outside their collection extent: {stray}"


def test_partial_ingest_is_reported_not_silent(caplog):
    stats = ingest_collection(FlakyClient(fail_after=3), "col-a", upsert)
    assert stats["stored"] > 0
    assert any("item paging failed" in r.message for r in caplog.records)


def test_paging_guard_terminates_on_a_stuck_token():
    stats = ingest_collection(StuckTokenClient(), "col-b", upsert)
    assert stats["stored"] <= MAX_ITEM_PAGES * 500


def test_degenerate_item_footprint_is_rejected():
    assert normalise_item({"id": "x", "bbox": [10, 10, 5, 5]}, "col-a") is None
```

Run the first test as a report rather than only as a gate when a catalog is first ingested: a handful of stray items is a data-quality note worth sending upstream, while a third of a collection sitting outside its declared extent means the extent field is being populated by something other than the items and should not be trusted for shortlisting at all.

The first test is the one that catches upstream data problems rather than code problems, and it will occasionally fail on legitimately odd data — a collection whose declared extent was never updated after items were added outside it. That is worth knowing about, which is why it asserts rather than warns.

## Gotchas & Edge Cases

**Item footprints that are the whole scene, not the usable part.** A satellite scene's footprint covers the full swath including the cloudy half. Store the usable fraction where the source reports it and require a threshold in coverage queries, or coverage counts overstate what is actually visible.

**Collections with millions of items and no need for them.** Ingesting every item of a global archive to answer questions about one country is a great deal of storage for no benefit. Restrict item ingest to your regions of interest and record that restriction, so a later query outside them reports "not indexed" rather than "not covered".

<figure class="diagram">
<svg viewBox="46 36 688 178" role="img" aria-labelledby="isc-scene-t isc-scene-d" xmlns="http://www.w3.org/2000/svg"><title id="isc-scene-t">Scene footprint against usable area</title><desc id="isc-scene-d">A satellite scene footprint covers a full rotated swath while cloud obscures part of it, so counting scenes overstates how much of the region was actually observed.</desc><rect x="46" y="36" width="688" height="178" fill="#ffffff"/><rect x="60" y="50" width="280" height="150" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="200" y="72" width="120" height="106" rx="6" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><text x="130" y="130" fill="#1f2937" font-size="12" text-anchor="middle">usable</text><text x="260" y="130" fill="#5b6471" font-size="12" text-anchor="middle">obscured</text><rect x="410" y="50" width="310" height="66" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="565" y="78" fill="#1f2937" font-size="12.5" text-anchor="middle">counting scenes: fully covered</text><text x="565" y="100" fill="#5b6471" font-size="12" text-anchor="middle">the footprint says yes</text><rect x="410" y="134" width="310" height="66" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="565" y="162" fill="#1f2937" font-size="12.5" text-anchor="middle">weighting by usable share: 55%</text><text x="565" y="184" fill="#5b6471" font-size="12" text-anchor="middle">the honest coverage figure</text></svg>
<figcaption><b>A footprint is where the sensor pointed, not what it saw.</b> Coverage counts that ignore the usable fraction are systematically optimistic in exactly the regions — persistently cloudy ones — where the question is worth asking.</figcaption>
</figure>

**Items with identifiers that are not stable across reprocessing.** Some archives reissue items with new identifiers after reprocessing, which duplicates coverage in the index. Deduplicate on the tuple of collection, footprint and capture date rather than trusting identifiers alone.

**Date fields that are ranges, not instants.** Composite products carry a start and an end, and storing only one of them makes temporal queries wrong at the boundaries. Store both, or store the range type your database offers.

**Foreign keys that block ingest ordering.** Items referencing a collection that has not been ingested yet will fail insertion. Ingest collections first in the same transaction, or stage items and resolve references afterwards; discovering this in production means a half-ingested catalog.

## Frequently Asked Questions

<details class="faq-item"><summary><span>How fresh does the item index need to be?</span></summary><p>As fresh as the questions being asked. A corpus supporting historical analysis can refresh weekly; one answering "what does the latest imagery show" needs to be within a day of the archive. Publish the index age alongside answers so a stale index degrades gracefully into a dated answer rather than a wrong one — an agent that says "as of last Tuesday" is far more useful than one that implies currency it does not have.</p></details>

<details class="faq-item"><summary><span>Should item geometry be the true footprint or its bounding box?</span></summary><p>The true footprint when the source provides it, because scene footprints are frequently rotated quadrilaterals whose bounding box overstates coverage by a third. That overstatement lands exactly where it hurts: at the edges of a coverage gap, where the question of whether a place was observed is genuinely in doubt.</p></details>

<details class="faq-item"><summary><span>What should a coverage query return when nothing is indexed?</span></summary><p>A distinguishable "not indexed" rather than a zero count. Those two states mean opposite things — one says the place was never observed, the other says nobody asked the archive about it — and collapsing them into a zero produces an agent that confidently reports absence of data it simply never fetched.</p></details>

<details class="faq-item"><summary><span>Is it worth storing item-level metadata beyond footprint and date?</span></summary><p>Only the fields a query will filter on: usable fraction, processing level, sensor mode. Everything else can be fetched from the source when an item is actually selected, which is rare compared to how often items are counted. Storing the full item record for millions of items inflates the index by an order of magnitude to serve a lookup that happens once per answer.</p></details>

Finally, record the ingest restrictions somewhere the query layer can read. An index that deliberately holds items for three regions is a correct design and a dangerous one if the query layer does not know the boundary: outside it, every coverage question returns zero, and zero is indistinguishable from a genuine gap unless the restriction is data rather than folklore.

## Related

- Up to the parent topic: [Spatial Metadata and Catalog Indexing](/geospatial-rag-pipelines/spatial-metadata-and-catalog-indexing/)
- [Mapping Catalog Fields to Retrieval Filters](/geospatial-rag-pipelines/spatial-metadata-and-catalog-indexing/mapping-catalog-fields-to-retrieval-filters/)
- Concept: [Vector-Raster Hybrid Processing](/spatial-llm-architecture-core-concepts/vector-raster-hybrid-processing/)
- Technique: [Filtering Retrieval by Bounding Box Before Vector Search](/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/filtering-retrieval-by-bounding-box-before-vector-search/)
