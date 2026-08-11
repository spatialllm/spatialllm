---
layout: overview.njk
order: 3
navLabel: RAG Pipelines
icon: retrieval
title: Geospatial RAG Pipelines
description: Retrieval-augmented generation for corpora where the coordinates matter — chunking, frame resolution, catalog selection, hybrid ranking and stores that filter on geometry.
slug: geospatial-rag-pipelines
type: overview
breadcrumb: Geospatial RAG Pipelines
datePublished: 2025-03-05
dateModified: 2026-08-11
---

# Geospatial RAG Pipelines

Retrieval-augmented generation assumes that finding the right passage is a semantic problem. For corpora carrying geometry it is not: the right passage is the one about the right place, in the right period, from a dataset fit for the question, and none of those properties survive a pipeline that treats coordinates as ordinary text. This section covers the engineering that keeps them intact, from the moment a document is split to the moment an answer cites the collection it came from.

The failure this whole section prevents is quiet. A general-purpose pipeline over a geometry-rich corpus does not crash; it returns fluent, well-sourced answers about the wrong catchment, drawn from a survey that predates the thing being asked about, measured in a reference frame nobody declared. Every stage below exists because one of those properties is lost by default and has to be deliberately preserved.

<figure class="diagram">
<svg viewBox="0 0 860 320" role="img" aria-labelledby="rag-arch-t rag-arch-d" xmlns="http://www.w3.org/2000/svg"><title id="rag-arch-t">The geospatial retrieval pipeline end to end</title><desc id="rag-arch-d">Documents are chunked without severing geometry, stamped with a reference frame and extent, indexed alongside their vectors, then queried through catalog selection, a spatial filter, hybrid ranking and proximity reranking before an answer is composed with provenance.</desc><rect x="0" y="0" width="860" height="320" fill="#ffffff"/><text x="430" y="34" fill="#5b6471" font-size="13" text-anchor="middle">Ingest — every stage preserves a property the next one depends on</text><g fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"><rect x="24" y="52" width="188" height="72" rx="8"/><rect x="232" y="52" width="188" height="72" rx="8"/><rect x="440" y="52" width="188" height="72" rx="8"/><rect x="648" y="52" width="188" height="72" rx="8"/></g><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="118" y="82">chunk whole features</text><text x="326" y="82">resolve the frame</text><text x="534" y="82">stamp extent</text><text x="742" y="82">index geometry + vector</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="118" y="104">no severed rings</text><text x="326" y="104">no silent WGS84</text><text x="534" y="104">computed, not declared</text><text x="742" y="104">one store, one snapshot</text></g><text x="430" y="164" fill="#5b6471" font-size="13" text-anchor="middle">Query — the population narrows before anything expensive runs</text><g fill="#e4f5ec" stroke="#12805c" stroke-width="2"><rect x="24" y="182" width="188" height="72" rx="8"/><rect x="232" y="182" width="188" height="72" rx="8"/><rect x="440" y="182" width="188" height="72" rx="8"/><rect x="648" y="182" width="188" height="72" rx="8"/></g><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="118" y="212">choose the dataset</text><text x="326" y="212">filter by region</text><text x="534" y="212">fuse lexical + dense</text><text x="742" y="212">rerank by proximity</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="118" y="234">catalog, not guesswork</text><text x="326" y="234">index-aware predicate</text><text x="534" y="234">names survive</text><text x="742" y="234">near and relevant</text></g><g stroke="#5b6471" stroke-width="2" marker-end="url(#rag-arch-a)"><line x1="214" y1="88" x2="228" y2="88"/><line x1="422" y1="88" x2="436" y2="88"/><line x1="630" y1="88" x2="644" y2="88"/><line x1="214" y1="218" x2="228" y2="218"/><line x1="422" y1="218" x2="436" y2="218"/><line x1="630" y1="218" x2="644" y2="218"/></g><defs><marker id="rag-arch-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/></marker></defs><rect x="24" y="272" width="812" height="36" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="430" y="295" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">Answer cites the collection, version and frame it relied on</text></svg>
<figcaption><b>Two halves that have to agree.</b> Every query-side stage depends on a property an ingest-side stage preserved: the region filter needs the stamped extent, the reranker needs the whole geometry, the citation needs the collection identifier. Skip one on ingest and the corresponding query stage silently stops working.</figcaption>
</figure>

## Chunking Without Destroying Geometry

A fixed-size splitter cuts a coordinate list in half and produces text no parser will accept. The fix is not a bigger chunk size but a no-cut map: scan the document for geometry spans first, then choose boundaries only from the paragraph breaks that fall outside them.

What makes this different from ordinary prose chunking is the absence of graceful degradation. A sentence cut in half still carries most of its meaning and still embeds into roughly the right region of vector space; half a polygon is not a smaller polygon, it is a syntax error that embeds as noise and retrieves as though it were content. There is no partial credit anywhere in the pipeline for a malformed ring, which is why the constraint has to be structural rather than a heuristic that usually holds.

```python
def is_safe_offset(offset: int, spans: list[tuple[int, int]]) -> bool:
    """A cut is safe only if it lies strictly outside every geometry span."""
    return not any(start < offset < end for start, end in spans)


def safe_boundaries(text: str, spans, budget_ok) -> list[int]:
    candidates = [m.start() for m in PARAGRAPH_BREAK.finditer(text)]
    safe = [c for c in candidates if is_safe_offset(c, spans)]
    if not safe:                       # nothing legal survived: emit the document whole
        return [len(text)]
    return [c for c in safe if budget_ok(c)] or [len(text)]
```

The consequence worth internalising is that chunks in a spatial corpus are uneven, and that unevenness is correct. A feature is the atomic unit; sizes vary because features vary. The full treatment, including what to do with the one polygon that exceeds any budget, is in [chunk-boundary strategies for spatial corpora](/geospatial-rag-pipelines/chunk-boundary-strategies-for-spatial-corpora/).

The other decision this stage forces is what a chunk is for. A feature chunk holds one geometry and its attributes and gives the sharpest retrieval with the weakest context. A section chunk holds a run of features under one heading and recovers the relationships between them at the cost of precision. A document chunk is right only for small files and is also, quietly, what the splitter falls back to when no safe boundary exists — which means a corpus with no paragraph structure silently changes its retrieval unit unless someone is watching the flag. Choose deliberately, record the choice in the metadata, and treat a rising rate of whole-document fallbacks as a corpus-shape problem rather than a chunker bug.

## Making Every Chunk Self-Describing

A retrieved chunk arrives without its document. The header that declared the projection, the caption that named the survey, the note about units — all of it is gone unless it was copied into metadata at ingest time. Four fields carry the load: the frame, the computed extent, the source feature identifiers, and the period.

```python
def build_chunk_meta(geometries, feature_ids, declared_epsg):
    """Compute metadata from the geometry, not from the document's claims."""
    if not geometries:
        return {"epsg": declared_epsg, "bbox": (0.0, 0.0, 0.0, 0.0),
                "feature_ids": tuple(feature_ids), "note": "no parseable geometry"}
    bounds = union_bounds(geometries)                 # computed, so it cannot go stale
    return {"epsg": declared_epsg, "bbox": bounds,
            "feature_ids": tuple(feature_ids), "note": ""}
```

The extent in particular must be computed rather than copied. A document's declared extent describes what its author believed it contained, often before an edit; a chunk's extent must describe what the chunk holds, because the spatial filter downstream reads it as fact. The details are in [carrying frame and extent metadata into every chunk](/geospatial-rag-pipelines/chunk-boundary-strategies-for-spatial-corpora/carrying-crs-and-extent-metadata-into-every-chunk/).

It is worth being explicit about who reads each field, because a schema justified that way stays small and stays correct. The extent is read by the pre-filter, which is why a loose one costs a few extra candidates and a tight one silently removes a feature from every spatially filtered query. The feature identifiers are read by the citation layer, which is why they must come from the source rather than from the chunker's own numbering — identifiers derived from position change on every re-export and break every citation made against the previous build. The period is read by any query that cares whether a description is current. Only the text is embedded. Fields that nothing reads are the ones that rot, and a rotted field that looks authoritative is worse than an absent one.

## Resolving the Reference Frame Before Reasoning

Coordinates without a declared frame are the most dangerous input this pipeline handles, because every downstream stage will happily process them and produce answers displaced by hundreds of metres. The frame must be established from retrieved evidence and library validation — never generated by a model — and the result must carry a confidence the answer layer can gate on.

```python
def resolve_crs(mention, index_query, sample_xy=None, threshold=0.55):
    for cand in retrieve_crs_candidates(mention, index_query):
        if crs_is_usable(cand.epsg, sample_xy):       # constructs, current, plausible extent
            conf = round(min(1.0, 0.5 + cand.weight / 2), 3)
            return ResolvedCRS(cand.epsg, conf, conf < threshold, f"validated from {cand.source}")
    # Deterministic fallback: degrade to WGS84 and make the uncertainty visible.
    return ResolvedCRS(4326, 0.0, True, "no candidate validated")
```

The flagged fallback is the load-bearing part. Silence is the failure mode: a pipeline that assumes geographic coordinates when it has no evidence produces confident measurements from a guess. See [retrieval-augmented CRS resolution](/geospatial-rag-pipelines/retrieval-augmented-crs-resolution/) for the scoring, and [resolving ambiguous EPSG codes from document context](/geospatial-rag-pipelines/retrieval-augmented-crs-resolution/resolving-ambiguous-epsg-codes-from-document-context/) for the case where a declaration exists but underdetermines the code.

Confidence has to be graded rather than binary, because most real cases sit between certainty and ignorance. A frame supported by a dataset header that declares it, validated by a round trip, deserves to answer measurement questions plainly. One supported only by co-occurrence with a place name in retrieved neighbours deserves to answer descriptive questions with a hedge and to refuse distances outright. Collapsing that spectrum into a single low-confidence flag forces every uncertain case into the same refusal and, in practice, leads to teams turning the flag off.

The other half of this stage is making the flag consequential. A confidence recorded only in a log line changes nothing about what the system says. It has to reach the answer path, where it turns a measured claim into an explicit statement about what could not be established — which is the one output a reader can act on, because it tells them exactly what to go and find.

## Choosing the Dataset Before Searching Its Contents

Spatial questions need two searches, not one. The first asks which collection covers this place, this period, at this resolution, under a licence you may use; the second searches inside the chosen collection. Collapsing them produces answers drawn from whatever dataset happened to embed nearby.

```sql
-- Search one: the catalog. Small table, structured filters, near-exact.
SELECT collection_id, title, resolution_m, licence
FROM   catalog_collections
WHERE  geom && ST_MakeEnvelope(:w, :s, :e, :n, 4326)
  AND  ST_Intersects(geom, ST_MakeEnvelope(:w, :s, :e, :n, 4326))
  AND  period && daterange(:from_date, :to_date)
  AND  (resolution_m IS NULL OR resolution_m <= :max_resolution_m)
ORDER  BY resolution_m NULLS LAST
LIMIT  20;
```

Fitness for purpose is computable from that metadata: a thirty-metre product cannot answer a question about one building, and a licence that forbids the use is not a penalty but an exclusion. Carrying the reasons forward is what lets an agent say "the best available data is coarser than your question needs" instead of answering anyway. This is developed in [spatial metadata and catalog indexing](/geospatial-rag-pipelines/spatial-metadata-and-catalog-indexing/).

There is a second gap inside this stage that catches most teams once. A collection's declared extent is a rectangle drawn around its observations, not a claim that every point inside was observed. Satellite archives are cloud-gapped, aerial programmes are campaign-based, and ground surveys are sparse points inside a national bounding box. Answering a coverage question from collection-level metadata alone produces confident statements about places nobody ever looked at, so any collection whose coverage is uneven needs its items indexed too, and coverage becomes a count rather than an assumption.

The output of this stage should be a shortlist with scores and caveats, not a single winner. A model handed one collection has nothing to reason about when that collection has limitations; a model handed three, with the reasons each scored as it did, can explain the trade-off it is making. It also makes the selection reviewable, which matters the first time an answer is disputed and someone asks why that dataset and not the obvious alternative.

<figure class="diagram">
<svg viewBox="16 42 758 208" role="img" aria-labelledby="rag-narrow-t rag-narrow-d" xmlns="http://www.w3.org/2000/svg"><title id="rag-narrow-t">How the candidate population narrows and what each stage costs</title><desc id="rag-narrow-d">A national corpus narrows to a region by an index lookup, then to a few hundred by vector comparison, then to a handful by fusion and reranking, with cost per stage falling as the set shrinks.</desc><rect x="16" y="42" width="758" height="208" fill="#ffffff"/><rect x="30" y="56" width="130" height="150" rx="8" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="180" y="76" width="130" height="110" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="330" y="92" width="130" height="78" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="480" y="102" width="130" height="58" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="630" y="110" width="130" height="42" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="12.5" text-anchor="middle" font-weight="600"><text x="95" y="126">corpus</text><text x="245" y="126">in region</text><text x="395" y="126">by vector</text><text x="545" y="128">fused</text><text x="695" y="128">reranked</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="95" y="148">2 000 000</text><text x="245" y="148">4 100</text><text x="395" y="148">200</text><text x="545" y="148">40</text><text x="695" y="148">8</text></g><g stroke="#5b6471" stroke-width="2" marker-end="url(#rag-narrow-a)"><line x1="162" y1="131" x2="176" y2="131"/><line x1="312" y1="131" x2="326" y2="131"/><line x1="462" y1="131" x2="476" y2="131"/><line x1="612" y1="131" x2="626" y2="131"/></g><defs><marker id="rag-narrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/></marker></defs><text x="400" y="232" fill="#1f2937" font-size="13" text-anchor="middle">The spatial filter removes 99.8% for the price of one index lookup</text></svg>
<figcaption><b>Order the stages by cost.</b> Everything after the region filter operates on a set small enough that cross-encoding and geometry distance are affordable. Reverse the order and the same pipeline is a hundred times more expensive for the same result.</figcaption>
</figure>

## Filtering by Region Before Comparing Vectors

The single most reliable performance decision in this section is putting the spatial predicate before the vector comparison, written so the index is actually used: the bounding-box operator first, the exact predicate second, and no function wrapped around the indexed column.

```sql
SELECT c.chunk_id, 1 - (c.embedding <=> :qvec) AS semantic
FROM   spatial_chunks c
WHERE  c.geom && :bbox                     -- index-aware pre-filter
  AND  ST_Intersects(c.geom, :region)      -- exact predicate on the survivors
ORDER  BY c.embedding <=> :qvec
LIMIT  200;
```

Two things break this in practice. Wrapping the geometry column in a transform makes the index unusable — transform the query region instead. And a filter on extent silently excludes every chunk whose extent is missing, so a corpus with partial geometry coverage needs an explicit lane for the rest. Both are covered in [filtering retrieval by bounding box before vector search](/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/filtering-retrieval-by-bounding-box-before-vector-search/).

Sizing the region is its own decision and does not belong in a constant. A question about protected trees on a plot wants tens of metres; a question about flood risk to a street wants a catchment; a question about what the local plan says wants an administrative boundary that may be twenty kilometres across. A pipeline with one radius answers the first with noise and the third with nothing. Where the classifier is unsure, widen rather than narrow: a region that is too large costs latency and dilutes ranking slightly, while one that is too small removes the correct answer from consideration entirely, and no reranking recovers a candidate that was never retrieved.

Record the region that was used alongside the results. When an answer is disputed, the first question is almost always which area was searched, and a pipeline that cannot answer it will be assumed to have searched the wrong one.

## Ranking Where Place and Meaning Both Count

Inside the region, two signals order the survivors: how well a candidate matches the subject, and how close it is. Fusing them beats sequencing them, because anything the first stage missed is unrecoverable in a sequential design.

```python
def fuse(semantic: float, distance_m: float, radius_m: float,
         w_semantic: float = 0.65, half_life_m: float = 1500.0) -> float:
    """Combine similarity with a decaying proximity term; outside the region scores zero."""
    if distance_m > radius_m:
        return 0.0                                    # a constraint, not a penalty
    if distance_m < 0 or not isfinite(distance_m):
        proximity = 0.0                               # unknown position scores as far
    else:
        proximity = 0.5 ** (distance_m / half_life_m)
    return round(w_semantic * max(0.0, min(1.0, semantic))
                 + (1.0 - w_semantic) * proximity, 6)
```

Exact token matching is the third signal and the one dense retrieval handles worst. Place names and reference codes are precisely what an embedding blurs and precisely what users type, so a lexical ranking runs alongside the dense one and the two are combined on rank rather than on score. See [spatial context retrieval and reranking](/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/) and [hybrid spatial and keyword retrieval](/geospatial-rag-pipelines/hybrid-spatial-keyword-retrieval/).

Geography constrains and does not vote. It is tempting to treat distance as a third score and blend all three into one weighted sum, and doing so lets a strong lexical match buy its way back in from the wrong region — the exact failure the filter exists to prevent. Filter first, fuse the two text rankings, then apply proximity within the surviving population. Each of the three then does one job, and when a result is wrong it is possible to say which of them produced it.

The shape of the proximity term matters more than its weight. An exponential decay expresses the intuition that the first kilometre matters far more than the tenth, which is true for site-specific questions and roughly true for most others. A linear decay makes a candidate at nine kilometres nearly as good as one at eight, which is right for a policy document and wrong for a site report. Most disagreements about how much distance should matter turn out to be disagreements about decay shape rather than about a scalar.

## Storing Vectors and Geometry Together

The store decision is usually argued on throughput and decided by filtering behaviour. A store that returns high recall unfiltered and loses a third of it when a bounding box removes 99% of the corpus is the wrong store for this workload, because the filtered case is the normal case.

```python
def index_footprint(n: int, dim: int, m: int = 16, bytes_per_link: int = 4) -> dict:
    """Vectors plus graph links — the memory bill nobody plans for."""
    vectors, links = n * dim * 4, n * m * 2 * bytes_per_link
    return {"vectors_gib": round(vectors / 1024**3, 2),
            "total_gib": round((vectors + links) / 1024**3, 2)}
```

Keeping geometry in the same store as the vector removes a two-system join and, more importantly, removes the possibility that the filter and the score disagree about which chunks exist. The criteria and the bake-off harness are in [spatial vector store selection](/geospatial-rag-pipelines/spatial-vector-store-selection/), with parameter sizing in [sizing HNSW parameters for spatial recall](/geospatial-rag-pipelines/spatial-vector-store-selection/sizing-hnsw-parameters-for-spatial-recall/).

Rebuild cost is the criterion that decides how the system evolves rather than how it performs. Embedding models change, chunking strategies change, and each change means reindexing. A store that takes six hours to rebuild makes those changes quarterly events and, over a couple of years, ossifies the pipeline around whatever model was current when it was built. One that rebuilds in twenty minutes makes reindexing a thing you do to test a hypothesis. That is an infrastructure decision with an entirely product-shaped consequence.

## Operating the Pipeline Over Time

Everything above describes a pipeline that is correct on the day it ships. Three things erode it afterwards, and none of them announce themselves.

The first is index quality. Records inserted after a graph index is built are wired into the graph as it exists at that moment, which over months produces a less well-connected structure than a full rebuild would. Nothing errors; recall simply drifts downward, and answers get slightly worse each month. The control is a scheduled recall measurement against brute-force truth under a realistic filter — not a scheduled rebuild, which treats the symptom without ever telling you whether it was needed.

The second is catalog staleness. New collections appear and existing ones extend their coverage, and a catalog refreshed six months ago will confidently report that no data exists for a region that has been well covered since spring. Publish the refresh age as a number, alert on it, and watch the normalisation rejection rate alongside it: a step change in rejections is the earliest visible sign that an upstream source has changed its schema, usually days or weeks before anyone notices the missing coverage in an answer.

The third is quieter still. Source documents get corrected, geometry gets resurveyed, and the chunk in the index still holds the superseded version. Version the corpus, record which version an answer used, and treat a correction as a reason to re-chunk the affected documents rather than to patch the store. A retrieval system that cannot say which version of a document it read is one whose answers cannot be reproduced, and reproducibility is the property that makes every other control in this section checkable.

<figure class="diagram">
<svg viewBox="16 32 748 214" role="img" aria-labelledby="rag-fail-t rag-fail-d" xmlns="http://www.w3.org/2000/svg"><title id="rag-fail-t">Four silent failures and the stage that prevents each</title><desc id="rag-fail-d">Severed geometry, an assumed reference frame, an unchosen dataset and a region-blind ranking each produce fluent wrong answers, and each is prevented by one specific stage of the pipeline.</desc><rect x="16" y="32" width="748" height="214" fill="#ffffff"/><rect x="30" y="46" width="360" height="86" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="410" y="46" width="340" height="86" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="30" y="146" width="360" height="86" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="410" y="146" width="340" height="86" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13" font-weight="600"><text x="52" y="76">severed geometry</text><text x="432" y="76">assumed frame</text><text x="52" y="176">unchosen dataset</text><text x="432" y="176">region-blind ranking</text></g><g fill="#5b6471" font-size="12"><text x="52" y="100">embeds as noise, parses as nothing</text><text x="52" y="120">prevented by the no-cut map</text><text x="432" y="100">answers displaced by hundreds of metres</text><text x="432" y="120">prevented by the flagged fallback</text><text x="52" y="200">answers from whatever embedded nearby</text><text x="52" y="220">prevented by the catalog search</text><text x="432" y="200">the eloquent document from elsewhere</text><text x="432" y="220">prevented by filter then fusion</text></g></svg>
<figcaption><b>None of these throw an exception.</b> That is what unites them: each produces a fluent, plausible, sourced answer, which is why the controls have to be structural rather than reactive.</figcaption>
</figure>

## What This Section Assumes

Three things from elsewhere on the site are treated as settled here rather than re-argued.

The first is representation. How a geometry becomes tokens — the trade between coordinate precision and token cost, the choice between well-known text, a structured object and a hierarchical cell identifier — belongs to [geometry tokenization strategies](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/), and every chunk in this section carries whatever that decision produced. The choice matters here mostly through its effect on chunk size: a representation that triples a polygon's token count triples the number of chunks a document needs.

The second is the reference frame as an ingestion-time concern. Documents that declare their projection are handled by [coordinate reference system normalization](/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/) before they reach a chunker; retrieval-augmented resolution exists for the harder residue where nothing is declared anywhere.

The third is what happens after retrieval. Once context is assembled, an agent decides which tool to call and how to recover when the call fails, which is the subject of [geospatial prompt engineering and tool routing](/geospatial-prompt-engineering-tool-routing/). The boundary between the two sections is worth stating plainly: retrieval decides what the model is allowed to know, routing decides what it does about it. Confusing them produces a system that retries a search when it should have called a geometry engine, or calls a geometry engine on a document it never actually read.

## Production Engineering Standards

1. **Every chunk parses.** Load each emitted chunk's geometry with a reader during the index build; a parse failure fails the build, because it proves a feature was severed.
2. **Every chunk carries a frame and an extent.** A missing frame is a hard error; an extent must be computed from the geometry, never copied from a document header.
3. **No frame is ever generated.** Reference systems come from retrieval plus library validation, and an unresolved frame degrades to WGS84 with a flag that changes what the agent may claim.
4. **The spatial predicate runs first, and the plan proves it.** Assert in a query test that the spatial index is used; a plan regression is silent and costs two orders of magnitude.
5. **Filtered recall is measured, not assumed.** Measure recall at three selectivities on every index build and fail the build if the tightest case drops more than five points below baseline.
6. **Both ranking components are visible.** Every returned candidate reports its semantic score, its distance and the fused total, so an odd result can be attributed rather than argued about.
7. **Every answer names its source.** Collection identifier and version reach the answer, or the build fails — an unattributed spatial claim cannot be checked and should not be made.
8. **Staleness is a published number.** Catalog refresh age, index build age and corpus version are exposed alongside answers, so a dated answer reads as dated rather than as current.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Does a geospatial corpus need a different embedding model?</span></summary><p>Usually not. The text in these corpora is ordinary technical prose, and a general model handles it well; what a specialised model would help with — reasoning about coordinates numerically — is not something embeddings do for any model. Where specialisation does pay is in the lexical half, through an analyser that keeps reference codes and place names intact, and in the metadata, which is where the actual spatial reasoning happens.</p></details>

<details class="faq-item"><summary><span>How many documents should reach the model's context?</span></summary><p>Fewer than the window allows. The useful number is set by how many genuinely distinct places and viewpoints the question needs, which is usually between four and ten. Filling the remaining space with weak candidates measurably degrades answers, because the model must decide which of twenty documents to trust and the weakest ones are the likeliest to contain a confidently phrased irrelevance. Cut the list where the fused score falls off, not where the token budget does.</p></details>

<details class="faq-item"><summary><span>What if the corpus has no geometry at all, only place names?</span></summary><p>Then geocode at ingest time and store the result as chunk metadata with a confidence, rather than resolving names at query time. Resolution is expensive, and a name that resolves once resolves the same way for every later query, which is a consistency property worth having. Keep the original name in the chunk text so the lexical half can still match it exactly — see <a href="/spatial-llm-architecture-core-concepts/geocoding-and-place-name-resolution/">geocoding and place-name resolution</a>.</p></details>

<details class="faq-item"><summary><span>Where should evaluation for this pipeline live?</span></summary><p>In a labelled retrieval set of its own, separate from the end-to-end agent evaluation. Retrieval quality and answer quality fail independently, and a combined score cannot tell you which one moved. Measure recall at a fixed depth over a few dozen queries spanning both site-specific and regional questions, and track it on every index build; the broader agent-level thresholds are the subject of <a href="/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/">evaluation and benchmarking for spatial LLMs</a>.</p></details>

## Related

- Section: [Spatial LLM Architecture and Core Concepts](/spatial-llm-architecture-core-concepts/) — the representation and evaluation groundwork this pipeline assumes
- Section: [Geospatial Prompt Engineering and Tool Routing](/geospatial-prompt-engineering-tool-routing/) — what an agent does with the context this pipeline assembles
- Topic: [Chunk-Boundary Strategies for Spatial Corpora](/geospatial-rag-pipelines/chunk-boundary-strategies-for-spatial-corpora/)
- Topic: [Retrieval-Augmented CRS Resolution](/geospatial-rag-pipelines/retrieval-augmented-crs-resolution/)
- Topic: [Spatial Context Retrieval and Reranking](/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/)
- Topic: [Spatial Vector Store Selection](/geospatial-rag-pipelines/spatial-vector-store-selection/)
- Topic: [Spatial Metadata and Catalog Indexing](/geospatial-rag-pipelines/spatial-metadata-and-catalog-indexing/)
- Topic: [Hybrid Spatial and Keyword Retrieval](/geospatial-rag-pipelines/hybrid-spatial-keyword-retrieval/)
