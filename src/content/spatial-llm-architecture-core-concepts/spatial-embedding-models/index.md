---
title: Spatial Embedding Models
description: Choose and evaluate embeddings for geometry-bearing corpora — what to embed, what not to, how to measure it honestly, and where a general model is already enough.
slug: spatial-embedding-models
type: topic
breadcrumb: Spatial Embedding Models
datePublished: 2025-02-18
dateModified: 2026-08-11
---

# Spatial Embedding Models

An embedding is a claim that similar things end up near each other. For geometry-bearing corpora that claim needs interrogating, because "similar" has at least three meanings — about the same subject, about the same place, and about the same kind of place — and no single vector space captures all of them. This topic covers what to embed, what to leave as metadata, and how to measure whether a candidate model is actually better for your corpus rather than better on a leaderboard.

It belongs to [spatial LLM architecture and core concepts](/spatial-llm-architecture-core-concepts/) and supplies the vectors that [spatial vector store selection](/geospatial-rag-pipelines/spatial-vector-store-selection/) has to store and [spatial context retrieval and reranking](/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/) has to rank. Its most consequential output is a single number — dimensionality — which sets the memory bill for the whole retrieval system.

<figure class="diagram">
<svg viewBox="16 38 748 188" role="img" aria-labelledby="sem-what-t sem-what-d" xmlns="http://www.w3.org/2000/svg"><title id="sem-what-t">What belongs in the vector and what belongs in metadata</title><desc id="sem-what-d">Descriptive prose and place names belong in the embedded text; coordinates, extents and dates belong in structured metadata where they can be filtered exactly rather than approximated.</desc><rect x="16" y="38" width="748" height="188" fill="#ffffff"/><rect x="30" y="52" width="340" height="160" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="410" y="52" width="340" height="160" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="200" y="84">embed this</text><text x="580" y="84">filter on this</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="200" y="114">the description of the place</text><text x="200" y="138">what happens there</text><text x="200" y="162">the name, spelled as written</text><text x="200" y="188">the attributes in prose</text><text x="580" y="114">the coordinates</text><text x="580" y="138">the bounding extent</text><text x="580" y="162">the capture date</text><text x="580" y="188">the licence and source</text></g></svg>
<figcaption><b>Coordinates in the vector are the classic mistake.</b> Embedding a number places it near other numbers of similar magnitude, which has nothing to do with proximity on the ground — and it consumes the dimensions that were meant to carry meaning.</figcaption>
</figure>

## Foundational Principles

**Do not embed coordinates.** A vector space arranges things by learned similarity, and coordinate strings have no useful similarity structure: 51.5 and 51.6 are adjacent, 51.5 and -0.1 are not, and the model has no way to know that the pair means London. Position belongs in a spatial index where it can be filtered exactly.

**Embed the name as written, and keep the lexical index too.** Place names are the strings users type and the strings dense models blur. Keep them in the embedded text because the surrounding prose gives them meaning, and keep the lexical half described in [hybrid spatial and keyword retrieval](/geospatial-rag-pipelines/hybrid-spatial-keyword-retrieval/) because only exact matching finds a rare toponym reliably.

**Dimensionality is an infrastructure decision.** Doubling it doubles the memory of every index and every replica, and the retrieval quality gain is usually small on corpora of ordinary size. Choose it against measured recall on your own data, as set out in [choosing vector dimensionality for spatial retrieval](/spatial-llm-architecture-core-concepts/spatial-embedding-models/choosing-vector-dimensionality-for-spatial-retrieval/).

## Step-by-Step Implementation Pipeline

### 1. Decide what text represents each chunk

The embedded text is a construction, not simply the chunk body. Including the feature's name, its type and a short parent context measurably improves retrieval for place-oriented questions; including the raw coordinates measurably harms it.

```python
import logging
from dataclasses import dataclass

log = logging.getLogger("spatial_embeddings")


@dataclass(frozen=True)
class Chunk:
    body: str
    name: str | None
    feature_type: str | None
    parents: tuple[str, ...]        # containing administrative names
    epsg: int | None


def embedding_text(chunk: Chunk, max_chars: int = 4000) -> str:
    """Build the string that gets embedded. Coordinates deliberately excluded."""
    head = []
    if chunk.name:
        head.append(chunk.name)
    if chunk.feature_type:
        head.append(chunk.feature_type)
    if chunk.parents:
        head.append(", ".join(chunk.parents[:2]))     # nearest containers only
    prefix = " — ".join(head)
    text = f"{prefix}\n{chunk.body}" if prefix else chunk.body
    if len(text) > max_chars:
        log.info("embedding text truncated from %d to %d chars", len(text), max_chars)
        text = text[:max_chars]
    return text
```

Truncating rather than splitting is right here because the chunk has already been sized for retrieval; a chunk whose embedding text exceeds the model's input is a chunking problem, and the log line is what surfaces it.

### 2. Choose candidates on the basis of your corpus, not a leaderboard

General-purpose benchmarks measure performance on general-purpose text. A corpus of survey reports, planning documents and technical assessments is not that, and the ranking frequently reorders. Shortlist on practical grounds — input length, dimensionality, licence, latency — and settle it by measurement.

```python
@dataclass(frozen=True)
class Candidate:
    name: str
    dim: int
    max_input_tokens: int
    normalises: bool               # does it return unit vectors already?


def shortlist(candidates, corpus_p95_tokens: int, memory_budget_gib: float,
              n_chunks: int) -> list[Candidate]:
    """Filter on hard constraints before measuring anything."""
    out = []
    for c in candidates:
        if c.max_input_tokens < corpus_p95_tokens:
            log.info("%s rejected: input limit %d below corpus p95 %d",
                     c.name, c.max_input_tokens, corpus_p95_tokens)
            continue
        gib = n_chunks * c.dim * 4 / 1024 ** 3
        if gib > memory_budget_gib:
            log.info("%s rejected: %.1f GiB of raw vectors exceeds the %.1f GiB budget",
                     c.name, gib, memory_budget_gib)
            continue
        out.append(c)
    return out
```

### 3. Measure on place-oriented queries, not generic ones

The evaluation set for an embedding decision needs the queries this corpus actually receives, and those skew toward names, types and local descriptions. Measuring on generic paraphrase pairs will rank models by a property you are not buying.

```python
def recall_at_k(model, queries, truth, encode_corpus, k: int = 10) -> float:
    """Recall over a labelled set of real place-oriented queries."""
    corpus_vecs = encode_corpus(model)
    hit = tot = 0
    for q, want in zip(queries, truth):
        try:
            got = set(nearest(model.encode(q), corpus_vecs, k))
        except Exception as exc:
            log.warning("encode failed for %r: %s — scoring as a miss", q, exc)
            got = set()
        hit += len(want & got)
        tot += len(want)
    return round(hit / tot, 4) if tot else 0.0
```

### 4. Measure under the filter, because that is the workload

Unfiltered recall is a property of the model; filtered recall is a property of the model and the corpus together, and it is what users experience. A model whose vectors cluster tightly by region can look excellent unfiltered and mediocre once a bounding box has removed everything outside one town.

```python
def recall_under_region(model, queries, truth, encode_corpus, regions, k: int = 10) -> float:
    """Recall when the candidate set is restricted to the query's own region."""
    corpus_vecs = encode_corpus(model)
    hit = tot = 0
    for q, want, region in zip(queries, truth, regions):
        eligible = [i for i, meta in enumerate(corpus_meta) if meta["region"] == region]
        if not eligible:
            continue
        got = set(nearest_within(model.encode(q), corpus_vecs, eligible, k))
        hit += len(want & got)
        tot += len(want)
    return round(hit / tot, 4) if tot else 0.0
```

<figure class="diagram">
<svg viewBox="66 7 632 239" role="img" aria-labelledby="sem-dim-t sem-dim-d" xmlns="http://www.w3.org/2000/svg"><title id="sem-dim-t">Recall and memory across embedding dimensionality</title><desc id="sem-dim-d">Recall rises steeply from low dimensionality and flattens well before the largest models, while memory grows linearly, so the useful choice sits at the knee rather than at the top.</desc><rect x="66" y="7" width="632" height="239" fill="#ffffff"/><text x="390" y="32" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">Retrieval recall and index memory against dimensionality</text><g fill="#e4f5ec" stroke="#12805c" stroke-width="2"><rect x="80" y="122" width="50" height="52" rx="4"/><rect x="240" y="88" width="50" height="86" rx="4"/><rect x="400" y="76" width="50" height="98" rx="4"/><rect x="560" y="72" width="50" height="102" rx="4"/></g><g fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"><rect x="136" y="158" width="30" height="16" rx="3"/><rect x="296" y="146" width="30" height="28" rx="3"/><rect x="456" y="118" width="30" height="56" rx="3"/><rect x="616" y="62" width="30" height="112" rx="3"/></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="118" y="196">256</text><text x="278" y="196">384</text><text x="438" y="196">768</text><text x="598" y="196">1536</text></g><text x="390" y="228" fill="#1f2937" font-size="12.5" text-anchor="middle">Wide bars are recall; narrow bars are memory — the knee sits near 384 to 768</text></svg>
<figcaption><b>The last doubling buys almost nothing and costs everything.</b> Between 768 and 1536 dimensions recall moves by under a point on most spatial corpora while index memory doubles, which is the trade that turns a single-machine deployment into a cluster.</figcaption>
</figure>

### 5. Normalise once, consistently

Cosine similarity assumes unit vectors. Some models return them and some do not, and a pipeline that normalises inconsistently produces scores that are not comparable between the two halves of its own corpus.

```python
import numpy as np


def normalise(vectors: np.ndarray, already_unit: bool) -> np.ndarray:
    """Return unit vectors. Idempotent, and safe on zero vectors."""
    if already_unit:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    zero = norms.squeeze() == 0
    if zero.any():
        log.warning("%d zero-length embedding(s); leaving them unnormalised", int(zero.sum()))
        norms[norms == 0] = 1.0
    return vectors / norms
```

### 6. Decide about fine-tuning by measuring the gap, not by ambition

Fine-tuning an embedding on domain text is a real improvement in some corpora and a maintenance burden in all of them. The decision should follow a measurement: how much of the retrieval failure is attributable to the embedding rather than to chunking, filtering or the lexical half.

```python
def attribution(recall_dense: float, recall_lexical: float, recall_hybrid: float) -> str:
    """Where is the headroom? Fine-tune only when the dense half is the binding constraint."""
    if recall_hybrid - max(recall_dense, recall_lexical) < 0.02:
        return "fusion is not contributing — fix the weaker half before touching the model"
    if recall_dense < recall_lexical - 0.10:
        return "dense half is the constraint — a domain-tuned embedding may help"
    return "dense half is competitive — invest in chunking or filtering instead"
```

Most teams that reach for fine-tuning discover, on running something like this, that their retrieval failures come from chunk boundaries or a missing spatial filter. Those are cheaper to fix and do not create a model artefact that has to be retrained whenever the corpus shifts.

### 7. Pin the model and record it with the index

An embedding is only comparable to other embeddings from the same model and the same version. Storing the model identifier alongside the index is what lets a later query detect that it is comparing vectors from two generations.

```python
def index_manifest(model_name: str, model_version: str, dim: int, normalised: bool) -> dict:
    return {"model": model_name, "version": model_version,
            "dim": dim, "normalised": normalised}


def assert_compatible(query_manifest: dict, index_manifest_: dict) -> None:
    for key in ("model", "version", "dim", "normalised"):
        if query_manifest[key] != index_manifest_[key]:
            raise ValueError(
                f"embedding mismatch on {key}: query {query_manifest[key]!r} "
                f"against index {index_manifest_[key]!r}")
```

### 8. Plan the migration before you need it

Changing an embedding means re-encoding the corpus and rebuilding the index, and doing that without downtime requires the plan to exist beforehand: build the new index alongside the old, verify recall on the labelled set, switch reads, then retire. A team that has not rehearsed this will treat the embedding as immutable, which is the real cost of not planning it.

```python
def migration_ready(new_recall: float, old_recall: float, tolerance: float = 0.01) -> bool:
    """Do not switch on faith; switch on a measurement against the same labelled set."""
    if new_recall + tolerance < old_recall:
        log.warning("new embedding recall %.3f is below current %.3f", new_recall, old_recall)
        return False
    return True
```

## Operating This Stage Over Time

Embedding choices age differently from most infrastructure: they get better around you. A model two generations old is not broken, and the pressure to migrate comes from a general sense that something newer exists rather than from a measurement. The labelled retrieval set is what turns that into a decision — run the new candidate against it, and migrate when the number moves by an amount that justifies the rebuild.

The corpus drifts too, and less visibly. A retrieval system that started on planning documents and has since absorbed field survey notes is being asked to embed a different register of text, and the model that measured best on the first corpus may not be best on the mixed one. Re-running the comparison annually, or whenever a substantial new source is added, costs an afternoon and occasionally changes the answer.

Watch for one specific failure that has no error: a partial re-encode. If half the corpus is embedded with one model version and half with another, every similarity comparison across that boundary is meaningless, and nothing raises. The manifest assertion in step 7 is the guard; it only works if it is checked at query time as well as at build time, because the mixed state arises during a migration rather than after it.

Finally, keep the dimensionality decision open. It is tempting to treat it as settled once the index exists, and it is one of the few parameters where a later reduction — through a model that supports shortened outputs, or through dimensionality reduction fitted on the corpus — can halve the memory bill for a measurable, and often acceptable, recall cost. Revisit it whenever the index outgrows its host, before reaching for a larger host.

Two habits make the rest of this cheaper. Keep the encoding step idempotent and content-addressed, so re-encoding a corpus skips everything whose embedding text has not changed — which turns a full rebuild from an overnight job into a short one for most changes. And keep the labelled retrieval set in the repository next to the encoder, so the comparison that justifies a model choice is something anyone can re-run rather than something that lived in one person's notebook.

## Failure Modes & Root Causes

**Coordinates in the embedded text.** Retrieval quality drops for no apparent reason after a chunking change that started including geometry. Root cause: numeric strings occupying vector space with no useful structure. Mitigation: build the embedding text explicitly, as in step 1, rather than embedding the raw chunk.

**Mixed model versions in one index.** Similarity scores that are meaningless across part of the corpus, with no error anywhere. Root cause: a partial re-encode. Mitigation: the manifest, asserted at query time.

**Leaderboard selection.** A model chosen on general benchmarks underperforms a smaller one on the actual corpus. Root cause: measuring the wrong thing. Mitigation: a labelled set of real queries, measured under the real filter.

**Inconsistent normalisation.** Cosine scores from two halves of the corpus are on different scales. Root cause: normalising conditionally on a per-batch flag. Mitigation: normalise once at write time, record it in the manifest.

## Production Validation Protocols

1. **Manifest assertion.** Assert model, version, dimensionality and normalisation match between query and index on every request.
2. **No-coordinate test.** Assert the embedding text contains no coordinate-shaped token, using a fixture chunk full of geometry.
3. **Filtered recall gate.** Measure recall under a realistic region filter on every index build; gate on a regression greater than two points.
4. **Zero-vector check.** Assert no stored vector has zero length; a zero vector matches nothing and usually means an empty embedding text.
5. **Dimensionality budget.** Recompute the index memory model whenever the model changes, and compare it against the host before the rebuild rather than after.
6. **Migration rehearsal.** Rebuild the index with the current model on a schedule, to keep the migration path exercised even when the model has not changed.

<figure class="diagram">
<svg viewBox="26 9 614 241" role="img" aria-labelledby="sem-attr-t sem-attr-d" xmlns="http://www.w3.org/2000/svg"><title id="sem-attr-t">Where retrieval headroom actually sits</title><desc id="sem-attr-d">Four candidate causes of retrieval failure — chunk boundaries, missing spatial filter, lexical gap and the embedding itself — with the embedding usually the smallest contributor.</desc><rect x="26" y="9" width="614" height="241" fill="#ffffff"/><text x="380" y="34" fill="#5b6471" font-size="13" text-anchor="middle">Typical attribution of retrieval failures on a spatial corpus</text><rect x="40" y="60" width="380" height="40" rx="5" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="40" y="110" width="300" height="40" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="40" y="160" width="200" height="40" rx="5" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="40" y="210" width="90" height="40" rx="5" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><g fill="#1f2937" font-size="12.5"><text x="440" y="86">chunk boundaries</text><text x="360" y="136">missing or wrong spatial filter</text><text x="260" y="186">lexical gap on names and codes</text><text x="150" y="236">the embedding model itself</text></g></svg>
<figcaption><b>The model is usually the smallest bar.</b> Teams reach for a better embedding first because it is the most tractable-looking change, and the three bars above it are cheaper to fix and larger.</figcaption>
</figure>

The zero-vector check is the cheapest of these and catches the most confusing failure. An empty embedding text — produced by a chunk whose body was whitespace, or by a truncation that removed everything — encodes to a vector that matches nothing and produces a chunk which is present in the index, counted in every statistic, and unreachable by any query.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Is there such a thing as a genuinely spatial embedding?</span></summary><p>There are models trained to embed geometry or trajectories, and they are useful for tasks where the shape itself is the subject — matching similar building footprints, clustering movement patterns. For document retrieval over corpora that happen to contain geometry, they solve a problem you do not have: the text is ordinary technical prose and the position is better handled by a spatial index. Reach for them when comparing shapes, not when finding documents.</p></details>

<details class="faq-item"><summary><span>Should the place name appear in the embedded text if it is also in a lexical index?</span></summary><p>Yes. The lexical index finds the name; the embedding needs it for context, because a chunk describing "the site" without naming it embeds ambiguously. Duplication between the two halves is not waste — they are doing different jobs with the same string, and removing it from either one measurably hurts.</p></details>

<details class="faq-item"><summary><span>How much does chunk length affect embedding quality?</span></summary><p>More than model choice, in most corpora. Very long chunks average away the specific content that makes them findable; very short ones lack the context that disambiguates them. The practical range is a few hundred to a thousand tokens of prose, and the fact that this range is set by retrieval quality rather than by the model's input limit is worth stating, because the limit is usually much higher and invites over-long chunks.</p></details>

<details class="faq-item"><summary><span>Can dimensionality be reduced after the fact?</span></summary><p>Sometimes, and it is worth knowing before you provision hardware. Some models are trained so that a prefix of the vector is itself a usable embedding, which makes reduction a slice; otherwise a projection fitted on the corpus can cut dimensionality substantially for a small recall cost. Either way, measure the reduced version on the labelled set — the loss is corpus-dependent and occasionally much larger than the general figures suggest.</p></details>

<details class="faq-item"><summary><span>Does the same model need to serve queries and documents?</span></summary><p>The same model and the same version, yes, unless it is explicitly an asymmetric model with separate query and document encoders — in which case use both halves as intended and record that in the manifest. Mixing a query from one model with documents from another produces a similarity that is arithmetically valid and semantically meaningless, and it is one of the few failures here that produces no symptom other than poor results.</p></details>

One closing note on expectations. For most geometry-bearing corpora a competent general-purpose embedding, a good chunking strategy and a working spatial filter will outperform a domain-tuned model bolted onto weak chunking, by a wide margin. The ordering of effort implied by that is unglamorous and reliable: fix the boundaries, add the filter, close the lexical gap, and only then ask whether the vectors are the constraint.

## Related

- Up to the section overview: [Spatial LLM Architecture and Core Concepts](/spatial-llm-architecture-core-concepts/)
- Technique: [Benchmarking Spatial Embedding Models for Vector GIS](/spatial-llm-architecture-core-concepts/spatial-embedding-models/benchmarking-spatial-embedding-models-for-vector-gis/)
- Technique: [Choosing Vector Dimensionality for Spatial Retrieval](/spatial-llm-architecture-core-concepts/spatial-embedding-models/choosing-vector-dimensionality-for-spatial-retrieval/)
- Related topic: [Spatial Vector Store Selection](/geospatial-rag-pipelines/spatial-vector-store-selection/)
- Related topic: [Hybrid Spatial and Keyword Retrieval](/geospatial-rag-pipelines/hybrid-spatial-keyword-retrieval/)
- Peer topic: [Geometry Tokenization Strategies](/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/)
