---
title: Hybrid Spatial and Keyword Retrieval
description: Combine lexical matching, dense retrieval and a geographic filter into one ranking, so exact place names and reference codes survive alongside semantic similarity.
slug: hybrid-spatial-keyword-retrieval
type: topic
breadcrumb: Hybrid Spatial and Keyword Retrieval
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Hybrid Spatial and Keyword Retrieval

Dense retrieval is very good at meaning and very bad at names. Ask it for "Kirkby Lonsdale" and it will cheerfully return documents about Kirkby, Lonsdale, and a dozen other places whose embeddings sit nearby, because to an embedding model a rare toponym looks like a slightly unusual arrangement of familiar subwords. Ask it for parcel reference `NY6512-04A` and it will do worse still. Hybrid retrieval fixes this by keeping a lexical index alongside the dense one and fusing their rankings, with a geographic filter constraining both.

This topic sits in [geospatial RAG pipelines](/geospatial-rag-pipelines/) and complements [spatial context retrieval and reranking](/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/), which handles the geometry side of ranking. Here the concern is the third signal — exact token overlap — and how to combine three score families that live on incompatible scales without one of them quietly dominating the other two.

<figure class="diagram">
<svg viewBox="16 38 768 227" role="img" aria-labelledby="hsk-three-t hsk-three-d" xmlns="http://www.w3.org/2000/svg"><title id="hsk-three-t">Three retrieval signals and what each one is good at</title><desc id="hsk-three-d">Lexical matching recovers rare names and codes, dense retrieval recovers paraphrase and synonymy, and the geographic filter removes everything outside the region. Each covers a failure of the other two.</desc><rect x="16" y="38" width="768" height="227" fill="#ffffff"/><rect x="30" y="52" width="230" height="128" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="285" y="52" width="230" height="128" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="540" y="52" width="230" height="128" rx="8" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><g fill="#1f2937" font-size="14" text-anchor="middle" font-weight="600"><text x="145" y="84">lexical</text><text x="400" y="84">dense</text><text x="655" y="84">geographic</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="145" y="114">finds: rare toponyms,</text><text x="145" y="134">codes, scheme numbers</text><text x="145" y="162">misses: paraphrase</text><text x="400" y="114">finds: synonyms,</text><text x="400" y="134">descriptions, intent</text><text x="400" y="162">misses: exact strings</text><text x="655" y="114">removes: everything</text><text x="655" y="134">outside the region</text><text x="655" y="162">decides nothing else</text></g><text x="400" y="222" fill="#1f2937" font-size="13" text-anchor="middle">Each signal covers the others&#8217; blind spot — none of them is a ranking on its own</text><text x="400" y="248" fill="#5b6471" font-size="12" text-anchor="middle">The filter constrains both rankings; only the first two are fused</text></svg>
<figcaption><b>Two rankings and one constraint.</b> It is tempting to treat geography as a third score and blend all three. Doing so lets a strong lexical match drag in a document from the wrong region, which is the exact failure the filter exists to prevent.</figcaption>
</figure>

## Foundational Principles

**Scores from different families are not comparable.** A lexical relevance score is unbounded and corpus-dependent; a cosine similarity sits in a narrow band near the top of its range. Adding them directly means whichever has the larger variance decides the ranking. Either normalise both to a common scale per query, or fuse on rank rather than on score.

**Geography constrains; it does not vote.** A candidate outside the region of interest is excluded, not penalised. Blending distance into the same weighted sum as the text scores means a sufficiently strong text match can always buy its way back in, and the resulting answers are the hardest kind to debug because every component behaved as designed.

**Rare terms are the whole point.** If a hybrid system's lexical half never changes the top result, it is not earning its cost — and the usual cause is a tokenizer that splits reference codes, or a stopword list that eats the distinguishing part of a place name. Verify on the hard cases, not the average ones.

## Step-by-Step Implementation Pipeline

### 1. Extract the terms worth matching exactly

Not every token deserves lexical weight. Place names, reference codes, scheme identifiers and unit designations do; ordinary vocabulary does not, because the dense side handles it better. A short extraction pass that identifies the exact-match candidates lets the lexical query be precise rather than broad.

```python
import re
from dataclasses import dataclass

# Codes: letters and digits joined by separators, at least one of each.
_CODE = re.compile(r"\b(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\b")
_TITLECASE_RUN = re.compile(r"\b(?:[A-Z][a-zÀ-ſ]+)(?:[ -](?:[A-Z][a-zÀ-ſ]+|of|the|upon|on))*\b")

@dataclass(frozen=True)
class ExactTerms:
    codes: tuple[str, ...]
    names: tuple[str, ...]

def extract_exact_terms(query: str, stopnames: frozenset[str] = frozenset()) -> ExactTerms:
    """Pull out the tokens that must match literally; everything else is the dense half's job."""
    codes = tuple(dict.fromkeys(m.group(0) for m in _CODE.finditer(query)))
    names = tuple(
        n for n in dict.fromkeys(m.group(0).strip() for m in _TITLECASE_RUN.finditer(query))
        if n.lower() not in stopnames and len(n) > 2
    )
    return ExactTerms(codes, names)
```

The `stopnames` set is worth maintaining by hand. Sentence-initial capitalisation makes "The", "Where" and "Flood" look like proper nouns, and a lexical query that insists on matching them literally will return nothing at all — a failure mode that presents as "hybrid retrieval made results worse", which is technically accurate and entirely fixable.

### 2. Run both retrievals against the same filtered population

Both halves must see the same candidate universe, or their ranks cannot be fused. That means the geographic filter is applied to both, not bolted onto one.

```sql
-- One filtered population; two orderings over it.
WITH in_region AS (
    SELECT id, body, tsv, embedding
    FROM   spatial_chunks
    WHERE  geom && ST_MakeEnvelope(:w, :s, :e, :n, 4326)
      AND  ST_Intersects(geom, ST_MakeEnvelope(:w, :s, :e, :n, 4326))
)
SELECT id,
       ts_rank_cd(tsv, plainto_tsquery('english', :terms)) AS lexical,
       1 - (embedding <=> :qvec)                           AS dense
FROM   in_region
WHERE  tsv @@ plainto_tsquery('english', :terms)
    OR embedding <=> :qvec < 0.65
LIMIT  400;
```

The `OR` is deliberate: a candidate that only one half likes must still reach the fusion stage, because the entire value of hybrid retrieval lies in the candidates one half would have missed. An `AND` here produces an intersection that is usually smaller than either result set and reliably worse than both.

### 3. Fuse on rank, not on raw score

Reciprocal rank fusion sidesteps the normalisation problem entirely: it discards the scores and combines positions, which are comparable by construction. It is unglamorous, it has one parameter, and it is very hard to beat without a labelled set large enough to tune against.

```python
def rrf(rank_lists: list[list[str]], k: int = 60, weights: list[float] | None = None) -> dict:
    """Reciprocal rank fusion over any number of rankings of document ids."""
    if not rank_lists:
        return {}
    if weights is None:
        weights = [1.0] * len(rank_lists)
    if len(weights) != len(rank_lists):
        raise ValueError("weights must match the number of rank lists")
    scores: dict[str, float] = {}
    for ranking, weight in zip(rank_lists, weights):
        for position, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + position)
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))
```

The constant dampens the advantage of rank one over rank two; a smaller value makes the fusion more decisive and more brittle. Sixty is the conventional default and behaves sensibly across corpus sizes, but it is a knob worth understanding before it is a knob worth turning. Tuning it and the weights against real query mixes is covered in [tuning fusion weights for toponym-heavy queries](/geospatial-rag-pipelines/hybrid-spatial-keyword-retrieval/tuning-fusion-weights-for-toponym-heavy-queries/).

### 4. Boost documents that match an exact code

Rank fusion treats a lexical hit on a rare code the same as a lexical hit on a common word, because both are just positions in a list. When a query contains an unambiguous identifier, a document containing that identifier is almost certainly the right answer, and a modest explicit boost captures that.

```python
def apply_exact_boost(fused: dict, doc_terms: dict, terms: ExactTerms,
                      boost: float = 0.5) -> dict:
    """Lift documents containing an exact code; names get a smaller lift than codes."""
    if not terms.codes and not terms.names:
        return fused
    out = dict(fused)
    for doc_id, base in fused.items():
        present = doc_terms.get(doc_id, frozenset())
        code_hits = sum(1 for c in terms.codes if c in present)
        name_hits = sum(1 for n in terms.names if n in present)
        if code_hits or name_hits:
            out[doc_id] = base * (1.0 + boost * code_hits + (boost / 3) * name_hits)
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))
```

Multiplying rather than adding keeps the boost proportional to the fused score, so a document that both halves already liked benefits more than one scraped from the bottom of a single list. That is the behaviour you want: an exact code in an otherwise irrelevant document is a coincidence more often than a signal.

### 5. Measure the two halves separately before measuring the whole

A hybrid system that performs well overall can be hiding a broken half. Evaluate lexical-only, dense-only and fused recall on the same query set; if fused barely beats the better single half, one of them is contributing nothing and the extra infrastructure is unearned.

```python
def compare_halves(queries, truth, lexical_search, dense_search, fuse_fn, k=10) -> dict:
    """Recall at k for each half and for the fusion — the diagnostic that matters."""
    def recall(search):
        hit = tot = 0
        for q in queries:
            want = truth.get(q, set())
            if not want:
                continue
            got = set(search(q)[:k])
            hit += len(want & got)
            tot += len(want)
        return round(hit / tot, 4) if tot else 0.0
    return {
        "lexical": recall(lexical_search),
        "dense": recall(dense_search),
        "fused": recall(lambda q: list(fuse_fn([lexical_search(q), dense_search(q)]))),
    }
```

### 6. Separate the query mix before drawing conclusions

An aggregate recall number over a mixed query set hides the effect this whole topic exists to produce. Split the evaluation set into name-and-code queries and descriptive queries, and report both. The expected pattern is that lexical wins the first group decisively, dense wins the second, and fusion is close to the winner in each — which is exactly the outcome that justifies running both.

If instead fusion sits below the better half on both groups, the fusion weights are wrong. If it sits below on only one group, the query classifier feeding those weights is the thing to fix. Distinguishing those two diagnoses is impossible from an aggregate number, which is why the split is not optional.

<figure class="diagram">
<svg viewBox="46 7 628 242" role="img" aria-labelledby="hsk-mix-t hsk-mix-d" xmlns="http://www.w3.org/2000/svg"><title id="hsk-mix-t">Recall by query type for each retrieval half and for the fusion</title><desc id="hsk-mix-d">Bars comparing lexical, dense and fused recall on name-and-code queries against descriptive queries. Each half wins one group decisively while the fusion stays close to the winner in both.</desc><rect x="46" y="7" width="628" height="242" fill="#ffffff"/><text x="390" y="32" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">Recall at ten, split by what the query is made of</text><g fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"><rect x="110" y="62" width="54" height="112" rx="4"/><rect x="440" y="134" width="54" height="40" rx="4"/></g><g fill="#e4f5ec" stroke="#12805c" stroke-width="2"><rect x="176" y="118" width="54" height="56" rx="4"/><rect x="506" y="66" width="54" height="108" rx="4"/></g><g fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"><rect x="242" y="68" width="54" height="106" rx="4"/><rect x="572" y="72" width="54" height="102" rx="4"/></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="203" y="196">names and codes</text><text x="533" y="196">descriptive questions</text></g><g fill="#1f2937" font-size="12"><text x="60" y="232">lexical</text><text x="220" y="232">dense</text><text x="360" y="232">fused</text></g><text x="600" y="232" fill="#5b6471" font-size="12" text-anchor="middle">taller is better</text></svg>
<figcaption><b>The shape that justifies the cost.</b> Fusion is not expected to beat both halves on either group — it is expected to stay near the winner on both, so one system serves a query mix that neither half handles alone.</figcaption>
</figure>

### 7. Classify the query, then choose the weights

A single fusion weight serves a mixed query set badly. A question that is essentially an identifier lookup wants the lexical half to dominate; a question phrased entirely in ordinary language wants the dense half to. A cheap classifier over the extracted terms gets most of this right without a model call.

```python
def classify_query(query: str, terms: ExactTerms) -> str:
    """Bucket a query by what it is mostly made of; drives the fusion weights."""
    tokens = [t for t in re.findall(r"[A-Za-z0-9-]+", query) if len(t) > 1]
    if not tokens:
        return "descriptive"                       # nothing to match on literally
    exact_tokens = sum(len(t.split()) for t in terms.names) + len(terms.codes)
    share = exact_tokens / len(tokens)
    if terms.codes and share > 0.4:
        return "identifier"                        # a lookup wearing a question's clothes
    if share > 0.25:
        return "named-place"
    return "descriptive"

FUSION_WEIGHTS = {                                 # (lexical, dense)
    "identifier":  (2.0, 0.6),
    "named-place": (1.2, 1.0),
    "descriptive": (0.6, 1.6),
}

def weights_for(query: str, terms: ExactTerms) -> list[float]:
    return list(FUSION_WEIGHTS.get(classify_query(query, terms), (1.0, 1.0)))
```

Three buckets is deliberately few. Every additional bucket needs its own evaluation slice to justify its weights, and a classifier with eight categories and forty labelled queries is fitting noise. Start with these three, watch which one accumulates complaints, and split that one when you have the evidence to.

Note that even the identifier bucket keeps a non-trivial dense weight. Setting it to zero would turn an identifier query into a pure lookup, which fails badly when the identifier is slightly wrong — a transposed digit, an old scheme reference — and the dense half is the only thing that can recover the intended document from the surrounding words.

### 8. Log which half supplied each result

When a result is wrong, the first useful question is which signal produced it. Record, for every returned document, its position in each source ranking and the weight applied. This costs a few bytes per result and turns "hybrid retrieval returned something odd" from an unfalsifiable complaint into a specific, fixable observation about one half.

```python
def explain(doc_id: str, rank_lists: list[list[str]], names: list[str]) -> dict:
    """Per-document provenance across the fused rankings."""
    out = {"doc_id": doc_id, "positions": {}}
    for ranking, name in zip(rank_lists, names):
        out["positions"][name] = ranking.index(doc_id) + 1 if doc_id in ranking else None
    return out
```

## Failure Modes & Root Causes

**The tokenizer eats the identifier.** A parcel code is split into three tokens, none of which is distinctive, so lexical search returns everything and nothing. Root cause: a general-purpose text analyser applied to structured strings. Mitigation: index codes in a separate keyword field with no stemming, and match them exactly.

**Fusion dominated by one half.** Results are indistinguishable from dense-only. Root cause: raw score addition with incompatible scales, or a weight set once and never revisited. Mitigation: rank fusion as in step 3, and the split evaluation in step 6.

**Region leakage.** A strongly matching document from outside the area appears in results. Root cause: geography treated as a score rather than a filter, or the filter applied to only one of the two retrievals. Mitigation: the shared filtered population in step 2.

**Empty results on rare names.** The one query type hybrid retrieval exists to serve returns nothing, because the lexical query demanded that every extracted term match. Root cause: conjunctive lexical matching over noisy term extraction. Mitigation: match terms disjunctively and let the boost in step 4 express preference rather than requirement.

## Production Validation Protocols

1. **Rare-term regression set.** Maintain a fixed set of queries containing codes and unusual place names, and assert each returns its known document in the top three.
2. **Half-contribution gate.** Assert fused recall exceeds the better single half on the mixed set; when it does not, the fusion is decoration.
3. **Filter parity assertion.** Assert both halves receive the same filtered population; a divergence here is invisible in output and fatal to fusion.
4. **Analyser test.** Assert that a representative reference code survives tokenisation as a single term; this test catches an analyser change months before a user does.
5. **Boost bound.** Assert the exact-match boost cannot promote a document that failed the geographic filter — the boost operates within the population, never around it.
6. **Latency budget.** Track combined latency separately from each half; hybrid retrieval doubles the query fan-out and the tail is where that shows.

<figure class="diagram">
<svg viewBox="26 36 708 178" role="img" aria-labelledby="hsk-rrf-t hsk-rrf-d" xmlns="http://www.w3.org/2000/svg"><title id="hsk-rrf-t">Reciprocal rank fusion over two rankings</title><desc id="hsk-rrf-d">Two ranked lists are combined by summing reciprocal ranks, so a document ranked moderately by both halves outranks one ranked first by a single half.</desc><rect x="26" y="36" width="708" height="178" fill="#ffffff"/><rect x="40" y="50" width="180" height="150" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><text x="130" y="76" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">lexical ranking</text><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="130" y="104">1. doc-A</text><text x="130" y="128">2. doc-C</text><text x="130" y="152">3. doc-D</text><text x="130" y="180">doc-B absent</text></g><rect x="250" y="50" width="180" height="150" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="340" y="76" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">dense ranking</text><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="340" y="104">1. doc-B</text><text x="340" y="128">2. doc-C</text><text x="340" y="152">3. doc-A</text><text x="340" y="180">doc-D absent</text></g><rect x="500" y="50" width="220" height="150" rx="8" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><text x="610" y="76" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">fused</text><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="610" y="104">1. doc-C — liked by both</text><text x="610" y="128">2. doc-A — first, then third</text><text x="610" y="152">3. doc-B — first, then absent</text><text x="610" y="180">4. doc-D — third, then absent</text></g></svg>
<figcaption><b>Agreement beats enthusiasm.</b> Doc-C tops neither list and wins the fusion, because two independent signals both put it near the front. That is the property worth having when one signal is systematically blind to names and the other to meaning.</figcaption>
</figure>

The rare-term regression set is the gate to build first and the one most often skipped, because it requires someone to sit down and write out thirty queries that are hard for the reasons this topic cares about. That afternoon of work pays for itself the first time an index configuration change silently breaks code matching, which it will, because analyser settings are exactly the kind of configuration that gets adjusted for one reason and breaks something unrelated.

The analyser test deserves to run on every build rather than nightly. It is a single assertion over a single string, it takes milliseconds, and it catches the highest-impact silent failure in this whole pipeline: the moment a reference code stops being one token, every identifier query in the system degrades to fuzzy matching, and nothing else in the test suite notices.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Is rank fusion always better than tuned score fusion?</span></summary><p>Not always, but it is better than untuned score fusion by a wide margin, and untuned is what most systems ship. Score fusion can win when you have a labelled set large enough to fit normalisation parameters per query type and the discipline to refit them when the corpus changes. If either condition is missing — and usually one is — rank fusion gives most of the benefit with none of the maintenance.</p></details>

<details class="faq-item"><summary><span>Should the lexical index cover the same chunks as the dense one?</span></summary><p>Yes, and this matters more than it sounds. If the two indexes hold different units — sentences on one side, feature groups on the other — their ranks refer to different objects and fusion becomes meaningless. Build both from the same chunking pass, with the same identifiers, so a position in one list and a position in the other are statements about the same thing.</p></details>

<details class="faq-item"><summary><span>How do I handle multilingual or historic place names?</span></summary><p>With an alias table, applied at query expansion time rather than at index time. Expanding the query to include known aliases keeps the index simple and makes the alias set inspectable and correctable; baking aliases into the index means every correction requires a rebuild. For historic names specifically, record the period each name was current, since a document from 1950 and one from last year may legitimately use different names for the same place.</p></details>

<details class="faq-item"><summary><span>Does the geographic filter make lexical search redundant for place names?</span></summary><p>No, because the filter answers "is this document about somewhere in the region" while the name answers "is this document about this specific place". A regional filter around a town still admits hundreds of documents about neighbouring villages, and the toponym is what separates them. The two work together: the filter sets the population, the name ranks within it.</p></details>

## Related

- Up to the section overview: [Geospatial RAG Pipelines](/geospatial-rag-pipelines/)
- Technique: [Fusing Keyword and Vector Scores for Place Queries](/geospatial-rag-pipelines/hybrid-spatial-keyword-retrieval/fusing-bm25-and-vector-scores-for-place-queries/)
- Technique: [Tuning Fusion Weights for Toponym-Heavy Queries](/geospatial-rag-pipelines/hybrid-spatial-keyword-retrieval/tuning-fusion-weights-for-toponym-heavy-queries/)
- Peer topic: [Spatial Context Retrieval and Reranking](/geospatial-rag-pipelines/spatial-context-retrieval-and-reranking/)
- Peer topic: [Spatial Vector Store Selection](/geospatial-rag-pipelines/spatial-vector-store-selection/)
- Concept: [Geocoding and Place-Name Resolution](/spatial-llm-architecture-core-concepts/geocoding-and-place-name-resolution/)
