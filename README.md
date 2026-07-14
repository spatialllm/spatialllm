<p align="center">
  <a href="https://www.spatialllm.org">
    <img src="https://www.spatialllm.org/assets/img/og-image.png" alt="Spatial LLM &amp; AI Agent Workflows" width="100%">
  </a>
</p>

<h1 align="center">Spatial LLM &amp; AI Agent Workflows</h1>

<p align="center">
  <strong>A production-focused resource for building, validating, and deploying spatial-aware AI systems.</strong><br>
  <a href="https://www.spatialllm.org">www.spatialllm.org</a>
</p>

---

Large language models are fluent with text but reckless with geography. A model will happily
return a plausible-looking polygon in the wrong hemisphere, invent coordinates with impossible
precision, mismatch a coordinate reference system, or emit SQL that quietly triggers a full-table
scan. None of those failures show up in a text benchmark — they show up in production, on a map,
in front of a user.

**Spatial LLM &amp; AI Agent Workflows** documents the engineering patterns that make spatial AI
deployable: the validation gates, deterministic fallbacks, and observability that keep geographic
reasoning correct at its edges. Every guide is grounded in runnable Python and PostGIS — GeoPandas,
Shapely, PyProj, and rasterio at the data layer; structured prompting, schema-grounded tool
dispatch, and observable agent routing at the inference layer.

The site is written for AI/ML engineers, spatial data scientists, Python GIS developers, and
platform teams who are comfortable with both transformer pipelines and spatial joins.

## What the site covers

The material is organized around three areas, each with in-depth guides:

### 🧭 [Spatial LLM Architecture &amp; Core Concepts](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/)
Validation-first ingestion, geometry tokenization, spatial embeddings, deterministic fallback
routing, and — new in this release — a full [evaluation &amp; benchmarking](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/)
area covering spatial IoU measurement, coordinate-hallucination detection, and regression test
harnesses for spatial agents.

### 🔀 [Geospatial Prompt Engineering &amp; Tool Routing](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/)
Prompt-to-spatial-SQL generation, async vs. sync geoprocessing, function-calling schemas for
spatial tools, multi-step agent orchestration with checkpoints, error mapping, and routing across
GeoPandas and PostGIS backends.

### 🔎 [Geospatial RAG Pipelines](https://www.spatialllm.org/geospatial-rag-pipelines/)
Retrieval-augmented generation over geographic corpora: spatial vector store selection,
retrieval-augmented CRS resolution, chunk-boundary strategies that never sever a geometry, and
reranking search results by both semantic relevance and spatial proximity.

## Why it's different

- **Every code sample is production-shaped.** Explicit error handling and a deterministic fallback
  path are non-negotiable — even in the shortest examples.
- **Index-aware by default.** PostGIS examples always show the bounding-box pre-filter (`&&`) before
  the spatial predicate, so queries stay off the sequential-scan path.
- **Correctness is measurable.** Spatial metrics (IoU, distance thresholds, precision bounds) are
  written out with real math, and the evaluation guides turn them into CI gates.
- **Hand-authored diagrams.** Each architecture and data-flow diagram is an original inline SVG that
  adapts to light and dark themes — no stock imagery.

## Explore

- 🌐 **Live site:** [www.spatialllm.org](https://www.spatialllm.org)
- 🚀 **Start here:** [Coordinate Reference System Normalization](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/coordinate-reference-system-normalization/) ·
  [Geometry Tokenization Strategies](https://www.spatialllm.org/spatial-llm-architecture-core-concepts/geometry-tokenization-strategies/) ·
  [Prompt-to-Spatial-SQL Generation](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/)

## Tech

The site is a static build produced with [Eleventy](https://www.11ty.dev/) and served on Cloudflare.
Content is authored in Markdown, diagrams are inline SVG, and spatial math is rendered with KaTeX.

```bash
npm install      # install dependencies
npm run build    # build the static site into _site/
npm start        # serve locally with live reload
```

---

<p align="center"><sub>Built for engineers shipping spatial AI to production. Contributions and corrections welcome.</sub></p>
