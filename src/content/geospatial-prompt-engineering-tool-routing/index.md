---
layout: overview.njk
order: 2
navLabel: Prompt Design
icon: routing
title: Geospatial Prompt Design and Tool Routing
description: How a spatial agent turns a sentence into checked geoprocessing — plan structure, tool schemas, generated SQL, routing between backends, topology, failure handling and budgets.
slug: geospatial-prompt-engineering-tool-routing
type: overview
breadcrumb: Prompt Design and Tool Routing
datePublished: 2025-02-11
dateModified: 2026-08-11
---

# Geospatial Prompt Design and Tool Routing

A language model that can describe a spatial analysis is not the same thing as a system that can perform one. Between the sentence a person types and the answer they read sits a plan that must be structured, tools that must be described precisely enough to call correctly, statements that must be checked before they touch a database, geometry that must be valid, failures that must be classified, and a budget that all of it has to fit inside. This section covers that middle layer — the part that turns a fluent description into a computation somebody can rely on.

The failure this section exists to prevent is not a crash. A spatial agent built without these controls produces answers: confident, well-formatted, in the right units, about the wrong area, measured in a reference system nobody declared, from a step that silently returned nothing. Everything below is a control against one specific way that happens by default.

<figure class="diagram">
<svg viewBox="0 0 860 330" role="img" aria-labelledby="gpe-arch-t gpe-arch-d" xmlns="http://www.w3.org/2000/svg"><title id="gpe-arch-t">From a sentence to a checked result</title><desc id="gpe-arch-d">A request becomes a structured plan, each step is bound to a described tool and routed to a backend, generated statements are validated before execution, results are checked between steps, and failures are classified rather than retried blindly — all inside a stated cost and latency budget.</desc><rect x="0" y="0" width="860" height="330" fill="#ffffff"/><defs><marker id="gpe-arch-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/></marker></defs><text x="430" y="34" fill="#5b6471" font-size="13" text-anchor="middle">Decide — nothing has run yet, and everything is still cheap to reject</text><g fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"><rect x="24" y="52" width="188" height="72" rx="8"/><rect x="232" y="52" width="188" height="72" rx="8"/><rect x="440" y="52" width="188" height="72" rx="8"/><rect x="648" y="52" width="188" height="72" rx="8"/></g><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="118" y="82">structure the plan</text><text x="326" y="82">bind to described tools</text><text x="534" y="82">route to a backend</text><text x="742" y="82">validate the statement</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="118" y="104">named steps, stated units</text><text x="326" y="104">schemas, not prose</text><text x="534" y="104">from estimates</text><text x="742" y="104">parse, do not match</text></g><text x="430" y="176" fill="#5b6471" font-size="13" text-anchor="middle">Execute — every step is checked before the next one builds on it</text><g fill="#e4f5ec" stroke="#12805c" stroke-width="2"><rect x="24" y="194" width="188" height="72" rx="8"/><rect x="232" y="194" width="188" height="72" rx="8"/><rect x="440" y="194" width="188" height="72" rx="8"/><rect x="648" y="194" width="188" height="72" rx="8"/></g><g fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600"><text x="118" y="224">run and checkpoint</text><text x="326" y="224">check the output</text><text x="534" y="224">enforce topology</text><text x="742" y="224">classify any failure</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="118" y="246">resume, do not replay</text><text x="326" y="246">empty is not success</text><text x="534" y="246">invalid input, invalid answer</text><text x="742" y="246">retry only what can recover</text></g><g stroke="#5b6471" stroke-width="2" marker-end="url(#gpe-arch-a)"><line x1="214" y1="88" x2="228" y2="88"/><line x1="422" y1="88" x2="436" y2="88"/><line x1="630" y1="88" x2="644" y2="88"/><line x1="214" y1="230" x2="228" y2="230"/><line x1="422" y1="230" x2="436" y2="230"/><line x1="630" y1="230" x2="644" y2="230"/></g><rect x="24" y="284" width="812" height="36" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="430" y="307" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">All of it inside one budget — the answer that arrives too late is not an answer</text></svg>
<figcaption><b>Two halves, one budget.</b> Everything in the upper row is reversible and costs almost nothing; everything in the lower row spends real time against a database. The whole design is an argument for moving as many decisions as possible into the row where being wrong is free.</figcaption>
</figure>

## Turning a Sentence Into a Plan You Can Inspect

The first decision is whether the model produces an answer or a plan. Asking for an answer directly gets one opaque call whose only outcomes are acceptance and rejection; asking for a plan gets a list of named steps with named inputs, which can be read, corrected, costed, partially cached and partially re-run.

That difference compounds at every later stage in this section. A plan can be checked against an allow list before anything executes. A plan can be priced, so an expensive request gets a confirmation rather than a four-minute silence. A plan that fails at step five can resume at step five. None of that is available for a single generated call, which is why the shape of the output is the most consequential choice in the whole design rather than a matter of style.

Planning is also where ambiguity gets resolved, and where it must be resolved visibly. People ask for things near a place, in the units they happen to think in, over data they assume is current. Each of those is an assumption the plan has to state — a threshold, a unit, a reference system, a snapshot — because an assumption written into the plan can be corrected by a reader and the same assumption applied silently cannot. [LLM-assisted geoprocessing pipelines](/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/) covers the decomposition itself, and [multi-step spatial agent orchestration](/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/) covers what happens once a plan has more steps than one thing can hold in mind.

## Describing Tools So They Are Called Correctly

A tool schema is not documentation. It is the only interface through which the model perceives what the system can do, and every ambiguity in it becomes a class of wrong call that arrives forever.

The leverage is concentrated in a few places. Parameter names are read reliably and descriptions are read sometimes, so a parameter called `distance_metres` prevents an error that three sentences of prose about units will not. Enumerated values remove a whole space of near-miss strings. Bounded numbers make an unreasonable argument impossible to express rather than merely unwise. Required fields stop the model from omitting the thing it was least sure about, which is usually the thing that matters most.

What is left over — the constraints a schema cannot express, like one field being required only when another has a particular value — has to be checked in code, and the check should fail in exactly the same shape as a schema rejection. The model does not benefit from knowing which layer rejected it; it benefits from being told which field to change and what an acceptable value looks like. [Spatial function-calling schemas](/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/) works through the shape of those definitions and the failure messages that go with them.

## Generated Statements Are Untrusted Input

Once a plan reaches the database, a second problem appears: the statement about to run was shaped by a sentence a stranger typed. That makes it untrusted regardless of how carefully the prompt was written, because prompt instructions are a preference expressed to a system whose entire purpose is to be persuaded by text.

The boundary that works sits between generation and execution, and it is structural. Parse the statement rather than matching patterns in it, because anything that can be written two ways will eventually be written the way the pattern misses. Check tables and functions against an allow list, so that anything new is unknown rather than permitted. Reject anything that is not a single read. And run the whole thing on a connection whose privileges make a mistake in the layers above it survivable.

<figure class="diagram">
<svg viewBox="0 32 780 228" role="img" aria-labelledby="gpe-gate-t gpe-gate-d" xmlns="http://www.w3.org/2000/svg"><title id="gpe-gate-t">Where each control sits relative to execution</title><desc id="gpe-gate-d">Plan validation, schema validation and statement parsing all happen before execution and cost nothing to fail; output checks, topology enforcement and error classification happen after execution and have already spent time and money.</desc><rect x="0" y="32" width="780" height="228" fill="#ffffff"/><rect x="30" y="46" width="330" height="170" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="420" y="46" width="330" height="170" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="195" y="76">before execution — free to fail</text><text x="585" y="76">after execution — already paid for</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="195" y="108">the plan is well formed</text><text x="195" y="134">arguments match the schema</text><text x="195" y="160">the statement parses to allowed parts</text><text x="195" y="186">the estimated cost is acceptable</text><text x="585" y="108">the output is not silently empty</text><text x="585" y="134">the geometry is valid</text><text x="585" y="160">the count is plausible</text><text x="585" y="186">the failure has a class</text></g><text x="390" y="242" fill="#1f2937" font-size="13" text-anchor="middle">Every control moved from the right box to the left one is a failure that stops costing anything</text></svg>
<figcaption><b>The asymmetry is the design.</b> A rejection on the left costs a parse; the same rejection on the right costs a query, a wait, and a person's attention. Most of the engineering in this section is an attempt to move checks leftward.</figcaption>
</figure>

The same reasoning applies to expense, not just safety. A syntactically perfect statement with no spatial predicate is a cross join that will run for twenty minutes, and the planner's cost estimate — taken before execution — catches it for the price of an explain. Rejecting on estimated cost with a message about narrowing the area is far better behaviour than a query nobody can cancel. [Prompt-to-spatial-SQL generation](/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/) covers both the safety and the cost checks, including the reference-system agreement that is the most common source of a confidently wrong number.

## Choosing Where the Work Runs

Spatial work can happen in the database or in the process, and neither is the default. The database wins when the data is large and already indexed there, because moving a million rows across a connection costs more than the computation does. The in-process library wins when the data is small, already loaded, or the operation is genuinely awkward to express in SQL.

The crossover between those two is a property of your data and your hardware, not a constant to be borrowed. Measuring it once, on real inputs, is cheaper than the debugging that follows from a threshold copied out of a tutorial — and it needs re-measuring whenever the data grows materially or the deployment changes, because nothing raises an error when a routing threshold goes stale.

The most useful shape in practice is not choosing one backend but splitting the work: a spatial predicate in the database reduces millions of rows to hundreds, and the awkward part of the computation then runs on those hundreds where expressing it is easy. The mistake this avoids is filtering in the process, which transfers everything in order to discard nearly all of it. [GeoPandas and PostGIS tool routing](/geospatial-prompt-engineering-tool-routing/geopandas-postgis-tool-routing/) covers the estimates that drive the decision and the fallback when the chosen backend is unavailable.

## Keeping Geometry Valid on the Way Through

Invalid geometry does not stay invalid at constant severity. A self-intersecting ring that survives one overlay produces several fragments in the next, and by the third operation the original shape is unrecoverable and the area is a number with no meaning. That is why validity is checked at the boundary, where a failure names its producer, rather than downstream where it names an innocent step.

The violations that actually occur are few: rings that cross themselves, rings that do not close, near-zero-width slivers left by imprecise overlay, and gaps where neighbouring polygons were meant to share an edge. The first two are unambiguous errors. The last two are only violations relative to a tolerance somebody chose, which is precisely why that tolerance has to be recorded alongside the data rather than living in a constant somewhere.

Repair is usually the right response and it must never be silent. Closing a ring restores obvious intent; snapping neighbouring boundaries together moves real lines and is a decision. Both are acceptable, and neither is acceptable unrecorded, because the first person to compare a result against its source will otherwise find a discrepancy nobody can account for. [Topology rule enforcement via LLMs](/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/) covers the checks, the repairs, and the tolerance choices that decide which is which.

## Checking Results Between Steps, Not Only at the End

A chain of six operations checked only at the end tells you that something in a chain of six is wrong. The same chain checked after every step names the operation that broke, stops the cost accumulating, and makes the fix local. That difference is diagnostic before it is economic.

Four checks catch nearly everything and all four are cheap. Is the result empty — the most common silent failure, and one that is indistinguishable from a real answer of "none". Is the count plausible against the input count, since a join gone wrong produces millions and a bad filter produces zero, and neither raises an error. Are the geometries valid. Do the reference systems agree. None of these requires knowing what the data means, which is what makes them worth running every time rather than when something already looks wrong.

The emptiness case deserves particular care, because "no hospitals within two kilometres" is a real and useful answer that looks exactly like a projection mismatch. Flagging it with the step that produced it, rather than passing it silently along, is what lets the difference be worked out later. [Validating intermediate geoprocessing outputs](/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/validating-intermediate-geoprocessing-outputs/) covers the checks and the thresholds that keep them from becoming noise.

## Failure Classes, Not Retry Counts

Retrying a failure that cannot recover is the most expensive habit an agent can develop, because it costs three attempts to learn something the first response already said. Four classes are enough to fix that.

<figure class="diagram">
<svg viewBox="0 32 780 224" role="img" aria-labelledby="gpe-fail-t gpe-fail-d" xmlns="http://www.w3.org/2000/svg"><title id="gpe-fail-t">Four failure classes and the response each deserves</title><desc id="gpe-fail-d">Transient failures are worth one jittered retry; capability failures should degrade to another route; input failures should be corrected and re-planned; fatal failures should stop and be reported.</desc><rect x="0" y="32" width="780" height="224" fill="#ffffff"/><g><rect x="30" y="46" width="172" height="150" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="216" y="46" width="172" height="150" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="402" y="46" width="172" height="150" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="588" y="46" width="162" height="150" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/></g><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="116" y="80">transient</text><text x="302" y="80">capability</text><text x="488" y="80">input</text><text x="669" y="80">fatal</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="116" y="112">a timeout or a blip</text><text x="116" y="138">one retry, jittered</text><text x="116" y="170">then treat as fatal</text><text x="302" y="112">this route cannot</text><text x="302" y="138">degrade to another</text><text x="302" y="170">no retry helps</text><text x="488" y="112">the request is wrong</text><text x="488" y="138">correct and re-plan</text><text x="488" y="170">the model can fix it</text><text x="669" y="112">not permitted</text><text x="669" y="138">stop and report</text><text x="669" y="170">retrying is waste</text></g><text x="390" y="238" fill="#1f2937" font-size="13" text-anchor="middle">Only one of the four classes is worth trying again — the other three retry identically forever</text></svg>
<figcaption><b>Three quarters of failures should never be retried.</b> A policy that retries without classifying spends three attempts confirming something the first response already stated plainly, and does it inside a budget that had room for an alternative route.</figcaption>
</figure>

The classification also decides what the reader sees. Raw service text names tables and stack frames, which tells an attacker about your schema and tells an ordinary user nothing at all. A derived message says what happened, whether it is worth trying again, and the one thing the reader can change — three clauses that fit in a sentence, with a correlation identifier for the case where somebody needs the detail. [Error mapping for spatial API calls](/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/) covers the classification, the circuit breaker that keeps a slow dependency from consuming every budget, and the messages themselves.

## Long Work, Short Conversations

Some spatial operations take minutes. A conversation cannot wait minutes, so the work has to leave the request path with a handle that can be checked later — and the decision to send it there has to come from an estimate rather than from discovering it by waiting.

A job that has left the path needs four states rather than two, because queued, running, done and failed each permit a different response from the caller. It needs atomic result writes, so that cancellation or a restart leaves nothing rather than leaving a partial file that parses and reads as complete. And it needs a bounded queue with an admission decision at the door, because an unbounded queue converts a load problem into a latency problem that grows without limit and without any error being raised.

The same reasoning produces rate limiting. A service that accepts everything degrades for every caller simultaneously; one that rejects at a stated limit keeps accepted work fast and tells the rest when to come back. A rejection carrying a retry time is a service rather than a failure — without it, every polite client becomes an impolite one, because immediate retry is the only strategy left. [Async and synchronous geoprocessing workflows](/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/) covers the job lifecycle, the admission decision and the backpressure signals.

## Everything Fits in a Budget

Cost and latency are not a final consideration; they are the constraint that decides how many of the controls above can run. A budget that is only checked at the end is a report. A budget carried through the plan, decremented as steps complete, is a control that can shorten a route, drop an optional refinement, or refuse before spending.

Caching is where most of the recovered budget comes from, and its whole difficulty is knowing when a cached answer is still true. A result keyed on the operation, its inputs and its parameters can be reused safely; the same result keyed on the question that produced it will be served for a question that only looks similar. Reference data that changes weekly and live data that changes constantly need different lifetimes, and treating them the same guarantees one of them is wrong.

Estimating before running is what makes the rest of it work. A plan whose cost can be projected from row counts and past measurements can be confirmed with the reader, routed to a cheaper backend, or split — all before anything runs. [Cost and latency budgets for spatial agents](/geospatial-prompt-engineering-tool-routing/cost-and-latency-budgets-for-spatial-agents/) covers the estimates, the cache keys and the degradation choices.

## Operating This Layer Over Time

The controls in this section share a failure mode: they are all thresholds, allow lists and defaults, and every one of them goes stale silently. A routing threshold measured against last year's data, a tolerance copied from a tutorial, a cache lifetime set when the data updated weekly, an allow list broadened under delivery pressure — none of these raise an error when they stop being right, and all of them change what the system answers.

The maintenance that works is a small set of counters rather than an audit. Rejections by reason. Routing decisions by backend, against whether the chosen one was slower than predicted. Cache hits against staleness complaints. Validation failures by check. Repairs by kind. Each is one number, each is meaningless alone, and together they say which of the assumptions built into the system have drifted away from the data.

The second habit is reviewing the lists as wholes rather than as individual additions. Every entry added to an allow list is individually reasonable, and the aggregate slowly becomes a deny list with extra steps. That shape is only visible when somebody reads the whole thing in one sitting, a few times a year, which no test can substitute for.

## The Standards Worth Holding Everything To

Five properties separate a spatial agent that can be operated from one that merely demonstrates well, and each of them is a decision made early that becomes expensive to add later.

Every answer should be reconstructable. That means the plan, the routing decision, the parameters and the data snapshot are all recorded together, so that a person who disputes a number can be shown how it was produced rather than told it was computed correctly. Reconstruction is also how a regression gets diagnosed, since the only way to tell a data change from a code change is to have both recorded.

Every assumption should be visible. Units, thresholds, reference systems and time windows are supplied by the system far more often than by the reader, and each of them changes the answer. Stating them in the plan and surfacing them in the response costs a clause and removes the most common category of confidently wrong result.

Every failure should have a class before it has a message. The classification decides whether to retry, whether to degrade, and what the reader sees, and deriving all three from one decision keeps them consistent as the system grows.

Every expensive action should be estimable in advance. If a plan's cost can be projected, it can be confirmed, routed, split or refused; if it cannot, the only available control is a timeout, which spends the whole budget before telling you anything.

Every control should be structural rather than instructed. A rule expressed in a schema, a parse tree or a database privilege holds regardless of what text arrives; the same rule expressed in a prompt holds until someone phrases a request differently.

None of these require unusual infrastructure. They require deciding, at the point where each control is written, that its result will be recorded and its assumption stated — which is a habit rather than a component, and considerably easier to establish at the beginning than to retrofit across a system that already works.

## What This Section Assumes

It assumes the representation questions are settled — that the system has a position on how geometry is encoded, how coordinates are described to a model, and what a reference frame means in your data. Those belong to [spatial LLM architecture and core concepts](/spatial-llm-architecture-core-concepts/), and an agent built on unsettled answers there will produce controls that check the wrong properties.

It also assumes the model has the context it needs. The retrieval side — deciding which dataset a question is about, filtering by region before ranking, keeping citations attached to answers — is covered in [geospatial RAG pipelines](/geospatial-rag-pipelines/). This section starts once the context exists and the question is understood, and it ends when a checked result is ready to be described.

## Frequently Asked Questions

<details class="faq-item"><summary><span>How much of this is necessary for a small internal tool?</span></summary><p>The plan structure, the parameterised values and the least-privilege connection, essentially always — those three are cheap and they prevent the failures that are worst to discover later. Circuit breakers, checkpointing and cost estimation earn their place when chains get deep or dependencies get flaky, which for an internal tool may be never. What does not scale down is validation: an internal tool answering the wrong question is exactly as wrong as a public one.</p></details>

<details class="faq-item"><summary><span>Can the model be given the database connection directly?</span></summary><p>Only in a context where every possible statement it could produce is acceptable, which in practice means a read-only replica of public data. The problem is not that the model is careless; it is that the statement was shaped by text from a person whose intentions are unknown, and no amount of instruction converts that into a boundary. Structural validation in front of a scoped connection gives the same capability without the property that makes it dangerous.</p></details>

<details class="faq-item"><summary><span>Where should the reference system be decided?</span></summary><p>In the plan, explicitly, and checked at every step boundary. A system inherited from whichever dataset was loaded first is correct until the second dataset arrives, at which point distances become meaningless without anything changing in the query. Making the system a stated part of each step means a mismatch is a rejection rather than a number.</p></details>

<details class="faq-item"><summary><span>How should this be evaluated?</span></summary><p>As two separate things: whether plans are correct, and whether execution is correct. A labelled set of requests with expected plan structures catches the first, cheaply and without running anything; a smaller set with expected results catches the second. Combining them into one score hides which half moved, which is the only thing the score would have been useful for.</p></details>

<details class="faq-item"><summary><span>Does a smaller model change any of this?</span></summary><p>It raises the value of everything structural and lowers the value of everything conversational. Tighter schemas, more enumerations, shorter plans and more aggressive validation all matter more with a weaker generator, and they cost nothing with a stronger one — which is a reasonable argument for building as though the model were weaker than it is.</p></details>

## Related

- Section: [Spatial LLM Architecture and Core Concepts](/spatial-llm-architecture-core-concepts/) — the representation and evaluation groundwork this layer assumes
- Section: [Geospatial RAG Pipelines](/geospatial-rag-pipelines/) — how the context an agent reasons over gets assembled
- Topic: [LLM-Assisted Geoprocessing Pipelines](/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/)
- Topic: [Spatial Function-Calling Schemas](/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/)
- Topic: [Prompt-to-Spatial-SQL Generation](/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/)
- Topic: [GeoPandas and PostGIS Tool Routing](/geospatial-prompt-engineering-tool-routing/geopandas-postgis-tool-routing/)
- Topic: [Topology Rule Enforcement via LLMs](/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/)
- Topic: [Multi-Step Spatial Agent Orchestration](/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/)
- Topic: [Error Mapping for Spatial API Calls](/geospatial-prompt-engineering-tool-routing/error-mapping-for-spatial-api-calls/)
- Topic: [Async vs Sync Geoprocessing Workflows](/geospatial-prompt-engineering-tool-routing/async-vs-sync-geoprocessing-workflows/)
- Topic: [Cost and Latency Budgets for Spatial Agents](/geospatial-prompt-engineering-tool-routing/cost-and-latency-budgets-for-spatial-agents/)
