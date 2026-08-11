---
title: Spatial Function-Calling Schemas
description: Design tool schemas a model can fill correctly — typed geometry, explicit units, required frames, closed enumerations, and validation that rejects rather than repairs.
slug: spatial-function-calling-schemas
type: topic
breadcrumb: Function-Calling Schemas
datePublished: 2025-04-01
dateModified: 2026-08-11
---

# Spatial Function-Calling Schemas

A tool schema is a prompt. Every field name, type and description is an instruction the model reads more carefully than anything in the system message, and a schema that admits ambiguity will receive ambiguous arguments — a radius with no unit, a coordinate pair with no frame, a geometry as a string of unknown dialect. This topic is about designing schemas where the wrong call is difficult to express.

It belongs to [geospatial prompt engineering and tool routing](/geospatial-prompt-engineering-tool-routing/) and is the interface through which everything else in the section is invoked. Its output feeds [prompt-to-spatial-SQL generation](/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/) and the routing decisions in [GeoPandas and PostGIS tool routing](/geospatial-prompt-engineering-tool-routing/geopandas-postgis-tool-routing/).

<figure class="diagram">
<svg viewBox="16 16 748 230" role="img" aria-labelledby="sfc-amb-t sfc-amb-d" xmlns="http://www.w3.org/2000/svg"><title id="sfc-amb-t">An ambiguous parameter and the four things it could mean</title><desc id="sfc-amb-d">A radius field with no unit is filled with metres, kilometres, feet or degrees depending on the phrasing of the question, and all four are accepted by a schema that only requires a number.</desc><rect x="16" y="16" width="748" height="230" fill="#ffffff"/><rect x="270" y="30" width="240" height="52" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="390" y="62" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">radius: number</text><rect x="30" y="128" width="170" height="66" rx="8" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="216" y="128" width="170" height="66" rx="8" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="402" y="128" width="170" height="66" rx="8" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><rect x="588" y="128" width="162" height="66" rx="8" fill="#eef2f7" stroke="#5b6471" stroke-width="2"/><g fill="#1f2937" font-size="13" text-anchor="middle"><text x="115" y="156">500 metres</text><text x="301" y="156">500 kilometres</text><text x="487" y="156">500 feet</text><text x="669" y="156">500 degrees</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="115" y="178">intended</text><text x="301" y="178">a thousand times too far</text><text x="487" y="178">a third of the way</text><text x="669" y="178">meaningless</text></g><text x="390" y="228" fill="#1f2937" font-size="13" text-anchor="middle">The schema accepts all four; only the field name can rule three of them out</text></svg>
<figcaption><b>Naming the unit is the whole fix.</b> A field called <code>radius_m</code> receives metres far more reliably than one called <code>radius</code> with a description mentioning metres, because the name is read every time and the description is not.</figcaption>
</figure>

## Foundational Principles

**Units live in the field name.** A parameter called `radius_m` is unambiguous at every reading; one called `radius` with a description is unambiguous only when the description is attended to. This single convention removes most unit errors.

**Geometry is a typed object, never a string.** A string parameter accepts any dialect and any frame, and validating it means parsing it. A structured geometry with a required frame field makes the wrong call impossible to express rather than merely detectable.

**Enumerations are closed and short.** A predicate parameter that accepts free text will receive plausible inventions. Closed enumerations of four or five values are filled correctly almost always, and an invalid value is rejected rather than passed through.

## Step-by-Step Implementation Pipeline

### 1. Type geometry properly

The geometry parameter is where most schema design pays off. Requiring a frame alongside the coordinates prevents the single most damaging class of tool-call error.

```python
GEOMETRY_SCHEMA = {
    "type": "object",
    "required": ["type", "coordinates", "epsg"],
    "additionalProperties": False,
    "properties": {
        "type": {"enum": ["Point", "LineString", "Polygon", "MultiPolygon"]},
        "coordinates": {"type": "array"},
        "epsg": {
            "type": "integer",
            "description": "Reference frame of the coordinates. Use 4326 unless the "
                           "source stated another. Never guess.",
        },
    },
}
```

Making the frame required rather than defaulted is deliberate. A default of 4326 is correct most of the time and silently wrong the rest, and the cases where it is wrong are exactly the ones where the model had information it did not use.

### 2. Close every enumeration

Predicates, aggregates and units are all closed sets, and expressing them as enumerations means an invalid value fails at the boundary rather than deep inside a query builder.

```python
TOOL_SCHEMA = {
    "name": "spatial_query",
    "description": "Find features in a region. Returns at most `limit` features.",
    "parameters": {
        "type": "object",
        "required": ["layer", "region", "predicate"],
        "additionalProperties": False,
        "properties": {
            "layer": {"enum": ["parcels", "buildings", "roads", "zoning"]},
            "region": GEOMETRY_SCHEMA,
            "predicate": {"enum": ["intersects", "within", "contains"]},
            "radius_m": {
                "type": "number", "minimum": 1, "maximum": 50_000,
                "description": "Only for predicate=dwithin. Distance in metres.",
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
        },
    },
}
```

Setting `additionalProperties` to false is worth the strictness. A model that invents a plausible extra parameter — `buffer`, `crs`, `simplify` — is telling you about a capability it expected, and rejecting the call surfaces that as a design question rather than silently ignoring the field.

### 3. Validate and reject, never repair

Validation happens once, at the boundary, and produces either a well-formed call or a rejection naming the field. Repairing a call produces something nobody wrote.

```python
import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("tool_validation")


class ArgumentRejected(ValueError):
    """The call cannot be executed as written; the field is named."""


@dataclass(frozen=True)
class Validated:
    tool: str
    arguments: dict
    notes: tuple[str, ...]


def validate_call(tool: str, arguments: dict, schema: dict) -> Validated:
    """Check against the schema and the semantic rules. Never mutates the arguments."""
    errors = list(schema_errors(arguments, schema["parameters"]))
    if errors:
        raise ArgumentRejected("; ".join(errors))

    notes = []
    if arguments.get("predicate") == "dwithin" and "radius_m" not in arguments:
        raise ArgumentRejected("radius_m is required when predicate is 'dwithin'")
    if arguments.get("predicate") != "dwithin" and "radius_m" in arguments:
        notes.append("radius_m ignored: it applies only to 'dwithin'")

    epsg = arguments["region"]["epsg"]
    if epsg not in PERMITTED_FRAMES:
        raise ArgumentRejected(f"epsg {epsg} is not a frame this system accepts")
    return Validated(tool, arguments, tuple(notes))
```

The conditional-requirement check is the kind of rule a schema cannot express and a validator must. Expressing it as a rejection with a specific message lets the model correct itself in one turn; expressing it as a silent default produces a query about the wrong thing. The schema-level mechanics are developed in [validating tool arguments with GeoJSON Schema](/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/validating-tool-arguments-with-geojson-schema/).

### 4. Bound every numeric parameter

An unbounded numeric field will eventually receive a value that turns a cheap operation into an expensive one. Minimums and maximums in the schema are read by the model and enforced by the validator.

```python
BOUNDS = {
    "radius_m": (1, 50_000),
    "limit": (1, 200),
    "simplify_tolerance_m": (0, 500),
    "buffer_m": (-1_000, 10_000),          # negative buffers are legitimate and bounded
}


def check_bounds(arguments: dict) -> None:
    for field, (lo, hi) in BOUNDS.items():
        if field in arguments and not lo <= arguments[field] <= hi:
            raise ArgumentRejected(
                f"{field}={arguments[field]} outside the permitted range {lo}..{hi}")
```

<figure class="diagram">
<svg viewBox="16 32 748 214" role="img" aria-labelledby="sfc-shape-t sfc-shape-d" xmlns="http://www.w3.org/2000/svg"><title id="sfc-shape-t">Schema choices that make wrong calls hard to express</title><desc id="sfc-shape-d">Named units, required frames, closed enumerations and bounded numbers each remove a class of incorrect call at the point of writing rather than at the point of validation.</desc><rect x="16" y="32" width="748" height="214" fill="#ffffff"/><rect x="30" y="46" width="360" height="86" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="410" y="46" width="340" height="86" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="30" y="146" width="360" height="86" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="410" y="146" width="340" height="86" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="13" font-weight="600"><text x="52" y="76">unit in the name</text><text x="432" y="76">frame required</text><text x="52" y="176">closed enumerations</text><text x="432" y="176">bounded numbers</text></g><g fill="#5b6471" font-size="12"><text x="52" y="102">removes unit confusion</text><text x="52" y="122">read on every call</text><text x="432" y="102">removes silent assumptions</text><text x="432" y="122">the model must decide</text><text x="52" y="202">removes invented values</text><text x="52" y="222">rejection is specific</text><text x="432" y="202">removes runaway cost</text><text x="432" y="222">the model sees the ceiling</text></g></svg>
<figcaption><b>Four conventions, four whole classes of error.</b> Each one is a few characters of schema and removes a failure that would otherwise need detection, classification, a message and a retry.</figcaption>
</figure>

### 5. Write descriptions for the model, not for a document

Tool descriptions are read on every call and are the most reliably attended text in the whole prompt. They should say what the tool does, what it returns, and the one thing most likely to be got wrong — not repeat the parameter list.

```python
GOOD_DESCRIPTION = (
    "Find features of one layer inside a region. Returns up to `limit` features with "
    "their identifiers and attributes, plus a flag when results were truncated. "
    "The region must carry its own reference frame; do not assume one."
)

POOR_DESCRIPTION = (
    "This function performs a spatial query against the configured spatial database "
    "using the parameters supplied, executing the appropriate predicate."
)
```

The second says nothing a reader could act on and consumes the same context. The first tells the model what comes back, which shapes whether it calls the tool again.

### 6. Return results the model can reason about

A tool that returns a bare list teaches the model that the list is everything. Returning the count, the truncation flag and the parameters that produced it is what lets an agent describe its own answer accurately.

```python
def tool_result(rows: list[dict], truncated: bool, considered: int,
                notes: tuple[str, ...]) -> dict:
    return {
        "features": rows,
        "returned": len(rows),
        "considered": considered,
        "truncated": truncated,
        "notes": list(notes),
    }
```

### 7. Keep the tool count small

Every tool in the schema competes for the model's attention, and a spatial agent with fourteen narrowly scoped tools chooses badly among them. Three or four broad tools with well-designed parameters produce better calls than a tool per operation.

```python
TOOLS = ["spatial_query", "spatial_measure", "spatial_relate", "spatial_summarise"]


def audit_tool_surface(schemas: list[dict], max_tools: int = 6) -> list[str]:
    """Flag a surface that has grown past what a model can choose between well."""
    problems = []
    if len(schemas) > max_tools:
        problems.append(f"{len(schemas)} tools exceeds the {max_tools} that choose reliably")
    names = [s["name"] for s in schemas]
    for name in names:
        if sum(1 for other in names if other.startswith(name[:8])) > 2:
            problems.append(f"tools sharing a prefix with {name!r} may be confusable")
    return problems
```

### 8. Version the schema and record which version produced a call

Schemas change, and a call recorded without its schema version cannot be replayed or explained. Stamping the version on every validated call makes a later investigation tractable.

```python
def stamp(validated: Validated, schema_version: str) -> dict:
    return {"tool": validated.tool, "arguments": validated.arguments,
            "schema_version": schema_version, "notes": list(validated.notes)}
```

### 9. Give the model a worked example per tool

Examples shape tool calls more strongly than descriptions do, and one accurate example per tool costs less context than a paragraph of prose. The example should show the reference form of geometry, a realistic layer, and the parameter most often omitted.

```python
def example_call(tool: str) -> dict:
    """One realistic call per tool, generated from the live schema constants."""
    return {
        "spatial_query": {
            "layer": "parcels",
            "region": {"type": "Point", "coordinates": [-3.19, 55.95], "epsg": 4326},
            "predicate": "intersects",
            "limit": 25,
        },
        "spatial_measure": {
            "from_ref": "place:osm-12345",
            "to_ref": "place:osm-67890",
            "rule": "edge",
        },
    }[tool]
```

Generating the examples rather than writing them keeps them correct as the schema changes. An example naming a layer that has since been renamed teaches the model to produce calls that will be rejected, and the rejection rate rises for reasons nobody connects to a migration weeks earlier.

Keep the examples short. A worked example with every optional parameter filled teaches the model that every parameter is expected, which produces over-specified calls and crowds the context with values that will be ignored.

### 10. Decide what the tool does with an ambiguous but valid call

Some calls pass validation and are still not what the user meant — a region that is a point when the question implied an area, a limit of one when the question asked what is nearby. These are not rejections, and treating them as ordinary calls loses information the tool has and the model does not.

```python
def advisory_notes(arguments: dict) -> list[str]:
    """Valid but suspicious calls get a note back, not a rejection."""
    notes = []
    region = arguments.get("region", {})
    if region.get("type") == "Point" and arguments.get("predicate") in {"contains"}:
        notes.append("a point cannot contain features; did you mean 'intersects'?")
    if arguments.get("limit", 0) == 1:
        notes.append("limit is 1; results will not describe the area")
    if region.get("epsg") == 4326 and arguments.get("radius_m", 0) > 20_000:
        notes.append("a radius above 20 km spans several administrative areas")
    return notes
```

Returning these as notes alongside a real result is better than rejecting, because the call may be exactly right and the tool cannot know. The note costs a line, appears in the model's context on the next turn, and is frequently the thing that produces a better second call without a round of rejection.

## Operating This Stage Over Time

Schemas accrete. Each new capability adds a parameter, and after a year a tool that took four arguments takes eleven — at which point the model fills the ones it recognises and omits the rest, producing calls that are valid and not what anyone intended. Reviewing parameter usage periodically, and removing anything that has never been set, keeps the surface honest.

The second drift is in descriptions. They are written once, the behaviour changes, and nothing forces them to agree — a tool described as returning up to fifty features that now returns two hundred will produce agents that under-ask. Generating the numeric parts of a description from the same constants the code uses removes that whole class of divergence.

Rejection rates are the signal worth watching. A field that is frequently rejected is a field the schema describes badly, and the fix is usually in the name or the description rather than in the model. Track rejections per field, not in aggregate, because one bad field can dominate a rate that looks like a general problem.

Finally, resist the temptation to repair. Every rejection is an argument for a small accommodation — defaulting a missing frame, clamping an out-of-range radius, accepting a synonym for an enumeration value — and each one individually is reasonable. Together they produce a boundary that accepts almost anything and quietly transforms it, which is the state the validation existed to prevent.

## Failure Modes & Root Causes

**The unitless number.** A radius of five hundred arrives meaning kilometres. Root cause: a field name without its unit. Mitigation: the unit in the name, and bounds that make an absurd value fail.

**The assumed frame.** Coordinates arrive with no frame and are treated as geographic. Root cause: a defaulted rather than required field. Mitigation: require it; reject when absent.

**The invented enumeration value.** A predicate of "near" or "touching" arrives where the enumeration lists four values. Root cause: a free-text field, or an enumeration the model never saw because it was in the description. Mitigation: real enumerations, rejected on mismatch.

**The tool surface nobody can choose between.** Fourteen tools with overlapping names produce calls to the wrong one. Root cause: a tool per operation. Mitigation: a few broad tools with good parameters, audited for count and confusability.

## Production Validation Protocols

1. **Schema-strictness assertion.** Assert every tool schema sets `additionalProperties` to false and requires its frame field.
2. **Unit-naming test.** Assert every numeric parameter representing a physical quantity has its unit in the name.
3. **Bounds coverage.** Assert every numeric parameter has a minimum and a maximum, and that the bounds appear in the schema the model sees.
4. **Rejection-message test.** Assert every rejection names a field; a message that does not is unactionable for the model.
5. **Per-field rejection rates.** Publish rejections by field and alert on a step change; the fix for a bad field is in the schema.
6. **Description-drift check.** Assert numeric claims in descriptions match the constants in code, generated rather than written.

<figure class="diagram">
<svg viewBox="0 0 760 250" role="img" aria-labelledby="sfc-count-t sfc-count-d" xmlns="http://www.w3.org/2000/svg"><title id="sfc-count-t">Call accuracy against the number of tools offered</title><desc id="sfc-count-d">Accuracy is high with a handful of broad tools and falls as the surface grows, because the model must first choose correctly among increasingly similar options.</desc><rect x="0" y="0" width="760" height="250" fill="#ffffff"/><text x="380" y="34" fill="#1f2937" font-size="13" text-anchor="middle" font-weight="600">Correct tool chosen, by size of the tool surface</text><g fill="#e4f5ec" stroke="#12805c" stroke-width="2"><rect x="90" y="66" width="60" height="108" rx="4"/><rect x="230" y="72" width="60" height="102" rx="4"/></g><g fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"><rect x="370" y="102" width="60" height="72" rx="4"/></g><g fill="#fdeaee" stroke="#b3324f" stroke-width="2"><rect x="510" y="128" width="60" height="46" rx="4"/><rect x="650" y="146" width="60" height="28" rx="4"/></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="120" y="196">3 tools</text><text x="260" y="196">6</text><text x="400" y="196">10</text><text x="540" y="196">14</text><text x="680" y="196">20</text></g><text x="380" y="228" fill="#1f2937" font-size="12.5" text-anchor="middle">Adding a tool for a capability that a parameter could express usually costs accuracy</text></svg>
<figcaption><b>Capability is not the same as surface area.</b> The same functionality expressed as four parameterised tools is used more accurately than as twenty specific ones, and the difference grows with how similar the specific ones are.</figcaption>
</figure>

The per-field rejection rate is the measurement that changes designs rather than merely reporting on them. A field rejected on a fifth of calls is a field whose name or enumeration is wrong, and no amount of prompt engineering elsewhere will fix it — whereas renaming it usually does, immediately and permanently.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should geometry be passed inline or by reference?</span></summary><p>By reference wherever the geometry already exists in the system — a place identifier, a saved region, a previous result — and inline only for geometry the user supplied. References are shorter, unambiguous about frame, and cannot be corrupted in transit through a context window. A schema offering both, with the reference form listed first, gets the reference form most of the time.</p></details>

<details class="faq-item"><summary><span>How much should a description explain?</span></summary><p>What the tool does, what it returns, and the one mistake most likely to be made — three sentences. Longer descriptions are read less carefully and crowd out other tools' descriptions. If a tool genuinely needs a paragraph to explain, that is usually a sign its parameters are doing too much and it should be two tools or one with a closed enumeration.</p></details>

<details class="faq-item"><summary><span>Is it worth returning errors to the model rather than to the user?</span></summary><p>For input errors, yes — that is the whole point of naming the field. A model told that `radius_m` was out of range corrects it in one turn, where a user told the same thing has to guess what the agent asked for. For fatal and capability errors, the model should be told what happened but not encouraged to retry, which is a matter of the message rather than the mechanism.</p></details>

<details class="faq-item"><summary><span>What about optional parameters that change the tool's meaning?</span></summary><p>Avoid them, and split the tool instead. A parameter whose presence changes what the tool does — a mode flag, an alternate geometry field — produces calls where the model sets the flag and forgets the field it now needs. Two tools with clear names are easier to choose between than one tool with a mode, even though the surface is nominally larger.</p></details>

<details class="faq-item"><summary><span>Should the schema expose the layer list, or accept any layer?</span></summary><p>Expose it, as an enumeration, and regenerate it from the live schema. A free-text layer parameter receives plausible names that do not exist, and the resulting error is a capability failure that could have been a schema rejection. Where the list is long enough to be unwieldy, that is a signal to group layers rather than to open the field.</p></details>

## Related

- Up to the section overview: [Geospatial Prompt Engineering and Tool Routing](/geospatial-prompt-engineering-tool-routing/)
- Technique: [Designing JSON Schemas for PostGIS Tool Calls](/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/designing-json-schemas-for-postgis-tool-calls/)
- Technique: [Validating Tool Arguments with GeoJSON Schema](/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/validating-tool-arguments-with-geojson-schema/)
- Peer topic: [Prompt-to-Spatial-SQL Generation](/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/)
- Peer topic: [Multi-Step Spatial Agent Orchestration](/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/)
- Related technique: [Mapping Catalog Fields to Retrieval Filters](/geospatial-rag-pipelines/spatial-metadata-and-catalog-indexing/mapping-catalog-fields-to-retrieval-filters/)
