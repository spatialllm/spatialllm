---
title: Validating Tool Arguments with GeoJSON Schema
description: Use JSON Schema to reject malformed geometry before a tool runs — coordinate order, ring closure, bounded values — plus the semantic checks a schema cannot express.
slug: validating-tool-arguments-with-geojson-schema
type: howto
breadcrumb: Validating with GeoJSON Schema
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Validating Tool Arguments with GeoJSON Schema

A geometry argument is the one place where a tool call can be syntactically perfect and geometrically nonsense. Schema validation catches a surprising amount of that — closure, coordinate bounds, structural nesting — and its limits are as worth knowing as its coverage. This guide covers both, and it is the validation half of [spatial function-calling schemas](/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/).

## When to Use This Approach

Wherever a model supplies geometry rather than referencing it. Where geometry is passed by identifier, the schema has almost nothing to check and the checks belong elsewhere.

| Argument | Schema catches | Schema misses |
|----------|----------------|---------------|
| Inline polygon | Closure, nesting, bounds | Self-intersection |
| Bounding box | Length, ordering, bounds | Whether it is the right box |
| Point | Bounds, dimension | Whether it is on land |
| Layer identifier | Enumeration | Whether the caller may read it |

<figure class="diagram">
<svg viewBox="16 38 748 212" role="img" aria-labelledby="vta-two-t vta-two-d" xmlns="http://www.w3.org/2000/svg"><title id="vta-two-t">What the schema settles and what it cannot</title><desc id="vta-two-d">Structural properties like ring closure and coordinate bounds are decidable from the document alone, while validity, plausibility and permission require knowledge the schema does not have.</desc><rect x="16" y="38" width="748" height="212" fill="#ffffff"/><rect x="30" y="52" width="340" height="160" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="410" y="52" width="340" height="160" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="200" y="84">the schema decides</text><text x="580" y="84">code has to decide</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="200" y="114">rings are closed</text><text x="200" y="140">coordinates are in range</text><text x="200" y="166">nesting depth is right</text><text x="580" y="114">the ring does not cross itself</text><text x="580" y="140">the area is plausible</text><text x="580" y="166">the caller may read the layer</text></g><text x="390" y="234" fill="#1f2937" font-size="13" text-anchor="middle">Both boxes must return failures in the same shape, or the model cannot act on them</text></svg>
<figcaption><b>The division is about information, not effort.</b> Everything on the left is decidable from the document alone; everything on the right needs the world outside it, which is why no schema will ever cover the second column.</figcaption>
</figure>

## Implementation

Start from the GeoJSON structure and tighten it. The published GeoJSON schema is permissive by design; a tool schema should be narrower than the format allows, because a tool usually accepts one geometry type rather than all of them.

```python
POLYGON_ARG = {
    "type": "object",
    "required": ["type", "coordinates"],
    "additionalProperties": False,
    "properties": {
        "type": {"const": "Polygon"},
        "coordinates": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "array",
                "minItems": 4,                       # a closed ring needs four positions
                "items": {
                    "type": "array",
                    "minItems": 2, "maxItems": 3,
                    "prefixItems": [
                        {"type": "number", "minimum": -180, "maximum": 180},
                        {"type": "number", "minimum": -90, "maximum": 90},
                    ],
                },
            },
        },
    },
}
```

`additionalProperties: false` is doing more work than it looks. Without it, a model that invents a `crs` or `properties` field gets silent acceptance and the field is ignored — which means the caller believes it specified something it did not.

Ring closure is the one geometric property a schema cannot express, so it goes in the same validation pass with the same failure shape.

```python
def validate_geometry_arg(value: dict) -> list[str]:
    problems = [format_error(e) for e in VALIDATOR.iter_errors(value)]
    if problems:
        return problems                              # structure first, semantics after

    for i, ring in enumerate(value["coordinates"]):
        if ring[0] != ring[-1]:
            problems.append(f"coordinates[{i}]: ring is not closed")
    return problems


def format_error(err) -> str:
    path = "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in err.absolute_path)
    return f"{path or 'root'}: {err.message}"
```

<figure class="diagram">
<svg viewBox="16 38 764 220" role="img" aria-labelledby="vta-order-t vta-order-d" xmlns="http://www.w3.org/2000/svg"><title id="vta-order-t">Why coordinate bounds catch a reversed pair only sometimes</title><desc id="vta-order-d">A longitude and latitude swapped outside the latitude range is caught by the bounds, while a swap where both values are small produces a valid position in the wrong place.</desc><rect x="16" y="38" width="764" height="220" fill="#ffffff"/><rect x="30" y="52" width="700" height="46" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="50" y="80" fill="#1f2937" font-size="12.5">longitude 151, latitude -33 reversed: -33 is a valid longitude, 151 is not a latitude — caught</text><rect x="30" y="108" width="620" height="46" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="50" y="136" fill="#1f2937" font-size="12.5">longitude 12, latitude 41 reversed: both in range — accepted, and in the wrong country</text><rect x="30" y="164" width="700" height="46" rx="6" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="50" y="192" fill="#1f2937" font-size="12.5">the extra check: does the geometry fall inside the expected working area?</text><text x="390" y="240" fill="#1f2937" font-size="13" text-anchor="middle">Bounds catch the obvious reversals — an extent check catches the rest</text></svg>
<figcaption><b>Bounds checking is necessary and partial.</b> Near the equator and prime meridian a swapped pair is perfectly valid, which is why an expected-extent check earns its place alongside the schema rather than instead of it.</figcaption>
</figure>

## Validation & Testing

Test the schema against malformed inputs that a permissive validator would accept, since those are the ones it exists to catch.

```python
BAD = [
    ({"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1]]]}, "not closed"),
    ({"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]], "crs": "EPSG:4326"},
     "additional properties"),
    ({"type": "Polygon", "coordinates": [[[200, 0], [1, 0], [1, 1], [200, 0]]]}, "out of range"),
    ({"type": "MultiPolygon", "coordinates": []}, "wrong type"),
]


@pytest.mark.parametrize("value,why", BAD)
def test_rejected(value, why):
    assert validate_geometry_arg(value), why
```

Then test the message shape rather than only the outcome. The point of validation here is that the model can correct the call, and a test asserting that each failure names a path and a constraint is what keeps that property from decaying.

## Gotchas & Edge Cases

**Coordinate order.** GeoJSON is longitude first; most people, and a good deal of software, say latitude first. The schema's bounds catch the reversal only when one value exceeds the latitude range, which is why a working-extent check belongs beside it.

**Winding order.** Exterior rings counter-clockwise, interior rings clockwise. No schema expresses this, and getting it wrong produces a polygon whose interior is the outside — an area calculation that returns a number describing the rest of the planet.

**Ring closure with floating point.** A ring closed to within a rounding error is not closed. Comparing positions exactly is correct here; anything approximate turns a clear rejection into a tolerance argument.

**Very large coordinate arrays.** A model that emits a ten-thousand-vertex polygon inline has usually misunderstood the task, and validating it costs real time. A `maxItems` on the ring is a blunt instrument that catches this before the geometry engine does.

**Three-dimensional positions.** Allowing a third element is usually right, and silently discarding it is not. If the tool ignores elevation, say so in the description rather than accepting a value that has no effect.

<figure class="diagram">
<svg viewBox="16 38 748 212" role="img" aria-labelledby="vta-fail-t vta-fail-d" xmlns="http://www.w3.org/2000/svg"><title id="vta-fail-t">What a rejection tells the model</title><desc id="vta-fail-d">A rejection naming the path, the constraint and an acceptable value lets the next attempt succeed, while a generic invalid-arguments message produces the same call again.</desc><rect x="16" y="38" width="748" height="212" fill="#ffffff"/><g><rect x="30" y="52" width="228" height="150" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="276" y="52" width="228" height="150" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="522" y="52" width="228" height="150" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/></g><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="144" y="84">names the path</text><text x="390" y="84">names the constraint</text><text x="636" y="84">names nothing</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="144" y="116">coordinates[0][3]</text><text x="144" y="142">the exact position</text><text x="144" y="168">no searching required</text><text x="390" y="116">maximum is 180</text><text x="390" y="142">what would be accepted</text><text x="390" y="168">the retry is informed</text><text x="636" y="116">invalid arguments</text><text x="636" y="142">no path, no constraint</text><text x="636" y="168">the same call returns</text></g><text x="390" y="234" fill="#1f2937" font-size="13" text-anchor="middle">Most wasted tool-calling budget goes to retries of a call nothing explained</text></svg>
<figcaption><b>The message is part of the interface.</b> A validator that rejects correctly and explains poorly costs more than one that is slightly more permissive and says exactly which field to change.</figcaption>
</figure>

## The Checks the Schema Cannot Make

Three semantic checks belong immediately after validation, and all three should fail in the schema's message shape so the model does not have to distinguish them.

Validity is the first: a ring that crosses itself passes every structural check and produces meaningless area. The geometry library decides this in microseconds, and the result should be a rejection naming the ring rather than a repair, because a model that supplied a self-intersecting polygon usually meant a different polygon rather than a repaired version of that one.

Plausibility is the second. A bounding box covering a hemisphere, supplied for a question about a neighbourhood, is structurally perfect and almost certainly a mistake — often a unit confusion or a reversed pair that happened to stay in range. Comparing the geometry's extent against the working area catches it, and the check costs one comparison.

Permission is the third, and it is the one most often deferred to the database. Deferring it is correct as a boundary and wrong as the only check, because the resulting error arrives as a database failure rather than as an argument rejection — which means the model sees an infrastructure problem where it should see a field it may not use. Checking the layer identifier against the caller's readable set keeps the failure in the right category and keeps the message useful.

## Operating This Step Over Time

Track validation failures per tool and per field. A field that fails regularly is badly named or badly typed, and the fix is in the schema rather than in the prompt; a tool that fails across many fields is usually one whose description does not match what it does.

Review the tightening periodically against what the tools actually accept. A schema that has grown stricter than the implementation rejects legitimate calls, and one that has grown looser passes arguments the implementation then handles badly — both are found by generating a few hundred valid calls and checking that the schema accepts every one.

Keep an eye on where geometry is arriving inline that should be arriving by reference. A model supplying coordinates it copied out of a previous result is doing something the system should be doing for it, and every such call carries a fresh opportunity for a rounding difference or a truncated ring. A rising share of inline geometry usually means a missing identifier — some result the system produced but never gave a name to — and adding the name removes the whole category of malformed argument rather than validating it more carefully.

The working-extent check needs revisiting whenever the deployment covers new ground. Set once for a single region and left alone, it starts rejecting perfectly good geometry the day the system is used somewhere else, and the rejection message will talk about plausibility rather than about a configured bound — which makes it one of the more confusing failures to diagnose from the outside.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should the full GeoJSON specification be supported?</span></summary><p>Rarely. A tool that computes an area needs polygons, and accepting every geometry type means handling cases the implementation was never designed for. Restricting the type with a `const` and saying so in the description produces better calls than accepting everything and failing later, and it removes a class of confusion from the model's side entirely.</p></details>

<details class="faq-item"><summary><span>Where should the reference system be declared?</span></summary><p>As a sibling parameter, not inside the geometry. GeoJSON's own answer is that coordinates are longitude and latitude on a specific datum, and embedding an alternative inside the object contradicts the format while being easy to miss. A separate enumerated field is visible, checkable and impossible to ignore.</p></details>

<details class="faq-item"><summary><span>Is it worth validating geometry the system produced itself?</span></summary><p>At boundaries, yes, and not between internal calls. The expensive part is not the check but the round trip through serialisation, so validating at the point where geometry enters or leaves the system catches what matters without paying for it repeatedly. Internal calls that share a representation can trust it.</p></details>

<details class="faq-item"><summary><span>How should a partially valid argument set be handled?</span></summary><p>Report every problem at once rather than the first. A model told about one field will fix it and be told about the next, and three round trips do the work of one. Collecting all validation errors before returning is a small change to the validator and a large change to how many attempts a call takes.</p></details>

## Related

- Up to the parent topic: [Spatial Function-Calling Schemas](/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/)
- [Designing JSON Schemas for PostGIS Tool Calls](/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/designing-json-schemas-for-postgis-tool-calls/)
- Related topic: [Topology Rule Enforcement via LLMs](/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/)
- Related topic: [Prompt-to-Spatial-SQL Generation](/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/)
