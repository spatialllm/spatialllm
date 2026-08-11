---
title: Planning Buffer, Overlay and Dissolve Chains from a Prompt
description: Turn a sentence into an ordered buffer-overlay-dissolve chain with stated units and reference systems, then check the order and cost before any of it runs.
slug: planning-buffer-overlay-dissolve-chains-from-a-prompt
type: howto
breadcrumb: Planning Chains
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Planning Buffer, Overlay and Dissolve Chains from a Prompt

Most spatial questions people actually ask resolve into the same three operations in some order: grow something by a distance, combine it with something else, then merge the pieces. Getting that order right, with the units and reference systems stated, is nearly the whole job of planning — and it is where a plan quietly becomes an answer to a different question. This guide covers the chain specifically, and it is the concrete half of [LLM-assisted geoprocessing pipelines](/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/).

## When to Use This Approach

Use an explicit chain wherever the request contains a distance, a combination, or an aggregation — which is most of them.

| Request shape | Chain | The trap |
|---------------|-------|----------|
| "Within 500 m of a river" | buffer → intersect | Buffering in degrees |
| "Parcels touching any protected area" | overlay → dissolve | Dissolving before overlay |
| "Total area within 1 km of any school" | buffer → dissolve → intersect | Double-counting overlaps |
| "Land in both zones" | intersect only | Adding a buffer nobody asked for |

<figure class="diagram">
<svg viewBox="16 38 748 212" role="img" aria-labelledby="pbo-two-t pbo-two-d" xmlns="http://www.w3.org/2000/svg"><title id="pbo-two-t">Dissolving before against after the overlay</title><desc id="pbo-two-d">Dissolving buffers before intersecting removes double counting where buffers overlap, while intersecting first and dissolving afterwards produces an inflated area.</desc><rect x="16" y="38" width="748" height="212" fill="#ffffff"/><rect x="30" y="52" width="340" height="160" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="410" y="52" width="340" height="160" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="200" y="84">dissolve, then intersect</text><text x="580" y="84">intersect, then dissolve</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="200" y="114">overlapping buffers merge first</text><text x="200" y="140">each area counted once</text><text x="200" y="166">the total is the real total</text><text x="580" y="114">overlaps intersect separately</text><text x="580" y="140">shared area counted twice</text><text x="580" y="166">the total is inflated, silently</text></g><text x="390" y="234" fill="#1f2937" font-size="13" text-anchor="middle">Both orders run without error and return a number — only one of them is the answer</text></svg>
<figcaption><b>Order is not a performance question here.</b> The wrong order returns a different number with the same units, the same column names and no indication that anything is amiss, which is why the check has to be structural rather than a review of the result.</figcaption>
</figure>

## Implementation

The plan is a list of steps, each naming its operation, its inputs by reference, and every parameter that would otherwise be a default.

```python
@dataclass
class Step:
    op: str                     # buffer | intersect | dissolve | ...
    inputs: list[str]           # names of prior outputs or source layers
    output: str
    params: dict                # distance_metres, crs, dissolve_by, ...


PLAN_SCHEMA = {
    "type": "object",
    "required": ["steps", "assumptions"],
    "properties": {
        "steps": {"type": "array", "minItems": 1, "maxItems": 8, "items": STEP_SCHEMA},
        "assumptions": {
            "type": "object",
            "required": ["crs", "distance_unit"],
            "properties": {
                "crs": {"type": "string", "pattern": "^EPSG:[0-9]{4,6}$"},
                "distance_unit": {"enum": ["metres", "feet"]},
            },
        },
    },
}
```

Forcing `assumptions` to be present and structured is what makes the units visible. A model that must fill in a reference system will pick one; a model that may omit it will, and the buffer will run in whatever the data happened to be in.

The order check runs over the plan before anything executes. It is a handful of rules, each of which encodes a mistake that produces a plausible wrong answer.

```python
def check_order(plan) -> list[str]:
    problems = []
    for i, step in enumerate(plan.steps):
        if step.op == "buffer" and not is_projected(step.params["crs"]):
            problems.append(f"step {i}: buffering in a geographic system — distances are degrees")
        if step.op == "area" and any(s.op == "buffer" for s in plan.steps[:i]):
            if not any(s.op == "dissolve" for s in plan.steps[:i]):
                problems.append(f"step {i}: area after buffers with no dissolve — overlaps double count")
        if step.op == "dissolve" and i + 1 < len(plan.steps):
            if plan.steps[i + 1].op == "buffer":
                problems.append(f"step {i}: dissolving before a buffer is usually the wrong order")
    return problems
```

<figure class="diagram">
<svg viewBox="16 38 728 220" role="img" aria-labelledby="pbo-crs-t pbo-crs-d" xmlns="http://www.w3.org/2000/svg"><title id="pbo-crs-t">What a buffer distance means in each reference system</title><desc id="pbo-crs-d">A buffer of 500 in a projected system is 500 metres, while the same number in a geographic system is 500 degrees, which is larger than the planet.</desc><rect x="16" y="38" width="728" height="220" fill="#ffffff"/><rect x="30" y="52" width="700" height="46" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="50" y="80" fill="#1f2937" font-size="12.5">projected system, buffer 500: five hundred metres, as intended</text><rect x="30" y="108" width="580" height="46" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="50" y="136" fill="#1f2937" font-size="12.5">geographic system, buffer 500: five hundred degrees — no error raised</text><rect x="30" y="164" width="640" height="46" rx="6" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><text x="50" y="192" fill="#1f2937" font-size="12.5">geographic system, buffer 0.0045: roughly right at one latitude, wrong at others</text><text x="390" y="240" fill="#1f2937" font-size="13" text-anchor="middle">The third case is the dangerous one — it looks correct and is only correct locally</text></svg>
<figcaption><b>The nearly-right case causes the most trouble.</b> A degree-approximation tuned for one city gives answers that drift with latitude, so the result is defensible where it was tested and wrong everywhere else.</figcaption>
</figure>

## Validation & Testing

Test the planner on requests, not on plans. A fixture set of a few dozen sentences with expected step sequences catches regressions cheaply and without touching a database.

```python
CASES = [
    ("schools within 500 m of a main road",
     ["buffer", "intersect"]),
    ("total area of parkland within 1 km of any school",
     ["buffer", "dissolve", "intersect", "area"]),
    ("parcels that touch a protected area",
     ["intersect"]),
]


@pytest.mark.parametrize("prompt,expected", CASES)
def test_plan_shape(prompt, expected):
    plan = planner.plan(prompt)
    assert [s.op for s in plan.steps] == expected
    assert check_order(plan) == []
```

Separately, test the order checker with plans that are deliberately wrong. It is the component most likely to be quietly broken by a refactor, because a checker that returns an empty list always looks like it is working.

## Gotchas & Edge Cases

**Buffer in a geographic system.** The most common single error, and it produces a result rather than a failure. Requiring a projected system in the assumptions, and rejecting plans that buffer without one, closes it entirely.

**Dissolve placed after the area calculation.** Areas summed over overlapping buffers double-count the overlaps. The rule is that any aggregation over buffered geometry needs a dissolve between them, and it is worth stating that as a check rather than trusting the plan.

**Choosing the projection by convenience.** A national grid is fine within its country and increasingly wrong outside it. Where the region spans zones, an equal-area projection appropriate to the extent is the honest choice, and it should appear in the assumptions rather than being applied inside a helper.

**Overlay against unclean inputs.** Intersecting geometry that has not been validated produces slivers, and slivers propagate into the dissolve as fragments that inflate feature counts. Validity is a precondition of the overlay step, not a separate concern.

**Chains that grow.** A model with a long example list will decompose a two-step request into five, each individually reasonable. Capping the step count in the schema is crude and effective; reviewing the median step count over time catches the drift before the cap starts firing.

<figure class="diagram">
<svg viewBox="9 52 761 194" role="img" aria-labelledby="pbo-chain-t pbo-chain-d" xmlns="http://www.w3.org/2000/svg"><title id="pbo-chain-t">The canonical four-step chain</title><desc id="pbo-chain-d">Buffer in a projected system, dissolve the overlapping buffers, intersect with the target layer, then aggregate — each step depending on a property the previous one established.</desc><rect x="9" y="52" width="761" height="194" fill="#ffffff"/><defs><marker id="pbo-chain-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/></marker></defs><rect x="26" y="66" width="160" height="118" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="212" y="66" width="160" height="118" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="398" y="66" width="160" height="118" rx="8" fill="#efe9fd" stroke="#6d4bbd" stroke-width="2"/><rect x="584" y="66" width="160" height="118" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="106" y="100">buffer</text><text x="292" y="100">dissolve</text><text x="478" y="100">intersect</text><text x="664" y="100">aggregate</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="106" y="130">in a projected system</text><text x="106" y="154">distance in metres</text><text x="292" y="130">overlaps merge</text><text x="292" y="154">nothing counted twice</text><text x="478" y="130">against the target layer</text><text x="478" y="154">valid inputs only</text><text x="664" y="130">area or count</text><text x="664" y="154">over clean geometry</text></g><g stroke="#5b6471" stroke-width="2" marker-end="url(#pbo-chain-a)"><line x1="188" y1="125" x2="208" y2="125"/><line x1="374" y1="125" x2="394" y2="125"/><line x1="560" y1="125" x2="580" y2="125"/></g><text x="390" y="228" fill="#1f2937" font-size="13" text-anchor="middle">Each step assumes something the previous one guaranteed — reorder and the guarantee is gone</text></svg>
<figcaption><b>The dependencies are the reason order matters.</b> The dissolve exists so the aggregate is not double-counted, and the projection exists so the buffer means metres; move either and the step that depended on it keeps running against an assumption that is no longer true.</figcaption>
</figure>

## Showing the Plan Before Running It

For anything expensive, the plan is worth putting in front of the reader before it executes — not as a technical listing but as a sentence describing what will happen and under which assumptions. "I'll take everything within 500 metres of a river, measured in the national grid, and total the parkland inside it" is a plan a person can correct, and the correction costs nothing at that point.

The value is concentrated in the assumptions rather than the steps. Readers rarely dispute that a buffer should precede an intersection; they frequently dispute a threshold, a unit, or which version of a dataset was used, and every one of those is invisible unless the plan states it. A confirmation that only shows step names therefore gets agreement without conveying anything.

Tie the confirmation to estimated cost rather than showing it always. A plan that will finish in two seconds should just run, and one that will take four minutes and produce a number somebody will put in a document should be agreed first. That threshold is the same estimate used for routing, which means the confirmation costs nothing extra to implement and appears exactly when a reader's attention is worth interrupting.

## Operating This Step Over Time

Track how often plans are rejected by the order checker and which rule fired. A rule that fires constantly is telling you the prompt needs an example rather than that the checker is working, and the fix belongs upstream.

Watch the assumption values too. If the same reference system appears in almost every plan, it is effectively a default and should become one explicitly, which removes a chance for the model to pick something else. If several appear, the ones that are wrong for their regions are visible directly.

The checker itself needs maintenance of a kind that is easy to forget. Each rule encodes a mistake somebody actually made, and rules added in response to incidents accumulate without anyone removing the ones that no longer apply — a check against an operation that has since been dropped from the vocabulary, for instance, or one that duplicates a constraint the schema now enforces. Reading the rule list alongside the schema once or twice a year keeps the two from drifting into contradiction, which is the state in which a plan is rejected for a reason nobody can explain.

Keep a small set of worked examples current as well. The fixtures used in testing double as the reference for what a good plan looks like, and when the vocabulary changes they are the first thing that should be updated — before the prompt, because the prompt's examples should be drawn from them rather than written independently.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Should the model choose the projection?</span></summary><p>It should propose one and the system should validate it against the extent. A model has a reasonable sense of which national grid suits which country, and no way to know whether a request spans two of them. Checking the proposed system against the bounding box of the inputs catches the case where a locally correct choice is regionally wrong.</p></details>

<details class="faq-item"><summary><span>What if the request needs an operation the plan vocabulary lacks?</span></summary><p>Fail explicitly rather than approximating. A model asked for a network distance and given only buffer, overlay and dissolve will produce a straight-line approximation and present it as the answer, which is worse than a refusal naming the missing capability. Keeping the vocabulary small is fine; keeping it small silently is not.</p></details>

<details class="faq-item"><summary><span>How should the dissolve field be chosen?</span></summary><p>From the question, and stated. "Total area" dissolves everything; "area by district" dissolves by district. The two produce very different answers and the difference is a single parameter, which makes it exactly the sort of thing that should appear in the plan rather than being inferred inside an implementation.</p></details>

<details class="faq-item"><summary><span>Is it worth caching intermediate chain outputs?</span></summary><p>The buffer and dissolve steps, yes — they are expensive, deterministic, and frequently reused across questions about the same features. The intersection usually is not, because it depends on both inputs and the second one varies with the question. Keying on operation, inputs and parameters makes the distinction fall out naturally.</p></details>

## Related

- Up to the parent topic: [LLM-Assisted Geoprocessing Pipelines](/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/)
- [Decomposing Natural Language into Geoprocessing Steps](/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/decomposing-natural-language-into-geoprocessing-steps/)
- [Validating Intermediate Geoprocessing Outputs](/geospatial-prompt-engineering-tool-routing/llm-assisted-geoprocessing-pipelines/validating-intermediate-geoprocessing-outputs/)
- Related topic: [Topology Rule Enforcement via LLMs](/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/)
