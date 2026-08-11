---
title: Constraining Generated SQL to an Allow-Listed Function Set
description: Parse generated SQL and check every function and table against an allow list, so anything unrecognised is rejected rather than permitted — and keep the list honest over time.
slug: constraining-generated-sql-to-an-allow-listed-function-set
type: howto
breadcrumb: Allow-Listed Functions
datePublished: 2026-08-11
dateModified: 2026-08-11
---

# Constraining Generated SQL to an Allow-Listed Function Set

The difference between an allow list and a deny list is what happens to something nobody thought of. A deny list permits it; an allow list rejects it. For SQL shaped by text a stranger typed, that difference is the whole control — and it only holds if the check reads a parse tree rather than the statement's text. This guide covers building and keeping that list, the enforcement half of [prompt-to-spatial-SQL generation](/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/).

## When to Use This Approach

Any system where a generated statement reaches a database. The size of the list varies enormously; its existence does not.

| Deployment | List size | Notes |
|------------|-----------|-------|
| Public-facing analytics | A few dozen functions | Tight, reviewed, read-only |
| Internal analyst tool | A hundred or so | Broader, still enumerated |
| Read replica of open data | Wide | Still an allow list, still no writes |
| Anything with private data | Narrow | Scoped tables matter more than functions |

<figure class="diagram">
<svg viewBox="16 38 748 212" role="img" aria-labelledby="cas-two-t cas-two-d" xmlns="http://www.w3.org/2000/svg"><title id="cas-two-t">Allow list against deny list for an unrecognised function</title><desc id="cas-two-d">An allow list rejects anything it does not recognise, while a deny list permits anything its author did not anticipate.</desc><rect x="16" y="38" width="748" height="212" fill="#ffffff"/><rect x="30" y="52" width="340" height="160" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="410" y="52" width="340" height="160" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="200" y="84">allow list</text><text x="580" y="84">deny list</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="200" y="114">unknown means rejected</text><text x="200" y="140">a new function needs a decision</text><text x="200" y="166">coverage is provable</text><text x="580" y="114">unknown means permitted</text><text x="580" y="140">a new function arrives allowed</text><text x="580" y="166">coverage is a hope</text></g><text x="390" y="234" fill="#1f2937" font-size="13" text-anchor="middle">A deny list works perfectly against everything its author already knew about</text></svg>
<figcaption><b>The two are not variations on one idea.</b> They differ precisely in the case you cannot enumerate, which is the only case a security control exists to handle.</figcaption>
</figure>

## Implementation

Parse first. A statement is checked by walking its tree, never by searching its text, because whitespace, comments and casing all defeat text matching while changing nothing about what runs.

```python
import sqlglot
from sqlglot import exp

ALLOWED_FUNCTIONS = {
    "st_area", "st_astext", "st_buffer", "st_centroid", "st_contains",
    "st_distance", "st_dwithin", "st_intersects", "st_intersection",
    "st_isvalid", "st_length", "st_makevalid", "st_transform", "st_union",
    "count", "sum", "avg", "min", "max",
}
ALLOWED_TABLES = {"parcels", "rivers", "protected_areas", "schools"}


def check(sql: str, dialect: str = "postgres") -> list[str]:
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except sqlglot.ParseError as exc:
        return [f"unparseable: {exc}"]

    if len(statements) != 1:
        return ["more than one statement"]

    tree = statements[0]
    if not isinstance(tree, exp.Select):
        return [f"not a read: {type(tree).__name__.lower()}"]

    problems = []
    for node in tree.find_all(exp.Anonymous, exp.Func):
        name = (node.sql_name() or "").lower()
        if name and name not in ALLOWED_FUNCTIONS:
            problems.append(f"function not allowed: {name}")
    for table in tree.find_all(exp.Table):
        if table.name.lower() not in ALLOWED_TABLES:
            problems.append(f"table not allowed: {table.name}")
    return problems
```

Two structural checks matter as much as the lists. Rejecting anything that is not a single statement closes the case where a benign read is followed by something else. Rejecting anything that is not a `SELECT` means the list never has to reason about writes at all.

The rejection message returned to the model should name the specific item, because that is what makes the next attempt succeed rather than repeat.

```python
def to_model_feedback(problems: list[str]) -> str:
    return (
        "The statement was rejected: " + "; ".join(problems) +
        ". Rewrite it using only the functions and tables described in the schema."
    )
```

<figure class="diagram">
<svg viewBox="16 32 749 224" role="img" aria-labelledby="cas-layer-t cas-layer-d" xmlns="http://www.w3.org/2000/svg"><title id="cas-layer-t">The four checks and what each one closes</title><desc id="cas-layer-d">Single statement, read only, allowed functions and allowed tables each close a distinct class of problem, and all four run against the parse tree before execution.</desc><rect x="16" y="32" width="749" height="224" fill="#ffffff"/><g><rect x="30" y="46" width="172" height="150" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="216" y="46" width="172" height="150" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="402" y="46" width="172" height="150" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="588" y="46" width="162" height="150" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/></g><g fill="#1f2937" font-size="13.5" text-anchor="middle" font-weight="600"><text x="116" y="80">one statement</text><text x="302" y="80">read only</text><text x="488" y="80">allowed functions</text><text x="669" y="80">allowed tables</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="116" y="112">no stacking</text><text x="116" y="138">a second is never</text><text x="116" y="170">a legitimate request</text><text x="302" y="112">no writes at all</text><text x="302" y="138">the list ignores</text><text x="302" y="170">everything but selects</text><text x="488" y="112">enumerated, not filtered</text><text x="488" y="138">unknown is rejected</text><text x="488" y="170">coverage is provable</text><text x="669" y="112">scoped to the caller</text><text x="669" y="138">not just to the schema</text><text x="669" y="170">privileges back it up</text></g><text x="390" y="238" fill="#1f2937" font-size="13" text-anchor="middle">All four are answerable from the parse tree — none of them requires running anything</text></svg>
<figcaption><b>Rejection costs a parse.</b> Every check here runs in under a millisecond against a tree the generator already produced, which means there is no version of this worth skipping for performance.</figcaption>
</figure>

## Validation & Testing

Test with statements designed to slip past a text matcher, because those are what a text matcher would have missed.

```python
BYPASS_ATTEMPTS = [
    "SELECT /* st_area */ pg_read_file('/etc/passwd')",
    "SELECT ST_Area(geom) FROM parcels; DROP TABLE parcels",
    "SELECT\n\tPG_SLEEP(10)",
    "select st_area(geom) from PARCELS_SECRET",
    "SELECT ST_Area(geom) FROM parcels UNION SELECT version()",
]


@pytest.mark.parametrize("sql", BYPASS_ATTEMPTS)
def test_rejected(sql):
    assert check(sql), f"should have been rejected: {sql}"


def test_ordinary_query_passes():
    assert check("SELECT ST_Area(geom) FROM parcels WHERE ST_DWithin(geom, %s, 500)") == []
```

Also assert that the parse failure path rejects. A malformed statement that produces a parse error must not fall through to execution on the theory that the database will reject it anyway — that reasoning is correct today and becomes wrong the first time somebody adds a fallback.

## Gotchas & Edge Cases

**Operators that are functions.** Spatial operators like `&&` and `<->` do not appear as function nodes in most parsers. They need their own enumeration, and forgetting them leaves a gap that looks like thoroughness.

**Schema-qualified names.** `public.st_area` and `st_area` are the same function and different strings. Normalise qualified names before comparison, or the list will reject legitimate statements and someone will widen it in the wrong direction.

**Common table expressions and subqueries.** Tables referenced inside a `WITH` clause or a nested select are still tables. A checker that only inspects the top-level `FROM` misses them entirely, which is why walking the whole tree matters more than checking the obvious places.

**The list broadening under pressure.** Every addition is individually justified and the aggregate drifts toward permissiveness. Reviewing the whole list periodically, rather than each addition as it arrives, is the only review that sees what it has become.

**Relying on the list alone.** It is one layer. The connection underneath should be read-only and scoped so that a gap in the list is a bug rather than a breach, and the two together are what make the design survivable.

<figure class="diagram">
<svg viewBox="16 38 728 212" role="img" aria-labelledby="cas-parse-t cas-parse-d" xmlns="http://www.w3.org/2000/svg"><title id="cas-parse-t">Where a text matcher fails and a parser does not</title><desc id="cas-parse-d">Comments, unusual whitespace, case differences and nested subqueries all change the text of a statement without changing what it does, and only a parser is unaffected.</desc><rect x="16" y="38" width="728" height="212" fill="#ffffff"/><rect x="30" y="52" width="700" height="46" rx="6" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="50" y="80" fill="#1f2937" font-size="12.5">parse tree: comments, spacing and casing are already gone before the check runs</text><rect x="30" y="108" width="560" height="46" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="50" y="136" fill="#1f2937" font-size="12.5">text match: a comment between the name and its parenthesis defeats it</text><rect x="30" y="164" width="480" height="46" rx="6" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="50" y="192" fill="#1f2937" font-size="12.5">text match: a table inside a subquery is never inspected</text><text x="390" y="238" fill="#1f2937" font-size="13" text-anchor="middle">The parser removes the entire category of problem rather than enumerating it</text></svg>
<figcaption><b>This is why the parse is not optional.</b> Each text-matching failure has an individual fix, and the set of them is unbounded — which is the situation a structural check exists to escape.</figcaption>
</figure>

## Generating the List Rather Than Writing It

A hand-written list drifts from the schema it describes. Generating it from the catalogue — the functions actually installed, the tables the connection can actually read — and then narrowing it by hand keeps the two in step and makes additions deliberate rather than accidental.

The generated starting point also answers a question that is otherwise guesswork: what is currently reachable. A connection with more privileges than anyone realised shows up immediately as a list far longer than expected, which is a finding worth having before it is a finding somebody else has. The narrowing pass is then a review of a real inventory rather than an attempt to imagine one.

The same generation should feed the schema description given to the model, so that what it is told about and what it is permitted to use are the same set. A model shown a table it cannot query will use it, get rejected, and try again — and the wasted round trips are entirely avoidable, since both descriptions come from the same source. Keeping them together also means that removing a table removes it from both places at once, which is the sort of change that otherwise gets done in one place and remembered in the other several weeks later.

## Operating This Step Over Time

Count rejections by reason, and keep the counts separated by function and table rather than aggregated into a single total. A function rejected repeatedly is usually a legitimate capability the list has not caught up with, and the right response is to consider adding it; one rejected once is worth reading in full, because a single unusual attempt is the shape most worth understanding. Both are invisible without the counter and obvious with it, and the separation is what lets you tell a missing capability from an unusual request at a glance.

Re-generate the candidate list after every schema migration and diff it against the enforced one. New tables and functions appear silently otherwise, and the diff is a two-minute review that catches a table added for one purpose becoming reachable for all of them.

Keep the bypass test set growing. Every rejected statement that turned out to be interesting — an unusual construct, a nesting nobody had considered, a function reached through an operator — belongs in the fixtures, because a test written from a real attempt is worth several written from imagination. The set also documents what the check is known to handle, which is the question anyone reviewing it will ask first.

Watch for the check being skipped. The most common way an allow list stops working is not that it is broadened but that a new code path reaches the database without passing through it — a background job, an export, a debugging endpoint that was never meant to survive. Routing every statement through one function, and asserting in tests that no other call site exists, is the structural version of that discipline; a periodic search for direct execution calls is the cheap version.

## Frequently Asked Questions

<details class="faq-item"><summary><span>Does the parser need to match the database exactly?</span></summary><p>Close enough that anything it accepts, the database interprets the same way. A parser that is more permissive than the database is safe — the database rejects what it does not understand. One that parses a construct differently is the risk, which is why the dialect should be set explicitly and why parse failures must reject rather than pass through.</p></details>

<details class="faq-item"><summary><span>What about user-defined functions?</span></summary><p>Treat them exactly like built-ins: enumerated individually, never by schema or prefix. A rule allowing everything in one schema means every future function added there is permitted by default, which reintroduces the deny-list property the whole approach exists to avoid.</p></details>

<details class="faq-item"><summary><span>Should rejections be shown to the user?</span></summary><p>Not the details. Naming the rejected function tells an attacker what to try next and tells an ordinary user nothing actionable. The reader gets a short sentence about being unable to answer that question and an invitation to rephrase; the specifics go to the log and to the model, which can act on them.</p></details>

<details class="faq-item"><summary><span>How does this interact with cost checks?</span></summary><p>They are separate and sequential. The allow list decides whether a statement is permitted; the planner's cost estimate decides whether it is affordable. A statement can be entirely legitimate and ruinously expensive, and conflating the two produces either a list that rejects reasonable queries or a cost check nobody added.</p></details>

## Related

- Up to the parent topic: [Prompt-to-Spatial-SQL Generation](/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/)
- [Preventing SQL Injection in LLM-Generated Spatial Queries](/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/preventing-sql-injection-in-llm-generated-spatial-queries/)
- [Generating Valid PostGIS Queries from Natural Language](/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/generating-valid-postgis-queries-from-natural-language/)
- Related topic: [Spatial Function-Calling Schemas](/geospatial-prompt-engineering-tool-routing/spatial-function-calling-schemas/)
