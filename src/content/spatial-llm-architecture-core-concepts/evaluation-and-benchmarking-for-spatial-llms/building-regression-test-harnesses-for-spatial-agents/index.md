---
title: Building Regression Test Harnesses for Spatial Agents
description: Build a harness that runs a versioned case set deterministically, distinguishes its own drift from the agent's, and turns every production failure into a permanent test.
slug: building-regression-test-harnesses-for-spatial-agents
type: howto
breadcrumb: Regression Test Harnesses
datePublished: 2025-02-06
dateModified: 2026-08-11
---

# Building Regression Test Harnesses for Spatial Agents

A spatial agent has too many moving parts for spot checks to be meaningful: a chunking change, a library upgrade and a prompt edit all move the same numbers, and none of them announces itself. A regression harness is the apparatus that makes those movements attributable — a versioned case set, a deterministic runner, and a self-check that fails before the agent is blamed. This guide builds it, as the machinery behind [evaluation and benchmarking for spatial LLMs](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/).

## When to Use This Approach

Build the harness before the second person joins the project, not after the first mysterious regression. Its value comes from history, and history only accumulates if the harness exists early.

| Change being made | What the harness must isolate |
|-------------------|-------------------------------|
| Prompt or instruction edit | Agent behaviour, everything else pinned |
| Model or version change | Agent behaviour, tokenizer recorded |
| Geometry library upgrade | Harness drift — the self-check fires first |
| Chunking or retrieval change | Retrieval quality, scored separately |
| Truth-data correction | Nothing — historical comparisons are invalidated |

The last row is the one teams get wrong. Correcting a truth geometry improves the case set and makes every previous score incomparable, so it must bump the case-set version rather than being applied silently.

<figure class="diagram">
<svg viewBox="16 32 748 214" role="img" aria-labelledby="rth-attr-t rth-attr-d" xmlns="http://www.w3.org/2000/svg"><title id="rth-attr-t">Four things that move the same number</title><desc id="rth-attr-d">A score change can come from the agent, the harness, the libraries or the truth data, and without pinning and self-checks all four look identical in a report.</desc><rect x="16" y="32" width="748" height="214" fill="#ffffff"/><rect x="30" y="46" width="360" height="86" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="410" y="46" width="340" height="86" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="30" y="146" width="360" height="86" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/><rect x="410" y="146" width="340" height="86" rx="8" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><g fill="#1f2937" font-size="13" font-weight="600"><text x="52" y="76">the agent changed</text><text x="432" y="76">a library changed</text><text x="52" y="176">the harness changed</text><text x="432" y="176">the truth data changed</text></g><g fill="#5b6471" font-size="12"><text x="52" y="102">what you wanted to measure</text><text x="52" y="122">everything else must be pinned</text><text x="432" y="102">repair behaviour moves scores</text><text x="432" y="122">pin versions, record them</text><text x="52" y="202">a scoring edit, a new fixture</text><text x="52" y="222">the self-check fires first</text><text x="432" y="202">a corrected geometry</text><text x="432" y="222">bump the case-set version</text></g></svg>
<figcaption><b>Only the top-left is a finding.</b> The other three are the reasons a team spends a week investigating a model that never changed, and each one is prevented by a version stamp rather than by cleverness.</figcaption>
</figure>

## Implementation

The harness pins everything it can, runs the case set, and records the environment alongside the results so a future comparison knows what it is comparing.

```python
import hashlib
import json
import logging
import platform
from dataclasses import dataclass, asdict
from typing import Callable, Sequence

log = logging.getLogger("regression_harness")


@dataclass(frozen=True)
class Environment:
    case_set_version: str
    agent_version: str
    tokenizer: str
    geos_version: str
    proj_version: str
    python: str


@dataclass(frozen=True)
class Run:
    environment: Environment
    results: tuple[dict, ...]
    case_set_digest: str


def case_set_digest(cases: Sequence[dict]) -> str:
    """A stable digest of the case set, so a silent edit is detectable."""
    payload = json.dumps(
        [{k: c[k] for k in sorted(c) if k != "notes"} for c in cases],
        sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def capture_environment(case_set_version: str, agent_version: str,
                        tokenizer: str) -> Environment:
    """Record what could move a score, before anything runs."""
    try:
        from shapely import geos_version_string
        geos = geos_version_string
    except Exception:                                 # a missing library is itself a finding
        geos = "unavailable"
        log.warning("geometry library version could not be determined")
    try:
        import pyproj
        proj = pyproj.__proj_version__
    except Exception:
        proj = "unavailable"
    return Environment(case_set_version, agent_version, tokenizer,
                       geos, proj, platform.python_version())
```

Recording the geometry and projection library versions is not bureaucracy. A geometry engine upgrade changes repair behaviour on invalid input, which moves overlap scores by a percent or two — indistinguishable from a model regression unless the version is in the record.

The runner itself must be deterministic and must fail fast when its own behaviour has changed.

```python
FROZEN = {"case-001": 0.8734, "case-014": 0.0000, "case-102": 1.0000}


class HarnessDrift(AssertionError):
    """The harness scores differently from before — investigate it, not the agent."""


def self_check(score_case: Callable[[str], float], tolerance: float = 1e-4) -> None:
    """Score three frozen cases before the sweep. Fails loudly if the harness moved."""
    for case_id, expected in FROZEN.items():
        try:
            got = score_case(case_id)
        except Exception as exc:
            raise HarnessDrift(f"self-check case {case_id} raised: {exc}") from exc
        if abs(got - expected) > tolerance:
            raise HarnessDrift(
                f"harness drift on {case_id}: expected {expected}, got {got:.4f}. "
                "Check library versions and truth data before blaming the agent.")


def run(cases: Sequence[dict], answer: Callable[[dict], dict],
        score: Callable[[dict, dict], dict], env: Environment) -> Run:
    """Run the sweep. Individual case failures are results, never aborts."""
    self_check(lambda cid: score(_case(cases, cid), answer(_case(cases, cid)))["score"])

    results = []
    for case in sorted(cases, key=lambda c: c["case_id"]):     # deterministic order
        try:
            produced = answer(case)
        except Exception as exc:                                # a crash is a result
            log.warning("case %s raised: %s", case["case_id"], exc)
            results.append({"case_id": case["case_id"], "family": case["family"],
                            "score": 0.0, "parsed": False, "note": f"agent raised: {exc}"})
            continue
        results.append({**score(case, produced), "case_id": case["case_id"],
                        "family": case["family"], "region": case.get("region", "all")})
    return Run(env, tuple(results), case_set_digest(cases))
```

Two properties make this harness trustworthy. Cases run in sorted order, so a change in dictionary iteration cannot reorder a sweep and produce a different sample under a partial run. And an agent crash becomes a scored result rather than an aborted sweep, because a crash on one case is exactly the kind of regression the harness exists to catch — and a sweep that stops at the first one measures nothing.

<figure class="diagram">
<svg viewBox="0 56 780 174" role="img" aria-labelledby="rth-grow-t rth-grow-d" xmlns="http://www.w3.org/2000/svg"><title id="rth-grow-t">How a case set should grow</title><desc id="rth-grow-d">Cases are added from production failures and never retired, so the set accumulates every regression the system has ever had and each one is checked on every sweep.</desc><rect x="0" y="56" width="780" height="174" fill="#ffffff"/><rect x="30" y="70" width="140" height="90" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="196" y="70" width="140" height="90" rx="8" fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2"/><rect x="362" y="70" width="180" height="90" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="568" y="70" width="182" height="90" rx="8" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><g fill="#1f2937" font-size="12.5" text-anchor="middle" font-weight="600"><text x="100" y="102">seed set</text><text x="266" y="102">+ failures</text><text x="452" y="102">+ more failures</text><text x="659" y="102">+ every one since</text></g><g fill="#5b6471" font-size="12" text-anchor="middle"><text x="100" y="128">50 cases</text><text x="266" y="128">80</text><text x="452" y="128">140</text><text x="659" y="128">300 and rising</text></g><text x="390" y="212" fill="#1f2937" font-size="13" text-anchor="middle">Nothing is retired — a case that has passed for a year is the one that catches the next rewrite</text></svg>
<figcaption><b>Growth is the point, not a problem to manage.</b> Each case costs a second to run and encodes one failure somebody once had to diagnose, which makes retiring old cases the most expensive kind of tidying available.</figcaption>
</figure>

## Validation & Testing

```python
def test_self_check_fires_before_the_sweep():
    def drifted(_case_id):
        return 0.5                                    # nothing like the frozen expectations
    try:
        self_check(drifted)
    except HarnessDrift as exc:
        assert "before blaming the agent" in str(exc)
        return
    raise AssertionError("harness drift must stop the sweep")


def test_agent_crash_becomes_a_scored_result():
    def crashing(_case):
        raise RuntimeError("tool timeout")
    out = run(CASES, crashing, score_fn, ENV)
    assert len(out.results) == len(CASES)
    assert all(r["score"] == 0.0 and not r["parsed"] for r in out.results)


def test_case_set_digest_changes_when_truth_changes():
    before = case_set_digest(CASES)
    edited = [{**CASES[0], "truth_geometry": "POLYGON((0 0,1 0,1 1,0 0))"}, *CASES[1:]]
    assert case_set_digest(edited) != before


def test_ordering_is_deterministic():
    a = run(CASES, answer_fn, score_fn, ENV)
    b = run(list(reversed(CASES)), answer_fn, score_fn, ENV)
    assert [r["case_id"] for r in a.results] == [r["case_id"] for r in b.results]
```

The third test is the one that keeps truth-data changes honest. A digest that moves when a truth geometry is edited turns a silent invalidation of history into a visible one, and the comparison tooling can then refuse to plot two runs with different digests on the same axis.

## Gotchas & Edge Cases

**A harness that stops at the first failure.** Standard test-runner behaviour and exactly wrong here: the sweep is a measurement, not a build, and a partial measurement is worse than none because it looks complete.

**Non-determinism from the agent.** A sampled model produces different answers on identical input, so a single sweep conflates real change with sampling noise. Lower the temperature for evaluation runs, or run a subset repeatedly and report the spread — but say which you did.

**Environment captured after the run.** Library versions read at the end of a long sweep can differ from those at the start in a container that was updated mid-run. Capture first, and store the record with the results rather than alongside them.

<figure class="diagram">
<svg viewBox="16 24 744 206" role="img" aria-labelledby="rth-stop-t rth-stop-d" xmlns="http://www.w3.org/2000/svg"><title id="rth-stop-t">A sweep that aborts against one that completes</title><desc id="rth-stop-d">Stopping at the first failing case reports a fraction of the set and looks like a completed run; scoring every case produces a measurement in which the failures are visible as results.</desc><rect x="16" y="24" width="744" height="206" fill="#ffffff"/><text x="30" y="62" fill="#b3324f" font-size="13" font-weight="600">aborts</text><rect x="170" y="38" width="90" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="266" y="38" width="60" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><text x="500" y="64" fill="#5b6471" font-size="12">the remaining 280 cases never ran</text><text x="30" y="152" fill="#12805c" font-size="13" font-weight="600">completes</text><rect x="170" y="128" width="90" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><rect x="266" y="128" width="60" height="42" rx="5" fill="#fdeaee" stroke="#b3324f" stroke-width="2"/><rect x="332" y="128" width="330" height="42" rx="5" fill="#e4f5ec" stroke="#12805c" stroke-width="2"/><text x="497" y="154" fill="#1f2937" font-size="12" text-anchor="middle">280 more cases, scored</text><text x="380" y="212" fill="#1f2937" font-size="13" text-anchor="middle">A sweep is a measurement, not a build — a partial one looks complete and is not</text></svg>
<figcaption><b>Test-runner instincts are wrong here.</b> Failing fast is right when the goal is a green build and wrong when the goal is a distribution, because the cases after the first failure are exactly the ones that tell you how widespread it is.</figcaption>
</figure>

**Cases whose truth was derived from the system.** A case labelled from the agent's own past output tests that the agent still agrees with itself. Label from the source data, ideally by someone who did not build the agent.

**Digest computed over unordered data.** A digest that changes when a dictionary is re-serialised is worse than none, because it invalidates history for no reason. Sort keys, exclude free-text notes, and test that a cosmetic edit does not move it.

**A frozen self-check that is silently updated.** When the self-check fails and somebody updates the expected values to match, the harness has lost its only defence against its own drift. Changing a frozen value should require the same evidence as changing a threshold.

## Frequently Asked Questions

<details class="faq-item"><summary><span>How often should the sweep run?</span></summary><p>On every change to the agent, and on a schedule regardless — nightly is common and daily is enough. The scheduled run is what catches drift that no change caused: a library updated by a base-image refresh, a data source that changed shape, an external service whose behaviour moved. Those are invisible to a change-triggered sweep because nothing in your repository changed.</p></details>

<details class="faq-item"><summary><span>Should the harness call real external services?</span></summary><p>For the agent under test, yes — a harness that mocks the geometry engine is testing the mock. For anything slow or rate-limited, record and replay: capture real responses once, replay them thereafter, and refresh the recordings on a schedule. That keeps sweeps fast and deterministic while still exercising the real integration on a known cadence.</p></details>

<details class="faq-item"><summary><span>What belongs in a case beyond the question and the truth?</span></summary><p>The family, the region, and a note explaining why the case exists. The first two drive the report breakdown; the third is what stops a future maintainer from deleting a case that looks redundant. A case labelled "added after the Newport incident, checks that ambiguous toponyms refuse rather than guess" survives a spring clean that a bare question and geometry will not.</p></details>

<details class="faq-item"><summary><span>How should the harness handle cases that are expected to fail?</span></summary><p>Mark them explicitly rather than removing them, and assert that they still fail. A known limitation that starts passing is information — either it was fixed or the case stopped testing what it used to — and a harness that only tracks successes cannot tell you either way.</p></details>

One organisational note. The case set belongs in the same repository as the agent, versioned alongside it, so a change to behaviour and the case that covers it land in the same commit. A case set in a separate store drifts out of step within weeks, and the first sign is a sweep that fails against a version of the agent that no longer exists.

## Related

- Up to the parent topic: [Evaluation and Benchmarking for Spatial LLMs](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/)
- [Setting Release Thresholds for Spatial Agents](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/setting-release-thresholds-for-spatial-agents/)
- [Detecting Hallucinated Coordinates in LLM Output](/spatial-llm-architecture-core-concepts/evaluation-and-benchmarking-for-spatial-llms/detecting-hallucinated-coordinates-in-llm-output/)
- Related topic: [Multi-Step Spatial Agent Orchestration](/geospatial-prompt-engineering-tool-routing/multi-step-spatial-agent-orchestration/)
