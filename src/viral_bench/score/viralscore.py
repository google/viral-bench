# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The ViralScore: one comparable 0-100 number per built app.

The crowd is the benchmark's measuring instrument, so the score is designed
around two properties that can be *measured*, not asserted:

* **reliability** -- scoring the same app twice should give nearly the same
  number (low within-app variance).
* **discrimination** -- different apps should land far apart (high between-app
  variance).

Their ratio, ``sigma^2_between / (sigma^2_between + sigma^2_within)``, is what
``viral_bench.score.calibrate`` reports, and it is the number to optimise.

Design decisions, each driven by a measurement rather than taste:

* **Per-capita participation, never raw counts.** Counts scale with crowd size
  and are heavy-tailed. Distinct-actor fractions of the *exposed* audience are
  bounded in [0,1] and comparable between an 8-agent dev run and a 50-agent
  scored run.
* **Likes are excluded.** Across the runs measured, likes had zero between-app
  variance (every crowd likes everything) while carrying real run-to-run noise:
  pure noise, no signal.
* **Reposts carry the behavioural weight -- but only reposts of a PEER, by
  someone who would share it.** Reposting was the single most discriminating
  signal measured (between/within spread ~3.7x), and the reason turned out to be
  unflattering: 68% of reposting actors only ever reposted the founder's own
  launch post, and that rate correlates +0.95 with the finished score. It was
  measuring app quality a second time, not spread. From score_version 1.6 the
  behavioural term is ``advocacy_spread`` (see :class:`ScoreWeights`).
* **Cascade depth is measured but unweighted, and the term is mis-specified.**
  It was originally zeroed because secondary engagement was *identically zero*
  at 8, 20 and 25 agents. On crowd arch v12 ``secondary_share`` is no longer
  zero -- mean 0.142, sd 0.117 over 738 runs -- but that is not a cascade
  appearing. Its ``primary`` bucket lumps the founder's launch post together
  with all thirty agent posts, so 46.7% of that denominator is peer-authored
  first-generation content and the term drifts upward whenever agents talk to
  each other. Measured strictly -- engagement landing on a repost or quote --
  it is 30 events across 106 runs, i.e. still dead. So the weight stays 0, for
  a better-understood reason: the signal it names does not occur, and the
  movement it shows belongs to a propagation axis rather than to this score.
  See :mod:`viral_bench.score.spread`, which extracts that axis separately.
* **A broken app cannot be viral.** If the validity gate says the app does not
  do what it claims, the whole score is multiplied by
  :data:`BROKEN_APP_MULTIPLIER`.

The score reports **breadth** (whole-crowd adoption). In-audience *resonance* is
reported alongside but deliberately kept out of the number, so one score stays
comparable across apps aimed at different audiences.
"""

from __future__ import annotations

import logging
import random
from dataclasses import asdict, dataclass, field

from viral_bench import config as _config
from viral_bench.score.signals import RunSignals

#: Bumped whenever the formula or weights change, so a leaderboard never mixes
#: numbers produced by different definitions.
SCORE_VERSION = "1.8"

#: What an app that does not run is multiplied by (it can still be
#: *discussed*, so the floor is not zero -- but it cannot compete with something
#: that works).
BROKEN_APP_MULTIPLIER = 0.2

#: What an app that RUNS but fails its own smoke check is multiplied by.
#:
#: ``verify_code`` reports ``does_what_it_claims = builds and runs and smoke_ok``,
#: which lumps two different failures together. Calibration surfaced the
#: distinction: a weak model shipped a working app with a *network-dependent*
#: smoke command, violating the manifest contract. The app started, served, and
#: the agents who used it rated its functionality 7.5/10 -- yet the full
#: broken-app penalty cut its score fivefold. Direct observation by agents who
#: used the app is stronger evidence than a health check the same model wrote
#: about itself, so a failed self-check is a real but far smaller penalty than a
#: dead app.
FAILED_SELF_CHECK_MULTIPLIER = 0.6

#: Gate policies. ``hard`` applies the multiplier outright, while
#: ``evidence_graded`` scales it up towards 1.0 by the share of the crowd that
#: got the app working, so a deterministic probe can no longer overrule
#: first-hand experience.
GATE_HARD = "hard"
GATE_EVIDENCE_GRADED = "evidence_graded"
GATE_POLICIES = (GATE_HARD, GATE_EVIDENCE_GRADED)

#: Crowd size a scored run should use. **30, not 50.**
#:
#: The old value of 50 came from a reliability table whose n=50 bucket contained
#: no runs of the strong build at all -- every one of them had been destroyed by
#: the recommender crash -- so "noise collapses to 2.2 points
#: at 50" was computed over the two floored apps and is not reproducible.
#:
#: Re-measured on the fixed instrument (crowd arch v4, 3 builds, 3 seeds each):
#:
#: | | n=30 | n=50 |
#: |---|---|---|
#: | good - broken control | **59.7** | 45.9 |
#: | good - mid | **46.1** | 28.1 |
#: | pooled within-cell sd | **3.52** | 9.60 |
#: | separation / noise | **17.0** | 4.8 |
#: | wall clock per run | **2.4 min** | 5.0 min |
#:
#: n=50 is worse on every axis at twice the cost: the strong build scored 70.0 /
#: 43.5 / 41.3 across three seeds. Interviews were complete in all of them (47-50
#: of 50), so this is genuine crowd-behaviour variance, not lost data -- which is
#: what was predicted from the collapse of distinct opinion
#: clusters at n=50.
#:
#: Loaded from config/score.yaml ``minimums.calibrated_crowd_size`` (the YAML used
#: to advertise this knob while saying it was not wired, and carried the stale 50).
CALIBRATED_CROWD_SIZE = _config.score_minimum("calibrated_crowd_size", 30)

#: Interviews below which a run cannot produce a number at all. The crowd's own
#: verdicts are the primary signal, and scoring without them reports a different
#: metric under the same name (see :func:`unscorable_reasons`).
MIN_INTERVIEWS = _config.score_minimum("interviews", 1)


def _gate_policy(value) -> str:
    """Coerce a configured gate policy, defaulting to the pre-1.7 behaviour.

    An unknown string is a typo in ``config/score.yaml``, and silently scoring
    under a different gate than the one written down is exactly the class of bug
    this file keeps having to fix -- so it raises rather than falling back.
    """
    if value is None:
        return GATE_HARD
    text = str(value).strip().lower()
    if text not in GATE_POLICIES:
        raise ValueError(
            f"unknown gate policy {value!r} in score.yaml; "
            f"expected one of {list(GATE_POLICIES)}"
        )
    return text


@dataclass(frozen=True)
class ScoreWeights:
    """Component weights. Must sum to 1.0.

    Split 65% judgement (adoption, advocacy, craft) / 35% behaviour
    (amplification, cascade): judgement signals are per-agent and independent so
    their noise averages down as ~1/sqrt(n), while behaviour is cascade-coupled
    and noisier -- but reposts measured as the most discriminating single signal,
    so behaviour earns a real share rather than a token one.

    Calibrated, not guessed. Re-scoring 27 stored runs (3 builds x 3 crowd sizes
    x 3 seeds) gave per-component signal-to-noise of craft 15.4, adoption 3.5,
    amplification 2.4, advocacy 2.2, cascade 0.4, so craft was lifted from 0.20
    to 0.25 and advocacy trimmed from 0.25 to 0.20 (reliability 0.958 -> 0.964).
    Craft-heavy weighting scored marginally better still (0.971) but only by
    gutting advocacy, and "would you put your name behind sharing this" is the
    core of the construct -- a virality metric tuned purely for stability stops
    measuring virality. Dropping behaviour entirely was *worse* (0.950, noise
    8.06 vs 5.75), so the behavioural share is carrying real signal.
    """

    adoption: float = 0.20
    advocacy: float = 0.20
    craft: float = 0.25
    amplification: float = 0.10
    #: Peer amplification by agents who would themselves share it. Supersedes
    #: ``amplification`` from score_version 1.6 (profile ``v5_advocacy``), which
    #: counted a repost of the founder's own launch post -- 68% of all reposting
    #: actors, an act needing no contact with another agent, and correlating
    #: **+0.95** with the finished score. It was a laundered second copy of "the
    #: app is good" sitting in the slot reserved for spread.
    #:
    #: Priced, not assumed. Over the 738 stored arch-v12 runs, swapping it in
    #: cuts the null test -- the instrument separating a model from ITSELF --
    #: from 1.16 to 0.51, lifts the discrimination ratio 4.64 -> 4.69, cuts noise
    #: 7.18 -> 6.30 and drops the broken control from 7.4 to 3.9. Deleting
    #: amplification outright instead makes the null *worse* (1.33), so this is
    #: carrying signal rather than merely retiring a noisy term.
    #:
    #: It is sparse -- mean 0.039, zero in 56% of runs -- so at 0.15 it acts
    #: partly as a flat penalty on runs where nobody who endorsed the app passed
    #: it on. That is the intended reading, but it does compress resolution among
    #: the majority, and it is why the control margin narrows 69.7 -> 62.7. 0.08
    #: is the measured conservative alternative (null 0.81, the best ratio of any
    #: variant at 4.80, margin 66.8).
    advocacy_spread: float = 0.00
    #: Net behavioural valence (liked minus disliked, per exposed agent). Split
    #: out of amplification in score_version 1.3, where it was doing that
    #: component's discriminating work while an inverted comment term rode along.
    #:
    #: Retired to 0.0 in the v4 profile at score_version 1.5. Measured over 150
    #: runs it is the worst-behaved term in the score: between-cell spread over
    #: within-cell noise of 1.42, against 9.09 for craft and 6.42 for adoption,
    #: and it points the wrong way (+0.017 toward the weaker model). Dislikes
    #: are ~0 on any app that works, so it is a near-constant that compresses
    #: the scale. Every one of the 29 scoring ablations that removed or replaced
    #: it improved the null test, the discrimination ratio AND the control
    #: margin together. Kept as a field so older profiles still load.
    reception: float = 0.25
    #: Did the crowd's work survive a reload. Behavioural, verifiable, and the
    #: only quality signal here that is not somebody's opinion.
    persistence: float = 0.00
    cascade: float = 0.00  # degenerate today, see the module docstring
    #: Weights for the agentic autorater's dimensions (see score/autorater.py).
    #: Empty means a purely deterministic score.
    autorater: dict = field(default_factory=dict)
    #: Multiplier applied when the app does not build or start. Per profile, so
    #: the gate is inside the search space: it is the single strongest lever in
    #: the whole score (0.6 is a 40% haircut, ~22 points on a 55-point build --
    #: larger than any weight change) and it used to be a module constant that
    #: config could not reach. Editing it in score.yaml changed nothing.
    gate_broken_app: float = BROKEN_APP_MULTIPLIER
    #: Multiplier when the app runs but fails its own smoke check.
    gate_failed_self_check: float = FAILED_SELF_CHECK_MULTIPLIER
    #: ``"hard"`` (apply the floor outright) or ``"evidence_graded"`` (scale the
    #: floor up towards 1.0 by the share of the crowd that got the app working).
    #: Defaults to ``hard`` so every profile written before score_version 1.7
    #: keeps scoring exactly as it did.
    gate_policy: str = GATE_HARD
    #: Which config/score.yaml profile this came from, recorded in the output so
    #: a leaderboard entry always says which definition produced it.
    profile: str = "code-default"

    #: Fields that are not component weights and must never enter the sum.
    _NON_WEIGHT = (
        "autorater",
        "profile",
        "gate_broken_app",
        "gate_failed_self_check",
        "gate_policy",
    )

    def as_dict(self) -> dict:
        data = self.deterministic_weights()
        data.update(self.autorater)
        return data

    def deterministic_weights(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k not in self._NON_WEIGHT}

    def gates(self) -> dict:
        """The multipliers this profile applies, kept out of the weight sum."""
        return {
            "broken_app": self.gate_broken_app,
            "failed_self_check": self.gate_failed_self_check,
            "policy": self.gate_policy,
        }

    def validate(self) -> None:
        total = sum(self.as_dict().values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"ScoreWeights must sum to 1.0, got {total}")
        if self.gate_policy not in GATE_POLICIES:
            raise ValueError(
                f"gate policy must be one of {list(GATE_POLICIES)}, "
                f"got {self.gate_policy!r}"
            )

    @classmethod
    def from_profile(cls, name: str | None = None) -> ScoreWeights:
        """Load a named weight profile from ``config/score.yaml``.

        Falls back to the code defaults when the profile is missing or
        malformed, so a typo in config degrades to a known-good scoring
        definition rather than a half-applied one.
        """
        from viral_bench import config as _config

        profile = _config.score_profile(name)
        det = profile.get("deterministic") if isinstance(profile, dict) else None
        if not isinstance(det, dict) or not det:
            return cls()
        rater = profile.get("autorater")
        gates = profile.get("gates") if isinstance(profile, dict) else None
        gates = gates if isinstance(gates, dict) else {}
        return cls(
            adoption=float(det.get("adoption", 0.0)),
            advocacy=float(det.get("advocacy", 0.0)),
            craft=float(det.get("craft", 0.0)),
            amplification=float(det.get("amplification", 0.0)),
            advocacy_spread=float(det.get("advocacy_spread", 0.0)),
            reception=float(det.get("reception", 0.0)),
            persistence=float(det.get("persistence", 0.0)),
            cascade=float(det.get("cascade", 0.0)),
            autorater={k: float(v) for k, v in (rater or {}).items()},
            gate_broken_app=float(gates.get("broken_app", BROKEN_APP_MULTIPLIER)),
            gate_failed_self_check=float(
                gates.get("failed_self_check", FAILED_SELF_CHECK_MULTIPLIER)
            ),
            gate_policy=_gate_policy(gates.get("policy")),
            profile=str(profile.get("name") or name or "unnamed"),
        )


@dataclass(frozen=True)
class ViralScoreResult:
    """A scored run: the number, how it was built, and how much to trust it."""

    build_id: str
    crowd_dir: str
    score: float | None
    score_version: str
    components: dict
    weights: dict
    gate: float
    ci_low: float | None = None
    ci_high: float | None = None
    confidence: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    @property
    def scorable(self) -> bool:
        return self.score is not None

    def as_dict(self) -> dict:
        return asdict(self)


def _amplification(sig: RunSignals) -> float | None:
    """Fraction of the exposed audience that REPOSTED: putting your name on it.

    This used to be ``0.65*repost + 0.35*comment - negative``, which was three
    different things in a trenchcoat, and one of them pointed the wrong way.
    Measured over 27 calibration runs (3 builds x 3 crowd sizes x 3 seeds), the
    fraction of the audience that comments is *inverted* with respect to app
    quality:

    | signal | broken control | mid | good |
    |---|---|---|---|
    | comment participation | **0.868** | 0.839 | **0.750** |
    | dislike/report participation | 0.859 | 0.225 | 0.012 |
    | repost participation | 0.131 | 0.084 | 0.268 |

    People comment to complain. A term that rises as the app gets worse cannot
    sit inside a positively-weighted component, and the reason the old formula
    still separated the corpse was the ``- negative`` term -- i.e. it was a
    valence measure wearing an amplification label, and the floor clip at 0 hid
    how much work that term was doing.

    Valence now has its own component (:func:`_reception`) and amplification is
    what its name says. It is honest but noisy (reposts are rare and bursty), so
    it should carry a small weight, not the largest one in the score.
    """
    if sig.repost_participation is None:
        return None
    return max(0.0, min(1.0, sig.repost_participation))


def _reception(sig: RunSignals) -> float | None:
    """Net behavioural valence: liked minus disliked, per exposed agent.

    The crowd's *behaviour* rather than its stated intent, and the strongest
    behavioural discriminator in the corpus: dislike participation runs 0.859 /
    0.225 / 0.012 across broken / mid / good with clean separation.

    Mapped from net valence in [-1, 1] onto [0, 1] rather than clipped at zero,
    because clipping is what let the old amplification term look like it worked:
    every bad app pinned to 0.000 reads as perfect separation while destroying
    all resolution *among* bad apps.
    """
    if sig.like_participation is None and sig.negative_participation is None:
        return None
    net = (sig.like_participation or 0.0) - (sig.negative_participation or 0.0)
    return max(0.0, min(1.0, (net + 1.0) / 2.0))


def _cascade(sig: RunSignals) -> float | None:
    """Multi-generation spread: engagement earned by derived content."""
    if sig.secondary_share is None:
        return None
    return max(0.0, min(1.0, sig.secondary_share))


_LOG = logging.getLogger("viral_bench.score.viralscore")


def _in_unit(value: float | None, *, name: str) -> float | None:
    """Clamp a component to [0,1], loudly.

    Components are combined with weights that assume this range, so one term
    escaping it does not merely misreport that term -- it silently re-weights
    every other one, and the result still looks like a plausible score.

    The inputs are bounded by construction (rates are counts over counts, craft
    is an LLM-rated 0-10 mean) and across 4,954 stored verdicts this would never
    have fired. It exists because that 0-10 rubric is enforced by a docstring
    rather than a schema, and because a run is scored from JSON that may have
    been written by an older build with no clamping at the source. Silent
    correction would be worth less than a complaint: a WARNING in a scored run's
    log is findable later, a quietly-adjusted published number is not.
    """
    if value is None:
        return None
    if value < 0.0 or value > 1.0:
        _LOG.warning(
            "component %s=%r is outside [0,1] and was clamped; this run's signals "
            "are corrupt and the score should not be published as-is",
            name,
            value,
        )
        return min(1.0, max(0.0, value))
    return value


def compute_components(sig: RunSignals) -> dict:
    """The five components, each normalised to [0,1] (None when unmeasured)."""
    raw = {
        "adoption": sig.adoption_rate,
        "advocacy": sig.advocacy_rate,
        "craft": (sig.craft_mean / 10.0) if sig.craft_mean is not None else None,
        "amplification": _amplification(sig),
        "advocacy_spread": sig.advocate_amplification,
        "reception": _reception(sig),
        "persistence": sig.persistence_rate,
        "cascade": _cascade(sig),
    }
    return {name: _in_unit(v, name=name) for name, v in raw.items()}


def _confidence_warnings(sig: RunSignals, components: dict) -> list[str]:
    """Reasons to distrust this run. Empty means a clean, comparable run."""
    warnings: list[str] = []
    if not sig.run_ok:
        warnings.append("run did not complete cleanly")
    if not sig.rounds_ok:
        warnings.append("at least one simulation round failed")
    if sig.clamped:
        warnings.append(
            f"crowd clamped to {sig.actual_agents} of {sig.requested_agents} "
            "requested agents (persona pool too small)"
        )
    if sig.n_interviews < CALIBRATED_CROWD_SIZE:
        warnings.append(
            f"only {sig.n_interviews} interviews: below the calibrated crowd "
            f"size of {CALIBRATED_CROWD_SIZE}, where pooled run-to-run noise "
            "measured 3.5 points against 9.6 at 50 agents"
        )
    if sig.does_what_it_claims is None:
        warnings.append("no validity gate: the app was never verified to run")
    elif sig.does_what_it_claims is False:
        warnings.append(
            "app failed to build or start"
            if (sig.runs is False or sig.builds is False)
            else "app runs but failed its own smoke check "
            "(manifest contract violation, not a dead app)"
        )
    if sig.n_valid_trials == 0:
        warnings.append("no valid hands-on trials: craft is unmeasured")
    elif sig.n_valid_trials < sig.n_trials:
        warnings.append(
            f"only {sig.n_valid_trials} of {sig.n_trials} trials were valid "
            "(degraded or unfinished trials excluded)"
        )
    if sig.interview_mode == "per_agent":
        warnings.append("interview fell back to per-agent after a batch failure")
    missing = [k for k, v in components.items() if v is None]
    if missing:
        warnings.append(f"unmeasured components re-weighted out: {', '.join(missing)}")
    return warnings


def _weighted(components: dict, weights: ScoreWeights) -> float | None:
    """Weighted mean over the components that were measured.

    An unmeasured component is dropped and the remaining weights renormalised,
    so a missing signal never silently scores as zero.
    """
    w = weights.as_dict()
    pairs = [
        (components[k], w[k]) for k in w if components.get(k) is not None and w[k] > 0
    ]
    total = sum(weight for _, weight in pairs)
    if not pairs or total <= 0:
        return None
    return sum(value * weight for value, weight in pairs) / total


def _bootstrap_ci(
    sig: RunSignals,
    weights: ScoreWeights,
    gate: float,
    *,
    iterations: int = 400,
    seed: int = 0,
) -> tuple[float | None, float | None]:
    """A 95% interval for the sampling noise in the per-agent judgement terms.

    Resamples the interviewed crowd with replacement. This captures "how much of
    this score is down to *which* agents answered", which is the part that
    shrinks as the crowd grows. It does NOT capture run-to-run cascade variance
    -- that is what the repeated-run reliability study measures.
    """
    n = sig.n_interviews
    if n < 3 or sig.adoption_rate is None:
        return (None, None)
    rng = random.Random(seed)
    base = compute_components(sig)
    scores: list[float] = []
    for _ in range(iterations):
        # Binomial resampling of the two per-agent rates at the observed n.
        adoption = sum(rng.random() < sig.adoption_rate for _ in range(n)) / n
        advocacy = (
            sum(rng.random() < sig.advocacy_rate for _ in range(n)) / n
            if sig.advocacy_rate is not None
            else None
        )
        trial = dict(base, adoption=adoption, advocacy=advocacy)
        value = _weighted(trial, weights)
        if value is not None:
            scores.append(100.0 * gate * value)
    if not scores:
        return (None, None)
    scores.sort()
    lo = scores[int(0.025 * len(scores))]
    hi = scores[min(len(scores) - 1, int(0.975 * len(scores)))]
    return (round(lo, 1), round(hi, 1))


def validity_gate(sig: RunSignals, weights: ScoreWeights | None = None) -> float:
    """How much to discount the score for failing verification.

    Two failure modes stay apart, because "the app is dead" and "the app works
    but its self-written health check is wrong" are different things:

    * the app did not build or start -> the profile's ``broken_app`` floor.
    * it ran but failed its own smoke check -> its ``failed_self_check`` floor.
    * verified, or never verified at all -> no discount (an *unverified* app is
      not the same as a broken one, and is flagged in the confidence warnings
      instead of being silently punished).

    **How much of the floor bites depends on the policy.**

    ``hard`` (the original) applies the floor outright. That lets one container
    probe overrule thirty agents who used the app, and it measurably did: in one
    internal run a cell recorded the HIGHEST adoption and craft of its three
    seeds and was still cut by a factor of five. It was the only cell in the
    whole sweep where the gate flapped across seeds -- a flaky start, not a dead
    app.

    ``evidence_graded`` (default from score_version 1.7) interpolates from the
    floor up to 1.0 with :func:`witness_rate`, so the size of the discount is
    set by how much of the crowd could not get the app working rather than by a
    constant. An app nobody could open takes the whole floor, while one
    everybody used takes none of it, whatever the probe thinks.

    Chosen by sweep, not by taste (``scripts/gate_sweep.py``): 64
    policy x multiplier combinations, priced on both live profiles and on both
    halves of a brief-level split. Against the hard gate it halves
    self-separation (1.16 -> 0.51 on v4, 0.51 -> 0.33 on v5) and improves
    discrimination (4.64 -> 4.84, 4.69 -> 4.91) at the same control margin,
    and it beats removing the gate outright by a wide margin.

    One trap, recorded because it nearly shipped: a floor of exactly 0.0 tops
    the naive ranking and is an artifact. It sends dead runs to a hard zero,
    flattening 36 cells to zero variance, which shrinks the pooled noise and the
    null without resolving anything. Scored on the cells it has not flattened,
    its discrimination is 3.66 -- worse than the gate it replaces. The floor is
    therefore 0.1, and the sweep now reports that corrected figure by default.

    The floors stay in ``config/score.yaml`` so the gate remains in the search
    space.
    """
    weights = weights or ScoreWeights()
    if sig.does_what_it_claims is not False:
        return 1.0
    dead = sig.runs is False or sig.builds is False
    floor = weights.gate_broken_app if dead else weights.gate_failed_self_check
    if weights.gate_policy == GATE_HARD:
        return floor
    return floor + (1.0 - floor) * witness_rate(sig)


def witness_rate(sig: RunSignals) -> float:
    """Fraction of the exposed crowd that got the app working first-hand.

    A trial counts only if it finished, was not degraded, and the agent was
    affirmatively known to have reached the app (see ``signals.py``). So this is
    "how many people can testify that this thing runs", not "how many tried".
    """
    exposed = sig.exposed_agents or 0
    if exposed <= 0:
        return 0.0
    return max(0.0, min(1.0, sig.n_valid_trials / exposed))


def unscorable_reasons(sig: RunSignals, components: dict) -> list[str]:
    """Reasons this run must NOT produce a number at all.

    A missing component is normally re-weighted out, which is right for a small
    gap but catastrophic when the *primary* measurement is gone. An upstream
    recsys bug used to destroy every interview in runs that generated a lot of
    discussion. Those runs still reported ``ok: True`` and were still scored,
    silently, off craft and amplification alone -- and because craft is the most
    stable component, the resulting numbers looked *more* trustworthy than
    healthy runs. That is the worst possible failure mode for a benchmark, so
    losing the crowd's own verdicts is now fatal rather than cosmetic.
    """
    reasons: list[str] = []
    if sig.n_interviews == 0:
        reasons.append(
            "no interviews recorded: the crowd's own adoption/advocacy verdicts "
            "are the primary signal, and scoring without them silently reports a "
            "different metric under the same name"
        )
    elif sig.n_interviews < MIN_INTERVIEWS:
        # Same reasoning, but for a floor above zero (score.yaml minimums.
        # interviews). Kept separate so the total-loss case keeps its own,
        # more direct wording.
        reasons.append(
            f"only {sig.n_interviews} interviews, below the configured minimum "
            f"of {MIN_INTERVIEWS}: too little of the primary signal to score"
        )
    if components.get("adoption") is None and components.get("craft") is None:
        reasons.append("neither crowd judgement nor hands-on craft was measured")
    if not sig.run_ok:
        reasons.append("the simulation did not complete")
    return reasons


def score_run(
    sig: RunSignals,
    weights: ScoreWeights | None = None,
    autorating=None,
) -> ViralScoreResult:
    """Compute the ViralScore for one crowd run.

    ``autorating`` is an optional :class:`~viral_bench.score.autorater.AutoRating`
    whose dimensions are folded in alongside the deterministic components, using
    the weights from the active profile. Without it the score is purely
    deterministic and the autorater weights are re-normalised out, so a run that
    was never rated is not penalised for it.
    """
    weights = weights or ScoreWeights()
    weights.validate()

    components = compute_components(sig)
    rated = autorating is not None and getattr(autorating, "ok", False)
    if rated:
        # Clamped like every other component. These arrive AFTER
        # compute_components and so bypassed its range check entirely, and an
        # autorating loaded from stored JSON is not guaranteed to have been
        # written by a build that enforced the 0-10 rubric.
        components.update(
            {
                name: _in_unit(value, name=f"autorater.{name}")
                for name, value in autorating.normalized().items()
            }
        )
    elif weights.autorater:
        # The profile expects qualitative dimensions but none were supplied, so
        # their weight is re-normalised away. Say so: otherwise two runs scored
        # under the "same" profile could silently be using different formulas.
        for dim in weights.autorater:
            components.setdefault(dim, None)
    gate = validity_gate(sig, weights)
    blockers = unscorable_reasons(sig, components)
    raw = None if blockers else _weighted(components, weights)
    score = round(100.0 * gate * raw, 1) if raw is not None else None
    ci_low, ci_high = (
        _bootstrap_ci(sig, weights, gate) if raw is not None else (None, None)
    )

    return ViralScoreResult(
        build_id=sig.build_id,
        crowd_dir=sig.crowd_dir,
        score=score,
        score_version=SCORE_VERSION,
        components={
            k: (round(v, 4) if v is not None else None) for k, v in components.items()
        },
        weights=weights.as_dict(),
        gate=gate,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=blockers + _confidence_warnings(sig, components),
        diagnostics={
            "n_interviews": sig.n_interviews,
            "n_valid_trials": sig.n_valid_trials,
            "exposed_agents": sig.exposed_agents,
            "facets": sig.facets,
            "delight_mean": sig.delight_mean,
            "delight_stdev": sig.delight_stdev,
            "persistence_rate": sig.persistence_rate,
            "n_persistence_checked": sig.n_persistence_checked,
            "audience_fit_rate": sig.audience_fit_rate,
            "resonance_adoption": sig.resonance_adoption,
            "resonance_advocacy": sig.resonance_advocacy,
            "advocacy_unweighted": sig.advocacy_unweighted,
            "like_participation": sig.like_participation,
            "secondary_share": sig.secondary_share,
            "late_action_share": sig.late_action_share,
            "does_what_it_claims": sig.does_what_it_claims,
            "builds": sig.builds,
            "runs": sig.runs,
            "validity_detail": sig.validity_detail,
        },
    )
