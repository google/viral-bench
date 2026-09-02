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

"""Read the frozen fleet and its crowd runs into one comparable corpus.

Everything the loop needs to answer "does this instrument separate two founder
models?" is on disk: ``builds/fleet.json`` says which builds are in the frozen
fleet, each ``build.json`` says how that build went, and each crowd run's
``run_summary.json`` (plus an optional stored ``autorating.json``) re-scores for
free. This module turns that into a corpus and computes the statistics the
benchmark is judged on. It is pure and offline -- no LLM calls, no simulation.

**The headline statistic is a paired, per-idea gap in ViralScore points.**

Cohen's d over pooled runs is the wrong instrument here, for a reason that is
easy to miss and fatal if you do. Pooling every run of a model puts *idea
difficulty* in the denominator: a fleet spanning "sliding tile game" and "local
LLM runner" has enormous between-idea spread that has nothing to do with which
model built them. A profile can then improve d by **compressing every score
toward the middle** -- shrinking the denominator faster than the numerator --
which looks like better discrimination while actually destroying it.

Pairing removes that. Each idea is built by both models, so differencing within
an idea cancels idea difficulty exactly, and the mean of those differences is in
points -- the unit the question is actually asked in ("how many points better?"),
not a ratio that can be inflated by squeezing the scale.

Two guards travel with it, because a gap on its own proves nothing:

* :func:`self_separation` runs the *same* statistic on one model against itself
  by splitting its seeds. An instrument that separates a model from itself is
  measuring noise, and its model gap means nothing.
* :func:`control_separation` checks the deliberately broken app still lands far
  below every real build. An instrument that cannot see a corpse cannot be
  trusted to rank two working apps.
"""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass, field, replace
from pathlib import Path

from viral_bench.crowd.sim_defaults import CROWD_ARCH_VERSION
from viral_bench.score.signals import RUN_SUMMARY, extract_signals
from viral_bench.score.viralscore import ScoreWeights, score_run

#: Builds whose recorded model starts with this are controls, not founder-model
#: builds, and are excluded from every model comparison.
CONTROL_PREFIX = "control/"

#: Controls come in two kinds, and they answer opposite questions. A *negative*
#: control (``control/broken``) is a deliberately dead app: the instrument must
#: put it at the floor, which is what gate G3 checks. A *positive* control
#: (``control/fullstack-ok``) is a deliberately correct one: the instrument must
#: put it high, which is the only way to catch a crowd that has quietly stopped
#: liking anything. Pooling them would let a good control lift the floor and
#: satisfy the separation gate with no separation at all.
POSITIVE_CONTROL_MARKERS = ("-ok", "-good")


def control_kind(model: str) -> str:
    """``"negative"``, ``"positive"``, or ``""`` for a real founder build."""
    if not model.startswith(CONTROL_PREFIX):
        return ""
    tail = model[len(CONTROL_PREFIX) :]
    return (
        "positive"
        if any(marker in tail for marker in POSITIVE_CONTROL_MARKERS)
        else "negative"
    )


@dataclass(frozen=True)
class FleetBuild:
    """One founder build in the frozen fleet, as recorded in ``build.json``."""

    build_id: str
    idea_id: str
    model: str
    status: str
    rounds_run: int | None = None
    turns_spent: int | None = None
    shipped_early: bool | None = None
    qa_verified: bool | None = None
    build_seconds: float | None = None
    config: dict = field(default_factory=dict)
    #: True for a control build (broken OR deliberately correct), which is never
    #: a founder-model build and must never enter a model comparison.
    is_control: bool = False
    #: "negative" (must land at the floor), "positive" (must land high), or ""
    #: for a real founder build.
    control: str = ""
    #: Which named sweep this build belongs to, e.g. "r4" (see scripts/cohort.py).
    cohort: str = ""
    #: Which independent build of this (idea, model) pair this is. The fleet can
    #: be built more than once under the same config; comparing those replicates
    #: is the only way to separate "this model is better" from "this build got
    #: lucky". 0 means the build is not in the fleet index.
    replicate: int = 0
    #: How the founder chose to orchestrate, in the dynamic arm only (``None``
    #: elsewhere -- the other arms have no such choice to record). Carried up to
    #: the scoring layer because in that arm the orchestration is half the
    #: result: "did the model that scored higher also delegate more, or
    #: differently?" is the question the arm exists to answer, and it cannot be
    #: asked from a build.json the corpus never reads.
    orchestration: dict | None = None

    #: Statuses whose builds must LEAVE the denominator rather than sit at the
    #: floor.
    #:
    #: Only ``provider_refusal`` so far, and the distinction from
    #: ``manifest_missing`` is the whole point. A model that ran its turns and
    #: shipped no launch contract IS a result -- it was asked to found an app and
    #: did not -- so it is floored, and 16 of the solo arm's cells are exactly
    #: that, concentrated in the weakest models. A model whose output a provider
    #: safety filter blocked produced no evidence either way; flooring it would
    #: confound provider POLICY with model CAPABILITY, and asymmetrically, since
    #: whichever provider filters hardest would lose the most score.
    UNSCORABLE_STATUSES = frozenset({"provider_refusal"})

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def unscorable(self) -> bool:
        """True if this build is excluded from scoring rather than floored."""
        return self.status in self.UNSCORABLE_STATUSES

    @property
    def control_kind(self) -> str:
        """This build's control kind, defaulting an unlabelled control to negative.

        Every control that existed before positive ones did was a corpse, and a
        control silently belonging to NEITHER kind would vanish from the
        separation gate -- which would make the gate pass by having nothing to
        measure. Defaulting is the safe direction: a mislabelled positive
        control makes G3 harder, never easier.
        """
        return self.control or ("negative" if self.is_control else "")


@dataclass(frozen=True)
class FleetSpec:
    """Which builds constitute *the fleet under test* right now.

    ``builds/fleet.json`` is append-only and now holds several eras: two
    replicates of a 4-agent team fleet, a Workspace arm, a solo arm, and two
    different model pairs. Every one of those is a legitimate past experiment
    and none of them may be pooled with the current one -- so "the fleet" has to
    be a named, explicit subset rather than "whatever the index contains".

    A build qualifies when its recorded founder configuration matches
    ``structure``, its fleet-index replicate matches ``replicate``, and its model
    is one of the two under comparison. That is checked against each
    ``build.json``, so a build made under a different shape can never drift into
    the comparison by being listed in the index.
    """

    model_a: str
    model_b: str
    structure: str
    replicate: int
    #: Additional models swept under the SAME founder configuration. The paired
    #: headline stays (model_a, model_b) -- a paired statistic needs two columns
    #: -- while every model here is built, scored and reported alongside them.
    extra_models: tuple[str, ...] = ()

    @property
    def models(self) -> tuple[str, ...]:
        return (self.model_a, self.model_b, *self.extra_models)

    @property
    def short_models(self) -> tuple[str, ...]:
        return tuple(_short_model(m) for m in self.models)

    def wants(self, build: FleetBuild) -> bool:
        return (
            _short_model(build.model) in set(self.short_models)
            and structure_name(build.config) == self.structure
            and build.replicate == self.replicate
        )


#: The fleet a scored comparison covers: every idea built by each named model,
#: with the founder configuration and replicate pinned so two runs of the same
#: cell are genuinely the same experiment.
#:
#: Empty by default. A fleet is a description of YOUR run -- fill it in (or pass
#: ``--models``) rather than inheriting someone else's model pair.
CURRENT_FLEET = FleetSpec(
    model_a="",
    model_b="",
    structure="solo",
    replicate=1,
    extra_models=(),
)


#: Cached idea lookup for brief-currency checks.
_IDEA_CACHE: dict | None = None


def structure_name(config: dict) -> str:
    """Name the founder configuration from a build record's config.

    Mirrors scripts/build_fleet.py STRUCTURES. Kept here because the scoring
    layer must be able to separate arms without importing a script.

    The dynamic arm is recognised by its recorded name rather than derived from
    ``(n_agents, collab)`` like the other three, and it has to be: it runs one
    orchestrator process, so it derives as ``solo`` and would be pooled with the
    solo baseline -- two different experiments scored as one cell, which is the
    exact failure this function exists to prevent.
    """
    if config.get("structure") == "dynamic":
        return "dynamic"
    agents = config.get("n_agents")
    collab = config.get("collab")
    if agents == 1:
        return "solo"
    if agents == 4 and collab == "local":
        return "team"
    return ""


@dataclass(frozen=True)
class ScoredRun:
    """One crowd run over one build, re-scored from artifacts on disk."""

    crowd_dir: str
    build_id: str
    idea_id: str
    model: str
    seed: int
    n_agents: int
    score: float | None
    #: Which crowd instrument produced this run. Runs from two architectures are
    #: not comparable, so the corpus keeps only one version at a time.
    arch_version: str = "0"
    #: The validity-gate multiplier the score was computed with: 1.0 means the
    #: app was verified (or unverified), below 1.0 means it was measured NOT to
    #: do what it claims.
    gate: float = 1.0
    #: The app did not build or did not start. Distinct from "runs but fails its
    #: own smoke check", which is a manifest-contract violation by an app agents
    #: nonetheless used successfully. The score already separates the two at 0.2x
    #: and 0.6x; anything reading the gate must keep them apart too.
    dead: bool = False
    #: The founder shipped no usable ``viralbench.json``, so the crowd had
    #: nothing to launch. A stricter statement than ``dead``: the app may be
    #: perfectly good code, but no user -- and no harness -- can start it.
    undeliverable: bool = False
    #: Who is at fault when ``app_start_failed``: "harness", "app", or "".
    #:
    #: Only a HARNESS fault is excluded. An app that will not start because of its
    #: own defect is floored, not dropped -- see ``scorable``.
    app_start_fault: str = ""
    #: The app could not be started, so no agent ever saw it.
    #:
    #: Deliberately not the same thing as ``dead``. ``dead`` is a verdict about the
    #: app -- we ran it and it does not work, so it is capped at the broken-app
    #: floor. This is a verdict about US: the dominant cause on the r3 sweep was
    #: materializing the app from its shipped git branch without the dependencies
    #: its own .gitignore excludes, and one such app, served by hand, returned
    #: HTTP 200 in 55 ms and rendered 1,216 DOM nodes with zero console errors.
    #: Scoring a model down for that manufactures a capability difference.
    #:
    #: So these runs are EXCLUDED from model means and reported as a coverage
    #: caveat, the same treatment ``harness_failed`` gets on the build side.
    #: Measured on r3 the choice does not change any arm's ranking (team 54.0 vs
    #: 52.1 floored, and the arm order is identical either way), which is what
    #: makes it a reporting decision rather than a thumb on the scale.
    app_start_failed: bool = False
    #: Which named sweep this run belongs to, e.g. "r4". Read from the stored
    #: summary, so a run says which corpus it is part of without being joined
    #: back to a build list that may since have moved.
    cohort: str = ""
    #: Which build of this cell produced the run (see ``FleetBuild.replicate``).
    replicate: int = 0
    #: Which founder configuration built the app: "solo", "team", "dynamic"
    #: (or "" when unknown). A cell is (idea, model, STRUCTURE): the
    #: same idea built by one agent and by a four-agent team are different
    #: experiments, and pooling them turns a real between-structure difference
    #: into apparent seed noise.
    structure: str = ""
    blockers: list[str] = field(default_factory=list)
    components: dict = field(default_factory=dict)

    @property
    def gated(self) -> bool:
        """True if this run's app failed verification (dead OR bad self-check)."""
        return self.gate < 1.0

    @property
    def scorable(self) -> bool:
        """Whether this run may enter a model's mean.

        A start failure disqualifies a run ONLY when the harness is at fault.
        That is the narrow case where the number would describe our packaging
        rather than the model: the app was materialized without its dependencies,
        or run under a toolchain the founder never had, and scoring a model down
        for it manufactures a capability difference.

        An app that will not start because of its OWN defect -- a syntax error in
        the shipped source, a manifest naming a file that was never committed --
        stays in and is floored. The crowd did everything right, nobody could use
        the thing, and noticing that is the entire point of the benchmark. It
        needs no special arithmetic: the validity gate already sees
        ``runs=False``, and with a witness rate of zero it lands on the
        broken-app floor of 0.1x, applied to components the interviews still
        produce. Measured on floored runs already on disk: 0.0-0.1 out of 100.

        Excluding here rather than at each call site is deliberate. The seed-0
        pass reported "complete" while 26 builds had produced nothing, and the way
        that happened was a hole being invisible to whichever aggregation forgot
        to filter. ``FleetCorpus.app_start_failures()`` keeps it visible.
        """
        if self.score is None:
            return False
        # Scorable only when the app is AFFIRMATIVELY at fault. An unattributed
        # start failure -- a run written before the attribution existed, or a
        # reason nothing could classify -- is excluded, because the safe default
        # is to leave a build out of a model's mean rather than to floor one
        # nobody has judged.
        return not (self.app_start_failed and self.app_start_fault != "app")


@dataclass
class PairedGap:
    """A paired per-idea difference between two models, in ViralScore points."""

    model_a: str
    model_b: str
    n_ideas: int
    gap: float | None
    ci_low: float | None
    ci_high: float | None
    wins_a: int
    wins_b: int
    per_idea: dict[str, float] = field(default_factory=dict)

    @property
    def excludes_zero(self) -> bool:
        if self.ci_low is None or self.ci_high is None:
            return False
        return self.ci_low > 0.0 or self.ci_high < 0.0


def _short_model(model: str) -> str:
    """Strip any provider prefix so 'google-vertex/x' and 'x' compare equal."""
    return model.rsplit("/", 1)[-1]


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _build_seconds(record: dict) -> float | None:
    phases = record.get("phases")
    if not isinstance(phases, list) or not phases:
        return None
    total = 0.0
    for phase in phases:
        if isinstance(phase, dict) and isinstance(phase.get("duration_s"), int | float):
            total += float(phase["duration_s"])
    return round(total, 1) or None


def load_builds(builds_root: Path) -> dict[str, FleetBuild]:
    """Every build under ``builds/work`` that has a readable record."""
    work = builds_root / "work"
    out: dict[str, FleetBuild] = {}
    if not work.is_dir():
        return out
    for child in sorted(work.iterdir()):
        record = _read_json(child / "build.json")
        if not record or not record.get("build_id"):
            continue
        raw_model = record.get("model", "")
        out[record["build_id"]] = FleetBuild(
            build_id=record["build_id"],
            idea_id=record.get("idea_id", ""),
            model=_short_model(raw_model),
            status=record.get("status", "unknown"),
            rounds_run=record.get("rounds_run"),
            turns_spent=record.get("turns_spent"),
            shipped_early=record.get("shipped_early"),
            qa_verified=record.get("qa_verified"),
            build_seconds=_build_seconds(record),
            config={
                "structure": record.get("structure"),
                "n_agents": record.get("n_agents"),
                "collab": record.get("collab"),
                "max_rounds": record.get("max_rounds"),
                "min_rounds": record.get("min_rounds"),
                "max_turns": record.get("max_turns"),
            },
            orchestration=(
                record.get("orchestration")
                if isinstance(record.get("orchestration"), dict)
                else None
            ),
            cohort=str(record.get("cohort") or ""),
            is_control=raw_model.startswith(CONTROL_PREFIX),
            control=control_kind(raw_model),
        )
    return out


def fleet_replicates(builds_root: Path) -> dict[str, int]:
    """Map each fleet build id to its replicate number.

    Read once from ``builds/fleet.json`` rather than looked up per build: the
    obvious implementation scans every entry for every build and is quadratic,
    which is invisible at 100 builds and silly at 1,000.
    """
    index = _read_json(builds_root / "fleet.json") or {}
    entries = index.get("entries")
    if not isinstance(entries, dict):
        return {}
    return {
        e["build_id"]: int(e.get("replicate", 1) or 1)
        for e in entries.values()
        if isinstance(e, dict) and e.get("build_id")
    }


def fleet_statuses(builds_root: Path) -> dict[str, str]:
    """Map build id -> the status recorded in the FLEET INDEX.

    ``build.json`` carries the status the harness assigned while the build was
    running; ``fleet.json`` carries the verdict for the cell, which can be revised
    afterwards -- a provider refusal is only recognisable once its error text is
    classified, and ``--reclassify-infra`` does that retroactively. Where the two
    disagree the index is right, because it is the later and better-informed of
    the two.

    Kept beside ``fleet_replicates`` and applied the same way, since ``replicate``
    has exactly this shape: a fact about the cell that the build record cannot
    know on its own.
    """
    index = _read_json(builds_root / "fleet.json") or {}
    entries = index.get("entries")
    if not isinstance(entries, dict):
        return {}
    return {
        e["build_id"]: str(e.get("status") or "")
        for e in entries.values()
        if isinstance(e, dict) and e.get("build_id") and e.get("status")
    }


def apply_fleet_status(
    builds: dict[str, FleetBuild], builds_root: Path
) -> dict[str, FleetBuild]:
    """Overlay the index's verdict onto each build, in place-ish.

    Only for statuses the build record cannot self-assign. Everything else keeps
    what the harness wrote, so this cannot quietly rewrite an ordinary outcome.
    """
    for build_id, status in fleet_statuses(builds_root).items():
        if build_id in builds and status in FleetBuild.UNSCORABLE_STATUSES:
            builds[build_id] = replace(builds[build_id], status=status)
    return builds


def fleet_build_ids(builds_root: Path) -> set[str]:
    """Build ids listed in ``builds/fleet.json`` (empty if there is no fleet)."""
    return set(fleet_replicates(builds_root))


def _load_autorating(crowd_dir: Path):
    """Re-hydrate a stored autorating, if the run has one. Never calls an LLM."""
    data = _read_json(crowd_dir / "autorating.json")
    if not isinstance(data, dict) or not isinstance(data.get("dimensions"), dict):
        return None
    from viral_bench.score.autorater import AutoRating, DimensionRating, _coerce

    dims = {}
    for name, value in data["dimensions"].items():
        if isinstance(value, dict) and value.get("score") is not None:
            # Clamped to the 0-10 rubric, like every other way a rating enters
            # the score. This is a THIRD reader of stored autorater JSON (after
            # the live path and AutoRating.from_dict) and it had the same bare
            # float(): normalized() divides by 10 straight into a weighted term
            # that must be in [0,1].
            score = _coerce(value["score"])
            if score is None:
                continue
            dims[name] = DimensionRating(
                score=score,
                samples=[float(s) for s in value.get("samples", [])],
                spread=float(value.get("spread", 0.0)),
                evidence=list(value.get("evidence", [])),
                reason=str(value.get("reason", "")),
            )
    if not dims:
        return None
    return AutoRating(
        build_id=str(data.get("build_id", "")),
        model=str(data.get("model", "")),
        repeats=int(data.get("repeats", 0)),
        dimensions=dims,
    )


def score_corpus(
    builds_root: Path,
    builds: dict[str, FleetBuild],
    weights: ScoreWeights | None = None,
) -> list[ScoredRun]:
    """Re-score every crowd run on disk whose build we know about.

    A run that cannot be scored is kept with ``score=None`` and its blockers, so
    missing data is visible rather than quietly dropped from the denominator.
    """
    weights = weights or ScoreWeights.from_profile()
    runs: list[ScoredRun] = []
    for summary_path in sorted(builds_root.glob(f"*/*/{RUN_SUMMARY}")):
        crowd_dir = summary_path.parent
        summary = _read_json(summary_path)
        if not summary:
            continue
        build_id = summary.get("build_id", "")
        build = builds.get(build_id)
        if build is None:
            continue
        config = summary.get("config") or {}
        try:
            signals = extract_signals(crowd_dir)
            result = score_run(signals, weights, _load_autorating(crowd_dir))
        except (FileNotFoundError, ValueError, KeyError):
            continue
        runs.append(
            ScoredRun(
                crowd_dir=str(crowd_dir),
                build_id=build_id,
                idea_id=build.idea_id,
                model=build.model,
                seed=int(config.get("seed", 0) or 0),
                n_agents=int(config.get("n_agents", 0) or 0),
                arch_version=str(summary.get("crowd_arch_version", "0")),
                gate=float(result.gate),
                dead=signals.builds is False or signals.runs is False,
                undeliverable=bool(summary.get("undeliverable")),
                app_start_failed=bool(summary.get("app_start_failed")),
                # Default "harness" so a run recorded before the attribution
                # existed keeps the old, conservative treatment rather than being
                # silently promoted into the denominator.
                app_start_fault=str(summary.get("app_start_fault") or "harness")
                if summary.get("app_start_failed")
                else "",
                cohort=str(summary.get("cohort") or ""),
                replicate=build.replicate,
                # From the BUILD's recorded config, not the crowd summary: the
                # crowd's n_agents is the crowd size, not the founder team's.
                structure=structure_name(build.config),
                score=result.score,
                blockers=list(result.confidence),
                components=dict(result.components),
            )
        )
    return runs


@dataclass(frozen=True)
class BuildCoverage:
    """How complete an arm is, counted in BUILDS rather than in models.

    Carries ``uncovered_ids`` because "99.2% covered" is not actionable and eight
    named builds are. Every completeness claim in a results document should be
    able to name what it is missing, or it is not a completeness claim.
    """

    structure: str
    builds: int
    covered: int
    runs: int
    #: seed-count -> how many builds have exactly that many scored seeds.
    seed_histogram: dict[int, int] = field(default_factory=dict)
    #: Builds with NO scored run whose failure we have attributed to the harness.
    app_start_failed: int = 0
    #: Builds excluded from the denominator entirely (a provider refused them).
    refused: int = 0
    uncovered_ids: list[str] = field(default_factory=list)

    @property
    def fraction(self) -> float:
        return self.covered / self.builds if self.builds else 0.0

    @property
    def complete(self) -> bool:
        return self.builds > 0 and self.covered == self.builds

    def at_least(self, seeds: int) -> int:
        """How many builds reached ``seeds`` scored seeds or more."""
        return sum(n for depth, n in self.seed_histogram.items() if depth >= seeds)

    def summary(self) -> str:
        depths = ", ".join(
            f"{depth}:{self.seed_histogram[depth]}"
            for depth in sorted(self.seed_histogram)
        )
        tail = (
            f", {self.app_start_failed} app_start_failed"
            if self.app_start_failed
            else ""
        )
        if self.refused:
            tail += f", {self.refused} excluded (provider refusal)"
        return (
            f"{self.structure or 'fleet'}: {self.covered}/{self.builds} builds "
            f"({100 * self.fraction:.1f}%), {self.runs} runs, seeds[{depths}]{tail}"
        )


@dataclass
class FleetCorpus:
    """The frozen fleet plus every crowd run scored over it.

    ``arch_version`` filters the corpus to ONE crowd architecture. Re-scoring is
    free, so a formula change can be applied retroactively to every run -- but a
    change to the crowd itself cannot, and pooling runs from before and after it
    would mean comparing two models through two different instruments.
    """

    builds: dict[str, FleetBuild]
    runs: list[ScoredRun]
    fleet_ids: set[str]
    profile: str = ""
    arch_version: str = CROWD_ARCH_VERSION

    def _current(self, runs: list[ScoredRun]) -> list[ScoredRun]:
        return [r for r in runs if r.arch_version == self.arch_version]

    def fleet_builds(self) -> list[FleetBuild]:
        return [b for bid, b in self.builds.items() if bid in self.fleet_ids]

    def fleet_runs(self, *, scorable_only: bool = True) -> list[ScoredRun]:
        # A refused build is excluded even where runs exist for it. Belt and
        # braces: the sweep skips those cells, but a run made before the status
        # was assigned would otherwise be scored at the floor, which is the exact
        # outcome the exclusion exists to avoid.
        refused = {b.build_id for b in self.builds.values() if b.unscorable}
        return self._current(
            [
                r
                for r in self.runs
                if r.build_id in self.fleet_ids
                and r.build_id not in refused
                and (r.scorable or not scorable_only)
            ]
        )

    def control_runs(
        self, *, scorable_only: bool = True, kind: str = "negative"
    ) -> list[ScoredRun]:
        """Runs of a control build of one kind (default: the broken one).

        Defaulting to "negative" keeps every existing caller meaning what it
        meant. A positive control must never be pooled in here: it would raise
        the floor the separation gate measures against, and a gate that gets
        easier when you add a GOOD app to the corpus is not a gate.
        """
        return self._current(
            [
                r
                for r in self.runs
                if self.builds[r.build_id].control_kind == kind
                and (r.scorable or not scorable_only)
            ]
        )

    def restrict(self, *, replicate: int) -> FleetCorpus:
        """A view of this corpus containing only one replicate of the fleet.

        Lets every existing statistic -- the paired gap, the nulls, the control
        check -- be re-run on a single fleet without reimplementing any of them,
        which is what makes "does the gap replicate?" a two-line question.
        """
        keep = {b.build_id for b in self.builds.values() if b.replicate == replicate}
        return FleetCorpus(
            builds=self.builds,
            runs=[r for r in self.runs if r.build_id in keep],
            fleet_ids=self.fleet_ids & keep,
            profile=self.profile,
            arch_version=self.arch_version,
        )

    def replicates(self) -> list[int]:
        """Replicate numbers present in the fleet, ascending."""
        return sorted({b.replicate for b in self.fleet_builds() if b.replicate})

    def stale_runs(self) -> list[ScoredRun]:
        """Runs produced by an older crowd architecture, excluded from the corpus."""
        return [r for r in self.runs if r.arch_version != self.arch_version]

    def cells(self) -> dict[tuple[str, str, str], list[ScoredRun]]:
        """Scorable fleet runs grouped by (idea_id, model, structure).

        Structure is part of the key. It was not, and once the fleet gained a
        structure axis that silently pooled a solo build, a team build and a
        Workspace build of the same idea into ONE cell -- so a genuine
        between-structure difference was counted as within-cell noise.
        Measured: within_cell_sd read 13.73 while the true seed-to-seed
        variation is about 3.5, which would have made every paired comparison
        look four times noisier than it is.
        """
        out: dict[tuple[str, str, str], list[ScoredRun]] = {}
        for run in self.fleet_runs():
            out.setdefault((run.idea_id, run.model, run.structure), []).append(run)
        return out

    def refusals(self) -> list[FleetBuild]:
        """Fleet builds excluded because a provider refused to produce them.

        Surfaced rather than merely dropped, for the reason every exclusion here
        is: an invisible hole is how a report came to claim "complete, all 10
        models x 4 pipelines" over a corpus missing 26 builds.
        """
        return [b for b in self.fleet_builds() if b.unscorable]

    def app_start_failures(self, *, fault: str = "") -> list[ScoredRun]:
        """Fleet runs whose app never started, optionally narrowed by fault.

        Both kinds are reported. Only the harness-fault ones are excluded from
        scoring, but an app that could not start is worth naming either way --
        floored or dropped, it is a build no user could have used.
        """
        return [
            r
            for r in self._current(self.runs)
            if r.build_id in self.fleet_ids
            and r.app_start_failed
            and (not fault or r.app_start_fault == fault)
        ]

    def build_coverage(self, *, structure: str = "") -> BuildCoverage:
        """How many FLEET BUILDS have a scored run, and at how many seeds.

        THE GRAIN IS THE POINT. Coverage used to be asserted at model x arm, and
        at that grain the seed-0 pass was genuinely complete: all ten models had
        runs in all four arms. It was also missing 26 of 1,000 builds, because a
        model's mean is taken over the runs that exist and a hole simply drops out
        of it rather than showing up as one. The report went out saying "complete,
        all 10 models x 4 pipelines", and that claim was true and misleading at
        once.

        A build is covered when at least one of its runs is scorable. Seeds are
        counted per build so a table can state its own depth instead of implying
        the target seed count was reached everywhere.
        """
        every = [
            b
            for b in self.fleet_builds()
            if not structure or structure_name(b.config) == structure
        ]
        # An excluded build is not a gap we still owe: no amount of sweeping will
        # ever produce a score for it. Held out of the denominator so completeness
        # is reachable, and counted so the exclusion stays visible.
        refused = [b for b in every if b.unscorable]
        builds = [b for b in every if not b.unscorable]
        seeds_by_build: dict[str, set[int]] = {}
        for run in self.fleet_runs():
            if structure and run.structure != structure:
                continue
            seeds_by_build.setdefault(run.build_id, set()).add(run.seed)
        covered = [b for b in builds if seeds_by_build.get(b.build_id)]
        # No fault filter: this is intersected with the UNCOVERED builds below,
        # and a build whose app is at fault is scorable and therefore covered, so
        # it drops out on its own. Filtering here as well would undercount the
        # excluded ones, which are the only kind that leave a hole.
        failed = {
            r.build_id
            for r in self.app_start_failures()
            if not structure or r.structure == structure
        }
        histogram: dict[int, int] = {}
        for build in builds:
            depth = len(seeds_by_build.get(build.build_id, ()))
            histogram[depth] = histogram.get(depth, 0) + 1
        return BuildCoverage(
            structure=structure,
            builds=len(builds),
            refused=len(refused),
            covered=len(covered),
            runs=sum(len(s) for s in seeds_by_build.values()),
            seed_histogram=histogram,
            app_start_failed=len(failed - set(seeds_by_build)),
            uncovered_ids=sorted(
                b.build_id for b in builds if not seeds_by_build.get(b.build_id)
            ),
        )

    def cells_pooling_structures(self) -> dict[tuple[str, str], list[ScoredRun]]:
        """Runs grouped by (idea_id, model), POOLING founder structures.

        Only for questions that are genuinely about coverage rather than about
        effect size -- "does this cell have enough seeds yet?". Do NOT use it to
        compare scores: pooling a solo build with a team build of the same idea
        counts a real between-structure difference as within-cell noise, which is
        the bug that made within_cell_sd read 13.3 against a true value near 3.5.
        The name is deliberately awkward so that using it is a decision.
        """
        out: dict[tuple[str, str], list[ScoredRun]] = {}
        for run in self.fleet_runs():
            out.setdefault((run.idea_id, run.model), []).append(run)
        return out


def _has_current_brief(builds_root: Path, build_id: str) -> bool:
    """True if this build was given the brief the corpus currently defines."""
    from viral_bench.founder.prompts import brief_fingerprint
    from viral_bench.ideas import load_ideas

    global _IDEA_CACHE
    if _IDEA_CACHE is None:
        _IDEA_CACHE = {i.idea_id: i for i in load_ideas()}
    record = _read_json(builds_root / "work" / build_id / "build.json")
    if not record:
        return False
    stored = record.get("brief_fingerprint") or ""
    idea = _IDEA_CACHE.get(record.get("idea_id"))
    return bool(stored) and idea is not None and stored == brief_fingerprint(idea)


def load_corpus(
    repo_root: Path | str = ".",
    weights: ScoreWeights | None = None,
    *,
    arch_version: str = CROWD_ARCH_VERSION,
    current_brief_only: bool = False,
    spec: FleetSpec | None = None,
) -> FleetCorpus:
    """Load and score the whole corpus from ``<repo_root>/builds``.

    ``current_brief_only`` keeps only builds given the brief the corpus defines
    NOW. It defaults to False so that loading stays a faithful read of what is on
    disk -- filtering is an ANALYSIS policy, and a loader that silently drops
    data is a bad default. Every analysis script passes True.

    ``fleet.json`` accumulates every entry the index has ever
    held, so without it the corpus mixes eras: measured here, 163 fleet builds of
    which only 75 were post-pivot, the other 88 having been told to "prefer plain
    static files" and that "no backend required" -- the opposite of the current
    brief, plus 50 builds of a model no longer under test.

    Two runs from different eras are not comparable, and pooling them does not
    merely add noise, it adds *structured* noise: it inflated ``within_cell_sd``
    to 13.3 against a true seed-to-seed variation near 3.5, which would make
    every paired comparison look four times noisier than it is and hide real
    differences behind manufactured error bars.

    Pass False to analyse an older fleet on its own terms (the 2.5-vs-3.6
    comparison, for instance, is legitimately about pre-pivot builds).
    """
    builds_root = Path(repo_root) / "builds"
    weights = weights or ScoreWeights.from_profile()
    builds = apply_fleet_status(load_builds(builds_root), builds_root)
    replicates = fleet_replicates(builds_root)
    for build_id, replicate in replicates.items():
        if build_id in builds:
            builds[build_id] = replace(builds[build_id], replicate=replicate)
    fleet_ids = set(replicates)
    if current_brief_only:
        fleet_ids = {b for b in fleet_ids if _has_current_brief(builds_root, b)}
    if spec is not None:
        # Only the FLEET is narrowed, never ``runs``: the negative control is not
        # a fleet build and must survive the filter, or control_separation loses
        # the one measurement that proves the instrument can see a corpse.
        fleet_ids = {b for b in fleet_ids if b in builds and spec.wants(builds[b])}
    return FleetCorpus(
        builds=builds,
        runs=score_corpus(builds_root, builds, weights),
        fleet_ids=fleet_ids,
        profile=weights.profile,
        arch_version=arch_version,
    )


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def _bootstrap_ci(
    values: list[float], *, iterations: int = 4000, seed: int = 0
) -> tuple[float | None, float | None]:
    """Percentile bootstrap over the per-idea differences (resamples IDEAS).

    Resampling ideas -- not runs -- is the point: the uncertainty that matters is
    "would this gap survive a different draw of 25 product ideas?", not "would it
    survive another seed of the same idea".
    """
    if len(values) < 3:
        return (None, None)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        statistics.fmean(rng.choice(values) for _ in range(n))
        for _ in range(iterations)
    )
    return (
        round(means[int(0.025 * iterations)], 2),
        round(means[min(iterations - 1, int(0.975 * iterations))], 2),
    )


def paired_gap(
    corpus: FleetCorpus, model_a: str, model_b: str, *, seed: int = 0
) -> PairedGap:
    """Mean per-idea (A - B) ViralScore gap over ideas both models built."""
    cells = corpus.cells()
    # Pair on (idea, structure), not idea alone. A cell key now carries the
    # founder configuration, and comparing model A's solo build against model
    # B's team build would attribute a structure difference to the model.
    keys_a = {(idea, st) for (idea, model, st) in cells if model == model_a}
    keys_b = {(idea, st) for (idea, model, st) in cells if model == model_b}
    shared = sorted(keys_a & keys_b)
    per_pair: dict[str, float] = {}
    for idea, st in shared:
        a = statistics.fmean(r.score for r in cells[(idea, model_a, st)])
        b = statistics.fmean(r.score for r in cells[(idea, model_b, st)])
        per_pair[f"{idea}[{st}]" if st else idea] = round(a - b, 2)
    per_idea = per_pair
    deltas = list(per_idea.values())
    low, high = _bootstrap_ci(deltas, seed=seed)
    return PairedGap(
        model_a=model_a,
        model_b=model_b,
        n_ideas=len(deltas),
        gap=round(statistics.fmean(deltas), 2) if deltas else None,
        ci_low=low,
        ci_high=high,
        wins_a=sum(1 for d in deltas if d > 0),
        wins_b=sum(1 for d in deltas if d < 0),
        per_idea=per_idea,
    )


def self_separation(corpus: FleetCorpus, model: str, *, seed: int = 0) -> PairedGap:
    """The same paired statistic run on ONE model against itself.

    Seeds within a cell are split by parity into two pseudo-"models". Any gap
    here is pure instrument noise, so it is the yardstick the real gap has to
    beat: a model gap of 4 points means little if the instrument separates a
    model from itself by 3.
    """
    cells = corpus.cells()
    per_idea: dict[str, float] = {}
    for (idea, cell_model, structure), runs in sorted(cells.items()):
        if cell_model != model or len(runs) < 2:
            continue
        label = f"{idea}[{structure}]" if structure else idea
        ordered = sorted(runs, key=lambda r: (r.seed, r.crowd_dir))
        first = [r.score for r in ordered[0::2]]
        second = [r.score for r in ordered[1::2]]
        if not first or not second:
            continue
        per_idea[label] = round(statistics.fmean(first) - statistics.fmean(second), 2)
    deltas = list(per_idea.values())
    low, high = _bootstrap_ci(deltas, seed=seed)
    return PairedGap(
        model_a=f"{model}/even-seeds",
        model_b=f"{model}/odd-seeds",
        n_ideas=len(deltas),
        gap=round(statistics.fmean(deltas), 2) if deltas else None,
        ci_low=low,
        ci_high=high,
        wins_a=sum(1 for d in deltas if d > 0),
        wins_b=sum(1 for d in deltas if d < 0),
        per_idea=per_idea,
    )


def within_cell_sd(corpus: FleetCorpus) -> float | None:
    """Pooled seed-to-seed SD inside (idea, model) cells: the noise floor."""
    variances: list[float] = []
    for runs in corpus.cells().values():
        if len(runs) > 1:
            variances.append(statistics.variance([r.score for r in runs]))
    if not variances:
        return None
    return round((sum(variances) / len(variances)) ** 0.5, 2)


@dataclass
class ControlSeparation:
    """How far the deliberately broken app sits below every *working* build."""

    n_control_runs: int
    control_mean: float | None
    control_max: float | None
    weakest_real_cell: str = ""
    weakest_real_mean: float | None = None
    margin: float | None = None
    #: Real (idea, model) cells whose app does not build or start. These are
    #: corpses too, so they are excluded from the comparison -- and bounded
    #: separately.
    dead_cells: dict[str, float] = field(default_factory=dict)
    #: Highest score any DEAD real build achieved. If an app that does not run
    #: can outscore one that does, the gate is not doing its job.
    dead_max: float | None = None
    #: Median of the working cells: the bulk the control has to be far below.
    working_median: float | None = None
    #: Working cells scoring at or below the control's mean, and how many there
    #: are in total. Some real builds genuinely are worse than a deliberately
    #: broken one, so this is a bounded exception count rather than zero.
    below_control: list[str] = field(default_factory=list)
    n_working_cells: int = 0


def control_separation(corpus: FleetCorpus) -> ControlSeparation:
    """Compare the broken control against the weakest **working** real cell.

    "Far below every real build" cannot mean *every* build, because some real
    builds are themselves dead -- one model shipped an app that does not start,
    and it correctly scores 7.4, right next to the control. A check that reads
    that as instrument failure would be punishing the score for being right.

    Dead means ``builds is False or runs is False``, NOT the softer "runs but
    fails its own smoke check". The score already separates those at 0.2x and
    0.6x because they are different failures: a self-check violation is an app
    the crowd used successfully whose author wrote the wrong health command, and
    it belongs in the working set. Conflating them (which the first version of
    this check did) flagged a 39.4-scoring, well-liked build as a corpse
    outscoring working ones.
    """
    control = [r.score for r in corpus.control_runs()]
    cells = corpus.cells()
    working: dict[str, float] = {}
    dead: dict[str, float] = {}
    for (idea, model, structure), runs in cells.items():
        if not runs:
            continue
        mean = statistics.fmean(r.score for r in runs)
        target = dead if any(r.dead for r in runs) else working
        label = f"{idea}[{model}/{structure}]" if structure else f"{idea}[{model}]"
        target[label] = round(mean, 2)

    base = ControlSeparation(
        n_control_runs=len(control),
        control_mean=round(statistics.fmean(control), 2) if control else None,
        control_max=round(max(control), 2) if control else None,
        dead_cells=dead,
        dead_max=round(max(dead.values()), 2) if dead else None,
        n_working_cells=len(working),
    )
    if not control or not working:
        return base
    control_mean = statistics.fmean(control)
    weakest = min(working, key=lambda k: working[k])
    base.weakest_real_cell = weakest
    base.weakest_real_mean = working[weakest]
    base.margin = round(working[weakest] - max(control), 2)
    base.working_median = round(statistics.median(working.values()), 2)
    base.below_control = sorted(
        (name for name, score in working.items() if score <= control_mean),
        key=lambda n: working[n],
    )
    return base


# --------------------------------------------------------------------------- #
# Replicate statistics: is a measured gap the models, or the builds?
# --------------------------------------------------------------------------- #


@dataclass
class VarianceComponents:
    """How much of a score moves with the crowd, and how much with the build."""

    #: SD of the same app scored by different crowds.
    crowd_sd: float | None
    #: SD attributable to *which app the model produced* for a given brief.
    build_sd: float | None
    #: Cells that had more than one build, i.e. the evidence for ``build_sd``.
    n_cells: int = 0
    n_seed_groups: int = 0
    median_abs_diff: float | None = None
    max_abs_diff: float | None = None

    @property
    def ratio(self) -> float | None:
        """Build noise as a multiple of crowd noise."""
        if not self.crowd_sd or self.build_sd is None:
            return None
        return round(self.build_sd / self.crowd_sd, 2)


def variance_components(corpus: FleetCorpus) -> VarianceComponents:
    """Split run-to-run variance into crowd noise and build noise.

    Two very different things move a ViralScore, and a single-replicate corpus
    cannot tell them apart:

    * **crowd noise** -- the same app judged by a different draw of 30 users.
      Directly observable as the spread across seeds within one build.
    * **build noise** -- the model, sampled at temperature, producing a
      *different app* from the same brief. Only observable once a brief has been
      built more than once.

    Build noise is recovered by algebra rather than measured directly. The
    spread of a cell's build means contains both terms::

        Var(build means) = sigma_build^2 + sigma_crowd^2 / n_seeds

    so ``sigma_build^2 = Var(build means) - sigma_crowd^2 / n_seeds``, clamped at
    zero. This is the number that says whether a per-brief result means anything:
    if build noise dwarfs crowd noise, one build of one brief is an anecdote no
    matter how many crowds judge it.
    """
    by_build: dict[tuple[str, str, str], list[float]] = {}
    for run in corpus.fleet_runs():
        by_build.setdefault((run.idea_id, run.model, run.build_id), []).append(
            run.score
        )

    seed_groups = [v for v in by_build.values() if len(v) > 1]
    if not seed_groups:
        return VarianceComponents(crowd_sd=None, build_sd=None)
    crowd_var = statistics.fmean(statistics.variance(v) for v in seed_groups)
    crowd_sd = crowd_var**0.5
    n_seeds = statistics.fmean(len(v) for v in seed_groups)

    cells: dict[tuple[str, str], list[float]] = {}
    for (idea, model, _bid), scores in by_build.items():
        cells.setdefault((idea, model), []).append(statistics.fmean(scores))

    multi = [means for means in cells.values() if len(means) > 1]
    if not multi:
        return VarianceComponents(
            crowd_sd=round(crowd_sd, 2),
            build_sd=None,
            n_seed_groups=len(seed_groups),
        )
    between_var = statistics.fmean(statistics.variance(m) for m in multi)
    build_var = max(0.0, between_var - crowd_var / n_seeds)
    spreads = [max(m) - min(m) for m in multi]
    return VarianceComponents(
        crowd_sd=round(crowd_sd, 2),
        build_sd=round(build_var**0.5, 2),
        n_cells=len(multi),
        n_seed_groups=len(seed_groups),
        median_abs_diff=round(statistics.median(spreads), 2),
        max_abs_diff=round(max(spreads), 2),
    )


@dataclass
class WinnerFlip:
    """A brief whose winner was not the same in every replicate."""

    idea_id: str
    per_replicate: dict[int, float]

    @property
    def swing(self) -> float:
        return round(
            max(self.per_replicate.values()) - min(self.per_replicate.values()), 2
        )


def winner_flips(corpus: FleetCorpus, model_a: str, model_b: str) -> list[WinnerFlip]:
    """Briefs where the two models swapped places between replicates.

    The aggregate gap can be rock solid while individual briefs are noise. This
    is the check that stops a reader treating one row of a per-brief table as a
    finding about a model's ability on that kind of software.
    """
    reps = corpus.replicates()
    if len(reps) < 2:
        return []
    per_idea: dict[str, dict[int, float]] = {}
    for rep in reps:
        gap = paired_gap(corpus.restrict(replicate=rep), model_a, model_b)
        for idea, delta in gap.per_idea.items():
            per_idea.setdefault(idea, {})[rep] = delta
    flips = []
    for idea, deltas in per_idea.items():
        if len(deltas) < 2:
            continue
        signs = {d > 0 for d in deltas.values()}
        if len(signs) > 1:
            flips.append(WinnerFlip(idea_id=idea, per_replicate=deltas))
    return sorted(flips, key=lambda f: -f.swing)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials.

    Used instead of the normal approximation because the interesting cases here
    sit at the boundary: one model failed 0 of 50 builds, and the normal
    approximation reports a zero-width interval for that, which is nonsense.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


@dataclass
class DeliveryStats:
    """How reliably a model shipped a launchable build."""

    model: str
    attempted: int
    shipped: int
    #: Builds that failed on their own deliverable contract (manifest missing or
    #: malformed), as opposed to harness faults, which are ours and are retried.
    manifest_failures: int
    per_replicate: dict[int, tuple[int, int]] = field(default_factory=dict)

    @property
    def failure_rate(self) -> float:
        return self.manifest_failures / self.attempted if self.attempted else 0.0

    @property
    def failure_ci(self) -> tuple[float, float]:
        return wilson_interval(self.manifest_failures, self.attempted)


#: Build statuses that mean "the model did not deliver a launchable app".
MANIFEST_FAILURE_STATUSES = frozenset({"manifest_missing", "manifest_invalid"})


def delivery_stats(corpus: FleetCorpus, model: str) -> DeliveryStats:
    """Build-delivery record for one model, pooled and split by replicate."""
    mine = [b for b in corpus.fleet_builds() if b.model == model]
    per_rep: dict[int, tuple[int, int]] = {}
    for rep in corpus.replicates():
        subset = [b for b in mine if b.replicate == rep]
        fails = sum(1 for b in subset if b.status in MANIFEST_FAILURE_STATUSES)
        per_rep[rep] = (fails, len(subset))
    return DeliveryStats(
        model=model,
        attempted=len(mine),
        shipped=sum(1 for b in mine if b.ok),
        manifest_failures=sum(1 for b in mine if b.status in MANIFEST_FAILURE_STATUSES),
        per_replicate=per_rep,
    )
