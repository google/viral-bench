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

"""The agentic autorater: an LLM that reads the crowd's trajectory and rates it.

The deterministic components count things -- how many adopted, how many
reposted, what the facet means were. They cannot read. Three questions decide
whether an app really landed, and all of them require reading:

* **substance** -- was the praise specific and earned, or reflexive? "Clean and
  useful" and "the zero-backend share link is the reason I'd switch" are the
  same +1 adoption and very different evidence.
* **severity** -- how damaging is the strongest criticism actually raised? A
  crowd can be mildly positive about an app with one fatal flaw.
* **word_of_mouth** -- did agents genuinely persuade each other, or did fifty
  agents independently form fifty opinions? Only the first is virality.

These sit *alongside* the deterministic metrics rather than replacing them: the
counts measured far better signal-to-noise than an LLM judgement will, so they
keep the bulk of the weight (see ``config/score.yaml``).

Three properties make this a measurement rather than a vibe:

* **Blinded.** The pack never names the founder model. A rater that can infer
  which model it is judging may rate reputation instead of artifact -- fatal for
  a benchmark whose entire purpose is comparing models.
* **Cited.** Every dimension must cite evidence ids it relied on, so a rating
  can be audited against the run rather than taken on trust.
* **Repeated.** The rater runs k times at temperature 0 and the MEDIAN is used.
  A single LLM sample is not a measurement, and the median is robust to the
  occasional outlier reading.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import asdict, dataclass, field

from viral_bench.providers import Capability, check_support, make_client, resolve
from viral_bench.score.evidence import EvidencePack

_LOG = logging.getLogger("viral_bench.score.autorater")

#: The dimensions the rater scores. Keys must match ``profiles.*.autorater`` in
#: config/score.yaml.
DIMENSIONS: tuple[str, ...] = ("substance", "severity", "word_of_mouth")

#: Extra attempts per sample before that sample is written off. Failures here
#: are transient (a throttled call, a malformed JSON body), so a retry usually
#: recovers the measurement rather than repeating the failure.
_SAMPLE_RETRIES = 2

#: Output ceiling for one rating sample.
#:
#: The cap went 2048 -> 8192 -> none, and the first two were guesses. 2048
#: demonstrably cost a sample: the only stored rating in the corpus records
#: "sample 2: unparseable response", and a reply truncated mid-object is a
#: silently lost measurement rather than a visible failure. On recent Gemini
#: thinking tokens are charged against this same budget, so a tight cap throttles
#: the rater's reasoning before it ever reaches the JSON.
#:
#: "None" is no longer expressible -- the Anthropic API requires a ceiling, so the
#: provider layer always sends one -- and the number is pinned here rather than
#: inherited from the layer's default so that changing that default cannot
#: quietly reintroduce a cap this rater was measured to lose samples under. It is
#: roughly forty times the longest reply the rubric can produce.
_SAMPLE_MAX_TOKENS = 8192

_RUBRIC = """\
You are rating how a simulated crowd responded to a newly launched app. You are
a measuring instrument for a benchmark, not a reviewer: be consistent, be
sceptical, and justify everything with evidence from the transcript.

Rate these three dimensions from 0 to 10. Use the FULL range and default to the
middle -- most launches are unremarkable.

1. substance (0-10) -- Did the app EARN specific praise for things it actually
   does? You are rating the APP through the evidence, not the crowd's eloquence.
   0-2  nothing earned: no praise, or praise for an app nobody got working
   3-4  only generic approval ("nice idea", "clean UI") with no working feature
        named, or specifics that describe what it PROMISES rather than what an
        agent saw it do
   5-6  one or two concrete things it demonstrably did, amid filler
   7-8  several concrete capabilities agents actually exercised and valued
   9-10 the crowd names precisely what it does better than what they already use
   DECISIVE RULE: praise counts only when tied to something an agent OBSERVED
   the app do. A well-argued takedown of a broken app is LOW substance, not
   high: the reasoning is excellent and the app earned nothing. If the
   transcript shows the app failed to run, or that nobody completed its core
   task, substance CANNOT exceed 2 however thoughtful the discussion is.

2. severity (0-10) -- How damaging is the STRONGEST criticism actually raised?
   NOTE THE DIRECTION: 10 means NO meaningful criticism, 0 means fatal flaws.
   0-2  fatal: it does not work, loses data, or is unusable
   3-4  serious: a core feature is broken or a dealbreaker for most users
   5-6  real but survivable gaps
   7-8  minor nitpicks only
   9-10 no substantive criticism despite people looking
   Judge the WORST substantiated problem, not the average sentiment. An absence
   of criticism because nobody engaged is not a 9 -- if few agents actually used
   the app, say so and stay near the middle.

3. word_of_mouth (0-10) -- Did the APP spread through the crowd on its merits?
   This is a virality measure, so DIRECTION MATTERS: it must reward an app
   people pass on, not a controversy people argue about.
   0-2  nobody passed it on; opinions are independent and parallel
   3-4  acknowledgements only, or discussion that is mostly complaints and
        post-mortems about why it does not work
   5-6  some genuine back-and-forth on its merits; mild interest spreading
   7-8  agents recommend it to each other and build on each other's reasons for
        using it; a sceptic is talked round
   9-10 a real cascade: it is quoted, reposted and pressed on others, and the
        enthusiasm compounds
   DECISIVE RULE: score the spread of ADVOCACY, not the volume of conversation.
   A broken app that generates a long thread of criticism is 0-3, not 8 -- the
   crowd is warning each other off, which is the opposite of word of mouth.

Return ONLY a JSON object, no prose, in exactly this form:
{"substance": {"score": <0-10>, "evidence": ["<id>"], "reason": "<one line>"},
 "severity": {"score": <0-10>, "evidence": ["<id>"], "reason": "<one line>"},
 "word_of_mouth": {"score": <0-10>, "evidence": ["<id>"], "reason": "<one line>"}}

Cite evidence by the ids shown in the transcript, e.g. "agent 7", "COMMENT 12".
"""


@dataclass
class DimensionRating:
    """One dimension's rating, aggregated over repeats."""

    score: float
    samples: list[float] = field(default_factory=list)
    spread: float = 0.0
    evidence: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class AutoRating:
    """The autorater's output for one crowd run."""

    build_id: str
    model: str
    repeats: int
    dimensions: dict[str, DimensionRating] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.dimensions)

    def normalized(self) -> dict[str, float]:
        """Dimension scores as 0..1, ready to be weighted into the composite."""
        return {k: v.score / 10.0 for k, v in self.dimensions.items()}

    def as_dict(self) -> dict:
        return {
            "build_id": self.build_id,
            "model": self.model,
            "repeats": self.repeats,
            "errors": self.errors,
            "dimensions": {k: asdict(v) for k, v in self.dimensions.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> AutoRating:
        """Rebuild a rating persisted by :meth:`as_dict` (``autorating.json``).

        Rating is expensive and already paid for, so every consumer that
        re-scores stored runs reads it back from disk rather than calling the
        model again. A malformed or truncated file yields a rating whose ``ok``
        is False -- the run is then scored deterministically and *counted* as
        unrated, rather than raising and taking a whole sweep down with it.
        """
        if not isinstance(data, dict):
            return cls(build_id="", model="", repeats=0)
        dims: dict[str, DimensionRating] = {}
        raw = data.get("dimensions")
        for name, d in (raw or {}).items() if isinstance(raw, dict) else ():
            if not isinstance(d, dict) or d.get("score") is None:
                continue
            # Clamp on the way in, exactly as _coerce does for a live response.
            # This path reads JSON written by some earlier run, so it must not
            # assume the writer enforced the rubric: normalized() divides by 10
            # and the result is weighted straight into the composite, so a stored
            # score of 50 would contribute 5.0 to a term required to be in [0,1].
            score = _coerce(d["score"])
            if score is None:
                continue
            dims[str(name)] = DimensionRating(
                score=score,
                samples=[float(s) for s in (d.get("samples") or [])],
                spread=float(d.get("spread") or 0.0),
                evidence=[str(e) for e in (d.get("evidence") or [])],
                reason=str(d.get("reason") or ""),
            )
        return cls(
            build_id=str(data.get("build_id") or ""),
            model=str(data.get("model") or ""),
            repeats=int(data.get("repeats") or 0),
            dimensions=dims,
            errors=[str(e) for e in (data.get("errors") or [])],
        )


def _parse_response(text: str) -> dict:
    """Pull the JSON object out of a model reply, tolerating fences/preamble."""
    if not text:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    raw = fenced.group(1) if fenced else None
    if raw is None:
        start, end = text.find("{"), text.rfind("}")
        raw = text[start : end + 1] if start != -1 and end > start else ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _coerce(value) -> float | None:
    try:
        return max(0.0, min(10.0, float(value)))
    except (TypeError, ValueError):
        return None


def rate_pack(
    pack: EvidencePack,
    *,
    client=None,
    model: str | None = None,
    temperature: float | None = None,
    repeats: int | None = None,
) -> AutoRating:
    """Run the autorater over an evidence pack and aggregate the repeats.

    ``client`` is anything speaking the provider interface --
    ``generate(messages) -> Reply`` (see :mod:`viral_bench.providers`) -- and the
    default is built from ``autorater.model`` in config/score.yaml. Injecting one
    keeps this unit-testable without network access.

    With no model configured and none injected the rater is **disabled**: it
    returns a rating whose ``ok`` is False rather than raising or reaching for a
    model of its own. See the note at that branch for why that is the right
    failure.
    """
    from viral_bench import config as _config

    cfg = _config.autorater_config()
    model = model or str(cfg.get("model") or "")
    temperature = cfg.get("temperature", 0.0) if temperature is None else temperature
    repeats = int(repeats if repeats is not None else cfg.get("repeats", 3))

    if client is None:
        if not model:
            # Off, not broken. ViralBench ships no default model, and picking one
            # here would bill an account the user never named -- and worse, would
            # rate every run with a model that appears in no configuration and so
            # in no write-up. The rater is an optional component
            # (``autorater.enabled`` in config/score.yaml): a run without it is
            # scored deterministically with its weight renormalised away and
            # *counted* as unrated, which is exactly what an unusable rating
            # already produces. Raising instead would take a whole sweep down
            # over a setting the sweep never needed.
            _LOG.warning(
                "autorater disabled: no model configured (set autorater.model in "
                "config/score.yaml to a 'provider/model' string)"
            )
            return AutoRating(
                build_id=pack.build_id,
                model="",
                repeats=0,
                errors=["no autorater model configured"],
            )
        client = _default_client(model, temperature)

    prompt = f"{_RUBRIC}\n\n=== TRANSCRIPT ===\n{pack.to_prompt()}\n"
    rating = AutoRating(build_id=pack.build_id, model=model, repeats=repeats)

    samples: dict[str, list[float]] = {d: [] for d in DIMENSIONS}
    evidence: dict[str, list[str]] = {d: [] for d in DIMENSIONS}
    reasons: dict[str, str] = {}
    for attempt in range(repeats):
        # Retry a failed draw instead of dropping it. A lost sample is not a
        # neutral event: it shrinks the median's support, and at the default
        # repeats=3 a single loss leaves an even number of samples for every
        # dimension. The one stored rating in the corpus lost a sample exactly
        # this way and nothing anywhere recorded that the rating was thinner
        # than it claimed to be.
        data = None
        for retry in range(_SAMPLE_RETRIES + 1):
            try:
                reply = client.generate([{"role": "user", "content": prompt}])
                data = _parse_response(reply.text)
            except Exception as exc:  # noqa: BLE001 - a bad sample must not kill the rating
                rating.errors.append(
                    f"sample {attempt} try {retry}: {type(exc).__name__}: {exc}"
                )
                data = None
            else:
                if data:
                    break
                rating.errors.append(f"sample {attempt} try {retry}: unparseable")
        if not data:
            rating.errors.append(f"sample {attempt}: dropped after retries")
            continue
        for dim in DIMENSIONS:
            item = data.get(dim)
            if not isinstance(item, dict):
                continue
            score = _coerce(item.get("score"))
            if score is None:
                continue
            samples[dim].append(score)
            ev = item.get("evidence")
            if isinstance(ev, list):
                evidence[dim].extend(str(e) for e in ev[:6])
            reasons.setdefault(dim, str(item.get("reason", ""))[:300])

    for dim in DIMENSIONS:
        vals = samples[dim]
        if not vals:
            continue
        rating.dimensions[dim] = DimensionRating(
            score=round(statistics.median(vals), 2),
            samples=vals,
            # Spread across repeats IS the rater's own reliability. A dimension
            # that swings wildly between identical calls should not be trusted
            # with weight, and this makes that visible instead of hidden.
            spread=round(max(vals) - min(vals), 2) if len(vals) > 1 else 0.0,
            evidence=list(dict.fromkeys(evidence[dim]))[:6],
            reason=reasons.get(dim, ""),
        )
    if not rating.dimensions:
        _LOG.warning("autorater produced no usable dimensions: %s", rating.errors)
    return rating


def _default_client(model: str, temperature: float):
    """Build the rater's client for one ``provider/model`` string.

    The injection seam. Everything about *how* to reach a model -- credentials,
    base URL, retries, error classification -- belongs to
    :mod:`viral_bench.providers`; what belongs here is the two things the rater
    itself requires of whatever it is pointed at.

    **JSON mode**, because the rater parses its own replies and unconstrained
    prose is an unparseable sample, i.e. a silently lost measurement. It is a
    hard requirement rather than a preference, so a provider that cannot do it is
    refused here rather than discovered three hundred ratings into a sweep.

    **An output ceiling that bounds nothing** -- see :data:`_SAMPLE_MAX_TOKENS`.

    Nothing else, and in particular no transport of its own. The rater used to
    carry a copy of the crowd's Vertex switch, and the cost of that is worth
    remembering now that it has none: when the simulation moved to Vertex and the
    rater was left on the Developer API, the rater went on hitting that API's
    throttle alone. Its three dimensions carry 15% of the active profile and a
    failed rating is renormalised away, so those runs were scored under a
    different profile than rated ones -- and the loss ran uneven by arm (solo 36%
    unrated against team 8%), which is the one way it cannot be dismissed as
    noise.
    """
    spec = resolve(model)
    check_support(spec, Capability.JSON_MODE, stage="autorater")
    return make_client(
        spec,
        json_mode=True,
        temperature=temperature,
        max_tokens=_SAMPLE_MAX_TOKENS,
    )
