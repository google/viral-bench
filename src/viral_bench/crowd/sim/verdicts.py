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

"""Aggregate the crowd's verdicts into comparable distributions.

The crowd is the benchmark's measuring instrument, so its output has to *separate*
a great app from a mediocre one. Each first-hand trier records a ``finish_trial``
verdict (would_use / would_share / delight 0-10) and every crowd member answers an
end-of-run interview. This module reduces both into raw distributions -- adoption
and share rates plus the delight spread (mean/median/stdev/min/max + a histogram).

It deliberately computes NO single composite score: turning these raw signals into
a ViralScore is a separate, later stage. Kept free of any ``oasis`` import so it is
unit-testable in the main environment (``simulation`` itself needs the crowd venv).
"""

from __future__ import annotations

import json
import re
import sqlite3
import statistics

from viral_bench.crowd.sim.prompts import CONSIDER_MARKER

#: Leading list/quote decoration a model puts in front of a field name.
_BULLET = re.compile(r"(?m)^[\s>]*(?:[-*+\u2022]|\d+[.)])\s+")
#: Emphasis characters that decorate a field name. ``_`` is deliberately NOT
#: here: ``would_use`` needs its underscore.
_EMPHASIS = re.compile(r"[*`~]")


def normalize_reply(text: str) -> str:
    """Strip markdown decoration so field lines parse.

    The field regexes are line-anchored, so ``**would_use**: yes`` used to parse
    as nothing at all and the whole reply was discarded as a non-answer. Both
    verdicts lost that way in the stored corpus were a ``yes``, so the loss was
    not even unbiased.
    """
    if not text:
        return ""
    return _EMPHASIS.sub("", _BULLET.sub("", text))


def _json_payload(text: str) -> dict | None:
    """The first JSON object in a reply, if it holds any field we want."""
    if not text or "{" not in text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    keys = {"would_use", "would_share", "delight", "score", "for_me"}
    return obj if keys & {str(k).lower() for k in obj} else None


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("yes", "true", "y"):
            return True
        if low in ("no", "false", "n"):
            return False
    return None


def parse_bool_field(text: str, field_name: str) -> bool | None:
    """Parse a ``field: yes/no`` answer from an interview reply.

    Prefers a JSON payload when the model emitted one, then falls back to the
    line format, tolerating markdown decoration around the field name.
    """
    payload = _json_payload(text)
    if payload is not None:
        for key, value in payload.items():
            if str(key).lower() == field_name.lower():
                coerced = _coerce_bool(value)
                if coerced is not None:
                    return coerced
    m = re.search(
        rf"(?im)^\s*{field_name}\s*[:=]\s*(yes|no|true|false|y|n)\b",
        normalize_reply(text),
    )
    if not m:
        return None
    return m.group(1).lower() in ("yes", "true", "y")


def parse_score(text: str) -> int | None:
    """Parse a ``score:`` / ``delight:`` 0-10 answer from an interview reply."""
    payload = _json_payload(text)
    if payload is not None:
        for key, value in payload.items():
            if str(key).lower() in ("delight", "score"):
                try:
                    return max(0, min(10, int(float(value))))
                except (TypeError, ValueError):
                    pass
    m = re.search(
        r"(?im)^\s*(?:score|delight)\s*[:=]\s*(\d{1,2})", normalize_reply(text)
    )
    if not m:
        return None
    return max(0, min(10, int(m.group(1))))


def _mean_of(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(statistics.fmean(vals), 2) if vals else None


def distribution(
    rows: list[dict], dropped: dict | None = None, expected: int | None = None
) -> dict:
    """Summarise verdict rows into a comparable distribution.

    Raw signals only (adoption/share rates + the delight distribution). Rows may
    carry ``None`` for a field an agent did not supply (e.g. a trier that never
    finished, or an interview reply missing a score); those are ignored per field.

    ``dropped`` records replies that never became rows at all. Every drop has to
    be counted: ``n`` is what the score's confidence checks read, so a run that
    quietly discarded half its crowd looked identical to a small healthy one.

    ``expected`` is how many agents were ASKED, and it is what makes the loss
    figure mean what its name says. Without it, loss was
    ``dropped / (rows + dropped)`` -- a count of failed *attempts*, not of lost
    *data* -- so a run whose repair pass worked perfectly was recorded as having
    lost 12% of its crowd. Measured over the first 14 runs of the v11 sweep,
    every one heard from 30 of 30 agents, and one was failed by the health gate
    regardless, purely for having needed four retries. That gate fires hardest
    on runs where the model dropped turns, which tracks load and the apps that
    generate the most output -- so it deletes the busiest runs, not the emptiest.
    """
    uses = [r["would_use"] for r in rows if r.get("would_use") is not None]
    shares = [r["would_share"] for r in rows if r.get("would_share") is not None]
    delights = [r["delight"] for r in rows if r.get("delight") is not None]
    dist: dict = {"n": len(rows), "per_agent": rows}
    if expected:
        dist["expected"] = expected
        dist["coverage"] = round(min(1.0, len(rows) / expected), 3)
        dist["loss_rate"] = round(max(0.0, 1.0 - len(rows) / expected), 3)
    if dropped is not None:
        n_dropped = sum(dropped.values())
        dist["dropped"] = dict(dropped)
        dist["n_dropped"] = n_dropped
        considered = len(rows) + n_dropped
        # Attempts that had to be retried: a diagnostic about the model and the
        # load, NOT a measure of missing data. Keeping them apart is the whole
        # fix -- conflated, a repair pass that worked reads as a failure.
        dist["retry_rate"] = round(n_dropped / considered, 3) if considered else 0.0
        if not expected:
            dist["loss_rate"] = dist["retry_rate"]
    if uses:
        dist["would_use_rate"] = round(sum(uses) / len(uses), 3)
    if shares:
        dist["would_share_rate"] = round(sum(shares) / len(shares), 3)

    # Per-facet means (hands-on trials only) -- the diagnostic breakdown, plus
    # the multi-item `craft` mean the ViralScore actually reads.
    for facet in ("functionality", "usability", "design", "simplicity", "craft"):
        value = _mean_of(rows, facet)
        if value is not None:
            dist[f"{facet}_mean"] = value

    # Two things only a hands-on agent can establish, reported and not scored.
    #
    # The denominator is deliberately "agents who checked", not "agents": a
    # trial that never reloaded says nothing about persistence, and folding its
    # silence in as a "no" would report the crowd's incuriosity as an app
    # defect. ``*_checked`` carries how much evidence there actually is.
    for field in ("work_survived", "saw_other_users"):
        answered = [r[field] for r in rows if r.get(field) is not None]
        dist[f"{field}_checked"] = len(answered)
        if answered:
            dist[f"{field}_rate"] = round(
                sum(bool(v) for v in answered) / len(answered), 3
            )

    # Audience fit: breadth (whole crowd) vs resonance (people it is aimed at).
    fits = [r["for_me"] for r in rows if r.get("for_me") is not None]
    if fits:
        dist["audience_fit_rate"] = round(sum(fits) / len(fits), 3)
        in_audience = [r for r in rows if r.get("for_me")]
        if in_audience:
            uses_in = [
                r["would_use"] for r in in_audience if r.get("would_use") is not None
            ]
            shares_in = [
                r["would_share"]
                for r in in_audience
                if r.get("would_share") is not None
            ]
            dist["in_audience"] = {
                "n": len(in_audience),
                "would_use_rate": round(sum(uses_in) / len(uses_in), 3)
                if uses_in
                else None,
                "would_share_rate": round(sum(shares_in) / len(shares_in), 3)
                if shares_in
                else None,
                "delight_mean": _mean_of(in_audience, "delight"),
            }
    if delights:
        dist["delight_mean"] = round(statistics.fmean(delights), 2)
        dist["delight_median"] = statistics.median(delights)
        dist["delight_stdev"] = (
            round(statistics.pstdev(delights), 2) if len(delights) > 1 else 0.0
        )
        dist["delight_min"] = min(delights)
        dist["delight_max"] = max(delights)
        # Counts indexed by score 0..10, so the shape of the spread is explicit.
        dist["delight_histogram"] = [delights.count(i) for i in range(11)]
    return dist


def _had_effect(toolkit) -> bool | None:
    """Did this trial do anything beyond opening and looking?

    Defensive because a toolkit may be absent (agent never ran) and because
    trace-like objects are also constructed in tests and from stored artifacts.
    """
    fn = getattr(getattr(toolkit, "trace", None), "had_effect", None)
    if not callable(fn):
        return None
    try:
        return bool(fn())
    except Exception:  # noqa: BLE001 - a reporting field must never fail a run
        return None


def trier_verdicts(crowd) -> dict:
    """Aggregate every first-hand ``finish_trial`` verdict.

    Includes a latecomer that the feed converted -- it used the app, so its
    verdict is hands-on evidence like any other -- and excludes one that never
    opened it, whose silence belongs in the conversion rate rather than in
    craft. Reading it any other way would let an app be marked down for the
    people it failed to interest, twice.
    """
    rows: list[dict] = []
    hands_on = [
        aid
        for aid in getattr(crowd, "hands_on_ids", crowd.trier_ids)
        if aid in crowd.trier_ids
        or any(
            s.action == "open"
            for s in getattr(
                getattr(crowd.trier_toolkits.get(aid), "trace", None), "steps", []
            )
            or []
        )
    ]
    for aid in hands_on:
        toolkit = crowd.trier_toolkits.get(aid)
        verdict = toolkit.trace.verdict if toolkit else None
        row = {
            "agent_id": aid,
            "username": crowd.persona_by_id[aid].username,
            "would_use": verdict.would_use if verdict else None,
            "would_share": verdict.would_share if verdict else None,
            "delight": verdict.delight if verdict else None,
            "finished": verdict is not None,
            # Only count hands-on evidence that was actually hands-on: a
            # degraded trial (e.g. a web app observed over static HTTP with no
            # browser) never ran the app, so its craft rating is not evidence.
            "degraded": bool(getattr(toolkit.trace, "degraded", False))
            if toolkit
            else None,
            # Stronger than degraded: the agent never reached the app at all and
            # is describing the source tree, or an error page, or nothing. The
            # scorer drops these from craft entirely.
            "app_reachable": getattr(toolkit.trace, "app_reachable", None)
            if toolkit
            else None,
            # How much of the app the agent actually exercised, so "opened it and
            # left" is distinguishable from "used it".
            "had_effect": _had_effect(toolkit),
            "n_steps": getattr(getattr(toolkit, "trace", None), "n_steps", None),
        }
        if verdict is not None:
            for facet in ("functionality", "usability", "design", "simplicity"):
                row[facet] = getattr(verdict, facet, None)
            row["craft"] = verdict.craft
            row["work_survived"] = getattr(verdict, "work_survived", None)
            row["saw_other_users"] = getattr(verdict, "saw_other_users", None)
        rows.append(row)
    return distribution(rows)


def conversion_stats(crowd, decisions: dict[int, dict] | None = None) -> dict:
    """How many not-yet-users the FEED talked into actually trying the app.

    This is the one number here that virality is not allowed to be handed. Every
    other signal counts something the crowd did to a post; this counts people
    who went and used a product because of what they read, and an app nobody can
    recommend convincingly cannot fake it.

    A latecomer counts as converted once its trace shows an ``open``. Trying and
    failing still counts: the feed did its job and the app did not, which the
    craft and validity signals exist to catch separately.
    """
    ids = list(getattr(crowd, "latecomer_ids", []) or [])
    decisions = decisions or {}
    rows: list[dict] = []
    for aid in ids:
        toolkit = crowd.trier_toolkits.get(aid)
        trace = getattr(toolkit, "trace", None)
        steps = getattr(trace, "steps", []) or []
        verdict = getattr(trace, "verdict", None)
        decision = decisions.get(aid) or {}
        rows.append(
            {
                "agent_id": aid,
                "username": crowd.persona_by_id[aid].username,
                # What the agent SAID when asked, with no tool in reach.
                "decided": decision.get("decided"),
                "because": decision.get("because", ""),
                "convinced_by": decision.get("convinced_by", ""),
                # What it then DID. The two are reported separately because a
                # decision that never turns into use is a different failure
                # from one that does.
                "converted": any(step.action == "open" for step in steps),
                "reached_app": getattr(trace, "app_reachable", None) is True,
                "would_use": getattr(verdict, "would_use", None),
                "delight": getattr(verdict, "delight", None),
                "n_steps": len(steps),
            }
        )
    answered = [r for r in rows if r["decided"] is not None]
    yes = sum(1 for r in answered if r["decided"])
    converted = sum(1 for r in rows if r["converted"])
    return {
        "n_latecomers": len(rows),
        "n_answered": len(answered),
        "n_decided_yes": yes,
        # The headline: of the people who had not tried it, how many did the
        # feed talk into trying it. Earned, not computed.
        "conversion_rate": round(yes / len(answered), 3) if answered else None,
        "n_converted": converted,
        "opened_rate": round(converted / len(rows), 3) if rows else None,
        "per_agent": rows,
    }


def _extract_why(response: str) -> str:
    """Pull the free-text rationale out of a structured interview reply."""
    for line in (response or "").splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("why"):
            _, _, rest = stripped.partition(":")
            return rest.strip()[:400]
    # Unstructured reply: keep a clipped version rather than losing it entirely.
    return " ".join((response or "").split())[:400]


def _interview_rows(db_path: str, *, marker: str | None) -> list[tuple]:
    """Interview trace rows, either ONLY the marked ones or only the unmarked.

    OASIS stores every INTERVIEW action under one name, and a latecomer's
    "are you going to try this?" turn is one. Without this split the aggregator
    -- which keeps the FIRST answer per agent -- would record that as the
    agent's final verdict on the app.
    """
    con = sqlite3.connect(db_path)
    try:
        fetched = con.execute(
            "SELECT user_id, info FROM trace WHERE action = 'interview'"
        ).fetchall()
    finally:
        con.close()
    out = []
    for user_id, info in fetched:
        try:
            payload = json.loads(info)
        except (json.JSONDecodeError, TypeError):
            payload = {}
        prompt = str(payload.get("prompt") or "")
        is_marked = CONSIDER_MARKER in prompt
        if (marker is None and is_marked) or (marker is not None and not is_marked):
            continue
        out.append((user_id, info))
    return out


def consideration_decisions(db_path: str) -> dict[int, dict]:
    """Each latecomer's "am I going to try this?" answer, by agent id."""
    out: dict[int, dict] = {}
    try:
        rows = _interview_rows(db_path, marker=CONSIDER_MARKER)
    except sqlite3.Error:
        return out
    for user_id, info in rows:
        if user_id in out:
            continue
        try:
            response = json.loads(info).get("response", "")
        except (json.JSONDecodeError, TypeError):
            continue
        decided = parse_bool_field(response, "try")
        out[user_id] = {
            "decided": decided,
            "because": _field_text(response, "because"),
            "convinced_by": _field_text(response, "convinced_by"),
        }
    return out


def _field_text(response: str, name: str) -> str:
    for line in (normalize_reply(response) or "").splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(f"{name}:"):
            return stripped.partition(":")[2].strip()[:300]
    return ""


def interview_verdicts(db_path: str, crowd) -> dict:
    """Parse + aggregate the whole crowd's end-of-run interview verdicts."""
    rows: list[dict] = []
    dropped = {"duplicate": 0, "bad_record": 0, "unparseable": 0, "db_error": 0}
    try:
        fetched = _interview_rows(db_path, marker=None)
    except sqlite3.Error:
        dropped["db_error"] = 1
        return distribution(rows, dropped, expected=len(crowd.crowd_ids))
    seen: set[int] = set()
    for user_id, info in fetched:
        # One verdict per agent. A batch interview failure is retried per-agent,
        # which can re-ask someone; counting them twice would silently
        # double-weight that persona. First answer wins, mirroring the
        # "a trier's verdict is final" rule on the hands-on side.
        if user_id in seen:
            dropped["duplicate"] += 1
            continue
        try:
            response = json.loads(info).get("response", "")
        except (json.JSONDecodeError, TypeError):
            dropped["bad_record"] += 1
            continue
        use = parse_bool_field(response, "would_use")
        share = parse_bool_field(response, "would_share")
        score = parse_score(response)
        # A reply with nothing parseable is a non-answer (a skipped turn after a
        # model failure, or a refusal). Counting it would inflate n with a row
        # carrying no signal, making a degraded run look better-evidenced than
        # it is -- and n is what the score's confidence checks read. It is
        # counted rather than vanishing: a parser that silently eats replies is
        # indistinguishable from a crowd that never answered.
        if use is None and share is None and score is None:
            dropped["unparseable"] += 1
            continue
        seen.add(user_id)
        persona = crowd.persona_by_id.get(user_id)
        rows.append(
            {
                "agent_id": user_id,
                "username": persona.username if persona else str(user_id),
                "tier": crowd.tier_by_id.get(user_id, "?"),
                "would_use": use,
                "would_share": share,
                "delight": score,
                "for_me": parse_bool_field(response, "for_me"),
                # The verbatim rationale. The parsed yes/no/score is what the
                # deterministic score reads, but the reasoning is what an
                # agentic rater needs -- "no, it leaks my data" and "no, I
                # already use something else" are the same number and very
                # different signals about the app.
                "why": _extract_why(response),
            }
        )
    rows.sort(key=lambda r: r["agent_id"])
    return distribution(rows, dropped, expected=len(crowd.crowd_ids))


#: Fraction of the CROWD that may go unheard before a run is unusable.
#:
#: Measured against coverage (agents who produced a parseable verdict over
#: agents asked), not against retries. Losses are not random -- they concentrate
#: in the runs with the most discussion, i.e. the best apps -- so a lenient
#: threshold here silently biases the very comparison the benchmark exists to
#: make, and a threshold applied to the wrong quantity throws away healthy runs
#: for the same reason.
MAX_VERDICT_LOSS_RATE = 0.10


#: Share of model turns that may be lost before the run stops being usable. A
#: skipped turn is recorded by OASIS as an agent doing nothing, so throttling
#: reads as crowd indifference -- and if two founder models are ever measured
#: under different load, that becomes a difference between the models.
MAX_TURN_SKIP_RATE = 0.10

#: Share of turns that may end at the ``max_iteration`` ceiling before the run
#: stops being usable. Held tighter than the skip rate because the budget
#: arithmetic (``trial_max_steps + social_headroom``) is designed so this cannot
#: happen at all: anything above noise means that guarantee has broken, and the
#: agents it silences are the ones who engaged MOST -- i.e. the best apps.
MAX_TURN_BUDGET_EXHAUSTION_RATE = 0.02


def undeliverable_validity(build_id: str) -> dict:
    """The validity verdict for a build with no launch contract.

    ``False``, not ``None``. The distinction is the whole point: ``None`` means
    *we could not check* (a fault in our harness, which must never be charged to
    the app and carries no score penalty), while ``False`` means *we checked and
    it does not run*. A build with no ``viralbench.json`` is the second thing --
    there is no command to try, and that is the founder's omission, not ours. It
    therefore takes the same broken-app multiplier as an app that fails to start.
    """
    return {
        "build_id": build_id,
        "builds": False,
        "runs": False,
        "does_what_it_claims": False,
        "detail": (
            "undeliverable: no valid viralbench.json, so there is no documented "
            "way to install, start or test this app"
        ),
    }


def assess_run_health(
    *,
    error: str | None,
    rounds_run: int,
    rounds_expected: int,
    round_log: list[dict],
    verdicts: dict,
    interview_expected: bool,
    turn_stats: dict | None = None,
) -> dict:
    """Decide whether a run is usable, and say why if it is not.

    A run's ``ok`` used to mean only "the orchestrator did not raise". Six stored
    runs therefore report ``ok: true`` having completed 2 of 4 rounds and
    collected zero interviews. Because the interview is the *primary* scoring
    signal, those runs were then scored off craft alone -- the most stable
    component -- so the resulting number looked *more* trustworthy than a healthy
    run's. A benchmark cannot have a failure mode where losing data increases
    apparent confidence.

    Returns ``{"ok": bool, "failures": [str, ...]}``.
    """
    failures: list[str] = []
    if error:
        failures.append(f"run raised: {error}")

    expected = int(rounds_expected or 0)
    if expected and rounds_run < expected:
        lost = [r.get("round") for r in round_log if not r.get("ok", True)]
        failures.append(
            f"only {rounds_run} of {expected} rounds completed"
            + (f" (failed: {lost})" if lost else "")
        )

    if interview_expected:
        interviews = verdicts.get("interviews") or {}
        n = int(interviews.get("n") or 0)
        if n == 0:
            failures.append(
                "zero interview verdicts: the primary scoring signal is gone"
            )
        else:
            # Coverage when it is recorded, loss_rate otherwise (older runs).
            expected = int(interviews.get("expected") or 0)
            loss = (
                max(0.0, 1.0 - n / expected)
                if expected
                else float(interviews.get("loss_rate") or 0.0)
            )
            if loss > MAX_VERDICT_LOSS_RATE:
                failures.append(
                    f"{loss:.0%} of the crowd produced no usable verdict "
                    f"(limit {MAX_VERDICT_LOSS_RATE:.0%}); "
                    f"{n} of {expected or '?'} answered"
                )
    stats = turn_stats or {}
    skipped = int(stats.get("skipped") or 0)
    turns = int(stats.get("turns") or 0)
    if turns and skipped / turns > MAX_TURN_SKIP_RATE:
        limited = int(stats.get("skipped_rate_limited") or 0)
        failures.append(
            f"{skipped} of {turns} model turns were lost "
            f"({skipped / turns:.0%}, {limited} to rate limits): the crowd's "
            "silence is ours, not the app's"
        )
    exhausted = int(stats.get("budget_exhausted") or 0)
    if turns and exhausted / turns > MAX_TURN_BUDGET_EXHAUSTION_RATE:
        failures.append(
            f"{exhausted} of {turns} turns hit the max_iteration ceiling "
            f"({exhausted / turns:.0%}): those agents were cut off mid-turn, not "
            "quiet. Raise simulation.social_headroom in config/crowd.yaml"
        )
    return {"ok": not failures, "failures": failures}
