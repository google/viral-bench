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

"""Shared crowd-simulation defaults (dependency-free, importable in both envs).

These constants are needed both by the main-process launcher
(:mod:`viral_bench.crowd.launch`, Python 3.12) and by the in-crowd-env runner
(:mod:`viral_bench.crowd.sim.runner`, Python 3.11). They live here -- with no
``oasis``/``camel`` imports -- so the 3.12 side can read them without pulling in
the simulation stack.
"""

from __future__ import annotations

from viral_bench import config as _config

#: Version of the crowd ARCHITECTURE -- the personas, prompts, feed, rounds,
#: interview and trial mechanics that decide what a run measures. Bump it on any
#: change that alters what the crowd does, so runs produced by two different
#: instruments are never pooled into one comparison.
#:
#: SCORE_VERSION covers the formula applied to a run's artifacts (and re-scoring
#: is free). This covers the run itself, which is NOT free to redo -- so it has
#: to be recorded at the time, or the corpus silently becomes a mixture.
#:
#: 1: the instrument as inherited on 2026-08-01 (post bug-fix branch).
#: 2: craft facets defined per app type, so cli and bot trials rate them at all
#:    (0 of 25 did under v1, while 259 of 293 web trials did).
#: 3: the feed carries author handles and every account has a bio, so agents can
#:    name each other and the interest recommender has something to rank on.
#: 4: the v3 "talk to PEOPLE" nudge is cut back to naming only. v3 collapsed
#:    reactor discrimination (adopt good-minus-broken 0.435 -> 0.023) by turning
#:    the reactor's job from judging the app into socialising about it.
#: 5: reachability is measured for cli and bot too, not just web. Under v4 a CLI
#:    whose every invocation exited 1 still produced 8 "valid" trials rating it
#:    craft 8.0-8.8 -- the fabricated-craft failure in the 48% of the bench that
#:    had no reachability check at all.
#: 6: the interview re-asks agents whose reply did not parse. A batch step that
#:    "succeeded" is not an answer -- a model turn lost to throttling becomes a
#:    synthetic "(no response)" -- and those losses concentrate in the runs with
#:    the most discussion, i.e. the best apps.
#: 7: every founder build is simulated, including ones with no usable
#:    viralbench.json. Under v6 those were skipped entirely, so a model that
#:    failed to ship a launch contract disappeared from the denominator while
#:    the other model's bad-but-runnable apps stayed in and lowered its average.
#: 8: agents can SEE. A screenshot reaches the model as an image part instead of
#:    a sentence about a screenshot, and agents can upload a file. Under v7 the
#:    entire percept was text, so the design facet -- 22% of the score, asking
#:    how an app looks -- was being answered for apps that render no text to
#:    look at. Measured over 2,088 stored web trials: apps whose output is a
#:    canvas or an image averaged design 6.89, while apps a text-only agent
#:    could genuinely read averaged 6.64. The blind apps scored HIGHER, which is
#:    the signature of a number produced from the pitch rather than the artifact.
#:    The bench is also web-only now, so cli and bot trials no longer exist.
#: 9: the crowd actually USES the apps. Under v8, 36% of triers recorded a
#:    verdict without one successful click, type or upload -- the modal
#:    trajectory was open -> look -> finish, i.e. rating a landing page -- and
#:    they scored delight 5.94 against 7.05 for triers who interacted 3-5 times.
#:    Clicks resolved to hidden elements in closed modals (109 of 136 timeouts),
#:    uploads failed 70% of the time by driving the styled label instead of the
#:    display:none input behind it, and reposting -- the most direct virality
#:    mechanic -- fired 0.8 times per 30-agent run because the prompt named it
#:    once and then spent four sentences discouraging amplification. Measured on
#:    one build after the fixes: never-interacted 54% -> 0%, reposts 0.7 -> 9.0.
#: 10: **everyone can try the app.** Triers go from 8 of 30 to all 30; each
#:    trier publishes its own post instead of commenting on the launch; the
#:    follow graph gains interest neighbourhoods instead of being one star on
#:    the founder; the feed is ranked (max_rec_posts 20 < the ~40 posts a run
#:    now produces) so reach is earned rather than handed out identically to
#:    every agent in 47 of 47 runs; server-backed apps hand each agent its own
#:    account and ask it to check persistence and whether it can see anyone
#:    else's work; agents get `reload_page` and `select_option`, without which
#:    "did my work survive" and "choose from this menu" were untestable; and a
#:    verdict filed without ever calling open_app is pushed back once (32 of 352
#:    stored trials rated an app they never opened, from reading its source).
#:    Rounds 4 -> 3: the last two rounds were 83% and 90% do_nothing.
#: 11: a failed SUB-RESOURCE stops voiding the trial. Any "requestfailed ...
#:     net::ERR" was read as "the main request failed", which marked the trial
#:     degraded and struck its craft out of the score. Over the first 660 trials
#:     of the v10 sweep that voided 106 (16%) and not one was the app failing to
#:     load: Server-Sent Events aborted by navigating away, fonts.gstatic.com
#:     with no external network, XHRs cancelled by a reload, revoked blob: URLs
#:     from a download. All four are markers of a MORE capable app, so the bias
#:     ran against the builds the bench exists to reward -- image_compressor
#:     lost 30 of 30 trials, collaborative_table 23 of 60. Only a failed
#:     document navigation now counts.
#: 12: interview loss is measured as COVERAGE (agents who answered / agents
#:     asked) instead of as failed attempts. It was dropped/(rows+dropped), so a
#:     run whose repair pass recovered every agent was recorded as having lost
#:     12% of its crowd and the health gate failed it. All 14 runs of the v11
#:     sweep heard from 30 of 30 agents; one was thrown away regardless. The
#:     gate fired hardest on runs where the model dropped turns, which tracks
#:     load and the apps that provoke the most output -- so it deleted the
#:     busiest runs, not the emptiest ones.
#: 13: ALLOCATED AND REJECTED. Two crowd rounds instead of three. Round 3 is
#:     90% do_nothing (4,235 of 4,680 turns across 156 runs) and on a matched
#:     10-brief arm dropping it looked like the best change available. Run over
#:     the full fleet at 3 seeds it is worse on every axis that matters:
#:     within-cell noise 7.60 -> 9.35, discrimination 3.27 -> 2.70, and it
#:     separates Gemini from ITSELF by +6.67 points against v12's -1.16. Its
#:     156 runs stay on disk tagged "13" as the evidence. The instrument is
#:     therefore still v12, and the version stays 12 -- bumping it would exclude
#:     the 150 runs that are the current corpus in exchange for nothing.
#: 14: the app the crowd sees is no longer missing its dependencies, and the
#:     crowd no longer loses turns to a request Vertex refuses. All three changes
#:     alter what an agent actually observes, so v12 runs may not be pooled with
#:     these:
#:
#:     * Per-build dependency provisioning (``founder/provision.py``). A build is
#:       materialized by ``git clone`` of its shipped branch, so it legitimately
#:       arrives without the ``node_modules``/``.venv`` its own .gitignore
#:       excludes -- and a ``pip install`` in a manifest setup step was being
#:       thrown away, because ContainerRuntime runs setup under ``--rm`` where
#:       site-packages is not a mount. Apps that could not start now start.
#:     * Node 20 -> 22 in the runtime image, matching the host the founder builds
#:       and self-tests on. ``node:sqlite`` and ``unsupported engine`` were our
#:       toolchain being recorded as a model that cannot ship.
#:     * The ``[function_response, image]`` request shape is split into two
#:       turns. Vertex rejected it as "Requests ending with a model turn are not
#:       supported" -- a message naming the wrong problem, since that turn's role
#:       is "user", which is why the guard written for it never fired. It cost
#:       54,522 agent turns across the r3 sweep; measured on one build, 6 of 368
#:       turns before and 0 of 312 after.
#:
#:     Skipping 13 is deliberate: it is taken, by 156 runs still on disk.
CROWD_ARCH_VERSION = "14"

#: The crowd's model, as ``provider/model`` -- from config/crowd.yaml
#: (``simulation.model.id``), overridable with ``--model``.
#:
#: There is deliberately no fallback. It used to read ``gemini-2.0-flash``,
#: which made a Google account a silent prerequisite for running the benchmark at
#: all: a user who had configured some other provider still got a crowd that
#: tried to call Gemini, and failed doing it. An unset value is not an error
#: here -- it is an error at the point a model is actually built, where
#: :func:`viral_bench.crowd.sim.model.crowd_model` can say which file to edit and
#: that ``viral-bench init`` will do it for you.
DEFAULT_MODEL = _config.crowd_model_id("")

#: Interest-based recommender (TWHIN-BERT). Use "twitter" for a light wiring smoke.
DEFAULT_RECSYS = "twhin-bert"

#: Crowd size for a scored run (calibrate up toward ~30-50; start smaller).
DEFAULT_AGENTS = _config.crowd_agents(8)

#: How many crowd members first-hand try the app. Negative means ALL of them,
#: which is the default: see config/crowd.yaml for the measurement that decided
#: it (hands-on verdicts separate builds at between/within 7.17 against 3.66 for
#: feed-only verdicts, with less run-to-run noise despite a quarter the sample).
DEFAULT_TRIERS = _config.crowd_triers(-1)

#: Agents who hold the app tools but were told not to use them unless the feed
#: convinces them. 0 disables the tier.
#:
#: Their conversion rate -- how many go and try it because of what they read --
#: is the only virality measurement the simulation EARNS rather than computes
#: from counts. It exists because comparing each agent's first-hand verdict with
#: its post-discussion verdict across 44 runs found the fraction talked INTO an
#: app was 0.00 in every one: once everybody has tried it, discussion can sink
#: an app but can never lift it, so half of word-of-mouth was unobservable.
DEFAULT_LATECOMERS = _config.crowd_latecomers(0)

#: Rounds of interaction after the launch is seeded. Round 1 is hands-on.
DEFAULT_ROUNDS = _config.crowd_rounds(3)

#: Reproducible crowd selection + tiering.
DEFAULT_SEED = _config.crowd_seed(0)

#: Run the end-of-run interview (the primary scoring signal).
DEFAULT_INTERVIEW = _config.crowd_interview(True)

#: Crowd sampling temperature. Higher gives livelier, more varied social
#: behaviour; lower makes the measurement more repeatable.
DEFAULT_TEMPERATURE = _config.crowd_temperature(0.7)

#: Output token ceiling per crowd LLM call. ``None`` -- the default -- sends no
#: ``max_output_tokens`` at all, so the model's own maximum applies.
#:
#: This used to be 8192, which was wrong in two ways. recent Gemini charges
#: thinking tokens against ``max_output_tokens``, so it was a *combined*
#: think-plus-answer budget, and a long deliberation truncated the reply
#: mid-sentence. And the crowd's job is rich, differentiated reactions: a cap
#: here trades away exactly the signal this benchmark exists to measure.
DEFAULT_MAX_TOKENS = _config.crowd_max_tokens(None)

#: Max concurrent crowd LLM requests.
DEFAULT_SEMAPHORE = _config.crowd_semaphore(12)

#: Hard cap on interaction steps within one trier's trial.
DEFAULT_TRIAL_MAX_STEPS = _config.crowd_trial_max_steps(24)

#: Tool-call rounds a trier is allowed for its SOCIAL turn -- posting, commenting,
#: liking, following -- on top of whatever the hands-on trial consumed.
#:
#: A trier spends one OASIS step doing both: it drives the app AND says something
#: about it, out of a single ``max_iteration`` budget. That makes the two compete,
#: and the app always wins because the trial runs first. Under the original
#: budget of 10 this was severe -- triers spending >=10 calls on the app averaged
#: 0.05 social actions and 37 of 39 were silenced outright, which for a VIRALITY
#: benchmark is the worst possible failure: the only agents with first-hand
#: experience are the ones who never get to talk about it.
#:
#: Re-measured over 600 triers on arch v8 it no longer reproduces (0% silenced,
#: mean 1.61 posts/comments each) because the budget was raised to 30 while the
#: median trial is 5 steps. But the margin was luck, not design: the worst
#: observed trier used 25 of 30, and a trial is separately allowed up to
#: ``DEFAULT_TRIAL_MAX_STEPS`` (24), so a single thorough trial can still starve
#: the voice with 6 calls to spare. Silently -- a truncated agent simply stops.
#:
#: So the budget is DERIVED rather than guessed: a trier always gets its full
#: trial allowance plus this headroom, which makes "a thorough trial can never
#: silence a trier" a property of the arithmetic instead of a property of the
#: workload happening to stay small.
DEFAULT_SOCIAL_HEADROOM = _config.crowd_social_headroom(16)

#: Per-step tool-call budget for a trier. See DEFAULT_SOCIAL_HEADROOM for why
#: this is computed rather than set.
DEFAULT_MAX_ITERATION_TRIER = DEFAULT_TRIAL_MAX_STEPS + DEFAULT_SOCIAL_HEADROOM

#: Per-step tool-call budget for a reactor. Reactors never run a trial -- they
#: read the feed and react -- so their budget is purely social.
DEFAULT_MAX_ITERATION_REACTOR = _config.crowd_max_iteration_reactor(8)

#: How long to wait for a web app to actually answer HTTP before giving up.
DEFAULT_START_WAIT = _config.crowd_start_wait(90.0)

#: Successful app interactions (click / type / select / upload / press) a trier
#: must have made before ``finish_trial`` accepts a verdict without pushing back.
#:
#: **ADVISORY, not enforced, and measured to be a no-op.** The guard fires once
#: per trial and the next call always goes through, because "I tried and could
#: not operate it" has to stay recordable. Raising it therefore changes nothing:
#: matched arms at 1, 3 and 5 produce 3.12 / 3.15 / 3.18 substantive actions per
#: trial, 25% of trials reaching 5+ actions in all three, and identical craft
#: (7.80 / 7.81 / 7.80), delight and adoption.
#:
#: That is not a defect to fix, on the evidence. Verdicts are already invariant
#: to depth on the apps where depth is cheap -- client-side trials at 1-2
#: actions rate delight 6.98 against 7.02 at 3-5 -- so forcing more interaction
#: would buy cost, not signal. It is kept as a knob because it is the honest
#: place to change this if a future app corpus needs it.
DEFAULT_MIN_INTERACTIONS = _config.crowd_min_interactions(1)

#: Tools withheld from the crowd, by name. An ablation lever: removing a verb
#: and re-measuring is the only way to know what it was contributing.
DEFAULT_DISABLED_TOOLS: tuple[str, ...] = ()

#: Show the "this environment provides GEMINI_API_KEY" notice on open.
DEFAULT_ENV_NOTICE = _config.crowd_env_notice(True)

#: How many interest-neighbours each agent follows (0 = hub-and-founder only).
DEFAULT_FOLLOW_PEERS = _config.crowd_follow_peers(3)
