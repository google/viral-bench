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

"""The agent-facing tool surface: what a crowd agent is actually handed.

:class:`AppInteractionToolkit` turns the interaction layer into a small set of
**tools an LLM agent calls** to try one app. Each tool returns a plain string
(the observation, ready to read) and records a step on the shared
:class:`~viral_bench.crowd.interaction.trace.InteractionTrace`, so the agent's
transcript *is* the evidence.

Every app is a web app, so there is one tool set:

* ``open_app``, ``look``, ``click``, ``type_text``, ``press_key``,
  ``upload_file``, ``screenshot``
* ``finish_trial`` -- the agent's structured "would I use/share this?" verdict,
  and the handoff into scoring

Two ways to use it:

* hand the tools to an autonomous agent --
  :meth:`AppInteractionToolkit.as_camel_tools` wraps them as CAMEL
  ``FunctionTool``\\s that drop straight into ``SocialAgent(tools=...)``; or
* drive a scripted / default trial without an agent -- :func:`try_app` (used by
  the CLI, tests, and the founder's ``verify.try_app``).

Everything is ``async`` (the crowd loop is async, and the browser must be); the
CLI and tests wrap calls in ``asyncio.run``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from viral_bench.crowd.interaction.browser import BrowserConfig, BrowserEngine
from viral_bench.crowd.interaction.clients import AppClient, Observation
from viral_bench.crowd.interaction.fixtures import available_fixtures
from viral_bench.crowd.interaction.imagery import mark_image
from viral_bench.crowd.interaction.session import manifest_or_none, open_trial
from viral_bench.crowd.interaction.trace import InteractionTrace, TrialVerdict
from viral_bench.founder.apphost import AppHost

# App types. Both are web apps -- the scope says how much lives on the server --
# so they share one client, one tool set, and one rubric. The split survives as a
# label for analysis, not as a code path.

#: Shown when a verb needs a real browser and this trial has not got one (the
#: degraded static-HTTP fallback can fetch a page but cannot drive it). The
#: warning matters: an agent that cannot click must not conclude the app is
#: broken, or a harness limitation is scored as a product defect.
_NO_BROWSER_MSG = (
    "Cannot {action}: this trial is running without a browser, so only the page's "
    "served HTML is visible. This is a limitation of how you are viewing the app, "
    "NOT a fault in the app -- do not describe or rate it as broken on this basis."
)

_LOG = logging.getLogger("viral_bench.crowd.toolkit")


class TrialLocked(RuntimeError):
    """Raised inside a locked trial; surfaced to the agent as a plain refusal."""


#: Top of the rubric every 0-10 verdict field is scored on.
_FACET_MAX = 10

_WEB = "client-app"
_FULL_STACK = "full-stack-app"

#: What the four craft facets MEAN. One definition for the whole fleet.
#:
#: These were previously per-app-type, because the corpus was a third terminal
#: programs and a web-shaped description ("how it looks and feels") reads as
#: inapplicable to a CLI -- and an LLM asked to rate an inapplicable facet omits
#: the whole optional set. Measured over the stored corpus at the time: 259/293
#: single-page-app trials carried facets and 0 of 25 cli+bot trials did, so their
#: craft silently collapsed to a single ``delight`` item, a different measurement
#: wearing the same key. With every idea now a web app the hazard is gone and one
#: definition is the honest one.
_FACET_GUIDE: dict[str, str] = {
    "functionality": "Does it actually work and do what it claims, 0-10.",
    "usability": "Could you figure it out and get the job done, 0-10.",
    "design": "Visual and interaction craft -- how it looks and feels, 0-10.",
    "simplicity": "Focused and friction-free rather than bloated, 0-10.",
}

_FINISH_DOC = """End the trial and record your verdict about the app.

        Call this ONCE, after you have actually used the app, to report whether
        you -- as this user -- would adopt and share it. Be a tough, honest critic:
        most apps are forgettable, so use the full scale and do not inflate. The
        verdict is final: calling this again later does not change it.

        Rate every 0-10 field on the same scale: 0-2 broken or unusable; 3-4
        works but boring or derivative; 5-6 fine but not worth mentioning; 7-8
        genuinely good; 9-10 rare and exceptional. Default to the middle and
        reserve 9-10 for standouts. Score the facets independently -- an ugly app
        can be highly functional, and a beautiful one can be useless.

        ALL FOUR facet scores are REQUIRED. They are defined below for THIS kind
        of app, so every one of them applies -- do not leave any of them out.

        Args:
            would_use: Keep using this app? Only yes if it beats what you already
                use for this.
            would_share: Share/repost it to others? A high bar -- only yes if you'd
                put your own name behind recommending it.
            delight: Overall, how delightful was it, 0-10.
            notes: A short, blunt justification grounded in what you saw/did.
            functionality: {functionality}
            usability: {usability}
            design: {design}
            simplicity: {simplicity}
            work_survived: Did what you made still exist after `reload_page`?
                1 yes, 0 no, -1 you did not check. Answer honestly -- -1 is
                fine and is not the same as "no".
            saw_other_users: Could you see anything ANOTHER person had made in
                this same app? 1 yes, 0 no (and you looked), -1 not applicable
                or you did not look.
        """


def finish_trial_doc(app_type: str) -> str:
    """The ``finish_trial`` description an agent sees.

    Takes ``app_type`` for call-site compatibility; the rubric is now the same
    for every app, because every app is a web app.
    """
    del app_type  # one rubric for the whole fleet
    return _FINISH_DOC.format(**_FACET_GUIDE)


class AppInteractionToolkit:
    """A per-build bundle of app-interaction tools for one agent's trial.

    Construct it with a ``build_id`` (and, for the crowd, a shared ``AppHost`` and
    ``BrowserEngine``); call :meth:`tools` / :meth:`as_camel_tools` to get the
    agent-callable functions. The underlying app trial is opened lazily on the
    first tool call and released by :meth:`close`.
    """

    def __init__(
        self,
        build_id: str,
        *,
        app_host: AppHost | None = None,
        container: bool = True,
        browser_engine: BrowserEngine | None = None,
        browser_config: BrowserConfig | None = None,
        use_browser: bool | None = None,
        env_map: dict[str, str] | None = None,
        max_steps: int = 40,
        start_wait: float = 90.0,
        app_type: str | None = None,
        locked: bool = False,
        min_interactions: int = 1,
        disabled_tools: tuple[str, ...] = (),
        env_notice: bool = True,
    ) -> None:
        self.build_id = build_id
        # A build whose manifest is missing or malformed has no app_type of its
        # own, and reading one used to raise here -- in the constructor, before
        # any agent existed, which is why such builds could not be simulated at
        # all. The caller passes the idea's declared scope instead, so an
        # undeliverable build still gets a crowd that tries it and finds nothing.
        manifest = manifest_or_none(build_id)
        self.undeliverable = manifest is None
        self.app_type = app_type or (
            manifest.app_type if manifest is not None else _WEB
        )
        self._app_host = app_host
        self._container = container
        self._browser_engine = browser_engine
        self._browser_config = browser_config
        self._use_browser = use_browser
        self._env_map = env_map
        self.max_steps = max_steps
        self._start_wait = start_wait

        #: A locked trial refuses every app verb. It exists for the latecomer
        #: tier: an agent that has NOT decided to try the app must not be able
        #: to drift into trying it. Told in the prompt to hold off, 8 of 8
        #: latecomers opened the app anyway -- a model handed a tool uses it --
        #: so the decision is taken in a separate turn with no tool in reach,
        #: and only then is the trial unlocked.
        self._locked = locked
        #: How many successful interactions a verdict needs behind it before
        #: finish_trial stops pushing back. See DEFAULT_MIN_INTERACTIONS.
        self._min_interactions = max(0, int(min_interactions))
        #: Verbs withheld from this agent (an ablation lever).
        self._disabled = frozenset(disabled_tools)
        self._env_notice = env_notice
        self._client: AppClient | None = None
        self._trace = InteractionTrace(build_id=build_id, app_type=self.app_type)
        self._finished = False
        #: Whether we have already asked this agent to actually use the app.
        self._nudged = False

    # -- lifecycle ----------------------------------------------------------

    @property
    def trace(self) -> InteractionTrace:
        return self._trace

    _LOCKED_MSG = (
        "You have not decided to try this app, so you have not opened it. "
        "Nothing here is available to you. React to the feed as someone who has "
        "not used it -- do not describe or rate the product itself."
    )

    @property
    def locked(self) -> bool:
        return self._locked

    def unlock(self) -> None:
        """Let this agent use the app: it decided the feed had earned it."""
        self._locked = False

    async def _ensure_open(self) -> AppClient:
        if self._locked:
            raise TrialLocked(self._LOCKED_MSG)
        if self._client is None:
            self._client = await open_trial(
                self.build_id,
                app_host=self._app_host,
                container=self._container,
                browser_engine=self._browser_engine,
                browser_config=self._browser_config,
                use_browser=self._use_browser,
                env_map=self._env_map,
                start_wait=self._start_wait,
                trace=self._trace,
                undeliverable_app_type=self.app_type,
                env_notice=self._env_notice,
            )
            # Auto-initialize the interaction surface so the first real tool call
            # already has something to observe.
            await self._client.open()
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def __aenter__(self) -> AppInteractionToolkit:
        await self._ensure_open()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    def _over_budget(self) -> bool:
        return self._trace.n_steps >= self.max_steps

    _BUDGET_MSG = (
        "Interaction step budget reached. Call finish_trial(...) now with your "
        "verdict based on what you have already seen."
    )

    # -- web tools ----------------------------------------------------------

    async def open_app(self) -> str:
        """Open the app in the browser and describe what is on screen.

        Returns the page title, visible text, and the controls you can interact
        with. Use this first for a web app.
        """
        if self._locked:
            return self._LOCKED_MSG
        client = await self._ensure_open()
        return (await client.observe()).render()

    async def look(self) -> str:
        """Re-read the current screen: title, visible text, controls, and any
        console errors. Use it after an action to see what changed."""
        if self._locked:
            return self._LOCKED_MSG
        client = await self._ensure_open()
        return (await client.observe()).render()

    async def click(self, target: str) -> str:
        """Click a control identified by its visible text or accessible name.

        Args:
            target: What to click, as a person would name it, e.g. ``"Roll"``,
                ``"Start game"``, or a CSS selector like ``"css=#submit"``.

        Returns the resulting screen state (so you can see what the click did).
        """
        if self._locked:
            return self._LOCKED_MSG
        client = await self._ensure_open()
        if self._over_budget():
            return self._BUDGET_MSG
        if not hasattr(client, "click"):
            return _NO_BROWSER_MSG.format(action="click")
        return (await client.click(target)).render()

    async def type_text(self, target: str, text: str) -> str:
        """Type ``text`` into an input identified by its label/placeholder.

        Args:
            target: The field to type into, e.g. ``"Search"`` or ``"your name"``.
            text: The text to enter.

        Returns the resulting screen state.
        """
        if self._locked:
            return self._LOCKED_MSG
        client = await self._ensure_open()
        if self._over_budget():
            return self._BUDGET_MSG
        if not hasattr(client, "type_text"):
            return _NO_BROWSER_MSG.format(action="type")
        return (await client.type_text(target, text)).render()

    async def press_key(self, key: str) -> str:
        """Press a keyboard key (e.g. ``"Enter"``, ``"ArrowUp"``, ``"a"``).

        Useful for games and shortcuts. Returns the resulting screen state.
        """
        if self._locked:
            return self._LOCKED_MSG
        client = await self._ensure_open()
        if self._over_budget():
            return self._BUDGET_MSG
        if not hasattr(client, "press_key"):
            return f"This app is a {self.app_type}; key presses do not apply."
        return (await client.press_key(key)).render()

    async def select_option(self, target: str, value: str) -> str:
        """Choose a value from a dropdown / ``<select>`` menu.

        Clicking a dropdown does not choose anything, so without this an app
        whose main control is a menu -- pick a format, pick a theme, pick a
        target language -- cannot be operated at all.

        Args:
            target: The dropdown, by its label, e.g. ``"Export format"``.
            value: The option to choose, by its visible text or value.

        Returns the resulting screen state.
        """
        if self._locked:
            return self._LOCKED_MSG
        client = await self._ensure_open()
        if self._over_budget():
            return self._BUDGET_MSG
        if not hasattr(client, "select_option"):
            return _NO_BROWSER_MSG.format(action="use a dropdown")
        return (await client.select_option(target, value)).render()

    async def reload_page(self) -> str:
        """Reload the app in your browser and report what is still there.

        This is how you find out whether your work was actually SAVED. Make
        something first, then reload: if it is still there the app really
        persisted it, and if it vanished the app only ever held it in memory.
        For anything that claims to store, share or sync your data, this is the
        single most important thing you can check -- and other people using this
        same app right now may have added things you will only see after a
        reload.
        """
        if self._locked:
            return self._LOCKED_MSG
        client = await self._ensure_open()
        if self._over_budget():
            return self._BUDGET_MSG
        if not hasattr(client, "reload"):
            return _NO_BROWSER_MSG.format(action="reload the page")
        return (await client.reload()).render()

    async def upload_file(self, fixture: str, target: str = "") -> str:
        """Upload one of the sample files to the app, for apps that take a file.

        Many apps open with "drop in a photo / PDF / screenshot" -- use this to
        get past that step. Available sample files:

        {fixtures}

        Args:
            fixture: Which sample file to upload, e.g. ``"photo.png"``.
            target: Optional. The upload control's label, if the page has more
                than one. Leave empty to use the first file input on the page --
                which is usually right, since upload buttons are often styled
                labels with no accessible name.

        Returns the resulting screen state.
        """
        if self._locked:
            return self._LOCKED_MSG
        client = await self._ensure_open()
        if self._over_budget():
            return self._BUDGET_MSG
        if not hasattr(client, "upload_file"):
            return _NO_BROWSER_MSG.format(action="upload a file")
        return (await client.upload_file(fixture, target or None)).render()

    async def screenshot(self, note: str = "") -> str:
        """Look at the app as an IMAGE, to judge how it actually looks.

        Use this when the visual result is the point -- layout, colour, spacing,
        a rendered chart, a generated image, anything drawn on a canvas. The text
        and ARIA snapshot you get from `look` cannot show you any of that.

        Args:
            note: A short label for the shot, e.g. ``"after winning"``.
        """
        if self._locked:
            return self._LOCKED_MSG
        client = await self._ensure_open()
        if not hasattr(client, "screenshot"):
            return f"This app is a {self.app_type}; screenshots do not apply."
        obs = await client.screenshot(note)
        # The rendered string carries a marker naming the PNG; the model
        # transport turns that into a real image part. Tools must return str, so
        # the image cannot simply be returned here.
        return mark_image(obs.render(), obs.screenshot)

    # -- universal ----------------------------------------------------------

    #: Actions that count as genuinely operating the app, as opposed to reading
    #: it. `look`, `reload` and `screenshot` are observation, not use.
    _INTERACTIONS = ("click", "type", "press", "upload", "select")

    def n_interactions(self) -> int:
        """How many interactions with the app have SUCCEEDED."""
        return sum(
            1
            for step in self._trace.steps
            if step.action in self._INTERACTIONS
            and "could not" not in (step.summary or "").lower()
        )

    def _did_interact(self) -> bool:
        """True if the trial has cleared its interaction floor."""
        return self.n_interactions() >= self._min_interactions

    def _app_is_reachable(self) -> bool:
        """True if the app actually served something to this agent."""
        return bool(getattr(self._trace, "app_reachable", False))

    def _ever_opened(self) -> bool:
        """True if this agent ever tried to open the app at all."""
        return any(step.action == "open" for step in self._trace.steps)

    async def finish_trial(
        self,
        would_use: bool,
        would_share: bool,
        delight: int,
        notes: str = "",
        functionality: int = -1,
        usability: int = -1,
        design: int = -1,
        simplicity: int = -1,
        work_survived: int = -1,
        saw_other_users: int = -1,
    ) -> str:
        """End the trial and record your verdict about the app.

        Call this ONCE, after you have actually used the app, to report whether
        you -- as this user -- would adopt and share it. Be a tough, honest critic:
        most apps are forgettable, so use the full scale and do not inflate. The
        verdict is final: calling this again later does not change it.

        Rate every 0-10 field on the same scale: 0-2 broken or unusable; 3-4
        works but boring or derivative; 5-6 fine but not worth mentioning; 7-8
        genuinely good; 9-10 rare and exceptional. Default to the middle and
        reserve 9-10 for standouts. Score the facets independently -- an ugly app
        can be highly functional, and a beautiful one can be useless.

        Args:
            would_use: Keep using this app? Only yes if it beats what you already
                use for this.
            would_share: Share/repost it to others? A high bar -- only yes if you'd
                put your own name behind recommending it.
            delight: Overall, how delightful was it, 0-10.
            notes: A short, blunt justification grounded in what you saw/did.
            functionality: Does it actually work and do what it claims, 0-10.
            usability: Could you figure it out and get the job done, 0-10.
            design: Visual and interaction craft -- how it looks and feels, 0-10.
            simplicity: Focused and friction-free rather than bloated, 0-10.
            work_survived: Did what you made still exist after `reload_page`?
                1 yes, 0 no, -1 you did not check.
            saw_other_users: Could you see anything ANOTHER person had made in
                this same shared app? 1 yes, 0 no, -1 not applicable.
        """
        # The verdict is recorded ONCE and is final. An agent that is re-activated
        # in a later round (with the app already tried) tends to call this again
        # with a contentless "already did this" note -- which used to overwrite the
        # real, evidence-grounded verdict and could even flip would_use/share.
        if self._finished:
            existing = self._trace.verdict
            if existing is not None:
                return (
                    "Your verdict for this app is already recorded "
                    f"(would_use={existing.would_use}, "
                    f"would_share={existing.would_share}, "
                    f"delight={existing.delight}/10) and is final. "
                    "Do not call finish_trial again; just act on the feed."
                )

        # Push back ONCE on a verdict reached without using the app.
        #
        # Measured over 792 trials on the solo fleet: 284 triers (35.9%) judged
        # an app without a single successful click, type, upload or keypress, and
        # 218 of those never even attempted one -- the modal trajectory was
        # `open -> look -> finish`. They were rating a landing page. It shows in
        # the scores: those triers gave delight 5.94 / would_use 0.55, against
        # 7.05 / 0.81 for triers who interacted 3-5 times, so the benchmark's
        # hands-on signal was substantially a reading-comprehension signal.
        #
        # A trier is the only agent who CAN find out whether the app works, so
        # the refusal is worth one round-trip. It is deliberately not a hard
        # block: an app that will not load cannot be operated, and "I tried and
        # it was broken" is a real verdict we must still be able to record. The
        # second call always goes through.
        # A locked trial has no verdict to give: this agent decided not to try
        # the app, and a rating from someone who did not use it is exactly the
        # hearsay the hands-on tier exists to replace.
        if self._locked:
            return self._LOCKED_MSG

        # Never opened the app at all -- not "opened it and it was broken", but
        # went straight from being handed the tools to filing a verdict. 32 of
        # 352 trials in the stored corpus did this, all of them on two builds
        # whose app genuinely would not start, and they wrote confident craft
        # ratings reconstructed from reading the source ("the SQLite persistence
        # works robustly"). The old nudge could not catch them: it was gated on
        # the app being *reachable*, which is exactly what a trial that never
        # opened anything is not. One push-back, then the verdict stands --
        # "I could not get it to start" has to remain recordable.
        if not self._ever_opened() and not self._nudged:
            self._nudged = True
            return (
                "You have not opened this app yet -- you have not called "
                "open_app once, so you have seen nothing of the product itself. "
                "Reading the source is not using the app. Call open_app, try "
                "the thing it is for, and then record your verdict. If it will "
                "not start, call finish_trial again and say exactly that: a "
                "dead app is a real and valuable finding."
            )

        if self._app_is_reachable() and not self._did_interact() and not self._nudged:
            self._nudged = True
            done = self.n_interactions()
            return (
                f"You have used this app {done} time(s) -- that is not enough to "
                f"judge it. Do the main thing it is FOR, end to end, and check "
                f"the result: type real input and read what comes out, play an "
                f"actual round, upload a file, change a setting, follow the "
                f"export or share path. Take at least "
                f"{self._min_interactions} real actions before deciding. If you "
                f"genuinely cannot operate it, call finish_trial again and say "
                f"so: that is a real finding and will be recorded."
            )

        # -1 is the "not supplied" sentinel: the facets are optional so a
        # malformed tool call still records the core verdict rather than failing.
        #
        # Everything is CLAMPED to the 0-10 rubric. Nothing enforced the upper
        # bound before -- an agent answering `design: 100` had it recorded
        # verbatim, and craft is the mean of these divided by 10, so one
        # hallucinated value silently pushes a component above 1.0 and
        # re-weights every other term in the score. Zero of 4,954 stored
        # verdicts were out of range under the current crowd model, so this is
        # prevention rather than repair: the rubric lives in a docstring, not a
        # schema, and the first model from another family to answer on a
        # different scale would corrupt a published comparison with no error and
        # no log line.
        def _clamped(value: object, *, field: str) -> int | None:
            if value is None:
                return None
            try:
                raw = int(value)
            except (TypeError, ValueError):
                _LOG.warning(
                    "trial %s: non-numeric %s=%r, dropped", self.build_id, field, value
                )
                return None
            if raw < 0:
                return None  # the "not supplied" sentinel
            if raw > _FACET_MAX:
                _LOG.warning(
                    "trial %s: %s=%d is outside the 0-%d rubric, clamped",
                    self.build_id,
                    field,
                    raw,
                    _FACET_MAX,
                )
                return _FACET_MAX
            return raw

        # delight is required, so a missing/unparseable value falls back to the
        # midpoint rather than dropping the whole verdict.
        delight_value = _clamped(delight, field="delight")

        def _tristate(value: object) -> bool | None:
            """-1 / anything unparseable means "did not check", not "no"."""
            try:
                raw = int(value)
            except (TypeError, ValueError):
                return None
            return None if raw < 0 else bool(raw)

        verdict = TrialVerdict(
            would_use=bool(would_use),
            would_share=bool(would_share),
            delight=_FACET_MAX // 2 if delight_value is None else delight_value,
            notes=notes,
            functionality=_clamped(functionality, field="functionality"),
            usability=_clamped(usability, field="usability"),
            design=_clamped(design, field="design"),
            simplicity=_clamped(simplicity, field="simplicity"),
            work_survived=_tristate(work_survived),
            saw_other_users=_tristate(saw_other_users),
        )
        # Tell the agent, in its own turn, if it never got the app working. It
        # is about to post its take to the feed, and 7 of 8 triers on one build
        # praised "grounded citations" and "Privacy Audit Cards" on an app whose
        # every invocation exited 1 -- reconstructed from reading the source and
        # narrated as first-hand use. The verdict is still recorded (the crowd
        # is not censored), but it is marked, excluded from craft downstream, and
        # the agent is told plainly what it actually observed.
        unreachable = self._trace.app_reachable is False
        self._trace.record(
            "finish",
            args={
                "would_use": verdict.would_use,
                "would_share": verdict.would_share,
                "delight": verdict.delight,
                "functionality": verdict.functionality,
                "usability": verdict.usability,
                "design": verdict.design,
                "simplicity": verdict.simplicity,
                "craft": verdict.craft,
                "work_survived": verdict.work_survived,
                "saw_other_users": verdict.saw_other_users,
            },
            summary=notes or "(verdict recorded)",
            ok=True,
        )
        self._trace.finish(verdict)
        self._finished = True
        craft = verdict.craft
        message = (
            f"Recorded verdict: would_use={verdict.would_use}, "
            f"would_share={verdict.would_share}, delight={verdict.delight}/10"
            + (f", craft={craft}/10." if craft is not None else ".")
        )
        if unreachable:
            message += (
                " NOTE: you never got this app to actually work in this trial --"
                " nothing you ran succeeded. Your quality ratings are recorded but"
                " excluded from the hands-on evidence, because reading the source"
                " is not the same as using the app. When you post about it, say"
                " what you actually observed: that you could not get it running."
            )
        return message

    # -- tool export --------------------------------------------------------

    def _finish_trial_tool(self) -> Callable:
        """``finish_trial`` with the facet definitions for THIS app type.

        The bound method keeps one signature and one implementation; only the
        description the model reads changes, so the recorded verdict schema is
        identical for every app, so two builds' verdicts stay comparable.
        """
        import functools

        @functools.wraps(self.finish_trial)
        async def finish_trial(
            would_use: bool,
            would_share: bool,
            delight: int,
            notes: str = "",
            functionality: int = -1,
            usability: int = -1,
            design: int = -1,
            simplicity: int = -1,
            work_survived: int = -1,
            saw_other_users: int = -1,
        ) -> str:
            return await self.finish_trial(
                would_use,
                would_share,
                delight,
                notes,
                functionality,
                usability,
                design,
                simplicity,
                work_survived,
                saw_other_users,
            )

        finish_trial.__doc__ = finish_trial_doc(self.app_type)
        return finish_trial

    def tools(self) -> list[Callable]:
        """Return the agent-callable tools for driving a web app.

        ``disabled_tools`` withholds verbs by name. Removing a verb and
        re-measuring is the only way to find out what it was contributing, and
        finish_trial is never removable.
        """
        offered = [
            self.open_app,
            self.look,
            self.click,
            self.type_text,
            self.select_option,
            self.press_key,
            self.upload_file,
            self.reload_page,
            self.screenshot,
        ]
        kept = [fn for fn in offered if fn.__name__ not in self._disabled]
        return [*kept, self._finish_trial_tool()]

    def as_camel_tools(self) -> list:
        """Wrap :meth:`tools` as CAMEL ``FunctionTool``s for a ``SocialAgent``.

        Imported lazily so the interaction layer does not require CAMEL to be
        installed unless you actually wire the tools into an agent. Install the
        crowd extra (``uv sync --extra crowd``) to use this.
        """
        try:
            from camel.toolkits import FunctionTool
        except ImportError as exc:  # pragma: no cover - exercised only w/o camel
            raise ImportError(
                "camel-ai is required for as_camel_tools(); install the crowd "
                "extra with `uv sync --extra crowd`."
            ) from exc
        return [FunctionTool(tool) for tool in self.tools()]

    # -- scripted / default driving ----------------------------------------

    async def run_script(self, script: Sequence[dict]) -> InteractionTrace:
        """Run a fixed list of actions (no agent). Each item is ``{"action":
        <name>, ...args}`` naming a tool, e.g. ``{"action": "click", "target":
        "Roll"}``. Unknown actions are recorded as failed steps and skipped."""
        await self._ensure_open()
        for item in script:
            action = item.get("action", "")
            args = {k: v for k, v in item.items() if k != "action"}
            fn = getattr(self, action, None)
            if not callable(fn) or action.startswith("_"):
                self._trace.record(
                    action or "unknown",
                    args=args,
                    summary=f"unknown action {action!r}",
                    ok=False,
                )
                continue
            await fn(**args)
        return self._trace

    async def run_default(self) -> InteractionTrace:
        """Run a small "smoke" interaction with the app (no agent).

        Enough to prove the app genuinely works when driven -- the canonical
        replacement for a blind HTTP GET. Not a substitute for a real agent's
        judgement; it exists for the validity probe, the CLI, and tests.
        """
        await self._ensure_open()
        await self.look()
        # Try the first obvious control, then observe the effect.
        client = self._client
        obs: Observation | None = None
        if client is not None:
            obs = await client.observe()
        if obs and obs.elements:
            await self.click("css=button, [role=button], a[href]")
            await self.look()
        await self.screenshot("default-smoke")
        return self._trace


async def try_app(
    build_id: str,
    *,
    app_host: AppHost | None = None,
    container: bool = True,
    browser_engine: BrowserEngine | None = None,
    browser_config: BrowserConfig | None = None,
    use_browser: bool | None = None,
    env_map: dict[str, str] | None = None,
    script: Sequence[dict] | None = None,
    max_steps: int = 40,
) -> InteractionTrace:
    """Open a trial, drive it (scripted or default), and return the trace.

    This is the no-agent entry point: the canonical, human-like "use the app"
    that replaces a blind HTTP GET. Pass ``script`` to run a fixed sequence of
    actions, or leave it out for a type-appropriate default smoke interaction.
    """
    toolkit = AppInteractionToolkit(
        build_id,
        app_host=app_host,
        container=container,
        browser_engine=browser_engine,
        browser_config=browser_config,
        use_browser=use_browser,
        env_map=env_map,
        max_steps=max_steps,
    )
    try:
        if script is not None:
            await toolkit.run_script(script)
        else:
            await toolkit.run_default()
    finally:
        await toolkit.close()
    return toolkit.trace


# The upload tool advertises the sample files in its own docstring, which is
# what the agent actually reads. Filled in here so the catalogue can never
# drift from fixtures.FIXTURE_DESCRIPTIONS.
AppInteractionToolkit.upload_file.__doc__ = (
    AppInteractionToolkit.upload_file.__doc__ or ""
).replace("{fixtures}", available_fixtures().replace("\n", "\n        "))
