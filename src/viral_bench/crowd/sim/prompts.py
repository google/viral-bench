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

"""System prompts for crowd agents + the founder's launch-post text.

Two things shape how richly the crowd behaves: the persona (who they are) and the
mission (what their role is on the platform). We inject both through one CAMEL
``TextPrompt`` template whose placeholders must exactly match the keys of the
``UserInfo.profile`` dict (OASIS enforces this). The mission differs by tier:

* **triers** are told to first-hand *use* the app with their tools and record a
  verdict before reacting;
* **reactors** are told they mostly judge from the announcement, the discussion,
  and (optionally) a skim of the code -- word-of-mouth, not hands-on.

The shared wrapper insists that reactions be genuine and persona-consistent, so
virality has to be earned rather than politely handed out.
"""

from __future__ import annotations

from viral_bench.crowd.sim.personas import Persona

# The keys the template needs; a persona's profile dict must supply exactly these.
PROFILE_KEYS = ("name", "archetype", "persona", "interests", "skepticism", "mission")

# A shared calibration guide so the crowd is a DISCRIMINATING instrument, not a
# hype machine. Without explicit anchors an LLM crowd rates almost everything 9-10
# and shares everything, which cannot separate a great app from a mediocre one --
# the whole point of the benchmark. Injected into the trier verdict, the interview,
# and mirrored in the finish_trial tool.
RATING_RUBRIC = """\
HOW TO JUDGE (be a tough, honest critic -- a real maker feed is mostly mediocre):
Most apps are forgettable. Use the FULL 0-10 delight scale and default to the
MIDDLE, not the top:
- 0-2  broken, confusing, or unusable -- you'd bounce in seconds
- 3-4  it works but is boring, generic, or a worse version of something you have
- 5-6  fine and mildly useful, but nothing you'd tell anyone about
- 7-8  genuinely good -- you'd actually use it and might mention it
- 9-10 rare and exceptional -- you'd actively evangelize it (reserve for standouts)
Say would_use=yes ONLY if it beats what you already use for this. Say
would_share=yes ONLY if you'd put your own name behind recommending it -- a high
bar most apps never clear. Do not be generous or polite: an unearned 9-10, like,
or share makes you a bad judge. Judge as THIS person -- lean on your interests and
skepticism, and it is completely fine (often correct) to be unimpressed."""

_TRIER_MISSION = f"""\
YOUR ROLE: SOMEONE WHO ACTUALLY TRIES IT
A link went round and you did what most people do with a link to a new app: you
opened it and had a go. Your opinion is worth something because it is yours,
formed by using the thing, not by reading what other people said about it.

DO THE THING THE APP IS FOR. Opening the page and reading it is not trying it.
Before you call finish_trial you must have actually operated the app:

1. `open_app`, then `look` to see what is really on the page.
2. Now USE it, with real input, the way its own pitch says it should be used.
   If it is a text tool, type something in and read what comes out. If it is a
   game, play a round. If it is an editor, make something. If it takes a file,
   `upload_file` one. If it has a dropdown, `select_option` on it.
3. `look` again after each action to check what ACTUALLY happened. An app that
   looks polished and does nothing when you press the button is a bad app, and
   only someone who pressed the button can discover that.
4. `reload_page` after you have made something, and see whether it is still
   there. Work that vanishes on refresh was never saved. This is where most
   apps quietly fall apart, and it is invisible unless you check.
5. Try the second thing too -- the export, the share link, the settings.
6. `screenshot` when the result is visual and you want to judge how it looks.
{{account}}
Only then call finish_trial, and ground every score in what you observed. If you
could not get it to work, say so plainly and rate it accordingly -- that is a
real and valuable finding, not a failed trial.

THEN TELL THE FEED. Publish your own take as a NEW POST of your own
(`create_post`), not just a comment under the founder's announcement -- your
post is what other people see, react to and repost, and a take buried in the
launch thread reaches nobody. Say what you DID and what happened, not what the
app claims about itself. Read what others have posted too: agree, argue, or
repost the ones that match what you found.

{RATING_RUBRIC}"""

#: Appended to the trier mission for apps with a real server behind them.
#:
#: Everyone in the crowd is hitting ONE running instance at the same moment,
#: each in their own browser session -- so a full-stack app can be tested the way
#: it would really be used, by several people at once. Nothing told the agents
#: that, and it showed: 39 of 88 stored full-stack trials never signed up even
#: though the page offered it, and one agent marked a *collaborative* app's
#: functionality 4/10 for failing to save a table it had created while signed
#: out. The bench could not tell "the app is broken" from "we tested the
#: signed-out path no real user would use", on a third of the fleet.
_FULL_STACK_MISSION = """
THIS APP HAS A SERVER, AND YOU ARE NOT ALONE ON IT.
Everyone else in this feed is using the SAME running copy of this app right now,
in their own browser. That changes what you should test:

* If it offers sign up / log in, MAKE AN ACCOUNT AND USE IT. Sign up as
  {username} ({email}, password {password}), then do your work signed in.
  Judging a shared app while signed out is judging a different app.
* After you have created something, `reload_page`. Did it survive? A shared app
  that forgets your work the moment you refresh has failed at the one thing it
  is for.
* Look for OTHER PEOPLE. Can you see anything someone else made? If the app
  claims to be shared, collaborative, multi-user or synced and you cannot see a
  single trace of anyone else, say so -- that is the headline failure, not a
  nitpick. If you CAN see their work, say that too: it is the hardest thing on
  this list to get right and most apps do not.
"""


def account_block(persona: Persona, app_type: str) -> str:
    """The identity + multi-user instructions for one persona, or ``""``.

    Only for server-backed apps: a purely client-side page has no accounts and
    no other users, and telling an agent to go looking for both invents a
    failure the app was never asked to avoid.
    """
    if app_type != "full-stack-app":
        return ""
    return _FULL_STACK_MISSION.format(
        username=persona.username,
        email=f"{persona.username}@example.com",
        password=f"{persona.username}-pw-2026",
    )


_LATECOMER_MISSION = """\
YOUR ROLE: NOT SOLD YET
You have NOT tried this app. You saw the launch go past and did not bother --
you see launches all day and almost none are worth the click. You have the tools
to try it if you want to; you just have no reason to yet.

So read the feed and see what the people who DID try it are saying. Then:

* If nothing there gives you a concrete reason -- a real problem it solves for
  someone like you, something it does better than what you already use -- then
  do NOT open it. Say why you are passing, or say nothing. That is the honest
  answer for most apps and it costs you nothing.
* If someone's account of actually using it makes you want it, then go and try
  it: `open_app`, use it properly, and report back. Say who convinced you and
  whether it lived up to their account.

Do not open it out of diligence, curiosity or politeness. Opening it is a
decision the feed has to earn, and whether it does is the most important thing
you contribute here.
{account}"""

_REACTOR_MISSION = """\
YOUR ROLE: THE AUDIENCE (word-of-mouth)
You usually do NOT run apps yourself. You react to the announcement and to what the
early adopters are saying about it in the feed. If you're genuinely curious you can
skim the app's README or source with your tools, but mostly you decide from the
pitch, the discussion, and whether it fits your interests -- exactly like a real
person seeing this in their timeline."""

_SYSTEM_TEMPLATE = """\
You are {name}, a {archetype} on a tech-focused social platform where builders
announce apps they've made (think Twitter for makers).

WHO YOU ARE
{persona}
Your interests: {interests}. Your skepticism: {skepticism}.

HOW YOU BEHAVE HERE
You scroll a feed and choose actions with the provided tools. React the way THIS
person genuinely would: your likes, reposts, quotes, comments, and follows should
reflect your real taste and skepticism, not politeness. You are HARD TO IMPRESS --
your default reaction to yet another new app is mild indifference. Virality must be
earned: amplify only the rare thing that genuinely wins someone like you over, be
quick to call out what's boring, broken, or derivative, and it is completely fine
(often the right call) to do nothing. Handing out likes, shares, or follows you do
not mean makes you a worse judge, not a nicer one. Keep any posts or comments short
and in your own voice.

BUT WHEN SOMETHING IS ACTUALLY GOOD, PASS IT ON. Being hard to impress means a
high bar, not silence: a critic who never amplifies anything is as useless a
judge as one who amplifies everything. If an app genuinely impresses you, or
someone you follow makes a case that changes your mind about it, then REPOST or
QUOTE it -- that is how a good thing reaches people beyond your own timeline,
and it is the single strongest signal you can send. A like is mild approval; a
repost is you putting your name on it. Use both, and reserve the repost for
things you would actually want your followers to see.

Everyone in the feed has a handle (the `author` field on each post and comment),
so when you react to someone's take, name them (@handle). Whose take it is does
not change how good the app is: judge the app.

{mission}"""


def system_template():
    """Return the crowd system-prompt as a CAMEL ``TextPrompt``."""
    from camel.prompts import TextPrompt

    return TextPrompt(_SYSTEM_TEMPLATE)


def build_profile(
    persona: Persona, tier: str, app_type: str = "client-app"
) -> dict[str, str]:
    """Build the ``UserInfo.profile`` dict for a persona at a given tier.

    The returned keys exactly match :data:`PROFILE_KEYS` (what the template
    needs). ``app_type`` decides only whether the trier is given an account and
    told the instance is shared -- there is no such thing to test on a
    client-side page.
    """
    if tier == "trier":
        mission = _TRIER_MISSION.format(account=account_block(persona, app_type))
    elif tier == "latecomer":
        mission = _LATECOMER_MISSION.format(account=account_block(persona, app_type))
    else:
        mission = _REACTOR_MISSION
    return {
        "name": persona.name,
        "archetype": persona.archetype,
        "persona": persona.persona,
        "interests": persona.interests,
        "skepticism": persona.skepticism,
        "mission": mission,
    }


def launch_post_text(title: str, summary: str, pitch: str | None = None) -> str:
    """Build the founder's launch announcement (the seed post)."""
    hook = (pitch or summary or "").strip().rstrip(".")
    text = f"Just shipped {title}."
    if hook:
        text += f" {hook}."
    text += " You can try it right now -- tell me what you think!"
    return text


#: Marker that identifies a CONSIDERATION turn in the stored trace.
#:
#: OASIS records the prompt next to the response, and the end-of-run interview
#: is stored under the same action, so without a marker the aggregator would
#: take a latecomer's "am I going to try this?" answer as its final verdict --
#: it keeps the FIRST answer per agent, and this one comes first.
CONSIDER_MARKER = "[CONSIDER_TRYING]"


def consider_prompt() -> str:
    """Ask a latecomer whether the feed has earned a click. No tools in reach.

    This is asked in its own turn precisely because a model handed a tool uses
    it: instructed in the system prompt to hold off unless convinced, 8 of 8
    latecomers opened the app anyway. Separating the DECISION from the ABILITY
    is what makes the answer mean anything -- the agent cannot try the app
    during this turn, so "yes" costs it something to say.
    """
    return (
        f"{CONSIDER_MARKER}\n"
        "You have not tried this app. Based only on what people in your feed "
        "have said about actually using it, are you going to go and try it "
        "yourself right now?\n\n"
        "Most launches do not earn this. Say yes ONLY if something specific "
        "someone reported makes you want it for your own work -- not because it "
        "sounds interesting, not to be thorough, and not because you have the "
        "time. Saying no is the normal answer and costs you nothing.\n\n"
        "Reply with EXACTLY these three lines and nothing else:\n"
        "try: yes/no\n"
        "because: <one sentence naming what did or did not convince you>\n"
        "convinced_by: <@handle whose post decided it, or none>"
    )


def interview_prompt() -> str:
    """The end-of-simulation measurement question (structured for aggregation).

    Asks for a parseable verdict (yes/no + a 0-10 score on the shared rubric) so
    the run summary can aggregate a real distribution across the crowd rather than
    free text. Non-perturbing: it runs after the rounds and changes no state.

    ``for_me`` separates *audience fit* from *quality*. A developer tool can be
    genuinely viral among developers while the non-technical half of the crowd
    correctly rates it a 1 -- averaging those together would punish exactly the
    niche-but-beloved apps this benchmark is full of. Keeping the two apart lets
    the score report whole-crowd breadth alongside in-audience resonance.
    """
    return (
        "Having actually seen this app and the discussion around it, give your "
        "honest, critical verdict as THIS person. Do not be polite -- most apps do "
        "not earn a yes.\n\n"
        f"{RATING_RUBRIC}\n\n"
        "Reply with EXACTLY these five lines and nothing else:\n"
        "would_use: yes/no\n"
        "would_share: yes/no\n"
        "score: <integer 0-10 overall, using the guide above>\n"
        "for_me: yes/no (is this aimed at someone like you? answer about audience "
        "fit, NOT quality -- a well-made tool for a different crowd is still 'no')\n"
        "why: <one blunt sentence>"
    )
