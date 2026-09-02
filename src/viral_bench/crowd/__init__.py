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

"""The crowd side of ViralBench: OASIS agents that try founder-built apps.

Where :mod:`viral_bench.founder` *builds* apps, :mod:`viral_bench.crowd` is the
simulated population that *uses* them and decides whether to adopt and share them
(the loop the ViralScore is read from).

This package is built in two layers:

* :mod:`viral_bench.crowd.interaction` -- give an agent hands to drive a
  running app *like a human would*, per app type (click/type through a web app in
  a real browser, run a CLI with real inputs, hold a multi-turn bot
  conversation), and record a structured evidence trace. This is the substrate
  the scoring stage consumes, and it is what this module ships first.
* the scoring/crowd-simulation layer (turning those interaction traces into
  social reactions on OASIS and a composite ViralScore) is built on top of the
  interaction layer and is added separately.

The interaction layer builds directly on the founder runtime primitives
(:mod:`viral_bench.founder.runtime`, :mod:`viral_bench.founder.apphost`,
:mod:`viral_bench.founder.runner`) so a founder-built app is exercised through
exactly the same manifest-driven runtime whether a human tests it, the founder's
QA verifies it, or the crowd tries it.
"""

from __future__ import annotations
