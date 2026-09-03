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

"""Pytest wiring for the viewer's tests.

Everything substantial lives in :mod:`vizfixtures`; this file only re-exports the
fixtures so pytest can find them.

That split is not tidiness, it is collision avoidance. Pytest imports test modules
by bare module name with their directory on ``sys.path``, so a suite dropped into a
repo that already has a ``tests/*/conftest.py`` will find *that* one for a plain
``from conftest import ...`` -- whichever was imported first wins, silently and
confusingly. Helpers therefore live under a name nothing else is likely to use, and
the test modules are prefixed ``test_viz_`` for the same reason. Verified against
the benchmark repo, whose ``tests/crowd/`` has both a ``conftest.py`` and a
``test_trace.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vizfixtures import builds, traced  # noqa: E402,F401
