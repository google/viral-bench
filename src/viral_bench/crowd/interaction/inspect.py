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

"""Let a crowd agent read a founder-built app's source, like a curious dev would.

Not everyone in a crowd runs an app before reacting -- plenty of people skim the
README or glance at the code and form an opinion from that. This module gives
agents that cheap, read-only capability: a small toolkit that exposes an app's
README, its file tree, and individual files, all sandboxed to the build's app
directory.

It is deliberately much cheaper than a real trial (no browser, no container, no
LLM inside the tool), so it is the natural tool surface for the many "reactor"
agents (and a complement to the full interaction toolkit for the few "trier"
agents). Like :mod:`~viral_bench.crowd.interaction.toolkit`, everything here is a
plain callable that :meth:`CodeInspectionToolkit.as_camel_tools` wraps as CAMEL
``FunctionTool``\\s for an OASIS ``SocialAgent(tools=...)``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from viral_bench.founder.build import load_build_record

__all__ = ["CodeInspectionToolkit"]

# Directories/files that are never worth showing an agent (build scratch, VCS,
# caches, binaries) -- keeps the file tree and reads focused on real source.
_SKIP_DIRS = {
    ".git",
    ".opencode",
    "__pycache__",
    "node_modules",
    ".venv",
    "dist",
    "build",
}
_README_NAMES = ("README.md", "README", "README.txt", "readme.md")
_MAX_FILE_BYTES = 20_000
_MAX_TREE_ENTRIES = 200
_MAX_READ_CHARS = 8_000
# Extensions we treat as readable text; everything else is reported, not dumped.
_TEXT_SUFFIXES = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".html",
    ".css",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
    ".sh",
    ".rs",
    ".go",
    ".java",
    ".rb",
    ".c",
    ".h",
    ".cpp",
    ".sql",
    ".svg",
    ".env.example",
    ".gitignore",
}


class CodeInspectionToolkit:
    """Read-only view of one build's source tree, as agent-callable tools."""

    def __init__(
        self,
        build_id: str,
        *,
        app_dir: str | Path | None = None,
        max_file_bytes: int | None = None,
        max_tree_entries: int | None = None,
    ) -> None:
        self.build_id = build_id
        if app_dir is not None:
            self._root = Path(app_dir).resolve()
        else:
            record = load_build_record(build_id)
            self._root = Path(record.app_dir).resolve()
        # The crowd keeps the tight caps: a trier skimming source is a side
        # activity, and a 200-file dump would swamp its context. The rubric
        # grader raises them, because a `source` item -- "the API key appears in
        # no client-served asset" -- is a claim about the WHOLE tree, and a
        # truncated tree would silently turn a real leak into a pass.
        self._max_file_bytes = max_file_bytes or _MAX_FILE_BYTES
        self._max_tree_entries = max_tree_entries or _MAX_TREE_ENTRIES

    # -- path safety --------------------------------------------------------

    def _safe(self, rel: str) -> Path | None:
        """Resolve ``rel`` under the app root, refusing any escape."""
        candidate = (self._root / rel).resolve()
        try:
            candidate.relative_to(self._root)
        except ValueError:
            return None
        return candidate

    # -- tools --------------------------------------------------------------

    def read_readme(self) -> str:
        """Read the app's README (its pitch and how-to), if it has one."""
        if not self._root.is_dir():
            return f"(app source for {self.build_id} not found)"
        for name in _README_NAMES:
            path = self._root / name
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="replace")
                return text[:_MAX_READ_CHARS] + (
                    "\n... [truncated]" if len(text) > _MAX_READ_CHARS else ""
                )
        return "(this app has no README)"

    def list_files(self) -> str:
        """List the app's source files (a compact tree) to see how it's built."""
        if not self._root.is_dir():
            return f"(app source for {self.build_id} not found)"
        entries: list[str] = []
        for path in sorted(self._root.rglob("*")):
            if any(part in _SKIP_DIRS for part in path.relative_to(self._root).parts):
                continue
            if path.is_dir():
                continue
            rel = path.relative_to(self._root).as_posix()
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            entries.append(f"  {rel} ({size} bytes)")
            if len(entries) >= self._max_tree_entries:
                entries.append("  ... [more files omitted]")
                break
        if not entries:
            return "(no source files found)"
        return f"Files in {self.build_id}:\n" + "\n".join(entries)

    def read_file(self, path: str) -> str:
        """Read one source file by its relative path (e.g. ``app.js``, ``README.md``).

        Args:
            path: File path relative to the app root. Cannot escape the app dir.
        """
        target = self._safe(path)
        if target is None:
            return f"Refused: {path!r} is outside the app directory."
        if not target.is_file():
            return f"No such file: {path!r}. Use list_files to see what exists."
        if target.suffix and target.suffix not in _TEXT_SUFFIXES:
            return f"{path!r} is not a readable text file ({target.suffix})."
        try:
            data = target.read_bytes()[: self._max_file_bytes]
        except OSError as exc:
            return f"Could not read {path!r}: {exc}"
        text = data.decode("utf-8", errors="replace")
        oversized = target.stat().st_size > self._max_file_bytes
        suffix = "\n... [truncated]" if oversized else ""
        return f"# {path}\n{text}{suffix}"

    def grep(self, pattern: str, *, max_hits: int = 200) -> str:
        """Search every readable source file for a regular expression.

        The one primitive the crowd never needed and a grader cannot work
        without: several rubric items are whole-tree claims (no third-party CDN
        is referenced anywhere, the API key is not in any client-served asset),
        and answering those by reading files one at a time is neither reliable
        nor affordable.

        Args:
            pattern: Python regular expression, matched case-insensitively.
            max_hits: Stop after this many matching lines.
        """
        if not self._root.is_dir():
            return f"(app source for {self.build_id} not found)"
        try:
            needle = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            return f"Bad pattern {pattern!r}: {exc}"

        hits: list[str] = []
        for path in sorted(self._root.rglob("*")):
            rel_parts = path.relative_to(self._root).parts
            if any(part in _SKIP_DIRS for part in rel_parts):
                continue
            if not path.is_file():
                continue
            if path.suffix and path.suffix not in _TEXT_SUFFIXES:
                continue
            try:
                text = path.read_bytes()[: self._max_file_bytes].decode(
                    "utf-8", errors="replace"
                )
            except OSError:
                continue
            rel = path.relative_to(self._root).as_posix()
            for number, line in enumerate(text.splitlines(), start=1):
                if needle.search(line):
                    hits.append(f"{rel}:{number}: {line.strip()[:200]}")
                    if len(hits) >= max_hits:
                        head = f"{len(hits)}+ matches for {pattern!r}:\n"
                        return head + "\n".join(hits)
        if not hits:
            return f"No match for {pattern!r} in {self.build_id}."
        return f"{len(hits)} matches for {pattern!r}:\n" + "\n".join(hits)

    # -- export -------------------------------------------------------------

    def tools(self) -> list[Callable]:
        """Return the agent-callable code-inspection tools.

        **``grep`` is deliberately absent.** It exists for the rubric grader,
        which calls it directly rather than through this list. Handing it to the
        crowd would give this era's agents a capability every earlier run
        lacked, and these additions are additive precisely so existing crowd
        runs stay comparable. See :meth:`grader_tools`.
        """
        return [self.read_readme, self.list_files, self.read_file]

    def grader_tools(self) -> list[Callable]:
        """The crowd's tools plus ``grep`` -- for the rubric grader only."""
        return [*self.tools(), self.grep]

    def as_camel_tools(self) -> list:
        """Wrap :meth:`tools` as CAMEL ``FunctionTool``s (lazy camel import)."""
        try:
            from camel.toolkits import FunctionTool
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "camel-ai is required for as_camel_tools(); install the crowd extra."
            ) from exc
        return [FunctionTool(tool) for tool in self.tools()]
