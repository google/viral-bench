# How to contribute

We'd love to accept your patches and contributions to this project.

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our community guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Contribution process

### Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Development setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/), which pins the
whole resolution in `uv.lock`. Install it once, then:

```bash
git clone https://github.com/google/viral-bench.git
cd viral-bench

uv sync                    # create the venv and install from the lock file
uv run pre-commit install  # run the lint/format hooks before each commit
```

Before opening a pull request, run what CI runs:

```bash
uv run pytest             # tests
uv run ruff check .       # lint
uv run ruff format --check .
```

`uv run ruff format .` and `uv run ruff check --fix .` fix most of what those
report.

### Tests that skip themselves

A bare `uv run pytest` is expected to skip part of the suite, and that is not a
broken checkout. Several stages drive real infrastructure, so their tests are
gated on that infrastructure actually being present:

- browser tests (`tests/crowd/`, `tests/rubric/`) need Playwright plus a system
  Chrome/Chromium — see `browser_available()` in
  `src/viral_bench/crowd/interaction/browser.py`;
- `tests/test_container_runtime.py` needs a container runtime (Podman or
  Docker) and a locally built `viralbench-runtime` image
  (`docker/Containerfile`);
- `tests/crowd/test_crowd_env_smoke.py` needs the separate crowd virtualenv
  (`.venv-crowd`), created by `scripts/setup_crowd_env.sh`.

`scripts/setup_founder_env.sh --check` and `scripts/setup_crowd_env.sh --check`
report what your machine is missing, installing nothing. If you are changing one
of those subsystems, set it up rather than relying on the skip — a skipped test
proves nothing.

### What we look for

- A test for any new behavior. `tests/` is organised by subsystem; put the new
  one next to the closest existing case.
- Comments that explain *why*, not *what*. Most of the non-obvious constants and
  defaults here were chosen for a measured reason, and that reason is recorded
  next to them. Keep that up.
- Small, focused pull requests. They are easier to review and faster to merge.
- Commit messages in the imperative mood ("Add X", "Fix Y"), saying why the
  change is being made.
