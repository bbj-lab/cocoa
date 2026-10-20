# Contributing to Cocoa

Thank you for helping to improve Cocoa! We appreciate every contribution, from a
bug report or a typo fix to a new feature. ✨ Please keep discussions kind,
patient, and constructive: our contributors come from medicine, statistics, and
software, and we want everyone to feel welcome.

## Getting oriented

Start with the [README](README.md) and the
[documentation](https://cocoa-tokenizer.readthedocs.io). The
[MIMIC-IV demo](recipes/mimic-iv-demo.md) runs the whole pipeline on public data
and is the quickest way to see what each stage does. [CLAUDE.md](CLAUDE.md) maps
the codebase and its conventions, for human contributors as well as coding
agents. New to pull requests? See
[GitHub flow](https://docs.github.com/en/get-started/using-github/github-flow).

## What we're looking for

We welcome bug reports (ideally with a small example that reproduces the
problem), bug fixes, corrections to the docs, new recipes, new config options or
support for new kinds of source tables, new tests, and improvements to the
visualizer.

Please open an issue to discuss these before you start on them:

- a change to a shipped default config in [src/cocoa/config/](src/cocoa/config/),
  which changes the output for every user
- a new pipeline stage or CLI command
- a change to what an existing config produces

If you aren't sure whether your idea fits, open an issue and ask.

## Patient data

**Never put real patient data in an issue, a pull request, a commit, or a
screenshot**, including the visualizer's html and pdf exports of a real subject.
Git ignores `data-raw/` and `processed/` to guard against this. To reproduce a
problem, use the synthetic data from [tests/synth.py](tests/synth.py) or the
public MIMIC-IV demo.

## Issues

[Search the existing issues](https://github.com/bbj-lab/cocoa/issues) before you
open a new one. For a bug, please include:

- your Cocoa version (`pip show cocoa-tokenizer`), Python version, and polars
  version
- the command you ran, and the config you passed with `-c`, if any
- the full traceback, and what you expected to happen

To work on an existing issue, leave a comment on it so that others know.

## Making changes

1. Fork and clone the repository, then install it in development mode (Python
   3.11 or newer):

    ```sh
    python -m venv .venv && . .venv/bin/activate
    pip install -e '.[all]'
    ```

2. Create a working branch from `master`.

As you work:

- When you change behavior, add or update a test first. The tests in
  [tests/](tests/) run on synthetic data only, so `pytest` needs no real data.
- Keep configs backward compatible with the releases on PyPI: a config that
  worked with a released version, and lacks a key you add, must give the same
  output as before. Read a new key with `cfg.get(key, fallback)`, where the
  fallback reproduces the old behavior, and add a test showing it does. The
  Conventions section of [CLAUDE.md](CLAUDE.md) spells out the full rule.
- Update the [README](README.md), the [recipes](recipes/), the
  [API docs](docs/api/), and the shipped configs to match. The docs site
  includes each recipe from [recipes/](recipes/), so edit it there; a new recipe
  also needs a stub in [docs/recipes/](docs/recipes/) and a `nav` entry in
  [mkdocs.yml](mkdocs.yml).
- Match the surrounding style. Ruff formats and lints the Python (line length 88,
  double quotes), and Prettier wraps the Markdown, following
  [.prettierrc.toml](.prettierrc.toml).

Before you commit, format, lint, and test:

```sh
ruff format .
ruff check . --fix
pytest
```

## Pull requests

Open a pull request against `master` on
[bbj-lab/cocoa](https://github.com/bbj-lab/cocoa).

- Describe what you changed and why. If it resolves an issue,
  [link it](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue),
  for example with `Closes #123`.
- [Allow maintainer edits](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/allowing-changes-to-a-pull-request-branch-created-from-a-fork)
  so that a maintainer can update your branch before merging.
- CI runs `pytest` on every supported Python version, and again with each
  dependency at the lowest version [pyproject.toml](pyproject.toml) allows.
  Please make sure these pass. It also reports test coverage to
  [Codecov](https://app.codecov.io/gh/bbj-lab/cocoa), which is for information
  only and never fails a check; run `pytest --cov` to see it locally.
- A maintainer will review your pull request and may ask questions or suggest
  changes. Mark each conversation as resolved once you've addressed it.

Cocoa is open source under the [MIT License](LICENSE.md), and by opening a pull
request, you agree that your contribution is licensed under it. Once merged, it
ships with the next release of
[`cocoa-tokenizer`](https://pypi.org/project/cocoa-tokenizer/). Thank you! 🎉
