# Welcome to the Cocoa contributing guide

Thank you for considering spending your time helping to improve Cocoa! We
appreciate everyone who pitches in, whether by reporting a bug, asking a
question, fixing a typo, or adding a feature. ✨

Cocoa is open source under the [MIT License](LICENSE.md), and everything you
contribute is licensed under that same license. See [Licensing](#licensing) for
the details.

Please keep discussions in issues and pull requests kind, patient, and
constructive. Our contributors come from medicine, statistics, and software, and
we want everyone to feel welcome.

This guide gives an overview of the contribution workflow, from opening an issue
to creating a pull request, getting it reviewed, and getting it merged.

## New contributor guide

To get an overview of the project, read the [README](README.md) and the
[documentation](https://cocoa-tokenizer.readthedocs.io). The
[MIMIC-IV demo](recipes/mimic-iv-demo.md) runs the whole pipeline on public data,
and it's the quickest way to see what each stage does. For a map of the codebase,
read [CLAUDE.md](CLAUDE.md). It holds the project's architecture notes and
conventions, and it's written for human contributors as well as coding agents.

Here are some resources to help you get started with open source contributions:

- [Finding ways to contribute to open source on GitHub](https://docs.github.com/en/get-started/exploring-projects-on-github/finding-ways-to-contribute-to-open-source-on-github)
- [Set up Git](https://docs.github.com/en/get-started/git-basics/set-up-git)
- [GitHub flow](https://docs.github.com/en/get-started/using-github/github-flow)
- [Collaborating with pull requests](https://docs.github.com/en/github/collaborating-with-pull-requests)

## Contribution types and what we're looking for

We welcome:

- bug reports, especially ones with a small example that reproduces the problem
- bug fixes
- corrections and clarifications to the README, the recipes, and the API docs
- new recipes showing how you use Cocoa on your own data
- new configuration options, and support for new kinds of source tables
- tests for behavior that isn't covered yet
- improvements to the timeline visualizer

Please open an issue to discuss these before you start on them:

- a change to a shipped default config in [src/cocoa/config/](src/cocoa/config/),
  which changes the output for every user
- a new pipeline stage or CLI command
- a change to what an existing config produces

If you aren't sure whether your idea fits, open an issue and ask.

## Patient data

Cocoa was built for electronic health records. **Never put real patient data in
an issue, a pull request, a commit, or a screenshot.** This includes the
visualizer's html and pdf exports of a real subject's timeline. Git ignores
`data-raw/` and `processed/` so that no data is committed from them by mistake.
To reproduce a problem, use the synthetic data that
[tests/synth.py](tests/synth.py) generates, or the public MIMIC-IV demo.

## Getting started

### Issues

#### Create a new issue

If you spot a problem or have an idea, first
[search the existing issues](https://github.com/bbj-lab/cocoa/issues) to see
whether someone has already reported it. If no one has, open a new issue. For a
bug, please include:

- your Cocoa version (`pip show cocoa-tokenizer`), Python version, and polars
  version
- the command you ran, and the config you passed with `-c`, if any
- the full error message or traceback
- what you expected to happen

#### Solve an issue

Browse the [existing issues](https://github.com/bbj-lab/cocoa/issues) to find one
that interests you. If you'd like to work on one, leave a comment so that others
know, and feel free to ask questions there before you start.

### Make changes

1. Fork the repository and clone your fork.
2. Install Cocoa in development mode (Python 3.11 or newer):

    ```sh
    python -m venv .venv && . .venv/bin/activate
    pip install -e '.[all]'
    ```

3. Create a working branch from `master` and start making your changes.

As you work:

- When you change behavior, add or update a test first. The tests in
  [tests/](tests/) run on synthetic data only, so `pytest` needs no real data.
- Keep configs backward compatible with the releases on PyPI. A config that
  worked with a released version must keep working, with the same output, even
  though it lacks any key you add. Read a new key with `cfg.get(key, fallback)`,
  where the fallback reproduces the old behavior. Then add a test showing that a
  config without the key still gives the old output. The Conventions section of
  [CLAUDE.md](CLAUDE.md) spells out this rule in full.
- Update the [README](README.md), the [recipes](recipes/), the
  [API docs](docs/api/), and the shipped configs to match your change. The docs
  site includes each recipe from [recipes/](recipes/), so edit it there. A new
  recipe also needs a stub in [docs/recipes/](docs/recipes/) and an entry in the
  `nav` of [mkdocs.yml](mkdocs.yml).
- Match the style of the surrounding code. Ruff formats and lints the Python
  (line length 88, double quotes). Prettier wraps the Markdown, following
  [.prettierrc.toml](.prettierrc.toml).

### Commit your update

Before you commit, format, lint, and test:

```sh
ruff format .
ruff check . --fix
pytest
```

Then commit your changes with a message that says what changed and why. Review
your own diff before you push. It's the quickest way to catch a leftover
debugging line or a file that doesn't belong.

### Pull request

Before you open a pull request, please run the full test suite from the root of
the repository and make sure it passes:

```sh
pytest
```

The tests generate their own synthetic data, so you don't need any real data to
run them. Once they pass, open a pull request against `master` on
[bbj-lab/cocoa](https://github.com/bbj-lab/cocoa).

- Describe what you changed and why, so that reviewers can follow your reasoning.
- If your pull request resolves an issue,
  [link it to the issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue),
  for example with `Closes #123` in the description.
- Enable the checkbox to
  [allow maintainer edits](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/allowing-changes-to-a-pull-request-branch-created-from-a-fork)
  so that a maintainer can update your branch before merging.
- CI runs `pytest` on every supported Python version and again with each
  dependency at the oldest version [pyproject.toml](pyproject.toml) allows. It
  also runs `ruff check`. Please make sure these pass.
- A maintainer will review your pull request and may ask questions or request
  changes, either as suggested changes you can apply directly or as comments.
  Mark each conversation as resolved once you've addressed it.
- By opening a pull request, you agree that your contribution is licensed under
  the project's MIT License (see [Licensing](#licensing)).

### Your PR is merged!

Congratulations 🎉🎉 The Cocoa team thanks you ✨

Once your pull request is merged, your contribution ships with the next release
of [`cocoa-tokenizer`](https://pypi.org/project/cocoa-tokenizer/), along with its
[documentation](https://cocoa-tokenizer.readthedocs.io).

Now that you're a contributor, we hope you'll keep going. Take a look at the
[open issues](https://github.com/bbj-lab/cocoa/issues) to find your next one.
