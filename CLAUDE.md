# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this is

**Cocoa** (`cocoa-tokenizer` on PyPI, imported as `cocoa`) is a configurable
pipeline that turns raw event tables — originally electronic health records —
into tokenized timelines for training/evaluating generative sequence models. It
ships a CLI (`cocoa`) and is driven entirely by YAML config. Companion projects
in the same lab: [cotorra](https://github.com/bbj-lab/cotorra) (trainer) and
coreopsis.

The pipeline has three stages, each a `Configurable` subclass with a shipped
default config in [src/cocoa/config/](src/cocoa/config/):

1. **Collate** ([collator.py](src/cocoa/collator.py)) — pull raw parquet/csv
   tables into one denormalized
   [MEDS](https://github.com/Medical-Event-Data-Standard/meds)-like frame of
   `(subject_id, time, code, numeric_value, text_value)` events, plus
   train/tuning/held_out subject splits, chronological or random (see
   `subject_splits.how` below). → `meds.parquet`,
   `subject_splits.parquet`
2. **Tokenize** ([tokenizer.py](src/cocoa/tokenizer.py)) — convert events to
   integer token sequences. Learns a vocabulary (`lookup`) and quantile `bins`
   **on training data only**, then freezes them. Adds BOS/EOS, optional
   clock/time-spacer tokens. → `tokens_times.parquet`, `tokenizer.yaml`
3. **Winnow** ([winnower.py](src/cocoa/winnower.py)) — split held-out timelines
   at a threshold (a duration after the first event or after the start time, or
   the first occurrence of a token) into past/future and flag outcome tokens for
   evaluation. → `{split}_for_inference.parquet`

Data flows strictly stage-to-stage through files in `--processed-data-home`. The
one exception is `cocoa winnow --output-home` (`-o`;
`Winnower(output_home=...)`), which reads from the processed dir but writes its
`{split}_for_inference.parquet` to another directory, so that one dataset can be
winnowed under several configs. `Winnower.copy_processed_data` first copies the
processed dir's top-level files there, so the output dir is a processed dir in
its own right (`cocoa visualize` reads it); it skips subdirectories, since the
output dir may be nested in the processed dir, and skips
`*_for_inference.parquet`, which may come from another config.

Alongside the stages, **Visualize** ([visualizer.py](src/cocoa/visualizer.py))
renders one subject's timeline from a processed dir as a self-contained,
interactive html page, either served on localhost or written to a file with
`--export-html`, and as a static pdf with `--export-pdf` (drawn by
[pdf.py](src/cocoa/pdf.py)). It is a `Configurable` too (default
[visualization.yaml](src/cocoa/config/visualization.yaml): lanes, palette, prefix
names, code descriptions), but it only reads the stages' outputs. It draws the
learned vocabulary as a second page (`cocoa visualize` without a subject id,
`Visualizer.render_vocabulary`): each binned code's training values against its
cut points, every word with its training count, the prefixes, and the timelines
by split, as `tokenize --verbose` summarizes them. It serves or saves that page
the same ways, as a pdf too (`render_vocabulary_pdf`). Each page's template and js
live in [src/cocoa/assets/](src/cocoa/assets/) (`timeline.*`, `vocabulary.*`),
with the icon and fonts; `timeline.css` holds the tokens and parts both pages
use, and `vocabulary.css` only what the vocabulary page adds.

Ahead of the stages, **`cocoa get-demo`** ([demo.py](src/cocoa/demo.py)) writes
the public data the docs run on: the MIMIC-IV demo's CLIF tables, which
[clifpy](https://pypi.org/project/clifpy/) ships in its package, to a raw data
dir, along with the four tables the shipped collation config reads that clifpy
derives from them. clifpy comes from the optional `demo` extra and is imported
only when the command runs, so the rest of cocoa never needs it.

## Commands

```sh
# Dev install (Python >= 3.11)
python -m venv .venv && . .venv/bin/activate
pip install -e '.[all]'          # all = dev + docs + test extras
pip install -e '.[all,demo]'     # plus clifpy, for cocoa get-demo (Python <= 3.14)

# The MIMIC-IV demo's CLIF tables, from clifpy, to start from
cocoa get-demo -r data-raw/demo [--verbose]

# Run the pipeline (or a single stage: collate | tokenize | winnow)
cocoa pipeline -r <raw-data-home> -p <processed-data-home> [--verbose]
cocoa <stage> -c <config.yaml> ... # -c overrides the shipped default for that stage
cocoa <stage> ... n_bins=5 '~key'  # trailing overrides edit single keys of it
cocoa pipeline ... tokenization.n_bins=5  # in pipeline, each key names its stage
cocoa winnow -p <processed-data-home> -o <dir>  # winnowed files go to <dir>
cocoa <stage> -h                   # help; --verbose prints summary stats
cocoa combine-datasets <dir> <dir> ... -o <output-dir>

# View one subject's timeline (serves on 127.0.0.1:8765 until Ctrl-C)
cocoa visualize <subject-id> -p <processed-data-home> [--open | -e out.html]
# ...or, given no subject id, the vocabulary and its bins over the training data
cocoa visualize -p <processed-data-home> [--open | -e out.html]

# Format + lint (run before committing)
ruff format .
ruff check . --fix

# Tests (pytest, synthetic data only)
pytest
pytest --cov                       # with line and branch coverage, as CI measures it

# Docs (mkdocs-material, published to readthedocs)
mkdocs build
mkdocs serve --dev-addr 127.0.0.1:8001

# Release: bump pyproject.toml `version`, commit, then push a signed tag for it
git tag -s vYY.M.patch -m "message"
git push origin vYY.M.patch
```

There's a pytest suite in [tests/](tests/) — run it with `pytest`. CI
([.github/workflows/tests.yml](.github/workflows/tests.yml)) runs it on every
push and pull request, and weekly, on Python 3.11 through 3.15, and once more on
3.11 with each dependency at the lowest version [pyproject.toml](pyproject.toml)
allows; it doesn't run ruff, so format and lint locally before pushing. Keep the
matrix in step with `requires-python`, and raise a floor in `pyproject.toml` when
code starts needing a newer dependency, or the lowest job fails. Each matrix job
also uploads its coverage to [Codecov](https://app.codecov.io/gh/bbj-lab/cocoa)
by OIDC, so no token secret is needed (fork PRs upload tokenless). Coverage is
informational only: the upload step can't fail the job (`continue-on-error`), and
[codecov.yml](codecov.yml) makes its statuses always pass and waits for all five
uploads before reporting (`after_n_builds`; keep it equal to the matrix size).
Coverage counts `src/cocoa` only and skips the `__main__` blocks (see
`[tool.coverage]` in [pyproject.toml](pyproject.toml)). On 3.15, `pyproject.toml`
excludes deepdiff 9.1, whose `cachebox<6` has no 3.15 wheels and won't build
there. The matrix jobs also install the `demo` extra, except on 3.15: clifpy pins
pandas 2.3.3, which has no 3.15 wheels. The lowest job leaves it out too, since
pyarrow, which clifpy imports, needs numpy 2 without declaring it, and fails to
import at cocoa's numpy floor. The `demo` extra also caps duckdb below 2, which
clifpy doesn't: clifpy 2.1.0's dose conversion calls a relation's `fetchone()`,
which duckdb 2.0 removed, so lift the cap only once a clifpy release stops doing
so. [tests/test_demo.py](tests/test_demo.py) runs
`cocoa get-demo` on clifpy's real tables where clifpy is installed, skips those
tests where it isn't, and fails them where it won't import; its other tests use a
stand-in `clifpy` package and run everywhere. [tests/synth.py](tests/synth.py)
generates synthetic CLIF-like raw data, and
[tests/conftest.py](tests/conftest.py) turns it into shared fixtures: a
session-scoped `pipeline` (collate → tokenize → winnow once with shipped
defaults) and a per-test `runner` for re-running individual stages with
overridden configs in an isolated tmp dir — no real patient data needed. Test
files mostly mirror `src/cocoa` modules, except tokenizer coverage is split into
core behavior (`test_tokenizer.py`), serialization/transfer
(`test_tokenizer_io.py`), and configurable options (`test_tokenizer_options.py`),
and the visualizer's into its timeline (`test_visualizer.py`) and its vocabulary
page (`test_visualizer_vocabulary.py`), plus cross-cutting suites for subject
splits, timezones, and full pipeline integration. Helpers that more than one test
file needs, such as `copy_of` and `as_written_by_26_6`, live in
[tests/conftest.py](tests/conftest.py) too. Beyond the fixtures running the
shipped defaults, no test should
depend on a shipped config carrying dataset-specific items (a lane, a prefix
name, a code description); a test that needs them writes its own config, as
`VIZ_CFG` in [tests/test_visualizer.py](tests/test_visualizer.py) does. A test
that reads console output passes it through `plain()` from
[tests/conftest.py](tests/conftest.py) first: typer styles its help and errors
whenever `GITHUB_ACTIONS` is set, and rich styles everything under `FORCE_COLOR`
or `PY_COLORS`, so substring checks on the raw text pass locally but fail in CI.

Most modules also keep an `if __name__ == "__main__"` block that self-tests
against a local processed dataset (e.g. `./processed/mimic/`, for which
[recipes/mimic-iv-demo.md](recipes/mimic-iv-demo.md) builds a public stand-in,
`./processed/demo/`); run a module directly (`python -m cocoa.tokenizer`) to
exercise it against real data — the tokenizer block also asserts round-trip
save/load equality, and the visualizer block asserts its pages reference no
external URLs. When you change behavior, add/update a pytest case first.

## Architecture notes

- **Config resolution** ([configurable.py](src/cocoa/configurable.py)): every
  stage loads a user `-c` config if given, otherwise the shipped default YAML,
  edits it with the command line's trailing `overrides`, and then merges
  non-`None` kwargs on top. The two YAMLs are never merged, so a user config that
  omits a key does **not** inherit that key from the default (this is why the
  backward-compatibility rule under Conventions matters). Read
  `Configurable.__init__` before changing merge logic. Config access is
  OmegaConf; use `.get(k, default)` for optional keys.
- **Overrides** (`apply_overrides`, the same function as cotorra's): `key=value`
  sets a key, adding it if absent, so a typo adds a stray key rather than
  failing; `~key` deletes one; a Hydra-style leading `+`/`++` is ignored.
  Deletion matters where a key's presence is what counts (the winnower's
  `threshold` keys), since `key=null` leaves it present. `cocoa pipeline` routes
  each override to the stage its first key names (`collation.`, `tokenization.`,
  `winnowing.`) and checks all three through `Configurable.load_cfg` before
  collating. Under `--tokenizer-home`, `Tokenizer.from_yaml` takes the saved
  config as written and makes it read-only: it is never merged with the shipped
  default, a `-c` config, overrides, or kwargs, so a key it lacks stays absent
  and falls back through `cfg.get` as it did when that tokenizer was saved
  (`cocoa tokenize` notes that `-c` and overrides don't apply when given with
  it). The one key it adds is a missing `fused`, as `false` (see Conventions).
- **Polars everywhere**, lazy by default. Frames are built as `LazyFrame` and
  written with `sink_parquet(..., engine="streaming")` to stay memory-bounded;
  `--verbose` forces collection for stats and can OOM on large data.
- **Reruns write the same files.** polars' `sort` is unstable and `group_by`
  returns groups in no set order, so a sort that decides an output's order needs
  keys that leave ties only between rows identical in every column written.
  `get_subject_splits` sorts groups on `(first_time, id)`, or on `(digest, id)`
  under `uniformly_random`; `tokenize_data` on
  `(time, priority, to_tokenize, numeric_value, subject_id)`, which also orders
  the rows of `tokens_times.parquet` (by first token, then `subject_id`); and
  `get_bins` on `code`, so `created_dttm` is the only line of `tokenizer.yaml`
  that changes between runs. Row order in `meds.parquet` and
  `subject_splits.parquet` still varies; nothing reads them by position.
  `shuffle_rows` (off by default) then reorders the rows of
  `tokens_times.parquet` with `Tokenizer.shuffle_rows`, a lazy sort on a hash of
  the row index with a fixed seed, so reruns stay identical.
- **`subject_splits.how` picks the order the split fractions cut.** `time_based`
  (the default, and the fallback for a config without it) orders groups by
  `first_time`; `uniformly_random` by the sha256 hex digest of each group id cast
  to a string, which is as good as a uniform shuffle and the same on every run,
  machine, and polars version. Don't swap in `pl.Expr.hash`, as `shuffle_rows`
  uses: its values may change with polars' version, which would move subjects
  across splits for anyone re-collating;
  `test_uniformly_random_ranks_by_the_digest_of_each_id` pins the order.
  `Collator.__init__` validates `how`, so a typo fails before `meds.parquet` is
  written.
- **Config-embedded expressions**: `filter_expr` / `with_col_expr` / `agg_expr`
  in a collation config are Polars expression _strings_ `eval`'d via
  `Collator.slightly_safer_eval`. This is intentionally powerful and **not a
  security boundary** — a config is as trusted as Python. Never run an untrusted
  config.
- **`drop_duplicate_events` dedupes within an entry, not across entries.** The
  `df.unique()` sits at the end of `Collator.get_entry`, so it collapses copies
  one entry emits of the same event — a result posted by two systems, a row
  multiplied by a join — comparing the whole collated event (`subject_id`,
  `time`, `code`, `numeric_value`, `text_value`). Two _entries_ that emit the
  same event still contribute a row each to `get_all`'s concat. Off by default.
- **`max_lookback_days` / `max_lookahead_days` window the whole event stream.**
  `Collator.drop_events_outside_window` runs once at the end of `get_all`, left-
  joining each event to its subject's reference `start_time`/`end_time`
  (`validate="m:1"`, so a reference table with a repeated `subject_id` raises
  rather than duplicating events). Limits are inclusive, in 24-hour days of
  elapsed time, built at `default_time_unit`. An unset limit, a null bound, or a
  subject absent from the reference frame keeps the event. Both are `null` (off)
  by default.
- **Vocabulary/bins are frozen after training.** `UNK` is always token `0`. Reuse
  a learned tokenizer across datasets with
  `cocoa tokenize --tokenizer-home <path>/tokenizer.yaml` (see
  [recipes/tokenizer-transfer.md](recipes/tokenizer-transfer.md)).
- **`min_training_ct` prunes the vocabulary's long tail.** Words seen fewer than
  that many times in the training split get no token and fall through to `UNK`;
  `BOS`/`EOS`/`CLCK//`/`TIME//` are exempt so structure survives on small data.
  `lookup` carries a `count` column (null for `UNK`), serialized as a separate
  `counts` block in `tokenizer.yaml` — `from_yaml` tolerates its absence in
  tokenizers written before it existed. The shipped default is `0` (off), because
  the threshold is an absolute count: any value large enough to be useful on a
  hospital-scale corpus unks nearly everything in the few-subject datasets the
  tests build, so tests exercising it pin the value themselves.
- **`include_hours_to_end_time` writes a per-token column** of fractional hours
  from each token to its subject's `end_time`, joined from
  `subject_splits.parquet` (so it is the collation config's
  `reference.end_time`). It counts down toward that time and goes negative beyond
  it, which `EOS` and trailing spacers can be; being a duration, it is tz- and
  unit-invariant. `Winnower.add_outcome_flags` splits it into `_past`/`_future`
  through its `optional` list — a new per-token column the tokenizer writes
  conditionally has to be named there too, or it rides along unsplit the way
  `numeric_values` does. The visualizer also reads such columns by name, so it
  only displays the ones it knows about.
- **Winnowing thresholds are picked by presence**, in `run_thresholding`'s order:
  `threshold.duration_s` (or the legacy top-level `horizon_s`), then
  `threshold.after_start_time_s`, then `threshold.first_occurrence`; the first
  present wins, so switching from the shipped `duration_s` takes
  `~threshold.duration_s`. `duration_s` counts from each timeline's first event
  (its `BOS`), `after_start_time_s` from its subject's `start_time` in
  `subject_splits.parquet` (the collation config's `reference.start_time`); they
  agree only on timelines that open at their start time, and most of the demo's
  open earlier, in the emergency department. Under both, the past is the tokens
  strictly before the cut and a kept timeline ends after it; `after_start_time_s`
  also drops timelines that begin at or after the cut, so every past still opens
  with `BOS`.
- **Codes** are `PREFIX//value`. The value is lowercased, whitespace and commas
  become `_`, and runs of `_` collapse to one; the prefix is left as written, and
  `text_value` is only lowercased with whitespace→`_`. The `ordering` list in the
  tokenization config breaks ties between events at the same timestamp; a prefix
  missing from `ordering` sorts after the listed ones but still before `EOS`,
  which `tokenize_data` always puts last. The exception is `TIME`, which then
  sorts first. Spacers are rows of their own, one per gap between a subject's
  consecutive instants (so their `numeric_value` is null). When adding a new
  event prefix, add it to `ordering` too, and give it a name under `prefixes` in
  `visualization.yaml`; otherwise its lane is labeled with the bare prefix.
- **The visualizer only reads; it is coupled to the stages' formats.** It loads
  `tokenizer.yaml` with PyYAML's C loader rather than OmegaConf, for speed on
  large vocabularies. It reads `tokens_times.parquet`, plus
  `subject_splits.parquet` (pass-through columns become subject fields) and
  `{split}_for_inference.parquet` when present. It also reads what released
  versions wrote (anything since 26.6.2). Before 26.9.0, times were naive UTC,
  which `get_timeline` localizes; a test rewrites today's output into that format
  (`as_written_by_26_6`) and checks it draws the same. When tokens were written
  without `include_numeric_values`, it recovers values from `meds.parquet` by
  re-binning them the way `Tokenizer.bin_data` does, and the vocabulary page's
  `get_histograms` re-bins the training split's values from `meds.parquet` the
  same way, in polars, to count each bin; a test checks those counts against the
  fused words' training counts. It spots a fused bin as an uppercase `_Q<n>`,
  which only works because code values are lowercased. So a change to binning,
  to which values bins are learned on, or to code normalization needs a matching
  change in `Visualizer.get_values` / `get_histograms` / `FUSED_BIN`. Its
  histograms take each code's mean over its values sorted, and the timelines'
  summaries over rows sorted by `subject_id`, so that sums don't depend on the
  order rows arrive in and a rerun renders the same page, `generated` aside.
  A page has to stay self-contained:
  fonts and icon are inlined as base64, with no external URLs, so an exported
  file can be sent to someone without cocoa or the data. Its static
  `#cocoa-fallback` note tells a viewer that doesn't run scripts (an email or
  file preview) to open it in a browser; the script removes the note once the
  page is drawn. The pdf is drawn in Python from the same payload, by a
  dependency-free writer that embeds the packaged Gotham whole (as Type0 fonts
  addressed by glyph id, with a ToUnicode map so text copies out) and sets codes
  and tokens in standard Courier (WinAnsi text); a character a font lacks prints
  as `?`. A missing font file falls back to Helvetica, as the page falls back to
  system fonts. `TimelinePdf` ports the page's lanes, axis ticks, and binned rows
  from `timeline.js`, and `VocabularyPdf` ports `vocabulary.js`'s histograms,
  tables, and number formats (`fmtPct`, `niceTicks`, and the like, rounding
  halves up as javascript does), so a change to a payload or to how its page
  draws it needs a matching change in the pdf that ports it. Both lay out through
  `Flow`, which holds the paging, headings, and tables they share.
- **Times** are normalized on load to the collation config's `default_timezone`
  (`Collator.to_default_tz`; `UTC` if unset) and stay **tz-aware** for the rest
  of the pipeline: tz-aware columns are instant-preserved, tz-naive columns are
  assumed to be local times in that zone (ambiguous DST times take the later
  instant; nonexistent ones raise `ComputeError`). A csv has no schema, so its
  datetimes arrive as `String` and `Collator.parse_datetime` resolves them with
  `str.to_datetime(time_zone=...)` before that branch — one call covers both
  rules, since an offset-bearing string converts and a bare one localizes.
  Downstream duration math (spacers, winnowing thresholds/horizons) works on
  instants and is therefore tz-invariant, but `CLCK//HH` tokens carry the _local_
  hour — the zone is part of what that vocabulary means, and `tokenizer.yaml`
  does not record it, so a transferred tokenizer only agrees with a new dataset
  if both were collated in the same zone.
- **Time precision** is the collation config's `default_time_unit` (`ms` / `us` /
  `ns`; polars' default `us` if unset, validated in `Collator.__init__`). Each
  raw datetime column is repinned to it in `to_default_tz` — `convert_time_zone`
  keeps the source unit, so the tz-aware branch casts explicitly while the
  tz-naive branch pins the unit in its cast. Downstream stages take their times
  and durations from that column, so the unit propagates to
  `tokens_times.parquet` and the inference frames without further configuration.
- **`cocoa get-demo` takes clifpy's demo as clifpy ships it.** `demo.copy_tables`
  copies every `clif_*.parquet` under clifpy's `data/clif_demo/` (not its
  subdirectories, such as the `3.0/` on clifpy's main branch), dropping the
  `__index_level_0__` column pandas left in 2.1.0's tables. `demo.derive_tables`
  then runs clifpy's respiratory-support waterfall, its dose conversions, and its
  SOFA scores, as the CLIF-MIMIC ETL's `scripts/clifpy_tables.py` does, with the
  preferred units of coreopsis's `run_clifpy.py` (`CONTINUOUS_UNITS`,
  `INTERMITTENT_UNITS`), so each medication's converted doses share one unit.
  SOFA rows are timed at each stay's last record, which in clifpy's tables
  usually comes hours before discharge, not after it, as in the ETL's. clifpy's
  output is swallowed unless `--verbose`; its logs go to a temporary directory,
  and `demo.release_logs` closes the handlers clifpy leaves on its loggers so
  that the directory can be removed.

## Conventions

- **Style**: ruff, line length 88, double quotes, LF,
  `skip-magic-trailing-comma`. Lint set is `E,F,I` (isort included; first-party =
  `cocoa`, `cotorra`, `coreopsis`).
- Files open with `#!/usr/bin/env python3` and a short lowercase module
  docstring; method docstrings are terse and lowercase. Match the surrounding
  terseness.
- **Type hints**: when writing a new function, try to hint its parameters and
  return (`-> None` included), but only where the hints come out clean: short,
  readable, and accurate. If a precise hint would sprawl over several lines, or
  need a `cast`, a `# type: ignore`, or reshaped code to satisfy a type checker
  (none runs here), use a simpler one like `dict[str, Any]` or leave it off.
  Spell out `X | None` whenever the default is `None`, and hint a list read from
  a config `collections.abc.Sequence`, since OmegaConf hands it over as a
  `ListConfig`. Annotations are evaluated at import on Python 3.11, so quote
  forward references and skip 3.12+ names like `typing.override`; in `cli.py`,
  typer reads a command's parameter annotations, so changing one changes the CLI.
- **Config changes stay backward compatible with what's on PyPI** as far as
  possible. A config that worked with a released version, and lacks a key added
  since, must keep working exactly as it did under that release. A `-c` config
  replaces the shipped default instead of merging with it, so an old config never
  picks up the new key from there. Anything not yet released (a key, a fallback,
  a file format that so far exists only on a branch or in an unpublished tag) can
  change without concern, and there's no need to stay compatible with it. To
  check, `pip index versions cocoa-tokenizer` lists the releases, and
  `git grep <key> v<version> -- src/` shows whether one had a key. For anything
  released, that means:
    - read a new key with `cfg.get(key, fallback)`, where the fallback reproduces
      the previous behavior, whatever the shipped default sets. Never use
      `cfg.key` / `cfg["key"]`, which raise on a missing key.
      `drop_duplicate_events`, `default_time_unit`, `min_training_ct`,
      `include_hours_to_end_time`, `shuffle_rows`, `max_lookback_days` /
      `max_lookahead_days`, and `subject_splits.how` all follow this pattern.
    - don't rename a key, repurpose it, or change what an existing value means.
      If a key has to change, keep accepting the old spelling.
    - the `cfg` block saved in `tokenizer.yaml` is a config too, and copies
      written by released versions outlive the code that wrote them. Code that
      reads one (`from_yaml`, the winnower, the visualizer) must fall back to
      what the tokenizer did when it wrote that file, which need not be today's
      fallback. For example, `to_yaml` now records the resolved `fused`, so a
      yaml without it was tokenized unfused, even though a config without `fused`
      now fuses. When a new key changes tokenization, record its resolved value
      the same way.
    - add a pytest case showing that a config without the new key produces the
      old output.
- New pipeline stages subclass `Configurable`, set `default_file`, ship a default
  YAML under `src/cocoa/config/` (packaged via `package-data` in
  [pyproject.toml](pyproject.toml)), and expose `save_all(verbose)`. Add the YAML
  and the class to `SHIPPED` and the `claimed` set in
  [tests/test_configurable.py](tests/test_configurable.py), which checks that
  every shipped YAML is packaged and claimed by a stage. Non-YAML package files,
  like the visualizer's under `src/cocoa/assets/`, need their own `package-data`
  globs.
- New CLI commands go in [cli.py](src/cocoa/cli.py) as typer commands using
  `rich` for output, mirroring the existing timing/output-path print pattern.
- **Versioning is CalVer** `YY.M.patch` (e.g. `26.6.1`); releases are signed git
  tags `vYY.M.patch`, published to PyPI. Not every tag was published: 26.6.0 was
  the first PyPI release, and `v26.4.0` and `v26.6.3` never shipped.
  `__version__` comes from installed package metadata, not a literal. Now pushing
  a `v*` tag to `bbj-lab/cocoa` publishes it: it runs
  [publish.yml](.github/workflows/publish.yml) (cotorra's, but for the names),
  which builds the sdist and wheel and uploads them to PyPI and TestPyPI by
  trusted publishing (OIDC, as for Codecov, so no token secret), in parallel jobs
  through the `pypi` and `testpypi` environments, so neither index's failure
  blocks the other; TestPyPI skips a version it already has. It fails unless the
  tag is `v` + `version`, points to the commit built, and is signed
  (`git tag -s`, not a lightweight tag) with burkh4rt's release key: it imports
  the public keys GitHub serves for that account and accepts only the one whose
  fingerprint it pins (`RELEASE_KEY`), so letting anyone else release means
  changing that. Each file gets two Sigstore-signed attestations, both keyless
  (signed for the workflow's OIDC identity): SLSA build provenance from the build
  job, kept by GitHub (`gh attestation verify <file> --repo bbj-lab/cocoa`), and
  the publish action's PEP 740 attestations, uploaded beside it to each index
  (`pypi-attestations verify pypi --repository https://github.com/bbj-lab/cocoa pypi:<file>`).
  PyPI refuses any requirement that's a direct URL reference, even in an extra,
  so before building it drops every extra holding one (cocoa has none; cotorra's
  `gen` does). Don't build and upload by hand, which skips the checks and the
  attestations.
- Keep [README.md](README.md), the [recipes/](recipes/) (mirrored into
  [docs/recipes/](docs/recipes/)), the per-module pages in
  [docs/api/](docs/api/), and the shipped default configs in sync when behavior
  changes. The README is the PyPI long description, and the recipes are the
  primary user-facing docs. [recipes/mimic-iv-demo.md](recipes/mimic-iv-demo.md)
  runs the shipped defaults end to end on the MIMIC-IV demo, as `cocoa get-demo`
  writes it from clifpy, and quotes what they make of it (event and split counts,
  vocabulary size, timeline lengths, the first held-out hospitalization). The
  other docs take their examples from the same run: the README's output tables
  and bin edges, the docs' paths (`data-raw/demo`, `processed/demo`), and the
  counts quoted by the other recipes, which run variants of it (split by
  `anchor_year_group`, or collated in `America/New_York`). So rerun it, and
  refresh those, after changing a shipped config, binning, splitting, or what
  `cocoa get-demo` derives, and when raising the clifpy floor: the recipe names
  the clifpy release its numbers come from, and another may ship other tables.

## Gotchas

- Don't commit anything under `data-raw/` or `processed/` (real patient data;
  gitignored and symlinked to shared storage on HPC).
- Editing a shipped `src/cocoa/config/*.yaml` changes the default behavior for
  every user — usually you want a separate config passed with `-c` instead.
- `combine-datasets` does **not** refuse to merge processed dirs whose tokenizer
  configs differ. It diffs the yamls (ignoring `created_dttm`) and logs a warning
  with the diff, but still combines the parquets and writes the _first_ input's
  `tokenizer.yaml`, so check its output for `Configuration mismatch`. A dir
  tokenized with `--tokenizer-home` records `is_training: false` where its source
  records `true`, so combining the two always warns, about that key alone. It
  also handles a legacy Int64-token schema from tokenizers `<= 26.4.0`. Only
  `tokenizer.yaml` is written to the processed dir, so a `default_timezone` or
  `default_time_unit` mismatch escapes that config diff and instead surfaces as a
  polars `SchemaError` on the datetime columns — which the legacy-schema fallback
  does not repair, since it only recasts token columns.
