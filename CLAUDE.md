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
   chronological train/tuning/held_out subject splits. → `meds.parquet`,
   `subject_splits.parquet`
2. **Tokenize** ([tokenizer.py](src/cocoa/tokenizer.py)) — convert events to
   integer token sequences. Learns a vocabulary (`lookup`) and quantile `bins`
   **on training data only**, then freezes them. Adds BOS/EOS, optional
   clock/time-spacer tokens. → `tokens_times.parquet`, `tokenizer.yaml`
3. **Winnow** ([winnower.py](src/cocoa/winnower.py)) — split held-out timelines
   at a threshold (a duration or the first occurrence of a token) into
   past/future and flag outcome tokens for evaluation. →
   `{split}_for_inference.parquet`

Data flows strictly stage-to-stage through files in `--processed-data-home`. The
one exception is `cocoa winnow --output-home` (`-o`; `Winnower(output_home=...)`),
which reads from the processed dir but writes its `{split}_for_inference.parquet`
to another directory, so that one dataset can be winnowed under several configs.
`Winnower.copy_processed_data` first copies the processed dir's top-level files
there, so the output dir is a processed dir in its own right (`cocoa visualize`
reads it); it skips subdirectories, since the output dir may be nested in the
processed dir, and skips `*_for_inference.parquet`, which may come from another
config.

Alongside the stages, **Visualize** ([visualizer.py](src/cocoa/visualizer.py))
renders one subject's timeline from a processed dir as a self-contained,
interactive html page, either served on localhost or written to a file with
`--export-html`, and as a static pdf with `--export-pdf` (drawn by
[pdf.py](src/cocoa/pdf.py)). It is a `Configurable` too (default
[visualization.yaml](src/cocoa/config/visualization.yaml): lanes, palette, prefix
names, code descriptions), but it only reads the stages' outputs. Its page
template, css, js, icon, and fonts live in
[src/cocoa/assets/](src/cocoa/assets/).

## Commands

```sh
# Dev install (Python >= 3.10)
python -m venv .venv && . .venv/bin/activate
pip install -e '.[all]'          # all = dev + docs + test extras

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

# Format + lint (run before committing)
ruff format .
ruff check . --fix

# Tests (pytest, synthetic data only)
pytest

# Docs (mkdocs-material, published to readthedocs)
mkdocs build
mkdocs serve --dev-addr 127.0.0.1:8001
```

There's a pytest suite in [tests/](tests/) — run it with `pytest`; there's still
no CI. [tests/synth.py](tests/synth.py) generates synthetic CLIF-like raw data,
and [tests/conftest.py](tests/conftest.py) turns it into shared fixtures: a
session-scoped `pipeline` (collate → tokenize → winnow once with shipped
defaults) and a per-test `runner` for re-running individual stages with
overridden configs in an isolated tmp dir — no real patient data needed. Test
files mostly mirror `src/cocoa` modules, except tokenizer coverage is split into
core behavior (`test_tokenizer.py`), serialization/transfer
(`test_tokenizer_io.py`), and configurable options (`test_tokenizer_options.py`),
plus cross-cutting suites for subject splits, timezones, and full pipeline
integration. Beyond the fixtures running the shipped defaults, no test should
depend on a shipped config carrying dataset-specific items (a lane, a prefix
name, a code description); a test that needs them writes its own config, as
`VIZ_CFG` in [tests/test_visualizer.py](tests/test_visualizer.py) does.

Most modules also keep an `if __name__ == "__main__"` block that self-tests
against a local processed dataset (e.g. `./processed/mimic/`); run a module
directly (`python -m cocoa.tokenizer`) to exercise it against real data — the
tokenizer block also asserts round-trip save/load equality, and the visualizer
block asserts the page references no external URLs. When you change behavior,
add/update a pytest case first.

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
  `threshold.duration_s` vs. `threshold.first_occurrence`), since `key=null`
  leaves it present. `cocoa pipeline` routes each override to the stage its first
  key names (`collation.`, `tokenization.`, `winnowing.`) and checks all three
  through `Configurable.load_cfg` before collating. Under `--tokenizer-home`,
  `Tokenizer.from_yaml` passes the saved config as kwargs, so it wins over
  overrides just as it does over `-c`.
- **Polars everywhere**, lazy by default. Frames are built as `LazyFrame` and
  written with `sink_parquet(..., engine="streaming")` to stay memory-bounded;
  `--verbose` forces collection for stats and can OOM on large data.
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
- **Codes** are `PREFIX//value`. The value is lowercased, whitespace and commas
  become `_`, and runs of `_` collapse to one; the prefix is left as written, and
  `text_value` is only lowercased with whitespace→`_`. The `ordering` list in the
  tokenization config breaks ties between events at the same timestamp; a prefix
  missing from `ordering` sorts after the listed ones but still before `EOS`,
  which `tokenize_data` always puts last. When adding a new event prefix, add it
  to `ordering` too, and give it a name under `prefixes` in `visualization.yaml`;
  otherwise its lane is labeled with the bare prefix.
- **The visualizer only reads; it is coupled to the stages' formats.** It loads
  `tokenizer.yaml` with PyYAML's C loader rather than OmegaConf, for speed on
  large vocabularies. It reads `tokens_times.parquet`, plus
  `subject_splits.parquet` (pass-through columns become subject fields) and
  `{split}_for_inference.parquet` when present. It also reads what released
  versions wrote (anything since 26.6.2). Before 26.9.0, times were naive UTC,
  which `get_timeline` localizes; a test rewrites today's output into that format
  (`as_written_by_26_6`) and checks it draws the same. When tokens were written
  without `include_numeric_values`, it recovers values from `meds.parquet` by
  re-binning them the way `Tokenizer.bin_data` does. It spots a fused bin as an
  uppercase `_Q<n>`, which only works because code values are lowercased. So a
  change to binning or to code normalization needs a matching change in
  `Visualizer.get_values` / `FUSED_BIN`. The page has to stay self-contained:
  fonts and icon are inlined as base64, with no external URLs, so an exported
  file can be sent to someone without cocoa or the data. Its static
  `#cocoa-fallback` note tells a viewer that doesn't run scripts (an email or
  file preview) to open it in a browser; the script removes the note once the
  page is drawn. The pdf is drawn in Python from the same payload, by a
  dependency-free writer that embeds the packaged Gotham whole (as Type0 fonts
  addressed by glyph id, with a ToUnicode map so text copies out) and sets codes
  and tokens in standard Courier (WinAnsi text); a character a font lacks prints
  as `?`. A missing font file falls back to Helvetica, as the page falls back to
  system fonts. It ports the page's lanes, axis ticks, and binned rows from
  `timeline.js`, so a change to the payload or to how the page draws it needs a
  matching change in `TimelinePdf`.
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

## Conventions

- **Style**: ruff, line length 88, double quotes, LF,
  `skip-magic-trailing-comma`. Lint set is `E,F,I` (isort included; first-party =
  `cocoa`, `cotorra`, `coreopsis`).
- Files open with `#!/usr/bin/env python3` and a short lowercase module
  docstring; method docstrings are terse and lowercase. Match the surrounding
  terseness.
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
      `drop_duplicate_events`, `default_time_unit`, `min_training_ct`, and
      `include_hours_to_end_time` all follow this pattern.
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
  `__version__` comes from installed package metadata, not a literal.
- Keep [README.md](README.md), the [recipes/](recipes/) (mirrored into
  [docs/recipes/](docs/recipes/)), the per-module pages in
  [docs/api/](docs/api/), and the shipped default configs in sync when behavior
  changes. The README is the PyPI long description, and the recipes are the
  primary user-facing docs.

## Gotchas

- Don't commit anything under `data-raw/` or `processed/` (real patient data;
  gitignored and symlinked to shared storage on HPC).
- Editing a shipped `src/cocoa/config/*.yaml` changes the default behavior for
  every user — usually you want a separate config passed with `-c` instead.
- `combine-datasets` does **not** refuse to merge processed dirs whose tokenizer
  configs differ. It diffs the yamls (ignoring `created_dttm`) and logs a warning
  with the diff, but still combines the parquets and writes the _first_ input's
  `tokenizer.yaml`, so check its output for `Configuration mismatch`. It also
  handles a legacy Int64-token schema from tokenizers `<= 26.4.0`. Only
  `tokenizer.yaml` is written to the processed dir, so a `default_timezone` or
  `default_time_unit` mismatch escapes that config diff and instead surfaces as a
  polars `SchemaError` on the datetime columns — which the legacy-schema fallback
  does not repair, since it only recasts token columns.
