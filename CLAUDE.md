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

Data flows strictly stage-to-stage through files in `--processed-data-home`.

## Commands

```sh
# Dev install (Python >= 3.11)
python -m venv .venv && . .venv/bin/activate
pip install -e '.[all]'          # all = dev + docs + test extras

# Run the pipeline (or a single stage: collate | tokenize | winnow)
cocoa pipeline -r <raw-data-home> -p <processed-data-home> [--verbose]
cocoa <stage> -c <config.yaml> ... # -c overrides the shipped default for that stage
cocoa <stage> -h                   # help; --verbose prints summary stats

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
integration.

Each module also keeps an `if __name__ == "__main__"` block that self-tests
against a local processed dataset (e.g. `./processed/mimic/`); run a module
directly (`python -m cocoa.tokenizer`) to exercise it against real data — the
tokenizer block also asserts round-trip save/load equality. When you change
behavior, add/update a pytest case first.

## Architecture notes

- **Config resolution** ([configurable.py](src/cocoa/configurable.py)): every
  stage merges, in increasing precedence, the shipped default YAML → a user `-c`
  config → non-`None` kwargs. A user config that omits a key does **not** inherit
  that key from the default when passed explicitly — read `Configurable.__init__`
  before changing merge logic. Config access is OmegaConf; use `.get(k, default)`
  for optional keys.
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
  `reference.end_time`). It counts down toward that time and goes negative
  beyond it, which `EOS` and trailing spacers can be; being a duration, it is
  tz- and unit-invariant. `Winnower.add_outcome_flags` splits it into
  `_past`/`_future` through its `optional` list — a new per-token column the
  tokenizer writes conditionally has to be named there too, or it rides along
  unsplit the way `numeric_values` does.
- **Codes** are `PREFIX//value` (lowercased, whitespace→`_`). The `ordering` list
  in the tokenization config breaks ties between events at the same timestamp; a
  prefix missing from `ordering` sorts last. When adding a new event prefix, add
  it to `ordering` too.
- **Times** are normalized on load to the collation config's `default_timezone`
  (`Collator.to_default_tz`; `UTC` if unset) and stay **tz-aware** for the rest
  of the pipeline: tz-aware columns are instant-preserved, tz-naive columns are
  assumed to be local times in that zone (ambiguous DST times take the later
  instant; nonexistent ones raise `ComputeError`). A csv has no schema, so its
  datetimes arrive as `String` and `Collator.parse_datetime` resolves them with
  `str.to_datetime(time_zone=...)` before that branch — one call covers both
  rules, since an offset-bearing string converts and a bare one localizes.
  Downstream duration math (spacers, winnowing thresholds/horizons) works on
  instants and is therefore tz-invariant, but `CLCK//HH` tokens carry the
  _local_ hour — the zone is part of what that vocabulary means, and
  `tokenizer.yaml` does not record it, so a transferred tokenizer only agrees
  with a new dataset if both were collated in the same zone.
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
- New pipeline stages subclass `Configurable`, set `default_file`, ship a default
  YAML under `src/cocoa/config/` (packaged via `package-data` in
  [pyproject.toml](pyproject.toml)), and expose `save_all(verbose)`.
- New CLI commands go in [cli.py](src/cocoa/cli.py) as typer commands using
  `rich` for output, mirroring the existing timing/output-path print pattern.
- **Versioning is CalVer** `YY.M.patch` (e.g. `26.6.1`); releases are signed git
  tags `vYY.M.patch`. `__version__` comes from installed package metadata, not a
  literal.
- Keep [README.md](README.md), the [recipes/](recipes/) (mirrored into
  [docs/recipes/](docs/recipes/)), and the shipped default configs in sync when
  behavior changes — the README is the PyPI long description and the recipes are
  the primary user-facing docs.

## Gotchas

- Don't commit anything under `data-raw/` or `processed/` (real patient data;
  gitignored and symlinked to shared storage on HPC).
- Editing a shipped `src/cocoa/config/*.yaml` changes the default behavior for
  every user — usually you want a separate config passed with `-c` instead.
- `combine-datasets` refuses to merge processed dirs whose tokenizer configs
  differ (it diffs the yamls, ignoring `created_dttm`); it also handles a legacy
  Int64-token schema from tokenizers `<= 26.4.0`. Only `tokenizer.yaml` is
  written to the processed dir, so a `default_timezone` or `default_time_unit`
  mismatch escapes that config diff and instead surfaces as a polars
  `SchemaError` on the datetime columns — which the legacy-schema fallback does
  not repair, since it only recasts token columns.
