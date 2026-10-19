<p align="center">
<img src="https://raw.githubusercontent.com/burkh4rt/cocoa/master/img/cocoa-bean.png" alt="cocoa bean" width="400"/>
</p>

# Cocoa: a configurable collator

[![PyPI Version](https://img.shields.io/pypi/v/cocoa-tokenizer)](https://pypi.org/project/cocoa-tokenizer/)
[![DOI](https://raw.githubusercontent.com/burkh4rt/cocoa/master/img/1174829117.svg)](https://doi.org/10.5281/zenodo.20413460)
[![SWH](https://archive.softwareheritage.org/badge/origin/https://github.com/bbj-lab/cocoa/)](https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/bbj-lab/cocoa)

> ☕️ Chicago's second favorite bean

## About

This repo provides a configurable way to collate data from multiple sources into
a single denormalized dataframe and create tokenized timelines from the results.
It benefits from previous experience collating data to train foundation models on
tokenized electronic health records. [^1] [^2] [^3] [^4] [^5]

<!-- cards-anchor -->

## Installation

Install the latest release from PyPI:

```sh
pip install cocoa-tokenizer
```

This installs the `cocoa` command. To see it run end to end on public data,
follow the
[MIMIC-IV demo](https://github.com/bbj-lab/cocoa/blob/master/recipes/mimic-iv-demo.md).
The examples below come from that demo and use its paths: it reads CLIF tables
from `data-raw/demo` and writes processed data to `processed/demo`.

## (1) Collation

The collator pulls from raw data tables (parquet or csv) and combines them into a
single denormalized dataframe in a
[MEDS](https://github.com/Medical-Event-Data-Standard/meds)-like format. Each row
in the output represents an event with a `subject_id`, `time`, `code` (all
mandatory), and optional `numeric_value` / `text_value` columns.

Collation is driven by a YAML config (the package ships a default; see
[`./src/cocoa/config/collation.yaml`](https://github.com/bbj-lab/cocoa/blob/master/src/cocoa/config/collation.yaml))
that specifies:

- A **reference table** with a primary key (`subject_id`), start/end times, and
  optional augmentation joins (e.g. joining a patient demographics table).
- A list of **entries**, each mapping a source table (or the reference frame
  itself via `table: REFERENCE`) to the output schema. Each entry declares which
  column provides the `code`, `time`, and optionally `numeric_value`, and
  `text_value`. Codes can be given a prefix `prefix`. Some preprocessing can be
  done with optional entries for `filter_expr`, `with_col_expr`, `agg_expr`, and
  `key`. These take the form of polars expressions that are evaluated and applied
  to the dataframe during loading. _Mild checks are performed when evaluating
  these expressions, but in general, the yaml config is just as powerful as the
  python. Check all yaml files prior to use._
- **Subject splits** (`train_frac` / `tuning_frac`) that partition subjects
  chronologically into train, tuning, and held-out sets.

A collation config has three top-level sections: identifiers, subject splits, and
the reference + entries that define which events to extract.

### Identifiers and splits

```yaml
subject_id: hospitalization_id # the atomic unit of interest
group_id: patient_id # multiple subjects can belong to a group
default_timezone: UTC # or, e.g. America/Chicago; times are stored in this zone
default_time_unit: us # precision of stored times: 'ms', 'us', or 'ns'
drop_duplicate_events: false # collapse rows an entry repeats verbatim
max_lookback_days: null # drop events more than this many days before start_time
max_lookahead_days: null # drop events more than this many days after end_time

subject_splits:
    train_frac: 0.7
    tuning_frac: 0.1
    # the remainder is held out
```

`subject_id` is the column that uniquely identifies each subject (e.g. a
hospitalization). `group_id` is an optional higher-level grouping column.
Subjects are sorted chronologically and split into train / tuning / held-out sets
according to the specified fractions; with `group_id`, groups are sorted instead,
each by its earliest start time, keeping every group in one split. Ties are
broken by id, so the split is the same every run.

All timestamps are stored as timezone-aware datetimes in `default_timezone` (an
IANA name like `America/Chicago`, defaulting to `UTC` if unset). Columns that
already carry a timezone are converted instant-preserving; columns without one
are assumed to be local times in `default_timezone` and labeled as such. Where a
daylight-saving shift makes such a local time ambiguous, the later instant is
used; a local time that a shift skips over is an error. A csv carries no schema,
so its datetimes arrive as strings and are parsed under the same two rules: one
written with a UTC offset is converted, one without is localized.

The timezone is part of the stored dtype, so it travels with the data through the
rest of the pipeline. Durations are computed from instants and are therefore
unaffected by the choice: time spacing tokens, and the winnowing thresholds and
horizons, measure elapsed physical time across a daylight-saving shift. Clock
tokens are different — `CLCK//HH` marks an hour of the _local_ day in
`default_timezone` (see [Tokenization](#2-tokenization)). And because processed
timestamps keep their timezone, `cocoa combine-datasets` cannot merge processed
directories that were collated with different `default_timezone` values.

Timestamps are stored at `default_time_unit` precision, one of the three polars
datetime units — `ms`, `us` (the default), or `ns`. Raw columns are repinned to
it on load whatever precision they arrive at, so a single unit holds across
tables; set it to match an upstream table or a downstream consumer. As with the
timezone, the unit is part of the stored dtype and travels with the data, so
`combine-datasets` likewise cannot merge directories collated at different units.
Durations are derived from the stored instants, so the choice changes no token or
label — but note that `ns` cannot represent times outside 1677–2262.

`drop_duplicate_events` collapses duplicate events as each entry is collated. Raw
tables can carry the same event more than once — a result posted by two systems,
a vital charted twice, a row multiplied by a join — and each copy would otherwise
become another token at the same timestamp. Rows are compared on the whole
collated event, so two copies are dropped to one only when their `subject_id`,
`time`, `code`, `numeric_value`, and `text_value` all agree.

`max_lookback_days` and `max_lookahead_days` bound how far a subject's events may
reach outside its reference window, from `start_time` to `end_time` (see
[Reference table](#reference-table)). An event more than `max_lookback_days`
before its subject's start time, or more than `max_lookahead_days` after its end
time, is dropped; one exactly at the limit is kept. The limits apply to every
entry, unlike `reference_key`, which holds a single entry to the window itself. A
day is 24 hours of elapsed time, so like the other durations the limits are
unaffected by daylight-saving shifts, and fractional days are allowed. Either
limit can be set alone, and `0` drops everything outside the window on that side.
Unset or `null` (the default), a limit drops nothing; so does a null bound, or a
subject missing from the reference table, since there is no time to measure from.

### Reference table

The reference table is the primary static table to which other static information
can be joined:

```yaml
reference:
    table: clif_hospitalization
    start_time: admission_dttm
    end_time: discharge_dttm

    augmentation_tables:
        - table: clif_patient
          key: patient_id
          validation: "m:1"
          with_col_expr: pl.lit("age").alias("AGE")
```

- `table` — the name of the parquet (or csv) file in `--raw-data-home` (without
  the extension).
- `start_time` / `end_time` — columns that define the subject's time window; used
  to filter events from other tables when `reference_key` is set (see below).
- `augmentation_tables` — optional list of tables to join onto the reference
  frame. Each needs a `key` to join on and a `validation` mode (e.g. `"m:1"`).
  You can also add computed columns via `with_col_expr`.

### Pass-through columns

The `pass_through_columns` option allows you to preserve static columns from the
reference table and include them in the output files. This is useful for
demographic and contextual data that should accompany the collated events:

```yaml
pass_through_columns:
  - age_at_admission
  - admission_type_category
  ...
```

Columns specified in this list will be copied from the reference table to:

- `subject_splits.parquet`
- `*_for_inference.parquet` files (e.g., `train_for_inference.parquet`,
  `tuning_for_inference.parquet`, `held_out_for_inference.parquet`) — for use in
  downstream tasks where you may need subject metadata alongside predictions

### Entries

The `entries` list defines the events to extract. Every entry produces rows with
the columns `subject_id`, `time`, `code`, `numeric_value`, and `text_value`. The
entry's fields tell the collator which source columns map to these outputs.

**Required fields:**

| Field   | Description                                                         |
| ------- | ------------------------------------------------------------------- |
| `table` | Source table name, or `REFERENCE` to pull from the reference frame. |
| `code`  | Column whose values become the event code.                          |
| `time`  | Column whose values become the event timestamp.                     |

**Optional fields:**

| Field           | Description                                                                                                                      |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `prefix`        | String prepended to the code (separated by `//`), e.g. `LAB-RES`.                                                                |
| `numeric_value` | Column to use as the numeric value for the event.                                                                                |
| `text_value`    | Column to use as the text value for the event.                                                                                   |
| `filter_expr`   | A Polars expression (or list of expressions) to filter rows before extraction.                                                   |
| `with_col_expr` | A Polars expression (or list) to add computed columns before extraction.                                                         |
| `agg_expr`      | A Polars aggregation expression (or list) applied via `group_by(...).agg(...)` before extraction.                                |
| `key`           | Grouping key used with `agg_expr`. Defaults to `subject_id` when not provided.                                                   |
| `reference_key` | Join the source table to the reference frame on this key and keep only rows within the subject's `start_time`–`end_time` window. |

**Examples:**

- A simple categorical event from the reference frame:

    ```yaml
    - table: REFERENCE
      prefix: DSCG
      code: discharge_category
      time: discharge_dttm
    ```

    creates codes such as `DSCG//expired`, `DSCG//home`, `DSCG//hospice` with
    time `discharge_dttm`.

- A numeric event from an external table:

    ```yaml
    - table: clif_labs
      prefix: LAB-RES
      code: lab_category
      numeric_value: lab_value_numeric
      time: lab_result_dttm
    ```

    creates codes such as `LAB-RES//alt` and `LAB-RES//ast` with numeric_value
    `lab_value_numeric` at time `lab_result_dttm`.

- Tables can be filtered prior to extraction with `filter_expr`:

    ```yaml
    - table: clif_position
      prefix: POSN
      filter_expr: pl.col("position_category") == "prone"
      code: position_category
      time: recorded_dttm
    ```

    selects only rows where `pl.col("position_category") == "prone"`

- Multiple filters can be applied as a list:

    ```yaml
    - table: clif_medication_admin_intermittent_converted
      prefix: MED-INT
      filter_expr:
          - pl.col("mar_action_category") == "given"
          - pl.col("_convert_status") == "success"
      code: med_category
      numeric_value: med_dose_converted
      time: admin_dttm
    ```

- Pre-aggregating events before token extraction with `agg_expr`:

    ```yaml
    - table: clif_crrt_therapy
      prefix: LABEL
      filter_expr: pl.col("crrt_mode_category").is_not_null()
      with_col_expr: pl.lit("crrt_init").alias("code")
      agg_expr:
          - pl.col("code").sort_by("recorded_dttm").first()
          - pl.col("recorded_dttm").min()
      code: code
      time: recorded_dttm
    ```

- Creating a computed column with `with_col_expr` to use as the code:

    ```yaml
    - table: clif_respiratory_support_processed
      prefix: RESP
      with_col_expr: pl.lit("fio2_set").alias("code")
      filter_expr: pl.col("fio2_set").is_finite()
      code: code
      numeric_value: fio2_set
      time: recorded_dttm
    ```

- The `reference_key` can be used to restrict events to a subject's time window:

    ```yaml
    - table: clif_code_status
      prefix: CODE
      code: code_status_category
      time: admission_dttm
      reference_key: patient_id
    ```

### Outputs

- `meds.parquet` gives a table of the collated events. Here are some of the
  demo's, from the minutes around hospitalization 27708593's transfer to the ICU,
  sorted by time:

    ```
    ┌────────────┬─────────────────────────┬───────────────────────┬───────────────┬────────────┐
    │ subject_id ┆ time                    ┆ code                  ┆ numeric_value ┆ text_value │
    │ ---        ┆ ---                     ┆ ---                   ┆ ---           ┆ ---        │
    │ str        ┆ datetime[μs, UTC]       ┆ str                   ┆ f32           ┆ str        │
    ╞════════════╪═════════════════════════╪═══════════════════════╪═══════════════╪════════════╡
    │ 27708593   ┆ 2177-03-15 10:49:00 UTC ┆ VTL//heart_rate       ┆ 143.0         ┆ null       │
    │ 27708593   ┆ 2177-03-15 10:49:00 UTC ┆ VTL//respiratory_rate ┆ 13.0          ┆ null       │
    │ 27708593   ┆ 2177-03-15 10:49:00 UTC ┆ VTL//spo2             ┆ 98.0          ┆ null       │
    │ 27708593   ┆ 2177-03-15 10:49:00 UTC ┆ VTL//weight_kg        ┆ 95.099998     ┆ null       │
    │ 27708593   ┆ 2177-03-15 10:49:26 UTC ┆ XFR-IN//icu           ┆ null          ┆ null       │
    │ …          ┆ …                       ┆ …                     ┆ …             ┆ …          │
    │ 27708593   ┆ 2177-03-15 11:00:00 UTC ┆ ASMT//cam_mental      ┆ null          ┆ no         │
    │ 27708593   ┆ 2177-03-15 11:00:00 UTC ┆ ASMT//cam_total       ┆ null          ┆ no         │
    │ 27708593   ┆ 2177-03-15 11:00:00 UTC ┆ ASMT//gcs_eye         ┆ 4.0           ┆ null       │
    │ 27708593   ┆ 2177-03-15 11:00:00 UTC ┆ ASMT//gcs_motor       ┆ 6.0           ┆ null       │
    │ 27708593   ┆ 2177-03-15 11:00:00 UTC ┆ ASMT//gcs_total       ┆ 15.0          ┆ null       │
    └────────────┴─────────────────────────┴───────────────────────┴───────────────┴────────────┘
    ```

- `subject_splits.parquet` gives a table of every subject_id with its split
  assignment, its start and end times, and its pass-through columns (left out
  here):

    ```
    ┌────────────┬──────────┬─────────────────────────┬─────────────────────────┐
    │ subject_id ┆ split    ┆ start_time              ┆ end_time                │
    │ ---        ┆ ---      ┆ ---                     ┆ ---                     │
    │ str        ┆ str      ┆ datetime[μs, UTC]       ┆ datetime[μs, UTC]       │
    ╞════════════╪══════════╪═════════════════════════╪═════════════════════════╡
    │ 25563031   ┆ train    ┆ 2110-04-11 20:08:00 UTC ┆ 2110-04-14 20:00:00 UTC │
    │ 20291550   ┆ train    ┆ 2110-11-30 11:31:00 UTC ┆ 2110-12-10 20:53:00 UTC │
    │ …          ┆ …        ┆ …                       ┆ …                       │
    │ 29366372   ┆ tuning   ┆ 2167-05-04 02:24:00 UTC ┆ 2167-05-11 17:57:00 UTC │
    │ 22954658   ┆ tuning   ┆ 2167-11-08 00:05:00 UTC ┆ 2167-11-11 19:23:00 UTC │
    │ …          ┆ …        ┆ …                       ┆ …                       │
    │ 27708593   ┆ held_out ┆ 2177-03-12 12:15:00 UTC ┆ 2177-03-19 19:25:00 UTC │
    │ 29281842   ┆ held_out ┆ 2178-07-04 02:13:00 UTC ┆ 2178-07-09 01:20:00 UTC │
    └────────────┴──────────┴─────────────────────────┴─────────────────────────┘
    ```

## (2) Tokenization

The tokenizer consumes the collated parquet output and converts events into
integer token sequences suitable for sequence models. It:

1. Adds `BOS` / `EOS` (beginning/end-of-sequence) tokens to each subject's
   timeline.
2. Optionally inserts configurable clock tokens to mark the passage of time.
3. Computes quantile-based bins for numeric values (from training data only).
4. Optionally inserts configurable time spacing tokens between events.
5. Maps codes (and optionally their binned values) to integer tokens via a
   vocabulary that is formed during training and is frozen for tuning/held-out
   data.
6. Aggregates per-subject token sequences according to time, and then
   configurable sort order.

Tokenization is driven by its own YAML config (the package ships a default; see
[`./src/cocoa/config/tokenization.yaml`](https://github.com/bbj-lab/cocoa/blob/master/src/cocoa/config/tokenization.yaml))
that specifies:

- `n_bins` — number of quantile bins for numeric values.
- `fused` — whether to fuse the code, binned value, and text value into a single
  token (`true`, the default) or keep them as separate tokens (`false`).
- `include_numeric_values` — whether to include raw numeric values alongside
  tokens in the output (`false` by default).
- `include_hours_to_end_time` — whether to include, alongside tokens, the
  fractional hours from each token to its subject's `end_time` as recorded in
  `subject_splits.parquet` (`false` by default). Values count down toward the end
  time and go negative for tokens beyond it, which the `EOS` and time spacing
  tokens of a timeline running past `end_time` can be.
- `min_training_ct` — minimum number of times a word must occur in the training
  split to earn a place in the vocabulary. `0` (the default) disables the
  threshold and keeps every word seen while training. Raise it to prune the long
  tail of one-off codes. Structural tokens (`BOS`, `EOS`, and any `CLCK//` or
  `TIME//` tokens) are exempt and always learned.
- `shuffle_rows` — whether to shuffle the order of subjects' rows in
  `tokens_times.parquet` (`false` by default). Each subject's row is unchanged;
  only the order of rows changes, the same way on every run.
- `insert_spacers` — whether to insert time spacing tokens between events. Each
  marks the gap between two consecutive instants and opens the later one, ahead
  of the events there.
- `insert_clocks` — whether to insert clock tokens at specified times.
- `ordering` — the priority order of code prefixes when sorting events within the
  same timestamp. A prefix missing from it sorts after the listed ones, though
  still before `EOS`, which always ends the timeline. If `TIME` is missing, time
  spacing tokens instead sort first, so they still open their instant.
- `spacers` — mapping of time intervals (e.g., `5m-15m`, `1h-2h`) to their lower
  bounds in minutes, used for time spacing tokens.
- `clocks` — list of hour strings (e.g., `00`, `04`, ...) at which to insert
  clock tokens. Hours are read in the collation config's `default_timezone`, so
  across a daylight-saving shift a repeated hour yields two tokens and a skipped
  hour yields none.

### Outputs

- `tokens_times.parquet` gives one row per subject, ordered by when each timeline
  opens and then by `subject_id` (or shuffled, under `shuffle_rows`). By default
  it has three columns:
    - `subject_id`
    - `tokens` — the integer token sequence for the subject's timeline.
    - `times` — a parallel list of timestamps, one per token, indicating when
      each event occurred.

    A `numeric_values` column, holding the corresponding values for numeric value
    tokens, is added only when `include_numeric_values` is set, and an
    `hours_to_end_time` column only when `include_hours_to_end_time` is.

    For the demo, the table looks like this:

    ```
    ┌────────────┬───────────────────┬─────────────────────────────────┐
    │ subject_id ┆ tokens            ┆ times                           │
    │ ---        ┆ ---               ┆ ---                             │
    │ str        ┆ list[u32]         ┆ list[datetime[μs, UTC]]         │
    ╞════════════╪═══════════════════╪═════════════════════════════════╡
    │ 25563031   ┆ [86, 5, … 103]    ┆ [2110-04-11 20:08:00 UTC, 2110… │
    │ 20291550   ┆ [86, 1088, … 103] ┆ [2110-11-30 09:45:00 UTC, 2110… │
    │ …          ┆ …                 ┆ …                               │
    │ 21477991   ┆ [86, 1090, … 103] ┆ [2201-12-11 08:04:13 UTC, 2201… │
    └────────────┴───────────────────┴─────────────────────────────────┘
    ```

    In this example, token 86 corresponds to the beginning-of-sequence token
    (`BOS`), token 103 to the end-of-sequence token (`EOS`), and the tokens in
    between correspond to the subject's clinical events in chronological order
    (with ties broken by the configured `ordering`). In fused mode each event is
    a single token; in unfused mode an event with a numeric value becomes two
    tokens (code + quantile bin).

- `tokenizer.yaml` is a plain yaml file that contains information about the
  configuration, learned vocabulary, and bins. This file is sufficient to
  reconstitute the tokenizer object. Currently, there's an entry for the lookup
  that maps strings to tokens:

    ```yaml
    lookup:
      UNK: 0
      ADMN//direct: 1
      ADMN//ed: 2
      ADMN//elective: 3
      AGE//age_Q0: 4
      …
    ```

    and an entry for bin cutpoints, listed by code:

    ```yaml
    bins:
      LAB-RES//platelet_count:
        - 34.0
        - 83.0
        - 115.0
        - 145.0
        - 170.0
        - 202.0
        - 241.0
        - 292.0
        - 359.0
      VTL//heart_rate:
        - 67.0
        - 75.0
        - 81.0
        - 85.0
        - 90.0
        - 95.0
        - 101.0
        - 106.0
        - 114.0
      …
    ```

    The lists following each key correspond to the cutpoints for the associated
    category.

<!-- prettier-ignore-start -->
> [!TIP]
> To train a generative event model on this data, check out our configurable
> trainer: [🦜 cotorra](https://pypi.org/project/cotorra/)
<!-- prettier-ignore-end -->

## (3) Winnowing

The winnower prepares timelines for evaluation by splitting each at a threshold
and flagging the outcome tokens on either side. It:

1. Loads the tokenized timelines of each split it is configured to prepare.
2. Splits each subject's timeline at a configurable threshold, either a duration
   after the timeline's first event, a duration after the subject's start time,
   or the first occurrence of a specified token, separating events into "past"
   (before the threshold) and "future" (after it).
3. Checks for the presence of outcome tokens in both the past and future periods.
4. Filters out subjects whose timelines never reach the threshold, ensuring
   subjects have sufficient observation time.
5. Outputs a winnowed dataset suitable for inference and evaluation tasks.

Winnowing is driven by a YAML config (the package ships a default; see
[`./src/cocoa/config/winnowing.yaml`](https://github.com/bbj-lab/cocoa/blob/master/src/cocoa/config/winnowing.yaml))
that specifies:

- `outcome_tokens` — list of event codes to track as outcomes (e.g.,
  `XFR-IN//icu`, `DSCG//expired`). The winnower creates binary flags for each
  outcome indicating whether that token appears in the past or future period.
- `threshold` — defines how the threshold is set. Currently supported options are
  as follows:
    - `duration_s` (integer) thresholds a given duration (in seconds) after the
      timeline's first event
    - `after_start_time_s` (integer) thresholds a given duration (in seconds)
      after the subject's start time, the `start_time` of the
      [reference table](#reference-table); see
      [below](#duration_s-vs-after_start_time_s) for how it differs from
      `duration_s`
    - `first_occurrence` (token string) thresholds after the first occurrence of
      the provided token

  Set only one of them: given several, the winnower uses the first in this list.
- `horizon_after_threshold_s` is an optional parameter that allows you to set a
  prediction window (in seconds) after the threshold is triggered
- `splits` — an optional list selecting which splits to prepare (defaults to
  `held_out`). Rep-based inference also requires the `train` and `tuning` splits.

**Example configuration:**

```yaml
outcome_tokens: # supports patterns with fnmatch
    - XFR-IN//icu # ICU transfer
    - RESP//imv # invasive mechanical ventilation event
    - DSCG//expired # discharge due to death
    - LABEL//* # any kind of label token
threshold:
    # choose one and only one of the following (given several, the first wins)
    duration_s: !!int 86400 # 24h after the timeline's first event
    # after_start_time_s: !!int 86400 # 24h after the subject's start_time
    # first_occurrence: XFR-IN//icu

horizon_after_threshold_s: !!int 2592000 # 30d outcome window after prediction threshold

splits: # select which splits to prepare
    - train
    - tuning
    - held_out
```

### `duration_s` vs. `after_start_time_s`

Both thresholds cut a timeline a set number of seconds after some time, but not
after the same time:

- `duration_s` counts from the timeline's **first event**, the time of its `BOS`
  token, so every past spans the same stretch of record. Where the cut falls in
  a stay depends on what was recorded first: a timeline that opens before its
  admission is cut that much earlier in the stay, and one that opens after it,
  that much later.
- `after_start_time_s` counts from the subject's **start time**: the reference
  table's `start_time` column (`admission_dttm` in the shipped collation
  config), as `subject_splits.parquet` records it. Every stay is cut at the same
  point, such as 24 hours after admission, whatever the timeline holds, and
  events recorded before the start time land in the past.

The two cut a timeline at the same point only if it opens at its start time. With
the shipped collation config, which times its `REFERENCE` entries at
`admission_dttm`, no timeline opens after its admission, but many open before
it: on the demo, 203 of the 275 timelines do, by as much as a day, 180 of them
with an arrival in the emergency department (`XFR-IN//ed`). Held-out
hospitalization 29079034 arrives there 20 hours before its admission, so
`duration_s: 86400` cuts it less than 4 hours into its stay, with 57 tokens in
its past, where `after_start_time_s: 86400` cuts it a day in, with 197.

Under either, an event at exactly the cut falls in the future, and a timeline is
kept only if it runs past the cut. Under `after_start_time_s`, it must also begin
before the cut, since its past would otherwise be empty; under `duration_s`, the
past always holds `BOS`.

### Outputs

- `held_out_for_inference.parquet` has columns for each outcome token (e.g.,
  `XFR-IN//icu_past`, `XFR-IN//icu_future`) indicating whether that outcome
  occurred in the respective time period.
- `train_for_inference.parquet` and `tuning_for_inference.parquet` are also
  provided; these are required to make rep-based predictions

These files are written to the processed data directory unless `cocoa winnow` is
given `--output-home` / `-o`, which sends them to another directory (created if
need be) and leaves the processed data directory untouched. The files of the
processed data directory are copied there too, so that it is a complete processed
dataset of its own (one `cocoa visualize -p` can read, say); its subdirectories
are not, nor are any `*_for_inference.parquet` files already there, which may
come from another configuration. That way one tokenized dataset can be winnowed
under several configurations, each into its own directory:

```sh
cocoa winnow -p processed/demo -o processed/demo/winnowed-24h
cocoa winnow -p processed/demo -o processed/demo/winnowed-admit-24h \
    '~threshold.duration_s' threshold.after_start_time_s=86400
cocoa winnow -p processed/demo -o processed/demo/winnowed-icu \
    '~threshold.duration_s' threshold.first_occurrence=XFR-IN//icu
```

On the demo, the first keeps 50 of the 55 held-out hospitalizations, those whose
timelines run past 24 hours; the second the same 50, which also run past 24 hours
after admission, though it cuts 24 of them later; and the third 26, those with a
transfer to the ICU.

## Usage

We provide a CLI that should be sufficient for most use cases (to change a few
config keys from the command line, see
[Overriding config keys](#overriding-config-keys)):

```
 Usage: cocoa [OPTIONS] COMMAND [ARGS]...

 Configurable collation and tokenization (v26.10.1)

╭─ Options ───────────────────────────────────────────────────────────────╮
│ --install-completion            Install completion for the current      │
│                                 shell.                                  │
│ --show-completion               Show completion for the current shell,  │
│                                 to copy it or customize the             │
│                                 installation.                           │
│ --help                -h        Show this message and exit.             │
╰─────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────╮
│ collate           Collate raw data into a denormalized format.          │
│ tokenize          Tokenize collated data into integer sequences.        │
│ winnow            Winnow timelines for evaluation.                      │
│ pipeline          Run the full pipeline: collate, tokenize, & winnow.   │
│ combine-datasets  Combine multiple processed datasets into one.         │
│ visualize         Visualize a subject's timeline as an interactive html │
│                   page.                                                 │
╰─────────────────────────────────────────────────────────────────────────╯
```

with commands:

- `cocoa collate`

    ```
    Usage: cocoa collate [OPTIONS] [overrides]...

    Collate raw data into a denormalized format.

    Reads collation configuration and produces a MEDS-like parquet file
    with collated events.

    ╭─ Arguments ─────────────────────────────────────────────────────────────╮
    │   overrides      <str>  Config overrides: key=value sets a key, adding  │
    │                         it if need be, and ~key deletes one             │
    ╰─────────────────────────────────────────────────────────────────────────╯
    ╭─ Options ───────────────────────────────────────────────────────────────╮
    │    --collation-config     -c      <path>  Collation configuration file  │
    │                                           (overrides default)           │
    │ *  --raw-data-home        -r      <str>   Raw data directory [required] │
    │ *  --processed-data-home  -p      <str>   Processed data directory      │
    │                                           [required]                    │
    │    --verbose              -v              Verbose logging for collate;  │
    │                                           this may cause memory issues  │
    │                                           with large datasets           │
    │    --help                 -h              Show this message and exit.   │
    ╰─────────────────────────────────────────────────────────────────────────╯
    ```

- `cocoa tokenize`

    ```
    Usage: cocoa tokenize [OPTIONS] [overrides]...

    Tokenize collated data into integer sequences.

    Reads collated parquet files and produces tokenized timelines with
    vocabulary and bin information.

    ╭─ Arguments ─────────────────────────────────────────────────────────────╮
    │   overrides      <str>  Config overrides: key=value sets a key, adding  │
    │                         it if need be, and ~key deletes one             │
    ╰─────────────────────────────────────────────────────────────────────────╯
    ╭─ Options ───────────────────────────────────────────────────────────────╮
    │    --tokenization-config  -c      <path>  Tokenization configuration    │
    │                                           file (overrides default)      │
    │ *  --processed-data-home  -p      <str>   Processed data directory      │
    │                                           [required]                    │
    │    --tokenizer-home       -t      <str>   Load a previously learned     │
    │                                           tokenizer from this           │
    │                                           tokenizer.yaml file (reuses   │
    │                                           its frozen vocabulary and     │
    │                                           bins)                         │
    │    --verbose              -v              Verbose logging for tokenize; │
    │                                           this may cause memory issues  │
    │                                           with large datasets           │
    │    --help                 -h              Show this message and exit.   │
    ╰─────────────────────────────────────────────────────────────────────────╯
    ```

- `cocoa winnow`

    ```
    Usage: cocoa winnow [OPTIONS] [overrides]...

    Winnow timelines for evaluation.

    Splits each timeline of the configured splits into a past and a future
    at a threshold, drops those that never reach it, and flags which outcome
    tokens fall on each side.

    ╭─ Arguments ─────────────────────────────────────────────────────────────╮
    │   overrides      <str>  Config overrides: key=value sets a key, adding  │
    │                         it if need be, and ~key deletes one             │
    ╰─────────────────────────────────────────────────────────────────────────╯
    ╭─ Options ───────────────────────────────────────────────────────────────╮
    │    --winnowing-config     -c      <path>  Winnowing configuration file  │
    │                                           (overrides default)           │
    │ *  --processed-data-home  -p      <str>   Processed data directory      │
    │                                           [required]                    │
    │    --output-home          -o      <str>   Directory to write the        │
    │                                           winnowed files to, along with │
    │                                           copies of the processed data  │
    │                                           directory's files (defaults   │
    │                                           to the processed data         │
    │                                           directory)                    │
    │    --verbose              -v              Verbose logging for winnow;   │
    │                                           prints summary statistics     │
    │    --help                 -h              Show this message and exit.   │
    ╰─────────────────────────────────────────────────────────────────────────╯
    ```

- `cocoa pipeline`

    ```
    Usage: cocoa pipeline [OPTIONS] [overrides]...

    Run the full pipeline: collate, tokenize, & winnow.

    ╭─ Arguments ─────────────────────────────────────────────────────────────╮
    │   overrides      <str>  Config overrides, each key starting with the    │
    │                         stage it's for: e.g. tokenization.n_bins=5 or   │
    │                         ~winnowing.threshold.duration_s                 │
    ╰─────────────────────────────────────────────────────────────────────────╯
    ╭─ Options ───────────────────────────────────────────────────────────────╮
    │    --collation-config             <path>  Collation configuration file  │
    │                                           (overrides default)           │
    │    --tokenization-config          <path>  Tokenization configuration    │
    │                                           file (overrides default)      │
    │    --winnowing-config             <path>  Winnowing configuration file  │
    │                                           (overrides default)           │
    │ *  --raw-data-home        -r      <str>   Raw data directory [required] │
    │ *  --processed-data-home  -p      <str>   Processed data directory      │
    │                                           [required]                    │
    │    --verbose              -v              Verbose logging for pipeline  │
    │                                           steps                         │
    │    --help                 -h              Show this message and exit.   │
    ╰─────────────────────────────────────────────────────────────────────────╯
    ```

- `cocoa combine-datasets`

    ```
    Usage: cocoa combine-datasets [OPTIONS] {input_data_dirs}...

    Combine multiple processed datasets into one.

    Merges the parquet files of the input directories and keeps the first
    directory's tokenizer.yaml, warning if another's differs.

    ╭─ Arguments ─────────────────────────────────────────────────────────────╮
    │ *    input_data_dirs      <str>  Processed data directories to combine  │
    │                                  [required]                             │
    ╰─────────────────────────────────────────────────────────────────────────╯
    ╭─ Options ───────────────────────────────────────────────────────────────╮
    │ *  --output-data-dir  -o      <str>  Output directory for combined data │
    │                                      [required]                         │
    │    --help             -h             Show this message and exit.        │
    ╰─────────────────────────────────────────────────────────────────────────╯
    ```

- `cocoa visualize`

    ```
    Usage: cocoa visualize [OPTIONS] {subject_id} [overrides]...

    Visualize a subject's timeline as an interactive html page.

    Serves the page on localhost until interrupted, or saves it as a
    self-contained html file with --export-html, a static pdf with
    --export-pdf, or both.

    ╭─ Arguments ─────────────────────────────────────────────────────────────╮
    │ *    subject_id      <str>  Subject whose timeline to show [required]   │
    │      overrides       <str>  Config overrides: key=value sets a key,     │
    │                             adding it if need be, and ~key deletes one  │
    ╰─────────────────────────────────────────────────────────────────────────╯
    ╭─ Options ───────────────────────────────────────────────────────────────╮
    │ *  --processed-data-home   -p      <str>   Processed data directory     │
    │                                            [required]                   │
    │    --visualization-config  -c      <path>  Visualization configuration  │
    │                                            file (overrides default)     │
    │    --export-html           -e      <path>  Save the timeline to this    │
    │                                            self-contained html file     │
    │                                            instead of serving it        │
    │    --export-pdf                    <path>  Save a static pdf of the     │
    │                                            timeline to this file        │
    │                                            instead of serving it; with  │
    │                                            --export-html, save both     │
    │    --host                          <str>   Address to serve the page on │
    │                                            [default: 127.0.0.1]         │
    │    --port                          <int>   Port to serve the page on; 0 │
    │                                            picks any                    │
    │                                            [default: 8765]              │
    │    --open                                  Open the page in a web       │
    │                                            browser                      │
    │    --show-winnowing                        Show the winnowing split     │
    │                                            into past and future, and    │
    │                                            its outcome flags            │
    │    --help                  -h              Show this message and exit.  │
    ╰─────────────────────────────────────────────────────────────────────────╯
    ```

### Overriding config keys

To change a few keys without writing a new config file, list them after a
command's options, as dotted paths into its config:

```sh
cocoa tokenize -p processed/demo n_bins=5 insert_spacers=true
```

`cocoa pipeline` takes them too, each key starting with the stage it's for:

```sh
cocoa pipeline -r data-raw/demo -p processed/demo \
    collation.default_timezone=America/New_York tokenization.n_bins=5 \
    '~winnowing.threshold.duration_s' winnowing.threshold.first_occurrence=XFR-IN//icu
```

It checks every stage's overrides before collating, so an unreadable winnowing
override fails at once rather than after the first two stages have run.

The syntax borrows from
[Hydra's](https://hydra.cc/docs/advanced/override_grammar/basic/):

- `key=value` sets a key, adding it if it isn't there, e.g. a
  `horizon_after_threshold_s` for winnowing. Keys aren't checked against the
  config, so a misspelled one is added quietly and does nothing. A leading `+` or
  `++`, as Hydra marks an addition, is accepted and ignored.
- `~key` deletes a key.

Overrides apply in order to the config file passed with `-c` (or to the shipped
default when none is). Values are read as YAML: `0.5` is a float, `[a, b]` a
list, and `{train_frac: 0.5}` a block, merged into any block already there.

A few things to know:

- Setting a key to `null` doesn't switch it off where its presence is what
  counts. The winnower uses whichever of `threshold.duration_s`,
  `threshold.after_start_time_s`, and `threshold.first_occurrence` is there (the
  first in that order, if several are), so switching between them takes `~` on
  one and setting the other, as above. Setting `threshold.after_start_time_s`
  alone changes nothing, since the shipped `threshold.duration_s` outranks it.
- Lists are replaced whole: `entries` can't be edited one item at a time, so a
  change to an entry calls for a config file.
- YAML decides each value's type, so quote what has to stay a string:
  `'clocks=["00", "12"]'`, since an unquoted `00` is the number 0, and
  `'other_color="#737373"'`, since an unquoted `#` starts a comment.
- With `--tokenizer-home`, the loaded tokenizer's saved config wins over
  overrides, as it does over `-c`.
- Quote anything starting with `~` in zsh, which otherwise reads it as a
  directory name and fails.

### Combining datasets

`cocoa combine-datasets` merges processed directories into one, such as the
timelines of two hospitals, or of the two periods that the
[tokenizer transfer recipe](https://github.com/bbj-lab/cocoa/blob/master/recipes/tokenizer-transfer.md)
splits the demo into:

```sh
cocoa combine-datasets processed/demo-2011-2013 processed/demo-2014-2016 \
    -o processed/demo-combined
```

Each parquet file of the first directory (`meds.parquet`,
`subject_splits.parquet`, `tokens_times.parquet`, and any
`{split}_for_inference.parquet`) is concatenated with the file of the same name
in each of the others, and its rows are shuffled, the same way on every run. Each
subject keeps the split it had, so the combined splits are the inputs' splits put
together, not a fresh split of the combined subjects.

The output gets the first directory's `tokenizer.yaml`, so every input's tokens
have to come from that one tokenizer: tokenize each of the others with
`--tokenizer-home` pointing at the first's `tokenizer.yaml`. The inputs'
`tokenizer.yaml` files are compared, but a difference isn't refused:
`combine-datasets` warns of a `Configuration mismatch`, logs the keys that
differ, and combines the inputs anyway. A tokenizer loaded with
`--tokenizer-home` records `is_training: false` where the one it came from
records `true`, so expect that key in the warning; any other key means the inputs
were tokenized differently, and their tokens don't mean the same thing.

A few more things to know:

- Subject ids must not repeat across the inputs. `combine-datasets` doesn't
  check, but `cocoa winnow` then fails on the combined directory, since it pairs
  each timeline with a single row of `subject_splits.parquet`.
- The first directory decides which files are combined: a parquet file only a
  later directory has is left out, and one that a later directory lacks is an
  error. Likewise, a column only the first input has is filled with nulls for the
  others, but a column only a later input has is an error, so list the input with
  the most columns first.
- Inputs collated with different `default_timezone` or `default_time_unit` values
  can't be combined; polars fails with a `SchemaError`.

<!-- prettier-ignore-start -->
> [!TIP]
> For common use cases, check out the [recipes](https://github.com/bbj-lab/cocoa/blob/master/recipes/README.md) section!
<!-- prettier-ignore-end -->

## Contributing

Contributions are welcome, from bug reports and typo fixes to new features. To
get started, see the
[contributing guide](https://github.com/bbj-lab/cocoa/blob/master/CONTRIBUTING.md).

[^1]:
    M. Burkhart, B. Ramadan, Z. Liao, K. Chhikara, J. Rojas, W. Parker, & B.
    Beaulieu-Jones, Foundation models for electronic health records:
    representation dynamics and transferability,
    [arXiv:2504.10422](https://doi.org/10.48550/arXiv.2504.10422)

[^2]:
    M. Burkhart, B. Ramadan, L. Solo, W. Parker, & B. Beaulieu-Jones,
    [Quantifying surprise in clinical care: Detecting highly informative events in electronic health records with foundation models](https://doi.org/10.1142/9789819824755_0013),
    Pacific Symposium on Biocomputing 31 (2026), 173–188

[^3]:
    L. Solo, M. McDermott, W. Parker, B. Ramadan, M. Burkhart, & B.
    Beaulieu-Jones, Efficient generative prediction for EHR foundation models:
    the SCOPE and REACH estimators,
    [arXiv:2602.03730](https://doi.org/10.48550/arXiv.2602.03730)

[^4]:
    I. Lee, L. Solo, M. Burkhart, B. Ramadan, W. Parker, & B. Beaulieu-Jones,
    Representation before training: a fixed-budget benchmark for generative
    medical event models,
    [arXiv:2604.16775](https://doi.org/10.48550/arXiv.2604.16775)

[^5]:
    M. Burkhart, L. Solo, I. Lee, S. Charles, Z. Liao, K. Chhikara, D. Therese,
    W.-T. Liao, C. Gao, W. Parker, & B. Beaulieu-Jones, Federated generative
    event models for tokenized electronic health records,
    [arXiv:2608.02939](https://doi.org/10.48550/arXiv.2608.02939)

<!--

Run in tmux:
```
tmux new -s co || tmux a -t co
```

Format:
```sh
ruff format .
ruff check . --fix
```

Send to bbj-lab1:
```
rsync -avht \
 --delete \
 --exclude "output" \
 --exclude "data-raw" \
 --exclude "processed/" \
 --exclude ".venv" \
 --exclude ".idea" \
 ~/Documents/chicago/cocoa \
 bbj-lab1:~
```


Send to randi:
```
for d in data-raw processed; do
	ln -s /gpfs/data/bbj-lab/users/burkh4rt/$d $d
done
```
```
rsync -avh \
 --exclude "output" \
 --exclude "processed" \
 --exclude "data-raw" \
 --exclude "logs" \
 --exclude "wandb" \
 --exclude ".venv/" \
 --exclude ".idea/" \
 ~/Documents/chicago/cocoa \
 randi:/gpfs/data/bbj-lab/users/burkh4rt
```

Send to pypi:
```
rm -rf dist
python3 -m pip install --upgrade build
python3 -m build
python3 -m pip install --upgrade twine
python3 -m twine upload --repository pypi dist/*
```

Make docs:
```
mkdocs build
mkdocs serve --dev-addr 127.0.0.1:8001
```

Make tag:
```
git tag -s v26.6.1 -m "docs may be read"
```
-->
