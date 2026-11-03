# Collation

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
- **Subject splits** (`how`, `train_frac` / `tuning_frac`) that partition
  subjects into train, tuning, and held-out sets, chronologically or at random.

A collation config has three top-level sections: identifiers, subject splits, and
the reference + entries that define which events to extract.

## Identifiers and splits

```yaml
subject_id: hospitalization_id # the atomic unit of interest
group_id: patient_id # multiple subjects can belong to a group
default_timezone: UTC # or, e.g. America/Chicago; times are stored in this zone
default_time_unit: us # precision of stored times: 'ms', 'us', or 'ns'
drop_duplicate_events: false # collapse rows an entry repeats verbatim
max_lookback_days: null # drop events more than this many days before start_time
max_lookahead_days: null # drop events more than this many days after end_time

subject_splits:
    how: time_based # or uniformly_random
    train_frac: 0.7
    tuning_frac: 0.1
    # the remainder is held out
```

`subject_id` is the column that uniquely identifies each subject (e.g. a
hospitalization). `group_id` is an optional higher-level grouping column.
Subjects are put in order and split into train / tuning / held-out sets according
to the specified fractions; with `group_id`, groups are ordered instead, keeping
every group in one split. `how` picks the order. Under `time_based`, the default
(and what a config without `how` gets), subjects are sorted chronologically and
groups each by its earliest start time, with ties broken by id. Under
`uniformly_random`, they're sorted by the SHA-256 digest of their id, an order as
good as a uniformly random shuffle and unrelated to time. Either way the split is
the same every run, and a random split depends on nothing but the ids: not on the
machine, nor on the versions of cocoa's dependencies.

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
`default_timezone` (see [Tokenization](tokenization.md)). And because processed
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

## Reference table

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

## Pass-through columns

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

## Entries

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

An entry with `table: REFERENCE` takes these options too. They apply to the
reference frame as the [reference block](#reference-table) builds it, so they can
use the columns its augmentation tables join on, and they find its `start_time`
and `end_time` already converted to `default_timezone`, where the options of an
entry over any other table see that table's raw columns. They shape that entry's
events alone: the other entries, the subject splits, and the lookback and
lookahead windows still see every row of the reference frame.

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
      time: start_dttm
      reference_key: patient_id
    ```

## Outputs

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
