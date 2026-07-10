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
tokenized electronic health records. [^1] [^2] [^3] [^4]

## Installation

Install the latest release from PyPI:

```sh
pip install cocoa-tokenizer
```

This installs the `cocoa` command. To work from source instead (e.g. for
development):

```sh
git clone git@github.com:bbj-lab/cocoa.git
cd cocoa
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

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

subject_splits:
    train_frac: 0.7
    tuning_frac: 0.1
    # the remainder is held out
```

`subject_id` is the column that uniquely identifies each subject (e.g. a
hospitalization). `group_id` is an optional higher-level grouping column.
Subjects are sorted chronologically and split into train / tuning / held-out sets
according to the specified fractions.

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
          with_col_expr: pl.lit("AGE").alias("AGE")
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

    creates codes such as `DSCG//assisted_living`, `DSCG//home`, `DSCG//hospice`
    with time `discharge_dttm`.

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
          - pl.col("code").first()
          - pl.col("recorded_dttm").first()
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

- `meds.parquet` gives a table of the collated events:

    ```
    ┌────────────┬─────────────────────┬──────────────────────────────┬───────────────┬────────────┐
    │ subject_id ┆ time                ┆ code                         ┆ numeric_value ┆ text_value │
    │ ---        ┆ ---                 ┆ ---                          ┆ ---           ┆ ---        │
    │ str        ┆ datetime[μs]        ┆ str                          ┆ f32           ┆ str        │
    ╞════════════╪═════════════════════╪══════════════════════════════╪═══════════════╪════════════╡
    │ 24591817   ┆ 2111-09-26 18:15:00 ┆ MED-CTS//sodium_chloride     ┆ 0.0           ┆ null       │
    │ 21343412   ┆ 2112-01-11 06:31:00 ┆ LAB-RES//albumin             ┆ 3.3           ┆ null       │
    │ 24894995   ┆ 2113-01-14 14:25:00 ┆ LAB-ORD//creatinine          ┆ null          ┆ null       │
    │ 20947416   ┆ 2110-12-12 18:41:00 ┆ LAB-RES//hemoglobin          ┆ 8.4           ┆ null       │
    │ 25082363   ┆ 2110-06-17 17:00:00 ┆ VTL//respiratory_rate        ┆ 30.0          ┆ null       │
    │ …          ┆ …                   ┆ …                            ┆ …             ┆ …          │
    │ 22074503   ┆ 2110-07-13 03:53:00 ┆ LAB-ORD//chloride            ┆ null          ┆ null       │
    │ 24524153   ┆ 2110-10-08 03:20:00 ┆ LAB-RES//glucose_serum       ┆ 179.0         ┆ null       │
    │ 28104308   ┆ 2112-03-22 14:31:00 ┆ LAB-RES//sodium              ┆ 137.0         ┆ null       │
    │ 23859742   ┆ 2110-08-21 21:35:00 ┆ LAB-RES//ptt                 ┆ 26.299999     ┆ null       │
    │ 25805890   ┆ 2110-10-03 11:00:00 ┆ LAB-ORD//eosinophils_percent ┆ null          ┆ null       │
    └────────────┴─────────────────────┴──────────────────────────────┴───────────────┴────────────┘
    ```

- `subject_splits.parquet` gives a table of all subject_id's and their
  corresponding split assignment:

    ```
    ┌────────────┬──────────┐
    │ subject_id ┆ split    │
    │ ---        ┆ ---      │
    │ str        ┆ str      │
    ╞════════════╪══════════╡
    │ 21081215   ┆ train    │
    │ 20302177   ┆ train    │
    │ …          ┆ …        │
    │ 27116134   ┆ tuning   │
    │ 29134959   ┆ tuning   │
    │ …          ┆ …        │
    │ 28150003   ┆ held_out │
    │ 22151813   ┆ held_out │
    └────────────┴──────────┘
    ```

## (2) Tokenization

The tokenizer consumes the collated parquet output and converts events into
integer token sequences suitable for sequence models. It:

1. Adds `BOS` / `EOS` (beginning/end-of-sequence) tokens to each subject's
   timeline.
2. Optionally inserts configurable clock tokens to mark the passage of time.
3. Optionally inserts configurable time spacing tokens between events.
4. Computes quantile-based bins for numeric values (from training data only).
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
  token (`true`) or keep them as separate tokens (`false`).
- `include_numeric_values` — whether to include raw numeric values alongside
  tokens in the output (`false` by default).
- `insert_spacers` — whether to insert time spacing tokens between events.
- `insert_clocks` — whether to insert clock tokens at specified times.
- `ordering` — the priority order of code prefixes when sorting events within the
  same timestamp.
- `spacers` — mapping of time intervals (e.g., `5m-15m`, `1h-2h`) to their lower
  bounds in minutes, used for time spacing tokens.
- `clocks` — list of hour strings (e.g., `00`, `04`, ...) at which to insert
  clock tokens.

### Outputs

- `tokens_times.parquet` gives one row per subject with three columns by default:
    - `subject_id`
    - `tokens` — the integer token sequence for the subject's timeline.
    - `times` — a parallel list of timestamps, one per token, indicating when
      each event occurred.
    - `numeric_values` - corresponding values for numeric value tokens (only if
      configured)

    The table will look something like this:

    ```
    ┌────────────────────┬─────────────────┬─────────────────────────────────┐
    │ subject_id         ┆ tokens          ┆ times                           │
    │ ---                ┆ ---             ┆ ---                             │
    │ str                ┆ list[u32]       ┆ list[datetime[μs]]              │
    ╞════════════════════╪═════════════════╪═════════════════════════════════╡
    │ 20002103           ┆ [20, 350, … 21] ┆ [2116-05-08 02:45:00, 2116-05-… │
    │ 20008372           ┆ [20, 350, … 21] ┆ [2110-10-30 13:03:00, 2110-10-… │
    │ …                  ┆ …               ┆ …                               │
    │ 29994865           ┆ [20, 364, … 21] ┆ [2111-01-28 21:49:00, 2111-01-… │
    └────────────────────┴─────────────────┴─────────────────────────────────┘
    ```

    In this example, token 20 corresponds to the beginning-of-sequence token
    (`BOS`), token 21 to the end-of-sequence token (`EOS`), and the tokens in
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

    and an entry for bin cutpoints:

    ```yaml
    bins:
      VTL//heart_rate:
        - 65.0
        - 70.0
        - 75.0
        - 80.0
        - 84.0
        - 89.0
        - 94.0
        - 100.0
        - 108.0
      LAB-RES//platelet_count:
        - 62.0
        - 114.0
        - 147.0
        - 175.0
        - 203.0
        - 233.0
        - 267.0
        - 314.0
        - 390.0
      …
    ```

    The lists following each key correspond to the cutpoints for the associated
    category.

<!-- prettier-ignore-start -->
> [!TIP]
> To train a generative event model on this data, check out our configurable
> trainer: [🦜 cotorra](https://github.com/bbj-lab/cotorra)
<!-- prettier-ignore-end -->

## (3) Winnowing

The winnower prepares held-out timelines for evaluation by filtering and flagging
subjects based on outcome criteria. It:

1. Loads held-out data from the tokenized timelines and associated timestamps.
2. Splits each subject's timeline at a configurable time horizon or at the first
   occurrence of a specified token, separating events into "past" (before the
   horizon) and "future" (after the horizon).
3. Checks for the presence of outcome tokens in both the past and future periods.
4. Filters out subjects whose timelines don't exceed the horizon duration,
   ensuring subjects have sufficient observation time.
5. Outputs a winnowed dataset suitable for inference and evaluation tasks.

Winnowing is driven by a YAML config (the package ships a default; see
[`./src/cocoa/config/winnowing.yaml`](https://github.com/bbj-lab/cocoa/blob/master/src/cocoa/config/winnowing.yaml))
that specifies:

- `outcome_tokens` — list of event codes to track as outcomes (e.g.,
  `XFR-IN//icu`, `DSCG//expired`). The winnower creates binary flags for each
  outcome indicating whether that token appears in the past or future period.
- `threshold` — defines how the threshold is set. Currently supported options are
  as follows:
    - `duration_s` (integer) thresholds after a given duration (in seconds)
    - `first_occurrence` (token string) thresholds after the first occurrence of
      the provided token
- `horizon_after_threshold_s` is an optional parameter that allows you to set a
  prediction window (in seconds) after the threshold is triggered

**Example configuration:**

```yaml
outcome_tokens: # supports patterns with fnmatch
    - XFR-IN//icu # ICU transfer
    - RESP//imv # invasive mechanical ventilation event
    - DSCG//expired # discharge due to death
    - LABEL//* # any kind of label token
threshold:
    # choose one and only one of the following
    duration_s: !!int 86400 # 24h
    # first_occurrence: XFR-IN//icu

horizon_after_threshold_s: !!int 2592000 # 30d outcome window after prediction threshold
```

### Outputs

- `held_out_for_inference.parquet` has columns for each outcome token (e.g.,
  `XFR-IN//icu_past`, `XFR-IN//icu_future`) indicating whether that outcome
  occurred in the respective time period.
- `train_for_inference.parquet` and `tuning_for_inference.parquet` are also
  provided; these are required to make rep-based predictions

## Usage

We provide a CLI that should be sufficient for most use cases:

```
 Usage: cocoa [OPTIONS] COMMAND [ARGS]...

 Configurable collation and tokenization (vXX.X.X)

╭─ Options ───────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.     │
│ --show-completion             Show completion for the current shell, to     │
│                               copy it or customize the installation.        │
│ --help                        Show this message and exit.                   │
╰─────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────╮
│ collate           Collate raw data into a denormalized format.              │
│ tokenize          Tokenize collated data into integer sequences.            │
│ winnow            Winnow held-out data for evaluation.                      │
│ pipeline          Run the full pipeline: collate, tokenize, & winnow.       │
│ combine-datasets  Combine multiple processed datasets into one.             │
╰─────────────────────────────────────────────────────────────────────────────╯
```

with commands:

- `cocoa collate`

    ```
    Usage: cocoa collate [OPTIONS]

    Collate raw data into a denormalized format.

    Reads collation configuration and produces a MEDS-like parquet file
    with collated events.

    ╭─ Options ───────────────────────────────────────────────────────────────────╮
    │    --collation-config     -c      PATH  Collation configuration file        │
    │                                         (overrides default)                 │
    │ *  --raw-data-home        -r      TEXT  Raw data directory [required]       │
    │ *  --processed-data-home  -p      TEXT  Processed data directory [required] │
    │    --verbose              -v            Verbose logging for collate; this   │
    │                                         may cause memory issues with large  │
    │                                         datasets                            │
    │    --help                 -h            Show this message and exit.         │
    ╰─────────────────────────────────────────────────────────────────────────────╯
    ```

- `cocoa tokenize`

    ```
    Usage: cocoa tokenize [OPTIONS]

    Tokenize collated data into integer sequences.

    Reads collated parquet files and produces tokenized timelines with
    vocabulary and bin information.

    ╭─ Options ───────────────────────────────────────────────────────────────────╮
    │    --tokenization-config  -c      PATH  Tokenization configuration file     │
    │                                         (overrides default)                 │
    │ *  --processed-data-home  -p      TEXT  Processed data directory [required] │
    │    --tokenizer-home       -t      TEXT  Load a previously learned tokenizer │
    │                                         from this tokenizer.yaml file       │
    │                                         (reuses its frozen vocabulary and   │
    │                                         bins)                               │
    │    --verbose              -v            Verbose logging for tokenize; this  │
    │                                         may cause memory issues with large  │
    │                                         datasets                            │
    │    --help                 -h            Show this message and exit.         │
    ╰─────────────────────────────────────────────────────────────────────────────╯
    ```

- `cocoa winnow`

    ```
    Usage: cocoa winnow [OPTIONS]

    Winnow held-out data for evaluation.

    Filters held-out timelines and assigns flags to disqualify certain subjects
    from evaluation based on the configured criteria.

    ╭─ Options ───────────────────────────────────────────────────────────────────╮
    │    --winnowing-config     -c      PATH  Winnowing configuration file        │
    │                                         (overrides default)                 │
    │ *  --processed-data-home  -p      TEXT  Processed data directory [required] │
    │    --verbose              -v            Verbose logging for winnow; prints  │
    │                                         summary statistics                  │
    │    --help                 -h            Show this message and exit.         │
    ╰─────────────────────────────────────────────────────────────────────────────╯
    ```

- `cocoa pipeline`

    ```
    Usage: cocoa pipeline [OPTIONS]

    Run the full pipeline: collate, tokenize, & winnow.

    ╭─ Options ───────────────────────────────────────────────────────────────────╮
    │    --collation-config             PATH  Collation configuration file        │
    │                                         (overrides default)                 │
    │    --tokenization-config          PATH  Tokenization configuration file     │
    │                                         (overrides default)                 │
    │    --winnowing-config             PATH  Winnowing configuration file        │
    │                                         (overrides default)                 │
    │ *  --raw-data-home        -r      TEXT  Raw data directory [required]       │
    │ *  --processed-data-home  -p      TEXT  Processed data directory [required] │
    │    --verbose              -v            Verbose logging for pipeline steps  │
    │    --help                 -h            Show this message and exit.         │
    ╰─────────────────────────────────────────────────────────────────────────────╯
    ```

<!-- prettier-ignore-start -->
> [!TIP]
> For common use cases, check out the [recipes](https://github.com/bbj-lab/cocoa/blob/master/recipes/README.md) section!
<!-- prettier-ignore-end -->

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
-->
