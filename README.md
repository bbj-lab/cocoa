# Cocoa: a configurable collator

> Chicago's second favorite bean

<img src="img/cocoa-bean.png" alt="cocoa bean" width="400" style="display: block;
margin: 0 auto; -webkit-mask-image: radial-gradient(
    ellipse at center,
    rgba(0,0,0,1) 50%,
    rgba(0,0,0,0) 100%
  );
  mask-image: radial-gradient(
    ellipse at center,
    rgba(0,0,0,1) 50%,
    rgba(0,0,0,0) 100%
  );"/>

## About

This repo provides a configurable way to collate data from multiple sources into
a single denormalized dataframe and create tokenized timelines from the results.

## Installation

You can use [uv](https://docs.astral.sh/uv/pip/) to create an environment for
running this code (with Python >= 3.12) as follows:

```sh
uv sync
uv run cocoa --help
```

## (1) Collation

The collator pulls from raw data tables (parquet or csv) and combines them into a
single denormalized dataframe in a
[MEDS](https://github.com/Medical-Event-Data-Standard/meds)-like format. Each row
in the output represents an event with a `subject_id`, `time`, `code` (all
mandatory), and optional `numeric_value` / `text_value` columns.

Collation is driven by a YAML config (like
[this](./config/collation/clif-21.yaml)) that specifies:

- A **reference table** with a primary key (`subject_id`), start/end times, and
  optional augmentation joins (e.g. joining a patient demographics table).
- A list of **entries**, each mapping a source table (or the reference frame
  itself via `table: REFERENCE`) to the output schema. Each entry declares which
  column provides the `code`, `time`, and optionally `numeric_value`, and
  `text_value`. Codes can be given a prefix `prefix`. Some preprocessing can be
  done with optional entries for `filter_expr`, `with_col_expr`, and `agg_expr`.
  These take the form of polars expressions that are evaluated and applied to the
  dataframe during loading. _Mild checks are performed when evaluating these
  expressions, but in general, the yaml config is just as powerful as the python.
  Check all yaml files prior to use._
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

The reference table is the primary static table to which everything else is
joined:

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

- `table` — the name of the parquet (or csv) file in `data_home` (without the
  extension).
- `start_time` / `end_time` — columns that define the subject's time window; used
  to filter events from other tables when `reference_key` is set (see below).
- `augmentation_tables` — optional list of tables to join onto the reference
  frame. Each needs a `key` to join on and a `validation` mode (e.g. `"m:1"`).
  You can also add computed columns via `with_col_expr`.

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
| `reference_key` | Join the source table to the reference frame on this key and keep only rows within the subject's `start_time`–`end_time` window. |

**Examples:**

A simple categorical event from the reference frame:

```yaml
- table: REFERENCE
  prefix: DSCG
  code: discharge_category
  time: discharge_dttm
```

A numeric event from an external table:

```yaml
- table: clif_labs
  prefix: LAB-RES
  code: lab_category
  numeric_value: lab_value_numeric
  time: lab_result_dttm
```

Filtering rows before extraction (single filter):

```yaml
- table: clif_position
  prefix: POSN
  filter_expr: pl.col("position_category") == "prone"
  code: position_category
  time: recorded_dttm
```

Multiple filters (applied as a list):

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

Creating a computed column with `with_col_expr` to use as the code:

```yaml
- table: clif_respiratory_support_processed
  prefix: RESP
  with_col_expr: pl.lit("fio2_set").alias("code")
  filter_expr: pl.col("fio2_set").is_finite()
  code: code
  numeric_value: fio2_set
  time: recorded_dttm
```

Using `reference_key` to restrict events to a subject's time window:

```yaml
- table: clif_code_status
  prefix: CODE
  code: code_status_category
  time: admission_dttm
  reference_key: patient_id
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

Tokenization is driven by its own YAML config (like
[this](./config/collation/clif-21.yaml)) that specifies:

- `n_bins` — number of quantile bins for numeric values.
- `fused` — whether to fuse the code, binned value, and text value into a single
  token (`true`) or keep them as separate tokens (`false`).
- `insert_spacers` — whether to insert time spacing tokens between events.
- `insert_clocks` — whether to insert clock tokens at specified times.
- `collated_inputs` — paths to the collated parquet files to tokenize.
- `subject_splits` — path to the subject splits parquet file.
- `ordering` — the priority order of code prefixes when sorting events within the
  same timestamp.
- `spacers` — mapping of time intervals (e.g., `5m-15m`, `1h-2h`) to their lower
  bounds in minutes, used for time spacing tokens.
- `clocks` — list of hour strings (e.g., `00`, `04`, ...) at which to insert
  clock tokens.

## Outputs

The tokenizer produces two main outputs:

### Tokenized timelines

`tokens_times.parquet` gives one row per subject with three columns:

- `subject_id`
- `tokens` — the integer token sequence for the subject's timeline.
- `times` — a parallel list of timestamps, one per token, indicating when each
  event occurred.

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

In this example, token 20 corresponds to the beginning-of-sequence token (`BOS`),
token 21 to the end-of-sequence token (`EOS`), and the tokens in between
correspond to the subject's clinical events in chronological order (with ties
broken by the configured `ordering`). In fused mode each event is a single token;
in unfused mode an event with a numeric value becomes two tokens (code + quantile
bin).

### A record of the tokenizer object

`tokenizer.yaml` is a plain yaml file that contains information about the
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
  ...
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
  ...
```

The lists following each key correspond to the cutpoints for the associated
category.

Both of these things are placed in `processed_data_home` as configured.

## Usage

All configuration lives under `config/`. The entrypoint is `config/main.yaml`,
which points to the collation and tokenization configs and sets shared paths:

```yaml
data_home: ~/path/to/raw/data
processed_data_home: ~/path/to/output

collation_config: ./config/collation/clif-21.yaml
tokenization_config: ./config/tokenization/clif-21.yaml
```

To use a different dataset or schema, create new YAML files under
`config/collation/` and `config/tokenization/` and update the paths in
`config/main.yaml`, or pass your options directly to these objects. Both the
`Collator` and `Tokenizer` classes also accept `**kwargs` that are merged on top
of the YAML config via OmegaConf, so any config value can be overridden
programmatically:

```python
from cocoa.collator import Collator
from cocoa.tokenizer import Tokenizer

collator = Collator(data_home="~/other/data")
tokenizer = Tokenizer(n_bins=20, fused=False)
```

We provide a CLI with the following commands:

```sh
# collate raw data into a denormalized parquet file
cocoa collate [-o OUTPUT_DIR]

# tokenize collated data into integer sequences
cocoa tokenize [-o OUTPUT_DIR]

# run both steps in sequence
cocoa pipeline [-o OUTPUT_DIR]

# display current configuration
cocoa info
```

<!--


Send to randi:
```
rsync -avht \
 --delete \
 --exclude "processed/" \
 --exclude ".venv/" \
 --exclude ".idea/" \
 ~/Documents/chicago/cocoa \
 randi:/gpfs/data/bbj-lab/users/burkh4rt
```

Send to bbj-lab1:
```
rsync -avht \
 --delete \
 --exclude "processed/" \
 --exclude ".venv/" \
 --exclude ".idea/" \
 ~/Documents/chicago/cocoa \
 bbj-lab1:~
```

-->
