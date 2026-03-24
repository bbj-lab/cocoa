# Cocoa: a configurable collator

> This repo provides a configurable way to collate data from multiple sources
> into a single denormalized dataframe and create tokenized timelines from the
> results.

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

## Installation

You can use [uv](https://docs.astral.sh/uv/pip/) to create an environment for
running this code (with Python >= 3.12) as follows:

```sh
uv sync
uv run cocoa --help
```

## Overview

Cocoa does two things: **(1) collation** and **(2) tokenization**.

### (1) Collation

The collator pulls from raw data tables (parquet or CSV) and combines them into a
single denormalized dataframe in a
[MEDS](https://github.com/Medical-Event-Data-Standard/meds)-like format. Each row
in the output represents an event with a `subject_id`, `time`, `code`, and
optional `numeric_value` / `text_value` columns.

Collation is driven by a YAML config that specifies:

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
  Check yaml files prior to use._
- **Subject splits** (`train_frac` / `tuning_frac`) that partition subjects
  chronologically into train, tuning, and held-out sets.

### (2) Tokenization

The tokenizer consumes the collated parquet output and converts events into
integer token sequences suitable for sequence models. It:

1. Adds `BOS` / `EOS` sentinel tokens to each subject's timeline.
2. Computes quantile-based bins for numeric values (from training data only).
3. Maps codes (and optionally their binned values) to integer tokens via a
   vocabulary that grows during training and is frozen for tuning/held-out data.
4. Aggregates per-subject token sequences in a configurable sort order.

Tokenization is driven by its own YAML config that specifies:

- `n_bins` — number of quantile bins for numeric values.
- `fused` — whether to fuse the code, binned value, and text value into a single
  token (`true`) or keep them as separate tokens (`false`).
- `collated_inputs` — paths to the collated parquet files to tokenize.
- `ordering` — the priority order of code prefixes when sorting events within the
  same timestamp.

The tokenizer produces two main outputs:

- **`tokens_times.parquet`** — one row per subject with three columns:
  - `subject_id`
  - `tokens` — the integer token sequence for the subject's timeline.
  - `times` — a parallel list of timestamps, one per token, indicating when each
    event occurred.
- **`tokenizer.yaml`** — a plain yaml file that contains information about the
  configuration, learned vocabulary, and bins

For example, a subject with two events might look like:

| subject_id | tokens           | times                                                        |
| ---------- | ---------------- | ------------------------------------------------------------ |
| "100"      | [1, 5, 8, 12, 2] | [2025-01-01, 2025-01-01, 2025-01-02, 2025-01-03, 2025-01-03] |

Here 1 is the BOS token, 2 is EOS, and the tokens in between correspond to the
subject's clinical events in chronological order (with ties broken by the
configured `ordering`). In fused mode each event is a single token; in unfused
mode an event with a numeric value becomes two tokens (code + quantile bin).

## Configuration

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
`config/main.yaml`.

Both the `Collator` and `Tokenizer` classes also accept `**kwargs` that are
merged on top of the YAML config via OmegaConf, so any config value can be
overridden programmatically:

```python
from cocoa.collator import Collator
from cocoa.tokenizer import Tokenizer

collator = Collator(data_home="~/other/data")
tokenizer = Tokenizer(n_bins=20, fused=False)
```

### Configuring the collator

A collation config has three top-level sections: identifiers, subject splits, and
the reference + entries that define which events to extract.

#### Identifiers and splits

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

#### Reference table

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

- `table` — the name of the parquet (or CSV) file in `data_home` (without the
  extension).
- `start_time` / `end_time` — columns that define the subject's time window; used
  to filter events from other tables when `reference_key` is set (see below).
- `augmentation_tables` — optional list of tables to join onto the reference
  frame. Each needs a `key` to join on and a `validation` mode (e.g. `"m:1"`).
  You can also add computed columns via `with_col_expr`.

#### Entries

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

## Usage

Cocoa provides a CLI with the following commands:

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
 --exclude "slurm/output/" \
 --exclude ".venv/" \
 --exclude ".idea/" \
 ~/Documents/chicago/cocoa \
 randi:/gpfs/data/bbj-lab/users/burkh4rt
```

-->
