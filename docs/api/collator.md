# Collator

The `Collator` is the first stage of the pipeline. It collects the many raw
tables that make up a dataset (parquet or csv) and **collates** them into a
single, denormalized event stream in a MEDS-like format: one row per event,
with a uniform schema of `subject_id`, `time`, `code`, `numeric_value`, and
`text_value`.

Which tables to read, how to filter and aggregate them, and how each event maps
onto a `code` are all driven by a collation config (`collation.yaml`). Timestamps
are normalized to timezone-naive UTC along the way.

## What it produces

Running the collator (via [`save_all`][cocoa.collator.Collator.save_all]) writes
two files to the processed-data directory:

- `meds.parquet` — the collated events for every subject.
- `subject_splits.parquet` — a `train` / `tuning` / `held_out` assignment for
  each subject, partitioned by first event time and the fractions in the config.

## How it works

1. [`get_reference_frame`][cocoa.collator.Collator.get_reference_frame] builds a
   per-subject "spine" (for example, a hospital stay with a start and end time),
   optionally joining in augmentation tables.
2. [`get_entry`][cocoa.collator.Collator.get_entry] turns one configured event
   type into standardized token rows — building the `code` string, casting
   values, and optionally restricting events to the reference window.
3. [`get_all`][cocoa.collator.Collator.get_all] concatenates every configured
   entry into the full event stream.
4. [`get_subject_splits`][cocoa.collator.Collator.get_subject_splits] assigns the
   data splits.

!!! warning "Config files are trusted input"
    Collation configs may contain Polars expression *strings* (for filters,
    added columns, and aggregations) that are evaluated at load time. These run
    through [`slightly_safer_eval`][cocoa.collator.Collator.slightly_safer_eval],
    which restricts the available namespace but is **not** secure against
    malicious input. Only run configs you trust.

::: cocoa.collator
