# Winnower

The `Winnower` is the final stage of the pipeline. It prepares timelines for
evaluation: it splits each timeline at a configurable cut-point into a **past**
portion (the context a model is given) and a **future** portion (what the model
must predict), and it attaches outcome labels. Its behavior is driven by a
winnowing config (`winnowing.yaml`).

The name is apt — the winnower also **filters out** subjects that cannot be
fairly evaluated: those whose timelines never reach the cut-point.

## What it produces

Running the winnower (via [`save_all`][cocoa.winnower.Winnower.save_all]) writes
one file per configured split to the processed-data directory:

- `{split}_for_inference.parquet` — the winnowed timelines with past/future
  splits and outcome flags (defaults to `held_out`).

To write them somewhere else, pass `output_home` (on the command line,
`--output-home` / `-o`); the directory is created if need be, and the
processed-data directory is then only read.
[`copy_processed_data`][cocoa.winnower.Winnower.copy_processed_data] copies the
processed-data directory's files there too, so that it is a complete processed
dataset of its own. It skips subdirectories, and any `*_for_inference.parquet`
already in the processed-data directory, which may have been winnowed under
another configuration. This lets one tokenized dataset be winnowed under several
configurations side by side, as here on the
[MIMIC-IV demo](../recipes/mimic-iv-demo.md):

```sh
cocoa winnow -p processed/demo -o processed/demo/winnowed-24h
cocoa winnow -p processed/demo -o processed/demo/winnowed-admit-24h \
    '~threshold.duration_s' threshold.after_start_time_s=86400
cocoa winnow -p processed/demo -o processed/demo/winnowed-icu \
    '~threshold.duration_s' threshold.first_occurrence=XFR-IN//icu
```

## How it works

[`prepare_winnowed_frame`][cocoa.winnower.Winnower.prepare_winnowed_frame] chains
these steps:

1. [`load_frame`][cocoa.winnower.Winnower.load_frame] — load the tokenized
   timelines for the requested split and compute elapsed times.
2. [`run_thresholding`][cocoa.winnower.Winnower.run_thresholding] — compute
   `last_valid`, the cut-point between past and future. This is derived from a
   duration after the timeline's first event (`threshold.duration_s`), a duration
   after the subject's start time (`threshold.after_start_time_s`), or the first
   occurrence of a token of interest (`threshold.first_occurrence`); given
   several, the first in that order wins. Timelines that never reach the
   cut-point are dropped, and under `after_start_time_s`, so are those that begin
   at or after it, which would have no past.
3. [`add_outcome_flags`][cocoa.winnower.Winnower.add_outcome_flags] — for each
   configured outcome token, add `*_past` and `*_future` boolean labels (for
   example `DSCG//expired_past` and `DSCG//expired_future`), so you can tell
   whether an outcome truly falls within the prediction window.

<!-- prettier-ignore-start -->
!!! note "`duration_s` vs. `after_start_time_s`"

    `duration_s` counts from each timeline's first event, the time of its `BOS`
    token, while `after_start_time_s` counts from the subject's `start_time` in
    `subject_splits.parquet`, the reference table's start time (`admission_dttm`
    in the shipped collation config). They cut a timeline at the same point only
    if it opens at its start time. One that opens earlier, as most of the demo's
    do, with an arrival in the emergency department, is cut that much earlier in
    its stay by `duration_s`; `after_start_time_s` cuts every stay at the same
    point, such as 24 hours after admission. See
    [`duration_s` vs. `after_start_time_s`](../index.md#duration_s-vs-after_start_time_s).
<!-- prettier-ignore-end -->

<!-- prettier-ignore-start -->
!!! note "Depends on tokenizer artifacts"

    The winnower reads the `tokenizer.yaml` written by the
    [Tokenizer](tokenizer.md) to resolve outcome-token patterns against the
    learned vocabulary, so tokenization must run first.
<!-- prettier-ignore-end -->

::: cocoa.winnower
