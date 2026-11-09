# Winnowing

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
      [reference table](collation.md#reference-table); see
      [below](#duration_s-vs-after_start_time_s) for how it differs from
      `duration_s`
    - `first_occurrence` (token string) thresholds after the first occurrence of
      the provided token

    Set only one of them: given several, the winnower uses the first in this
    list.

- `horizon_after_threshold_s` is an optional parameter that sets a prediction
  window (in seconds): the future keeps only the events within that long of the
  last event of the past, which under `first_occurrence` is the triggering token.
  Under `duration_s` and `after_start_time_s`, that event can come well before
  the cut, and the window closes as much earlier (see
  [within a horizon](../recipes/several-decision-times.md#within-a-horizon)).
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

## `duration_s` vs. `after_start_time_s`

Both thresholds cut a timeline a set number of seconds after some time, but not
after the same time:

- `duration_s` counts from the timeline's **first event**, the time of its `BOS`
  token, so every past spans the same stretch of record. Where the cut falls in a
  stay depends on what was recorded first: a timeline that opens before its
  admission is cut that much earlier in the stay, and one that opens after it,
  that much later.
- `after_start_time_s` counts from the subject's **start time**: the reference
  table's `start_time` column (`admission_dttm` in the shipped collation config),
  as `subject_splits.parquet` records it. Every stay is cut at the same point,
  such as 24 hours after admission, whatever the timeline holds, and events
  recorded before the start time land in the past.

The two cut a timeline at the same point only if it opens at its start time. With
the shipped collation config, which times its demographic `REFERENCE` entries at
`admission_dttm`, no timeline opens after its admission, but many open before it:
on the demo, 236 of the 310 timelines do, by as much as a day, 213 of them with
an arrival in the emergency department (`XFR-IN//ed`). Held-out hospitalization
29079034 arrives there 20 hours before its admission, so `duration_s: 86400` cuts
it less than 4 hours into its stay, with 57 tokens in its past, where
`after_start_time_s: 86400` cuts it a day in, with 197.

Under either, an event at exactly the cut falls in the future, and a timeline is
kept only if it runs past the cut. Under `after_start_time_s`, it must also begin
before the cut, since its past would otherwise be empty; under `duration_s`, the
past always holds `BOS`.

## Outputs

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

On the demo, the first keeps 57 of the 63 held-out hospitalizations, those whose
timelines run past 24 hours; the second 55 of those, the ones that also run past
24 hours after admission, cutting 25 of them later than the first; and the third
26, those with a transfer to the ICU. To winnow at several times into each stay,
for a model that predicts again and again as a stay goes on, see
[inference at several decision times](../recipes/several-decision-times.md).
