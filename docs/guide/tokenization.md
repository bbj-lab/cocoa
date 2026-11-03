# Tokenization

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

## Outputs

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
    │ 25563031   ┆ [88, 5, … 105]    ┆ [2110-04-11 20:08:00 UTC, 2110… │
    │ 20291550   ┆ [88, 1103, … 105] ┆ [2110-11-30 09:45:00 UTC, 2110… │
    │ …          ┆ …                 ┆ …                               │
    │ 21477991   ┆ [88, 1105, … 105] ┆ [2201-12-11 08:04:13 UTC, 2201… │
    └────────────┴───────────────────┴─────────────────────────────────┘
    ```

    In this example, token 88 corresponds to the beginning-of-sequence token
    (`BOS`), token 105 to the end-of-sequence token (`EOS`), and the tokens in
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
        - 40.0
        - 76.0
        - 108.0
        - 135.0
        - 158.0
        - 189.0
        - 229.0
        - 282.0
        - 348.0
      VTL//heart_rate:
        - 67.0
        - 75.0
        - 80.0
        - 84.0
        - 89.0
        - 94.0
        - 100.0
        - 105.0
        - 113.0
      …
    ```

    The lists following each key correspond to the cutpoints for the associated
    category.

<!-- prettier-ignore-start -->
> [!TIP]
> To train a generative event model on this data, check out our configurable
> trainer: [🦜 cotorra](https://pypi.org/project/cotorra/)
<!-- prettier-ignore-end -->
