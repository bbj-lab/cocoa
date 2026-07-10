# Tokenizer

The `Tokenizer` is the second stage of the pipeline. It takes the collated
events and turns them into per-subject **integer sequences** ("timelines"),
learning a vocabulary and numeric-binning scheme along the way. Its behavior is
driven by a tokenization config (`tokenization.yaml`).

Crucially, both the vocabulary and the numeric bins are learned **only on the
training split**, so no information leaks from tuning or held-out subjects.

## What it produces

Running the tokenizer (via [`save_all`][cocoa.tokenizer.Tokenizer.save_all])
writes two files to the processed-data directory:

- `tokens_times.parquet` — the tokenized timelines (`tokens`, `times`, and
  optionally `numeric_values`) for each subject.
- `tokenizer.yaml` — the learned state (lookup table, bins, and config),
  sufficient to reconstruct the tokenizer.

## How it works

The orchestrator [`get_all`][cocoa.tokenizer.Tokenizer.get_all] runs these steps
in order:

1. [`add_ends`][cocoa.tokenizer.Tokenizer.add_ends] — insert `BOS` / `EOS`
   markers at the start and end of each subject's timeline.
2. [`add_clocks`][cocoa.tokenizer.Tokenizer.add_clocks] — optionally insert
   `CLCK//HH` tokens at configured hours of the day.
3. [`bin_data`][cocoa.tokenizer.Tokenizer.bin_data] — discretize numeric values
   into quantile bins (`Q0`, `Q1`, …) using cut points learned by
   [`get_bins`][cocoa.tokenizer.Tokenizer.get_bins].
4. [`insert_time_spacers`][cocoa.tokenizer.Tokenizer.insert_time_spacers] —
   optionally insert `TIME//…` tokens encoding the gap between consecutive
   events.
5. [`tokenize_data`][cocoa.tokenizer.Tokenizer.tokenize_data] — map the
   vocabulary to integers via the lookup table from
   [`get_lookup`][cocoa.tokenizer.Tokenizer.get_lookup], sorting simultaneous
   events by the configured `ordering` priority. `UNK` is always token `0`.

## Reusing a trained tokenizer

A tokenizer's learned state round-trips through
[`to_yaml`][cocoa.tokenizer.Tokenizer.to_yaml] and
[`from_yaml`][cocoa.tokenizer.Tokenizer.from_yaml], so a tokenizer trained on one
dataset can be **frozen** (`is_training=False`) and applied to another — see the
[Tokenizer Transfer](../recipes/tokenizer-transfer.md) recipe.

!!! tip "It behaves like a mapping"
    A `Tokenizer` is callable and dict-like: `tkzr("EOS")` returns a token id
    (`0` for out-of-vocabulary words), `"foo" in tkzr` tests vocabulary
    membership, and `len(tkzr)` reports the vocabulary size.

::: cocoa.tokenizer
