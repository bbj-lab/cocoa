# Combining datasets

`cocoa combine-datasets` merges processed directories into one, such as the
timelines of two hospitals, or of the two periods that the
[tokenizer transfer recipe](../recipes/tokenizer-transfer.md) splits the demo
into:

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
