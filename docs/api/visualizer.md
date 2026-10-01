# Visualizer

The `Visualizer` renders one subject's tokenized timeline as a self-contained,
interactive html page, so you can see what tokenization made of a timeline. It is
not a pipeline stage: it only reads what the stages wrote to the processed-data
directory. Its behavior is driven by a visualization config
(`visualization.yaml`), which controls how the page looks, not what data it
shows.

## What it reads

From the processed-data directory:

- `tokenizer.yaml` — the vocabulary, bins, and tokenization config, used to
  decode each token back into a code, a bin, and a text value.
- `tokens_times.parquet` — the subject's tokens and times, plus `numeric_values`
  and `hours_to_end_time` when the tokenizer wrote them.
- `subject_splits.parquet` (if present) — the subject's split, start and end
  times, and any pass-through columns, shown as subject details.
- `meds.parquet` (if present) — numeric values, matched back to their tokens when
  the timeline was tokenized without `include_numeric_values`.
- `{split}_for_inference.parquet` (if present) — where the
  [Winnower](winnower.md) split the timeline into past and future, and its
  outcome flags.

## What it produces

[`render`][cocoa.visualizer.Visualizer.render] returns the page as one html
string, with its styles, script, fonts, and data inlined, so it works offline and
loads nothing from the network. [`save`][cocoa.visualizer.Visualizer.save] writes
it to a file. From the command line, `cocoa visualize` serves the page on
localhost, or saves it with `--export-html`.

## Configuring the page

Every section of `visualization.yaml` is optional, and an empty one means none:

- `lanes` — the rows of the timeline, top to bottom, each grouping one or more
  code prefixes under a name and a color.
- `palette` — colors for prefixes that no lane claims. Each such prefix gets a
  lane of its own, and the most frequent in training take the first colors, so a
  prefix keeps its color across a dataset's subjects.
- `other_color` — the color of prefixes past the end of the palette, and of
  `UNK`.
- `prefixes` — what each prefix denotes, shown on hover.
- `descriptions` — fuller descriptions of codes, keyed by the code without its
  bin or text value. Keys may be `fnmatch` patterns; an exact key wins over a
  pattern.

<!-- prettier-ignore-start -->
!!! note "Naming new prefixes"

    `cocoa visualize` lists the vocabulary prefixes that no lane claims. To name
    and color them, list them under `lanes` in a config passed with `-c`. A `-c`
    config replaces the packaged default rather than merging into it, so start
    from a copy of the default.
<!-- prettier-ignore-end -->

::: cocoa.visualizer
