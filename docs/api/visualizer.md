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
- `{split}_for_inference.parquet` (if present, and only when `show_winnowing` is
  on) — where the [Winnower](winnower.md) split the timeline into past and
  future, and its outcome flags.

It reads a processed directory from any release since 26.6.2; 26.6.0 and 26.6.1
fail before writing `tokenizer.yaml`. The page leaves out whatever an older
release didn't write. Before 26.9.1, `subject_splits.parquet` has no start or
end times, and `tokenizer.yaml` has no training counts, so the prefixes with the
most vocabulary words take the first palette colors. Before 26.9.0, times were
stored without a time zone, meaning UTC, and the page shows them in UTC.

## What it produces

[`render`][cocoa.visualizer.Visualizer.render] returns the page as one html
string, with its styles, script, fonts, and data inlined, so it works offline and
loads nothing from the network. [`save`][cocoa.visualizer.Visualizer.save] writes
it to a file. From the command line, `cocoa visualize` serves the page on
localhost, or saves it with `--export-html`.

A saved page is all anyone needs to see what you see: send the one file, and they
can open it in any current web browser, offline, without cocoa or the processed
data. Times show in the data's time zone, not the viewer's. Email and file
previews often show html without running its script, so until the timeline is
drawn the page shows a note asking to open the file in a web browser instead.

[`render_pdf`][cocoa.visualizer.Visualizer.render_pdf] draws the same timeline as
a static pdf, and [`save_pdf`][cocoa.visualizer.Visualizer.save_pdf] writes it to
a file (`cocoa visualize --export-pdf`). A pdf opens anywhere, in a phone's
message preview as readily as in a pdf reader, which makes it the file to send
when the recipient may not open html in a browser; send both, and they can open
the html to explore. Its first page shows the subject and every lane of the whole
timeline; the pages after give every code a row of its own, with binned values at
the height of their quantile, and then list the events, as many as the page's
table shows at once. It is set in the page's font, Gotham, which it embeds, with
codes and tokens in Courier, one of pdf's standard fonts; a character the font lacks
prints as `?`.

## Configuring the page

Every section of `visualization.yaml` is optional, and an empty one means none:

- `show_winnowing` — whether to show where the winnower split the timeline: the
  threshold and the shaded future in the plot, the future tokens, and a summary
  near the bottom of the page of which outcomes it flagged in the past and the
  future. Off by default, so the page shows none of it;
  `cocoa visualize --show-winnowing` turns it on without a config of your own.
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
