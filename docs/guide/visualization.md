# Visualization

The visualizer draws one subject's tokenized timeline as an interactive html
page, so you can see what collation and tokenization made of it, and draws the
learned vocabulary as another (see [The vocabulary page](#the-vocabulary-page)).
Unlike the three stages, it writes nothing to the processed data directory: it
only reads what they wrote there. For a timeline, it:

1. Loads the subject's tokens and times from `tokens_times.parquet`, and decodes
   each token with `tokenizer.yaml` into its code, quantile bin, and text value.
2. Looks up the subject's split, start and end times, and pass-through columns in
   `subject_splits.parquet`.
3. Finds the numeric value behind each binned token: in the `numeric_values`
   column if the tokenizer wrote one, or else in `meds.parquet`, binned again
   with the tokenizer's cutpoints to match each value to its token.
4. Groups codes into lanes by their prefix, one row of the timeline per lane.
5. Optionally marks where the winnower split the timeline into past and future,
   and which outcome tokens it flagged, from `{split}_for_inference.parquet`.
6. Serves the page on localhost, or saves it as a self-contained html file, a
   static pdf, or both.

Visualization is driven by a YAML config (the package ships a default; see
[`./src/cocoa/config/visualization.yaml`](https://github.com/bbj-lab/cocoa/blob/master/src/cocoa/config/visualization.yaml))
that decides how the page looks, not what data it shows. Every section is
optional, and an empty one means none. It specifies:

- `show_winnowing` — whether to show the winnower's split (`false` by default;
  `cocoa visualize --show-winnowing` turns it on without a config of your own).
  The page then marks the threshold, shades the future, and summarizes which
  outcomes were flagged on each side.
- `lanes` — the rows of the timeline, top to bottom, each grouping one or more
  code prefixes under a `name` and a `color`. A lane is drawn only for a subject
  with codes of one of its prefixes.
- `palette` — colors for prefixes that no lane claims. Each such prefix gets a
  lane of its own, below the configured ones, and the most frequent in training
  take the first colors not already worn by a lane, so a prefix keeps its color
  across a dataset's subjects.
- `other_color` — the color of prefixes past the end of the palette, and of
  `UNK`.
- `prefixes` — what each prefix denotes, shown on hover alongside the code; it
  also names the lane of a prefix that no lane claims.
- `descriptions` — fuller descriptions of codes, shown on hover, keyed by the
  code without its bin or text value. Keys may be `fnmatch` patterns; an exact
  key wins over a pattern. The shipped config has none.

**Example configuration:**

```yaml
show_winnowing: !!bool false

lanes: # top to bottom
    - name: Labs
      prefixes: [LAB-ORD, LAB-RES]
      color: "#007396" # lake
    - name: Vitals
      prefixes: [VTL]
      color: "#A4343A" # brick
    …

palette: # for prefixes no lane claims, most frequent first
    - "#59315F" # violet
    - "#007396" # lake
    …
other_color: "#737373" # dark greystone

prefixes: # shown on hover
    LAB-RES: Lab result
    VTL: Vital sign
    …

descriptions: # none ship; keys may be fnmatch patterns
    VTL//spo2: Oxygen saturation, by pulse oximetry
    ASMT//gcs_*: Glasgow Coma Scale
```

The shipped lanes cover the prefixes of the shipped collation config. When a
vocabulary has prefixes that no lane claims, `cocoa visualize` lists them; to
name and color their lanes, list them under `lanes` in a config passed with `-c`.
A `-c` config replaces the shipped default rather than merging into it, so start
from a copy of the default.

## The page

The page opens with the subject: its id and split, its start time, end time, and
span, how many tokens, events, and distinct codes its timeline holds, and its
pass-through columns. Below, the timeline draws each event as a mark in its lane,
against an axis in the data's time zone (the collation config's
`default_timezone`), whatever the viewer's zone:

<img src="https://raw.githubusercontent.com/burkh4rt/cocoa/master/img/vis-screenshot.png" alt="timeline of held-out hospitalization 27708593, with its first tokens listed" width="700"/>

To explore it:

- Drag to pan; pinch, ⌘/Ctrl-scroll, or double-click to zoom; and press `0` (or
  Reset) to see the whole timeline again. The strip above the axis shows the
  whole timeline, and dragging across it picks the window in view.
- Click a lane's name to open it, giving each of its codes a row of its own. A
  binned code's marks sit at the height of their quantile, from `Q0` at the
  bottom to the top bin at the top. "Open all lanes" opens every lane at once.
- Hover over a mark to see its time, how long after the start time it came, and,
  for each event in that row at that instant, its code, its value or bin (with
  the bin's range in training), its description, and its tokens.
- Type in the search box (or press `/`) to highlight the codes whose code, prefix
  name, lane, or description contains the text, and dim the rest.
- Click a mark, token, or table row, then step through the events with the arrow
  keys; while searching, only through the highlighted ones.

<img src="https://raw.githubusercontent.com/burkh4rt/cocoa/master/img/vis-screenshot-2.png" alt="timeline of held-out hospitalization 27708593, with the vitals lane opened" width="700"/>

Below the timeline, two tabs list what's in view: Tokens, the tokens grouped by
time, and Table, one row per event with its lane, code, bin, value, text value,
and tokens. With `show_winnowing`, a summary of the winnower's split follows, and
a footer records the cocoa version, the tokenizer's settings, and the processed
data directory the page was drawn from.

## Outputs

`cocoa visualize` takes a subject id and a processed data directory. By default,
it serves the page at `http://127.0.0.1:8765/` (or on a free port, if that one is
taken) until you press Ctrl-C, and `--open` opens it in a web browser:

```sh
cocoa visualize 27708593 -p processed/demo --open
```

To keep the page rather than serve it, save it to a file, or to two:

```sh
cocoa visualize 27708593 -p processed/demo \
    -e timeline.html --export-pdf timeline.pdf
```

- `--export-html` (`-e`) writes the page as one self-contained html file. Its
  styles, script, fonts, and data are inlined, so it loads nothing from the
  network: anyone can open it in a current web browser, offline, without cocoa or
  the processed data. Email and file previews often show html without running its
  script, so until the timeline is drawn, the page asks to be opened in a browser
  instead.
- `--export-pdf` writes a static pdf of the same timeline, which opens anywhere,
  even in a phone's message preview. Its first page shows the subject and every
  lane of the whole timeline; the pages after give every code a row of its own,
  with binned values at the height of their quantile, and then list the events.

Either file holds the subject's timeline, so share one made from real patient
data only as you would the data itself. The visualizer reads a processed data
directory written by any release since 26.6.2; the page leaves out whatever an
older release didn't write.

## The vocabulary page

Given no subject id, `cocoa visualize` draws what tokenization learned, rather
than one subject: the vocabulary, with its bins over the training data. It takes
the same config as a timeline, so a prefix wears the color its lane wears in a
timeline. As for a timeline, it serves the page, or saves it as one
self-contained html file with `--export-html` (`-e`), as a static pdf with
`--export-pdf`, or both:

```sh
cocoa visualize -p processed/demo --open
cocoa visualize -p processed/demo -e vocabulary.html --export-pdf vocabulary.pdf
```

`cocoa visualize` reads its first argument as a subject id whatever it says, so
to change the vocabulary page's config, pass a config of your own with `-c`; a
config override given without a subject id would be read as one.

Below the vocabulary's size, the page has five tabs:

- Binned codes: a histogram of each binned code's values in the training split,
  the values the tokenizer learned its bins on, with the cut points drawn across
  it and every other bin shaded. A code with at most 50 distinct values gets a
  bar per value; one with more gets about 50 buckets of a round width, over all
  but its most extreme values, which it counts below or above the range drawn
  instead. Each says how many of its bins hold values: where cut points tie, as
  they do for a score with only a few values, the bins between them stay empty,
  and ordering by the fewest bins puts such codes first. Click one to see its
  values' range, mean, and median, and a row per bin with its range, how many
  training values fall in it, and the word they tokenize to.
- Words: every word, with its token, lane, bin, text value, and training count,
  and how many words training saw only once or fewer than 10 times, which can
  help in choosing `min_training_ct`. Click a word of a binned code to see its
  bins.
- Prefixes: each prefix, with its lane, its words, codes, and binned codes, and
  its share of the training tokens.
- Timelines: what `cocoa tokenize --verbose` reports, for each split and for all
  of them: how many timelines and tokens, how many of the tokens are `UNK`, the
  timelines' lengths and durations, and the first and last events, with
  histograms of the lengths and durations.
- Tokenizer config: the config saved in `tokenizer.yaml`.

The search box finds the codes, words, and prefixes whose code, name, lane, or
description contains its text. The page reads `tokenizer.yaml` and
`tokens_times.parquet`, and `meds.parquet` and `subject_splits.parquet` for the
training values. Without those, it draws each bin's training count from the
counts of its words, where a fused vocabulary has them, or else just the cut
points. A tokenizer applied with `--tokenizer-home` keeps the counts of the data
it was learned on, so the page marks its vocabulary as transferred, and draws
this directory's training split against its bins.

The pdf lays out what the tabs show, one after another. Its first page gives the
vocabulary's size, the prefixes, and the timelines by split, with histograms of
the lengths and durations of each split's timelines, where the page draws one
split at a time. Then each binned code gets its histogram beside a row per bin,
as the page shows a code opened, followed by the words and the tokenizer's
config. It draws the first 500 binned codes and lists the first 2,000 words; the
html page has all of them.

Neither file holds anyone's timeline, but both summarize the training data, down
to each code's least and greatest value and each split's first and last event, so
share them as you would such statistics.
