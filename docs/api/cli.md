# CLI

Cocoa ships a command-line interface, `cocoa`, that drives every stage of the
pipeline. Each stage has its own command. There are also two convenience
commands, for running everything at once and for merging datasets, a viewer for
inspecting one subject's tokenized timeline, or the learned vocabulary, and a
command that writes the MIMIC-IV demo's tables to start from.

## Commands

| Command                  | What it does                                                 |
| ------------------------ | ------------------------------------------------------------ |
| `cocoa collate`          | Collate raw tables into a denormalized event stream.         |
| `cocoa tokenize`         | Tokenize collated data into integer timelines.               |
| `cocoa winnow`           | Prepare timelines for evaluation.                            |
| `cocoa pipeline`         | Run `collate`, `tokenize`, and `winnow` end-to-end.          |
| `cocoa combine-datasets` | Merge multiple processed datasets into one.                  |
| `cocoa visualize`        | Show a subject's timeline, or the vocabulary, as a web page. |
| `cocoa get-demo`         | Write the MIMIC-IV demo's CLIF tables, from clifpy.          |

Every stage command accepts `--processed-data-home` / `-p` (the working directory
for intermediate and output files) and `--verbose` / `-v` (extra logging and
summary statistics). Each single-stage command also takes an optional `-c` config
file that overrides the packaged default for that stage; `cocoa pipeline` takes
one per stage instead, as `--collation-config`, `--tokenization-config`, and
`--winnowing-config`. `cocoa visualize` takes `-p` and `-c` too, but has no
`--verbose`. `cocoa get-demo` takes only `--raw-data-home` / `-r`, the directory
to write the tables to, and `--verbose` / `-v`, which shows what clifpy prints;
it needs clifpy, which cocoa's `demo` extra installs
(`pip install 'cocoa-tokenizer[demo]'`).

To change individual config keys for one run, list them after the command's
options, as in `cocoa tokenize -p processed/demo n_bins=5`; for `cocoa pipeline`,
start each key with its stage, as in `tokenization.n_bins=5`. See
[Overriding config keys](../guide/configuration.md#overriding-config-keys) for
the syntax.

Run any command with `-h` / `--help` to see its full set of options:

```sh
cocoa --help
cocoa tokenize --help
```

## Typical usage

The examples here use the paths of the
[MIMIC-IV demo](../recipes/mimic-iv-demo.md), run from its working directory.
Write the demo's CLIF tables, from clifpy:

```sh
cocoa get-demo --raw-data-home data-raw/demo
```

Then run the whole pipeline in one go:

```sh
cocoa pipeline \
    --raw-data-home data-raw/demo \
    --processed-data-home processed/demo \
    --verbose
```

Or drive the stages individually — for example, to reuse a previously learned
tokenizer via `--tokenizer-home` (see the
[Tokenizer Transfer](../recipes/tokenizer-transfer.md) recipe, which splits the
demo into the CLIF tables of its earlier and later patients):

```sh
cocoa collate   --raw-data-home data-raw/demo-2014-2016 \
                --processed-data-home processed/demo-2014-2016
cocoa tokenize  --tokenizer-home processed/demo-2011-2013/tokenizer.yaml \
                --processed-data-home processed/demo-2014-2016
cocoa winnow    --processed-data-home processed/demo-2014-2016
```

Since the two halves then share a vocabulary, `cocoa combine-datasets` can merge
them into one processed dataset (see
[Combining datasets](../guide/combining-datasets.md)):

```sh
cocoa combine-datasets processed/demo-2011-2013 processed/demo-2014-2016 \
    -o processed/demo-combined
```

`cocoa winnow` writes into the processed-data directory unless given
`--output-home` / `-o`, which also gets a copy of the processed-data directory's
files (see the [Winnower](winnower.md)). So to winnow the same tokenized dataset
under several configurations, give each run its own output directory:

```sh
cocoa winnow -p processed/demo -o processed/demo/winnowed-24h
cocoa winnow -p processed/demo -o processed/demo/winnowed-admit-24h \
    '~threshold.duration_s' threshold.after_start_time_s=86400
cocoa winnow -p processed/demo -o processed/demo/winnowed-icu \
    '~threshold.duration_s' threshold.first_occurrence=XFR-IN//icu
```

To look over a subject's timeline, serve it on localhost (Ctrl-C stops the
server), save it as a self-contained html file with `--export-html` / `-e`, or
save a static pdf of it with `--export-pdf`; give both to save both (see the
[Visualizer](visualizer.md)). Hospitalization 27708593 is one of the demo's:

```sh
cocoa visualize 27708593 --processed-data-home processed/demo --open
cocoa visualize 27708593 --processed-data-home processed/demo -e timeline.html
cocoa visualize 27708593 --processed-data-home processed/demo \
                -e timeline.html --export-pdf timeline.pdf
```

Without a subject id, `cocoa visualize` draws the vocabulary that tokenization
learned instead, with each binned code's training values against its bins and the
timelines summarized by split, and serves or saves it the same ways (see
[The vocabulary page](visualizer.md#the-vocabulary-page)):

```sh
cocoa visualize --processed-data-home processed/demo --open
cocoa visualize --processed-data-home processed/demo -e vocabulary.html
cocoa visualize --processed-data-home processed/demo \
                -e vocabulary.html --export-pdf vocabulary.pdf
```

## Command reference

Each command's `--help`:

### `cocoa collate`

```
 Usage: cocoa collate [OPTIONS] [overrides]...

 Collate raw data into a denormalized format.

 Reads collation configuration and produces a MEDS-like parquet file
 with collated events.

╭─ Arguments ─────────────────────────────────────────────────────────────╮
│   overrides      <str>  Config overrides: key=value sets a key, adding  │
│                         it if need be, and ~key deletes one             │
╰─────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────╮
│    --collation-config     -c      <path>  Collation configuration file  │
│                                           (overrides default)           │
│ *  --raw-data-home        -r      <str>   Raw data directory [required] │
│ *  --processed-data-home  -p      <str>   Processed data directory      │
│                                           [required]                    │
│    --verbose              -v              Verbose logging for collate;  │
│                                           this may cause memory issues  │
│                                           with large datasets           │
│    --help                 -h              Show this message and exit.   │
╰─────────────────────────────────────────────────────────────────────────╯
```

### `cocoa tokenize`

```
 Usage: cocoa tokenize [OPTIONS] [overrides]...

 Tokenize collated data into integer sequences.

 Reads collated parquet files and produces tokenized timelines with
 vocabulary and bin information.

╭─ Arguments ─────────────────────────────────────────────────────────────╮
│   overrides      <str>  Config overrides: key=value sets a key, adding  │
│                         it if need be, and ~key deletes one             │
╰─────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────╮
│    --tokenization-config  -c      <path>  Tokenization configuration    │
│                                           file (overrides default)      │
│ *  --processed-data-home  -p      <str>   Processed data directory      │
│                                           [required]                    │
│    --tokenizer-home       -t      <str>   Load a previously learned     │
│                                           tokenizer from this           │
│                                           tokenizer.yaml file (reuses   │
│                                           its frozen config,            │
│                                           vocabulary, and bins)         │
│    --verbose              -v              Verbose logging for tokenize; │
│                                           this may cause memory issues  │
│                                           with large datasets           │
│    --help                 -h              Show this message and exit.   │
╰─────────────────────────────────────────────────────────────────────────╯
```

### `cocoa winnow`

```
 Usage: cocoa winnow [OPTIONS] [overrides]...

 Winnow timelines for evaluation.

 Splits each timeline of the configured splits into a past and a future
 at a threshold, drops those that never reach it, and flags which outcome
 tokens fall on each side.

╭─ Arguments ─────────────────────────────────────────────────────────────╮
│   overrides      <str>  Config overrides: key=value sets a key, adding  │
│                         it if need be, and ~key deletes one             │
╰─────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────╮
│    --winnowing-config     -c      <path>  Winnowing configuration file  │
│                                           (overrides default)           │
│ *  --processed-data-home  -p      <str>   Processed data directory      │
│                                           [required]                    │
│    --output-home          -o      <str>   Directory to write the        │
│                                           winnowed files to, along with │
│                                           copies of the processed data  │
│                                           directory's files (defaults   │
│                                           to the processed data         │
│                                           directory)                    │
│    --verbose              -v              Verbose logging for winnow;   │
│                                           prints summary statistics     │
│    --help                 -h              Show this message and exit.   │
╰─────────────────────────────────────────────────────────────────────────╯
```

### `cocoa pipeline`

```
 Usage: cocoa pipeline [OPTIONS] [overrides]...

 Run the full pipeline: collate, tokenize, & winnow.

╭─ Arguments ─────────────────────────────────────────────────────────────╮
│   overrides      <str>  Config overrides, each key starting with the    │
│                         stage it's for: e.g. tokenization.n_bins=5 or   │
│                         ~winnowing.threshold.duration_s                 │
╰─────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────╮
│    --collation-config             <path>  Collation configuration file  │
│                                           (overrides default)           │
│    --tokenization-config          <path>  Tokenization configuration    │
│                                           file (overrides default)      │
│    --winnowing-config             <path>  Winnowing configuration file  │
│                                           (overrides default)           │
│ *  --raw-data-home        -r      <str>   Raw data directory [required] │
│ *  --processed-data-home  -p      <str>   Processed data directory      │
│                                           [required]                    │
│    --verbose              -v              Verbose logging for pipeline  │
│                                           steps                         │
│    --help                 -h              Show this message and exit.   │
╰─────────────────────────────────────────────────────────────────────────╯
```

### `cocoa visualize`

```
 Usage: cocoa visualize [OPTIONS] [subject_id] [overrides]...

 Visualize a subject's timeline, or the vocabulary, as an interactive html
 page.

 Given a subject id, draws that subject's timeline; without one, draws
 the vocabulary: each binned code's training values against its bins,
 every word with its training count, and the timelines by split, as
 tokenize --verbose summarizes them. Serves the page on localhost until
 interrupted, or saves it as a self-contained html file with
 --export-html, as a static pdf with --export-pdf, or both.

╭─ Arguments ─────────────────────────────────────────────────────────────╮
│   subject_id      <str>  Subject whose timeline to draw; without one,   │
│                          draws the vocabulary                           │
│   overrides       <str>  Config overrides, after a subject id:          │
│                          key=value sets a key, adding it if need be,    │
│                          and ~key deletes one                           │
╰─────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────╮
│ *  --processed-data-home   -p      <str>   Processed data directory     │
│                                            [required]                   │
│    --visualization-config  -c      <path>  Visualization configuration  │
│                                            file (overrides default)     │
│    --export-html           -e      <path>  Save the page to this        │
│                                            self-contained html file     │
│                                            instead of serving it        │
│    --export-pdf                    <path>  Save a static pdf of the     │
│                                            page to this file instead of │
│                                            serving it; with             │
│                                            --export-html, save both     │
│    --host                          <str>   Address to serve the page on │
│                                            [default: 127.0.0.1]         │
│    --port                          <int>   Port to serve the page on; 0 │
│                                            picks any                    │
│                                            [default: 8765]              │
│    --open                                  Open the page in a web       │
│                                            browser                      │
│    --show-winnowing                        Show the winnowing split of  │
│                                            the subject's timeline into  │
│                                            past and future, and its     │
│                                            outcome flags                │
│    --help                  -h              Show this message and exit.  │
╰─────────────────────────────────────────────────────────────────────────╯
```

### `cocoa combine-datasets`

```
 Usage: cocoa combine-datasets [OPTIONS] {input_data_dirs}...

 Combine multiple processed datasets into one.

 Merges the parquet files of the input directories and keeps the first
 directory's tokenizer.yaml, warning if another's differs.

╭─ Arguments ─────────────────────────────────────────────────────────────╮
│ *    input_data_dirs      <str>  Processed data directories to combine  │
│                                  [required]                             │
╰─────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────╮
│ *  --output-data-dir  -o      <str>  Output directory for combined data │
│                                      [required]                         │
│    --help             -h             Show this message and exit.        │
╰─────────────────────────────────────────────────────────────────────────╯
```

### `cocoa get-demo`

```
 Usage: cocoa get-demo [OPTIONS]

 Write the MIMIC-IV demo's CLIF tables, from clifpy.

 Copies the tables of the demo that clifpy ships to a raw data
 directory, and derives from them, with clifpy, the four more that the
 shipped collation config reads. Needs clifpy, which cocoa's demo
 extra installs.

╭─ Options ───────────────────────────────────────────────────────────────╮
│ *  --raw-data-home  -r      <str>  Raw data directory to write to       │
│                                    [required]                           │
│    --verbose        -v             Show what clifpy prints as it        │
│                                    derives tables                       │
│    --help           -h             Show this message and exit.          │
╰─────────────────────────────────────────────────────────────────────────╯
```

---

::: cocoa.cli
