# CLI

Cocoa ships a command-line interface, `cocoa`, that drives every stage of the
pipeline. Each stage has its own command. There are also two convenience
commands, for running everything at once and for merging datasets, and a viewer
for inspecting one subject's tokenized timeline.

## Commands

| Command                  | What it does                                            |
| ------------------------ | ------------------------------------------------------- |
| `cocoa collate`          | Collate raw tables into a denormalized event stream.    |
| `cocoa tokenize`         | Tokenize collated data into integer timelines.          |
| `cocoa winnow`           | Prepare held-out timelines for evaluation.              |
| `cocoa pipeline`         | Run `collate`, `tokenize`, and `winnow` end-to-end.     |
| `cocoa combine-datasets` | Merge multiple processed datasets into one.             |
| `cocoa visualize`        | Show one subject's timeline as an interactive web page. |

Every stage command accepts `--processed-data-home` / `-p` (the working directory
for intermediate and output files) and `--verbose` / `-v` (extra logging and
summary statistics). Each also takes an optional `-c` config file that overrides
the packaged default for that stage. `cocoa visualize` takes `-p` and `-c` too,
but has no `--verbose`.

To change individual config keys for one run, list them after the command's
options, as in `cocoa tokenize -p processed/demo n_bins=5`; for `cocoa pipeline`,
start each key with its stage, as in `tokenization.n_bins=5`. See
[Overriding config keys](../index.md#overriding-config-keys) for the syntax.

Run any command with `-h` / `--help` to see its full set of options:

```sh
cocoa --help
cocoa tokenize --help
```

## Typical usage

The examples here use the paths of the
[MIMIC-IV demo](../recipes/mimic-iv-demo.md), run from its working directory. Run
the whole pipeline in one go:

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

`cocoa winnow` writes into the processed-data directory unless given
`--output-home` / `-o`, which also gets a copy of the processed-data directory's
files (see the [Winnower](winnower.md)). So to winnow the same tokenized dataset
under several configurations, give each run its own output directory:

```sh
cocoa winnow -p processed/demo -o processed/demo/winnowed-24h
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

---

::: cocoa.cli
