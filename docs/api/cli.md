# CLI

Cocoa ships a command-line interface, `cocoa`, that drives every stage of the
pipeline. Each stage has its own command, and there are two convenience commands
for running everything at once and for merging datasets.

## Commands

| Command | What it does |
|---------|--------------|
| `cocoa collate` | Collate raw tables into a denormalized event stream. |
| `cocoa tokenize` | Tokenize collated data into integer timelines. |
| `cocoa winnow` | Prepare held-out timelines for evaluation. |
| `cocoa pipeline` | Run `collate`, `tokenize`, and `winnow` end-to-end. |
| `cocoa combine-datasets` | Merge multiple processed datasets into one. |

Every stage command accepts `--processed-data-home` / `-p` (the working
directory for intermediate and output files) and `--verbose` / `-v` (extra
logging and summary statistics). Each also takes an optional `-c` config file
that overrides the packaged default for that stage.

Run any command with `-h` / `--help` to see its full set of options:

```sh
cocoa --help
cocoa tokenize --help
```

## Typical usage

Run the whole pipeline in one go:

```sh
cocoa pipeline \
    --raw-data-home /path/to/raw \
    --processed-data-home ./processed/mimic \
    --verbose
```

Or drive the stages individually — for example, to reuse a previously learned
tokenizer via `--tokenizer-home` (see the
[Tokenizer Transfer](../recipes/tokenizer-transfer.md) recipe):

```sh
cocoa collate   --raw-data-home /path/to/raw --processed-data-home ./processed/ucmc
cocoa tokenize  --tokenizer-home ./processed/mimic/tokenizer.yaml \
                --processed-data-home ./processed/ucmc
cocoa winnow    --processed-data-home ./processed/ucmc
```

::: cocoa.cli
