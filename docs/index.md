<p align="center">
<img src="https://raw.githubusercontent.com/burkh4rt/cocoa/master/img/cocoa-bean.png"
alt="cocoa bean" width="300" style="display: block; margin: 0 auto;
-webkit-mask-image: radial-gradient(ellipse at center, rgba(0,0,0,1) 50%, rgba(0,0,0,0) 100%);
mask-image: radial-gradient(ellipse at center, rgba(0,0,0,1) 50%, rgba(0,0,0,0) 100%);"/>
</p>

# Cocoa: a configurable collator

[![PyPI Version](https://img.shields.io/pypi/v/cocoa-tokenizer)](https://pypi.org/project/cocoa-tokenizer/)
[![DOI](https://raw.githubusercontent.com/burkh4rt/cocoa/master/img/1174829117.svg)](https://doi.org/10.5281/zenodo.20413460)

> ☕️ Chicago's second favorite bean

## About

Cocoa provides a configurable way to collate data from multiple sources into a
single denormalized dataframe and create tokenized timelines from the results.
It benefits from previous experience collating data to train foundation models
on tokenized electronic health records.

The pipeline has three stages:

| Stage | Class | CLI command |
|---|---|---|
| Collation | [`Collator`](api/collator.md) | `cocoa collate` |
| Tokenization | [`Tokenizer`](api/tokenizer.md) | `cocoa tokenize` |
| Winnowing | [`Winnower`](api/winnower.md) | `cocoa winnow` |

## Installation

```sh
git clone git@github.com:bbj-lab/cocoa.git
cd cocoa
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

Or from PyPI:

```sh
pip install cocoa-tokenizer
```

## Quick start

Run the full pipeline with:

```sh
cocoa pipeline \
    --raw-data-home /path/to/raw \
    --processed-data-home /path/to/processed \
    --verbose
```

Each stage can also be run individually — see the [CLI reference](api/cli.md).

For common scenarios (tokenizer transfer, date-based inference, adding new
tokens), see the [Recipes](recipes/index.md) section.
