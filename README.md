<!-- --8<-- [start:intro] -->

<p align="center">
<img src="https://raw.githubusercontent.com/burkh4rt/cocoa/master/img/cocoa-bean.png" alt="cocoa bean" width="400"/>
</p>

# Cocoa: a configurable collator

[![PyPI Version](https://img.shields.io/pypi/v/cocoa-tokenizer)](https://pypi.org/project/cocoa-tokenizer/)
[![DOI](https://raw.githubusercontent.com/burkh4rt/cocoa/master/img/1174829117.svg)](https://doi.org/10.5281/zenodo.20413460)
[![SWH](https://archive.softwareheritage.org/badge/origin/https://github.com/bbj-lab/cocoa/)](https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/bbj-lab/cocoa)
[![tests](https://github.com/bbj-lab/cocoa/actions/workflows/tests.yml/badge.svg)](https://github.com/bbj-lab/cocoa/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/bbj-lab/cocoa/graph/badge.svg?token=6QXX2PE38Q)](https://codecov.io/gh/bbj-lab/cocoa)

> ☕️ Chicago's second favorite bean

## About

This library provides a configurable way to (1) collate data from multiple
sources into a single denormalized dataframe, (2) tokenize the records into
sequences of discrete integers, (3) winnow the records in preparation for
downstream inference tasks, and (4) visualize both individual records and the
tokenization process. It benefits from previous experience preparing data for
training foundation models on tokenized electronic health records. [^1] [^2] [^3]
[^4] [^5]

<!-- --8<-- [end:intro] -->

<!-- --8<-- [start:install] -->

## Installation

Install the latest release from PyPI:

```sh
pip install cocoa-tokenizer
```

This installs the `cocoa` command. To see it run end to end on public data,
follow the
[MIMIC-IV demo](https://cocoa-tokenizer.readthedocs.io/latest/recipes/mimic-iv-demo/),
which installs cocoa with its `demo` extra (on Python 3.11 to 3.14):

```sh
pip install 'cocoa-tokenizer[demo]'
```

The extra brings [clifpy](https://pypi.org/project/clifpy/), the CLIF
consortium's Python package, which ships the demo as CLIF tables, and
`cocoa get-demo` writes them out. The examples below come from that demo and use
its paths: it reads CLIF tables from `data-raw/demo` and writes processed data to
`processed/demo`.

<!-- --8<-- [end:install] -->

## Usage

Each stage of the pipeline is a `cocoa` command, driven by a YAML config, and
`cocoa pipeline` runs the first three in turn:

```
collate  →  tokenize  →  winnow
                      ↘  visualize
```

The [documentation](https://cocoa-tokenizer.readthedocs.io/latest/) covers:

- [Collation](https://cocoa-tokenizer.readthedocs.io/latest/guide/collation/):
  pulling raw tables into a single denormalized,
  [MEDS](https://github.com/Medical-Event-Data-Standard/meds)-like table of
  events, and splitting the subjects into train, tuning, and held-out sets, with
  `cocoa collate`.
- [Tokenization](https://cocoa-tokenizer.readthedocs.io/latest/guide/tokenization/):
  turning each subject's events into a sequence of integer tokens, with a
  vocabulary and quantile bins learned on the training split, with
  `cocoa tokenize`.
- [Winnowing](https://cocoa-tokenizer.readthedocs.io/latest/guide/winnowing/):
  cutting timelines into a past and a future at a threshold and flagging the
  outcome tokens on each side, for evaluation, with `cocoa winnow`.
- [Visualization](https://cocoa-tokenizer.readthedocs.io/latest/guide/visualization/):
  drawing a subject's tokenized timeline, or the learned vocabulary, as an
  interactive page, with `cocoa visualize`.
- [Combining datasets](https://cocoa-tokenizer.readthedocs.io/latest/guide/combining-datasets/):
  merging processed directories into one with `cocoa combine-datasets`.
- [Configuration](https://cocoa-tokenizer.readthedocs.io/latest/guide/configuration/):
  the YAML config each stage reads, and how to change its keys on the command
  line.
- The [CLI reference](https://cocoa-tokenizer.readthedocs.io/latest/api/cli/),
  with every command's options, and the
  [recipes](https://cocoa-tokenizer.readthedocs.io/latest/recipes/), worked
  examples from a run on public data to tokenizer transfer.

## Contributing

Contributions are welcome, from bug reports and typo fixes to new features. To
get started, see the
[contributing guide](https://github.com/bbj-lab/cocoa/blob/master/CONTRIBUTING.md).

<!-- --8<-- [start:references] -->

[^1]:
    M. Burkhart, B. Ramadan, Z. Liao, K. Chhikara, J. Rojas, W. Parker, & B.
    Beaulieu-Jones, Foundation models for electronic health records:
    representation dynamics and transferability,
    [arXiv:2504.10422](https://doi.org/10.48550/arXiv.2504.10422)

[^2]:
    M. Burkhart, B. Ramadan, L. Solo, W. Parker, & B. Beaulieu-Jones,
    [Quantifying surprise in clinical care: Detecting highly informative events in electronic health records with foundation models](https://doi.org/10.1142/9789819824755_0013),
    Pacific Symposium on Biocomputing 31 (2026), 173–188

[^3]:
    L. Solo, M. McDermott, W. Parker, B. Ramadan, M. Burkhart, & B.
    Beaulieu-Jones, Efficient generative prediction for EHR foundation models:
    the SCOPE and REACH estimators,
    [arXiv:2602.03730](https://doi.org/10.48550/arXiv.2602.03730)

[^4]:
    I. Lee, L. Solo, M. Burkhart, B. Ramadan, W. Parker, & B. Beaulieu-Jones,
    Representation before training: a fixed-budget benchmark for generative
    medical event models,
    [arXiv:2604.16775](https://doi.org/10.48550/arXiv.2604.16775), _to appear in
    ML4H 2026_

[^5]:
    M. Burkhart, L. Solo, I. Lee, S. Charles, Z. Liao, K. Chhikara, D. Therese,
    W.-T. Liao, C. Gao, W. Parker, & B. Beaulieu-Jones, Federated generative
    event models for tokenized electronic health records,
    [arXiv:2608.02939](https://doi.org/10.48550/arXiv.2608.02939)

<!-- --8<-- [end:references] -->

<!--

Run in tmux:
```
tmux new -s co || tmux a -t co
```

Format:
```sh
ruff format .
ruff check . --fix
```

Send to bbj-lab1:
```
rsync -avht \
 --delete \
 --exclude "output" \
 --exclude "data-raw" \
 --exclude "processed/" \
 --exclude ".venv" \
 --exclude ".idea" \
 ~/Documents/chicago/cocoa \
 bbj-lab1:~
```


Send to randi:
```
for d in data-raw processed; do
	ln -s /gpfs/data/bbj-lab/users/burkh4rt/$d $d
done
```
```
rsync -avh \
 --exclude "output" \
 --exclude "processed" \
 --exclude "data-raw" \
 --exclude "logs" \
 --exclude "wandb" \
 --exclude ".venv/" \
 --exclude ".idea/" \
 ~/Documents/chicago/cocoa \
 randi:/gpfs/data/bbj-lab/users/burkh4rt
```

Make docs:
```
zensical build
zensical serve --dev-addr 127.0.0.1:8001
```

Make tag and send to pypi and testpypi (pushing a `v*` tag runs
`.github/workflows/publish.yml`, which builds and uploads to both):
```
git tag -s v26.10.4 -m "add provenance"
git push origin v26.10.4
git push upstream v26.10.4
```
-->
