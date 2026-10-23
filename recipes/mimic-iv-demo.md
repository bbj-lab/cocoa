## The MIMIC-IV demo, end to end: from raw tables to tokenized timelines

[MIMIC-IV](https://mimic.mit.edu) holds the deidentified health records of
patients at the Beth Israel Deaconess Medical Center in Boston, and only
credentialed users can download it. Its
[Clinical Database Demo](https://physionet.org/content/mimic-iv-demo/2.2/) [^1],
100 of those patients, is open to anyone. cocoa's shipped configs are written for
[CLIF](https://clif-icu.com/data-dictionary/data-dictionary-2.1.0), the Common
Longitudinal ICU data Format, and
[clifpy](https://github.com/Common-Longitudinal-ICU-data-Format/clifpy), the CLIF
consortium's Python package, ships the demo as CLIF tables. So in this example
we'll take the demo's tables from clifpy, derive four more with it, and then
collate, tokenize, and winnow the result with the shipped configs, and draw one
timeline. The run takes well under a minute on a laptop.

You'll need Python 3.11 to 3.14 and cocoa 26.10.3 or later, installed with its
`demo` extra, which brings clifpy. The numbers below come from cocoa 26.10.3 with
clifpy 2.1.0; another release of clifpy may ship other tables. clifpy and the
packages it brings take about 300 MB of disk, and the data, with everything made
from it, about 7 MB.

### 0. Make a working directory

```sh
mkdir -p mimic-iv-demo && cd mimic-iv-demo
```

Every command below runs from here. clifpy 2.1.0 pins pandas to 2.3.3, which has
no wheels for Python 3.15, so install cocoa in an environment of its own, made
with a Python from 3.11 to 3.14, such as `python3.14`:

```sh
python3.14 -m venv .venv && . .venv/bin/activate
pip install -U 'cocoa-tokenizer[demo]>=26.10.3'
```

### 1. Get the demo

clifpy keeps the demo's CLIF tables in its package, so nothing more has to be
downloaded. `cocoa get-demo` copies them to the directory given by `-r`, here
`data-raw/demo`, next to the `processed/demo` that cocoa writes in the steps
below:

```sh
cocoa get-demo -r data-raw/demo
```

clifpy 2.1.0 ships 14 tables, of 310 hospitalizations of the demo's 100 patients,
with times in UTC. As in MIMIC-IV, dates are shifted into the future to
deidentify them (admissions run from 2110 to 2201), each patient's by its own
amount, so that times line up within a patient but not across patients.

The shipped collation config also reads four tables that clifpy derives from
these, and `cocoa get-demo` derives them with clifpy and writes them next to the
others:

- `clif_respiratory_support_processed`, respiratory support after clifpy's
  waterfall, which infers missing devices and modes, carries settings forward,
  and adds a row at the end of each hour between a stay's first and last records;
- `clif_medication_admin_continuous_converted` and
  `clif_medication_admin_intermittent_converted`, medication doses converted to
  one unit per medication, so that the tokenizer bins each medication's doses on
  a single scale (clifpy flags a dose it can't convert, and the collation config
  then keeps its code but drops its value). The preferred units are the ones we
  use on the full MIMIC-IV (see
  [coreopsis](https://github.com/bbj-lab/coreopsis/blob/main/recipes/run_clifpy.py)),
  so that every dose clifpy can convert does; and
- `clif_sofa`, Sequential Organ Failure Assessment (SOFA) scores, from the worst
  values over each stay, timed at the stay's last record, after every value they
  score; most stays' last record comes some hours before discharge. clifpy
  stitches together a patient's hospitalizations that come at most 6 hours apart,
  so the 310 hospitalizations make 307 stays. The `SOFA//` tokens, and the
  `LABEL//sepsis_onset` derived from them, come near the end of a timeline.

clifpy is chatty, so `cocoa get-demo` keeps what it prints to itself unless given
`--verbose`, and ends with:

```
✓ Demo written in 1.09s.
  Output: …/mimic-iv-demo/data-raw/demo (18 tables, from clifpy 2.1.0)
```

### 2. Collate

With every table in place, cocoa's shipped collation config runs as is. `-r`
names the directory of tables to read, and `-p` the directory to write to, which
the later steps read from:

```sh
cocoa collate -r data-raw/demo -p processed/demo --verbose
```

This writes `processed/demo/meds.parquet`, the 297,060 events of the 310
hospitalizations, a row each with its `subject_id`, `time`, `code` (such as
`LAB-RES//sodium`), and any value; the shipped config makes each hospitalization
a subject (`subject_id: hospitalization_id`). It also writes
`processed/demo/subject_splits.parquet`, which assigns patients, with all of
their hospitalizations, to splits in the order of their first admissions: the
first 70 to train, the next 10 to tuning, and the last 20 to held out. Since each
patient's dates are shifted separately, that order isn't chronological in real
time. `--verbose` prints summaries as it goes, among them the split by
hospitalization:

```
┌──────────┬───────┬────────┐
│ split    ┆ count ┆ rate   │
│ ---      ┆ ---   ┆ ---    │
│ str      ┆ u32   ┆ f64    │
╞══════════╪═══════╪════════╡
│ train    ┆ 208   ┆ 0.671  │
│ tuning   ┆ 39    ┆ 0.1258 │
│ held_out ┆ 63    ┆ 0.2032 │
└──────────┴───────┴────────┘
```

### 3. Tokenize

```sh
cocoa tokenize -p processed/demo --verbose
```

The tokenizer learns its vocabulary, and the deciles of each code's numeric
values, on the 208 training hospitalizations, and applies both, frozen, to the
rest. With the shipped config, each event becomes one token: its code, with the
decile of its value appended when it has one, as in `LAB-RES//sodium_Q5`. It
writes `processed/demo/tokenizer.yaml`, the vocabulary and bin edges, and
`processed/demo/tokens_times.parquet`, a row for each hospitalization holding its
tokens, from `BOS` to `EOS`, and their times. `--verbose` prints the timelines'
lengths, from 10 tokens to 11,614, half of them shorter than 311, and a few
timelines token by token, and ends with:

```
✓ Tokenization completed in 0.49s.
  Output: …/mimic-iv-demo/processed/demo/tokens_times.parquet
  Output: …/mimic-iv-demo/processed/demo/tokenizer.yaml
  Vocabulary size: 1115 tokens
```

### 4. Winnow, and look at a timeline

`cocoa winnow` readies timelines for evaluating a model. It cuts each one 24
hours after its first event into a past and a future, keeps those that run past
the cut, and flags which outcome tokens (`XFR-IN//icu`, `RESP//imv`,
`DSCG//expired`, `ASMT//cam_total_yes`, and the `LABEL//` tokens) fall on each
side. It writes `processed/demo/train_for_inference.parquet`,
`processed/demo/tuning_for_inference.parquet`, and
`processed/demo/held_out_for_inference.parquet`, keeping 187, 36, and 57 of the
208, 39, and 63 hospitalizations, and `--verbose` prints how often each outcome
falls on each side:

```sh
cocoa winnow -p processed/demo --verbose
```

The cut comes 24 hours after a timeline's first event, not its admission. Most of
the demo's timelines open before their admission, 213 of them with an arrival in
the emergency department, so many are cut less than a day into the stay. To cut
each one 24 hours after its admission instead, swap the threshold for
`after_start_time_s`, which counts from the subject's start time; this keeps 184,
34, and 55:

```sh
cocoa winnow -p processed/demo -o processed/demo/winnowed-admit-24h \
    '~threshold.duration_s' threshold.after_start_time_s=86400
```

Then draw one timeline:

```sh
cocoa visualize 27708593 -p processed/demo --open
```

`cocoa visualize` serves the page until you press Ctrl-C; to keep it as a file
instead, pass `--export-html timeline.html` (or `--export-pdf timeline.pdf`).
Hospitalization 27708593 is the earliest admission in the held-out split, a
week-long elective neurosurgical stay with 17 hours in the surgical ICU. The page
draws its events in lanes by kind:

<img src="https://raw.githubusercontent.com/burkh4rt/cocoa/master/img/vis-screenshot.png" alt="timeline of training hospitalization 21759936, with the labs at one time listed" width="700"/>

Lanes can be displayed interactively:

<img src="https://raw.githubusercontent.com/burkh4rt/cocoa/master/img/vis-screenshot-2.png" alt="timeline of held-out hospitalization 27708593, with the vitals lane opened" width="700"/>

### Next steps

- `cocoa pipeline -r data-raw/demo -p processed/demo` collates, tokenizes, and
  winnows in one go (steps 2 to 4, short of drawing), and takes config overrides,
  each key prefixed by its stage: `collation.`, `tokenization.`, or `winnowing.`.
- Clock tokens mark every fourth hour (`CLCK//00`, `CLCK//04`, and so on) on the
  clock of the collation config's `default_timezone`, which the shipped config
  sets to `UTC`. MIMIC-IV's times are Boston's, so for local hours, run the
  pipeline again into a new directory:
  `cocoa pipeline -r data-raw/demo -p processed/demo-local collation.default_timezone=America/New_York tokenization.insert_clocks=true`.
  [Date-based generative inference](date-based-generative-inference.md) uses such
  clock tokens to cut timelines at midnight.
- To change what's collated or how it's tokenized, copy a shipped config from
  [src/cocoa/config](https://github.com/bbj-lab/cocoa/tree/master/src/cocoa/config),
  edit it, and pass it with `-c`;
  [configure a new event token](configure-new-token.md) walks through adding an
  entry.
- [Tokenizer transfer](tokenizer-transfer.md) splits the demo in two, by when its
  patients were treated, and applies a vocabulary learned on the earlier patients
  to the later ones.
- To train a model on the timelines, see
  [🦜 cotorra](https://pypi.org/project/cotorra/).

[^1]:
    A. Johnson, L. Bulgarelli, T. Pollard, S. Horng, L. A. Celi, & R. Mark,
    [MIMIC-IV Clinical Database Demo (version 2.2)](https://doi.org/10.13026/dp1f-ex47),
    PhysioNet (2023), RRID:SCR_007345
