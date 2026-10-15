## The MIMIC-IV demo, end to end: from raw tables to tokenized timelines

[MIMIC-IV](https://mimic.mit.edu) holds the deidentified health records of
patients at the Beth Israel Deaconess Medical Center in Boston, and only
credentialed users can download it. Its
[Clinical Database Demo](https://physionet.org/content/mimic-iv-demo/2.2/) [^1],
100 of those patients with their 275 hospitalizations, is open to anyone. cocoa's
shipped configs are written for
[CLIF](https://clif-icu.com/data-dictionary/data-dictionary-2.1.0), the Common
Longitudinal ICU data Format, so in this example we'll convert the demo to CLIF
with the [CLIF-MIMIC ETL](https://github.com/bbj-lab/CLIF-MIMIC), derive four
more tables with
[clifpy](https://github.com/Common-Longitudinal-ICU-data-Format/clifpy), the CLIF
consortium's Python package, and then collate, tokenize, and winnow the result
with the shipped configs, and draw one timeline. After the downloads, the run
takes well under a minute on a laptop.

You'll need a POSIX shell (macOS, Linux, or WSL) with git, curl, and unzip;
[uv](https://docs.astral.sh/uv/) 0.6.8 or later, which sets up separate
environments for the ETL and for clifpy; and cocoa 26.10 or later
(`pip install -U 'cocoa-tokenizer>=26.10'`; the numbers below come from 26.10.1).
The environments take a little over 1 GB of disk, and the data, with everything
made from it, about 47 MB.

### 0. Make a working directory

```sh
mkdir -p mimic-iv-demo && cd mimic-iv-demo
```

Every command below runs from here. Keep `'` out of its path, since the ETL can't
read data from such a path.

### 1. Download the demo

PhysioNet can be slow, so the 15 MB download may take several minutes:

```sh
curl -fL -o mimic-iv-demo-2.2.zip https://physionet.org/content/mimic-iv-demo/get-zip/2.2/
unzip -qo mimic-iv-demo-2.2.zip
```

This unpacks the MIMIC-IV tables, as compressed csv files, under
`mimic-iv-clinical-database-demo-2.2/hosp/` and `.../icu/`. As in MIMIC-IV, dates
are shifted into the future to deidentify them (admissions run from 2110 to
2201), each patient's by its own amount, so that times line up within a patient
but not across patients.

### 2. Convert it to CLIF

Clone the ETL's `demo` branch:

```sh
git clone --branch demo https://github.com/bbj-lab/CLIF-MIMIC.git
```

The branch adds
[a fix and three files](https://github.com/bbj-lab/CLIF-MIMIC/compare/main...demo)
to the ETL's `main`. The ETL was written for the full MIMIC-IV, which charts
every item (MIMIC's word for a kind of measurement) that the ETL maps at least
once. The demo's 100 patients lack a few, and without them two tables,
`patient_assessments` and `respiratory_support`, fail to build on `main`. On
`demo` both do without, and on data with every item, they come out with the same
rows as on `main`. One file is the ETL's config, `CLIF-MIMIC/config/config.json`,
which on `main` you write yourself. It points the ETL at the demo, at
`../mimic-iv-clinical-database-demo-2.2` (relative to `CLIF-MIMIC`, where the ETL
runs), and chooses the twelve tables that the steps below read. The other two are
scripts, one for step 3 and one that splits the demo in two for the
[tokenizer-transfer recipe](tokenizer-transfer.md).

Then run it. `--frozen` installs exactly the versions in the ETL's lockfile, and
`--no-dev` leaves out its development tools; the first run also installs the
Python version the ETL pins, 3.10.5:

```sh
(cd CLIF-MIMIC && uv run --frozen --no-dev python main.py)
```

If a virtual environment is active (cocoa's, say), uv first warns that it doesn't
match the ETL's and will be ignored, as it should be. The ETL converts the 16
MIMIC-IV tables it may read to parquet, under
`mimic-iv-clinical-database-demo-2.2/parquet/`, and then writes the CLIF tables,
with times in UTC, to `CLIF-MIMIC/output/rclif-2.1/`, and its logs to
`CLIF-MIMIC/output/logs/`. Two warnings are expected:
`no events for items [225448, 226237, 227581]; their columns stay empty`, from
that fix, and
`df length after augmentation and mapping is different from expected in last run`,
which compares against a count from the full MIMIC-IV. A table that fails to
build is logged as `error building ...`, but the ETL carries on, and still ends
with `finished building all clif tables!`. So check that it wrote 13 files, the
twelve tables and `clif_patient_assessments_raw_gcs.parquet`:

```sh
ls CLIF-MIMIC/output/rclif-2.1
```

The steps below read them from `data-raw/demo`, next to the `processed/demo` that
cocoa writes, so copy them there:

```sh
mkdir -p data-raw/demo
cp CLIF-MIMIC/output/rclif-2.1/*.parquet data-raw/demo
```

### 3. Derive the tables that clifpy makes

The shipped collation config also reads four tables that clifpy derives from the
CLIF tables:

- `clif_respiratory_support_processed`, respiratory support after clifpy's
  waterfall, which infers missing devices and modes, carries settings forward,
  and adds a row at the end of each hour between a stay's first and last records;
- `clif_medication_admin_continuous_converted` and
  `clif_medication_admin_intermittent_converted`, medication doses converted to
  one unit per medication, so that the tokenizer bins each medication's doses on
  a single scale (clifpy flags a dose it can't convert, and the collation config
  then keeps its code but drops its value); and
- `clif_sofa`, Sequential Organ Failure Assessment (SOFA) scores, from the worst
  values over each stay, timed at the stay's last record, just after discharge.
  So the `SOFA//` tokens, and the `LABEL//sepsis_onset` derived from them, come
  at the end of a timeline.

The script on the ETL's branch,
[`CLIF-MIMIC/scripts/clifpy_tables.py`](https://github.com/bbj-lab/CLIF-MIMIC/blob/demo/scripts/clifpy_tables.py),
writes them next to the others. Its preferred units are the ones we use on the
full MIMIC-IV (see
[coreopsis](https://github.com/bbj-lab/coreopsis/blob/main/recipes/run_clifpy.py)),
with dextrose split as the ETL now splits it, so that every dose clifpy can
convert does. Run it with clifpy 2.1.0:

```sh
uv run --isolated --no-project --managed-python --python 3.12 \
    --exclude-newer 2026-10-14T16:00:00Z --with clifpy==2.1.0 \
    python CLIF-MIMIC/scripts/clifpy_tables.py data-raw/demo
```

uv runs the script in a fresh environment of its own: `--isolated` keeps out any
virtual environment that's active or nearby (cocoa's, say), `--managed-python`
the packages of your own Python, and `--no-project` any project the working
directory sits in, while `--exclude-newer` resolves clifpy's dependencies as they
stood when this recipe was tested.

clifpy is chatty. It starts with
`Polars imported but failed runtime check, falling back to DuckDB`, as clifpy 2.1
predates polars 2, and along the way it warns of medications given a preferred
unit that the demo never charts, of a unit it can't convert to (`meq/min`, sodium
bicarbonate's own), and of assessments with both a numerical and a categorical
value; pandas adds a `FutureWarning`. It ends with
`SOFA computation completed. Results stored in self.sofa_df with shape: (272, 10)`,
and `data-raw/demo/` then holds 17 files. If you rerun the ETL, copy its tables
again and rerun this step too.

### 4. Collate

With every table in place, cocoa's shipped collation config runs as is. `-r`
names the directory of tables to read, and `-p` the directory to write to, which
the later steps read from:

```sh
cocoa collate -r data-raw/demo -p processed/demo --verbose
```

This writes `processed/demo/meds.parquet`, the 275,648 events of the 275
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
┌──────────┬───────┬──────┐
│ split    ┆ count ┆ rate │
│ ---      ┆ ---   ┆ ---  │
│ str      ┆ u32   ┆ f64  │
╞══════════╪═══════╪══════╡
│ train    ┆ 187   ┆ 0.68 │
│ tuning   ┆ 33    ┆ 0.12 │
│ held_out ┆ 55    ┆ 0.2  │
└──────────┴───────┴──────┘
```

### 5. Tokenize

```sh
cocoa tokenize -p processed/demo --verbose
```

The tokenizer learns its vocabulary, and the deciles of each code's numeric
values, on the 187 training hospitalizations, and applies both, frozen, to the
rest. With the shipped config, each event becomes one token: its code, with the
decile of its value appended when it has one, as in `LAB-RES//sodium_Q5`. It
writes `processed/demo/tokenizer.yaml`, the vocabulary and bin edges, and
`processed/demo/tokens_times.parquet`, a row for each hospitalization holding its
tokens, from `BOS` to `EOS`, and their times. `--verbose` prints the timelines'
lengths, from 10 tokens to 11,461 with a median of 339, and a few timelines token
by token, and ends with:

```
✓ Tokenization completed in 0.72s.
  Output: …/mimic-iv-demo/processed/demo/tokens_times.parquet
  Output: …/mimic-iv-demo/processed/demo/tokenizer.yaml
  Vocabulary size: 1100 tokens
```

### 6. Winnow, and look at a timeline

`cocoa winnow` readies timelines for evaluating a model. It cuts each one 24
hours after its first event into a past and a future, keeps those that run past
the cut, and flags which outcome tokens (`XFR-IN//icu`, `RESP//imv`,
`DSCG//expired`, `ASMT//cam_total_yes`, and the `LABEL//` tokens) fall on each
side. It writes `processed/demo/train_for_inference.parquet`,
`processed/demo/tuning_for_inference.parquet`, and
`processed/demo/held_out_for_inference.parquet`, keeping 169, 30, and 50 of the
187, 33, and 55 hospitalizations, and `--verbose` prints how often each outcome
falls on each side:

```sh
cocoa winnow -p processed/demo --verbose
```

The cut comes 24 hours after a timeline's first event, not its admission. Most
of the demo's timelines open before their admission, 180 of them with an arrival
in the emergency department, so many are cut less than a day into the stay. To
cut each one 24 hours after its admission instead, swap the threshold for
`after_start_time_s`, which counts from the subject's start time; this keeps
166, 30, and 50:

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
  winnows in one go (steps 4 to 6, short of drawing), and takes config overrides,
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
