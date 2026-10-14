## The MIMIC-IV demo, end to end: from raw tables to tokenized timelines

[MIMIC-IV](https://mimic.mit.edu) holds the deidentified health records of
patients at the Beth Israel Deaconess Medical Center in Boston, and only
credentialed users can download it. Its
[Clinical Database Demo](https://physionet.org/content/mimic-iv-demo/2.2/), 100
of those patients with their 275 hospitalizations, is open to anyone. cocoa's
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
made from it, about 45 MB.

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

The demo is released under the
[Open Data Commons Open Database License v1.0](https://physionet.org/content/mimic-iv-demo/view-license/2.2/),
which asks you to say where anything you share came from, and to share a database
derived from it (the CLIF tables or the timelines, say) under the same license.
If you publish anything made from it, cite it, along with MIMIC-IV and PhysioNet,
as [its page](https://physionet.org/content/mimic-iv-demo/2.2/) asks:

> Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R.
> (2023). MIMIC-IV Clinical Database Demo (version 2.2). PhysioNet.
> RRID:SCR_007345. <https://doi.org/10.13026/dp1f-ex47>

### 2. Convert it to CLIF

Clone the ETL's `demo` branch, at the commit this recipe was tested with:

```sh
git clone --branch demo https://github.com/bbj-lab/CLIF-MIMIC.git
git -C CLIF-MIMIC checkout -q 5be22e320e1d
```

The branch adds
[one fix](https://github.com/bbj-lab/CLIF-MIMIC/commit/5be22e320e1d) to the ETL's
`main`. The ETL was written for the full MIMIC-IV, which charts every item
(MIMIC's word for a kind of measurement) that the ETL maps at least once. The
demo's 100 patients lack a few, and without them two tables,
`patient_assessments` and `respiratory_support`, fail to build on `main`. On
`demo` both do without, and on data with every item, they come out with the same
rows as on `main`.

Point the ETL at the demo, by absolute path (as the ETL asks, hence `$PWD`), and
choose the twelve tables that the steps below read:

```sh
cat > CLIF-MIMIC/config/config.json <<EOF
{
  "current_workspace": "demo",
  "demo": {
    "mimic_csv_dir": "$PWD/mimic-iv-clinical-database-demo-2.2",
    "mimic_parquet_dir": ""
  },
  "create_mimic_parquet_from_csv": 1,
  "overwrite_existing_mimic_parquet": 0,
  "clif_output_dir_name": "",
  "clif_version": "2.1",
  "clif_tables": {
    "patient": 1,
    "hospitalization": 1,
    "adt": 1,
    "vitals": 1,
    "labs": 1,
    "patient_assessments": 1,
    "respiratory_support": 1,
    "medication_admin_continuous": 1,
    "medication_admin_intermittent": 1,
    "position": 1,
    "crrt_therapy": 1,
    "code_status": 1
  }
}
EOF
```

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

This script writes them next to the others. Its preferred units are the ones we
use on the full MIMIC-IV (see
[coreopsis](https://github.com/bbj-lab/coreopsis/blob/main/recipes/run_clifpy.py)),
with dextrose split as the ETL now splits it, so that every dose clifpy can
convert does:

```sh
cat > clifpy_tables.py <<'EOF'
import pathlib
import sys

import clifpy

clif = pathlib.Path(sys.argv[1]).resolve()
co = clifpy.ClifOrchestrator(
    data_directory=str(clif),
    filetype="parquet",
    timezone="UTC",
    output_directory=str(clif.parent / "clifpy"),  # for clifpy's logs
)

# load every vital first: otherwise the dose conversions load only weights, and
# sofa then finds no blood pressure
co.load_table("vitals")

co.load_table("respiratory_support")
co.respiratory_support.waterfall().df.to_parquet(
    clif / "clif_respiratory_support_processed.parquet"
)

co.convert_dose_units_for_continuous_meds(
    preferred_units={
        "acetaminophen": "mg/min", "albumin_infusion": "ml/hr",
        "alteplase": "mg/hr", "aminocaproic": "g/hr",
        "aminophylline": "mg/kg/hr", "amiodarone": "mg/min",
        "angiotensin": "ng/kg/min", "argatroban": "mcg/kg/min",
        "bivalirudin": "mg/kg/hr", "bumetanide": "mg/hr",
        "cisatracurium": "mg/kg/hr", "clevidipine": "mg/hr",
        "dexmedetomidine": "mcg/kg/hr", "dextrose_5_water": "ml/hr",
        "dextrose_10_water": "ml/hr", "dextrose_other": "g/hr",
        "diltiazem": "mg/hr", "dobutamine": "mcg/kg/min",
        "dopamine": "mcg/kg/min", "epinephrine": "mcg/kg/min",
        "epoprostenol": "ng/kg/min", "eptifibatide": "mcg/kg/min",
        "esmolol": "mcg/kg/min", "fentanyl": "mcg/hr", "furosemide": "mg/hr",
        "heparin": "u/hr", "hydromorphone": "mg/hr", "insulin": "u/hr",
        "ketamine": "mg/kg/hr", "labetalol": "mg/min", "lidocaine": "mg/min",
        "lorazepam": "mg/hr", "magnesium": "g/hr", "midazolam": "mg/hr",
        "milrinone": "mcg/kg/min", "morphine": "mg/hr", "naloxone": "mg/hr",
        "nicardipine": "mcg/kg/min", "nitroglycerin": "mcg/kg/min",
        "nitroprusside": "mcg/kg/min", "norepinephrine": "mcg/kg/min",
        "octreotide": "mcg/hr", "pantoprazole": "mg/hr",
        "pentobarbital": "mg/kg/hr", "phenylephrine": "mcg/kg/min",
        "procainamide": "mg/min", "propofol": "mcg/kg/min",
        "rocuronium": "mcg/kg/min", "sodium chloride": "ml/hr", "tpn": "ml/hr",
        "treprostinil": "ng/kg/min", "vasopressin": "u/hr",
        "vecuronium": "mg/kg/hr",
    },
    override=True,
)
co.medication_admin_continuous.df_converted.to_parquet(
    clif / "clif_medication_admin_continuous_converted.parquet"
)

co.convert_dose_units_for_intermittent_meds(
    preferred_units={
        "acetaminophen": "mg", "adenosine": "mg", "amiodarone": "mg",
        "bumetanide": "mg", "cisatracurium": "mg", "dextrose_5_water": "ml",
        "dextrose_10_water": "ml", "dextrose_other": "ml", "diazepam": "mg",
        "diltiazem": "mg", "epinephrine": "mg", "fentanyl": "mcg",
        "furosemide": "mg", "heparin": "u", "hydromorphone": "mg",
        "insulin": "u", "ketamine": "mcg", "labetalol": "mg", "lidocaine": "mg",
        "lorazepam": "mg", "magnesium": "g", "midazolam": "mg", "morphine": "mg",
        "naloxone": "mg", "propofol": "mg", "rocuronium": "mg",
        "sodium bicarbonate": "ml", "sodium chloride": "ml", "vecuronium": "mg",
        # mostly charted as "1 dose", which clifpy can't convert, but those
        # charted by mass then convert
        **dict.fromkeys(
            ["acyclovir", "amikacin", "ampicillin", "ampicillin_sulbactam",
             "azithromycin", "aztreonam", "caspofungin", "cefazolin", "cefepime",
             "ceftaroline", "ceftazidime", "ceftriaxone", "ciprofloxacin",
             "clindamycin", "colistin", "daptomycin", "doxycycline", "ertapenem",
             "erythromycin", "esomeprazole", "fluconazole", "foscarnet",
             "gentamicin", "imipenem", "levofloxacin", "linezolid", "meropenem",
             "metronidazole", "micafungin", "moxifloxacin", "nafcillin",
             "pantoprazole", "penicillin", "piperacillin_tazobactam", "rifampin",
             "tigecycline", "tobramycin", "vancomycin", "voriconazole"],
            "mg",
        ),
    },
    override=True,
)
co.medication_admin_intermittent.df_converted.to_parquet(
    clif / "clif_medication_admin_intermittent_converted.parquet"
)

# a row for each encounter_block, clifpy's name for a stay: the hospitalizations
# of a patient at most 6 hours apart, stitched together
sofa = co.compute_sofa_scores()
last = (
    co.wide_df.sort_values("event_time")
    .groupby("encounter_block", as_index=False)
    .agg(
        event_time=("event_time", "max"),
        hospitalization_id=("hospitalization_id", "last"),
    )
)
last.merge(sofa, on="encounter_block", validate="1:1").to_parquet(
    clif / "clif_sofa.parquet", index=False
)
EOF

uv run --isolated --no-project --managed-python --python 3.12 \
    --exclude-newer 2026-10-14T16:00:00Z --with clifpy==2.1.0 \
    python clifpy_tables.py CLIF-MIMIC/output/rclif-2.1
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
and `CLIF-MIMIC/output/rclif-2.1/` then holds 17 files. If you rerun the ETL,
rerun this step too.

### 4. Collate

With every table in place, cocoa's shipped collation config runs as is. `-r`
names the directory of tables to read, and `-p` the directory to write to, which
the later steps read from:

```sh
cocoa collate -r CLIF-MIMIC/output/rclif-2.1 -p processed --verbose
```

This writes `processed/meds.parquet`, the 275,648 events of the 275
hospitalizations, a row each with its `subject_id`, `time`, `code` (such as
`LAB-RES//sodium`), and any value; the shipped config makes each hospitalization
a subject (`subject_id: hospitalization_id`). It also writes
`processed/subject_splits.parquet`, which assigns patients, with all of their
hospitalizations, to splits in the order of their first admissions: the first 70
to train, the next 10 to tuning, and the last 20 to held out. Since each
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
cocoa tokenize -p processed --verbose
```

The tokenizer learns its vocabulary, and the deciles of each code's numeric
values, on the 187 training hospitalizations, and applies both, frozen, to the
rest. With the shipped config, each event becomes one token: its code, with the
decile of its value appended when it has one, as in `LAB-RES//sodium_Q5`. It
writes `processed/tokenizer.yaml`, the vocabulary and bin edges, and
`processed/tokens_times.parquet`, a row for each hospitalization holding its
tokens, from `BOS` to `EOS`, and their times. `--verbose` prints the timelines'
lengths, from 10 tokens to 11,461 with a median of 339, and a few timelines token
by token, and ends with:

```
✓ Tokenization completed in 0.72s.
  Output: …/mimic-iv-demo/processed/tokens_times.parquet
  Output: …/mimic-iv-demo/processed/tokenizer.yaml
  Vocabulary size: 1100 tokens
```

### 6. Winnow, and look at a timeline

`cocoa winnow` readies timelines for evaluating a model. It cuts each one 24
hours after its first event into a past and a future, keeps those that run past
the cut, and flags which outcome tokens (`XFR-IN//icu`, `RESP//imv`,
`DSCG//expired`, `ASMT//cam_total_yes`, and the `LABEL//` tokens) fall on each
side. It writes `processed/train_for_inference.parquet`,
`processed/tuning_for_inference.parquet`, and
`processed/held_out_for_inference.parquet`, keeping 169, 30, and 50 of the 187,
33, and 55 hospitalizations, and `--verbose` prints how often each outcome falls
on each side:

```sh
cocoa winnow -p processed --verbose
```

Then draw one timeline:

```sh
cocoa visualize 27708593 -p processed --show-winnowing --open
```

`cocoa visualize` serves the page until you press Ctrl-C; to keep it as a file
instead, pass `--export-html timeline.html` (or `--export-pdf timeline.pdf`).
Hospitalization 27708593 is the earliest admission in the held-out split, a
week-long elective neurosurgical stay with 17 hours in the surgical ICU. The page
draws its events in lanes by kind, and `--show-winnowing` marks the cut and sums
up, near the bottom, which outcomes fall on each side: here the ICU transfer, a
first hypotension and tachycardia, and `sepsis_onset` all come after it.

### Next steps

- `cocoa pipeline -r CLIF-MIMIC/output/rclif-2.1 -p processed` collates,
  tokenizes, and winnows in one go (steps 4 to 6, short of drawing), and takes
  config overrides, each key prefixed by its stage: `collation.`,
  `tokenization.`, or `winnowing.`.
- Clock tokens mark every fourth hour (`CLCK//00`, `CLCK//04`, and so on) on the
  clock of the collation config's `default_timezone`, which the shipped config
  sets to `UTC`. MIMIC-IV's times are Boston's, so for local hours, run the
  pipeline again into a new directory:
  `cocoa pipeline -r CLIF-MIMIC/output/rclif-2.1 -p processed-local collation.default_timezone=America/New_York tokenization.insert_clocks=true`.
  [Date-based generative inference](date-based-generative-inference.md) uses such
  clock tokens to cut timelines at midnight.
- To change what's collated or how it's tokenized, copy a shipped config from
  [src/cocoa/config](https://github.com/bbj-lab/cocoa/tree/master/src/cocoa/config),
  edit it, and pass it with `-c`;
  [configure a new event token](configure-new-token.md) walks through adding an
  entry.
- [Tokenizer transfer](tokenizer-transfer.md) applies the vocabulary learned here
  to another dataset.
- To train a model on the timelines, see
  [🦜 cotorra](https://pypi.org/project/cotorra/).
