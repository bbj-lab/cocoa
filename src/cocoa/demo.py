#!/usr/bin/env python3

"""
write the mimic-iv demo that clifpy ships, as clif tables, to a raw data directory
"""

import contextlib
import importlib.resources
import io
import logging
import pathlib
import tempfile
import warnings
from collections.abc import Iterator
from types import ModuleType

import polars as pl

# the tables that the shipped collation config reads besides clifpy's demo, which
# clifpy derives from it: respiratory support after its waterfall, doses converted
# to the units below, and sofa scores
DERIVED = (
    "clif_respiratory_support_processed",
    "clif_medication_admin_continuous_converted",
    "clif_medication_admin_intermittent_converted",
    "clif_sofa",
)

# one unit per medication, so that the tokenizer bins each one's doses on a single
# scale: those bbj-lab uses on the full mimic-iv, as in
# https://github.com/bbj-lab/coreopsis/blob/main/recipes/run_clifpy.py
CONTINUOUS_UNITS = {
    "acetaminophen": "mg/min",
    "albumin_infusion": "ml/hr",
    "alteplase": "mg/hr",
    "aminocaproic": "g/hr",
    "aminophylline": "mg/kg/hr",
    "amiodarone": "mg/min",
    "angiotensin": "ng/kg/min",
    "argatroban": "mcg/kg/min",
    "bivalirudin": "mg/kg/hr",
    "bumetanide": "mg/hr",
    "cisatracurium": "mg/kg/hr",
    "clevidipine": "mg/hr",
    "dexmedetomidine": "mcg/kg/hr",
    "dextrose": "ml/hr",
    "dextrose_in_water_d5w": "ml/hr",
    "diltiazem": "mg/hr",
    "dobutamine": "mcg/kg/min",
    "dopamine": "mcg/kg/min",
    "epinephrine": "mcg/kg/min",
    "epoprostenol": "ng/kg/min",
    "eptifibatide": "mcg/kg/min",
    "esmolol": "mcg/kg/min",
    "fentanyl": "mcg/hr",
    "furosemide": "mg/hr",
    "heparin": "u/hr",
    "hydromorphone": "mg/hr",
    "insulin": "u/hr",
    "ketamine": "mg/kg/hr",
    "labetalol": "mg/min",
    "lidocaine": "mg/min",
    "lorazepam": "mg/hr",
    "magnesium": "g/hr",
    "midazolam": "mg/hr",
    "milrinone": "mcg/kg/min",
    "morphine": "mg/hr",
    "naloxone": "mg/hr",
    "nicardipine": "mcg/kg/min",
    "nitroglycerin": "mcg/kg/min",
    "nitroprusside": "mcg/kg/min",
    "norepinephrine": "mcg/kg/min",
    "octreotide": "mcg/hr",
    "pantoprazole": "mg/hr",
    "pentobarbital": "mg/kg/hr",
    "phenylephrine": "mcg/kg/min",
    "procainamide": "mg/min",
    "propofol": "mcg/kg/min",
    "rocuronium": "mcg/kg/min",
    "sodium chloride": "ml/hr",
    "tpn": "ml/hr",
    "treprostinil": "ng/kg/min",
    "vasopressin": "u/hr",
    "vecuronium": "mg/kg/hr",
}

# many intermittent doses are charted as "1 dose", which converts to nothing, but
# asking for mg (or u) converts the ones charted by mass (or units)
INTERMITTENT_UNITS = {
    "acetaminophen": "mg",
    "acyclovir": "mg",
    "adenosine": "mg",
    "amikacin": "mg",
    "amiodarone": "mg",
    "ampicillin": "mg",
    "ampicillin_sulbactam": "mg",
    "azithromycin": "mg",
    "aztreonam": "mg",
    "bumetanide": "mg",
    "caspofungin": "mg",
    "cefazolin": "mg",
    "cefepime": "mg",
    "ceftaroline": "mg",
    "ceftazidime": "mg",
    "ceftriaxone": "mg",
    "ciprofloxacin": "mg",
    "cisatracurium": "mg",
    "clindamycin": "mg",
    "colistin": "mg",
    "daptomycin": "mg",
    "dextrose": "ml",
    "dextrose_in_water_d5w": "ml",
    "diazepam": "mg",
    "diltiazem": "mg",
    "doxycycline": "mg",
    "epinephrine": "mg",
    "ertapenem": "mg",
    "erythromycin": "mg",
    "esomeprazole": "mg",
    "fentanyl": "mcg",
    "fluconazole": "mg",
    "foscarnet": "mg",
    "furosemide": "mg",
    "gentamicin": "mg",
    "heparin": "u",
    "hydromorphone": "mg",
    "imipenem": "mg",
    "insulin": "u",
    "ketamine": "mcg",
    "labetalol": "mg",
    "levofloxacin": "mg",
    "lidocaine": "mg",
    "linezolid": "mg",
    "lorazepam": "mg",
    "magnesium": "g",
    "meropenem": "mg",
    "metronidazole": "mg",
    "micafungin": "mg",
    "midazolam": "mg",
    "morphine": "mg",
    "moxifloxacin": "mg",
    "nafcillin": "mg",
    "naloxone": "mg",
    "pantoprazole": "mg",
    "penicillin": "mg",
    "piperacillin_tazobactam": "mg",
    "propofol": "mg",
    "rifampin": "mg",
    "rocuronium": "mg",
    "sodium bicarbonate": "ml",
    "sodium chloride": "ml",
    "tigecycline": "mg",
    "tobramycin": "mg",
    "vancomycin": "mg",
    "vecuronium": "mg",
    "voriconazole": "mg",
}


def import_clifpy() -> ModuleType:
    """clifpy, or else an error that says how to install it"""
    try:
        import clifpy
    except ModuleNotFoundError as e:
        if e.name != "clifpy":  # clifpy is there, but something it needs isn't
            raise
        raise ModuleNotFoundError(
            "The demo comes from clifpy, which isn't installed; install it with "
            "pip install 'cocoa-tokenizer[demo]'",
            name="clifpy",
        ) from None
    return clifpy


@contextlib.contextmanager
def hushed(verbose: bool) -> Iterator[None]:
    """unless `verbose`, swallow what clifpy prints, logs, and warns"""
    if verbose:
        yield
        return
    # clifpy's console log binds to whatever sys.stdout is when it sets up
    sink = io.StringIO()
    with (
        contextlib.redirect_stdout(sink),
        contextlib.redirect_stderr(sink),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        yield


def release_logs() -> None:
    """close the handlers clifpy logs through, so that their files can go"""
    for name in list(logging.root.manager.loggerDict):
        if name == "clifpy" or name.startswith("clifpy."):
            logger = logging.getLogger(name)
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()


def copy_tables(clifpy: ModuleType, raw: pathlib.Path) -> list[pathlib.Path]:
    """copy the clif tables of clifpy's demo to `raw`"""
    bundled = importlib.resources.files(clifpy) / "data" / "clif_demo"
    with importlib.resources.as_file(bundled) as demo:
        tables = sorted(demo.glob("clif_*.parquet"))
        if not tables:
            raise FileNotFoundError(f"No demo tables found in {demo}")
        raw.mkdir(parents=True, exist_ok=True)
        for table in tables:
            # clifpy 2.1.0 wrote pandas's index into each table as a column
            pl.scan_parquet(table).drop("__index_level_0__", strict=False).sink_parquet(
                raw / table.name
            )
    return [raw / table.name for table in tables]


def derive_tables(
    clifpy: ModuleType, raw: pathlib.Path, log_dir: pathlib.Path | str
) -> list[pathlib.Path]:
    """write the tables in `DERIVED`, from the clif tables in `raw`, to `raw`"""
    co = clifpy.ClifOrchestrator(
        data_directory=str(raw),
        filetype="parquet",
        timezone="UTC",
        output_directory=str(log_dir),
    )
    # load every vital first: otherwise the dose conversions load only weights, and
    # sofa then finds no blood pressure
    co.load_table("vitals")
    co.load_table("respiratory_support")
    derived = {
        "clif_respiratory_support_processed": co.respiratory_support.waterfall().df
    }
    co.convert_dose_units_for_continuous_meds(
        preferred_units=CONTINUOUS_UNITS, override=True
    )
    derived["clif_medication_admin_continuous_converted"] = (
        co.medication_admin_continuous.df_converted
    )
    co.convert_dose_units_for_intermittent_meds(
        preferred_units=INTERMITTENT_UNITS, override=True
    )
    derived["clif_medication_admin_intermittent_converted"] = (
        co.medication_admin_intermittent.df_converted
    )
    # a row for each encounter_block, clifpy's name for a stay: the hospitalizations
    # of a patient at most 6 hours apart, stitched together; each is timed at its
    # stay's last record
    sofa = co.compute_sofa_scores()
    last = (
        co.wide_df.sort_values("event_time")
        .groupby("encounter_block", as_index=False)
        .agg(
            event_time=("event_time", "max"),
            hospitalization_id=("hospitalization_id", "last"),
        )
    )
    derived["clif_sofa"] = last.merge(sofa, on="encounter_block", validate="1:1")
    for name, df in derived.items():
        df.to_parquet(raw / f"{name}.parquet", index=False)
    return [raw / f"{name}.parquet" for name in derived]


def write_demo(
    raw_data_home: pathlib.Path | str, verbose: bool = False
) -> list[pathlib.Path]:
    """
    write the clif tables of the mimic-iv demo that clifpy ships to
    `raw_data_home`, along with the four that clifpy derives from them and the
    shipped collation config also reads; return the paths written
    """
    raw = pathlib.Path(raw_data_home).expanduser().resolve()
    with (
        hushed(verbose),
        tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as log_dir,
    ):
        clifpy = import_clifpy()
        try:
            return copy_tables(clifpy, raw) + derive_tables(clifpy, raw, log_dir)
        finally:
            release_logs()
