#!/usr/bin/env python3

"""the mimic-iv demo from clifpy: `cocoa get-demo` and cocoa.demo"""

import collections
import contextlib
import importlib.metadata
import importlib.resources
import importlib.util
import logging
import re
import sys

import polars as pl
import pytest
from conftest import plain
from polars.testing import assert_frame_equal
from typer.testing import CliRunner

from cocoa.cli import app
from cocoa.collator import Collator
from cocoa.demo import DERIVED, copy_tables, import_clifpy, write_demo

CLI = CliRunner()

Run = collections.namedtuple("Run", "path result")


def run(*args):
    """invoke the cli in-process; exceptions are captured on the result"""
    return CLI.invoke(app, [str(a) for a in args])


def squashed(text: str) -> str:
    """text with all whitespace dropped: rich wraps long paths mid-token"""
    return re.sub(r"\s+", "", plain(text))


def installed_clifpy():
    """clifpy, or else a skip if it isn't installed; one that won't import fails"""
    if importlib.util.find_spec("clifpy") is None:
        pytest.skip("clifpy isn't installed; cocoa's demo extra installs it")
    import clifpy

    return clifpy


def clifpy_modules() -> list[str]:
    return [m for m in sys.modules if m == "clifpy" or m.startswith("clifpy.")]


@pytest.fixture
def fake_clifpy(tmp_path, monkeypatch):
    """have `import clifpy` find a stand-in package, made by the function returned"""
    saved = {m: sys.modules.pop(m) for m in clifpy_modules()}
    root = tmp_path / "site"

    def make(source: str = "", tables: dict[str, pl.DataFrame] | None = None):
        demo = root / "clifpy" / "data" / "clif_demo"
        demo.mkdir(parents=True)
        (root / "clifpy" / "__init__.py").write_text(source)
        for name, df in (tables or {}).items():
            df.write_parquet(demo / f"{name}.parquet")
        monkeypatch.syspath_prepend(str(root))

    yield make
    for m in clifpy_modules():
        del sys.modules[m]
    sys.modules.update(saved)


def test_get_demo_without_clifpy_says_how_to_install_it(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "clifpy", None)  # so importing it fails
    dest = tmp_path / "demo"
    result = run("get-demo", "-r", dest)
    assert result.exit_code == 1
    assert squashed("pip install 'cocoa-tokenizer[demo]'") in squashed(result.output)
    assert not dest.exists()


def test_a_module_clifpy_needs_and_lacks_is_named_as_missing(tmp_path, fake_clifpy):
    fake_clifpy("import a_module_clifpy_needs\n")
    result = run("get-demo", "-r", tmp_path / "demo")
    assert result.exit_code == 1
    assert isinstance(result.exception, ModuleNotFoundError)
    assert result.exception.name == "a_module_clifpy_needs"


def test_a_clifpy_without_demo_tables_is_refused(tmp_path, fake_clifpy):
    fake_clifpy()
    dest = tmp_path / "demo"
    with pytest.raises(FileNotFoundError, match="No demo tables"):
        write_demo(dest)
    assert not dest.exists()


def test_copies_drop_the_index_that_pandas_wrote_in(tmp_path, fake_clifpy):
    patient = pl.DataFrame({"patient_id": ["1", "2"], "sex_category": ["F", "M"]})
    fake_clifpy(
        tables={
            "clif_patient": patient.with_columns(__index_level_0__=pl.int_range(2)),
            "clif_labs": pl.DataFrame({"hospitalization_id": ["3"]}),
            "not_a_clif_table": pl.DataFrame({"x": [0]}),
        }
    )
    written = copy_tables(import_clifpy(), tmp_path / "demo")
    assert [p.name for p in written] == ["clif_labs.parquet", "clif_patient.parquet"]
    assert_frame_equal(pl.read_parquet(written[1]), patient)


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> Run:
    """the demo, as `cocoa get-demo` writes it, run from an empty directory"""
    installed_clifpy()
    cwd = tmp_path_factory.mktemp("cwd")
    with contextlib.chdir(cwd):
        result = run("get-demo", "-r", "data-raw/demo")
    assert result.exit_code == 0, result.output
    return Run(cwd / "data-raw" / "demo", result)


@pytest.fixture(scope="module")
def bundled() -> dict[str, pl.DataFrame]:
    """the tables of clifpy's demo, as clifpy ships them"""
    clifpy = installed_clifpy()
    shipped = importlib.resources.files(clifpy) / "data" / "clif_demo"
    with importlib.resources.as_file(shipped) as demo:
        return {p.name: pl.read_parquet(p) for p in demo.glob("clif_*.parquet")}


def test_get_demo_writes_clifpy_tables_and_those_derived(demo, bundled):
    derived = {f"{name}.parquet" for name in DERIVED}
    assert {p.name for p in demo.path.iterdir()} == set(bundled) | derived
    out = squashed(demo.result.output)
    assert squashed(f"Output: {demo.path}") in out
    assert squashed(f"({len(bundled) + 4} tables") in out
    assert squashed(f"clifpy {importlib.metadata.version('clifpy')})") in out


def test_get_demo_copies_clifpy_tables_without_their_pandas_index(demo, bundled):
    for name, df in bundled.items():
        assert_frame_equal(
            pl.read_parquet(demo.path / name),
            df.drop("__index_level_0__", strict=False),
        )


def test_get_demo_leaves_no_logs_behind_nor_any_open(demo):
    assert not (demo.path.parents[1] / "output").exists()  # clifpy's default
    assert {p.suffix for p in demo.path.iterdir()} == {".parquet"}
    # a log file left open would keep windows from removing its directory
    assert not [
        h
        for name in list(logging.root.manager.loggerDict)
        if name == "clifpy" or name.startswith("clifpy.")
        for h in logging.getLogger(name).handlers
        if isinstance(h, logging.FileHandler)
    ]


def test_quiet_get_demo_keeps_clifpy_s_output_to_itself(demo):
    assert "ClifOrchestrator" not in demo.result.output


def test_verbose_get_demo_passes_clifpy_s_output_through(tmp_path):
    installed_clifpy()
    result = run("get-demo", "-r", tmp_path / "demo", "--verbose")
    assert result.exit_code == 0, result.output
    assert "ClifOrchestrator" in result.output


@pytest.mark.parametrize("kind", ["continuous", "intermittent"])
def test_each_medication_s_converted_doses_share_one_unit(demo, kind):
    df = pl.read_parquet(demo.path / f"clif_medication_admin_{kind}_converted.parquet")
    converted = df.filter(pl.col("_convert_status") == "success")
    assert converted.height > df.height / 2
    units = converted.group_by("med_category").agg(
        pl.col("med_dose_unit_converted").n_unique()
    )
    assert units.filter(pl.col("med_dose_unit_converted") > 1).height == 0


def test_respiratory_support_gains_the_rows_of_the_waterfall(demo):
    raw = pl.read_parquet(demo.path / "clif_respiratory_support.parquet")
    processed = pl.read_parquet(
        demo.path / "clif_respiratory_support_processed.parquet"
    )
    assert processed.height > raw.height
    assert set(processed["hospitalization_id"]) == set(raw["hospitalization_id"])


def test_sofa_scores_each_stay_once_on_one_of_its_hospitalizations(demo):
    sofa = pl.read_parquet(demo.path / "clif_sofa.parquet")
    hosp = pl.read_parquet(demo.path / "clif_hospitalization.parquet")
    assert sofa["encounter_block"].is_unique().all()
    assert set(sofa["hospitalization_id"]) <= set(hosp["hospitalization_id"])
    assert sofa["sofa_total"].null_count() == 0
    assert sofa["event_time"].dtype.time_zone == "UTC"


def test_the_shipped_collation_config_collates_the_demo(demo, tmp_path):
    Collator(raw_data_home=demo.path, processed_data_home=tmp_path).save_all()
    hosp = pl.read_parquet(demo.path / "clif_hospitalization.parquet")
    meds = pl.read_parquet(tmp_path / "meds.parquet")
    assert set(meds["subject_id"]) == set(hosp["hospitalization_id"])
