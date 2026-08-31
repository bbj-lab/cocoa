#!/usr/bin/env python3

"""the typer cli: help, stage invocations, config overrides, verbose output"""

import collections
import importlib.metadata
import pathlib
import re
import shutil
import subprocess
import sys

import polars as pl
import pytest
from conftest import default_cfg, write_cfg
from omegaconf import OmegaConf
from polars.testing import assert_frame_equal
from typer.testing import CliRunner

import cocoa.cli
from cocoa.cli import app

COMMANDS = ("collate", "tokenize", "winnow", "pipeline", "combine-datasets")

ARTIFACTS = (
    "meds.parquet",
    "subject_splits.parquet",
    "tokens_times.parquet",
    "tokenizer.yaml",
    "train_for_inference.parquet",
    "tuning_for_inference.parquet",
    "held_out_for_inference.parquet",
)
TOKENIZED = ARTIFACTS[:4]
TOKEN_COLS = ("tokens", "tokens_past", "tokens_future")

COCOA_EXE = pathlib.Path(sys.executable).parent / "cocoa"

CLI = CliRunner()

Run = collections.namedtuple("Run", "path result")


def run(*args):
    """invoke the cli in-process; exceptions are captured on the result"""
    return CLI.invoke(app, [str(a) for a in args])


def squashed(text: str) -> str:
    """text with all whitespace dropped: rich wraps long paths mid-token"""
    return re.sub(r"\s+", "", text)


def prefixes(meds: pl.DataFrame) -> set:
    """the code prefixes present in a collated frame"""
    return set(meds["code"].str.split("//").list.first().to_list())


def quantile_indices(vocab) -> set:
    """the Q indices appearing in a fused vocabulary"""
    return {int(m.group(1)) for w in vocab if (m := re.search(r"_Q(\d+)$", w))}


def lookup_of(processed) -> dict:
    """the vocabulary recorded in a processed dir's tokenizer.yaml"""
    return dict(OmegaConf.load(pathlib.Path(processed) / "tokenizer.yaml").lookup)


def bins_of(processed) -> dict:
    """the bin break points recorded in a processed dir's tokenizer.yaml"""
    cfg = OmegaConf.load(pathlib.Path(processed) / "tokenizer.yaml").bins
    return {k: list(v) for k, v in dict(cfg).items()}


def multisets(df: pl.DataFrame, col: str) -> dict:
    """subject_id -> sorted token multiset of a list-of-token column"""
    return {
        s: (None if t is None else sorted(t))
        for s, t in zip(df["subject_id"].to_list(), df[col].to_list())
    }


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory, raw_data) -> Run:
    """one `cocoa pipeline` invocation, shared by tests that only read it"""
    dest = tmp_path_factory.mktemp("cli_pipeline")
    result = run("pipeline", "-r", raw_data.root, "-p", dest)
    assert result.exit_code == 0, result.output
    return Run(dest, result)


@pytest.fixture(scope="module")
def stages_run(tmp_path_factory, raw_data) -> pathlib.Path:
    """the same work as `pipeline_run`, done by three separate invocations"""
    dest = tmp_path_factory.mktemp("cli_stages")
    for args in (
        ("collate", "-r", raw_data.root, "-p", dest),
        ("tokenize", "-p", dest),
        ("winnow", "-p", dest),
    ):
        result = run(*args)
        assert result.exit_code == 0, result.output
    return dest


@pytest.fixture
def tokenized(tmp_path, pipeline_run) -> pathlib.Path:
    """a writable copy of the collated + tokenized artifacts, per test"""
    dest = tmp_path / "tokenized"
    dest.mkdir()
    for f in TOKENIZED:
        shutil.copy(pipeline_run.path / f, dest / f)
    return dest


def test_cli_version_matches_installed_package_metadata():
    assert cocoa.cli.__version__ == importlib.metadata.version("cocoa-tokenizer")
    # calver YY.M.patch
    assert re.fullmatch(r"\d{2}\.\d{1,2}\.\d+", cocoa.cli.__version__)


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_top_level_help_lists_every_command_and_the_version(flag):
    result = run(flag)
    assert result.exit_code == 0
    out = squashed(result.output)
    for command in COMMANDS:
        assert command in out
    assert f"v{cocoa.cli.__version__}" in out


HELP_TEXT = {
    "collate": (
        ("Collate raw data", "denormalized", "--collation-config", "--raw-data-home"),
        ("--tokenizer-home", "--winnowing-config"),
    ),
    "tokenize": (
        ("Tokenize collated data", "vocabulary", "--tokenizer-home", "-t"),
        ("--raw-data-home", "--collation-config"),
    ),
    "winnow": (
        ("Winnow held-out data", "--winnowing-config", "--processed-data-home"),
        ("--raw-data-home", "--tokenizer-home"),
    ),
    "pipeline": (
        (
            "collate, tokenize, & winnow",
            "--collation-config",
            "--tokenization-config",
            "--winnowing-config",
            "--raw-data-home",
        ),
        ("--tokenizer-home",),
    ),
    "combine-datasets": (
        ("Combine multiple processed datasets", "--output-data-dir", "-o"),
        ("--processed-data-home", "--raw-data-home"),
    ),
}


@pytest.mark.parametrize("flag", ["-h", "--help"])
@pytest.mark.parametrize("command", COMMANDS)
def test_command_help_shows_its_own_options(command, flag):
    expected, unexpected = HELP_TEXT[command]
    result = run(command, flag)
    assert result.exit_code == 0
    out = squashed(result.output)
    assert squashed(f"Usage: cocoa {command}") in out
    for needle in expected:
        assert squashed(needle) in out, needle
    for needle in unexpected:
        assert squashed(needle) not in out, needle


def test_collate_writes_artifacts_and_prints_both_paths(runner):
    dest = runner.root / "not_yet_created"  # the cli should make it
    result = run("collate", "-r", runner.raw.root, "-p", dest)
    assert result.exit_code == 0, result.output
    assert (dest / "meds.parquet").exists()
    assert (dest / "subject_splits.parquet").exists()
    out = squashed(result.output)
    assert "meds.parquet" in out
    assert "subject_splits.parquet" in out
    meds = pl.read_parquet(dest / "meds.parquet")
    splits = pl.read_parquet(dest / "subject_splits.parquet")
    assert meds.height > 0
    assert set(meds["subject_id"].to_list()) == set(runner.raw.subject_ids)
    assert sorted(splits["subject_id"].to_list()) == sorted(runner.raw.subject_ids)


def test_tokenize_writes_artifacts_and_prints_the_vocabulary_size(runner):
    dest = runner.seed_collated()
    result = run("tokenize", "-p", dest)
    assert result.exit_code == 0, result.output
    assert (dest / "tokens_times.parquet").exists()
    assert (dest / "tokenizer.yaml").exists()
    vocab = lookup_of(dest)
    assert len(vocab) > 1
    out = squashed(result.output)
    assert "tokens_times.parquet" in out
    assert "tokenizer.yaml" in out
    assert squashed(f"Vocabulary size: {len(vocab)} tokens") in out
    tokens_times = pl.read_parquet(dest / "tokens_times.parquet")
    assert tokens_times.height == len(runner.raw.subject_ids)
    assert tokens_times["tokens"].explode().max() < len(vocab)


def test_winnow_writes_one_file_per_configured_split_and_prints_paths(
    tokenized, raw_data
):
    splits = default_cfg("winnowing")["splits"]
    assert len(splits) == 3  # the shipped default prepares all three
    result = run("winnow", "-p", tokenized)
    assert result.exit_code == 0, result.output
    out = squashed(result.output)
    for split in splits:
        assert (tokenized / f"{split}_for_inference.parquet").exists()
        assert f"{split}_for_inference.parquet" in out
    held_out = pl.read_parquet(tokenized / "held_out_for_inference.parquet")
    assert held_out.height > 0
    assert set(held_out["subject_id"].to_list()) <= set(
        raw_data.subjects_in_split("held_out")
    )


def test_pipeline_produces_every_artifact(pipeline_run, raw_data):
    for f in ARTIFACTS:
        assert (pipeline_run.path / f).exists(), f
    out = squashed(pipeline_run.result.output)
    for stage in ("Collation completed", "Tokenization completed", "Winnowing"):
        assert squashed(stage) in out
    assert squashed("Pipeline completed") in out
    tokens_times = pl.read_parquet(pipeline_run.path / "tokens_times.parquet")
    assert set(tokens_times["subject_id"].to_list()) == set(raw_data.subject_ids)


def test_pipeline_matches_three_separate_invocations(pipeline_run, stages_run):
    one, three = pipeline_run.path, stages_run

    meds = [pl.read_parquet(d / "meds.parquet") for d in (one, three)]
    assert meds[0].height > 0
    assert_frame_equal(*(m.sort(m.columns) for m in meds))

    splits = [pl.read_parquet(d / "subject_splits.parquet") for d in (one, three)]
    assert splits[0].height > 0
    assert_frame_equal(*(s.sort("subject_id") for s in splits))

    assert lookup_of(one) == lookup_of(three)
    assert bins_of(one) == bins_of(three)

    tt = [
        pl.read_parquet(d / "tokens_times.parquet").sort("subject_id")
        for d in (one, three)
    ]
    assert tt[0].height > 0
    assert tt[0]["subject_id"].to_list() == tt[1]["subject_id"].to_list()
    assert tt[0]["times"].to_list() == tt[1]["times"].to_list()
    # token *order* only agrees up to permutation within a (time, priority) tie:
    # Tokenizer.tokenize_data sorts unstably, so two runs over identical input
    # emit different sequences (reported as a bug, not asserted as desirable)
    assert multisets(tt[0], "tokens") == multisets(tt[1], "tokens")

    for split in ("train", "tuning", "held_out"):
        inf = [
            pl.read_parquet(d / f"{split}_for_inference.parquet").sort("subject_id")
            for d in (one, three)
        ]
        assert inf[0].height > 0
        assert_frame_equal(*(i.drop(TOKEN_COLS) for i in inf))
        for col in TOKEN_COLS:
            assert multisets(inf[0], col) == multisets(inf[1], col)


@pytest.mark.parametrize("flag", ["-c", "--collation-config"])
def test_collation_config_override_limits_the_collated_entries(runner, flag):
    cfg = default_cfg("collation")
    assert len(prefixes_wanted := {"SEX", "VTL"}) < len(
        {e["prefix"] for e in cfg["entries"]}
    )  # the default collates strictly more than we ask for below
    cfg["entries"] = [e for e in cfg["entries"] if e["prefix"] in prefixes_wanted]
    cfg_path = write_cfg(runner.root / "reduced_collation.yaml", cfg)
    dest = runner.dir()
    result = run("collate", flag, cfg_path, "-r", runner.raw.root, "-p", dest)
    assert result.exit_code == 0, result.output
    meds = pl.read_parquet(dest / "meds.parquet")
    assert prefixes(meds) == prefixes_wanted
    # one SEX event per subject, plus every vitals row carrying a time
    vitals = pl.read_parquet(runner.raw.root / "clif_vitals.parquet")
    assert (
        meds.height
        == len(runner.raw.subject_ids)
        + vitals.filter(pl.col("recorded_dttm").is_not_null()).height
    )


@pytest.mark.parametrize("flag", ["-c", "--tokenization-config"])
def test_tokenization_config_override_changes_the_number_of_bins(
    runner, tokenized, pipeline, flag
):
    assert max(quantile_indices(pipeline.vocab)) == 9  # default n_bins=10
    cfg = default_cfg("tokenization")
    cfg["n_bins"] = 4
    cfg_path = write_cfg(runner.root / "four_bins.yaml", cfg)
    result = run("tokenize", flag, cfg_path, "-p", tokenized)
    assert result.exit_code == 0, result.output
    bins = bins_of(tokenized)
    assert len(bins) > 0
    assert {len(breaks) for breaks in bins.values()} == {3}  # n_bins - 1
    assert max(quantile_indices(lookup_of(tokenized))) == 3


@pytest.mark.parametrize("flag", ["-c", "--winnowing-config"])
def test_winnowing_config_override_limits_the_prepared_splits(runner, tokenized, flag):
    cfg = default_cfg("winnowing")
    cfg["splits"] = ["held_out"]
    cfg_path = write_cfg(runner.root / "held_out_only.yaml", cfg)
    result = run("winnow", flag, cfg_path, "-p", tokenized)
    assert result.exit_code == 0, result.output
    assert (tokenized / "held_out_for_inference.parquet").exists()
    assert not (tokenized / "train_for_inference.parquet").exists()
    assert not (tokenized / "tuning_for_inference.parquet").exists()
    out = squashed(result.output)
    assert "held_out_for_inference.parquet" in out
    assert "train_for_inference.parquet" not in out
    assert "tuning_for_inference.parquet" not in out


def test_pipeline_honours_all_three_config_overrides(runner, raw_data):
    collation = default_cfg("collation")
    collation["subject_splits"] = {"train_frac": 0.5, "tuning_frac": 0.25}
    tokenization = default_cfg("tokenization")
    tokenization["n_bins"] = 5
    winnowing = default_cfg("winnowing")
    winnowing["splits"] = ["held_out"]
    paths = [
        write_cfg(runner.root / f"pipeline_{name}.yaml", cfg)
        for name, cfg in (
            ("collation", collation),
            ("tokenization", tokenization),
            ("winnowing", winnowing),
        )
    ]
    dest = runner.dir()
    result = run(
        "pipeline",
        "--collation-config",
        paths[0],
        "--tokenization-config",
        paths[1],
        "--winnowing-config",
        paths[2],
        "-r",
        raw_data.root,
        "-p",
        dest,
    )
    assert result.exit_code == 0, result.output

    # half the patients train, a quarter tuning, the rest held out, assigned
    # chronologically; counted here in subjects
    expected = collections.Counter()
    for i, patient in enumerate(raw_data.patient_ids):
        n = len(raw_data.patient_ids)
        split = "train" if i < n // 2 else "tuning" if i < 3 * n // 4 else "held_out"
        expected[split] += len(raw_data.subjects_of[patient])
    splits = pl.read_parquet(dest / "subject_splits.parquet")
    assert collections.Counter(splits["split"].to_list()) == expected

    assert max(quantile_indices(lookup_of(dest))) == 4  # n_bins=5
    assert (dest / "held_out_for_inference.parquet").exists()
    assert not (dest / "train_for_inference.parquet").exists()


@pytest.mark.parametrize("flag", ["-t", "--tokenizer-home"])
def test_tokenizer_home_reuses_the_learned_vocabulary(runner, pipeline_run, flag):
    """a transferred tokenizer keeps its frozen vocabulary on a new dataset"""
    cfg = default_cfg("collation")
    cfg["entries"] = [e for e in cfg["entries"] if e["prefix"] in {"SEX", "VTL", "AGE"}]
    cfg_path = write_cfg(runner.root / "subset_collation.yaml", cfg)
    subset = runner.dir()
    assert (
        run("collate", "-c", cfg_path, "-r", runner.raw.root, "-p", subset).exit_code
        == 0
    )
    transferred = runner.dir()
    for f in TOKENIZED[:2]:
        shutil.copy(subset / f, transferred / f)

    assert run("tokenize", "-p", subset).exit_code == 0  # learns its own vocabulary
    source = lookup_of(pipeline_run.path)
    assert 0 < len(lookup_of(subset)) < len(source)

    result = run(
        "tokenize", flag, pipeline_run.path / "tokenizer.yaml", "-p", transferred
    )
    assert result.exit_code == 0, result.output
    assert lookup_of(transferred) == source
    assert bins_of(transferred) == bins_of(pipeline_run.path)
    out = squashed(result.output)
    assert squashed("Using pretrained tokenizer") in out
    assert squashed(f"Vocabulary size: {len(source)} tokens") in out


@pytest.mark.parametrize("flag", ["-o", "--output-data-dir"])
def test_combine_datasets_merges_two_processed_dirs(tmp_path, pipeline_run, flag):
    inputs = []
    for name in ("first", "second"):
        d = tmp_path / name
        shutil.copytree(pipeline_run.path, d)
        inputs.append(d)
    dest = tmp_path / "combined"
    result = run("combine-datasets", *inputs, flag, dest)
    assert result.exit_code == 0, result.output
    out = squashed(result.output)
    assert squashed("Combine completed") in out
    assert squashed("Output at:") in out
    assert "combined" in out
    for f in ARTIFACTS:
        assert (dest / f).exists(), f
    for f in ("meds.parquet", "tokens_times.parquet", "held_out_for_inference.parquet"):
        before = pl.read_parquet(pipeline_run.path / f)
        after = pl.read_parquet(dest / f)
        assert before.height > 0
        assert after.height == 2 * before.height
    subjects = pl.read_parquet(dest / "tokens_times.parquet")["subject_id"].to_list()
    assert collections.Counter(subjects) == {
        s: 2
        for s in pl.read_parquet(pipeline_run.path / "tokens_times.parquet")[
            "subject_id"
        ].to_list()
    }
    assert lookup_of(dest) == lookup_of(pipeline_run.path)


@pytest.mark.parametrize("flag", ["-v", "--verbose"])
def test_verbose_collate_reports_row_and_subject_counts(runner, flag):
    dest = runner.dir()
    result = run("collate", flag, "-r", runner.raw.root, "-p", dest)
    assert result.exit_code == 0, result.output
    meds = pl.read_parquet(dest / "meds.parquet")
    assert meds.height > 0
    out = squashed(result.output)
    assert squashed(f"total rows: {meds.height}") in out
    assert squashed(f"unique subjects: {len(runner.raw.subject_ids)}") in out
    assert squashed("subjects by split") in out
    assert squashed("rows by split") in out
    for split in ("train", "tuning", "held_out"):
        assert split in out


def test_quiet_collate_omits_the_summary_statistics(runner):
    dest = runner.dir()
    result = run("collate", "-r", runner.raw.root, "-p", dest)
    assert result.exit_code == 0, result.output
    out = squashed(result.output)
    assert squashed("total rows:") not in out
    assert squashed("subjects by split") not in out


@pytest.mark.parametrize("flag", ["-v", "--verbose"])
def test_verbose_tokenize_reports_timeline_statistics(tokenized, flag):
    result = run("tokenize", flag, "-p", tokenized)
    assert result.exit_code == 0, result.output
    n_subjects = pl.read_parquet(tokenized / "tokens_times.parquet").height
    assert n_subjects > 0
    out = squashed(result.output)
    assert squashed(f"total rows: {n_subjects}") in out
    assert squashed("timeline length stats") in out
    assert squashed("timeline duration stats") in out
    assert squashed("split-level info") in out
    assert squashed("example timeline") in out


@pytest.mark.parametrize("flag", ["-v", "--verbose"])
def test_verbose_winnow_reports_outcome_rates_per_split(tokenized, flag):
    result = run("winnow", flag, "-p", tokenized)
    assert result.exit_code == 0, result.output
    out = squashed(result.output)
    for split in default_cfg("winnowing")["splits"]:
        assert squashed(f"Prepared split {split} for inference") in out
    for token in ("DSCG//expired", "RESP//imv", "XFR-IN//icu"):
        assert token in out
    assert "past" in out and "future" in out


@pytest.mark.parametrize("flag", ["-v", "--verbose"])
def test_verbose_pipeline_reports_statistics_for_every_stage(runner, flag):
    dest = runner.dir()
    result = run("pipeline", flag, "-r", runner.raw.root, "-p", dest)
    assert result.exit_code == 0, result.output
    out = squashed(result.output)
    assert squashed(f"unique subjects: {len(runner.raw.subject_ids)}") in out
    assert squashed("timeline length stats") in out
    assert squashed("Prepared split held_out for inference") in out


@pytest.mark.parametrize(
    "args,missing",
    [
        (("collate", "-r", "{raw}"), "--processed-data-home"),
        (("collate", "-p", "{out}"), "--raw-data-home"),
        (("tokenize",), "--processed-data-home"),
        (("winnow",), "--processed-data-home"),
        (("pipeline", "-r", "{raw}"), "--processed-data-home"),
        (("combine-datasets", "{out}"), "--output-data-dir"),
    ],
)
def test_missing_required_option_exits_nonzero(runner, args, missing):
    out_dir = runner.dir()
    result = run(*(a.format(raw=runner.raw.root, out=out_dir) for a in args))
    assert result.exit_code == 2
    assert squashed(f"Missing option '{missing}'") in squashed(result.output)
    assert not list(out_dir.iterdir())


def test_unknown_command_exits_nonzero():
    result = run("frobnicate")
    assert result.exit_code == 2
    assert squashed("No such command") in squashed(result.output)


def test_nonexistent_raw_data_home_surfaces_an_error(runner):
    dest = runner.dir()
    result = run("collate", "-r", runner.root / "no_such_raw_home", "-p", dest)
    assert result.exit_code == 1
    assert isinstance(result.exception, FileNotFoundError)
    assert "clif_hospitalization" in str(result.exception)
    assert not (dest / "meds.parquet").exists()


@pytest.mark.parametrize("command", ["tokenize", "winnow"])
def test_stage_without_its_inputs_surfaces_an_error(runner, command):
    dest = runner.dir()  # empty: nothing has been collated into it
    result = run(command, "-p", dest)
    assert result.exit_code == 1
    assert isinstance(result.exception, FileNotFoundError)
    assert not list(dest.iterdir())


@pytest.mark.parametrize(
    "argv",
    [
        [sys.executable, "-m", "cocoa.cli"],
        pytest.param(
            [str(COCOA_EXE)],
            marks=pytest.mark.skipif(
                not COCOA_EXE.exists(), reason="console script not installed"
            ),
        ),
    ],
    ids=["module", "console-script"],
)
def test_out_of_process_invocation_runs_the_pipeline(tmp_path, raw_data, argv):
    dest = tmp_path / "out_of_process"
    proc = subprocess.run(
        [*argv, "pipeline", "-r", str(raw_data.root), "-p", str(dest)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for f in ARTIFACTS:
        assert (dest / f).exists(), f
    assert squashed("Pipeline completed") in squashed(proc.stdout)
    tokens_times = pl.read_parquet(dest / "tokens_times.parquet")
    assert set(tokens_times["subject_id"].to_list()) == set(raw_data.subject_ids)
