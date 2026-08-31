#!/usr/bin/env python3

"""combining processed directories with cocoa.util.combine_processed_data"""

import datetime
import pathlib
import shutil

import polars as pl
import pytest
from conftest import default_cfg
from omegaconf import OmegaConf

from cocoa.util import combine_processed_data

ARTIFACTS = ("meds.parquet", "subject_splits.parquet", "tokens_times.parquet")
OLD_DTTM = "2000-01-01T00:00:00-06:00"


def copy_processed(src, dest, *, suffix: str = None) -> pathlib.Path:
    """copy a processed dir, optionally suffixing every subject_id"""
    src, dest = pathlib.Path(src), pathlib.Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if suffix is not None and f.suffix == ".parquet":
            pl.read_parquet(f).with_columns(
                pl.col("subject_id") + suffix
            ).write_parquet(dest / f.name)
        else:
            shutil.copy(f, dest / f.name)
    return dest


def canon(df: pl.DataFrame) -> pl.DataFrame:
    """a row-order-independent view: sorted by every non-nested column"""
    by = [c for c, t in df.schema.items() if not isinstance(t, pl.List)]
    return df.sort(by, nulls_last=True, maintain_order=True)


def read(d, name: str) -> pl.DataFrame:
    return pl.read_parquet(pathlib.Path(d) / name)


def subjects(d, name: str = "subject_splits.parquet") -> set:
    return set(read(d, name)["subject_id"].to_list())


@pytest.fixture(scope="module")
def inputs(tmp_path_factory, pipeline) -> tuple:
    """two processed dirs with identical configs but disjoint subject ids"""
    root = tmp_path_factory.mktemp("inputs")
    return (
        copy_processed(pipeline.path, root / "a", suffix="__a"),
        copy_processed(pipeline.path, root / "b", suffix="__b"),
    )


def test_combine_sums_rows_and_unions_subjects(inputs, tmp_path):
    a, b = inputs
    assert not subjects(a) & subjects(b), "inputs must be disjoint to be countable"
    out = pathlib.Path(combine_processed_data([a, b], tmp_path / "out"))
    for name in ARTIFACTS:
        n_a, n_b = read(a, name).height, read(b, name).height
        assert n_a > 0 and n_b > 0
        assert read(out, name).height == n_a + n_b
    assert subjects(out) == subjects(a) | subjects(b)
    assert subjects(out, "tokens_times.parquet") == subjects(a) | subjects(b)
    # one timeline per subject survives the concatenation
    assert read(out, "tokens_times.parquet")["subject_id"].n_unique() == len(
        subjects(out)
    )


@pytest.mark.parametrize("name", (*ARTIFACTS, "held_out_for_inference.parquet"))
def test_combine_preserves_rows_verbatim(inputs, tmp_path, name):
    a, b = inputs
    out = pathlib.Path(combine_processed_data([a, b], tmp_path / "out"))
    want = pl.concat([read(a, name), read(b, name)])
    got = read(out, name)
    assert got.height == want.height > 0
    assert canon(got).equals(canon(want))


def test_combine_does_not_modify_its_inputs(inputs, tmp_path):
    a, b = inputs
    before = {f: f.read_bytes() for d in (a, b) for f in sorted(d.iterdir())}
    assert len(before) >= 8  # artifacts of both inputs, none of them empty
    combine_processed_data([a, b], tmp_path / "out")
    assert {f: f.read_bytes() for f in before} == before


def test_combine_accepts_string_paths(inputs, tmp_path):
    a, b = inputs
    dest = tmp_path / "out"
    combine_processed_data([str(a), str(b)], str(dest))
    assert read(dest, "meds.parquet").height == (
        read(a, "meds.parquet").height + read(b, "meds.parquet").height
    )


def test_three_inputs_sum_rows(inputs, tmp_path):
    a, b = inputs
    c = copy_processed(a, tmp_path / "c", suffix="__c")
    out = pathlib.Path(combine_processed_data([a, b, c], tmp_path / "out"))
    for name in ARTIFACTS:
        want = sum(read(d, name).height for d in (a, b, c))
        assert read(out, name).height == want > 0
    assert subjects(out) == subjects(a) | subjects(b) | subjects(c)
    assert len(subjects(out)) == 3 * len(subjects(a))


def test_single_input_reproduces_it(inputs, tmp_path):
    a, _ = inputs
    out = pathlib.Path(combine_processed_data([a], tmp_path / "solo"))
    assert {f.name for f in out.iterdir()} == {f.name for f in a.iterdir()}
    for name in (*ARTIFACTS, "held_out_for_inference.parquet"):
        want, got = read(a, name), read(out, name)
        assert got.height == want.height > 0
        assert canon(got).equals(canon(want))


def test_output_directory_created_and_str_returned(inputs, tmp_path):
    a, _ = inputs
    dest = tmp_path / "deep" / "nested" / "out"
    assert not dest.exists()
    got = combine_processed_data([a], dest)
    assert isinstance(got, str)
    assert pathlib.Path(got).is_dir()
    assert pathlib.Path(got).samefile(dest)
    assert read(dest, "meds.parquet").height == read(a, "meds.parquet").height


def test_tokenizer_yaml_matches_inputs_with_refreshed_created_dttm(inputs, tmp_path):
    a, b = inputs
    dirs = [
        copy_processed(d, tmp_path / n) for d, n in zip(inputs, ("a", "b"), strict=True)
    ]
    for d in dirs:  # plant an old timestamp so a refresh is detectable
        cfg = OmegaConf.load(d / "tokenizer.yaml")
        cfg.created_dttm = OLD_DTTM
        (d / "tokenizer.yaml").write_text(OmegaConf.to_yaml(cfg))
    out = pathlib.Path(combine_processed_data(dirs, tmp_path / "out"))

    assert {f.name for f in out.glob("*.yaml")} == {"tokenizer.yaml"}
    want, got = (OmegaConf.load(d / "tokenizer.yaml") for d in (dirs[0], out))
    assert got.created_dttm != OLD_DTTM
    refreshed = datetime.datetime.fromisoformat(got.created_dttm)
    assert refreshed.tzinfo is not None
    assert refreshed > datetime.datetime.fromisoformat(OLD_DTTM)
    del want.created_dttm
    del got.created_dttm
    assert OmegaConf.to_container(got) == OmegaConf.to_container(want)
    # the substantive content really is there to have been compared
    assert len(got.lookup) > 100 and got.cfg.n_bins == 10


def test_matching_configs_do_not_warn(inputs, tmp_path, capsys):
    a, b = inputs
    capsys.readouterr()
    combine_processed_data([a, b], tmp_path / "out")
    assert "Configuration mismatch" not in capsys.readouterr().out


def test_mismatched_configs_warn_but_still_combine(runner, tmp_path, capsys):
    """different n_bins: a warning on stdout, and the parquets combine anyway"""
    a, b = runner.seed_collated(), runner.seed_collated()
    runner.tokenize(processed=a)
    cfg = default_cfg("tokenization")
    cfg["n_bins"] = 5
    runner.tokenize(cfg=cfg, processed=b)
    assert (
        OmegaConf.load(a / "tokenizer.yaml").cfg.n_bins
        != OmegaConf.load(b / "tokenizer.yaml").cfg.n_bins
    )
    n_a, n_b = (read(d, "tokens_times.parquet").height for d in (a, b))

    capsys.readouterr()  # drop the stage chatter
    out = pathlib.Path(combine_processed_data([a, b], tmp_path / "out"))
    printed = capsys.readouterr().out
    assert "Configuration mismatch" in printed
    assert "tokenizer.yaml" in printed
    assert "n_bins" in printed  # the deepdiff names the offending key
    assert read(out, "tokens_times.parquet").height == n_a + n_b > 0

    # actual behaviour: only the first input's tokenizer survives, so the
    # combined timelines are decoded against a vocabulary half of them
    # were not built with
    kept = OmegaConf.load(out / "tokenizer.yaml")
    del kept.created_dttm
    first = OmegaConf.load(a / "tokenizer.yaml")
    del first.created_dttm
    assert kept == first
    words_a = {v: k for k, v in dict(first.lookup).items()}
    words_b = {
        v: k for k, v in dict(OmegaConf.load(b / "tokenizer.yaml").lookup).items()
    }
    assert [i for i in words_a if i in words_b and words_a[i] != words_b[i]]


def test_split_tokens_times_files_are_excluded(inputs, tmp_path):
    dirs = [
        copy_processed(d, tmp_path / n) for d, n in zip(inputs, ("a", "b"), strict=True)
    ]
    for d in dirs:
        shutil.copy(d / "tokens_times.parquet", d / "train_tokens_times.parquet")
    out = pathlib.Path(combine_processed_data(dirs, tmp_path / "out"))
    assert not (out / "train_tokens_times.parquet").exists()
    assert read(out, "tokens_times.parquet").height == sum(
        read(d, "tokens_times.parquet").height for d in dirs
    )


def test_shuffle_is_deterministic(inputs, tmp_path):
    a, b = inputs
    o1 = pathlib.Path(combine_processed_data([a, b], tmp_path / "o1"))
    o2 = pathlib.Path(combine_processed_data([a, b], tmp_path / "o2"))
    for name in ARTIFACTS:
        f1, f2 = read(o1, name), read(o2, name)
        assert f1.height > 0
        assert f1.equals(f2)  # same rows in the same order
        assert (o1 / name).read_bytes() == (o2 / name).read_bytes()


def test_output_is_shuffled_not_concatenated(inputs, tmp_path):
    a, b = inputs
    out = pathlib.Path(combine_processed_data([a, b], tmp_path / "out"))
    got = read(out, "tokens_times.parquet")["subject_id"].to_list()
    want = (
        read(a, "tokens_times.parquet")["subject_id"].to_list()
        + read(b, "tokens_times.parquet")["subject_id"].to_list()
    )
    assert len(got) == len(want) > 2
    assert sorted(got) == sorted(want)
    assert got != want
    # and the two inputs are interleaved, not merely block-concatenated
    head = got[: len(got) // 2]
    assert any(s.endswith("__a") for s in head)
    assert any(s.endswith("__b") for s in head)


def test_missing_column_is_inserted_as_null(inputs, tmp_path):
    """an extra column in the first input becomes null for the others"""
    dirs = [
        copy_processed(d, tmp_path / n) for d, n in zip(inputs, ("a", "b"), strict=True)
    ]
    name = "held_out_for_inference.parquet"
    read(dirs[0], name).with_columns(pl.lit(True).alias("extra_flag")).write_parquet(
        dirs[0] / name
    )
    n_a, n_b = (read(d, name).height for d in dirs)
    out = pathlib.Path(combine_processed_data(dirs, tmp_path / "out"))

    got = read(out, name)
    assert got.height == n_a + n_b
    assert "extra_flag" in got.columns
    assert got["extra_flag"].null_count() == n_b > 0
    assert set(got.filter(pl.col("extra_flag"))["subject_id"]) == subjects(
        dirs[0], name
    )
    assert set(got.filter(pl.col("extra_flag").is_null())["subject_id"]) == subjects(
        dirs[1], name
    )


def test_extra_column_in_later_input_raises(inputs, tmp_path):
    """
    actual behaviour: missing_columns="insert" only fills columns absent from
    the *first* input, so the same two dirs combine in one order and fail in
    the other
    """
    dirs = [
        copy_processed(d, tmp_path / n) for d, n in zip(inputs, ("a", "b"), strict=True)
    ]
    name = "held_out_for_inference.parquet"
    read(dirs[1], name).with_columns(pl.lit(True).alias("extra_flag")).write_parquet(
        dirs[1] / name
    )
    with pytest.raises(pl.exceptions.SchemaError, match="extra column"):
        combine_processed_data(dirs, tmp_path / "out")


@pytest.mark.parametrize("legacy", ["first", "second"])
def test_legacy_int64_tokens_are_upcast(inputs, tmp_path, legacy):
    """tokenizers <= 26.4.0 wrote List(Int64) tokens; the fallback recasts them"""
    dirs = [
        copy_processed(d, tmp_path / n) for d, n in zip(inputs, ("a", "b"), strict=True)
    ]
    target = dirs[0 if legacy == "first" else 1]
    read(target, "tokens_times.parquet").with_columns(
        pl.col("tokens").cast(pl.List(pl.Int64))
    ).write_parquet(target / "tokens_times.parquet")
    assert read(target, "tokens_times.parquet").schema["tokens"] == pl.List(pl.Int64)
    with pytest.raises(pl.exceptions.SchemaError):  # the plain scan cannot do it
        pl.scan_parquet(
            [d / "tokens_times.parquet" for d in dirs], missing_columns="insert"
        ).collect()

    out = pathlib.Path(combine_processed_data(dirs, tmp_path / "out"))
    got = read(out, "tokens_times.parquet")
    assert got.schema["tokens"] == pl.List(pl.UInt32)
    a, b = inputs  # the untouched originals: values must round-trip
    want = pl.concat([read(a, "tokens_times.parquet"), read(b, "tokens_times.parquet")])
    assert got.height == want.height > 0
    assert canon(got).equals(canon(want))


def test_parquet_only_in_later_input_is_ignored(inputs, tmp_path):
    a, b = inputs
    b2 = copy_processed(b, tmp_path / "b2")
    pl.DataFrame({"subject_id": ["x"], "junk": [1]}).write_parquet(
        b2 / "orphan.parquet"
    )
    out = pathlib.Path(combine_processed_data([a, b2], tmp_path / "out"))
    assert not (out / "orphan.parquet").exists()
    assert {f.name for f in out.iterdir()} == {f.name for f in a.iterdir()}


def test_parquet_missing_from_later_input_raises(inputs, tmp_path):
    a, b = inputs
    b2 = copy_processed(b, tmp_path / "b2")
    (b2 / "meds.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        combine_processed_data([a, b2], tmp_path / "out")


def test_timezone_mismatch_escapes_config_diff_and_raises(runner, tmp_path, capsys):
    """
    two dirs collated in different zones carry identical tokenizer.yamls, so
    the config diff says nothing and polars fails on the datetime columns after
    the yaml has already been written to the output
    """
    utc = runner.full()
    cfg = default_cfg("collation")
    cfg["default_timezone"] = "America/Chicago"
    chi = runner.full(collation=cfg)
    assert read(utc.path, "meds.parquet").schema["time"].time_zone == "UTC"
    assert read(chi.path, "meds.parquet").schema["time"].time_zone == "America/Chicago"
    left, right = (OmegaConf.load(d / "tokenizer.yaml") for d in (utc.path, chi.path))
    del left.created_dttm
    del right.created_dttm
    assert left == right, "the zone is not recorded in tokenizer.yaml"

    dest = tmp_path / "out"
    capsys.readouterr()
    with pytest.raises(
        pl.exceptions.SchemaError, match="data type mismatch for column time"
    ):
        combine_processed_data([utc.path, chi.path], dest)
    assert "Configuration mismatch" not in capsys.readouterr().out
    assert (dest / "tokenizer.yaml").exists()  # partial output is left behind
