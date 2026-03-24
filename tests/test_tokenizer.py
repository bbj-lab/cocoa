import datetime
import pathlib
import tempfile

import polars as pl
import pytest
from omegaconf import OmegaConf

from cocoa.tokenizer import Tokenizer


def _make_collated_data(tmp_path: pathlib.Path):
    """Write collated meds + splits parquet files mirroring collator output."""
    # 4 subjects with SEX events + lab events (same shape as collator tests)
    meds = pl.DataFrame(
        {
            "subject_id": [
                "100",
                "100",
                "100",
                "101",
                "102",
                "102",
                "103",
                "103",
                "103",
            ],
            "time": [
                datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(2025, 1, 2, tzinfo=datetime.timezone.utc),
                datetime.datetime(2025, 1, 3, tzinfo=datetime.timezone.utc),
                datetime.datetime(2025, 3, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(2025, 2, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(2025, 2, 5, tzinfo=datetime.timezone.utc),
                datetime.datetime(2025, 4, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(2025, 4, 2, tzinfo=datetime.timezone.utc),
                datetime.datetime(2025, 4, 3, tzinfo=datetime.timezone.utc),
            ],
            "code": [
                "SEX//male",
                "LAB//glucose",
                "LAB//creatinine",
                "SEX//male",
                "SEX//female",
                "LAB//hemoglobin",
                "SEX//female",
                "LAB//glucose",
                "LAB//hemoglobin",
            ],
            "numeric_value": [None, 120.0, 1.2, None, None, 13.5, None, 110.0, 11.0],
            "text_value": [None] * 9,
        }
    ).cast({"numeric_value": pl.Float32, "time": pl.Datetime("us", "UTC")})
    meds.write_parquet(tmp_path / "meds.parquet")

    # splits: 100, 102 → train (earliest admit times), 101 → tuning, 103 → held_out
    splits = pl.DataFrame(
        {
            "subject_id": ["100", "102", "101", "103"],
            "split": ["train", "train", "tuning", "held_out"],
        }
    )
    splits.write_parquet(tmp_path / "splits.parquet")
    return tmp_path / "meds.parquet", tmp_path / "splits.parquet"


def _collect_lookup(lookup):
    """Return a Polars DataFrame for the lookup regardless of storage type.

    Accepts a dict, a Polars LazyFrame, or a Polars DataFrame.
    """
    if lookup is None:
        return None
    if isinstance(lookup, dict):
        return pl.DataFrame(
            {"to_tokenize": list(lookup.keys()), "token": list(lookup.values())}
        )
    # LazyFrame or DataFrame
    try:
        return lookup.collect()
    except Exception:
        return pl.DataFrame(lookup)


def _make_tokenizer(tmp_path: pathlib.Path, *, fused: bool = True, n_bins: int = 4):
    """Build a Tokenizer without reading config files from disk."""
    meds_path, splits_path = _make_collated_data(tmp_path)
    cfg = OmegaConf.create(
        {
            "data_home": str(tmp_path),
            "processed_data_home": str(tmp_path / "out"),
            "n_bins": n_bins,
            "fused": fused,
            "collated_inputs": [str(meds_path)],
            "subject_splits": str(splits_path),
            "ordering": ["BOS", "SEX", "LAB", "EOS"],
            # Defaults for clock and spacer tests
            "insert_clocks": False,
            "clocks": ["01", "02", "03", "04"],
            "insert_spacers": False,
            "spacers": {"short": 60, "medium": 120, "long": 240},
        }
    )
    tkzr = object.__new__(Tokenizer)
    tkzr.cfg = cfg
    tkzr.data_home = tmp_path
    tkzr.lookup = None
    tkzr.bins = None
    tkzr.subject_splits = None
    tkzr.is_training = True
    tkzr.created_dttm = datetime.datetime.now().isoformat()
    return tkzr


@pytest.fixture()
def tokenizer():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield _make_tokenizer(pathlib.Path(tmp_dir))


@pytest.fixture()
def unfused_tokenizer():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield _make_tokenizer(pathlib.Path(tmp_dir), fused=False)


def test_add_ends_adds_bos_eos(tokenizer):
    """BOS and EOS rows are added for every subject."""
    df = tokenizer.get_data()
    n_subjects = df.select("subject_id").collect().n_unique()
    result = tokenizer.add_ends(df).collect()
    bos = result.filter(pl.col("code") == "BOS")
    eos = result.filter(pl.col("code") == "EOS")
    assert len(bos) == n_subjects
    assert len(eos) == n_subjects


def test_get_bins_only_uses_training_data(tokenizer):
    """Bins should be computed from training subjects only."""
    df = tokenizer.get_data()
    df = tokenizer.add_ends(df)
    bins = tokenizer.get_bins(df).collect()
    # training subjects are 100 and 102, which have LAB//glucose, LAB//creatinine,
    # LAB//hemoglobin — so bins should have those codes
    binned_codes = set(bins["code"].to_list())
    assert "LAB//glucose" in binned_codes
    assert "LAB//creatinine" in binned_codes
    assert "LAB//hemoglobin" in binned_codes
    # SEX codes have no numeric values, so no bins
    assert not any(c.startswith("SEX") for c in binned_codes)


def test_bin_data_assigns_quantile_labels(tokenizer):
    """Numeric values get a Q-label; non-numeric rows get null."""
    df = tokenizer.get_data()
    df = tokenizer.add_ends(df)
    result = tokenizer.bin_data(df).collect()
    lab_rows = result.filter(pl.col("code").str.starts_with("LAB"))
    sex_rows = result.filter(pl.col("code").str.starts_with("SEX"))
    # labs have numeric values → should get binned_value like "Q3"
    assert lab_rows["binned_value"].is_null().sum() == 0
    assert all(v.startswith("Q") for v in lab_rows["binned_value"].to_list())
    # sex rows have no numeric values → null binned_value
    assert sex_rows["binned_value"].is_null().all()


def test_fused_tokenization_one_token_per_event(tokenizer):
    """In fused mode, each event becomes exactly one token."""
    df = tokenizer.get_data()
    df = tokenizer.add_ends(df)
    df = tokenizer.bin_data(df)
    df = df.join(tokenizer.subject_splits, on="subject_id", validate="m:1")
    df = tokenizer.insert_time_spacers(df)
    pre = tokenizer.get_pretokenized(df.filter(pl.col("split") == "train")).collect()
    # each original event should produce a single pre-tokenized string
    assert all(len(t) == 1 for t in pre["to_tokenize"].to_list())


def test_unfused_tokenization_multiple_tokens(unfused_tokenizer):
    """In unfused mode, events with numeric values produce >1 token."""
    tkzr = unfused_tokenizer
    df = tkzr.get_data()
    df = tkzr.add_ends(df)
    df = tkzr.bin_data(df)
    df = df.join(tkzr.subject_splits, on="subject_id", validate="m:1")
    df = tkzr.insert_time_spacers(df)
    pre = tkzr.get_pretokenized(df.filter(pl.col("split") == "train")).collect()
    lab_rows = pre.filter(pl.col("code").str.starts_with("LAB"))
    sex_rows = pre.filter(pl.col("code").str.starts_with("SEX"))
    # labs: code token + binned_value token = 2
    assert all(len(t) == 2 for t in lab_rows["to_tokenize"].to_list())
    # sex: just code token = 1
    assert all(len(t) == 1 for t in sex_rows["to_tokenize"].to_list())


def test_get_all_produces_one_row_per_subject(tokenizer):
    """After aggregation, there's one timeline per subject."""
    result = tokenizer.get_all().collect()
    assert result["subject_id"].n_unique() == len(result)
    assert len(result) == 4


def test_get_all_timelines_start_with_bos(tokenizer):
    """Every timeline should begin with BOS token (token 1)."""
    result = tokenizer.get_all().collect()
    lookup_df = _collect_lookup(tokenizer.lookup)
    bos_token = lookup_df.filter(pl.col("to_tokenize") == "BOS")["token"].to_list()[0]
    for tokens in result["tokens"].to_list():
        assert tokens[0] == bos_token


def test_get_all_timelines_contain_eos(tokenizer):
    """Every timeline should contain an EOS token."""
    result = tokenizer.get_all().collect()
    lookup_df = _collect_lookup(tokenizer.lookup)
    eos_token = lookup_df.filter(pl.col("to_tokenize") == "EOS")["token"].to_list()[0]
    for tokens in result["tokens"].to_list():
        assert eos_token in tokens


def test_vocab_frozen_after_get_all(tokenizer):
    """After get_all, the tokenizer and its vocab should be frozen."""
    tokenizer.get_all().collect()
    assert isinstance(tokenizer.lookup, (dict, pl.LazyFrame, pl.DataFrame))


def test_contains(tokenizer):
    """__contains__ reflects vocab membership after processing."""
    tokenizer.get_all().collect()
    lookup_df = _collect_lookup(tokenizer.lookup)
    toks = lookup_df["to_tokenize"].to_list()
    assert "BOS" in toks
    assert "EOS" in toks
    assert "never_a_real_code" not in toks


def test_yaml_round_trip(tokenizer):
    """Serializing to YAML and back preserves vocab and bins."""
    tokenizer.get_all().collect()
    yaml_str = tokenizer.to_yaml()
    restored = Tokenizer.from_yaml(yaml_str, done_training=True)
    restored_lookup = _collect_lookup(restored.lookup)
    orig_lookup = _collect_lookup(tokenizer.lookup)
    assert len(restored_lookup) == len(orig_lookup)
    assert not restored.is_training
    assert restored.bins is not None
    # bins have the same codes
    orig_codes = set(tokenizer.bins.collect()["code"].to_list())
    rest_codes = set(restored.bins.collect()["code"].to_list())
    assert orig_codes == rest_codes


def test_save_all_writes_files(tokenizer):
    """save_all produces the expected output files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out = pathlib.Path(tmp_dir) / "output"
        tokenizer.save_all(path=out)
        assert (out / "tokens_times.parquet").exists()
        assert (out / "tokenizer.yaml").exists()
        # can read parquet back
        df = pl.read_parquet(out / "tokens_times.parquet")
        assert len(df) == 4
        # can load tokenizer yaml
        restored = tokenizer.load(out / "tokenizer.yaml")
        restored_lookup = _collect_lookup(restored.lookup)
        orig_lookup = _collect_lookup(tokenizer.lookup)
        assert len(restored_lookup) == len(orig_lookup)


def test_save_and_load_yaml(tokenizer):
    def test_add_clocks_inserts_clock_tokens(tmp_path):
        """Clock tokens are inserted when insert_clocks is enabled."""
        cfg = OmegaConf.create(
            {
                "data_home": str(tmp_path),
                "processed_data_home": str(tmp_path / "out"),
                "n_bins": 4,
                "fused": True,
                "collated_inputs": [str(_make_collated_data(tmp_path)[0])],
                "subject_splits": str(_make_collated_data(tmp_path)[1]),
                "ordering": ["BOS", "SEX", "LAB", "EOS"],
                "insert_clocks": True,
                "clocks": ["01", "02", "03", "04"],
            }
        )
        tkzr = object.__new__(Tokenizer)
        tkzr.cfg = cfg
        tkzr.data_home = tmp_path
        tkzr.lookup = None
        tkzr.bins = None
        tkzr.subject_splits = None
        tkzr.is_training = True
        tkzr.created_dttm = datetime.datetime.now().isoformat()
        df = tkzr.get_data()
        df = tkzr.add_ends(df)
        df = tkzr.add_clocks(df)
        result = df.collect()
        # Check for CLCK tokens
        clock_rows = result.filter(pl.col("code").str.starts_with("CLCK"))
        assert len(clock_rows) > 0
        assert all(clock_rows["code"].str.starts_with("CLCK//"))

    def test_insert_time_spacers_inserts_spacer_tokens(tmp_path):
        """Time spacing tokens are inserted when insert_spacers is enabled."""
        cfg = OmegaConf.create(
            {
                "data_home": str(tmp_path),
                "processed_data_home": str(tmp_path / "out"),
                "n_bins": 4,
                "fused": True,
                "collated_inputs": [str(_make_collated_data(tmp_path)[0])],
                "subject_splits": str(_make_collated_data(tmp_path)[1]),
                "ordering": ["BOS", "SEX", "LAB", "EOS"],
                "insert_spacers": True,
                "spacers": {"short": 60, "medium": 120, "long": 240},
            }
        )
        tkzr = object.__new__(Tokenizer)
        tkzr.cfg = cfg
        tkzr.data_home = tmp_path
        tkzr.lookup = None
        tkzr.bins = None
        tkzr.subject_splits = None
        tkzr.is_training = True
        tkzr.created_dttm = datetime.datetime.now().isoformat()
        df = tkzr.get_data()
        df = tkzr.add_ends(df)
        df = tkzr.bin_data(df)
        df = tkzr.insert_time_spacers(df)
        result = df.collect()
        # Check for TIME spacer tokens
        assert "t_spacer" in result.columns
        spacer_rows = result.filter(pl.col("t_spacer").is_not_null())
        assert len(spacer_rows) > 0
        assert all(spacer_rows["t_spacer"].str.starts_with("TIME//"))

    """save/load round-trip via YAML preserves the tokenizer."""
    tokenizer.get_all().collect()
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = pathlib.Path(tmp_dir) / "tkzr.yaml"
        tokenizer.save(path)
        assert path.exists()
        restored = tokenizer.load(path)
        restored_lookup = _collect_lookup(restored.lookup)
        orig_lookup = _collect_lookup(tokenizer.lookup)
        assert len(restored_lookup) == len(orig_lookup)
        assert not restored.is_training
