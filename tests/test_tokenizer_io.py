#!/usr/bin/env python3

"""tokenizer serialization, reload, transfer, and its small api surface"""

import collections
import datetime
import importlib.metadata as meta
import pathlib
import shutil

import polars as pl
import pytest
import synth
from conftest import Processed, default_cfg
from omegaconf import OmegaConf

from cocoa.collator import Collator
from cocoa.tokenizer import Tokenizer

YAML_KEYS = {
    "lookup",
    "counts",
    "bins",
    "is_training",
    "cfg",
    "created_dttm",
    "cocoa_version",
}

MIN_HOSP = [
    {
        "admission_dttm": datetime.datetime(2024, 1, 1, 0, 0),
        "discharge_dttm": datetime.datetime(2024, 1, 2, 0, 0),
    }
]

# one known code with a value away from every learned cut point, one unseen code
MIN_VITALS = [
    {"recorded_dttm": datetime.datetime(2024, 1, 1, 3, 0), "vital_value": 100.0},
    {
        "recorded_dttm": datetime.datetime(2024, 1, 1, 6, 0),
        "vital_category": "quokka vital",
        "vital_value": 100.0,
    },
]


def _seed(src: pathlib.Path, dest: pathlib.Path) -> pathlib.Path:
    """copy collated artifacts out of one processed dir into another"""
    dest = pathlib.Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for f in ("meds.parquet", "subject_splits.parquet"):
        shutil.copy(pathlib.Path(src) / f, dest / f)
    return dest


def _by_time(path: pathlib.Path) -> pl.DataFrame:
    """
    tokenized timelines as (subject, timestamp) -> sorted tokens; the order of
    tokens sharing a timestamp is not reproducible run to run, this view is
    """
    return (
        pl.read_parquet(path)
        .explode("tokens", "times")
        .group_by("subject_id", "times")
        .agg(pl.col("tokens").sort())
        .sort("subject_id", "times")
    )


def _collate_minimal(runner, prefix: str = "VTL") -> pathlib.Path:
    """collate the hand-written one-subject dataset, optionally under a prefix"""
    raw = synth.write_minimal_dataset(
        runner.dir(), hospitalizations=MIN_HOSP, vitals=MIN_VITALS
    )
    cfg = synth.minimal_collation_cfg()
    cfg["entries"] = [{**cfg["entries"][0], "prefix": prefix}]
    dest = runner.dir()
    runner.collate(cfg=cfg, raw=raw, dest=dest)
    return dest


@pytest.fixture(scope="module")
def trained(tmp_path_factory, pipeline) -> Tokenizer:
    """a tokenizer learned once over the session collation, left in training mode"""
    dest = _seed(pipeline.path, tmp_path_factory.mktemp("learned"))
    tkzr = Tokenizer(processed_data_home=dest)
    tkzr.save_all()
    return tkzr


@pytest.fixture(scope="module")
def src_yaml(trained) -> pathlib.Path:
    """the tokenizer.yaml that transfer tests load from"""
    return trained.processed_data_home / "tokenizer.yaml"


@pytest.fixture(scope="module")
def src_vocab(src_yaml) -> dict:
    return dict(OmegaConf.load(src_yaml).lookup)


@pytest.fixture(scope="module")
def second(tmp_path_factory) -> Processed:
    """a second, independently generated dataset, collated with the defaults"""
    root = tmp_path_factory.mktemp("second")
    raw = synth.write_raw_dataset(root / "raw", n_patients=12)
    dest = root / "processed"
    dest.mkdir()
    Collator(raw_data_home=raw.root, processed_data_home=dest).save_all()
    return Processed(dest, raw)


# --- to_yaml / from_yaml -----------------------------------------------------


def test_to_yaml_from_yaml_preserves_artifacts_cfg_and_created_dttm(trained):
    cp = trained.from_yaml(trained.to_yaml())
    assert trained.lookup.height > 200 and trained.bins.height > 0
    assert cp.lookup.equals(trained.lookup)
    assert cp.bins.equals(trained.bins)
    assert cp.cfg == trained.cfg
    assert cp.created_dttm == trained.created_dttm
    assert len(cp) == len(trained)


def test_to_yaml_has_expected_top_level_keys_and_records_version(trained):
    y = OmegaConf.create(trained.to_yaml())
    assert set(y.keys()) == YAML_KEYS
    assert y.cocoa_version == meta.version("cocoa-tokenizer")
    assert y.created_dttm == trained.created_dttm
    assert dict(y.lookup) == dict(trained.lookup.select("to_tokenize", "token").rows())
    assert dict(y.counts) == dict(trained.lookup.select("to_tokenize", "count").rows())
    assert y.cfg.n_bins == trained.cfg.n_bins == 10


def test_from_yaml_tolerates_a_legacy_yaml_without_counts(trained):
    """tokenizers written before counts existed still load, with null counts"""
    y = OmegaConf.create(trained.to_yaml())
    del y.counts
    cp = trained.from_yaml(OmegaConf.to_yaml(y))
    assert cp.lookup.drop("count").equals(trained.lookup.drop("count"))
    assert cp.lookup["count"].null_count() == cp.lookup.height
    assert cp("EOS") == trained("EOS")


@pytest.mark.parametrize(
    "done_training,expected", [(True, False), (False, True)], ids=["frozen", "training"]
)
def test_from_yaml_freezes_unless_told_otherwise(trained, done_training, expected):
    """from_yaml lands in inference mode by default"""
    cp = trained.from_yaml(trained.to_yaml(), done_training=done_training)
    assert trained.is_training is True  # save_all does not freeze the learner
    assert cp.is_training is expected


def test_untrained_tokenizer_round_trips_with_null_artifacts(tmp_path):
    tkzr = Tokenizer(processed_data_home=tmp_path, n_bins=4)
    y = OmegaConf.create(tkzr.to_yaml())
    assert y.lookup is None and y.bins is None and y.cfg.n_bins == 4
    cp = tkzr.from_yaml(tkzr.to_yaml())
    assert cp.lookup is None and cp.bins is None
    assert len(cp) == 0
    assert cp.cfg.n_bins == 4


def test_from_yaml_prefers_saved_cfg_over_the_loading_config(runner, src_yaml):
    """the frozen tokenizer's own config wins over one passed at load time"""
    cfg = runner.cfg_path("tokenization", {**default_cfg("tokenization"), "n_bins": 4})
    loader = Tokenizer(tokenization_cfg=cfg, processed_data_home=runner.dir())
    assert loader.cfg.n_bins == 4
    loaded = loader.load(src_yaml)
    assert loaded.cfg.n_bins == 10
    assert loaded.bins.width == 10  # code + 9 breaks


def test_saved_n_bins_drives_bins_reconstruction(runner):
    dest = runner.seed_collated()
    cfg = {**default_cfg("tokenization"), "n_bins": 4}
    tkzr = runner.tokenize(processed=dest, cfg=cfg)
    assert tkzr.bins.width == 4
    back = Tokenizer(processed_data_home=dest).load(dest / "tokenizer.yaml")
    assert back.cfg.n_bins == 4
    assert back.bins.columns == ["code", "break_1", "break_2", "break_3"]
    assert back.bins.equals(tkzr.bins)
    assert back.lookup.equals(tkzr.lookup)


def test_from_yaml_targets_the_loading_processed_dir(runner, src_yaml):
    dest = runner.dir()
    loaded = Tokenizer(processed_data_home=dest).load(src_yaml)
    assert loaded.processed_data_home == dest.resolve()


# --- save / load on disc -----------------------------------------------------


def test_save_creates_missing_parents_and_writes_to_yaml(trained, tmp_path):
    path = tmp_path / "no" / "such" / "dir" / "tokenizer.yaml"
    trained.save(path)
    assert path.exists()
    assert path.read_text() == trained.to_yaml()


def test_load_from_disc_round_trips(trained, tmp_path):
    path = tmp_path / "saved.yaml"
    trained.save(path)
    back = Tokenizer(processed_data_home=tmp_path).load(str(path))  # the cli passes str
    assert back.lookup.equals(trained.lookup)
    assert back.bins.equals(trained.bins)
    assert back.created_dttm == trained.created_dttm
    assert back.cfg == trained.cfg
    assert back.is_training is False


def test_save_all_writes_tokenizer_yaml_matching_the_vocabulary(trained, src_yaml):
    y = OmegaConf.load(src_yaml)
    assert set(y.keys()) == YAML_KEYS
    assert dict(y.lookup) == dict(trained.lookup.select("to_tokenize", "token").rows())
    assert y.lookup["UNK"] == 0
    assert y.is_training is True  # a freshly learned tokenizer is saved unfrozen
    assert len(dict(y.bins)) == trained.bins.height


def test_relearning_the_same_collation_gives_the_same_vocabulary(trained, pipeline):
    """vocabulary learning is deterministic across runs"""
    assert len(pipeline.vocab) > 0
    assert dict(trained.lookup.select("to_tokenize", "token").rows()) == pipeline.vocab


# --- dunders -----------------------------------------------------------------


@pytest.mark.parametrize("word", ["BOS", "EOS", "VTL//heart_rate_Q0", "SEX//female"])
def test_call_maps_known_word_to_its_token(trained, pipeline, word):
    assert word in pipeline.vocab  # expectation comes from the serialized vocab
    assert trained(word) == pipeline.vocab[word]
    assert word in trained


@pytest.mark.parametrize("word", ["#$%^&*()", "", "quokka", "VTL//heart_rate_Q42"])
def test_call_maps_unknown_word_to_zero(trained, pipeline, word):
    assert word not in pipeline.vocab
    assert trained(word) == 0
    assert word not in trained


def test_unk_is_token_zero_and_reads_as_absent(trained):
    """UNK occupies token 0, so __contains__ reports it as out-of-vocabulary"""
    assert trained("UNK") == 0
    assert "UNK" in trained.lookup["to_tokenize"].to_list()
    assert "UNK" not in trained


def test_len_equals_vocabulary_size(trained, src_vocab):
    assert len(trained) == trained.lookup.height == len(src_vocab)
    assert len(trained) > 200


def test_str_reports_size_and_mode(trained):
    frozen = trained.from_yaml(trained.to_yaml())
    assert f"of {len(trained)} words" in str(trained)
    assert "Tokenizer" in str(trained)
    assert "in training mode" in str(trained) and "(frozen)" not in str(trained)
    assert "(frozen)" in str(frozen) and "in training mode" not in str(frozen)


def test_repr_includes_created_dttm(trained):
    assert repr(trained).startswith(str(trained))
    assert trained.created_dttm in repr(trained)
    assert trained.created_dttm not in str(trained)


def test_fresh_tokenizer_is_empty(tmp_path):
    tkzr = Tokenizer(processed_data_home=tmp_path)
    assert len(tkzr) == 0
    assert tkzr("BOS") == 0
    assert "BOS" not in tkzr
    assert "of 0 words in training mode" in str(tkzr)


# --- frozen artifacts --------------------------------------------------------


@pytest.mark.parametrize(
    "method,msg",
    [
        ("get_bins", "Bins must be learned during training"),
        ("get_lookup", "Lookup table must be learned during training"),
    ],
)
def test_frozen_tokenizer_without_artifacts_refuses_to_learn(tmp_path, method, msg):
    tkzr = Tokenizer(processed_data_home=tmp_path, is_training=False)
    assert tkzr.bins is None and tkzr.lookup is None
    with pytest.raises(AssertionError, match=msg):
        getattr(tkzr, method)(pl.LazyFrame())


@pytest.mark.parametrize("attr", ["bins", "lookup"])
def test_loaded_tokenizer_with_cleared_artifact_refuses_to_learn(
    trained, src_yaml, tmp_path, attr
):
    tkzr = Tokenizer(processed_data_home=tmp_path).load(src_yaml)
    assert getattr(tkzr, attr) is not None
    setattr(tkzr, attr, None)
    getter = {"bins": tkzr.get_bins, "lookup": tkzr.get_lookup}[attr]
    with pytest.raises(AssertionError, match="must be learned during training"):
        getter(pl.LazyFrame())


def test_frozen_tokenizer_missing_bins_fails_save_all(runner, src_yaml):
    dest = runner.seed_collated()
    tkzr = Tokenizer(processed_data_home=dest).load(src_yaml)
    tkzr.bins = None
    with pytest.raises(AssertionError, match="Bins must be learned during training"):
        tkzr.save_all()
    assert not (dest / "tokens_times.parquet").exists()


# --- tokenizer transfer ------------------------------------------------------


def test_transfer_reuses_the_source_vocabulary_and_bins(
    runner, second, trained, src_yaml
):
    """the transferred tokenizer is frozen, not relearned on the new dataset"""
    transferred = runner.tokenize(
        processed=_seed(second.path, runner.dir()), tokenizer_home=src_yaml
    )
    assert transferred.lookup.equals(trained.lookup)
    assert transferred.bins.equals(trained.bins)
    assert transferred.is_training is False

    # non-vacuity: the second dataset learns a different vocabulary on its own
    own = runner.tokenize(processed=_seed(second.path, runner.dir()))
    assert own.lookup.height > 100
    assert not own.lookup.equals(trained.lookup)
    assert not own.bins.equals(trained.bins)


def test_transfer_leaves_the_source_yaml_untouched(runner, second, src_yaml):
    before = src_yaml.read_text()
    runner.tokenize(processed=_seed(second.path, runner.dir()), tokenizer_home=src_yaml)
    assert src_yaml.read_text() == before


def test_transferred_yaml_is_frozen_and_keeps_source_provenance(
    runner, second, src_yaml, src_vocab
):
    dest = _seed(second.path, runner.dir())
    runner.tokenize(processed=dest, tokenizer_home=src_yaml)
    y = OmegaConf.load(dest / "tokenizer.yaml")
    src = OmegaConf.load(src_yaml)
    assert dict(y.lookup) == src_vocab
    assert dict(y.bins) == dict(src.bins)
    assert y.is_training is False  # a transferred tokenizer records itself as frozen
    assert y.created_dttm == src.created_dttm
    assert y.cocoa_version == meta.version("cocoa-tokenizer")


def test_transferred_timelines_are_well_formed(runner, second, src_yaml, src_vocab):
    dest = _seed(second.path, runner.dir())
    runner.tokenize(processed=dest, tokenizer_home=src_yaml)
    out = Processed(dest, second.manifest)
    subjects = sorted(second.manifest.subject_ids)
    assert out.tokens_times.height == len(subjects) > 0
    assert sorted(out.tokens_times["subject_id"].to_list()) == subjects
    for sid in subjects:
        line = out.timeline(sid)
        assert line[0] == "BOS" and line[-1] == "EOS"
        assert len(line) > 2
    every = out.tokens_times["tokens"].explode()
    assert every.max() < len(src_vocab)


def test_transfer_unks_codes_absent_from_the_source_vocabulary(
    runner, second, src_yaml, src_vocab
):
    """
    synth plants two labs that never occur in a training split, so the source
    vocabulary cannot contain them; under transfer they must land on 0
    """
    unseen_codes = [
        "LAB-RES//" + lab.replace(" ", "_")
        for lab in (synth.TUNING_ONLY_LAB, synth.HELD_OUT_ONLY_LAB)
    ]
    assert not [w for w in src_vocab if w.startswith(tuple(unseen_codes))]

    dest = _seed(second.path, runner.dir())
    runner.tokenize(processed=dest, tokenizer_home=src_yaml)
    events = pl.read_parquet(dest / "meds.parquet").filter(
        pl.col("code").is_in(unseen_codes)
    )
    assert events.height > 0
    flat = pl.read_parquet(dest / "tokens_times.parquet").explode("tokens", "times")
    for row in events.iter_rows(named=True):
        at = flat.filter(
            (pl.col("subject_id") == row["subject_id"])
            & (pl.col("times") == row["time"])
        )
        assert at.height > 0
        assert 0 in at["tokens"].to_list(), row


def test_transfer_maps_valueless_codes_to_their_source_tokens(
    runner, second, src_yaml, src_vocab
):
    """
    a code carrying neither a numeric nor a text value fuses to itself, so the
    source vocabulary alone determines its token in the new dataset
    """
    dest = _seed(second.path, runner.dir())
    runner.tokenize(processed=dest, tokenizer_home=src_yaml)
    meds = pl.read_parquet(dest / "meds.parquet").filter(
        pl.col("numeric_value").is_null() & pl.col("text_value").is_null()
    )
    assert meds.height > 100

    got = {
        (r["subject_id"], r["times"]): collections.Counter(r["tokens"])
        for r in _by_time(dest / "tokens_times.parquet").iter_rows(named=True)
    }
    want = {}
    for r in meds.iter_rows(named=True):
        key = (r["subject_id"], r["time"])
        want.setdefault(key, collections.Counter())[src_vocab.get(r["code"], 0)] += 1

    assert len(want) > 50
    seen = set()
    for key, counts in want.items():
        assert key in got, key
        assert counts <= got[key], key  # BOS/EOS and valued codes may be extra
        seen |= set(counts)
    assert len(seen - {0}) > 20


def test_transfer_applies_source_bins_and_unks_the_unseen_code(runner, src_yaml):
    breaks = list(dict(OmegaConf.load(src_yaml).bins)["VTL//heart_rate"])
    assert len(breaks) == 9
    assert all(abs(b - 100.0) > 1.0 for b in breaks)  # 100 is not near a cut point
    expected = "VTL//heart_rate_Q{}".format(sum(1 for b in breaks if b <= 100.0))

    dest = _collate_minimal(runner)
    runner.tokenize(processed=dest, tokenizer_home=src_yaml)
    out = Processed(dest, None)
    assert out.timeline("H0") == ["BOS", expected, "UNK", "EOS"]
    assert out.tokens_times["tokens"].item().to_list()[2] == 0


def test_without_transfer_the_minimal_dataset_learns_its_own_bins(runner):
    """the contrast case: one training value per code puts every cut point on it"""
    dest = _collate_minimal(runner)
    runner.tokenize(processed=dest)
    out = Processed(dest, None)
    assert out.timeline("H0") == [
        "BOS",
        "VTL//heart_rate_Q9",
        "VTL//quokka_vital_Q9",
        "EOS",
    ]


def test_unseen_prefix_tokenizes_to_unk_and_sorts_after_eos(runner, src_yaml):
    """a prefix missing from `ordering` sorts last, so it can follow EOS"""
    dest = _collate_minimal(runner, prefix="ZZZ")
    runner.tokenize(processed=dest, tokenizer_home=src_yaml)
    out = Processed(dest, None)
    assert pl.read_parquet(dest / "meds.parquet")["code"].to_list() == [
        "ZZZ//heart_rate",
        "ZZZ//quokka_vital",
    ]
    assert out.timeline("H0") == ["BOS", "UNK", "EOS", "UNK"]


def test_transfer_is_idempotent(runner, second, src_yaml):
    first = _seed(second.path, runner.dir())
    again = _seed(second.path, runner.dir())
    chained = _seed(second.path, runner.dir())
    a = runner.tokenize(processed=first, tokenizer_home=src_yaml)
    b = runner.tokenize(processed=again, tokenizer_home=src_yaml)
    c = runner.tokenize(processed=chained, tokenizer_home=first / "tokenizer.yaml")
    assert a.lookup.equals(b.lookup) and a.lookup.equals(c.lookup)
    assert a.bins.equals(c.bins)
    assert (first / "tokenizer.yaml").read_text() == (
        chained / "tokenizer.yaml"
    ).read_text()
    left = _by_time(first / "tokens_times.parquet")
    right = _by_time(chained / "tokens_times.parquet")
    assert left.height > 100
    assert left.equals(right)


def test_loaded_tokenizer_reproduces_the_original_tokens(runner, trained, src_yaml):
    """
    same collated data in, same tokens out -- but only up to the order of
    tokens sharing a timestamp, which cocoa does not reproduce run to run
    (tokenize_data sorts on ("time", "priority") and leaves ties unordered)
    """
    dest = _seed(trained.processed_data_home, runner.dir())
    runner.tokenize(processed=dest, tokenizer_home=src_yaml)
    src = trained.processed_data_home / "tokens_times.parquet"
    original = pl.read_parquet(src).sort("subject_id")
    again = pl.read_parquet(dest / "tokens_times.parquet").sort("subject_id")
    assert original.height > 0
    assert original["subject_id"].to_list() == again["subject_id"].to_list()
    assert original["times"].to_list() == again["times"].to_list()
    assert [sorted(t) for t in original["tokens"].to_list()] == [
        sorted(t) for t in again["tokens"].to_list()
    ]
    assert _by_time(src).equals(_by_time(dest / "tokens_times.parquet"))
