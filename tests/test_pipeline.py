#!/usr/bin/env python3

"""
end-to-end pipeline integration, and validation of the synthetic fixtures the
rest of the suite relies on
"""

import collections
import datetime
import itertools
import pathlib

import polars as pl
import pytest
import synth
from conftest import default_cfg
from omegaconf import OmegaConf

from cocoa.collator import Collator
from cocoa.tokenizer import Tokenizer
from cocoa.winnower import Winnower

ARTIFACTS = (
    "meds.parquet",
    "subject_splits.parquet",
    "tokens_times.parquet",
    "tokenizer.yaml",
    "train_for_inference.parquet",
    "tuning_for_inference.parquet",
    "held_out_for_inference.parquet",
)


def configured_tables() -> set:
    """every raw table the shipped default collation config reads"""
    cfg = default_cfg("collation")
    tables = {cfg["reference"]["table"]}
    tables |= {t["table"] for t in cfg["reference"].get("augmentation_tables", [])}
    tables |= {e["table"] for e in cfg["entries"]} - {"REFERENCE"}
    return tables


# --- the synthetic fixtures themselves ------------------------------------


def test_synthetic_dataset_covers_every_configured_table(raw_data):
    missing = {
        t for t in configured_tables() if not (raw_data.root / f"{t}.parquet").exists()
    }
    assert not missing, f"synth.py does not generate {missing}"


def test_synthetic_tables_match_declared_schemas(raw_data):
    for table, schema in synth.SCHEMAS.items():
        df = pl.read_parquet(raw_data.root / f"{table}.parquet")
        assert dict(df.schema) == schema, table
        assert df.height > 0, table


def test_manifest_agrees_with_generated_tables(raw_data):
    """the manifest is what every other module trusts; check it against the data"""
    hosp = pl.read_parquet(raw_data.root / "clif_hospitalization.parquet")
    adt = pl.read_parquet(raw_data.root / "clif_adt.parquet")
    vitals = pl.read_parquet(raw_data.root / "clif_vitals.parquet")
    resp = pl.read_parquet(raw_data.root / "clif_respiratory_support_processed.parquet")

    assert set(hosp["hospitalization_id"]) == set(raw_data.subject_ids)
    assert set(hosp["patient_id"]) == set(raw_data.patient_ids)
    assert raw_data.n_patients == hosp["patient_id"].n_unique()

    assert raw_data.expired == frozenset(
        hosp.filter(pl.col("discharge_category") == "Expired")["hospitalization_id"]
    )
    assert raw_data.icu == frozenset(
        adt.filter(pl.col("location_category") == "icu")["hospitalization_id"]
    )
    assert raw_data.imv == frozenset(
        resp.filter(pl.col("device_category") == "imv")["hospitalization_id"]
    )
    # the is_finite guard matters: polars compares NaN >= 130 as true, so the
    # planted NaN reading would otherwise look like a real tachycardia
    assert raw_data.tachy == frozenset(
        vitals.filter(
            (pl.col("vital_category") == "heart_rate")
            & (pl.col("vital_value") >= 130)
            & pl.col("vital_value").is_finite()
        )["hospitalization_id"]
    )
    assert raw_data.nan_vitals == frozenset(
        vitals.filter(pl.col("vital_value").is_nan())["hospitalization_id"]
    )
    assert raw_data.nan_vitals.isdisjoint(raw_data.tachy)
    for hid, hours in raw_data.los_hours.items():
        assert raw_data.discharge[hid] - raw_data.admission[hid] == datetime.timedelta(
            hours=hours
        )
    for pid, subjects in raw_data.subjects_of.items():
        assert set(subjects) == set(
            hosp.filter(pl.col("patient_id") == pid)["hospitalization_id"]
        )


def test_manifest_patient_order_is_chronological(raw_data):
    firsts = [
        min(raw_data.admission[s] for s in raw_data.subjects_of[p])
        for p in raw_data.patient_ids
    ]
    assert firsts == sorted(firsts)
    assert len(set(firsts)) == len(firsts), "ties would make the split ambiguous"


def test_synthetic_dataset_plants_both_sides_of_every_fact(raw_data):
    """each planted flag must be non-trivial: some subjects in, some out"""
    subjects = set(raw_data.subject_ids)
    for name in ("icu", "imv", "prone", "crrt", "expired", "pressor", "tachy"):
        planted = getattr(raw_data, name)
        assert planted, f"nothing planted for {name}"
        assert planted < subjects, f"{name} is planted for every subject"
    assert raw_data.long_stay_subjects(24) and raw_data.short_stay_subjects(24)


def test_split_helpers_partition_the_subjects(raw_data):
    parts = [
        set(raw_data.subjects_in_split(s)) for s in ("train", "tuning", "held_out")
    ]
    assert set().union(*parts) == set(raw_data.subject_ids)
    assert all(a.isdisjoint(b) for a in parts for b in parts if a is not b)
    assert all(parts), "every split must be non-empty for the suite to be meaningful"


# --- the pipeline ---------------------------------------------------------


def test_every_artifact_is_written(pipeline):
    for f in ARTIFACTS:
        assert (pipeline.path / f).exists(), f


def test_subject_sets_agree_across_artifacts(pipeline):
    from_meds = set(pipeline.meds["subject_id"].unique())
    from_splits = set(pipeline.splits["subject_id"])
    from_tokens = set(pipeline.tokens_times["subject_id"])
    assert from_meds == from_splits == from_tokens
    assert from_meds == set(pipeline.manifest.subject_ids)


def test_stages_write_only_their_own_artifacts(runner):
    dest = runner.dir()
    runner.collate(dest=dest)
    assert sorted(p.name for p in dest.iterdir()) == [
        "meds.parquet",
        "subject_splits.parquet",
    ]
    runner.tokenize(processed=dest)
    assert (dest / "tokenizer.yaml").exists()
    assert not any(p.name.endswith("_for_inference.parquet") for p in dest.iterdir())


def cotemporal_runs(tokens: list, times: list) -> list:
    """token multisets grouped into maximal runs of equal timestamps"""
    return [
        collections.Counter(t for _, t in grp)
        for _, grp in itertools.groupby(zip(times, tokens), key=lambda p: p[0])
    ]


def test_collation_is_reproducible(runner, pipeline):
    again = runner.collate(dest=runner.dir())
    key = ["subject_id", "time", "code", "numeric_value", "text_value"]
    reread = pl.read_parquet(again.processed_data_home / "meds.parquet")
    # row order is not stable across runs (streaming sink_parquet), content is
    assert pipeline.meds.sort(key).equals(reread.sort(key))
    splits = pl.read_parquet(again.processed_data_home / "subject_splits.parquet")
    assert pipeline.splits.sort("subject_id").equals(splits.sort("subject_id"))


def test_vocabulary_and_bins_are_reproducible(runner, pipeline):
    again = runner.full()
    assert pipeline.vocab == again.vocab
    left, right = (OmegaConf.to_container(p.tokenizer_yaml) for p in (pipeline, again))
    for c in (left, right):
        del c["created_dttm"]
    assert left == right


def test_timelines_are_reproducible_up_to_cotemporal_order(runner, pipeline):
    """
    times and token content are stable across runs, but the order of events
    sharing a timestamp is not: `ordering` only ranks prefixes, and neither the
    streaming collation write nor sort("time", "priority") is a stable sort
    """
    again = runner.full()
    a = pipeline.tokens_times.sort("subject_id")
    b = again.tokens_times.sort("subject_id")
    assert a["subject_id"].to_list() == b["subject_id"].to_list()
    assert a["times"].to_list() == b["times"].to_list()
    assert a.height > 0
    for i in range(a.height):
        assert cotemporal_runs(
            a["tokens"][i].to_list(), a["times"][i].to_list()
        ) == cotemporal_runs(b["tokens"][i].to_list(), b["times"][i].to_list()), a[
            "subject_id"
        ][i]


def test_tokenize_without_collating_first_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Tokenizer(processed_data_home=tmp_path).save_all()


def test_winnow_without_tokenizing_first_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Winnower(processed_data_home=tmp_path)


def test_collate_with_missing_raw_table_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No parquet / csv file found"):
        Collator(
            raw_data_home=tmp_path / "nowhere", processed_data_home=tmp_path / "out"
        ).save_all()


def test_pipeline_survives_a_tiny_dataset(runner):
    """a four-patient dataset should still produce well-formed timelines"""
    raw = synth.write_raw_dataset(runner.dir("raw_tiny"), n_patients=4)
    dest = runner.dir()
    runner.collate(raw=raw.root, dest=dest)
    tkzr = runner.tokenize(processed=dest)
    runner.winnow(processed=dest)
    tt = pl.read_parquet(dest / "tokens_times.parquet")
    assert tt.height == len(raw.subject_ids)
    assert set(pl.read_parquet(dest / "subject_splits.parquet")["split"]) == {
        "train",
        "tuning",
        "held_out",
    }
    assert len(tkzr) > 1
    for tokens, times in zip(tt["tokens"].to_list(), tt["times"].to_list()):
        assert len(tokens) == len(times) > 2


def test_empty_training_split_yields_a_vocabulary_of_only_unk(runner):
    """
    with a single patient the chronological split leaves train empty, and
    nothing raises: the vocabulary collapses to UNK and every token is 0
    """
    raw = synth.write_raw_dataset(runner.dir("raw_one"), n_patients=1)
    dest = runner.dir()
    runner.collate(raw=raw.root, dest=dest)
    tkzr = runner.tokenize(processed=dest)
    assert pl.read_parquet(dest / "subject_splits.parquet")["split"].to_list() == [
        "held_out"
    ]
    assert len(tkzr) == 1
    assert tkzr.lookup["to_tokenize"].to_list() == ["UNK"]
    tokens = pl.read_parquet(dest / "tokens_times.parquet")["tokens"].item().to_list()
    assert tokens and set(tokens) == {0}


# --- raw csv tables -------------------------------------------------------


def test_csv_tables_collate_like_parquet(runner):
    """load_table falls back to <table>.csv when no parquet is present"""
    as_parquet = synth.write_raw_dataset(runner.dir("raw_pq"), n_patients=6)
    as_csv = synth.write_raw_dataset(
        runner.dir("raw_csv"),
        n_patients=6,
        csv_tables=("clif_position", "clif_code_status"),
    )
    assert not (as_csv.root / "clif_position.parquet").exists()
    assert (as_csv.root / "clif_position.csv").exists()

    frames = []
    for raw in (as_parquet, as_csv):
        dest = runner.dir()
        runner.collate(raw=raw.root, dest=dest)
        frames.append(
            pl.read_parquet(dest / "meds.parquet")
            .filter(pl.col("code").str.contains(r"^(?:POSN|CODE)//"))
            .sort("subject_id", "time", "code")
        )
    assert frames[0].height > 0
    assert frames[0].equals(frames[1])


def test_csv_datetimes_must_be_iso8601(runner):
    """
    the csv path leaves times as strings for cast(pl.Datetime), which accepts
    "2024-01-01T09:00:00" but not the space-separated form many exports emit
    """
    raw = synth.write_raw_dataset(
        runner.dir("raw_csv_space"), n_patients=4, csv_tables=("clif_position",)
    )
    csv = raw.root / "clif_position.csv"
    csv.write_text(csv.read_text().replace("T", " ").replace(".000000", ""))
    with pytest.raises(pl.exceptions.InvalidOperationError, match="failed in column"):
        runner.collate(raw=raw.root, dest=runner.dir())


# --- identifier dtypes ----------------------------------------------------


def integer_id_dataset(dest: pathlib.Path) -> pathlib.Path:
    """raw tables keyed on integer ids, as many warehouses actually store them"""
    base = datetime.datetime(2024, 1, 1, 8, 0)
    hour = datetime.timedelta(hours=1)
    pl.DataFrame(
        {
            "hospitalization_id": [1001, 1002, 1003],
            "patient_id": [1, 2, 2],
            "admission_dttm": [base, base + 720 * hour, base + 1440 * hour],
            "discharge_dttm": [base + 40 * hour, base + 760 * hour, base + 1480 * hour],
            "age_at_admission": [61.0, 44.0, 44.5],
        }
    ).write_parquet(dest / "clif_hospitalization.parquet")
    pl.DataFrame(
        {
            "hospitalization_id": [1001, 1001, 1002, 1002, 1003, 1003],
            "recorded_dttm": [
                base + hour,
                base + 5 * hour,
                base + 721 * hour,
                base + 725 * hour,
                base + 1441 * hour,
                base + 1445 * hour,
            ],
            "vital_category": ["heart_rate"] * 6,
            "vital_value": [70.0, 90.0, 80.0, 100.0, 75.0, 95.0],
        }
    ).write_parquet(dest / "clif_vitals.parquet")
    return dest


def test_integer_subject_id_is_cast_to_string_by_every_stage(runner):
    """
    get_entry casts subject_id to String, so get_subject_splits must too or the
    two artifacts disagree and tokenization dies on the join
    """
    raw = integer_id_dataset(runner.dir("raw_int_ids"))
    dest = runner.dir()
    runner.collate(
        cfg=synth.minimal_collation_cfg(
            subject_splits={"train_frac": 0.5, "tuning_frac": 0.25}
        ),
        raw=raw,
        dest=dest,
    )
    meds = pl.read_parquet(dest / "meds.parquet")
    splits = pl.read_parquet(dest / "subject_splits.parquet")
    assert meds.schema["subject_id"] == pl.String
    assert splits.schema["subject_id"] == pl.String
    assert set(splits["subject_id"]) == {"1001", "1002", "1003"}

    runner.tokenize(processed=dest)
    tt = pl.read_parquet(dest / "tokens_times.parquet")
    assert tt.schema["subject_id"] == pl.String
    assert set(tt["subject_id"]) == {"1001", "1002", "1003"}
    for tokens, times in zip(tt["tokens"].to_list(), tt["times"].to_list()):
        assert len(tokens) == len(times) > 2


def test_integer_subject_id_survives_pass_through_columns(runner):
    """the pass_through_columns join is keyed on subject_id and must cast too"""
    raw = integer_id_dataset(runner.dir("raw_int_pass"))
    dest = runner.dir()
    runner.collate(
        cfg=synth.minimal_collation_cfg(
            subject_splits={"train_frac": 0.5, "tuning_frac": 0.25},
            pass_through_columns=["age_at_admission"],
        ),
        raw=raw,
        dest=dest,
    )
    splits = pl.read_parquet(dest / "subject_splits.parquet")
    assert splits.schema["subject_id"] == pl.String
    assert splits.height == 3
    assert dict(zip(splits["subject_id"], splits["age_at_admission"])) == {
        "1001": 61.0,
        "1002": 44.0,
        "1003": 44.5,
    }
    runner.tokenize(processed=dest)
    assert (dest / "tokens_times.parquet").exists()


def test_integer_group_id_keeps_a_patients_hospitalizations_together(runner):
    """grouping still works when the group key is an integer too"""
    raw = integer_id_dataset(runner.dir("raw_int_group"))
    dest = runner.dir()
    runner.collate(
        cfg=synth.minimal_collation_cfg(
            group_id="patient_id",
            subject_splits={"train_frac": 0.5, "tuning_frac": 0.25},
        ),
        raw=raw,
        dest=dest,
    )
    splits = pl.read_parquet(dest / "subject_splits.parquet")
    assert splits.schema["subject_id"] == pl.String
    by_subject = dict(zip(splits["subject_id"], splits["split"]))
    # 1002 and 1003 belong to patient 2, so they share a split
    assert by_subject["1002"] == by_subject["1003"]
