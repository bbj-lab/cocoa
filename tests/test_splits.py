#!/usr/bin/env python3

"""subject splits: labels, group integrity, chronology, fractions, pass-through"""

import collections
import datetime
import pathlib

import omegaconf
import polars as pl
import pytest
import synth
from conftest import default_cfg

DAY = datetime.timedelta(days=1)
HOUR = datetime.timedelta(hours=1)
STAY_BASE = datetime.datetime(2024, 3, 1, 8, 0, 0)
SPLITS = ("train", "tuning", "held_out")

# three groups, bounds int(3 * cumsum([0.34, 0.33])) -> (1, 2): one per split
THIRDS = {"train_frac": 0.34, "tuning_frac": 0.33}

# clif_hospitalization / clif_patient columns copied through by the default config
FROM_HOSPITALIZATION = ("age_at_admission", "admission_type_category")
FROM_PATIENT = (
    "race_category",
    "ethnicity_category",
    "sex_category",
    "language_category",
)


def _splits_of(collator) -> pl.DataFrame:
    return pl.read_parquet(collator.processed_data_home / "subject_splits.parquet")


def _map(df: pl.DataFrame) -> dict:
    """subject_id -> split"""
    return dict(zip(df["subject_id"], df["split"]))


def _subject_counts(df: pl.DataFrame) -> dict:
    return dict(collections.Counter(df["split"]))


def _patient_counts(df: pl.DataFrame, manifest) -> dict:
    """number of distinct patients per split"""
    groups = collections.defaultdict(set)
    for sid, split in zip(df["subject_id"], df["split"]):
        groups[split].add(manifest.patient_of[sid])
    return {k: len(v) for k, v in groups.items()}


def _first_admission(manifest) -> dict:
    """patient_id -> earliest admission over that patient's stays"""
    return {
        pid: min(manifest.admission[h] for h in hids)
        for pid, hids in manifest.subjects_of.items()
    }


def _blocks(keys: list, lo: int, hi: int) -> dict:
    """label a chronologically ordered list of keys by block membership"""
    return {
        k: "train" if n < lo else "tuning" if n < hi else "held_out"
        for n, k in enumerate(keys)
    }


def _raw_pass_through(manifest) -> dict:
    """subject_id -> the pass-through values as they sit in the raw tables"""
    hosp = pl.read_parquet(manifest.root / "clif_hospitalization.parquet")
    patient = pl.read_parquet(manifest.root / "clif_patient.parquet")
    by_patient = {
        r["patient_id"]: {c: r[c] for c in FROM_PATIENT}
        for r in patient.iter_rows(named=True)
    }
    return {
        r["hospitalization_id"]: {
            **{c: r[c] for c in FROM_HOSPITALIZATION},
            **by_patient[r["patient_id"]],
        }
        for r in hosp.iter_rows(named=True)
    }


def _stays_raw(runner, *, duplicate: str = None) -> pathlib.Path:
    """
    four stays over three patients, ten days apart; PA holds both the earliest
    and the latest stay, so grouping and non-grouping disagree
    """
    stays = (("H0", "PA", 0), ("H1", "PB", 10), ("H2", "PC", 20), ("H3", "PA", 30))
    if duplicate is not None:
        stays += tuple(s for s in stays if s[0] == duplicate)
    hosp = [
        {
            "hospitalization_id": hid,
            "patient_id": pid,
            "admission_dttm": STAY_BASE + off * DAY,
            "discharge_dttm": STAY_BASE + (off + 1) * DAY,
        }
        for hid, pid, off in stays
    ]
    vitals = [
        {
            "hospitalization_id": h["hospitalization_id"],
            "recorded_dttm": h["admission_dttm"] + HOUR,
            "vital_value": 80.0 + n,
        }
        for n, h in enumerate(hosp)
    ]
    return synth.write_minimal_dataset(
        runner.dir(), hospitalizations=hosp, vitals=vitals
    )


def _spec_raw(runner, spec, *, reverse: bool = False) -> pathlib.Path:
    """
    stays given as (hospitalization_id, patient_id, admit_day, discharge_day),
    days offset from STAY_BASE; a None admit day is written as a null admission
    """
    hosp = [
        {
            "hospitalization_id": hid,
            "patient_id": pid,
            "admission_dttm": None if adm is None else STAY_BASE + adm * DAY,
            "discharge_dttm": STAY_BASE + dis * DAY,
        }
        for hid, pid, adm, dis in spec
    ]
    vitals = [
        {
            "hospitalization_id": h["hospitalization_id"],
            "recorded_dttm": h["discharge_dttm"] - HOUR,
            "vital_value": 80.0 + n,
        }
        for n, h in enumerate(hosp)
    ]
    if reverse:  # the same rows, written in the opposite order
        hosp, vitals = hosp[::-1], vitals[::-1]
    return synth.write_minimal_dataset(
        runner.dir(), hospitalizations=hosp, vitals=vitals
    )


def _int_id_raw(runner) -> pathlib.Path:
    """two stays over two patients whose ids are integers rather than strings"""
    dest = runner.dir()
    ids = {"hospitalization_id": pl.Int64, "patient_id": pl.Int64}
    tables = {
        "clif_hospitalization": [
            {
                "hospitalization_id": 10,
                "patient_id": 1,
                "admission_dttm": STAY_BASE,
                "discharge_dttm": STAY_BASE + 2 * DAY,
                "age_at_admission": 61.5,
                "admission_type_category": "Inpatient",
                "discharge_category": "Home",
            },
            {
                "hospitalization_id": 11,
                "patient_id": 2,
                "admission_dttm": STAY_BASE + 10 * DAY,
                "discharge_dttm": STAY_BASE + 12 * DAY,
                "age_at_admission": 40.0,
                "admission_type_category": "Inpatient",
                "discharge_category": "Home",
            },
        ],
        "clif_vitals": [
            {
                "hospitalization_id": 10,
                "recorded_dttm": STAY_BASE + HOUR,
                "vital_category": "heart_rate",
                "vital_value": 80.0,
            },
            {
                "hospitalization_id": 11,
                "recorded_dttm": STAY_BASE + 10 * DAY + HOUR,
                "vital_category": "heart_rate",
                "vital_value": 90.0,
            },
        ],
    }
    for table, rows in tables.items():
        schema = {k: ids.get(k, v) for k, v in synth.SCHEMAS[table].items()}
        pl.DataFrame(rows, schema=schema, orient="row").write_parquet(
            dest / f"{table}.parquet"
        )
    return dest


# --- labels and coverage -----------------------------------------------------


def test_only_the_three_expected_labels_appear(pipeline):
    assert pipeline.splits.height == 48
    assert set(pipeline.splits["split"]) == set(SPLITS)


def test_meds_and_splits_cover_the_same_subjects_exactly_once(pipeline):
    splits, meds = pipeline.splits, pipeline.meds
    assert splits.height > 0 and meds.height > 0
    assert splits["subject_id"].n_unique() == splits.height  # the 1:1 join holds
    assert set(splits["subject_id"]) == set(meds["subject_id"])
    assert set(splits["subject_id"]) == set(pipeline.manifest.subject_ids)


def test_a_subject_with_no_events_is_still_split(runner):
    """
    the splits come from the reference table, so an event-free stay is labeled
    even though it never reaches meds or a tokenized timeline
    """
    hosp = [
        {"admission_dttm": STAY_BASE, "discharge_dttm": STAY_BASE + 2 * DAY},
        {
            "admission_dttm": STAY_BASE + 10 * DAY,
            "discharge_dttm": STAY_BASE + 12 * DAY,
        },
    ]
    vitals = [{"recorded_dttm": STAY_BASE + HOUR, "vital_value": 80.0}]  # H0 only
    processed = runner.minimal(hospitalizations=hosp, vitals=vitals)
    assert _map(processed.splits) == {"H0": "train", "H1": "train"}
    assert set(processed.meds["subject_id"]) == {"H0"}
    assert processed.tokens_times["subject_id"].to_list() == ["H0"]


# --- group integrity and chronology ------------------------------------------


def test_every_stay_of_a_patient_shares_one_split(pipeline):
    manifest = pipeline.manifest
    assigned = _map(pipeline.splits)
    multi = [p for p, hids in manifest.subjects_of.items() if len(hids) > 1]
    assert len(multi) == 8  # synth gives every fifth patient a second stay
    for pid in multi:
        hids = manifest.subjects_of[pid]
        assert all(h in assigned for h in hids), pid
        assert len({assigned[h] for h in hids}) == 1, pid


def test_splits_are_chronological_blocks_of_patients(pipeline):
    manifest = pipeline.manifest
    assigned = _map(pipeline.splits)
    first = _first_admission(manifest)
    assert len(set(first.values())) == len(first)  # no ties to break
    by_split = collections.defaultdict(list)
    for pid, t in first.items():
        by_split[assigned[manifest.subjects_of[pid][0]]].append(t)
    assert all(by_split[s] for s in SPLITS), by_split.keys()
    assert max(by_split["train"]) < min(by_split["tuning"])
    assert max(by_split["tuning"]) < min(by_split["held_out"])


def test_groups_are_ordered_by_their_earliest_admission(runner):
    """
    PA is admitted first but discharged in the middle and holds the latest
    admission of the four stays, while PB is discharged last: only ranking the
    groups by *min* admission puts PA in train, PB in tuning, PC held out
    """
    spec = (
        ("HA1", "PA", 0, 1),
        ("HA2", "PA", 30, 31),
        ("HB", "PB", 10, 40),
        ("HC", "PC", 20, 21),
    )
    cfg = synth.minimal_collation_cfg(group_id="patient_id", subject_splits=THIRDS)
    df = _splits_of(runner.collate(cfg=cfg, raw=_spec_raw(runner, spec)))
    assert _map(df) == {
        "HA1": "train",
        "HA2": "train",
        "HB": "tuning",
        "HC": "held_out",
    }


def test_a_null_admission_sorts_a_group_first(runner):
    """current behavior: polars sorts nulls first, so PA lands in train"""
    spec = (("HA", "PA", None, 5), ("HB", "PB", 10, 11), ("HC", "PC", 20, 21))
    cfg = synth.minimal_collation_cfg(group_id="patient_id", subject_splits=THIRDS)
    df = _splits_of(runner.collate(cfg=cfg, raw=_spec_raw(runner, spec)))
    assert _map(df) == {"HA": "train", "HB": "tuning", "HC": "held_out"}


def test_tied_first_times_keep_the_block_sizes(runner):
    """
    PA and PB share an admission instant; which of them is ranked first is not
    specified (polars sort is not stable), but the block sizes are, so the tie
    is broken *within* train/tuning and PC still sorts last
    """
    spec = (("HA", "PA", 0, 1), ("HB", "PB", 0, 2), ("HC", "PC", 20, 21))
    cfg = synth.minimal_collation_cfg(group_id="patient_id", subject_splits=THIRDS)
    assigned = _map(_splits_of(runner.collate(cfg=cfg, raw=_spec_raw(runner, spec))))
    assert len(assigned) == 3
    assert {assigned["HA"], assigned["HB"]} == {"train", "tuning"}
    assert assigned["HC"] == "held_out"


def test_reference_row_order_does_not_change_the_assignment(runner):
    """the ranking is by admission time, not by position in the raw table"""
    spec = (
        ("HA", "PA", 0, 1),
        ("HB", "PB", 10, 11),
        ("HC", "PC", 20, 21),
        ("HD", "PD", 30, 31),
    )
    cfg = synth.minimal_collation_cfg(
        group_id="patient_id", subject_splits={"train_frac": 0.5, "tuning_frac": 0.25}
    )
    # 4 patients, bounds int(4 * cumsum([0.5, 0.25])) -> (2, 3)
    expected = {"HA": "train", "HB": "train", "HC": "tuning", "HD": "held_out"}
    for reverse in (False, True):
        raw = _spec_raw(runner, spec, reverse=reverse)
        assert _map(_splits_of(runner.collate(cfg=cfg, raw=raw))) == expected, reverse


# --- block sizes and fractions -----------------------------------------------


def test_patient_and_subject_counts_are_the_expected_block_sizes(pipeline):
    patients = _patient_counts(pipeline.splits, pipeline.manifest)
    assert patients == {"train": 28, "tuning": 3, "held_out": 9}  # 40 patients
    subjects = _subject_counts(pipeline.splits)
    assert subjects == {"train": 33, "tuning": 4, "held_out": 11}  # 48 stays


def test_assignment_of_every_subject_matches_the_chronological_blocks(pipeline):
    """
    the first 28 patients by first admission train, the next 3 tune, the last 9
    are held out (see test_default_fractions_truncate_the_tuning_block for the
    28 / 31 bounds), and every stay inherits its patient's label
    """
    manifest = pipeline.manifest
    first = _first_admission(manifest)
    by_patient = _blocks(sorted(first, key=first.get), 28, 31)
    assert collections.Counter(by_patient.values()) == {
        "train": 28,
        "tuning": 3,
        "held_out": 9,
    }
    assigned = _map(pipeline.splits)
    assert len(assigned) == len(manifest.subject_ids) == 48
    assert assigned == {h: by_patient[p] for h, p in manifest.patient_of.items()}


def test_default_fractions_truncate_the_tuning_block(pipeline):
    """
    0.1 * 40 patients is 4, but cumsum([0.7, 0.1]) * 40 lands just under 32 in
    binary floating point and is truncated, so tuning gets 3 -- current behavior
    """
    counts = _patient_counts(pipeline.splits, pipeline.manifest)
    assert counts["tuning"] == 3
    assert counts["train"] + counts["tuning"] == 31


@pytest.mark.parametrize(
    "train_frac,tuning_frac,expected",
    [
        # int(40 * cumsum) -> (20, 30), (32, 40), (40, 40)
        (0.5, 0.25, {"train": 20, "tuning": 10, "held_out": 10}),
        (0.8, 0.2, {"train": 32, "tuning": 8, "held_out": 0}),
        (1.0, 0.0, {"train": 40, "tuning": 0, "held_out": 0}),
    ],
)
def test_configured_fractions_set_the_patient_counts(
    runner, train_frac, tuning_frac, expected
):
    cfg = default_cfg("collation")
    cfg["subject_splits"] = {"train_frac": train_frac, "tuning_frac": tuning_frac}
    df = _splits_of(runner.collate(cfg=cfg))
    assert df.height == 48
    assert _patient_counts(df, runner.raw) == {k: v for k, v in expected.items() if v}
    assert set(df["split"]) == {k for k, v in expected.items() if v}


@pytest.mark.parametrize(
    "fracs,expected",
    [
        # current behavior: nothing checks that the fractions are a partition
        (
            {"train_frac": 0.9, "tuning_frac": 0.5},
            ("train", "train", "train", "tuning"),
        ),
        (
            {"train_frac": 0.0, "tuning_frac": 0.5},
            ("tuning", "tuning") + 2 * ("held_out",),
        ),
        ({"train_frac": -0.5, "tuning_frac": 0.75}, ("tuning",) + 3 * ("held_out",)),
    ],
)
def test_degenerate_fractions_are_accepted_as_given(runner, fracs, expected):
    spec = (
        ("HA", "PA", 0, 1),
        ("HB", "PB", 10, 11),
        ("HC", "PC", 20, 21),
        ("HD", "PD", 30, 31),
    )
    cfg = synth.minimal_collation_cfg(group_id="patient_id", subject_splits=fracs)
    df = _splits_of(runner.collate(cfg=cfg, raw=_spec_raw(runner, spec)))
    assert _map(df) == dict(zip(("HA", "HB", "HC", "HD"), expected))


@pytest.mark.parametrize("missing", ["tuning_frac", "subject_splits"])
def test_missing_split_fractions_raise(runner, missing):
    """the fractions are read as attributes, so an absent key is not defaulted"""
    cfg = synth.minimal_collation_cfg(
        subject_splits={"train_frac": 0.5, "tuning_frac": 0.5}
    )
    if missing == "subject_splits":
        del cfg["subject_splits"]
    else:
        del cfg["subject_splits"][missing]
    spec = (("HA", "PA", 0, 1), ("HB", "PB", 10, 11))
    with pytest.raises(omegaconf.errors.ConfigAttributeError, match=missing):
        runner.collate(cfg=cfg, raw=_spec_raw(runner, spec))


# --- group_id ----------------------------------------------------------------


def test_without_group_id_splits_are_keyed_on_subject(runner):
    manifest = runner.raw
    cfg = default_cfg("collation")
    del cfg["group_id"]
    df = _splits_of(runner.collate(cfg=cfg))
    assert df.height == 48 and df["subject_id"].n_unique() == 48
    # int(48 * cumsum([0.7, 0.1])) -> (33, 38), now counting stays not patients
    order = sorted(manifest.subject_ids, key=lambda h: manifest.admission[h])
    assert _map(df) == _blocks(order, 33, 38)
    assert _subject_counts(df) == {"train": 33, "tuning": 5, "held_out": 10}
    # the stay-indexed bounds land on a different patient than the grouped ones
    assert _patient_counts(df, manifest) == {"train": 28, "tuning": 4, "held_out": 8}


def test_without_group_id_a_patients_stays_can_straddle_splits(runner):
    fracs = {"train_frac": 0.5, "tuning_frac": 0.25}
    grouped, ungrouped = (default_cfg("collation") for _ in range(2))
    grouped["subject_splits"] = ungrouped["subject_splits"] = fracs
    del ungrouped["group_id"]

    manifest = runner.raw
    multi = [p for p, hids in manifest.subjects_of.items() if len(hids) > 1]
    assert multi

    def straddlers(cfg):
        assigned = _map(_splits_of(runner.collate(cfg=cfg)))
        assert len(assigned) == 48
        return {
            p for p in multi if len({assigned[h] for h in manifest.subjects_of[p]}) > 1
        }

    split_up = straddlers(ungrouped)
    assert split_up  # at least one patient is torn apart without group_id
    assert not straddlers(grouped)


def test_group_id_keeps_a_patients_stays_together(runner):
    cfg = synth.minimal_collation_cfg(
        group_id="patient_id", subject_splits={"train_frac": 0.5, "tuning_frac": 0.5}
    )
    df = _splits_of(runner.collate(cfg=cfg, raw=_stays_raw(runner)))
    # 3 patients, bounds int(3 * [0.5, 1.0]) -> (1, 3): PA alone in train, and
    # H3 rides along with it despite being the latest stay of the four
    assert _map(df) == {"H0": "train", "H3": "train", "H1": "tuning", "H2": "tuning"}


def test_no_group_id_splits_one_patients_stays_apart(runner):
    cfg = synth.minimal_collation_cfg(
        subject_splits={"train_frac": 0.5, "tuning_frac": 0.5}
    )
    df = _splits_of(runner.collate(cfg=cfg, raw=_stays_raw(runner)))
    # 4 stays, bounds int(4 * [0.5, 1.0]) -> (2, 4): the two earliest stays train
    assert _map(df) == {"H0": "train", "H1": "train", "H2": "tuning", "H3": "tuning"}


# --- pass-through columns ----------------------------------------------------


def test_pass_through_columns_carry_raw_values(pipeline):
    df = pipeline.splits
    assert df.columns == [
        "subject_id",
        "split",
        "start_time",
        "end_time",
        *FROM_HOSPITALIZATION,
        *FROM_PATIENT,
    ]
    assert df.schema["age_at_admission"] == pl.Float64
    raw = _raw_pass_through(pipeline.manifest)
    assert len(raw) == df.height == 48
    checked = 0
    for row in df.iter_rows(named=True):
        want = raw[row["subject_id"]]
        assert row["age_at_admission"] == pytest.approx(want["age_at_admission"])
        for c in FROM_HOSPITALIZATION[1:] + FROM_PATIENT:
            assert row[c] == want[c], (row["subject_id"], c)
        checked += 1
    assert checked == 48
    # the two stays of a patient share the patient-level values but not the age
    two = [h for p, h in pipeline.manifest.subjects_of.items() if len(h) > 1][0]
    ages = df.filter(pl.col("subject_id").is_in(two))["age_at_admission"].to_list()
    assert len(ages) == 2 and ages[0] != ages[1]


def test_no_pass_through_columns_yields_only_subject_and_split(runner):
    cfg = default_cfg("collation")
    del cfg["pass_through_columns"]
    df = _splits_of(runner.collate(cfg=cfg))
    assert df.columns == ["subject_id", "split", "start_time", "end_time"]
    assert df.height == 48


def test_pass_through_columns_reach_the_inference_frame(pipeline):
    """every pass-through column, stay-level ones included, carries raw values"""
    inference = pipeline.inference("held_out")
    assert set(FROM_HOSPITALIZATION + FROM_PATIENT) <= set(inference.columns)
    assert set(inference["subject_id"]) <= set(pipeline.subjects_in_split("held_out"))
    raw = _raw_pass_through(pipeline.manifest)
    checked = 0
    for row in inference.iter_rows(named=True):
        want = raw[row["subject_id"]]
        assert row["age_at_admission"] == pytest.approx(want["age_at_admission"])
        for c in FROM_HOSPITALIZATION[1:] + FROM_PATIENT:
            assert row[c] == want[c], (row["subject_id"], c)
        checked += 1
    assert checked == inference.height > 0


@pytest.mark.parametrize(
    "columns,error,match",
    [
        (["nope"], pl.exceptions.ColumnNotFoundError, "nope"),
        (
            ["age_at_admission", "age_at_admission"],
            pl.exceptions.DuplicateError,
            "age_at_admission",
        ),
    ],
)
def test_bad_pass_through_columns_raise(runner, columns, error, match):
    cfg = synth.minimal_collation_cfg(pass_through_columns=columns)
    spec = (("HA", "PA", 0, 1), ("HB", "PB", 10, 11))
    with pytest.raises(error, match=match):
        runner.collate(cfg=cfg, raw=_spec_raw(runner, spec))


def test_pass_through_may_repeat_the_subject_id_column(runner):
    """current behavior: the id is copied under its raw name as well"""
    cfg = synth.minimal_collation_cfg(pass_through_columns=["hospitalization_id"])
    spec = (("HA", "PA", 0, 1), ("HB", "PB", 10, 11))
    df = _splits_of(runner.collate(cfg=cfg, raw=_spec_raw(runner, spec)))
    assert df.columns == [
        "subject_id",
        "split",
        "start_time",
        "end_time",
        "hospitalization_id",
    ]
    assert df["subject_id"].to_list() == df["hospitalization_id"].to_list()


# --- inference frames --------------------------------------------------------


def test_inference_frames_partition_by_split(pipeline):
    """
    each frame holds exactly the subjects of its split that survive the 24h
    default threshold -- winnowing must not leak subjects across splits
    """
    long_stays = set(pipeline.manifest.long_stay_subjects(24))
    assert len(long_stays) == 24  # of 48 stays; the rest are winnowed out
    for split in SPLITS:
        subjects = pipeline.inference(split)["subject_id"].to_list()
        assert len(set(subjects)) == len(subjects), split
        want = set(pipeline.subjects_in_split(split)) & long_stays
        assert set(subjects) == want, split


# --- duplicate reference rows ------------------------------------------------


def test_duplicate_reference_row_fails_the_pass_through_join(runner):
    cfg = synth.minimal_collation_cfg(pass_through_columns=["age_at_admission"])
    with pytest.raises(pl.exceptions.ComputeError, match="1:1"):
        runner.collate(cfg=cfg, raw=_stays_raw(runner, duplicate="H0"))


def test_duplicate_reference_row_duplicates_a_grouped_subject(runner):
    """
    current behavior: with group_id and no pass_through_columns nothing
    validates the reference table, so a repeated stay is emitted twice and
    only trips the tokenizer's m:1 join later
    """
    cfg = synth.minimal_collation_cfg(group_id="patient_id")
    dest = runner.dir()
    df = _splits_of(
        runner.collate(cfg=cfg, raw=_stays_raw(runner, duplicate="H0"), dest=dest)
    )
    assert df.height == 5
    assert dict(collections.Counter(df["subject_id"]))["H0"] == 2
    with pytest.raises(pl.exceptions.ComputeError, match="m:1"):
        runner.tokenize(processed=dest)


def test_duplicate_reference_row_fails_the_split_join_without_group_id(runner):
    """
    the group_by on subject_id collapses the pair, but joining the start/end
    times back on gets the repeat again, so the 1:1 validation catches it
    """
    cfg = synth.minimal_collation_cfg()
    with pytest.raises(pl.exceptions.ComputeError, match="1:1"):
        runner.collate(cfg=cfg, raw=_stays_raw(runner, duplicate="H0"))
