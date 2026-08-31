#!/usr/bin/env python3

"""collation of raw tables into meds.parquet"""

import datetime
import math
import pathlib
import re

import numpy as np
import polars as pl
import pytest
import synth
from conftest import default_cfg

from cocoa.collator import Collator

UTC = datetime.timezone.utc
HOUR = datetime.timedelta(hours=1)
ADM = datetime.datetime(2024, 3, 1, 8, 0, 0)  # admission of the hand-built stay

# the hand-built stay every `hand_collate` dataset is built around
BASE_TABLES = {
    "clif_hospitalization": [
        {
            "hospitalization_id": "H0",
            "patient_id": "P0",
            "admission_dttm": ADM,
            "discharge_dttm": ADM + 48 * HOUR,
            "age_at_admission": 61.5,
            "admission_type_category": "Inpatient",
            "discharge_category": "Home",
        }
    ],
    "clif_patient": [
        {
            "patient_id": "P0",
            "race_category": "White",
            "ethnicity_category": "Non-Hispanic",
            "sex_category": "Female",
            "language_category": "English",
        }
    ],
}


def utc(t: datetime.datetime) -> datetime.datetime:
    """a naive utc instant as a tz-aware one"""
    return t.replace(tzinfo=UTC)


def shipped_entries(table: str, prefix: str = None) -> list:
    """the entries the default config configures over `table`, verbatim"""
    entries = [
        e
        for e in default_cfg("collation")["entries"]
        if e["table"] == table and (prefix is None or e.get("prefix") == prefix)
    ]
    assert entries, f"no shipped entry for {table=} {prefix=}"
    return entries


def write_tables(
    dest: pathlib.Path, tables: dict, schemas: dict = None
) -> pathlib.Path:
    """write hand-written rows as raw parquet tables, using synth's schemas"""
    dest = pathlib.Path(dest)
    schemas = {**synth.SCHEMAS, **(schemas or {})}
    for name, rows in {**BASE_TABLES, **tables}.items():
        pl.DataFrame(rows, schema=schemas[name], orient="row").write_parquet(
            dest / f"{name}.parquet"
        )
    return dest


def hand_collate(
    runner, tables: dict, entries: list, *, schemas: dict = None, **overrides
) -> pl.DataFrame:
    """collate hand-written rows with a chosen subset of the shipped entries"""
    cfg = default_cfg("collation")
    cfg["entries"] = entries
    cfg.update(overrides)
    dest = runner.dir()
    runner.collate(cfg=cfg, raw=write_tables(runner.dir(), tables, schemas), dest=dest)
    return pl.read_parquet(dest / "meds.parquet")


def with_second_stay(tables: dict) -> dict:
    """the hand-built dataset plus a second hospitalization H1 of patient P1"""
    return {
        "clif_hospitalization": BASE_TABLES["clif_hospitalization"]
        + [
            {
                **BASE_TABLES["clif_hospitalization"][0],
                "hospitalization_id": "H1",
                "patient_id": "P1",
            }
        ],
        "clif_patient": BASE_TABLES["clif_patient"]
        + [{**BASE_TABLES["clif_patient"][0], "patient_id": "P1"}],
        **tables,
    }


def vital(n: int, *, hid: str = "H0", cat: str = "heart_rate", value=90.0) -> dict:
    """one vitals row, `n` hours after the hand-built admission"""
    return {
        "hospitalization_id": hid,
        "recorded_dttm": ADM + n * HOUR,
        "vital_category": cat,
        "vital_value": value,
    }


def events(meds: pl.DataFrame) -> list:
    """(time, code, numeric_value, text_value) tuples in a deterministic order"""
    return (
        meds.sort("time", "code", "text_value")
        .select("time", "code", "numeric_value", "text_value")
        .rows()
    )


def prefixes(meds: pl.DataFrame) -> set:
    return {c.split("//", 1)[0] for c in meds["code"]}


def subjects_with(meds: pl.DataFrame, code: str) -> set:
    return set(meds.filter(pl.col("code") == code)["subject_id"])


def raw(runner_or_manifest, table: str) -> pl.DataFrame:
    root = getattr(runner_or_manifest, "raw", runner_or_manifest).root
    return pl.read_parquet(root / f"{table}.parquet")


# --- schema ------------------------------------------------------------------


def test_meds_schema_is_exactly_the_meds_like_columns(pipeline):
    assert dict(pipeline.meds.schema) == {
        "subject_id": pl.String,
        "time": pl.Datetime("us", "UTC"),  # the configured default_timezone
        "code": pl.String,
        "numeric_value": pl.Float32,
        "text_value": pl.String,
    }


def test_subject_id_time_and_code_are_never_null(pipeline):
    meds = pipeline.meds
    assert meds.height > 1000
    assert meds.select(
        pl.col("subject_id", "time", "code").null_count()
    ).to_dicts() == [{"subject_id": 0, "time": 0, "code": 0}]


def test_every_collated_subject_is_a_planted_hospitalization(pipeline):
    assert set(pipeline.meds["subject_id"]) == set(pipeline.manifest.subject_ids)


# --- code construction -------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "RACE//black_or_african_american",  # "Black or African American"
        "ETHN//non-hispanic",  # "-" is not whitespace and survives
        "SEX//female",
        "ADMN//acute_care_transfer",
        "LAB-RES//white_blood_cell",  # "white blood cell"
        "LAB-ORD//white_blood_cell",
        "MED-INT//sodium_bicarbonate",
        "DSCG//skilled_nursing_facility",
        "CODE//dnr/dni",  # "DNR/DNI"; "/" is not whitespace either
        "RESP//assist_control-volume_control",
        "LABEL//hyperk_init",  # config writes pl.lit("hyperK_init")
    ],
)
def test_expected_code_is_present(pipeline, code):
    assert code in set(pipeline.meds["code"])


def test_codes_are_prefix_then_lowercased_whitespace_free_value(pipeline):
    configured = {e["prefix"] for e in default_cfg("collation")["entries"]}
    codes = set(pipeline.meds["code"])
    assert len(codes) > 50
    for c in codes:
        prefix, _, value = c.partition("//")
        assert prefix in configured, c  # prefix case is preserved verbatim
        assert value and value == value.lower(), c
        assert not re.search(r"\s", value), c


def test_output_prefixes_are_exactly_those_the_config_can_produce(pipeline):
    assert prefixes(pipeline.meds) == {
        e["prefix"] for e in default_cfg("collation")["entries"]
    }


def test_whitespace_runs_collapse_to_a_single_underscore(runner):
    meds = hand_collate(
        runner,
        {
            "clif_vitals": [
                {
                    "hospitalization_id": "H0",
                    "recorded_dttm": ADM + n * HOUR,
                    "vital_category": cat,
                    "vital_value": 90.0,
                }
                for n, cat in enumerate(
                    ["Heart  Rate", "Temp\tC", " Mean\n Arterial Pressure ", None]
                )
            ]
        },
        shipped_entries("clif_vitals", "VTL"),
    )
    # the last row's null category is dropped; nothing is stripped, so leading
    # and trailing whitespace each become an underscore
    assert events(meds) == [
        (utc(ADM), "VTL//heart_rate", 90.0, None),
        (utc(ADM + HOUR), "VTL//temp_c", 90.0, None),
        (utc(ADM + 2 * HOUR), "VTL//_mean_arterial_pressure_", 90.0, None),
    ]


def test_entry_without_a_prefix_yields_a_bare_code(runner):
    entry = dict(shipped_entries("clif_vitals", "VTL")[0])
    entry.pop("prefix")
    meds = hand_collate(
        runner,
        {
            "clif_vitals": [
                {
                    "hospitalization_id": "H0",
                    "recorded_dttm": ADM + HOUR,
                    "vital_category": "Heart Rate",
                    "vital_value": 90.0,
                }
            ]
        },
        [entry],
    )
    assert events(meds) == [(utc(ADM + HOUR), "heart_rate", 90.0, None)]


# --- filter_expr -------------------------------------------------------------


def test_posn_carries_prone_only(pipeline):
    meds, m = pipeline.meds, pipeline.manifest
    assert (
        raw(m, "clif_position").filter(pl.col("position_category") == "supine").height
    )
    assert m.prone
    posn = meds.filter(pl.col("code").str.starts_with("POSN"))
    assert set(posn["code"]) == {"POSN//prone"}
    assert set(posn["subject_id"]) == set(m.prone)


def test_crrt_null_mode_rows_are_filtered_out(pipeline):
    meds, m = pipeline.meds, pipeline.manifest
    nulls = raw(m, "clif_crrt_therapy").filter(pl.col("crrt_mode_category").is_null())
    assert nulls["hospitalization_id"].n_unique() == len(m.subject_ids)
    assert m.crrt
    crrt = meds.filter(pl.col("code").str.starts_with("CRRT"))
    assert set(crrt["code"]) == {"CRRT//cvvhdf"}
    assert set(crrt["subject_id"]) == set(m.crrt)


def test_med_cts_numeric_value_tracks_convert_status(pipeline):
    meds, m = pipeline.meds, pipeline.manifest
    src = raw(m, "clif_medication_admin_continuous_converted")
    ok = src.filter(pl.col("_convert_status") == "success").height
    bad = src.filter(pl.col("_convert_status") != "success").height
    assert ok and bad
    cts = meds.filter(pl.col("code").str.starts_with("MED-CTS"))
    assert cts.height == ok + bad
    assert cts["numeric_value"].is_not_null().sum() == ok
    assert cts["numeric_value"].is_null().sum() == bad


def test_med_cts_split_entries_keep_dose_only_when_converted(runner):
    meds = hand_collate(
        runner,
        {
            "clif_medication_admin_continuous_converted": [
                {
                    "hospitalization_id": "H0",
                    "admin_dttm": ADM + n * HOUR,
                    "med_category": "propofol",
                    "med_dose_converted": 10.0 * n,
                    "med_dose_unit_converted": "mcg/kg/min",
                    "_convert_status": status,
                }
                for n, status in enumerate(
                    ["success", "original unit dose is not recognized", None], start=1
                )
            ]
        },
        shipped_entries("clif_medication_admin_continuous_converted", "MED-CTS"),
    )
    # a null _convert_status satisfies neither branch, so that row disappears
    assert events(meds) == [
        (utc(ADM + HOUR), "MED-CTS//propofol", 10.0, None),
        (utc(ADM + 2 * HOUR), "MED-CTS//propofol", None, None),
    ]


def test_med_int_keeps_only_given_administrations(runner):
    meds = hand_collate(
        runner,
        {
            "clif_medication_admin_intermittent_converted": [
                {
                    "hospitalization_id": "H0",
                    "admin_dttm": ADM + n * HOUR,
                    "med_category": "morphine",
                    "med_dose_converted": 5.0 + n,
                    "med_dose_unit_converted": "mg",
                    "mar_action_category": action,
                    "_convert_status": status,
                }
                for n, (action, status) in enumerate(
                    [
                        ("given", "success"),
                        ("given", "user-preferred unit is not recognized"),
                        ("held", "success"),
                        (None, "success"),
                    ],
                    start=1,
                )
            ]
        },
        shipped_entries("clif_medication_admin_intermittent_converted"),
    )
    assert events(meds) == [
        (utc(ADM + HOUR), "MED-INT//morphine", 6.0, None),
        (utc(ADM + 2 * HOUR), "MED-INT//morphine", None, None),
    ]


def test_asmt_splits_into_numeric_and_text_events(runner):
    rows = [
        ("rass", -2.0, None),
        ("cam_total", None, "Positive"),
        ("cam_total", None, "Negative"),
        ("cam_total", None, "Unable to Assess"),
        ("gcs_total", None, "Verbal Response"),
        (None, 5.0, None),
    ]
    meds = hand_collate(
        runner,
        {
            "clif_patient_assessments": [
                {
                    "hospitalization_id": "H0",
                    "recorded_dttm": ADM + n * HOUR,
                    "assessment_category": cat,
                    "numerical_value": num,
                    "categorical_value": txt,
                }
                for n, (cat, num, txt) in enumerate(rows, start=1)
            ]
        },
        shipped_entries("clif_patient_assessments"),
    )
    # rass is quantitative, cam_total qualitative with Positive/Negative
    # rewritten to yes/no; the null category is dropped
    assert events(meds) == [
        (utc(ADM + HOUR), "ASMT//rass", -2.0, None),
        (utc(ADM + 2 * HOUR), "ASMT//cam_total", None, "yes"),
        (utc(ADM + 3 * HOUR), "ASMT//cam_total", None, "no"),
        (utc(ADM + 4 * HOUR), "ASMT//cam_total", None, "unable_to_assess"),
        (utc(ADM + 5 * HOUR), "ASMT//gcs_total", None, "verbal_response"),
    ]


def test_asmt_text_values_are_only_yes_and_no_by_default(pipeline):
    cam = pipeline.meds.filter(pl.col("code") == "ASMT//cam_total")
    assert cam.height > 100
    assert set(cam["text_value"]) == {"yes", "no"}
    assert cam["numeric_value"].is_null().all()


def test_resp_setting_entries_drop_non_finite_values(pipeline):
    meds, m = pipeline.meds, pipeline.manifest
    src = raw(m, "clif_respiratory_support_processed")
    for col, code in (
        ("fio2_set", "RESP//fio2_set"),
        ("peep_set", "RESP//peep_set"),
        ("tidal_volume_set", "RESP//tidal_volume_set"),
    ):
        finite = sum(v is not None and math.isfinite(v) for v in src[col])
        assert finite
        assert meds.filter(pl.col("code") == code).height == finite
    # one planted nan peep is the difference between peep and tidal volume
    assert src["peep_set"].is_nan().sum() == 1
    assert (
        meds.filter(pl.col("code") == "RESP//peep_set").height
        == meds.filter(pl.col("code") == "RESP//tidal_volume_set").height - 1
    )


# --- with_col_expr -----------------------------------------------------------


def test_sofa_component_codes_are_built_from_scores(runner):
    def sofa(n, **comps):
        row = {
            "hospitalization_id": "H0",
            "event_time": ADM + n * HOUR,
            "sofa_cv_97": 0,
            "sofa_cns": 0,
            "sofa_coag": 0,
            "sofa_liver": 0,
            "sofa_renal": 0,
            "sofa_resp": 0,
        }
        row.update(comps)
        row["sofa_total"] = sum(v for k, v in row.items() if k.startswith("sofa_"))
        return row

    meds = hand_collate(
        runner,
        {"clif_sofa": [sofa(1, sofa_cns=1), sofa(2, sofa_cv_97=2, sofa_resp=1)]},
        shipped_entries("clif_sofa", "SOFA"),
    )
    # only components scoring above zero are emitted
    assert events(meds) == [
        (utc(ADM + HOUR), "SOFA//cns-1", None, None),
        (utc(ADM + 2 * HOUR), "SOFA//cv-2", None, None),
        (utc(ADM + 2 * HOUR), "SOFA//resp-1", None, None),
    ]


def test_sofa_codes_never_carry_a_zero_score(pipeline):
    sofa = pipeline.meds.filter(pl.col("code").str.starts_with("SOFA//"))
    assert sofa.height > 100
    for c in set(sofa["code"]):
        component, _, score = c.removeprefix("SOFA//").partition("-")
        assert component in {"cv", "cns", "coag", "liver", "renal", "resp"}, c
        assert int(score) > 0, c


# --- agg_expr ----------------------------------------------------------------


def test_label_codes_occur_at_most_once_per_subject(pipeline):
    labels = pipeline.meds.filter(pl.col("code").str.starts_with("LABEL//"))
    assert labels.height > 100
    assert labels.group_by("subject_id", "code").len()["len"].max() == 1


@pytest.mark.parametrize(
    "code,planted,offset_h",
    [
        ("LABEL//hyperk_init", "hyperkalemia", 3),
        ("LABEL//crrt_init", "crrt", 4),
        ("LABEL//pressor_init", "pressor", 0),
    ],
)
def test_label_marks_the_first_qualifying_event(pipeline, code, planted, offset_h):
    m = pipeline.manifest
    expected = getattr(m, planted)
    assert expected
    assert subjects_with(pipeline.meds, code) == set(expected)
    got = dict(
        zip(*pipeline.meds.filter(pl.col("code") == code).select("subject_id", "time"))
    )
    assert got == {s: utc(m.admission[s]) + offset_h * HOUR for s in expected}


def test_tachy_label_is_also_triggered_by_a_nan_heart_rate(pipeline):
    """polars orders NaN above every number, so NaN >= 130 passes the filter"""
    meds, m = pipeline.meds, pipeline.manifest
    nan_hr = set(
        raw(m, "clif_vitals").filter(
            (pl.col("vital_category") == "heart_rate") & pl.col("vital_value").is_nan()
        )["hospitalization_id"]
    )
    assert nan_hr and not (nan_hr & set(m.tachy))
    # actual behaviour: the nan-carrying subject is labelled tachycardic too
    assert subjects_with(meds, "LABEL//tachy_init") == set(m.tachy) | nan_hr
    got = dict(
        zip(
            *meds.filter(pl.col("code") == "LABEL//tachy_init").select(
                "subject_id", "time"
            )
        )
    )
    # the planted 141 bpm reading is 2h after admission, the nan one 5h after
    assert got == {
        s: utc(m.admission[s]) + (2 if s in m.tachy else 5) * HOUR for s in got
    }


def test_agg_expr_takes_the_earliest_qualifying_row_not_the_earliest_row(runner):
    def sofa(n, total):
        return {
            "hospitalization_id": "H0",
            "event_time": ADM + n * HOUR,
            "sofa_cv_97": total,
            "sofa_cns": 0,
            "sofa_coag": 0,
            "sofa_liver": 0,
            "sofa_renal": 0,
            "sofa_resp": 0,
            "sofa_total": total,
        }

    meds = hand_collate(
        runner,
        {"clif_sofa": [sofa(1, 1), sofa(5, 3), sofa(9, 4)]},
        shipped_entries("clif_sofa", "LABEL"),
    )
    assert events(meds) == [(utc(ADM + 5 * HOUR), "LABEL//sepsis_onset", None, None)]


def test_agg_expr_emits_nothing_when_nothing_qualifies(runner):
    meds = hand_collate(
        runner,
        {
            "clif_sofa": [
                {
                    "hospitalization_id": "H0",
                    "event_time": ADM + HOUR,
                    "sofa_cv_97": 1,
                    "sofa_cns": 0,
                    "sofa_coag": 0,
                    "sofa_liver": 0,
                    "sofa_renal": 0,
                    "sofa_resp": 0,
                    "sofa_total": 1,
                }
            ]
        },
        shipped_entries("clif_sofa", "LABEL"),
    )
    assert meds.height == 0


# --- dropped rows ------------------------------------------------------------


def test_row_with_a_null_time_is_dropped(pipeline):
    meds, m = pipeline.meds, pipeline.manifest
    src = raw(m, "clif_vitals")
    affected = set(src.filter(pl.col("recorded_dttm").is_null())["hospitalization_id"])
    assert affected  # synth plants at least one null-time vital
    for hid in affected:
        kept = src.filter(
            (pl.col("hospitalization_id") == hid)
            & (pl.col("vital_category") == "heart_rate")
            & pl.col("recorded_dttm").is_not_null()
        ).height
        assert kept
        assert (
            meds.filter(
                (pl.col("subject_id") == hid) & (pl.col("code") == "VTL//heart_rate")
            ).height
            == kept
        )


def test_row_with_a_null_code_is_dropped(pipeline):
    meds, m = pipeline.meds, pipeline.manifest
    src = raw(m, "clif_patient_assessments")
    affected = set(
        src.filter(pl.col("assessment_category").is_null())["hospitalization_id"]
    )
    assert affected  # synth plants at least one null-category assessment
    for hid in affected:
        kept = src.filter(
            (pl.col("hospitalization_id") == hid)
            & pl.col("assessment_category").is_not_null()
        ).height
        assert kept
        asmt = meds.filter(
            (pl.col("subject_id") == hid) & pl.col("code").str.starts_with("ASMT")
        )
        assert asmt.height == kept
        assert not {"ASMT", "ASMT//"} & set(asmt["code"])


# --- numeric_value / text_value ---------------------------------------------


def test_numeric_value_is_populated_only_for_entries_configured_with_one(pipeline):
    entries = default_cfg("collation")["entries"]
    populated = prefixes(pipeline.meds.filter(pl.col("numeric_value").is_not_null()))
    assert populated == {e["prefix"] for e in entries if e.get("numeric_value")}
    assert len(populated) > 1


def test_text_value_is_populated_only_for_entries_configured_with_one(pipeline):
    entries = default_cfg("collation")["entries"]
    populated = prefixes(pipeline.meds.filter(pl.col("text_value").is_not_null()))
    assert populated == {e["prefix"] for e in entries if e.get("text_value")}
    assert populated


def test_lab_order_and_result_entries_differ_in_time_and_numeric_value(pipeline):
    meds = pipeline.meds
    ordered, resulted = (
        meds.filter(pl.col("code").str.starts_with(p)).with_columns(
            pl.col("code").str.strip_prefix(p).alias("cat")
        )
        for p in ("LAB-ORD//", "LAB-RES//")
    )
    assert ordered.height > 500
    assert ordered["numeric_value"].is_null().all()
    assert resulted["numeric_value"].is_not_null().all()
    # synth results every lab exactly an hour after it is ordered
    assert (
        ordered.select("subject_id", "cat", pl.col("time") + pl.duration(hours=1))
        .sort("subject_id", "cat", "time")
        .equals(
            resulted.select("subject_id", "cat", "time").sort(
                "subject_id", "cat", "time"
            )
        )
    )


def test_age_numeric_value_matches_the_raw_column(pipeline):
    m = pipeline.manifest
    src = raw(m, "clif_hospitalization")
    expected = dict(zip(src["hospitalization_id"], src["age_at_admission"]))
    age = pipeline.meds.filter(pl.col("code") == "AGE//age")
    assert age.height == len(expected)
    for hid, v in zip(age["subject_id"], age["numeric_value"]):
        assert v == pytest.approx(expected[hid], rel=1e-6)


# --- static events and coverage ---------------------------------------------


@pytest.mark.parametrize(
    "prefix,at",
    [
        ("RACE", "admission"),
        ("ETHN", "admission"),
        ("SEX", "admission"),
        ("AGE", "admission"),
        ("ADMN", "admission"),
        ("CODE", "admission"),
        ("DSCG", "discharge"),
    ],
)
def test_static_event_occurs_once_per_subject_at_the_expected_instant(
    pipeline, prefix, at
):
    m = pipeline.manifest
    when = {"admission": m.admission, "discharge": m.discharge}[at]
    df = pipeline.meds.filter(pl.col("code").str.starts_with(f"{prefix}//"))
    assert set(df["subject_id"]) == set(m.subject_ids)
    assert df.height == len(m.subject_ids)
    assert {s: t for s, t in zip(df["subject_id"], df["time"])} == {
        s: utc(t) for s, t in when.items()
    }


def test_dscg_expired_marks_exactly_the_expired_stays(pipeline):
    m = pipeline.manifest
    assert m.expired
    assert subjects_with(pipeline.meds, "DSCG//expired") == set(m.expired)


def test_xfr_in_icu_marks_exactly_the_icu_stays(pipeline):
    m = pipeline.manifest
    assert m.icu
    assert subjects_with(pipeline.meds, "XFR-IN//icu") == set(m.icu)


def test_every_subject_has_both_a_transfer_in_and_a_transfer_out(pipeline):
    meds, m = pipeline.meds, pipeline.manifest
    for prefix in ("XFR-IN//", "XFR-OUT//"):
        got = set(meds.filter(pl.col("code").str.starts_with(prefix))["subject_id"])
        assert got == set(m.subject_ids), prefix


# --- reference_key windowing and fix_date_to_time ---------------------------

EXTRA = {
    "table": "clif_extra_events",
    "prefix": "XTRA",
    "code": "extra_category",
    "reference_key": "patient_id",
}


def collate_extra(runner, entry) -> pl.DataFrame:
    """collate the extra-events table (plus one reference entry) alone"""
    cfg = default_cfg("collation")
    cfg["entries"] = shipped_entries("REFERENCE", "RACE") + [entry]
    dest = runner.dir()
    runner.collate(cfg=cfg, dest=dest)
    return pl.read_parquet(dest / "meds.parquet").filter(
        pl.col("code").str.starts_with("XTRA//")
    )


def test_reference_key_drops_events_outside_the_stay(runner):
    m = runner.raw
    src = raw(m, "clif_extra_events")
    assert set(src["extra_category"]) == {"inside window", "outside window"}
    xtra = collate_extra(runner, {**EXTRA, "time": "event_dttm"})
    # each patient's event is planted relative to their first stay, so it lands
    # on that stay only; the one ten days before admission is dropped
    assert set(xtra["code"]) == {"XTRA//inside_window"}
    assert {(s, t) for s, t in zip(xtra["subject_id"], xtra["time"])} == {
        (first := m.subjects_of[pid][0], utc(m.admission[first]) + HOUR)
        for pid in m.patient_ids
    }


def test_fix_date_to_time_moves_midnight_to_the_end_of_the_day(runner):
    m = runner.raw
    xtra = collate_extra(
        runner, {**EXTRA, "time": "event_date", "fix_date_to_time": True}
    )
    expected = set()
    for pid in m.patient_ids:
        first = m.subjects_of[pid][0]
        end_of_day = datetime.datetime.combine(
            (m.admission[first] + HOUR).date(), datetime.time(23, 59, 59)
        )
        if m.admission[first] <= end_of_day <= m.discharge[first]:
            expected.add((first, utc(end_of_day)))
    assert 0 < len(expected) < m.n_patients  # both branches are exercised
    assert set(xtra["code"]) == {"XTRA//inside_window"}
    assert {(s, t) for s, t in zip(xtra["subject_id"], xtra["time"])} == expected


def test_without_fix_date_to_time_a_date_stays_at_midnight(runner):
    # every planted event date is the date of an 08:00 admission, so midnight
    # falls before the stay begins and the window filter drops the row
    assert collate_extra(runner, {**EXTRA, "time": "event_date"}).height == 0


def test_reference_key_entry_before_any_reference_entry_raises(runner):
    """get_entry joins the cached reference frame, which nothing has filled yet"""
    cfg = default_cfg("collation")
    cfg["entries"] = [{**EXTRA, "time": "event_dttm"}]
    with pytest.raises(TypeError, match="LazyFrame"):
        runner.collate(cfg=cfg)


# --- slightly_safer_eval -----------------------------------------------------


@pytest.mark.parametrize(
    "expr,expected",
    [('str(3) + "x"', "3x"), ("int(2.7)", 2), ("float(1)", 1.0), ("bool(0)", False)],
)
def test_slightly_safer_eval_keeps_scalar_builtins(expr, expected):
    assert Collator.slightly_safer_eval(expr) == expected


def test_slightly_safer_eval_returns_polars_expressions():
    expr = Collator.slightly_safer_eval('pl.col("sofa_cv_97") > 0')
    assert isinstance(expr, pl.Expr)
    df = pl.DataFrame({"sofa_cv_97": [0, 2]})
    assert df.filter(expr)["sofa_cv_97"].to_list() == [2]


@pytest.mark.parametrize(
    "expr",
    [
        'open("/etc/passwd").read()',
        '__import__("os").listdir(".")',
        "eval('1')",
        "globals()",
    ],
)
def test_slightly_safer_eval_rejects_other_builtins(expr):
    with pytest.raises(NameError):
        Collator.slightly_safer_eval(expr)


# --- determinism -------------------------------------------------------------


def test_collating_twice_gives_identical_content(runner):
    a, b = runner.dir("a"), runner.dir("b")
    runner.collate(dest=a)
    runner.collate(dest=b)
    order = ("subject_id", "time", "code", "numeric_value", "text_value")
    left, right = (
        pl.read_parquet(d / "meds.parquet").sort(order, nulls_last=True) for d in (a, b)
    )
    assert left.height > 1000
    assert left.equals(right)
