#!/usr/bin/env python3

"""collation of raw tables into meds.parquet"""

import datetime
import math
import pathlib
import re
import zoneinfo

import polars as pl
import pytest
import synth
from conftest import default_cfg, write_cfg

from cocoa.collator import Collator

UTC = datetime.timezone.utc
HOUR = datetime.timedelta(hours=1)
ADM = datetime.datetime(2024, 3, 1, 8, 0, 0)  # admission of the hand-built stay
SOFA_SCORES = (
    "sofa_cv_97",
    "sofa_cns",
    "sofa_coag",
    "sofa_liver",
    "sofa_renal",
    "sofa_resp",
)
MED_CTS = "clif_medication_admin_continuous_converted"

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


def hand_collator(
    runner, tables: dict, *, schemas: dict = None, without=(), **overrides
) -> Collator:
    """
    a collator over hand-written rows, configured by the shipped default with
    keys replaced by `overrides` and the keys in `without` deleted
    """
    cfg = default_cfg("collation")
    cfg.update(overrides)
    for key in without:
        del cfg[key]
    return Collator(
        collation_cfg=runner.cfg_path("collation", cfg),
        raw_data_home=write_tables(runner.dir(), tables, schemas),
        processed_data_home=runner.dir(),
    )


def hand_collate(
    runner, tables: dict, entries: list, *, schemas: dict = None, **overrides
) -> pl.DataFrame:
    """collate hand-written rows with a chosen subset of the shipped entries"""
    collator = hand_collator(
        runner, tables, schemas=schemas, entries=entries, **overrides
    )
    collator.save_all()
    return pl.read_parquet(collator.processed_data_home / "meds.parquet")


def stays(*hids: str) -> dict:
    """the hand-built stay copied once per id, each with a patient of its own"""
    hosp, patient = (
        BASE_TABLES["clif_hospitalization"][0],
        BASE_TABLES["clif_patient"][0],
    )
    return {
        "clif_hospitalization": [
            {**hosp, "hospitalization_id": h, "patient_id": f"P-{h}"} for h in hids
        ],
        "clif_patient": [{**patient, "patient_id": f"P-{h}"} for h in hids],
    }


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


def lab(n: float, *, hid: str = "H0", cat: str = "sodium", value=140.0) -> dict:
    """a lab resulted `n` hours after admission, having been ordered an hour before"""
    return {
        "hospitalization_id": hid,
        "lab_order_dttm": ADM + (n - 1) * HOUR,
        "lab_result_dttm": ADM + n * HOUR,
        "lab_category": cat,
        "lab_value_numeric": value,
    }


def sofa(n: float, *, hid: str = "H0", **scores) -> dict:
    """one sofa row `n` hours after the hand-built admission; unnamed scores are 0"""
    row = {
        "hospitalization_id": hid,
        "event_time": ADM + n * HOUR,
        **dict.fromkeys(SOFA_SCORES, 0),
        **scores,
    }
    row["sofa_total"] = sum(row[s] for s in SOFA_SCORES)
    return row


def med_cts(n: float, *, hid: str = "H0", med: str = "propofol") -> dict:
    """one converted continuous medication row, `n` hours after the admission"""
    return {
        "hospitalization_id": hid,
        "admin_dttm": ADM + n * HOUR,
        "med_category": med,
        "med_dose_converted": 1.0,
        "med_dose_unit_converted": "mcg/kg/min",
        "_convert_status": "success",
    }


def extra(pid: str, t: datetime.datetime, cat: str) -> dict:
    """one patient-level row of the extra-events table"""
    return {
        "patient_id": pid,
        "event_dttm": t,
        "event_date": None,
        "extra_category": cat,
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


def test_commas_and_underscore_runs_in_a_code_become_one_underscore(runner):
    """a text_value only trades whitespace for underscores, so it keeps both"""

    def assessment(n, cat, txt):
        return {
            "hospitalization_id": "H0",
            "recorded_dttm": ADM + n * HOUR,
            "assessment_category": cat,
            "numerical_value": None,
            "categorical_value": txt,
        }

    meds = hand_collate(
        runner,
        {
            "clif_patient_assessments": [
                assessment(1, "Pain, Location", "Lower Back,  Left"),
                assessment(2, "gcs__motor", "obeys__commands"),
            ]
        },
        shipped_entries("clif_patient_assessments"),
    )
    assert events(meds) == [
        (utc(ADM + HOUR), "ASMT//pain_location", None, "lower_back,_left"),
        (utc(ADM + 2 * HOUR), "ASMT//gcs_motor", None, "obeys__commands"),
    ]


def test_a_numeric_code_column_is_written_as_text(runner):
    meds = hand_collate(
        runner,
        {"clif_sofa": [sofa(1, sofa_cns=3)]},
        [
            {
                "table": "clif_sofa",
                "prefix": "SOFA",
                "code": "sofa_total",
                "time": "event_time",
            }
        ],
    )
    assert events(meds) == [(utc(ADM + HOUR), "SOFA//3", None, None)]


def test_iso8601_string_time_column_is_parsed_to_the_same_instant(runner):
    """a raw table storing `time` as iso8601 text collates like a native datetime"""
    meds = hand_collate(
        runner,
        {
            "clif_vitals": [
                {
                    "hospitalization_id": "H0",
                    "recorded_dttm": "2024-03-01T09:00:00",
                    "vital_category": "heart_rate",
                    "vital_value": 90.0,
                }
            ]
        },
        shipped_entries("clif_vitals", "VTL"),
        schemas={
            "clif_vitals": {**synth.SCHEMAS["clif_vitals"], "recorded_dttm": pl.String}
        },
    )
    assert events(meds) == [(utc(ADM + HOUR), "VTL//heart_rate", 90.0, None)]


# --- load_table --------------------------------------------------------------

# vitals keyed on the table's own name for the hospitalization
VITALS_BY_ENCOUNTER = {
    "encounter_id": pl.String,
    "recorded_dttm": synth.DTTM,
    "vital_category": pl.String,
    "vital_value": pl.Float64,
}


def by_encounter(hid: str, n: float, value: float) -> dict:
    return {
        "encounter_id": hid,
        "recorded_dttm": ADM + n * HOUR,
        "vital_category": "heart_rate",
        "vital_value": value,
    }


def test_a_parquet_table_is_read_in_preference_to_a_csv_of_the_same_name(runner):
    raw = write_tables(runner.dir(), {"clif_vitals": [vital(1)]})
    # the csv holds another reading, which collating it would turn up
    pl.DataFrame(
        [vital(2, value=50.0)], schema=synth.SCHEMAS["clif_vitals"], orient="row"
    ).write_csv(raw / "clif_vitals.csv")
    cfg = {**default_cfg("collation"), "entries": shipped_entries("clif_vitals", "VTL")}
    dest = runner.dir()
    runner.collate(cfg=cfg, raw=raw, dest=dest)
    assert events(pl.read_parquet(dest / "meds.parquet")) == [
        (utc(ADM + HOUR), "VTL//heart_rate", 90.0, None)
    ]


def test_subject_id_str_names_a_tables_own_subject_column(runner):
    meds = hand_collate(
        runner,
        with_second_stay(
            {
                "vitals_by_encounter": [
                    by_encounter("H1", 1, 80.0),
                    by_encounter("H0", 2, 90.0),
                ]
            }
        ),
        [
            {
                **shipped_entries("clif_vitals", "VTL")[0],
                "table": "vitals_by_encounter",
                "subject_id_str": "encounter_id",
            }
        ],
        schemas={"vitals_by_encounter": VITALS_BY_ENCOUNTER},
    )
    assert sorted(meds.select("subject_id", "time", "numeric_value").rows()) == [
        ("H0", utc(ADM + 2 * HOUR), 90.0),
        ("H1", utc(ADM + HOUR), 80.0),
    ]


def test_table_options_apply_in_their_documented_order(runner):
    """
    subject_id_str, then filter_expr, then with_col_expr, then agg_expr: the filter
    reads raw values that the listed with_col_expr goes on to rescale, and the lone
    agg_expr groups on the column subject_id_str supplies, seeing each subject's
    rows, written out of time order, sorted on `time`
    """
    meds = hand_collate(
        runner,
        with_second_stay(
            {
                "vitals_by_encounter": [
                    by_encounter("H0", 3, 120.0),
                    by_encounter("H0", 1, 80.0),
                    by_encounter("H0", 2, 110.0),
                    by_encounter("H1", 5, 105.0),
                ]
            }
        ),
        [
            {
                "table": "vitals_by_encounter",
                "subject_id_str": "encounter_id",
                "prefix": "LABEL",
                "filter_expr": 'pl.col("vital_value") > 100',
                "with_col_expr": [
                    '(pl.col("vital_value") / 10).alias("vital_value")',
                    'pl.lit("first_high_hr").alias("code")',
                ],
                "agg_expr": "pl.all().first()",
                "code": "code",
                "numeric_value": "vital_value",
                "time": "recorded_dttm",
            }
        ],
        schemas={"vitals_by_encounter": VITALS_BY_ENCOUNTER},
    )
    assert sorted(
        meds.select("subject_id", "time", "code", "numeric_value").rows()
    ) == [
        ("H0", utc(ADM + 2 * HOUR), "LABEL//first_high_hr", 11.0),
        ("H1", utc(ADM + 5 * HOUR), "LABEL//first_high_hr", 10.5),
    ]


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


def test_crrt_entries_carry_the_blood_flow_rate_and_mark_initiation(runner):
    def crrt(n, mode, rate):
        return {
            "hospitalization_id": "H0",
            "recorded_dttm": ADM + n * HOUR,
            "crrt_mode_category": mode,
            "blood_flow_rate": rate,
        }

    meds = hand_collate(
        runner,
        {
            "clif_crrt_therapy": [
                crrt(1, None, 150.0),
                crrt(4, "CVVHDF", 200.0),
                crrt(10, "CVVHDF", 210.0),
            ]
        },
        shipped_entries("clif_crrt_therapy"),
    )
    assert events(meds) == [
        (utc(ADM + 4 * HOUR), "CRRT//cvvhdf", 200.0, None),
        (utc(ADM + 4 * HOUR), "LABEL//crrt_init", None, None),
        (utc(ADM + 10 * HOUR), "CRRT//cvvhdf", 210.0, None),
    ]


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


def test_resp_entries_split_a_row_into_mode_device_and_settings(runner):
    def resp(n, mode, device, fio2, peep, tidal_volume):
        return {
            "hospitalization_id": "H0",
            "recorded_dttm": ADM + n * HOUR,
            "mode_category": mode,
            "device_category": device,
            "fio2_set": fio2,
            "peep_set": peep,
            "tidal_volume_set": tidal_volume,
        }

    meds = hand_collate(
        runner,
        {
            "clif_respiratory_support_processed": [
                resp(1, "Assist Control-Volume Control", "IMV", 0.5, 8.0, 450.0),
                resp(2, None, "Nasal Cannula", None, float("nan"), float("inf")),
            ]
        },
        shipped_entries("clif_respiratory_support_processed"),
    )
    # the unventilated row has no mode, and no setting that is a finite number
    assert events(meds) == [
        (utc(ADM + HOUR), "RESP//assist_control-volume_control", None, None),
        (utc(ADM + HOUR), "RESP//fio2_set", 0.5, None),
        (utc(ADM + HOUR), "RESP//imv", None, None),
        (utc(ADM + HOUR), "RESP//peep_set", 8.0, None),
        (utc(ADM + HOUR), "RESP//tidal_volume_set", 450.0, None),
        (utc(ADM + 2 * HOUR), "RESP//nasal_cannula", None, None),
    ]


# --- with_col_expr -----------------------------------------------------------


def test_sofa_component_codes_are_built_from_scores(runner):
    meds = hand_collate(
        runner,
        {
            "clif_sofa": [
                sofa(1, sofa_cns=1),
                sofa(2, sofa_cv_97=2, sofa_resp=1),
                sofa(3, sofa_coag=1, sofa_liver=2, sofa_renal=4),
            ]
        },
        shipped_entries("clif_sofa", "SOFA"),
    )
    # only components scoring above zero are emitted
    assert events(meds) == [
        (utc(ADM + HOUR), "SOFA//cns-1", None, None),
        (utc(ADM + 2 * HOUR), "SOFA//cv-2", None, None),
        (utc(ADM + 2 * HOUR), "SOFA//resp-1", None, None),
        (utc(ADM + 3 * HOUR), "SOFA//coag-1", None, None),
        (utc(ADM + 3 * HOUR), "SOFA//liver-2", None, None),
        (utc(ADM + 3 * HOUR), "SOFA//renal-4", None, None),
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
    meds = hand_collate(
        runner,
        {
            "clif_sofa": [
                sofa(1, sofa_cv_97=1),
                sofa(5, sofa_cv_97=3),
                sofa(9, sofa_cv_97=4),
            ]
        },
        shipped_entries("clif_sofa", "LABEL"),
    )
    assert events(meds) == [(utc(ADM + 5 * HOUR), "LABEL//sepsis_onset", None, None)]


def test_agg_expr_emits_nothing_when_nothing_qualifies(runner):
    meds = hand_collate(
        runner,
        {"clif_sofa": [sofa(1, sofa_cv_97=1)]},
        shipped_entries("clif_sofa", "LABEL"),
    )
    assert meds.height == 0


def test_agg_expr_groups_by_key_when_one_is_given(runner):
    """a patient-level table, aggregated per patient and then windowed onto stays"""
    meds = hand_collate(
        runner,
        with_second_stay(
            {
                "clif_extra_events": [
                    extra("P0", ADM + 5 * HOUR, "later"),
                    extra("P0", ADM + 2 * HOUR, "earlier"),
                    extra("P1", ADM + HOUR, "only"),
                ]
            }
        ),
        shipped_entries("REFERENCE", "RACE")
        + [
            {
                **EXTRA,
                "time": "event_dttm",
                "key": "patient_id",
                "agg_expr": [
                    'pl.col("extra_category").sort_by("event_dttm").first()',
                    'pl.col("event_dttm").min()',
                ],
            }
        ],
    )
    xtra = meds.filter(pl.col("code").str.starts_with("XTRA//"))
    assert sorted(xtra.select("subject_id", "time", "code").rows()) == [
        ("H0", utc(ADM + 2 * HOUR), "XTRA//earlier"),
        ("H1", utc(ADM + HOUR), "XTRA//only"),
    ]


# --- the shipped LABEL definitions -------------------------------------------

# one stay per case: the label it should raise, the table its readings go in, the
# readings as (hours after admission, category, value), and the hour the label
# lands on (None for never); a case opens with a reading that would count were its
# category ignored, then has one just short of the threshold before one exactly
# at it, or just past it where the threshold itself does not count
LABEL_CASES = {
    "hyperk": (
        "hyperk_init",
        "clif_labs",
        [(1, "phosphate", 6.6), (2, "potassium", 6.49), (3, "potassium", 6.5)],
        3,
    ),
    "hypok": (
        "hypok_init",
        "clif_labs",
        [(1, "magnesium", 1.9), (2, "potassium", 2.5), (3, "potassium", 2.49)],
        3,
    ),
    "anemia": (
        "anemia_init",
        "clif_labs",
        [(1, "albumin", 3.1), (2, "hemoglobin", 7.0), (3, "hemoglobin", 6.99)],
        3,
    ),
    "hypona": (
        "hypona_init",
        "clif_labs",
        [(1, "potassium", 4.1), (2, "sodium", 120.0), (3, "sodium", 119.9)],
        3,
    ),
    "hyperna": (
        "hyperna_init",
        "clif_labs",
        [(1, "glucose", 210.0), (2, "sodium", 159.9), (3, "sodium", 160.0)],
        3,
    ),
    "tachy": (
        "tachy_init",
        "clif_vitals",
        [(1, "sbp", 150.0), (2, "heart_rate", 129.9), (3, "heart_rate", 130.0)],
        3,
    ),
    "hypertension_sbp": (
        "hypertension_init",
        "clif_vitals",
        [(1, "map", 185.0), (2, "sbp", 179.9), (3, "sbp", 180.0)],
        3,
    ),
    "hypertension_dbp": (
        "hypertension_init",
        "clif_vitals",
        [(1, "sbp", 150.0), (2, "dbp", 119.9), (3, "dbp", 120.0)],
        3,
    ),
    "hypotension_map": (
        "hypotension_init",
        "clif_vitals",
        [(1, "dbp", 50.0), (2, "map", 65.0), (3, "map", 64.9)],
        3,
    ),
    "hypotension_sbp": (
        "hypotension_init",
        "clif_vitals",
        [(1, "map", 80.0), (2, "sbp", 90.0), (3, "sbp", 89.9)],
        3,
    ),
    "sepsis": ("sepsis_onset", "clif_sofa", [(1, None, 1), (2, None, 2)], 2),
    **{
        f"pressor_{med}": (
            "pressor_init",
            MED_CTS,
            [(1, "propofol", None), (2, med, None)],
            2,
        )
        for med in (
            "angiotensin",
            "dopamine",
            "epinephrine",
            "norepinephrine",
            "phenylephrine",
            "vasopressin",
        )
    },
    "no_pressor": (
        "pressor_init",
        MED_CTS,
        [(1, "propofol", None), (2, "dobutamine", None)],
        None,
    ),
}

READING = {
    "clif_labs": lambda hid, n, cat, value: lab(n, hid=hid, cat=cat, value=value),
    "clif_vitals": lambda hid, n, cat, value: vital(n, hid=hid, cat=cat, value=value),
    "clif_sofa": lambda hid, n, _, total: sofa(n, hid=hid, sofa_cv_97=total),
    MED_CTS: lambda hid, n, med, _: med_cts(n, hid=hid, med=med),
}


def shipped_labels() -> list:
    return [e for e in default_cfg("collation")["entries"] if e["prefix"] == "LABEL"]


@pytest.fixture(scope="module")
def labelled(tmp_path_factory) -> pl.DataFrame:
    """the shipped LABEL entries, collated once over every case's stay"""
    tables = stays(*LABEL_CASES)
    for hid, (_, table, readings, _) in LABEL_CASES.items():
        tables.setdefault(table, []).extend(READING[table](hid, *r) for r in readings)
    cfg = default_cfg("collation")
    cfg["entries"] = [e for e in shipped_labels() if e["table"] in tables]
    root = tmp_path_factory.mktemp("labelled")
    (raw := root / "raw").mkdir()
    Collator(
        collation_cfg=write_cfg(root / "collation.yaml", cfg),
        raw_data_home=write_tables(raw, tables),
        processed_data_home=root / "processed",
    ).save_all()
    return pl.read_parquet(root / "processed" / "meds.parquet")


@pytest.mark.parametrize("case", LABEL_CASES)
def test_label_lands_on_the_first_reading_its_definition_admits(labelled, case):
    label, _, _, hour = LABEL_CASES[case]
    got = labelled.filter(
        (pl.col("subject_id") == case) & (pl.col("code") == f"LABEL//{label}")
    )
    assert got["time"].to_list() == ([] if hour is None else [utc(ADM + hour * HOUR)])


def test_every_shipped_label_has_a_case():
    """crrt_init is pinned with the CRRT entry it shares a table with"""
    shipped = {
        re.fullmatch(r'pl\.lit\("(\w+)"\)\.alias\("code"\)', e["with_col_expr"])
        .group(1)
        .lower()
        for e in shipped_labels()
    }
    assert shipped == {label for label, *_ in LABEL_CASES.values()} | {"crrt_init"}


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


def test_row_with_a_null_subject_id_is_dropped(runner):
    meds = hand_collate(
        runner,
        {"clif_vitals": [vital(1), vital(2, hid=None)]},
        shipped_entries("clif_vitals", "VTL"),
    )
    assert meds.select("subject_id", "time").rows() == [("H0", utc(ADM + HOUR))]


# --- drop_duplicate_events ---------------------------------------------------


REPEATED = {"clif_vitals": [vital(1), vital(1), vital(2)]}


def test_duplicate_events_collapse_to_one(runner):
    meds = hand_collate(
        runner,
        REPEATED,
        shipped_entries("clif_vitals", "VTL"),
        drop_duplicate_events=True,
    )
    assert events(meds) == [
        (utc(ADM + HOUR), "VTL//heart_rate", 90.0, None),
        (utc(ADM + 2 * HOUR), "VTL//heart_rate", 90.0, None),
    ]


@pytest.mark.parametrize(
    "cfg",
    [{}, {"drop_duplicate_events": False}, {"without": ["drop_duplicate_events"]}],
    ids=["shipped", "false", "absent"],
)
def test_duplicate_events_survive_unless_dropping_is_asked_for(runner, cfg):
    """
    the shipped default is off, and a config without the key, as written for the
    releases before 26.9.1, keeps them as those releases did
    """
    meds = hand_collate(runner, REPEATED, shipped_entries("clif_vitals", "VTL"), **cfg)
    assert events(meds) == [
        (utc(ADM + HOUR), "VTL//heart_rate", 90.0, None),
        (utc(ADM + HOUR), "VTL//heart_rate", 90.0, None),
        (utc(ADM + 2 * HOUR), "VTL//heart_rate", 90.0, None),
    ]


def test_duplicate_events_are_dropped_within_an_entry_not_across_entries(runner):
    """two entries emitting one event keep a copy each"""
    meds = hand_collate(
        runner,
        {"clif_vitals": [vital(1), vital(1)]},
        2 * shipped_entries("clif_vitals", "VTL"),
        drop_duplicate_events=True,
    )
    assert events(meds) == 2 * [(utc(ADM + HOUR), "VTL//heart_rate", 90.0, None)]


def test_contemporaneous_events_differing_in_value_are_not_duplicates(runner):
    """same subject, time, and code, but distinct measurements"""
    meds = hand_collate(
        runner,
        {"clif_vitals": [vital(1), vital(1, value=110.0)]},
        shipped_entries("clif_vitals", "VTL"),
        drop_duplicate_events=True,
    )
    assert meds.height == 2
    assert sorted(meds["numeric_value"]) == [90.0, 110.0]


def test_duplicates_are_dropped_per_subject_not_across_subjects(runner):
    meds = hand_collate(
        runner,
        with_second_stay({"clif_vitals": [vital(1), vital(1), vital(1, hid="H1")]}),
        shipped_entries("clif_vitals", "VTL"),
        drop_duplicate_events=True,
    )
    assert meds.height == 2
    assert set(meds["subject_id"]) == {"H0", "H1"}


# --- max_lookback_days / max_lookahead_days ----------------------------------

# hours after the hand-built admission; the stay is discharged at 48
SPREAD = (-49, -48, -13, -12, -1, 1, 47, 71, 72, 73)
SPREAD_VITALS = {"clif_vitals": [vital(n) for n in SPREAD]}


def hours_after_admission(meds: pl.DataFrame) -> list:
    return sorted((t - utc(ADM)) / HOUR for t in meds["time"])


def windowed(runner, tables: dict, **overrides) -> pl.DataFrame:
    """collate hand-written vitals under a lookback / lookahead"""
    return hand_collate(
        runner, tables, shipped_entries("clif_vitals", "VTL"), **overrides
    )


@pytest.mark.parametrize(
    "limits,kept",
    [
        ({"max_lookback_days": 2, "max_lookahead_days": 1}, SPREAD[1:-1]),
        ({"max_lookback_days": 2}, SPREAD[1:]),
        ({"max_lookahead_days": 1}, SPREAD[:-1]),
        ({"max_lookback_days": 0.5, "max_lookahead_days": 0.5}, SPREAD[3:7]),
        ({"max_lookback_days": 0, "max_lookahead_days": 0}, (1, 47)),
    ],
)
def test_events_beyond_the_lookback_or_lookahead_are_dropped(runner, limits, kept):
    """a limit is inclusive, may be fractional, and can be set on either side alone"""
    meds = windowed(runner, SPREAD_VITALS, **limits)
    assert hours_after_admission(meds) == list(kept)


@pytest.mark.parametrize(
    "cfg",
    [
        {},
        {"max_lookback_days": None, "max_lookahead_days": None},
        {"without": ["max_lookback_days", "max_lookahead_days"]},
    ],
    ids=["shipped", "null", "absent"],
)
def test_without_a_lookback_or_lookahead_every_event_survives(runner, cfg):
    """the shipped default is null, and a config without the keys keeps them too"""
    meds = windowed(runner, SPREAD_VITALS, **cfg)
    assert hours_after_admission(meds) == list(SPREAD)


def test_lookback_and_lookahead_measure_from_each_subjects_own_stay(runner):
    later = {
        **BASE_TABLES["clif_hospitalization"][0],
        "hospitalization_id": "H1",
        "patient_id": "P1",
        "admission_dttm": ADM + 240 * HOUR,
        "discharge_dttm": ADM + 288 * HOUR,
    }
    meds = windowed(
        runner,
        with_second_stay(
            {
                "clif_hospitalization": BASE_TABLES["clif_hospitalization"] + [later],
                "clif_vitals": [vital(1), vital(1, hid="H1"), vital(241, hid="H1")],
            }
        ),
        max_lookback_days=1,
        max_lookahead_days=1,
    )
    # H0's event at hour 1 is inside its stay, but ten days before H1's
    assert sorted(meds.select("subject_id", "time").rows()) == [
        ("H0", utc(ADM + HOUR)),
        ("H1", utc(ADM + 241 * HOUR)),
    ]


def test_a_lookahead_day_is_24_hours_of_elapsed_time(runner):
    """so seven days on from a stay ending before a dst shift ends an hour later"""
    # the stay is discharged at 08:00 CST on 3 March; 168 hours later is
    # 09:00 CDT on 10 March, not the 08:00 CDT seven calendar days would give
    meds = windowed(
        runner,
        # 08:30 and 09:30 local time on 10 March
        {"clif_vitals": [vital(216.5), vital(217.5)]},
        default_timezone="America/Chicago",
        max_lookahead_days=7,
    )
    assert [t.hour for t in meds["time"].dt.convert_time_zone("UTC")] == [13]


def test_a_missing_bound_leaves_that_side_of_the_window_open(runner):
    """no end_time, or no reference row at all, means nothing to measure from"""
    open_ended = {**BASE_TABLES["clif_hospitalization"][0], "discharge_dttm": None}
    meds = windowed(
        runner,
        {
            "clif_hospitalization": [open_ended],
            "clif_vitals": [vital(n) for n in (-49, 1, 1000)]
            + [vital(n, hid="H9") for n in (-49, 1000)],
        },
        max_lookback_days=2,
        max_lookahead_days=1,
    )
    assert sorted(meds.select("subject_id", "time").rows()) == [
        ("H0", utc(ADM + HOUR)),
        ("H0", utc(ADM + 1000 * HOUR)),
        ("H9", utc(ADM - 49 * HOUR)),
        ("H9", utc(ADM + 1000 * HOUR)),
    ]


def test_a_subject_repeated_in_the_reference_table_raises_under_a_window(runner):
    """rather than doubling its events in the join on the window's bounds"""
    collator = hand_collator(
        runner,
        {
            "clif_hospitalization": 2 * BASE_TABLES["clif_hospitalization"],
            "clif_vitals": [vital(1)],
        },
        entries=shipped_entries("clif_vitals", "VTL"),
        max_lookback_days=1,
    )
    with pytest.raises(pl.exceptions.ComputeError):
        collator.get_all().collect()


@pytest.mark.parametrize("key", ["max_lookback_days", "max_lookahead_days"])
@pytest.mark.parametrize("value", [-1, "7"])
def test_a_negative_or_non_numeric_lookback_or_lookahead_raises(runner, key, value):
    cfg = {**default_cfg("collation"), key: value}
    with pytest.raises(AssertionError, match=key):
        runner.collate(cfg=cfg)


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


def test_code_status_lands_on_the_stay_it_was_set_in(pipeline):
    """
    code_status is keyed by patient, so each status meets every stay of its
    patient; timed by its own start_dttm, it stays on the one it was set in, at
    that time, and one set before any stay is dropped
    """
    m = pipeline.manifest
    code = pipeline.meds.filter(pl.col("code").str.starts_with("CODE//"))
    assert sorted(code.select("subject_id", "time", "code").rows()) == sorted(
        (s, utc(t), f"CODE//{c.lower()}") for s, (t, c) in m.code_status.items()
    )
    assert raw(m, "clif_code_status").height > len(m.code_status)
    # a patient's stays differ in status, so a stray couldn't pass for its own
    assert any(
        len({m.code_status[s][1] for s in m.subjects_of[p]}) > 1 for p in m.patient_ids
    )


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


def test_a_transfer_enters_at_its_in_time_and_leaves_at_its_out_time(runner):
    def adt(location, start, end):
        return {
            "hospitalization_id": "H0",
            "in_dttm": ADM + start * HOUR,
            "out_dttm": ADM + end * HOUR,
            "location_category": location,
        }

    meds = hand_collate(
        runner,
        {"clif_adt": [adt("ED", 0, 4), adt("ICU", 4, 30)]},
        shipped_entries("clif_adt"),
    )
    assert events(meds) == [
        (utc(ADM), "XFR-IN//ed", None, None),
        (utc(ADM + 4 * HOUR), "XFR-IN//icu", None, None),
        (utc(ADM + 4 * HOUR), "XFR-OUT//ed", None, None),
        (utc(ADM + 30 * HOUR), "XFR-OUT//icu", None, None),
    ]


# --- the reference block -----------------------------------------------------

# a stay-level table, joined on hospitalization_id rather than patient_id
UNITS = {"hospitalization_id": pl.String, "unit_name": pl.String}
WEIGHTS = {
    "patient_id": pl.String,
    "recorded_dttm": synth.DTTM,
    "weight_kg": pl.Float64,
}


def reference_with(*augmentation_tables) -> dict:
    """the shipped reference block with its augmentation tables replaced"""
    return {
        **default_cfg("collation")["reference"],
        "augmentation_tables": list(augmentation_tables),
    }


PATIENTS = {"table": "clif_patient", "key": "patient_id", "validation": "m:1"}


def test_augmentation_tables_join_on_their_own_keys_and_keep_every_stay(runner):
    """
    the patient table is filtered before it is joined, leaving H1's patient no
    row to join, and the left join keeps H1 all the same
    """
    tables = with_second_stay(
        {
            "clif_hospitalization_unit": [
                {"hospitalization_id": "H0", "unit_name": "Medical ICU"},
                {"hospitalization_id": "H1", "unit_name": "Cardiac Ward"},
            ]
        }
    )
    tables["clif_patient"] = [
        BASE_TABLES["clif_patient"][0],
        {
            **BASE_TABLES["clif_patient"][0],
            "patient_id": "P1",
            "race_category": "Asian",
            "sex_category": "Male",
        },
    ]
    meds = hand_collate(
        runner,
        tables,
        shipped_entries("REFERENCE", "RACE")
        + shipped_entries("REFERENCE", "ADMN")
        + [
            {
                "table": "REFERENCE",
                "prefix": "UNIT",
                "code": "unit_name",
                "time": "admission_dttm",
            }
        ],
        schemas={"clif_hospitalization_unit": UNITS},
        reference=reference_with(
            {**PATIENTS, "filter_expr": 'pl.col("sex_category") == "Female"'},
            {
                "table": "clif_hospitalization_unit",
                "key": "hospitalization_id",
                "validation": "1:1",
            },
        ),
    )
    assert sorted(meds.select("subject_id", "code").rows()) == [
        ("H0", "ADMN//inpatient"),
        ("H0", "RACE//white"),
        ("H0", "UNIT//medical_icu"),
        ("H1", "ADMN//inpatient"),
        ("H1", "UNIT//cardiac_ward"),
    ]


ONE_PATIENT_TWO_STAYS = {
    "clif_hospitalization": BASE_TABLES["clif_hospitalization"]
    + [{**BASE_TABLES["clif_hospitalization"][0], "hospitalization_id": "H1"}]
}
ONE_PATIENT_TWO_ROWS = {"clif_patient": 2 * BASE_TABLES["clif_patient"]}


@pytest.mark.parametrize(
    "validation,tables,ok",
    [
        ("m:1", ONE_PATIENT_TWO_STAYS, True),
        ("1:1", ONE_PATIENT_TWO_STAYS, False),
        ("m:1", ONE_PATIENT_TWO_ROWS, False),
    ],
    ids=["many-to-one-two-stays", "one-to-one-two-stays", "many-to-one-repeat"],
)
def test_an_augmentation_join_is_validated_as_configured(
    runner, validation, tables, ok
):
    collator = hand_collator(
        runner, tables, reference=reference_with({**PATIENTS, "validation": validation})
    )
    if ok:
        assert collator.get_reference_frame().collect().height == 2
    else:
        with pytest.raises(pl.exceptions.ComputeError):
            collator.get_reference_frame().collect()


# a patient's weights, its latest first, to be aggregated to one row per patient
WEIGHED = with_second_stay(
    {
        "clif_patient_weight": [
            {"patient_id": "P0", "recorded_dttm": ADM, "weight_kg": 72.5},
            {"patient_id": "P0", "recorded_dttm": ADM - 720 * HOUR, "weight_kg": 70.0},
            {"patient_id": "P1", "recorded_dttm": ADM, "weight_kg": 80.0},
        ]
    }
)
LATEST_WEIGHT = {
    "table": "clif_patient_weight",
    "key": "patient_id",
    "validation": "m:1",
    "agg_expr": 'pl.col("weight_kg").last()',
}


def test_an_augmentation_table_is_aggregated_by_its_key_before_the_join(runner):
    """`key` groups the table's agg_expr, whose rows are sorted on its `time`"""
    collator = hand_collator(
        runner,
        WEIGHED,
        schemas={"clif_patient_weight": WEIGHTS},
        reference=reference_with(PATIENTS, {**LATEST_WEIGHT, "time": "recorded_dttm"}),
    )
    frame = collator.get_reference_frame().collect()
    assert dict(zip(frame["hospitalization_id"], frame["weight_kg"])) == {
        "H0": 72.5,
        "H1": 80.0,
    }


@pytest.mark.parametrize("engine", ("in-memory", "streaming"))
def test_an_augmentation_agg_expr_without_a_time_takes_the_table_order(runner, engine):
    """
    with no `time` to sort on, as an augmentation table or the reference block may
    lack, the agg_expr sees rows as the table lists them, so P0's last weight is
    its earlier one; load_table sorted on None here, which raised, or under polars
    2.0's streaming engine was skipped, unless polars dropped it ahead of an
    aggregation that ignores order, such as a max
    """
    collator = hand_collator(
        runner,
        WEIGHED,
        schemas={"clif_patient_weight": WEIGHTS},
        reference=reference_with(PATIENTS, LATEST_WEIGHT),
    )
    frame = collator.get_reference_frame().collect(engine=engine)
    assert dict(zip(frame["hospitalization_id"], frame["weight_kg"])) == {
        "H0": 70.0,
        "H1": 80.0,
    }


def test_the_reference_block_takes_the_options_of_a_table(runner):
    """here a filter_expr, keeping observation stays out of the reference frame"""
    tables = with_second_stay({})
    tables["clif_hospitalization"][1] = {
        **tables["clif_hospitalization"][1],
        "admission_type_category": "Observation",
    }
    collator = hand_collator(
        runner,
        tables,
        entries=shipped_entries("REFERENCE", "ADMN"),
        reference={
            **default_cfg("collation")["reference"],
            "filter_expr": 'pl.col("admission_type_category") != "Observation"',
        },
    )
    collator.save_all()
    for f in ("meds.parquet", "subject_splits.parquet"):
        df = pl.read_parquet(collator.processed_data_home / f)
        assert df["subject_id"].to_list() == ["H0"], f


def test_the_reference_frame_is_built_once_and_reused(runner):
    collator = hand_collator(runner, {})
    assert collator.reference_frame is None
    frame = collator.get_reference_frame()
    assert collator.get_reference_frame() is frame is collator.reference_frame


DSCG_ENTRY = shipped_entries("REFERENCE", "DSCG")[0]
EXPIRED = 'pl.col("discharge_category") == "Expired"'


def p1_stays(*stays: dict) -> dict:
    """the hand-built stay H0 of P0, who goes home, then P1's stays as given"""
    tables = with_second_stay({})
    first, second = tables["clif_hospitalization"]
    tables["clif_hospitalization"] = [first] + [{**second, **s} for s in stays]
    return tables


@pytest.mark.parametrize(
    "filter_expr, kept",
    [
        (EXPIRED, [("H1", "DSCG//expired")]),
        (
            ['pl.col("patient_id") == "P1"', 'pl.col("discharge_category") == "Home"'],
            [("H2", "DSCG//home")],
        ),
    ],
    ids=["string", "list"],
)
def test_a_reference_entry_filters_its_rows(runner, filter_expr, kept):
    """as an entry over any other table does, in either form"""
    tables = p1_stays({"discharge_category": "Expired"}, {"hospitalization_id": "H2"})
    meds = hand_collate(runner, tables, [{**DSCG_ENTRY, "filter_expr": filter_expr}])
    assert sorted(meds.select("subject_id", "code").rows()) == kept


def test_a_reference_entry_works_on_the_frame_the_reference_block_builds(runner):
    """
    so its options can use a column an augmentation table joins on, here
    clif_patient's sex_category, to filter rows and derive their codes
    """
    tables = p1_stays({})
    tables["clif_patient"][1] = {**tables["clif_patient"][1], "sex_category": "Male"}
    entry = {
        **DSCG_ENTRY,
        "filter_expr": 'pl.col("sex_category") == "Male"',
        "with_col_expr": (
            '(pl.col("sex_category") + " " + pl.col("discharge_category")).alias("who")'
        ),
        "code": "who",
    }
    meds = hand_collate(runner, tables, [entry])
    assert meds.select("subject_id", "code").rows() == [("H1", "DSCG//male_home")]


def test_a_reference_entry_aggregates_by_its_key_in_time_order(runner):
    """
    here keeping each patient's latest discharge: P1's later stay comes first in
    the table, so its rows have to be sorted by the entry's time beforehand
    """
    later = ADM + 96 * HOUR
    tables = p1_stays(
        {
            "hospitalization_id": "H2",
            "admission_dttm": ADM + 72 * HOUR,
            "discharge_dttm": later,
            "discharge_category": "Expired",
        },
        {},
    )
    cols = ("hospitalization_id", "discharge_category", "discharge_dttm")
    entry = {
        **DSCG_ENTRY,
        "key": "patient_id",
        "agg_expr": [f'pl.col("{c}").last()' for c in cols],
    }
    meds = hand_collate(runner, tables, [entry])
    assert sorted(meds.select("subject_id", "code", "time").rows()) == [
        ("H0", "DSCG//home", utc(ADM + 48 * HOUR)),
        ("H2", "DSCG//expired", utc(later)),
    ]


def test_a_reference_entry_s_options_shape_its_own_events_alone(runner):
    """the reference frame stays whole for the other entries and the splits"""
    collator = hand_collator(
        runner,
        p1_stays({"discharge_category": "Expired"}),
        entries=[
            {**DSCG_ENTRY, "filter_expr": EXPIRED},
            *shipped_entries("REFERENCE", "RACE"),
        ],
    )
    collator.save_all()
    meds = pl.read_parquet(collator.processed_data_home / "meds.parquet")
    assert sorted(meds.select("subject_id", "code").rows()) == [
        ("H0", "RACE//white"),
        ("H1", "DSCG//expired"),
        ("H1", "RACE//white"),
    ]
    splits = pl.read_parquet(collator.processed_data_home / "subject_splits.parquet")
    assert sorted(splits["subject_id"]) == ["H0", "H1"]


# --- reference_key windowing and fix_date_to_time ---------------------------

EXTRA = {
    "table": "clif_extra_events",
    "prefix": "XTRA",
    "code": "extra_category",
    "reference_key": "patient_id",
}


def collate_extra(runner, entry, **overrides) -> pl.DataFrame:
    """collate the extra-events table (plus one reference entry) alone"""
    cfg = default_cfg("collation")
    cfg["entries"] = shipped_entries("REFERENCE", "RACE") + [entry]
    cfg.update(overrides)
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


def test_reference_key_keeps_events_at_either_end_of_the_stay(runner):
    """the window is closed: an event at admission or at discharge is inside it"""
    tick, end = datetime.timedelta(microseconds=1), ADM + 48 * HOUR
    meds = hand_collate(
        runner,
        {
            "clif_extra_events": [
                extra("P0", ADM - tick, "before"),
                extra("P0", ADM, "admission"),
                extra("P0", end, "discharge"),
                extra("P0", end + tick, "after"),
            ]
        },
        shipped_entries("REFERENCE", "RACE") + [{**EXTRA, "time": "event_dttm"}],
    )
    xtra = meds.filter(pl.col("code").str.starts_with("XTRA//"))
    assert sorted(xtra["code"]) == ["XTRA//admission", "XTRA//discharge"]


def test_reference_key_window_compares_instants_across_a_dst_fall_back(runner):
    """
    the stay ends at the first of chicago's two 01:30s, so an event at the second
    01:10 reads earlier on the wall clock but comes 40 minutes after discharge
    """
    discharge = datetime.datetime(2024, 11, 3, 6, 30, tzinfo=UTC)  # 01:30 CDT
    minute = datetime.timedelta(minutes=1)
    aware = pl.Datetime("us", "UTC")
    meds = hand_collate(
        runner,
        {
            "clif_hospitalization": [
                {
                    **BASE_TABLES["clif_hospitalization"][0],
                    "admission_dttm": discharge - 24 * HOUR,
                    "discharge_dttm": discharge,
                }
            ],
            "clif_extra_events": [
                extra("P0", discharge - 10 * minute, "inside"),  # 01:20 CDT
                extra("P0", discharge + 40 * minute, "outside"),  # 01:10 CST
            ],
        },
        shipped_entries("REFERENCE", "RACE") + [{**EXTRA, "time": "event_dttm"}],
        schemas={
            "clif_hospitalization": {
                **synth.SCHEMAS["clif_hospitalization"],
                "admission_dttm": aware,
                "discharge_dttm": aware,
            },
            "clif_extra_events": {
                **synth.SCHEMAS["clif_extra_events"],
                "event_dttm": aware,
            },
        },
        default_timezone="America/Chicago",
    )
    xtra = meds.filter(pl.col("code").str.starts_with("XTRA//"))
    assert xtra["code"].to_list() == ["XTRA//inside"]


def test_reference_key_entry_may_take_its_time_from_the_reference_frame(runner):
    """
    a `time` the table lacks comes from the reference frame, where the window
    keeps every row: so the CODE entry released through 26.10.3, timed by
    admission_dttm, puts every status of a patient on each of their stays, at
    its admission
    """
    m = runner.raw
    cfg = default_cfg("collation")
    cfg["entries"] = shipped_entries("REFERENCE", "RACE") + [
        {
            "table": "clif_code_status",
            "prefix": "CODE",
            "code": "code_status_category",
            "time": "admission_dttm",
            "reference_key": "patient_id",
        }
    ]
    dest = runner.dir()
    runner.collate(cfg=cfg, dest=dest)
    code = pl.read_parquet(dest / "meds.parquet").filter(
        pl.col("code").str.starts_with("CODE//")
    )
    statuses = raw(m, "clif_code_status")
    assert sorted(code.select("subject_id", "time", "code").rows()) == sorted(
        (s, utc(m.admission[s]), f"CODE//{c.lower()}")
        for p, c in zip(statuses["patient_id"], statuses["code_status_category"])
        for s in m.subjects_of[p]
    )


@pytest.mark.parametrize("zone", ["UTC", "America/Chicago", "Asia/Tokyo"])
def test_fix_date_to_time_moves_midnight_to_the_end_of_the_local_day(runner, zone):
    """the end of the day means the end of the day in `default_timezone`"""
    m = runner.raw
    xtra = collate_extra(
        runner,
        {**EXTRA, "time": "event_date", "fix_date_to_time": True},
        default_timezone=zone,
    )
    expected = set()
    for pid in m.patient_ids:
        first = m.subjects_of[pid][0]
        end_of_day = datetime.datetime.combine(
            (m.admission[first] + HOUR).date(), datetime.time(23, 59, 59)
        )
        if m.admission[first] <= end_of_day <= m.discharge[first]:
            # the raw times are naive, so every wall clock reads in `zone`
            expected.add((first, end_of_day.replace(tzinfo=zoneinfo.ZoneInfo(zone))))
    assert 0 < len(expected) < m.n_patients  # both branches are exercised
    assert set(xtra["code"]) == {"XTRA//inside_window"}
    assert {(s, t) for s, t in zip(xtra["subject_id"], xtra["time"])} == expected


@pytest.mark.parametrize(
    "column_zone,zone", [("Asia/Tokyo", "UTC"), ("UTC", "America/Chicago")]
)
def test_fix_date_to_time_reads_a_tz_aware_midnight_in_its_own_zone(
    runner, column_zone, zone
):
    """a date cast to a timestamp is midnight in the zone it was cast in"""
    tz = zoneinfo.ZoneInfo(column_zone)
    day = ADM.date()  # the hand-built stay runs 2024-03-01 08:00 local for 48h
    midnight = datetime.datetime.combine(day, datetime.time(0, 0), tzinfo=tz)
    meds = hand_collate(
        runner,
        {
            "clif_extra_events": [
                {"patient_id": "P0", "event_dttm": midnight, "extra_category": "date"}
            ]
        },
        shipped_entries("REFERENCE", "RACE")
        + [{**EXTRA, "time": "event_dttm", "fix_date_to_time": True}],
        schemas={
            "clif_extra_events": {
                "patient_id": pl.String,
                "event_dttm": pl.Datetime("us", column_zone),
                "extra_category": pl.String,
            }
        },
        default_timezone=zone,
    )
    xtra = meds.filter(pl.col("code").str.starts_with("XTRA//"))
    # the end of that same calendar day, as the column's own zone reckons it
    assert xtra["time"].to_list() == [
        datetime.datetime.combine(day, datetime.time(23, 59, 59), tzinfo=tz)
    ]
    assert xtra.schema["time"] == pl.Datetime("us", zone)


def test_fix_date_to_time_leaves_real_timestamps_alone(runner):
    """only midnight moves: an intraday reading keeps its exact instant"""
    noon = ADM + 4 * HOUR  # 2024-03-01 12:00, inside the 48h stay
    midnight = datetime.datetime(2024, 3, 2, 0, 0)  # the second day of the stay
    dawn = midnight + 6 * HOUR
    meds = hand_collate(
        runner,
        {
            "clif_extra_events": [
                {"patient_id": "P0", "event_dttm": t, "extra_category": c}
                for t, c in [(noon, "noon"), (midnight, "date"), (dawn, "dawn")]
            ]
        },
        shipped_entries("REFERENCE", "RACE")
        + [{**EXTRA, "time": "event_dttm", "fix_date_to_time": True}],
    )
    xtra = meds.filter(pl.col("code").str.starts_with("XTRA//")).sort("time")
    end_of_day = midnight + datetime.timedelta(seconds=86399)
    assert list(zip(xtra["code"], xtra["time"])) == [
        ("XTRA//noon", utc(noon)),  # untouched
        ("XTRA//dawn", utc(dawn)),  # untouched
        ("XTRA//date", utc(end_of_day)),  # moved to the end of its own day
    ]
    # the point of the flag: the date-derived event cannot leak into the past,
    # so it lands after every real event of the day it belongs to
    assert utc(end_of_day) > utc(dawn) > utc(midnight)


def test_fix_date_to_time_on_a_day_whose_local_midnight_does_not_exist(runner):
    """cuba springs forward at 00:00, so the unfixed midnight cannot localize"""
    zone = "America/Havana"
    tz = zoneinfo.ZoneInfo(zone)
    skipped = datetime.date(2024, 3, 10)  # 00:00 -> 01:00 CDT in havana
    meds = hand_collate(
        runner,
        {
            "clif_hospitalization": [
                {
                    **BASE_TABLES["clif_hospitalization"][0],
                    "admission_dttm": datetime.datetime(2024, 3, 8, 8, 0),
                    "discharge_dttm": datetime.datetime(2024, 3, 12, 8, 0),
                }
            ],
            "clif_extra_events": [
                {"patient_id": "P0", "event_date": skipped, "extra_category": "date"}
            ],
        },
        shipped_entries("REFERENCE", "RACE")
        + [{**EXTRA, "time": "event_date", "fix_date_to_time": True}],
        schemas={
            "clif_extra_events": {
                "patient_id": pl.String,
                "event_date": pl.Date,
                "extra_category": pl.String,
            }
        },
        default_timezone=zone,
    )
    xtra = meds.filter(pl.col("code").str.starts_with("XTRA//"))
    assert xtra["time"].to_list() == [
        datetime.datetime.combine(skipped, datetime.time(23, 59, 59), tzinfo=tz)
    ]


def test_without_fix_date_to_time_a_date_stays_at_midnight(runner):
    # every planted event date is the date of an 08:00 admission, so midnight
    # falls before the stay begins and the window filter drops the row
    assert collate_extra(runner, {**EXTRA, "time": "event_date"}).height == 0


def test_reference_key_entry_needs_no_reference_entry_before_it(runner):
    """its join builds the reference frame when no REFERENCE entry has yet"""
    entry = {**EXTRA, "time": "event_dttm"}
    cfg = default_cfg("collation")
    cfg["entries"] = [entry]
    dest = runner.dir()
    runner.collate(cfg=cfg, dest=dest)
    alone = pl.read_parquet(dest / "meds.parquet")
    after = collate_extra(runner, entry)  # behind a REFERENCE entry, as shipped
    assert after.height > 0
    assert alone.sort("subject_id", "time", "code").equals(
        after.sort("subject_id", "time", "code")
    )


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
