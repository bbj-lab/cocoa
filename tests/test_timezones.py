#!/usr/bin/env python3

"""timezone handling: localization, instant preservation, local-hour clocks"""

import datetime
import zoneinfo

import polars as pl
import pytest
import synth
from conftest import default_cfg
from omegaconf import OmegaConf

UTC = datetime.timezone.utc
CHICAGO = "America/Chicago"
D = datetime.datetime


def localized(naive: D, zone: str) -> D:
    """read a naive wall clock in `zone` and return the utc instant it denotes"""
    return naive.replace(tzinfo=zoneinfo.ZoneInfo(zone)).astimezone(UTC)


def hospitalization(admit: D, discharge: D) -> list:
    return [{"admission_dttm": admit, "discharge_dttm": discharge}]


def vitals(*times: D) -> list:
    # a null value keeps the code unbinned, so timelines read VTL//heart_rate
    return [{"recorded_dttm": t, "vital_value": None} for t in times]


def run_minimal(runner, zone, hosp, vits, **kwargs):
    """collate/tokenize a minimal dataset with `default_timezone` set to `zone`"""
    return runner.minimal(
        hospitalizations=hosp,
        vitals=vits,
        collation=synth.minimal_collation_cfg(default_timezone=zone),
        **kwargs,
    )


def instants(processed, zone: str = "UTC") -> list:
    """the collated event instants, expressed in `zone`, in chronological order"""
    return (
        processed.meds.select(pl.col("time").dt.convert_time_zone(zone))
        .sort("time")["time"]
        .to_list()
    )


def with_prefix(words, prefix: str) -> list:
    return [w for w in words if w.startswith(prefix)]


def canonical(meds: pl.DataFrame) -> pl.DataFrame:
    """utc instants in a deterministic row order, for cross-zone comparison"""
    return (
        meds.with_row_index("i")
        .with_columns(pl.col("time").dt.convert_time_zone("UTC"))
        .sort("subject_id", "time", "code", "i")
        .drop("i")
    )


CLOCK_HOSP = hospitalization(D(2024, 1, 15, 0, 0), D(2024, 1, 17, 0, 0))
CLOCK_VITS = vitals(D(2024, 1, 15, 3, 30), D(2024, 1, 15, 21, 30))


def first_times(meds: pl.DataFrame) -> dict:
    """subject_id -> earliest collated instant, in utc"""
    return dict(
        meds.group_by("subject_id")
        .agg(pl.col("time").min().dt.convert_time_zone("UTC"))
        .iter_rows()
    )


def test_shipped_default_stores_utc_instants(pipeline):
    """the default config's UTC zone is carried by meds and tokens_times"""
    assert pipeline.meds.height > 1000
    assert pipeline.meds.schema["time"] == pl.Datetime("us", "UTC")
    assert pipeline.tokens_times.schema["times"].inner == pl.Datetime("us", "UTC")
    # the raw tables are naive utc, so every timeline starts at its admission
    assert first_times(pipeline.meds) == {
        hid: admit.replace(tzinfo=UTC)
        for hid, admit in pipeline.manifest.admission.items()
    }


@pytest.mark.parametrize("zone", ["UTC", CHICAGO, "Asia/Tokyo"])
def test_meds_time_zone_matches_configured_default_timezone(runner, zone):
    cfg = default_cfg("collation")
    cfg["default_timezone"] = zone
    dest = runner.dir()
    runner.collate(cfg=cfg, dest=dest)
    meds = pl.read_parquet(dest / "meds.parquet")
    assert meds.height > 1000
    assert meds.schema["time"] == pl.Datetime("us", zone)
    # the raw times are naive: each is read as a wall clock reading in `zone`
    assert first_times(meds) == {
        hid: localized(admit, zone) for hid, admit in runner.raw.admission.items()
    }


def test_naive_input_denotes_different_instants_in_different_zones(runner):
    """the same naive reading is 6h apart in winter between utc and chicago"""
    reading = D(2024, 1, 15, 12, 0)
    hosp = hospitalization(D(2024, 1, 15, 0, 0), D(2024, 1, 17, 0, 0))
    got = {
        zone: run_minimal(runner, zone, hosp, vitals(reading), naive_times=True)
        for zone in ("UTC", CHICAGO)
    }
    # stored wall clock is unchanged; only the zone attached to it differs
    for zone, processed in got.items():
        assert processed.meds.height == 1
        assert processed.meds.schema["time"] == pl.Datetime("us", zone)
        assert instants(processed, zone) == [
            reading.replace(tzinfo=zoneinfo.ZoneInfo(zone))
        ]
    assert instants(got["UTC"]) == [reading.replace(tzinfo=UTC)]
    assert instants(got[CHICAGO]) == [localized(reading, CHICAGO)]
    assert instants(got[CHICAGO])[0] - instants(got["UTC"])[0] == datetime.timedelta(
        hours=6
    )


def test_tz_aware_input_is_instant_preserving(runner):
    """new-york times collated as chicago == naive utc times collated as utc"""
    naive = synth.write_raw_dataset(runner.dir("raw_naive"), n_patients=8)
    aware = synth.write_raw_dataset(
        runner.dir("raw_aware"), tz="America/New_York", n_patients=8
    )
    cfg = default_cfg("collation")
    cfg["default_timezone"] = CHICAGO
    dest_utc, dest_chi = runner.dir(), runner.dir()
    runner.collate(raw=naive.root, dest=dest_utc)  # shipped default: UTC
    runner.collate(cfg=cfg, raw=aware.root, dest=dest_chi)
    meds_utc = pl.read_parquet(dest_utc / "meds.parquet")
    meds_chi = pl.read_parquet(dest_chi / "meds.parquet")
    assert meds_utc.height > 100
    assert meds_utc.schema["time"] == pl.Datetime("us", "UTC")
    assert meds_chi.schema["time"] == pl.Datetime("us", CHICAGO)
    assert canonical(meds_utc).equals(canonical(meds_chi))


def test_ambiguous_naive_local_time_takes_the_later_instant(runner):
    """2024-11-03 01:30 happens twice in chicago; the CST reading wins"""
    reading = D(2024, 11, 3, 1, 30)
    processed = run_minimal(
        runner,
        CHICAGO,
        hospitalization(D(2024, 11, 3, 0, 0), D(2024, 11, 4, 0, 0)),
        vitals(reading),
        naive_times=True,
    )
    assert instants(processed) == [D(2024, 11, 3, 7, 30, tzinfo=UTC)]  # 01:30 CST
    assert instants(processed) != [D(2024, 11, 3, 6, 30, tzinfo=UTC)]  # 01:30 CDT
    assert instants(processed, CHICAGO)[0].utcoffset() == datetime.timedelta(hours=-6)


def test_nonexistent_naive_local_time_raises(runner):
    """2024-03-10 02:30 is skipped by the chicago spring forward"""
    with pytest.raises(pl.exceptions.ComputeError):
        run_minimal(
            runner,
            CHICAGO,
            hospitalization(D(2024, 3, 10, 0, 0), D(2024, 3, 11, 0, 0)),
            vitals(D(2024, 3, 10, 2, 30)),
            naive_times=True,
        )


@pytest.mark.parametrize("omitted", [True, False], ids=["omitted", "null"])
def test_absent_default_timezone_falls_back_to_utc(runner, omitted):
    reading = D(2024, 1, 15, 12, 0)
    cfg = synth.minimal_collation_cfg()
    if omitted:
        cfg.pop("default_timezone")
    else:
        cfg["default_timezone"] = None
    processed = runner.minimal(
        hospitalizations=hospitalization(D(2024, 1, 15, 0, 0), D(2024, 1, 17, 0, 0)),
        vitals=vitals(reading),
        collation=cfg,
        naive_times=True,
    )
    assert processed.meds.schema["time"] == pl.Datetime("us", "UTC")
    assert instants(processed) == [reading.replace(tzinfo=UTC)]


def test_spacer_tokens_are_identical_across_zones_for_equal_instants(runner):
    """duration math runs on instants, so spacers do not depend on the zone"""
    start = D(2024, 1, 15, 0, 0)
    hosp = hospitalization(start, D(2024, 1, 20, 0, 0))
    # gaps of 20m, 3h, 31h and 2m: the last is below the 5m minimum spacer
    times = vitals(
        start,
        start + datetime.timedelta(minutes=20),
        start + datetime.timedelta(hours=3, minutes=20),
        start + datetime.timedelta(hours=34, minutes=20),
        start + datetime.timedelta(hours=34, minutes=22),
    )
    tkzr_cfg = default_cfg("tokenization")
    tkzr_cfg["insert_spacers"] = True
    got = {
        zone: run_minimal(runner, zone, hosp, times, tz="UTC", tokenization=tkzr_cfg)
        for zone in ("UTC", CHICAGO)
    }
    timelines = {zone: p.timeline("H0") for zone, p in got.items()}
    assert timelines["UTC"] == [
        "BOS",
        "VTL//heart_rate",
        "TIME//15m-1h",
        "VTL//heart_rate",
        "TIME//2h-6h",
        "VTL//heart_rate",
        "TIME//1d-3d",
        "VTL//heart_rate",
        "VTL//heart_rate",
        "EOS",
    ]
    assert timelines[CHICAGO] == timelines["UTC"]
    # without clock tokens the zone leaves no trace in the vocabulary either
    assert got[CHICAGO].vocab == got["UTC"].vocab


def test_naive_readings_across_spring_forward_lose_an_hour(runner):
    """
    identical wall clock readings are not identical durations: 25h of chicago
    wall clock across the spring forward is only 24h of elapsed time
    """
    hosp = hospitalization(D(2024, 3, 9, 0, 0), D(2024, 3, 12, 0, 0))
    times = vitals(D(2024, 3, 9, 12, 0), D(2024, 3, 10, 13, 0))
    tkzr_cfg = default_cfg("tokenization")
    tkzr_cfg["insert_spacers"] = True
    got = {
        zone: run_minimal(
            runner, zone, hosp, times, naive_times=True, tokenization=tkzr_cfg
        )
        for zone in ("UTC", CHICAGO)
    }
    elapsed = {zone: instants(p)[-1] - instants(p)[0] for zone, p in got.items()}
    assert elapsed["UTC"] == datetime.timedelta(hours=25)
    assert elapsed[CHICAGO] == datetime.timedelta(hours=24)
    # 1500 vs 1440 minutes straddle the 1d spacer break
    assert with_prefix(got["UTC"].timeline("H0"), "TIME//") == ["TIME//1d-3d"]
    assert with_prefix(got[CHICAGO].timeline("H0"), "TIME//") == ["TIME//12h-1d"]


def test_winnowing_threshold_follows_instants_not_wall_clock(runner):
    """the 24h threshold is met in utc but exactly missed in chicago"""
    hosp = hospitalization(D(2024, 3, 9, 0, 0), D(2024, 3, 12, 0, 0))
    times = vitals(D(2024, 3, 9, 12, 0), D(2024, 3, 10, 13, 0))
    win_cfg = default_cfg("winnowing")
    win_cfg["splits"] = ["train"]  # minimal_collation_cfg puts everyone in train
    got = {
        zone: run_minimal(
            runner, zone, hosp, times, naive_times=True, winnowing=win_cfg
        )
        for zone in ("UTC", CHICAGO)
    }
    kept = got["UTC"].inference("train")
    assert kept.height == 1
    assert kept["s_total_duration"].to_list() == [25 * 3600]
    assert kept["last_valid"].to_list() == [2]  # BOS and the first reading
    # 24h elapsed is not strictly greater than the 24h threshold
    assert got[CHICAGO].inference("train").height == 0


def clock_run(runner, zone, hosp=None, vits=None, **kwargs):
    """tokenize a minimal dataset with clocks under `default_timezone=zone`"""
    tkzr_cfg = default_cfg("tokenization")
    tkzr_cfg["insert_clocks"] = True
    return run_minimal(
        runner,
        zone,
        hosp if hosp is not None else CLOCK_HOSP,
        vits if vits is not None else CLOCK_VITS,
        tokenization=tkzr_cfg,
        **kwargs,
    )


def test_clock_tokens_carry_the_local_hour(runner):
    """
    events span 03:30-21:30 utc == 21:30-15:30 chicago, so the clock tokens
    are the chicago hours 00, 04, 08, 12 rather than the utc hours
    """
    processed = clock_run(runner, CHICAGO, tz="UTC")
    words = processed.timeline("H0")
    assert words == [
        "BOS",
        "VTL//heart_rate",
        "CLCK//00",
        "CLCK//04",
        "CLCK//08",
        "CLCK//12",
        "VTL//heart_rate",
        "EOS",
    ]
    assert processed.tokens_times.schema["times"].inner == pl.Datetime("us", CHICAGO)
    stamps = processed.tokens_times["times"].item().to_list()
    clocks = [(w, t) for w, t in zip(words, stamps) if w.startswith("CLCK//")]
    assert len(clocks) == 4
    for word, stamp in clocks:
        assert f"{stamp.hour:02d}" == word.removeprefix("CLCK//")
        assert stamp.minute == 0 and stamp.second == 0
    # midnight in chicago is 06:00 utc that january day
    assert clocks[0][1].astimezone(UTC) == D(2024, 1, 15, 6, 0, tzinfo=UTC)


def test_clock_tokens_differ_between_zones_for_the_same_instants(runner):
    """
    the tokenizer-transfer hazard: identical instants and identical tokenizer
    configs, but a vocabulary that depends on the collation zone
    """
    got = {zone: clock_run(runner, zone, tz="UTC") for zone in ("UTC", CHICAGO)}
    assert instants(got["UTC"]) == instants(got[CHICAGO])
    assert with_prefix(got["UTC"].timeline("H0"), "CLCK//") == [
        "CLCK//04",
        "CLCK//08",
        "CLCK//12",
        "CLCK//16",
        "CLCK//20",
    ]
    assert with_prefix(got[CHICAGO].timeline("H0"), "CLCK//") == [
        "CLCK//00",
        "CLCK//04",
        "CLCK//08",
        "CLCK//12",
    ]
    assert with_prefix(sorted(got["UTC"].vocab), "CLCK//") != with_prefix(
        sorted(got[CHICAGO].vocab), "CLCK//"
    )
    # nothing in tokenizer.yaml records which zone produced those hours
    assert OmegaConf.to_container(
        got["UTC"].tokenizer_yaml.cfg
    ) == OmegaConf.to_container(got[CHICAGO].tokenizer_yaml.cfg)
    assert "default_timezone" not in OmegaConf.to_container(
        got["UTC"].tokenizer_yaml.cfg
    )


def test_clock_tokens_are_identical_for_equal_naive_readings(runner):
    """
    the flip side: identical wall clock readings give identical clock tokens in
    every zone, even though they denote instants 6h apart
    """
    got = {zone: clock_run(runner, zone, naive_times=True) for zone in ("UTC", CHICAGO)}
    expected = ["CLCK//04", "CLCK//08", "CLCK//12", "CLCK//16", "CLCK//20"]
    for processed in got.values():
        assert with_prefix(processed.timeline("H0"), "CLCK//") == expected
    assert instants(got[CHICAGO])[0] - instants(got["UTC"])[0] == datetime.timedelta(
        hours=6
    )


def test_clock_tokens_follow_local_hours_across_a_dst_shift(runner):
    """
    a chicago stay spanning the spring forward keeps producing local 4-hourly
    clock tokens, so the same token maps to different utc offsets
    """
    processed = clock_run(
        runner,
        CHICAGO,
        hospitalization(D(2024, 3, 9, 0, 0), D(2024, 3, 13, 0, 0)),
        vitals(D(2024, 3, 9, 20, 0), D(2024, 3, 11, 20, 0)),
        tz="UTC",
    )
    words = processed.timeline("H0")
    stamps = processed.tokens_times["times"].item().to_list()
    clocks = [(w, t) for w, t in zip(words, stamps) if w.startswith("CLCK//")]
    # 2024-03-09 14:00 through 2024-03-11 15:00, chicago local
    assert [w for w, _ in clocks] == [
        "CLCK//16",
        "CLCK//20",
        "CLCK//00",
        "CLCK//04",
        "CLCK//08",
        "CLCK//12",
        "CLCK//16",
        "CLCK//20",
        "CLCK//00",
        "CLCK//04",
        "CLCK//08",
        "CLCK//12",
    ]
    for word, stamp in clocks:
        assert f"{stamp.hour:02d}" == word.removeprefix("CLCK//")
    assert clocks[0][1].utcoffset() == datetime.timedelta(hours=-6)  # CST
    assert clocks[-1][1].utcoffset() == datetime.timedelta(hours=-5)  # CDT
    assert clocks[0][1].astimezone(UTC) == D(2024, 3, 9, 22, 0, tzinfo=UTC)  # 16:00 CST
    assert clocks[5][1].astimezone(UTC) == D(
        2024, 3, 10, 17, 0, tzinfo=UTC
    )  # 12:00 CDT
    assert clocks[-1][1].astimezone(UTC) == D(2024, 3, 11, 17, 0, tzinfo=UTC)
    # the day of the shift is one hour short between local midnights
    first, second = (t.astimezone(UTC) for w, t in clocks if w == "CLCK//00")
    assert second - first == datetime.timedelta(hours=23)


def test_tz_aware_csv_input_silently_drops_the_offset(runner):
    """
    BUG (asserting current behaviour): scan_csv reads offset-bearing datetimes
    as strings, so `to_default_tz` misses the tz-aware branch and localizes the
    already-local reading -- new-york times land 5h early with no warning
    """
    raw = synth.write_raw_dataset(
        runner.dir("raw_csv"),
        n_patients=4,
        tz="America/New_York",
        csv_tables=("clif_vitals",),
    )
    dest = runner.dir()
    runner.collate(raw=raw.root, dest=dest)  # shipped default: UTC
    meds = pl.read_parquet(dest / "meds.parquet")
    vtl = meds.filter(pl.col("code").str.starts_with("VTL//"))
    assert vtl.height > 10
    observed = first_times(vtl)
    # the first vital of each stay is recorded at admission
    true_instants = {hid: a.replace(tzinfo=UTC) for hid, a in raw.admission.items()}
    dropped_offset = {
        hid: t.astimezone(zoneinfo.ZoneInfo("America/New_York")).replace(tzinfo=UTC)
        for hid, t in true_instants.items()
    }
    assert observed == dropped_offset
    assert observed != true_instants
    # the parquet-sourced reference times keep their offset, so a stay's own
    # vitals now start five hours before its admission event
    race = meds.filter(pl.col("code").str.starts_with("RACE//"))
    assert race.height == len(true_instants)
    assert first_times(race) == true_instants
