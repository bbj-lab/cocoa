#!/usr/bin/env python3

"""
winnowing: thresholding timelines into a past and a future, and flagging
outcome tokens in each tense
"""

import datetime
import os
import pathlib
import shutil

import polars as pl
import pytest
import synth
from conftest import Processed, default_cfg, write_cfg
from omegaconf.errors import ConfigAttributeError, ConfigKeyError

from cocoa.tokenizer import Tokenizer
from cocoa.winnower import Winnower

SPLITS = ("train", "tuning", "held_out")
DAY_S = 86400
HORIZON_S = 18000  # 5h; short enough to truncate every synthetic future
HOUR_S = 3600
ADMIT = datetime.datetime(2024, 1, 1, 8, 0, 0)
# seconds after admission of each subject's readings: H0 opens before its
# admission and H1 after it
OFF_START = {
    "H0": (-2 * HOUR_S, 23 * HOUR_S, 25 * HOUR_S),
    "H1": (3 * HOUR_S, 25 * HOUR_S, 28 * HOUR_S),
}
INPUTS = ("tokens_times.parquet", "subject_splits.parquet", "tokenizer.yaml")
# the non-wildcard entries of the shipped default outcome_tokens
LITERAL_OUTCOMES = ("XFR-IN//icu", "RESP//imv", "DSCG//expired", "ASMT//cam_total_yes")
STRUCTURAL = ("tokens_past", "tokens_future", "s_elapsed_past")


def seed(runner, pipeline) -> pathlib.Path:
    """a fresh directory holding only the artifacts the winnower reads"""
    dest = runner.dir()
    for f in INPUTS:
        shutil.copy(pipeline.path / f, dest / f)
    return dest


def cfg(**overrides) -> dict:
    """the shipped default winnowing config with keys replaced"""
    out = default_cfg("winnowing")
    out.update(overrides)
    return out


def rewinnow(runner, pipeline, split="held_out", **overrides) -> pl.DataFrame:
    """winnow the session tokenization again into an isolated directory"""
    dest = seed(runner, pipeline)
    runner.winnow(cfg=cfg(splits=[split], **overrides), processed=dest)
    return pl.read_parquet(dest / f"{split}_for_inference.parquet")


def winnower(runner, processed, **overrides) -> Winnower:
    """a winnower over `processed` with default keys replaced, not yet run"""
    return Winnower(
        winnowing_cfg=runner.cfg_path("winnowing", cfg(**overrides)),
        processed_data_home=processed,
    )


def offsets(times) -> list:
    """seconds from the first time, in plain python"""
    ts = list(times)
    return [int((t - ts[0]).total_seconds()) for t in ts]


def inference_files(path) -> set:
    return {p.name for p in pathlib.Path(path).glob("*_for_inference.parquet")}


def minimal_at(runner, readings: dict, unit: str = "us"):
    """
    collate and tokenize subjects all admitted at ADMIT, each with unbinned
    readings at the given seconds after it, everyone in train
    """
    return runner.minimal(
        hospitalizations=[
            {
                "hospitalization_id": h,
                "admission_dttm": ADMIT,
                "discharge_dttm": ADMIT + datetime.timedelta(days=3),
            }
            for h in readings
        ],
        vitals=[
            {
                "hospitalization_id": h,
                "recorded_dttm": ADMIT + datetime.timedelta(seconds=s),
            }
            for h, ss in readings.items()
            for s in ss
        ],
        collation=synth.minimal_collation_cfg(default_time_unit=unit),
    )


def spo2_twice(runner):
    """
    H0 reads heart rate at 0h, 2h, and 4h after ADMIT and spo2 at 1h and 3h;
    H1 reads only heart rate, at 0h and 1h
    """
    cats = {
        "H0": ("heart_rate", "spo2") * 2 + ("heart_rate",),
        "H1": ("heart_rate",) * 2,
    }
    return runner.minimal(
        hospitalizations=[
            {
                "admission_dttm": ADMIT,
                "discharge_dttm": ADMIT + datetime.timedelta(days=1),
            }
        ]
        * 2,
        vitals=[
            {
                "hospitalization_id": h,
                "recorded_dttm": ADMIT + datetime.timedelta(hours=n),
                "vital_category": c,
            }
            for h, cs in cats.items()
            for n, c in enumerate(cs)
        ],
    )


def winnow_train(runner, processed, threshold: dict, **extra) -> pl.DataFrame:
    """winnow the training split of a minimal dataset at `threshold`"""
    runner.winnow(
        cfg={
            "outcome_tokens": ["VTL//*"],
            "threshold": threshold,
            "splits": ["train"],
            **extra,
        },
        processed=processed.path,
    )
    return processed.inference("train")


def past_hours(d) -> dict:
    """each subject's past, as hours after its start_time"""
    return {
        r["subject_id"]: [
            (t - r["start_time"]).total_seconds() / HOUR_S
            for t in r["times"][: r["last_valid"]]
        ]
        for r in d.iter_rows(named=True)
    }


def outcomes(pipeline) -> set:
    """
    what the shipped default outcome_tokens should expand to, derived from the
    vocabulary without going through fnmatch
    """
    labels = {k for k in pipeline.vocab if k.startswith("LABEL//")}
    assert len(labels) > 1, "no LABEL tokens to expand against"
    assert set(LITERAL_OUTCOMES) <= set(pipeline.vocab)
    return labels | set(LITERAL_OUTCOMES)


@pytest.fixture(scope="module")
def extras(tmp_path_factory, pipeline) -> Processed:
    """
    the session collation tokenized with both optional per-token columns,
    hours_to_end_time and numeric_values, and its held_out winnowed by default
    """
    dest = tmp_path_factory.mktemp("extras")
    cfgs = tmp_path_factory.mktemp("extras_cfg")
    for f in ("meds.parquet", "subject_splits.parquet"):
        shutil.copy(pipeline.path / f, dest / f)
    tok = default_cfg("tokenization")
    tok.update(include_hours_to_end_time=True, include_numeric_values=True)
    Tokenizer(
        tokenization_cfg=write_cfg(cfgs / "tokenization.yaml", tok),
        processed_data_home=dest,
    ).save_all()
    Winnower(
        winnowing_cfg=write_cfg(cfgs / "winnowing.yaml", cfg(splits=["held_out"])),
        processed_data_home=dest,
    ).save_all()
    return Processed(dest, pipeline.manifest)


# --- output files ---------------------------------------------------------


def test_default_config_writes_one_file_per_configured_split(pipeline):
    assert default_cfg("winnowing")["splits"] == list(SPLITS)
    assert inference_files(pipeline.path) == {
        f"{s}_for_inference.parquet" for s in SPLITS
    }


@pytest.mark.parametrize("splits", ([], ["held_out"], ["train", "tuning"]))
def test_only_configured_splits_are_written(runner, pipeline, splits):
    dest = seed(runner, pipeline)
    runner.winnow(cfg=cfg(splits=splits), processed=dest)
    assert inference_files(dest) == {f"{s}_for_inference.parquet" for s in splits}


def test_without_splits_only_held_out_is_prepared(runner, pipeline):
    c = cfg()
    del c["splits"]
    dest = seed(runner, pipeline)
    runner.winnow(cfg=c, processed=dest)
    assert inference_files(dest) == {"held_out_for_inference.parquet"}
    d = pl.read_parquet(dest / "held_out_for_inference.parquet")
    assert d.equals(pipeline.inference("held_out"))


def test_null_splits_raises_type_error(runner, pipeline):
    """
    BUG (current behaviour asserted): a splits key whose entries are all
    commented out reads as null, not as a missing key, so .get hands it back
    instead of its held_out fallback, and save_all fails iterating it, a
    TypeError, before writing any split, where the visualizer reads a section
    emptied this way as none
    """
    dest = seed(runner, pipeline)
    with pytest.raises(TypeError):
        runner.winnow(cfg=cfg(splits=None), processed=dest)
    assert inference_files(dest) == set()


@pytest.mark.parametrize("horizon", ({}, {"horizon_after_threshold_s": HOUR_S}))
@pytest.mark.parametrize(
    "threshold",
    (
        {"duration_s": DAY_S},
        {"after_start_time_s": DAY_S},
        {"first_occurrence": "VTL//heart_rate"},
    ),
)
def test_a_split_without_subjects_gets_an_empty_file(runner, threshold, horizon):
    """
    everyone is in train, yet tuning and held_out get files, with no rows, typed
    as train's are; polars slices a list to List(Null) on an empty frame, which
    panicked flagging outcomes, under first_occurrence and, from polars 2.0, under
    after_start_time_s, until the winnower cast each slice back to its type
    """
    p = minimal_at(runner, OFF_START)
    assert set(p.splits["split"]) == {"train"}
    runner.winnow(
        cfg={
            "outcome_tokens": ["VTL//*"],
            "threshold": threshold,
            "splits": list(SPLITS),
            **horizon,
        },
        processed=p.path,
    )
    train = p.inference("train")
    assert train.height > 0
    for split in ("tuning", "held_out"):
        d = p.inference(split)
        assert d.height == 0 and d.schema == train.schema, split


def test_output_home_keeps_two_winnowings_of_one_dataset_apart(runner, pipeline):
    dest = seed(runner, pipeline)
    by_day, by_icu = (runner.root / "winnowed" / n for n in ("24h", "icu"))
    assert not by_day.parent.exists()  # the winnower should make it
    w = runner.winnow(cfg=cfg(splits=["held_out"]), processed=dest, output=by_day)
    assert w.output_home == by_day.resolve() != w.processed_data_home
    runner.winnow(
        cfg=cfg(splits=["held_out"], threshold={"first_occurrence": "XFR-IN//icu"}),
        processed=dest,
        output=by_icu,
    )
    assert inference_files(dest) == set()  # the processed dir is only read
    for d in (by_day, by_icu):
        # each output is a processed dir of its own
        assert {f.name for f in d.iterdir()} == {
            *INPUTS,
            "held_out_for_inference.parquet",
        }
        for f in INPUTS:
            assert (d / f).read_bytes() == (dest / f).read_bytes(), f
    # each matches winnowing that configuration in place
    day = pl.read_parquet(by_day / "held_out_for_inference.parquet")
    assert day.equals(pipeline.inference("held_out"))
    icu = pl.read_parquet(by_icu / "held_out_for_inference.parquet")
    assert icu.equals(
        rewinnow(runner, pipeline, threshold={"first_occurrence": "XFR-IN//icu"})
    )
    assert set(icu["subject_id"]) != set(day["subject_id"])


def test_output_home_skips_subdirectories_and_earlier_winnowings(runner, pipeline):
    """
    output dirs nested in the processed dir, as the docs suggest, aren't copied
    into one another, nor is a frame winnowed in place, perhaps differently
    """
    dest = seed(runner, pipeline)
    (dest / "notes.txt").write_text("hello")
    stale = "train_for_inference.parquet"
    shutil.copy(pipeline.path / stale, dest / stale)
    for name in ("winnowed-a", "winnowed-b"):
        runner.winnow(cfg=cfg(splits=["held_out"]), processed=dest, output=dest / name)
    assert {f.name for f in (dest / "winnowed-b").iterdir()} == {
        *INPUTS,
        "notes.txt",
        "held_out_for_inference.parquet",
    }


@pytest.mark.parametrize("output", (None, "same", "dotted", "dotdot", "symlink"))
def test_output_home_defaults_to_the_processed_dir(runner, pipeline, output):
    """any spelling of the processed dir is the processed dir, a symlink too"""
    dest = seed(runner, pipeline)
    link = runner.root / "link"
    if output == "symlink":
        link.symlink_to(dest, target_is_directory=True)
    output = {
        "same": dest,
        "dotted": dest / ".",
        "dotdot": dest / ".." / dest.name,
        "symlink": link,
    }.get(output)
    w = runner.winnow(cfg=cfg(splits=["held_out"]), processed=dest, output=output)
    assert w.output_home == w.processed_data_home == dest.resolve()
    assert w.copy_processed_data() == []
    assert {f.name for f in dest.iterdir()} == {
        *INPUTS,
        "held_out_for_inference.parquet",
    }


@pytest.mark.parametrize("processed", ("relative", "symlink"))
def test_output_home_matches_a_relative_or_symlinked_processed_dir(
    runner, pipeline, monkeypatch, processed
):
    """
    the processed dir is resolved before it's compared, so an output_home
    naming the same dir is it, and no file is copied onto itself
    """
    dest = seed(runner, pipeline)
    if processed == "relative":
        monkeypatch.chdir(dest.parent)
        processed = pathlib.Path(dest.name)
    else:
        processed = runner.root / "link"
        processed.symlink_to(dest, target_is_directory=True)
    w = runner.winnow(cfg=cfg(splits=["held_out"]), processed=processed, output=dest)
    assert w.processed_data_home == w.output_home == dest.resolve()
    assert w.copy_processed_data() == []
    assert {f.name for f in dest.iterdir()} == {
        *INPUTS,
        "held_out_for_inference.parquet",
    }


def test_processed_and_output_homes_expand_a_leading_tilde(
    runner, pipeline, monkeypatch
):
    dest = seed(runner, pipeline)
    for var in ("HOME", "USERPROFILE"):  # the latter on windows
        monkeypatch.setenv(var, str(runner.root))
    w = Winnower(
        winnowing_cfg=runner.cfg_path("winnowing", cfg(splits=["held_out"])),
        processed_data_home=f"~/{dest.name}",
        output_home="~/out",
    )
    assert w.processed_data_home == dest.resolve()
    assert w.output_home == (runner.root / "out").resolve()
    w.save_all()
    assert inference_files(runner.root / "out") == {"held_out_for_inference.parquet"}


@pytest.mark.parametrize("split", SPLITS)
def test_each_file_holds_only_subjects_of_that_split(pipeline, split):
    d = pipeline.inference(split)
    assert d.height > 0
    assert set(d["subject_id"]) <= set(pipeline.subjects_in_split(split))
    assert d["subject_id"].n_unique() == d.height


# --- duration thresholding ------------------------------------------------


@pytest.mark.parametrize("split", SPLITS)
def test_duration_threshold_keeps_exactly_the_long_stays(pipeline, split):
    m = pipeline.manifest
    kept = set(m.subjects_in_split(split)) & set(m.long_stay_subjects(24))
    dropped = set(m.subjects_in_split(split)) & set(m.short_stay_subjects(24))
    assert kept and dropped, "the split must exercise both sides of the threshold"
    assert set(pipeline.inference(split)["subject_id"]) == kept


def test_surviving_timelines_are_the_ones_spanning_more_than_the_threshold(pipeline):
    """cross-check the drop against timeline spans read out of tokens_times"""
    kept = set(pipeline.inference("held_out")["subject_id"])
    rows = pipeline.tokens_times.filter(
        pl.col("subject_id").is_in(pipeline.subjects_in_split("held_out"))
    )
    assert rows.height > len(kept) > 0
    spans = {
        r["subject_id"]: offsets(r["times"])[-1] for r in rows.iter_rows(named=True)
    }
    assert any(s > DAY_S for s in spans.values())
    assert any(s <= DAY_S for s in spans.values())
    for sid, span in spans.items():
        assert (span > DAY_S) == (sid in kept), sid


def test_duration_threshold_boundary_is_strict(runner):
    """
    a hand-built pair: H0 spans exactly 24h and is dropped, H1 spans 24h + 1s
    and survives with the event at exactly 24h landing in the future
    """
    a = datetime.datetime(2024, 1, 1, 8, 0, 0)
    at = [a + datetime.timedelta(seconds=s) for s in (0, 43200, DAY_S, DAY_S + 1)]
    p = runner.minimal(
        hospitalizations=[
            {"admission_dttm": a, "discharge_dttm": a + datetime.timedelta(days=4)}
        ]
        * 2,
        vitals=[
            {"hospitalization_id": h, "recorded_dttm": t, "vital_value": 70.0 + n}
            for h, ts in (("H0", at[:3]), ("H1", at))
            for n, t in enumerate(ts)
        ],
        winnowing={
            "outcome_tokens": ["VTL//*"],
            "threshold": {"duration_s": DAY_S},
            "splits": ["train"],
        },
    )
    assert sorted(p.subjects_in_split("train")) == ["H0", "H1"]
    d = p.inference("train")
    assert d["subject_id"].to_list() == ["H1"]
    r = d.row(0, named=True)
    assert r["s_total_duration"] == DAY_S + 1
    # times are BOS/vital at 0, vital at 12h, vital at 24h, vital/EOS at 24h+1s
    assert offsets(r["times"]) == [0, 0, 43200, DAY_S, DAY_S + 1, DAY_S + 1]
    assert r["last_valid"] == 3
    assert p.decode(r["tokens_past"]) == [
        "BOS",
        "VTL//heart_rate_Q2",
        "VTL//heart_rate_Q5",
    ]
    assert p.decode(r["tokens_future"])[-1] == "EOS"
    assert len(r["tokens_future"]) == 3


@pytest.mark.parametrize("split", SPLITS)
def test_last_valid_counts_events_strictly_within_the_horizon(pipeline, split):
    d = pipeline.inference(split)
    assert d.height > 0
    for r in d.iter_rows(named=True):
        assert r["last_valid"] == sum(1 for o in offsets(r["times"]) if o < DAY_S)
        # every survivor spans more than the horizon, so the future is non-empty
        assert 1 <= r["last_valid"] < len(r["tokens"])


# --- past / future partition ----------------------------------------------


@pytest.mark.parametrize("split", SPLITS)
def test_past_and_future_concatenate_to_the_timeline(pipeline, split):
    d = pipeline.inference(split)
    assert d.height > 0
    for r in d.iter_rows(named=True):
        assert list(r["tokens_past"]) + list(r["tokens_future"]) == list(r["tokens"])
        assert len(r["tokens_past"]) == r["last_valid"]
        assert list(r["s_elapsed_past"]) == list(r["s_elapsed"])[: r["last_valid"]]


def test_hours_to_end_time_is_split_like_the_tokens(extras):
    d = extras.inference("held_out")
    assert d.height > 0
    assert {"hours_to_end_time_past", "hours_to_end_time_future"} <= set(d.columns)
    ends = dict(extras.splits.select("subject_id", "end_time").iter_rows())
    for r in d.iter_rows(named=True):
        past, future = (
            list(r["hours_to_end_time_past"]),
            list(r["hours_to_end_time_future"]),
        )
        assert past + future == list(r["hours_to_end_time"])
        assert len(past) == r["last_valid"] == len(r["tokens_past"])
        assert len(future) == len(r["tokens_future"])
        end = ends[r["subject_id"]]
        assert past + future == [(end - t).total_seconds() / 3600 for t in r["times"]]


def test_horizon_after_threshold_truncates_hours_to_end_time_too(runner, extras):
    d = rewinnow(runner, extras, horizon_after_threshold_s=HORIZON_S)
    assert d.height > 0
    shortened = 0
    for r in d.iter_rows(named=True):
        future = list(r["hours_to_end_time_future"])
        assert len(future) == len(r["tokens_future"])
        whole = list(r["hours_to_end_time"])[r["last_valid"] :]
        assert future == whole[: len(future)]
        shortened += len(future) < len(whole)
    assert shortened == d.height, "the horizon truncated nothing"


def test_numeric_values_ride_along_unsplit(runner, extras):
    """
    add_outcome_flags splits only the per-token columns it names, and
    numeric_values is not one: it passes through whole, aligned with tokens,
    even under a horizon
    """
    whole = extras.tokens_times.select("subject_id", "numeric_values")
    assert whole["numeric_values"].explode(empty_as_null=False).drop_nulls().len() > 0
    for d in (
        extras.inference("held_out"),
        rewinnow(runner, extras, horizon_after_threshold_s=HORIZON_S),
    ):
        assert d.height > 0
        assert not [c for c in d.columns if c.startswith("numeric_values_")]
        got = d.select("subject_id", "numeric_values").sort("subject_id")
        ref = whole.filter(pl.col("subject_id").is_in(got["subject_id"].to_list()))
        assert got.equals(ref.sort("subject_id"))
        assert (d["numeric_values"].list.len() == d["tokens"].list.len()).all()


@pytest.mark.parametrize("split", SPLITS)
def test_s_elapsed_is_seconds_since_the_first_event(pipeline, split):
    d = pipeline.inference(split)
    assert d.height > 0
    for r in d.iter_rows(named=True):
        s = list(r["s_elapsed"])
        assert s == offsets(r["times"])
        assert s[0] == 0
        assert all(x <= y for x, y in zip(s, s[1:]))
        assert r["s_total_duration"] == s[-1]


# --- outcome flags --------------------------------------------------------


def test_outcome_columns_are_the_expanded_patterns(pipeline):
    d = pipeline.inference("held_out")
    got = {
        c
        for c in d.columns
        if (c.endswith("_past") or c.endswith("_future")) and c not in STRUCTURAL
    }
    assert got == {
        f"{t}_{tense}" for t in outcomes(pipeline) for tense in ("past", "future")
    }


def test_unmatched_literal_pattern_contributes_nothing(runner, pipeline):
    dest = seed(runner, pipeline)
    w = runner.winnow(
        cfg=cfg(
            outcome_tokens=["DSCG//expired", "NOPE//nope", "LABEL//*"],
            splits=["held_out"],
        ),
        processed=dest,
    )
    labels = {k for k in pipeline.vocab if k.startswith("LABEL//")}
    assert set(w.grokked_outcome_tokens) == labels | {"DSCG//expired"}
    d = pl.read_parquet(dest / "held_out_for_inference.parquet")
    assert not [c for c in d.columns if c.startswith("NOPE")]


@pytest.mark.parametrize(
    "patterns, expected, near",
    (
        # a literal is that word alone, not every word it begins
        (["VTL//heart_rate"], {"VTL//heart_rate"}, {"VTL//heart_rate_Q0"}),
        # ? is exactly one character
        (["XFR-IN//???"], {"XFR-IN//icu"}, {"XFR-IN//ed", "XFR-IN//ward"}),
        (
            ["DSCG//[eh]*"],
            {"DSCG//expired", "DSCG//home", "DSCG//hospice"},
            {"DSCG//skilled_nursing_facility"},
        ),
        (["DSCG//[!eh]*"], {"DSCG//skilled_nursing_facility"}, {"DSCG//home"}),
        # a word that several patterns match is an outcome once
        (
            ["XFR-*//icu", "XFR-IN//*", "XFR-IN//icu"],
            {f"XFR-IN//{x}" for x in ("ed", "icu", "stepdown", "ward")}
            | {"XFR-OUT//icu"},
            {"XFR-OUT//ed"},
        ),
        (["NOPE//*"], set(), set()),
        # case counts, and a fused bin's suffix is uppercase; fnmatch passes
        # both sides through os.path.normcase, which folds case on windows
        pytest.param(
            ["VTL//heart_rate_Q[01]", "VTL//heart_rate_q[23]"],
            {"VTL//heart_rate_Q0", "VTL//heart_rate_Q1"},
            {"VTL//heart_rate_Q2", "VTL//heart_rate_Q3"},
            marks=pytest.mark.skipif(
                os.path.normcase("A") != "A", reason="fnmatch folds case here"
            ),
        ),
    ),
    ids=(
        "literal",
        "any-char",
        "char-set",
        "not-char-set",
        "overlapping",
        "unmatched",
        "case",
    ),
)
def test_outcome_patterns_are_matched_with_fnmatch(
    runner, pipeline, patterns, expected, near
):
    """`near` holds words a looser reading of the patterns would also match"""
    assert near <= set(pipeline.vocab)
    w = winnower(runner, pipeline.path, outcome_tokens=patterns)
    assert sorted(w.grokked_outcome_tokens) == sorted(expected)
    schema = w.prepare_winnowed_frame().collect_schema()
    assert {c for c, t in schema.items() if t == pl.Boolean} == {
        f"{t}_{tense}" for t in expected for tense in ("past", "future")
    }


def test_a_lone_star_makes_every_word_an_outcome(runner, pipeline):
    w = winnower(runner, pipeline.path, outcome_tokens=["*"])
    assert sorted(w.grokked_outcome_tokens) == sorted(pipeline.vocab)
    assert {"UNK", "BOS", "EOS"} <= set(w.grokked_outcome_tokens)


def test_outcome_tokens_has_no_fallback(runner, pipeline):
    """unlike splits, a config must give outcome_tokens, or the winnower fails"""
    c = cfg()
    del c["outcome_tokens"]
    with pytest.raises(ConfigAttributeError):
        Winnower(
            winnowing_cfg=runner.cfg_path("winnowing", c),
            processed_data_home=pipeline.path,
        )


@pytest.mark.parametrize("split", SPLITS)
def test_outcome_flags_are_token_membership_in_each_tense(pipeline, split):
    d = pipeline.inference(split)
    assert d.height > 0
    seen = set()
    for r in d.iter_rows(named=True):
        tenses = {t: set(r[f"tokens_{t}"]) for t in ("past", "future")}
        for name in outcomes(pipeline):
            tok = pipeline.vocab[name]
            for tense, present in tenses.items():
                assert r[f"{name}_{tense}"] == (tok in present), (r["subject_id"], name)
                if r[f"{name}_{tense}"]:
                    seen.add((name, tense))
    assert len(seen) > 5, "no flags fired; the check would hold vacuously"


def test_expired_subject_is_flagged_in_exactly_one_tense(pipeline):
    m = pipeline.manifest
    tok = pipeline.vocab["DSCG//expired"]
    rows = list(pipeline.inference("held_out").iter_rows(named=True))
    expired = [r for r in rows if r["subject_id"] in m.expired]
    survived = [r for r in rows if r["subject_id"] not in m.expired]
    assert expired and survived
    for r in expired:
        assert r["DSCG//expired_past"] != r["DSCG//expired_future"]
        tense = "future" if r["DSCG//expired_future"] else "past"
        assert tok in list(r[f"tokens_{tense}"])
    for r in survived:
        assert not r["DSCG//expired_past"] and not r["DSCG//expired_future"]


# --- first-occurrence thresholding ----------------------------------------


def test_first_occurrence_keeps_only_subjects_carrying_the_token(runner, pipeline):
    m = pipeline.manifest
    icu = set(m.subjects_in_split("held_out")) & set(m.icu)
    without = set(m.subjects_in_split("held_out")) - set(m.icu)
    assert icu and without
    d = rewinnow(runner, pipeline, threshold={"first_occurrence": "XFR-IN//icu"})
    assert set(d["subject_id"]) == icu


def test_first_occurrence_ends_the_past_with_the_triggering_token(runner, pipeline):
    tok = pipeline.vocab["XFR-IN//icu"]
    d = rewinnow(runner, pipeline, threshold={"first_occurrence": "XFR-IN//icu"})
    assert d.height > 0
    for r in d.iter_rows(named=True):
        assert r["last_valid"] == list(r["tokens"]).index(tok) + 1
        assert list(r["tokens_past"])[-1] == tok
        assert len(r["tokens_future"]) > 0
        assert list(r["tokens_future"])[0] != tok
        assert r["XFR-IN//icu_past"] and not r["XFR-IN//icu_future"]


def test_first_occurrence_cuts_after_the_earliest_of_several(runner):
    """H0's second spo2 reading lands in its future; H1 never reads spo2"""
    p = spo2_twice(runner)
    d = winnow_train(runner, p, {"first_occurrence": "VTL//spo2"})
    assert d["subject_id"].to_list() == ["H0"]
    r = d.row(0, named=True)
    assert p.decode(r["tokens_past"]) == ["BOS", "VTL//heart_rate", "VTL//spo2"]
    assert p.decode(r["tokens_future"]) == [
        "VTL//heart_rate",
        "VTL//spo2",
        "VTL//heart_rate",
        "EOS",
    ]
    assert r["VTL//spo2_past"] and r["VTL//spo2_future"]


def test_first_occurrence_of_unknown_token_raises_config_key_error(runner, pipeline):
    # a threshold token outside the vocabulary is looked up in tokenizer.yaml
    # with no guard, so it surfaces as omegaconf's "Missing key ..." (a KeyError)
    with pytest.raises(ConfigKeyError):
        rewinnow(runner, pipeline, threshold={"first_occurrence": "NOT-A//token"})


def test_first_occurrence_absent_from_the_split_writes_an_empty_frame(runner, pipeline):
    """
    when no subject in the split carries the threshold token, the filtered frame
    is empty, and the split gets a file with no rows, as under the unreachable
    duration threshold below
    """
    used = set()
    for row in pipeline.tokens_times.filter(
        pl.col("subject_id").is_in(pipeline.subjects_in_split("held_out"))
    )["tokens"]:
        used |= {int(t) for t in row}
    absent = sorted(k for k, i in pipeline.vocab.items() if i not in used)
    assert absent, "every vocabulary token occurs in held_out"
    d = rewinnow(runner, pipeline, threshold={"first_occurrence": absent[0]})
    assert d.height == 0
    assert d.schema == pipeline.inference("held_out").schema


def test_unreachable_duration_threshold_writes_an_empty_frame(runner, pipeline):
    d = rewinnow(runner, pipeline, threshold={"duration_s": 10**9})
    assert d.height == 0
    assert {"last_valid", "tokens_past", "tokens_future"} <= set(d.columns)
    assert "DSCG//expired_future" in d.columns


# --- start-time thresholding ----------------------------------------------


@pytest.mark.parametrize("unit", ("ms", "us", "ns"))
def test_after_start_time_counts_from_the_start_time_not_the_first_event(runner, unit):
    """
    H0 opens 2h before its admission and H1 3h after it, so 24h after the first
    event is 22h after admission for H0 and 27h for H1, while after_start_time_s
    cuts both 24h after admission
    """
    p = minimal_at(runner, OFF_START, unit)
    past = {
        key: past_hours(winnow_train(runner, p, {key: DAY_S}))
        for key in ("duration_s", "after_start_time_s")
    }
    # BOS shares the time of the first reading
    assert past["duration_s"] == {"H0": [-2, -2], "H1": [3, 3, 25]}
    assert past["after_start_time_s"] == {"H0": [-2, -2, 23], "H1": [3, 3]}


def test_after_start_time_cuts_that_long_after_the_start_time(runner):
    """
    cut at admission, only H0 began before the cut; cut 26h after it, only H1
    runs past it, H0's last reading being at 25h
    """
    p = minimal_at(runner, OFF_START)
    past = {
        h: past_hours(winnow_train(runner, p, {"after_start_time_s": h * HOUR_S}))
        for h in (0, 26)
    }
    assert past == {0: {"H0": [-2, -2]}, 26: {"H1": [3, 3, 25]}}


def test_after_start_time_cuts_at_the_precision_of_the_start_time(runner):
    """
    at ns, a start_time moved 500 ns later moves the cut as far, so a reading a
    day after admission, at the cut before, falls just before it, in the past
    """
    p = minimal_at(runner, {"H0": (0, DAY_S, DAY_S + 1)}, "ns")
    at_start = {"after_start_time_s": DAY_S}
    assert winnow_train(runner, p, at_start)["last_valid"].to_list() == [2]
    f = p.path / "subject_splits.parquet"
    pl.read_parquet(f).with_columns(
        pl.col("start_time") + pl.duration(nanoseconds=500)
    ).write_parquet(f)
    assert winnow_train(runner, p, at_start)["last_valid"].to_list() == [3]


def test_after_start_time_keeps_only_timelines_spanning_the_cut(runner):
    """
    as under duration_s, a timeline must end after the cut, with an event at
    exactly the cut in the future; it must also begin before the cut, which
    under duration_s it always does
    """
    readings = {
        "H0": (0, DAY_S),  # ends at the cut
        "H1": (0, DAY_S, DAY_S + 1),  # ends just after it
        "H2": (DAY_S, 2 * DAY_S + 1),  # opens at it
    }
    p = minimal_at(runner, readings)
    d = winnow_train(runner, p, {"duration_s": DAY_S})
    assert sorted(d["subject_id"]) == ["H1", "H2"]
    d = winnow_train(runner, p, {"after_start_time_s": DAY_S})
    assert d["subject_id"].to_list() == ["H1"]
    r = d.row(0, named=True)
    assert offsets(r["times"]) == [0, 0, DAY_S, DAY_S + 1, DAY_S + 1]
    assert r["last_valid"] == 2  # BOS and the reading at admission
    assert p.decode(r["tokens_future"]) == ["VTL//heart_rate", "VTL//heart_rate", "EOS"]


@pytest.mark.parametrize("extra", ({}, {"horizon_after_threshold_s": HORIZON_S}))
def test_after_start_time_agrees_with_duration_on_timelines_opening_at_the_start(
    runner, pipeline, extra
):
    """every synthetic timeline opens at its admission, where the two coincide"""
    rows = pipeline.tokens_times.join(pipeline.splits, on="subject_id")
    assert (rows["times"].list.first() == rows["start_time"]).all()
    by_start = rewinnow(
        runner, pipeline, threshold={"after_start_time_s": DAY_S}, **extra
    )
    assert by_start.height > 0
    assert by_start.equals(rewinnow(runner, pipeline, **extra))


@pytest.mark.parametrize("shift_h", (-6, 6))
def test_after_start_time_follows_the_start_time_in_subject_splits(
    runner, pipeline, shift_h
):
    """
    moving every start_time moves the cut with it, though no event moves;
    a stay whose last event lands exactly on the moved cut is dropped
    """
    dest = seed(runner, pipeline)
    f = dest / "subject_splits.parquet"
    pl.read_parquet(f).with_columns(
        pl.col("start_time") + datetime.timedelta(hours=shift_h)
    ).write_parquet(f)
    runner.winnow(
        cfg=cfg(splits=["held_out"], threshold={"after_start_time_s": DAY_S}),
        processed=dest,
    )
    d = pl.read_parquet(dest / "held_out_for_inference.parquet")
    m, cut_h = pipeline.manifest, 24 + shift_h
    held_out = set(m.subjects_in_split("held_out"))
    assert held_out & {s for s, h in m.los_hours.items() if h == cut_h}, (
        "no held-out stay ends exactly at the cut"
    )
    assert set(d["subject_id"]) == held_out & set(m.long_stay_subjects(cut_h))
    for r in d.iter_rows(named=True):
        # synthetic timelines open at their admission
        assert r["last_valid"] == sum(
            1 for o in offsets(r["times"]) if o < cut_h * HOUR_S
        )


# --- horizon after the threshold ------------------------------------------


def test_horizon_after_threshold_truncates_the_future_to_a_prefix(runner, pipeline):
    full = {
        r["subject_id"]: r for r in pipeline.inference("held_out").iter_rows(named=True)
    }
    d = rewinnow(runner, pipeline, horizon_after_threshold_s=HORIZON_S)
    assert set(d["subject_id"]) == set(full)
    shortened = 0
    for r in d.iter_rows(named=True):
        ref = full[r["subject_id"]]
        assert r["last_valid"] == ref["last_valid"]  # the past is untouched
        got, whole = list(r["tokens_future"]), list(ref["tokens_future"])
        assert got == whole[: len(got)]
        shortened += len(got) < len(whole)
        # the retained future is exactly the events within the horizon of the
        # last event of the past
        times = list(r["times"])
        thresh = times[r["last_valid"] - 1]
        after = [(t - thresh).total_seconds() for t in times[r["last_valid"] :]]
        assert sum(1 for s in after if s <= HORIZON_S) == len(got)
        if len(got) < len(whole):
            assert after[len(got)] > HORIZON_S
    assert shortened == d.height, "the horizon truncated nothing"


def test_horizon_after_threshold_clears_outcomes_beyond_it(runner, pipeline):
    tok = pipeline.vocab["DSCG//expired"]
    full = {
        r["subject_id"]: r for r in pipeline.inference("held_out").iter_rows(named=True)
    }
    d = rewinnow(runner, pipeline, horizon_after_threshold_s=HORIZON_S)
    flipped = []
    for r in d.iter_rows(named=True):
        ref = full[r["subject_id"]]
        assert not (r["DSCG//expired_future"] and not ref["DSCG//expired_future"])
        if ref["DSCG//expired_future"] and not r["DSCG//expired_future"]:
            flipped.append(r)
    assert flipped, "no late outcome to clear"
    for r in flipped:
        # the death token really did fall outside the retained window
        cut = r["last_valid"] + len(r["tokens_future"])
        assert list(r["tokens"]).index(tok) >= cut
        assert tok not in list(r["tokens_future"])


def test_horizon_after_threshold_keeps_an_event_at_its_very_end(runner):
    """
    the window opens at the last event of the past, here H0's first spo2
    reading at 1h, so a window of 2h closes on its second, at 3h, and keeps it
    """
    p = spo2_twice(runner)
    for horizon_s, future in (
        (2 * HOUR_S, ["VTL//heart_rate", "VTL//spo2"]),
        (2 * HOUR_S - 1, ["VTL//heart_rate"]),
    ):
        d = winnow_train(
            runner,
            p,
            {"first_occurrence": "VTL//spo2"},
            horizon_after_threshold_s=horizon_s,
        )
        r = d.row(0, named=True)
        assert p.decode(r["tokens_past"]) == ["BOS", "VTL//heart_rate", "VTL//spo2"]
        assert p.decode(r["tokens_future"]) == future, horizon_s
        assert r["VTL//spo2_future"] == ("VTL//spo2" in future)


# --- threshold configuration errors ---------------------------------------


@pytest.mark.parametrize("threshold", ({}, None))
def test_no_threshold_criterion_raises_not_implemented(runner, pipeline, threshold):
    c = cfg(splits=["held_out"])
    if threshold is None:
        del c["threshold"]
    else:
        c["threshold"] = threshold
    dest = seed(runner, pipeline)
    with pytest.raises(NotImplementedError):
        runner.winnow(cfg=c, processed=dest)


@pytest.mark.parametrize(
    "threshold",
    (None, {"after_start_time_s": DAY_S}, {"first_occurrence": "XFR-IN//icu"}),
    ids=("no-block", "after_start_time_s", "first_occurrence"),
)
def test_legacy_horizon_s_without_threshold_duration_s_raises(
    runner, pipeline, threshold
):
    # BUG: run_thresholding reads self.cfg.threshold.duration_s as the eager
    # default of .get("horizon_s", ...), so horizon_s alone cannot be used, nor
    # beside the other thresholds, which it outranks
    c = cfg(splits=["held_out"], horizon_s=43200)
    if threshold is None:
        del c["threshold"]
    else:
        c["threshold"] = threshold
    dest = seed(runner, pipeline)
    with pytest.raises(ConfigAttributeError):
        runner.winnow(cfg=c, processed=dest)


def test_legacy_horizon_s_overrides_threshold_duration_s(runner, pipeline):
    m = pipeline.manifest
    d = rewinnow(runner, pipeline, horizon_s=43200)  # 12h, vs the 24h default
    expected = set(m.subjects_in_split("held_out")) & set(m.long_stay_subjects(12))
    kept_at_24h = set(pipeline.inference("held_out")["subject_id"])
    assert kept_at_24h < expected, "the shorter horizon must keep strictly more"
    assert set(d["subject_id"]) == expected
    for r in d.iter_rows(named=True):
        assert r["last_valid"] == sum(1 for o in offsets(r["times"]) if o < 43200)


def test_given_several_thresholds_the_first_in_order_wins(runner):
    """
    duration_s outranks after_start_time_s, which outranks first_occurrence,
    however the config orders them
    """
    p = minimal_at(runner, OFF_START)
    first = {"first_occurrence": "VTL//heart_rate"}
    by_duration = winnow_train(runner, p, {"duration_s": DAY_S})
    by_start = winnow_train(runner, p, {"after_start_time_s": DAY_S})
    by_first = winnow_train(runner, p, first)
    assert not by_duration.equals(by_start)
    assert not by_first.equals(by_duration) and not by_first.equals(by_start)
    assert winnow_train(
        runner, p, {"after_start_time_s": DAY_S, "duration_s": DAY_S}
    ).equals(by_duration)
    assert winnow_train(runner, p, {**first, "after_start_time_s": DAY_S}).equals(
        by_start
    )
    assert winnow_train(runner, p, {**first, "duration_s": DAY_S}).equals(by_duration)
    assert winnow_train(
        runner, p, {**first, "after_start_time_s": DAY_S, "duration_s": DAY_S}
    ).equals(by_duration)


@pytest.mark.filterwarnings("ignore:Comparisons with None")
def test_a_null_threshold_still_outranks_the_ones_after_it(runner, pipeline):
    """
    as the README warns, a threshold set to null, as by the override
    threshold.duration_s=null, is still present, so it still beats the next:
    a null duration_s compares each duration with null and keeps no stay
    """
    after = {"after_start_time_s": DAY_S}
    assert rewinnow(runner, pipeline, threshold=after).height > 0
    nulled = rewinnow(runner, pipeline, threshold={"duration_s": None, **after})
    assert nulled.height == 0


# --- reproducibility ------------------------------------------------------


def test_winnowing_twice_yields_identical_frames(runner, pipeline):
    dest = seed(runner, pipeline)
    runner.winnow(cfg=cfg(splits=["held_out"]), processed=dest)
    first = pl.read_parquet(dest / "held_out_for_inference.parquet")
    runner.winnow(cfg=cfg(splits=["held_out"]), processed=dest)
    second = pl.read_parquet(dest / "held_out_for_inference.parquet")
    assert first.height > 0
    assert first.equals(second)
    assert first.equals(pipeline.inference("held_out"))
