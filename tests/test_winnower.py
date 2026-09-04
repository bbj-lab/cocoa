#!/usr/bin/env python3

"""
winnowing: thresholding timelines into a past and a future, and flagging
outcome tokens in each tense
"""

import datetime
import pathlib
import shutil

import polars as pl
import pytest
from conftest import default_cfg
from omegaconf.errors import ConfigAttributeError, ConfigKeyError
from polars.exceptions import PanicException

SPLITS = ("train", "tuning", "held_out")
DAY_S = 86400
HORIZON_S = 18000  # 5h; short enough to truncate every synthetic future
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


def offsets(times) -> list:
    """seconds from the first time, in plain python"""
    ts = list(times)
    return [int((t - ts[0]).total_seconds()) for t in ts]


def inference_files(path) -> set:
    return {p.name for p in pathlib.Path(path).glob("*_for_inference.parquet")}


def outcomes(pipeline) -> set:
    """
    what the shipped default outcome_tokens should expand to, derived from the
    vocabulary without going through fnmatch
    """
    labels = {k for k in pipeline.vocab if k.startswith("LABEL//")}
    assert len(labels) > 1, "no LABEL tokens to expand against"
    assert set(LITERAL_OUTCOMES) <= set(pipeline.vocab)
    return labels | set(LITERAL_OUTCOMES)


# --- output files ---------------------------------------------------------


def test_default_config_writes_one_file_per_configured_split(pipeline):
    assert default_cfg("winnowing")["splits"] == list(SPLITS)
    assert inference_files(pipeline.path) == {
        f"{s}_for_inference.parquet" for s in SPLITS
    }


@pytest.mark.parametrize("splits", (["held_out"], ["train", "tuning"]))
def test_only_configured_splits_are_written(runner, pipeline, splits):
    dest = seed(runner, pipeline)
    runner.winnow(cfg=cfg(splits=splits), processed=dest)
    assert inference_files(dest) == {f"{s}_for_inference.parquet" for s in splits}


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


def _with_hours(runner):
    """a full run whose tokenization carries hours_to_end_time"""
    tok = default_cfg("tokenization")
    tok["include_hours_to_end_time"] = True
    return runner.full(tokenization=tok)


def test_hours_to_end_time_is_split_like_the_tokens(runner):
    processed = _with_hours(runner)
    d = processed.inference("held_out")
    assert d.height > 0
    assert {"hours_to_end_time_past", "hours_to_end_time_future"} <= set(d.columns)
    ends = dict(processed.splits.select("subject_id", "end_time").iter_rows())
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


def test_horizon_after_threshold_truncates_hours_to_end_time_too(runner):
    processed = _with_hours(runner)
    runner.winnow(
        cfg=cfg(splits=["held_out"], horizon_after_threshold_s=HORIZON_S),
        processed=processed.path,
    )
    d = pl.read_parquet(processed.path / "held_out_for_inference.parquet")
    assert d.height > 0
    shortened = 0
    for r in d.iter_rows(named=True):
        future = list(r["hours_to_end_time_future"])
        assert len(future) == len(r["tokens_future"])
        whole = list(r["hours_to_end_time"])[r["last_valid"] :]
        assert future == whole[: len(future)]
        shortened += len(future) < len(whole)
    assert shortened == d.height, "the horizon truncated nothing"


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


def test_first_occurrence_of_unknown_token_raises_config_key_error(runner, pipeline):
    # a threshold token outside the vocabulary is looked up in tokenizer.yaml
    # with no guard, so it surfaces as omegaconf's "Missing key ..." (a KeyError)
    with pytest.raises(ConfigKeyError):
        rewinnow(runner, pipeline, threshold={"first_occurrence": "NOT-A//token"})


def test_first_occurrence_absent_from_the_split_panics(runner, pipeline):
    """
    BUG (current behaviour asserted): when no subject in the split carries the
    threshold token the filtered frame is empty, last_valid comes out null and
    polars panics rather than writing an empty frame -- unlike the duration
    threshold below, which handles the same situation
    """
    used = set()
    for row in pipeline.tokens_times.filter(
        pl.col("subject_id").is_in(pipeline.subjects_in_split("held_out"))
    )["tokens"]:
        used |= {int(t) for t in row}
    absent = sorted(k for k, i in pipeline.vocab.items() if i not in used)
    assert absent, "every vocabulary token occurs in held_out"
    with pytest.raises(PanicException):
        rewinnow(runner, pipeline, threshold={"first_occurrence": absent[0]})


def test_unreachable_duration_threshold_writes_an_empty_frame(runner, pipeline):
    d = rewinnow(runner, pipeline, threshold={"duration_s": 10**9})
    assert d.height == 0
    assert {"last_valid", "tokens_past", "tokens_future"} <= set(d.columns)
    assert "DSCG//expired_future" in d.columns


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


def test_legacy_horizon_s_without_a_threshold_block_raises(runner, pipeline):
    # BUG: run_thresholding reads self.cfg.threshold.duration_s as the eager
    # default of .get("horizon_s", ...), so horizon_s alone cannot be used
    c = cfg(splits=["held_out"], horizon_s=43200)
    del c["threshold"]
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
