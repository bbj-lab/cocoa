#!/usr/bin/env python3

"""configurable tokenization options: fusion, bins, numerics, spacers, clocks"""

import collections
import datetime
import math
import re

import polars as pl
import pytest
import synth
from conftest import Processed, default_cfg

UTC = datetime.timezone.utc
JAN1 = datetime.datetime(2024, 1, 1)
MIN = datetime.timedelta(minutes=1)
HOUR = datetime.timedelta(hours=1)


def utc(*args) -> datetime.datetime:
    """a tz-aware utc instant, to compare against a stored time"""
    return datetime.datetime(*args, tzinfo=UTC)


def stay(hours: int = 72, start: datetime.datetime = JAN1) -> list:
    return [{"admission_dttm": start, "discharge_dttm": start + hours * HOUR}]


def vitals(times, *, values=None, category=None) -> list:
    """one vitals row per time; distinctly named unless `category` is given"""
    return [
        {
            "recorded_dttm": t,
            "vital_category": category if category is not None else f"E{i}",
            "vital_value": None if values is None else values[i],
        }
        for i, t in enumerate(times)
    ]


def tok_cfg(**overrides) -> dict:
    cfg = default_cfg("tokenization")
    cfg.update(overrides)
    return cfg


def tokenized(runner, **overrides):
    """tokenize the cached default collation with an overridden config"""
    dest = runner.seed_collated()
    tkzr = runner.tokenize(processed=dest, cfg=tok_cfg(**overrides))
    return tkzr, Processed(dest, runner.raw)


def read(p, subject_id: str = "H0") -> tuple:
    """(decoded words, times, numeric values) of one subject's timeline"""
    row = p.tokens_times.filter(pl.col("subject_id") == subject_id)
    assert row.height == 1, subject_id
    nums = (
        row["numeric_values"].item().to_list()
        if "numeric_values" in row.columns
        else None
    )
    return p.decode(row["tokens"].item().to_list()), row["times"].item().to_list(), nums


def marked(words, times, prefix: str) -> list:
    """(time, word) of every token whose word carries `prefix`"""
    return [(t, w) for w, t in zip(words, times) if w.startswith(prefix)]


def spacer_label(spacers: dict, gap_minutes: int) -> str | None:
    """the label the configured spacer table gives a gap, in plain python"""
    bounds = list(spacers.values())
    if gap_minutes < min(bounds):
        return None
    # polars `cut` is right-closed, so a gap landing exactly on a boundary
    # keeps the *lower* label (60 minutes is still `15m-1h`, not `1h-2h`)
    return list(spacers)[sum(1 for b in bounds[1:] if gap_minutes > b)]


def clock_marks(first, last, hours) -> list:
    """hour marks strictly after `first` and up to `last`, in plain python"""
    t = first.replace(minute=0, second=0, microsecond=0) + HOUR
    end = last.replace(minute=0, second=0, microsecond=0)
    out = []
    while t <= end:
        if t.strftime("%H") in hours:
            out.append((t, "CLCK//" + t.strftime("%H")))
        t += HOUR
    return out


# a single entry carrying a code, a numeric value and a text value at once
TEXT_ENTRY = [
    {
        "table": "clif_vitals",
        "prefix": "ASMT",
        "code": "vital_category",
        "numeric_value": "vital_value",
        "text_value": "flag",
        "time": "recorded_dttm",
        "with_col_expr": 'pl.lit("Yes indeed").alias("flag")',
    }
]

# two prefixes so that cotemporaneous codes can be ordered against each other
TWO_PREFIX_ENTRIES = [
    {
        "table": "clif_vitals",
        "prefix": prefix,
        "code": "vital_category",
        "time": "recorded_dttm",
        "filter_expr": f'pl.col("vital_category") == "{cat}"',
    }
    for prefix, cat in (("AAA", "a"), ("ZZZ", "z"))
]

TWO_PREFIX_VITALS = [
    {"recorded_dttm": JAN1 + 4 * HOUR, "vital_category": "a"},
    {"recorded_dttm": JAN1 + 4 * HOUR, "vital_category": "z"},
    {"recorded_dttm": JAN1 + 8 * HOUR, "vital_category": "a"},
]


def fusion_dataset(runner, *, fused: bool):
    """three readings of one code that has a bin and a text value"""
    return runner.minimal(
        hospitalizations=stay(24),
        vitals=[
            {"recorded_dttm": JAN1 + h * HOUR, "vital_value": v}
            for h, v in ((1, 10.0), (2, 20.0), (3, 30.0))
        ],
        collation=synth.minimal_collation_cfg(entries=TEXT_ENTRY),
        tokenization=tok_cfg(fused=fused, n_bins=2, include_numeric_values=True),
    )


def two_prefix_timeline(runner, ordering):
    p = runner.minimal(
        hospitalizations=stay(24),
        vitals=TWO_PREFIX_VITALS,
        collation=synth.minimal_collation_cfg(entries=TWO_PREFIX_ENTRIES),
        tokenization=tok_cfg(ordering=ordering),
    )
    return read(p)[0]


# --- fusion ----------------------------------------------------------------


def test_fused_vocabulary_has_no_bare_bin_or_text_words(pipeline):
    vocab = pipeline.vocab
    assert default_cfg("tokenization")["fused"] is True
    assert not [w for w in vocab if re.fullmatch(r"Q\d+", w)]
    assert "yes" not in vocab and "no" not in vocab
    # code, bin and text are glued together with "_" into one word
    assert {"ASMT//cam_total_yes", "ASMT//cam_total_no"} <= set(vocab)
    assert "ASMT//cam_total" not in vocab
    assert len([w for w in vocab if re.search(r"_Q\d+$", w)]) > 50


def test_unfused_vocabulary_has_bare_bin_and_text_words(runner, pipeline):
    _, p = tokenized(runner, fused=False)
    vocab = p.vocab
    assert {w for w in vocab if re.fullmatch(r"Q\d+", w)} == {
        f"Q{i}" for i in range(10)
    }
    assert "yes" in vocab and "no" in vocab
    assert "ASMT//cam_total" in vocab
    assert "ASMT//cam_total_yes" not in vocab
    assert not [w for w in vocab if re.search(r"_Q\d+", w)]
    # bins and text values are shared across codes instead of multiplied out
    assert len(vocab) < len(pipeline.vocab)


def test_unfused_timelines_longer_than_fused(runner, pipeline):
    _, p = tokenized(runner, fused=False)
    both = pipeline.tokens_times.select(
        "subject_id", pl.col("tokens").list.len().alias("n_fused")
    ).join(
        p.tokens_times.select(
            "subject_id", pl.col("tokens").list.len().alias("n_unfused")
        ),
        on="subject_id",
        validate="1:1",
    )
    assert both.height == pipeline.tokens_times.height > 0
    assert (both["n_unfused"] > both["n_fused"]).all()


def test_fused_joins_code_bin_and_text_into_one_token(runner):
    p = fusion_dataset(runner, fused=True)
    words, times, nums = read(p)
    assert words == [
        "BOS",
        "ASMT//heart_rate_Q0_yes_indeed",
        "ASMT//heart_rate_Q1_yes_indeed",
        "ASMT//heart_rate_Q1_yes_indeed",
        "EOS",
    ]
    assert times == [utc(2024, 1, 1, h) for h in (1, 1, 2, 3, 3)]
    assert nums == [None, 10.0, 20.0, 30.0, None]


def test_unfused_emits_code_bin_and_text_as_consecutive_tokens(runner):
    p = fusion_dataset(runner, fused=False)
    words, times, nums = read(p)
    assert words == (
        ["BOS"]
        + ["ASMT//heart_rate", "Q0", "yes_indeed"]
        + ["ASMT//heart_rate", "Q1", "yes_indeed"]
        + ["ASMT//heart_rate", "Q1", "yes_indeed"]
        + ["EOS"]
    )
    assert times == [utc(2024, 1, 1, h) for h in (1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3)]
    # the event's numeric value is repeated across each of its tokens
    assert nums == [None, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0, None]


# --- bins ------------------------------------------------------------------


@pytest.mark.parametrize("n_bins", [2, 4, 10, 20])
def test_n_bins_controls_break_columns_and_bin_indices(runner, n_bins):
    tkzr, p = tokenized(runner, n_bins=n_bins, fused=False)
    assert tkzr.bins.columns == ["code"] + [f"break_{i}" for i in range(1, n_bins)]
    assert tkzr.bins.height > 0
    for row in tkzr.bins.iter_rows(named=True):
        breaks = [row[f"break_{i}"] for i in range(1, n_bins)]
        assert all(b is not None for b in breaks), row["code"]
        assert breaks == sorted(breaks), row["code"]
    observed = {
        int(w[1:])
        for tokens in p.tokens_times["tokens"]
        for w in p.decode(tokens.to_list())
        if re.fullmatch(r"Q\d+", w)
    }
    assert observed == set(range(n_bins))


def test_n_bins_two_splits_at_the_median(runner):
    p = runner.minimal(
        hospitalizations=stay(24),
        vitals=vitals(
            [JAN1 + h * HOUR for h in (1, 2, 3, 4, 5)],
            values=[10.0, 20.0, 30.0, 40.0, 50.0],
            category="hr",
        ),
        tokenization=tok_cfg(n_bins=2),
    )
    assert p.tokenizer_yaml.bins["VTL//hr"] == [pytest.approx(30.0)]
    assert read(p)[0] == ["BOS"] + [f"VTL//hr_Q{q}" for q in (0, 0, 1, 1, 1)] + ["EOS"]


# --- numeric values --------------------------------------------------------


def test_numeric_values_column_absent_by_default(pipeline):
    assert default_cfg("tokenization")["include_numeric_values"] is False
    assert pipeline.tokens_times.columns == ["subject_id", "tokens", "times"]


def test_numeric_values_align_with_the_collated_events(runner):
    _, p = tokenized(runner, include_numeric_values=True)
    tt = p.tokens_times
    assert tt.height > 0
    assert "numeric_values" in tt.columns
    assert tt.select(
        (pl.col("tokens").list.len() == pl.col("numeric_values").list.len())
        .all()
        .alias("same_len")
    )["same_len"].item()

    seen, unks, nans = {}, 0, []
    for row in tt.iter_rows(named=True):
        words = p.decode(row["tokens"])
        for w, t, v in zip(words, row["times"], row["numeric_values"]):
            if w == "UNK":
                unks += 1  # an out-of-vocab code hides its own bin
            else:
                finite = v is not None and math.isfinite(v)
                assert bool(re.search(r"_Q\d+", w)) == finite, w
            if v is None:
                continue
            if v != v:
                nans.append(w)  # nan is neither used for breaks nor binned
            else:
                key = (row["subject_id"], t, round(float(v), 4))
                seen[key] = seen.get(key, 0) + 1
    assert unks > 0 and sum(seen.values()) > 1000
    # bin_data gates on is_finite, so the nan event keeps its bare code
    assert nans == ["VTL//heart_rate"]

    expected = {}
    for row in p.meds.filter(
        pl.col("numeric_value").is_not_null() & pl.col("numeric_value").is_not_nan()
    ).iter_rows(named=True):
        key = (row["subject_id"], row["time"], round(float(row["numeric_value"]), 4))
        expected[key] = expected.get(key, 0) + 1
    assert seen == expected


# --- min_training_ct -------------------------------------------------------

STRUCTURAL = ("BOS", "EOS", "CLCK//", "TIME//")


def training_counts(p) -> dict:
    """word -> occurrences over the training split, counted in plain python"""
    train = set(p.subjects_in_split("train"))
    seen = collections.Counter(
        int(t)
        for sbj, tokens in zip(p.tokens_times["subject_id"], p.tokens_times["tokens"])
        if sbj in train
        for t in tokens
    )
    return {w: seen[t] for w, t in p.vocab.items()}


def test_min_training_ct_zero_keeps_every_word_seen_in_training(runner):
    """the threshold is off at 0, so every training word earns a token"""
    tkzr, p = tokenized(runner, min_training_ct=0)
    counts = training_counts(p)
    assert min(c for w, c in counts.items() if w != "UNK") == 1
    assert tkzr.lookup.filter(pl.col("count") == 1).height > 0


@pytest.mark.parametrize("min_ct", [2, 25])
def test_min_training_ct_drops_words_below_the_threshold(runner, min_ct):
    """learned words clear the threshold; structural tokens are exempt"""
    tkzr, p = tokenized(runner, min_training_ct=min_ct)
    scant = tkzr.lookup.filter(pl.col("count") < min_ct).drop_nulls("count")
    assert all(w.startswith(STRUCTURAL) for w in scant["to_tokenize"])
    # nothing the tokenizer emits contradicts the counts it recorded
    for word, count in training_counts(p).items():
        if word == "UNK":
            continue
        assert count >= min_ct or word.startswith(STRUCTURAL), word


def test_min_training_ct_maps_the_words_it_drops_to_unk(runner):
    """a word pruned from the vocabulary tokenizes to UNK, not to nothing"""
    kept, _ = tokenized(runner, min_training_ct=0)
    pruned, p = tokenized(runner, min_training_ct=25)
    dropped = set(kept.lookup["to_tokenize"]) - set(pruned.lookup["to_tokenize"])
    assert dropped and not any(w.startswith(STRUCTURAL) for w in dropped)
    assert set(p.vocab) & dropped == set()
    # the events themselves survive: timelines keep their length, as UNK
    assert p.tokens_times.select(pl.col("tokens").list.len().sum()).item() == (
        Processed(kept.processed_data_home, runner.raw)
        .tokens_times.select(pl.col("tokens").list.len().sum())
        .item()
    )
    unks = [sum(1 for t in tokens if t == 0) for tokens in p.tokens_times["tokens"]]
    assert sum(unks) > 0


def test_min_training_ct_keeps_scant_structural_tokens(runner):
    """BOS/EOS and inserted clock/spacer tokens survive a threshold above them"""
    p = runner.minimal(
        hospitalizations=stay(24),
        vitals=vitals([JAN1 + h * HOUR for h in (1, 5, 9)], category="hr"),
        tokenization=tok_cfg(
            min_training_ct=1000, insert_clocks=True, insert_spacers=True
        ),
    )
    words = read(p)[0]
    assert words[0] == "BOS" and words[-1] == "EOS"
    assert any(w.startswith("CLCK//") for w in words)
    assert any(w.startswith("TIME//") for w in words)
    # the lone vital is far below the threshold and so is unked
    assert "VTL//hr" not in p.vocab
    assert 0 in p.tokens_times["tokens"].item().to_list()


# --- time spacers ----------------------------------------------------------

# (gap in minutes since the previous event, expected spacer label)
GAP_LABELS = [
    (4, None),  # below the smallest configured boundary: no spacer
    (5, "5m-15m"),  # exactly the smallest boundary: spacer
    (14, "5m-15m"),
    (15, "5m-15m"),  # a boundary value keeps the lower label
    (16, "15m-1h"),
    (59, "15m-1h"),
    (60, "15m-1h"),  # ditto
    (61, "1h-2h"),
    (120, "1h-2h"),
    (121, "2h-6h"),
]


def test_no_extra_tokens_are_inserted_by_default(pipeline):
    cfg = default_cfg("tokenization")
    assert cfg["insert_spacers"] is False and cfg["insert_clocks"] is False
    assert not [w for w in pipeline.vocab if w.startswith(("TIME//", "CLCK//"))]
    both = (
        pipeline.meds.group_by("subject_id")
        .len()
        .join(
            pipeline.tokens_times.select(
                "subject_id", pl.col("tokens").list.len().alias("n")
            ),
            on="subject_id",
            validate="1:1",
        )
    )
    assert both.height == pipeline.tokens_times.height > 0
    assert (both["n"] == both["len"] + 2).all()  # one token per event, plus BOS/EOS


def test_spacer_labels_at_and_around_the_configured_boundaries(runner):
    times, expected = [JAN1], ["BOS", "VTL//e0"]
    t = JAN1
    for i, (gap, label) in enumerate(GAP_LABELS):
        t += gap * MIN
        times.append(t)
        t += MIN  # a one minute follower, so no gap is measured against EOS
        times.append(t)
        if label is not None:
            expected.append(f"TIME//{label}")
        expected += [f"VTL//e{2 * i + 1}", f"VTL//e{2 * i + 2}"]
    expected.append("EOS")

    p = runner.minimal(
        hospitalizations=stay(72),
        vitals=vitals(times),
        tokenization=tok_cfg(insert_spacers=True),
    )
    words, stamps, _ = read(p)
    assert words == expected
    # nothing precedes the first event, and the spacer sits at the later event
    assert words[:2] == ["BOS", "VTL//e0"]
    assert stamps[words.index("TIME//5m-15m")] == utc(2024, 1, 1, 0, 10)


def test_custom_spacers_mapping_supplies_the_bucket_labels(runner):
    p = runner.minimal(
        hospitalizations=stay(24),
        vitals=vitals(
            [JAN1 + m * MIN for m in (0, 5, 15, 136, 137)]  # gaps 5, 10, 121, 1
        ),
        tokenization=tok_cfg(insert_spacers=True, spacers={"short": 10, "long": 120}),
    )
    assert read(p)[0] == [
        "BOS",
        "VTL//e0",
        "VTL//e1",  # a five minute gap is now under the smallest boundary
        "TIME//short",
        "VTL//e2",
        "TIME//long",
        "VTL//e3",
        "VTL//e4",
        "EOS",
    ]


def test_spacer_token_carries_the_following_events_numeric_value(runner):
    p = runner.minimal(
        hospitalizations=stay(24),
        vitals=vitals(
            [JAN1 + m * MIN for m in (0, 60, 61)],
            values=[10.0, 50.0, 90.0],
            category="hr",
        ),
        tokenization=tok_cfg(
            insert_spacers=True, n_bins=2, include_numeric_values=True
        ),
    )
    words, _, nums = read(p)
    assert words == [
        "BOS",
        "VTL//hr_Q0",
        "TIME//15m-1h",
        "VTL//hr_Q1",
        "VTL//hr_Q1",
        "EOS",
    ]
    # the spacer rides on the row of the event it precedes, and so repeats it
    assert nums == [None, 10.0, 50.0, 50.0, 90.0, None]


def test_spacer_tokens_match_the_gaps_between_neighbouring_times(runner):
    spacers = default_cfg("tokenization")["spacers"]
    _, p = tokenized(runner, insert_spacers=True)
    total, off_head = 0, 0
    assert p.tokens_times.height > 0
    for row in p.tokens_times.iter_rows(named=True):
        words, stamps = p.decode(row["tokens"]), list(row["times"])
        distinct = sorted(set(stamps))
        assert len(distinct) > 1, row["subject_id"]
        expected = []
        for prev, cur in zip(distinct, distinct[1:]):
            label = spacer_label(spacers, int((cur - prev).total_seconds() // 60))
            if label is not None:
                expected.append((cur, f"TIME//{label}"))
        assert marked(words, stamps, "TIME//") == expected, row["subject_id"]
        for i, w in enumerate(words):
            if not w.startswith("TIME//"):
                continue
            total += 1
            # the spacer rides on an event's row, so an event always follows it
            assert i + 1 < len(words) and stamps[i + 1] == stamps[i]
            if i > 0 and stamps[i - 1] == stamps[i]:
                off_head += 1  # ... but not necessarily the first of that instant
    assert total > 100
    assert off_head > 0


# --- clocks ----------------------------------------------------------------


def test_clock_tokens_appear_at_configured_hours_within_the_timeline(runner):
    p = runner.minimal(
        hospitalizations=stay(72),
        vitals=vitals(
            [datetime.datetime(2024, 1, 1, 9, 30), datetime.datetime(2024, 1, 2, 3, 10)]
        ),
        tokenization=tok_cfg(insert_clocks=True),
    )
    words, stamps, _ = read(p)
    assert words == [
        "BOS",
        "VTL//e0",
        "CLCK//12",
        "CLCK//16",
        "CLCK//20",
        "CLCK//00",
        "VTL//e1",
        "EOS",
    ]
    assert stamps == [
        utc(2024, 1, 1, 9, 30),
        utc(2024, 1, 1, 9, 30),
        utc(2024, 1, 1, 12),
        utc(2024, 1, 1, 16),
        utc(2024, 1, 1, 20),
        utc(2024, 1, 2, 0),
        utc(2024, 1, 2, 3, 10),
        utc(2024, 1, 2, 3, 10),
    ]


def test_custom_clocks_list_replaces_the_configured_hours(runner):
    p = runner.minimal(
        hospitalizations=stay(72),
        vitals=vitals(
            [datetime.datetime(2024, 1, 1, 9, 30), datetime.datetime(2024, 1, 2, 3, 10)]
        ),
        tokenization=tok_cfg(insert_clocks=True, clocks=["09", "21"]),
    )
    words, stamps, _ = read(p)
    # 09:00 is the truncated hour of the first event and so is excluded
    assert words == ["BOS", "VTL//e0", "CLCK//21", "VTL//e1", "EOS"]
    assert stamps[2] == utc(2024, 1, 1, 21)


def test_clock_tokens_repeat_once_per_matching_hour(runner):
    p = runner.minimal(
        hospitalizations=stay(24 * 5),
        vitals=vitals(
            [
                datetime.datetime(2024, 1, 1, 9, 30),
                datetime.datetime(2024, 1, 4, 10, 10),
            ]
        ),
        tokenization=tok_cfg(insert_clocks=True, clocks=["00"]),
    )
    words, stamps, _ = read(p)
    assert words == ["BOS", "VTL//e0"] + ["CLCK//00"] * 3 + ["VTL//e1", "EOS"]
    assert marked(words, stamps, "CLCK//") == [
        (utc(2024, 1, d, 0), "CLCK//00") for d in (2, 3, 4)
    ]


def test_clock_tokens_on_the_full_dataset_land_on_configured_hours(runner):
    hours = [str(h) for h in default_cfg("tokenization")["clocks"]]
    _, p = tokenized(runner, insert_clocks=True)
    assert {w for w in p.vocab if w.startswith("CLCK//")} == {
        f"CLCK//{h}" for h in hours
    }
    bounds = {
        row["subject_id"]: (row["lo"], row["hi"])
        for row in p.meds.group_by("subject_id")
        .agg(pl.col("time").min().alias("lo"), pl.col("time").max().alias("hi"))
        .iter_rows(named=True)
    }
    total = 0
    assert p.tokens_times.height > 0
    for row in p.tokens_times.iter_rows(named=True):
        words, stamps = p.decode(row["tokens"]), list(row["times"])
        found = marked(words, stamps, "CLCK//")
        lo, hi = bounds[row["subject_id"]]
        assert found == clock_marks(lo, hi, hours), row["subject_id"]
        for t, w in found:
            assert (t.minute, t.second, t.microsecond) == (0, 0, 0)
            assert t.strftime("%H") == w.removeprefix("CLCK//")
        total += len(found)
    assert total > 100


# --- spacers and clocks together -------------------------------------------

CLOCK_SPACER_VITALS = [
    datetime.datetime(2024, 1, 1, 23, 0),
    datetime.datetime(2024, 1, 2, 0, 20),
    datetime.datetime(2024, 1, 2, 0, 21),
]


def test_spacers_alone_bridge_a_gap_containing_an_unused_clock_hour(runner):
    p = runner.minimal(
        hospitalizations=stay(48),
        vitals=vitals(CLOCK_SPACER_VITALS),
        tokenization=tok_cfg(insert_spacers=True),
    )
    # 23:00 -> 00:20 is one 80 minute gap
    assert read(p)[0] == ["BOS", "VTL//e0", "TIME//1h-2h", "VTL//e1", "VTL//e2", "EOS"]


def test_spacers_are_measured_after_clocks_are_inserted(runner):
    p = runner.minimal(
        hospitalizations=stay(48),
        vitals=vitals(CLOCK_SPACER_VITALS),
        tokenization=tok_cfg(insert_spacers=True, insert_clocks=True),
    )
    words, stamps, _ = read(p)
    # the clock at 00:00 splits the same 80 minute gap into 60 + 20 minutes
    assert words == [
        "BOS",
        "VTL//e0",
        "TIME//15m-1h",
        "CLCK//00",
        "TIME//15m-1h",
        "VTL//e1",
        "VTL//e2",
        "EOS",
    ]
    assert stamps[2] == stamps[3] == utc(2024, 1, 2, 0)
    assert stamps[4] == stamps[5] == utc(2024, 1, 2, 0, 20)


def test_inserted_tokens_leave_the_event_tokens_and_their_times_intact(
    runner, pipeline
):
    base = {
        row["subject_id"]: [
            (w, t) for w, t in zip(pipeline.decode(row["tokens"]), row["times"])
        ]
        for row in pipeline.tokens_times.iter_rows(named=True)
    }
    assert len(base) > 0
    for prefix, overrides in (
        ("TIME//", {"insert_spacers": True}),
        ("CLCK//", {"insert_clocks": True}),
    ):
        _, p = tokenized(runner, **overrides)
        assert p.tokens_times.height == len(base)
        for row in p.tokens_times.iter_rows(named=True):
            kept = [
                (w, t)
                for w, t in zip(p.decode(row["tokens"]), row["times"])
                if not w.startswith(prefix)
            ]
            was = base[row["subject_id"]]
            assert [t for _, t in kept] == [t for _, t in was], row["subject_id"]
            # NB: cotemporaneous same-priority events may be permuted, so only
            # the multiset of (word, time) pairs is preserved
            assert sorted(kept) == sorted(was), row["subject_id"]


# --- ordering --------------------------------------------------------------


def test_prefix_missing_from_ordering_sorts_last(runner):
    # AAA is absent, so it takes priority len(ordering) and follows even EOS
    assert two_prefix_timeline(runner, ["BOS", "ZZZ", "EOS"]) == [
        "BOS",
        "ZZZ//z",
        "AAA//a",
        "EOS",
        "AAA//a",
    ]


def test_short_ordering_can_push_bos_behind_the_first_event(runner):
    assert two_prefix_timeline(runner, ["AAA", "BOS", "EOS"]) == [
        "AAA//a",
        "BOS",
        "ZZZ//z",
        "AAA//a",
        "EOS",
    ]


def test_ordering_breaks_ties_between_cotemporaneous_prefixes(runner):
    # swapping two prefixes in the ordering swaps their cotemporaneous tokens
    assert two_prefix_timeline(runner, ["BOS", "AAA", "ZZZ", "EOS"]) == [
        "BOS",
        "AAA//a",
        "ZZZ//z",
        "AAA//a",
        "EOS",
    ]
    assert two_prefix_timeline(runner, ["BOS", "ZZZ", "AAA", "EOS"]) == [
        "BOS",
        "ZZZ//z",
        "AAA//a",
        "AAA//a",
        "EOS",
    ]


def test_empty_ordering_raises_a_polars_schema_error(runner):
    # actual behaviour: an empty ordering makes the priority join key null-typed
    with pytest.raises(pl.exceptions.SchemaError):
        two_prefix_timeline(runner, [])
