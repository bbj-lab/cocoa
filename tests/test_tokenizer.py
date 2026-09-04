#!/usr/bin/env python3

"""core tokenization: timeline structure, vocabulary, and learned bins"""

import collections
import datetime
import itertools
import math
import re

import polars as pl
import pytest
import synth
from conftest import default_cfg
from omegaconf import OmegaConf

N_BINS = default_cfg("tokenization")["n_bins"]
ORDERING = list(default_cfg("tokenization")["ordering"])
PRIORITY = {p: i for i, p in enumerate(ORDERING)}

# labs planted in exactly one non-training split by synth
MARKERS = {
    "tuning": synth.TUNING_ONLY_LAB.replace(" ", "_"),
    "held_out": synth.HELD_OUT_ONLY_LAB.replace(" ", "_"),
}

Q_SUFFIX = re.compile(r"^(?P<word>.+)_Q(?P<q>\d+)$")


def nearest_rank_quantile(values, q):
    """value at rank round(q * (n - 1)); polars' default quantile"""
    ordered = sorted(values)
    return ordered[math.floor(q * (len(ordered) - 1) + 0.5)]


def finite_values(meds, code, subjects=None):
    """the non-null, non-nan numeric values recorded for `code`"""
    df = meds.filter(
        pl.col("code") == code,
        pl.col("numeric_value").is_not_null(),
        pl.col("numeric_value").is_not_nan(),
    )
    if subjects is not None:
        df = df.filter(pl.col("subject_id").is_in(list(subjects)))
    return df["numeric_value"].to_list()


def prefix_of(word):
    """the code prefix a decoded token belongs to"""
    return word.split("//")[0]


@pytest.fixture(scope="module")
def timelines(pipeline):
    """subject_id -> (decoded words, times)"""
    return {
        row["subject_id"]: (pipeline.decode(row["tokens"]), list(row["times"]))
        for row in pipeline.tokens_times.iter_rows(named=True)
    }


@pytest.fixture(scope="module")
def bins(pipeline):
    """code -> learned break points, as recorded in tokenizer.yaml"""
    return {k: list(v) for k, v in dict(pipeline.tokenizer_yaml.bins).items()}


@pytest.fixture(scope="module")
def train_subjects(pipeline):
    return pipeline.subjects_in_split("train")


# --- structure -------------------------------------------------------------


def test_tokens_times_schema_is_subject_tokens_times(pipeline):
    schema = pipeline.tokens_times.schema
    assert list(schema.keys()) == ["subject_id", "tokens", "times"]
    assert schema["subject_id"] == pl.String
    assert schema["tokens"] == pl.List(pl.UInt32)
    inner = schema["times"].inner
    assert isinstance(inner, pl.Datetime)
    # times stay tz-aware, in the zone the collation config asked for
    assert inner.time_zone is not None
    assert inner.time_zone == pipeline.meds.schema["time"].time_zone


def test_every_planted_subject_tokenized_exactly_once(pipeline):
    ids = pipeline.tokens_times["subject_id"]
    assert len(ids) == 48
    assert ids.n_unique() == len(ids)
    assert set(ids) == set(pipeline.manifest.subject_ids)
    assert set(ids) == set(pipeline.splits["subject_id"])


def test_tokens_and_times_have_equal_lengths(timelines):
    assert len(timelines) == 48
    for sid, (words, times) in timelines.items():
        assert len(words) == len(times), sid
        assert len(words) >= 3, sid  # BOS, at least one event, EOS


def test_times_are_non_decreasing(timelines):
    assert timelines
    for sid, (_, times) in timelines.items():
        assert all(a <= b for a, b in itertools.pairwise(times)), sid


def test_timelines_start_with_bos_and_end_with_eos(timelines):
    assert timelines
    for sid, (words, _) in timelines.items():
        assert words[0] == "BOS", sid
        assert words[-1] == "EOS", sid


def test_timeline_bounds_match_subject_event_times(pipeline, timelines):
    checked = 0
    for sid, (_, times) in timelines.items():
        events = pipeline.meds.filter(pl.col("subject_id") == sid)["time"].to_list()
        assert events, sid
        assert times[0] == min(events), sid
        assert times[-1] == max(events), sid
        checked += 1
    assert checked == 48


def test_fused_timeline_holds_one_token_per_event_plus_ends(pipeline, timelines):
    """with fusion on and no spacers/clocks, every meds row is one token"""
    counts = dict(pipeline.meds.group_by("subject_id").len().iter_rows())
    assert len(counts) == 48
    for sid, (words, times) in timelines.items():
        assert len(words) == counts[sid] + 2, sid
        events = pipeline.meds.filter(pl.col("subject_id") == sid)["time"].to_list()
        # BOS repeats the first event time and EOS the last
        assert sorted(times) == sorted(events + [min(events), max(events)]), sid


# --- vocabulary ------------------------------------------------------------


def test_unk_is_token_zero_and_vocabulary_is_contiguous(pipeline):
    vocab = pipeline.vocab
    assert vocab["UNK"] == 0
    assert len(vocab) > 100
    assert sorted(vocab.values()) == list(range(len(vocab)))


def test_lookup_has_no_duplicate_words_or_tokens(runner):
    tkzr = runner.tokenize(processed=runner.seed_collated())
    assert tkzr.lookup.height > 100
    assert tkzr.lookup["to_tokenize"].n_unique() == tkzr.lookup.height
    assert tkzr.lookup["token"].n_unique() == tkzr.lookup.height
    assert tkzr.lookup.filter(pl.col("to_tokenize") == "UNK")["token"].item() == 0


def test_lookup_counts_training_occurrences_of_each_word(pipeline):
    """every vocabulary word carries how often it was seen while training"""
    train = pipeline.subjects_in_split("train")
    seen = collections.Counter(
        int(t)
        for tokens in pipeline.tokens_times.filter(pl.col("subject_id").is_in(train))[
            "tokens"
        ]
        for t in tokens
    )
    counts = pipeline.counts
    assert set(counts) == set(pipeline.vocab)
    assert counts["UNK"] is None  # UNK is never learned, so it has no count
    assert sum(seen.values()) > 1000
    assert all(counts[w] == seen[t] for w, t in pipeline.vocab.items() if w != "UNK")


def test_counts_are_learned_on_the_training_split_only(runner):
    """held-out occurrences neither create vocabulary nor inflate counts"""
    tkzr = runner.tokenize(processed=runner.seed_collated())
    counted = tkzr.lookup.filter(pl.col("to_tokenize") != "UNK")
    assert counted.height == tkzr.lookup.height - 1
    assert counted["count"].min() >= 1
    assert counted["count"].null_count() == 0
    assert (
        counted["count"].sum()
        < tkzr.get_all().select(pl.col("tokens").list.len().sum()).collect().item()
    )  # the whole corpus is larger than the training split


def test_vocabulary_words_are_alphabetical_after_unk(pipeline):
    words = [w for w, _ in sorted(pipeline.vocab.items(), key=lambda kv: kv[1])]
    assert words[0] == "UNK"
    assert words[1:] == sorted(words[1:])


def test_every_timeline_token_is_in_the_vocabulary(pipeline, timelines):
    seen = {int(t) for tokens in pipeline.tokens_times["tokens"] for t in tokens}
    assert len(seen) > 100
    assert seen <= set(pipeline.vocab.values())
    assert max(seen) < len(pipeline.vocab)


def test_every_vocabulary_word_appears_in_a_training_timeline(
    timelines, train_subjects, pipeline
):
    assert len(train_subjects) == 33
    used = set()
    for sid in train_subjects:
        used.update(timelines[sid][0])
    missing = set(pipeline.vocab) - used - {"UNK"}
    assert not missing


def test_training_timelines_have_no_unk_tokens(timelines, train_subjects):
    assert train_subjects
    for sid in train_subjects:
        assert "UNK" not in timelines[sid][0], sid


@pytest.mark.parametrize("split", ["tuning", "held_out"])
def test_non_training_only_codes_are_absent_from_the_vocabulary(pipeline, split):
    marker = MARKERS[split]
    planted = pipeline.meds.filter(pl.col("code").str.ends_with(marker))
    assert planted.height > 0  # synth really planted it
    assert set(planted["subject_id"]) == set(pipeline.subjects_in_split(split))
    assert set(planted["code"]) == {f"LAB-ORD//{marker}", f"LAB-RES//{marker}"}
    assert not [w for w in pipeline.vocab if marker in w]
    assert not [c for c in dict(pipeline.tokenizer_yaml.bins) if marker in c]


@pytest.mark.parametrize("split", ["tuning", "held_out"])
def test_unk_tokens_land_exactly_on_the_non_training_marker_labs(
    pipeline, timelines, split
):
    marker = MARKERS[split]
    labs = pl.read_parquet(pipeline.manifest.root / "clif_labs.parquet")
    planted = dict(
        labs.filter(pl.col("lab_category").str.replace_all(" ", "_") == marker)
        .group_by("hospitalization_id")
        .len()
        .iter_rows()
    )
    subjects = pipeline.subjects_in_split(split)
    assert subjects and set(planted) == set(subjects)
    for sid in subjects:
        words, times = timelines[sid]
        # each planted lab row is collated twice: ordered, then resulted
        assert sum(w == "UNK" for w in words) == 2 * planted[sid] > 0, sid
        unk_times = sorted(t for w, t in zip(words, times) if w == "UNK")
        marker_times = sorted(
            pipeline.meds.filter(
                pl.col("subject_id") == sid, pl.col("code").str.ends_with(marker)
            )["time"].to_list()
        )
        assert unk_times == marker_times, sid


# --- bins ------------------------------------------------------------------


def test_bins_have_n_bins_minus_one_breaks(pipeline, bins):
    assert len(bins) == 19
    for code, breaks in bins.items():
        assert len(breaks) == N_BINS - 1, code
    assert len(dict(pipeline.tokenizer_yaml.bins)) == len(bins)


def test_bins_cover_exactly_the_codes_numeric_in_training(
    pipeline, bins, train_subjects
):
    numeric_in_training = set(
        pipeline.meds.filter(
            pl.col("subject_id").is_in(train_subjects),
            pl.col("numeric_value").is_not_null(),
            pl.col("numeric_value").is_not_nan(),
        )["code"]
    )
    assert numeric_in_training == set(bins)


def test_bins_equal_training_quantiles(pipeline, bins, train_subjects):
    assert bins
    for code, breaks in bins.items():
        values = finite_values(pipeline.meds, code, train_subjects)
        assert len(values) >= 9, code
        expected = [nearest_rank_quantile(values, i / N_BINS) for i in range(1, N_BINS)]
        assert breaks == pytest.approx(expected, rel=1e-6), code


@pytest.mark.parametrize(
    "code", ["VTL//heart_rate", "LAB-RES//potassium", "AGE//age", "RESP//peep_set"]
)
def test_bins_differ_from_quantiles_over_all_subjects(pipeline, bins, code):
    """held-out values must not move the break points"""
    everyone = finite_values(pipeline.meds, code)
    training = finite_values(pipeline.meds, code, pipeline.subjects_in_split("train"))
    assert len(everyone) > len(training) > 0
    over_all = [nearest_rank_quantile(everyone, i / N_BINS) for i in range(1, N_BINS)]
    assert bins[code] != pytest.approx(over_all, rel=1e-9)
    assert bins[code] == pytest.approx(
        [nearest_rank_quantile(training, i / N_BINS) for i in range(1, N_BINS)],
        rel=1e-6,
    )


def test_bins_ignore_nan_and_null_numeric_values(pipeline, bins):
    nan_rows = pipeline.meds.filter(
        pl.col("code") == "VTL//heart_rate", pl.col("numeric_value").is_nan()
    )
    assert nan_rows.height == 1  # synth plants one nan heart rate
    assert nan_rows["subject_id"].item() in pipeline.subjects_in_split("train")
    null_rows = pipeline.meds.filter(
        pl.col("code") == "MED-CTS//propofol", pl.col("numeric_value").is_null()
    )
    assert null_rows.height > 0  # unconverted doses carry no value
    for code, breaks in bins.items():
        assert not any(b != b for b in breaks), code


def test_bins_are_non_decreasing(bins):
    assert bins
    for code, breaks in bins.items():
        assert all(a <= b for a, b in itertools.pairwise(breaks)), code


# --- binning ---------------------------------------------------------------


@pytest.mark.parametrize(
    "code", ["VTL//heart_rate", "LAB-RES//potassium", "AGE//age", "MED-INT//morphine"]
)
def test_q_suffix_counts_breaks_at_or_below_the_value(pipeline, timelines, bins, code):
    """the Q index of a fused token is #{break <= value}, hand-counted here"""
    breaks = bins[code]
    expected = collections.defaultdict(list)
    rows = pipeline.meds.filter(
        pl.col("code") == code,
        pl.col("numeric_value").is_not_null(),
        pl.col("numeric_value").is_not_nan(),
    ).select("subject_id", "time", "numeric_value")
    assert rows.height > 30
    for sid, time, value in rows.iter_rows():
        q = sum(b <= value for b in breaks)
        assert 0 <= q <= N_BINS - 1
        expected[(sid, time)].append(f"{code}_Q{q}")
    nans = pipeline.meds.filter(
        pl.col("code") == code, pl.col("numeric_value").is_nan()
    ).select("subject_id", "time")

    seen = collections.defaultdict(list)
    wanted = re.compile(rf"^{re.escape(code)}_Q\d+$")
    for sid, (words, times) in timelines.items():
        for word, time in zip(words, times):
            if wanted.match(word):
                seen[(sid, time)].append(word)

    assert set(seen) == set(expected)
    for key, words in expected.items():
        assert sorted(seen[key]) == sorted(words), key
    for sid, time in nans.iter_rows():
        # a non-finite value is not binned at all, so it emits no _Q token
        assert (sid, time) not in seen


def test_q_index_stays_within_the_bin_range(pipeline):
    suffixed = [m for m in map(Q_SUFFIX.match, pipeline.vocab) if m]
    assert len(suffixed) > 100
    for m in suffixed:
        assert 0 <= int(m.group("q")) <= N_BINS - 1, m.group(0)


def test_codes_without_numeric_values_get_no_q_suffix(pipeline):
    numeric = pipeline.meds.group_by("code").agg(
        has_value=pl.col("numeric_value").is_not_null().any()
    )
    never = set(numeric.filter(~pl.col("has_value"))["code"])
    assert {"RESP//imv", "LAB-ORD//sodium", "DSCG//home"} <= never
    binned = {m.group("word") for m in map(Q_SUFFIX.match, pipeline.vocab) if m}
    assert not (never & binned)
    # propofol has converted (numeric) and unconverted (bare) administrations
    assert "MED-CTS//propofol" in pipeline.vocab
    assert f"MED-CTS//propofol_Q{N_BINS - 1}" in pipeline.vocab


def test_bos_and_eos_never_carry_a_q_suffix(pipeline, timelines):
    assert {"BOS", "EOS"} <= set(pipeline.vocab)
    assert not [w for w in pipeline.vocab if w.startswith(("BOS_", "EOS_"))]
    assert not [
        w
        for words, _ in timelines.values()
        for w in words
        if prefix_of(w) in ("BOS", "EOS") and w not in ("BOS", "EOS")
    ]


def test_median_bins_give_hand_computed_tokens(runner):
    """two bins put values below the median in Q0 and the rest in Q1"""
    admit = datetime.datetime(2024, 1, 1, 0, 0)
    processed = runner.minimal(
        hospitalizations=[
            {"admission_dttm": admit, "discharge_dttm": admit + datetime.timedelta(1)}
        ],
        vitals=[
            {"recorded_dttm": admit + datetime.timedelta(hours=h), "vital_value": v}
            for h, v in [(1, 1.0), (2, 2.0), (3, 3.0), (4, None)]
        ],
        tokenization={**default_cfg("tokenization"), "n_bins": 2},
    )
    assert dict(processed.tokenizer_yaml.bins)["VTL//heart_rate"] == [2.0]
    assert processed.timeline("H0") == [
        "BOS",
        "VTL//heart_rate_Q0",
        "VTL//heart_rate_Q1",
        "VTL//heart_rate_Q1",
        "VTL//heart_rate",  # a null value gets no Q suffix
        "EOS",
    ]


def test_n_bins_sets_break_count_and_q_range(runner):
    n_bins = 4
    processed_dir = runner.seed_collated()
    tkzr = runner.tokenize(
        processed=processed_dir, cfg={**default_cfg("tokenization"), "n_bins": n_bins}
    )
    assert tkzr.bins.width == n_bins  # code + n_bins - 1 breaks
    vocab = dict(OmegaConf.load(processed_dir / "tokenizer.yaml").lookup)
    suffixed = [m for m in map(Q_SUFFIX.match, vocab) if m]
    assert suffixed
    assert {int(m.group("q")) for m in suffixed} == set(range(n_bins))
    breaks = dict(zip(tkzr.bins["code"], tkzr.bins.rows()))
    values = finite_values(
        pl.read_parquet(processed_dir / "meds.parquet"),
        "VTL//heart_rate",
        pl.read_parquet(processed_dir / "subject_splits.parquet")
        .filter(pl.col("split") == "train")["subject_id"]
        .to_list(),
    )
    assert list(breaks["VTL//heart_rate"][1:]) == pytest.approx(
        [nearest_rank_quantile(values, i / n_bins) for i in range(1, n_bins)], rel=1e-6
    )


def test_nan_numeric_value_gets_no_bin_suffix(pipeline, timelines, bins):
    """
    bin_data gates on is_finite, so a nan is neither used to learn the breaks
    nor assigned one: the event keeps its bare code, the same shape the config
    uses for events that carry no numeric_value at all
    """
    row = pipeline.meds.filter(
        pl.col("code") == "VTL//heart_rate", pl.col("numeric_value").is_nan()
    ).row(0, named=True)
    words, times = timelines[row["subject_id"]]
    at_nan = [
        w
        for w, t in zip(words, times)
        if t == row["time"] and w.startswith("VTL//heart_rate")
    ]
    assert at_nan == ["VTL//heart_rate"]


# --- ordering --------------------------------------------------------------


def test_cotemporaneous_tokens_follow_the_configured_ordering(
    timelines, train_subjects
):
    runs = 0
    for sid in train_subjects:  # training timelines carry no UNK to misread
        words, times = timelines[sid]
        for time, group in itertools.groupby(zip(times, words), key=lambda p: p[0]):
            priorities = [PRIORITY[prefix_of(w)] for _, w in group]
            if len(priorities) > 1:
                runs += 1
                assert all(a <= b for a, b in itertools.pairwise(priorities)), (
                    sid,
                    time,
                    priorities,
                )
    assert runs > 100


def test_admission_tokens_are_ordered_by_prefix(timelines):
    """everything recorded at admission, collapsed to one entry per prefix"""
    words, times = timelines["H00000"]
    at_admission = [w for w, t in zip(words, times) if t == times[0]]
    assert [p for p, _ in itertools.groupby(map(prefix_of, at_admission))] == [
        "BOS",
        "AGE",
        "SEX",
        "RACE",
        "ETHN",
        "ADMN",
        "XFR-IN",
        "MED-CTS",
        "MED-INT",
        "LAB-ORD",
        "CODE",
        "VTL",
        "ASMT",
        "RESP",
        "SOFA",
        "LABEL",
    ]


def minimal_two_prefix_cfg():
    """collate heart_rate under VTL and spo2 under ZZZ, absent from `ordering`"""
    return synth.minimal_collation_cfg(
        entries=[
            {
                "table": "clif_vitals",
                "prefix": prefix,
                "filter_expr": f'pl.col("vital_category") == "{category}"',
                "code": "vital_category",
                "numeric_value": "vital_value",
                "time": "recorded_dttm",
            }
            for prefix, category in (("VTL", "heart_rate"), ("ZZZ", "spo2"))
        ]
    )


def test_prefix_absent_from_ordering_sorts_last_among_cotemporaneous(runner):
    admit = datetime.datetime(2024, 1, 1, 0, 0)
    assert "ZZZ" not in ORDERING
    processed = runner.minimal(
        hospitalizations=[
            {"admission_dttm": admit, "discharge_dttm": admit + datetime.timedelta(1)}
        ],
        vitals=[
            {"recorded_dttm": admit + datetime.timedelta(hours=1), "vital_value": 60.0},
            {
                "recorded_dttm": admit + datetime.timedelta(hours=1),
                "vital_category": "spo2",
                "vital_value": 95.0,
            },
            {"recorded_dttm": admit + datetime.timedelta(hours=2), "vital_value": 70.0},
        ],
        collation=minimal_two_prefix_cfg(),
    )
    timeline = processed.timeline("H0")
    assert [prefix_of(w) for w in timeline] == ["BOS", "VTL", "ZZZ", "VTL", "EOS"]


def test_prefix_absent_from_ordering_sorts_after_eos(runner):
    """
    a prefix missing from `ordering` outranks even EOS, so an event at the last
    timestamp is emitted after the end-of-sequence token -- see bugs
    """
    admit = datetime.datetime(2024, 1, 1, 0, 0)
    processed = runner.minimal(
        hospitalizations=[
            {"admission_dttm": admit, "discharge_dttm": admit + datetime.timedelta(1)}
        ],
        vitals=[
            {"recorded_dttm": admit + datetime.timedelta(hours=1), "vital_value": 60.0},
            {
                "recorded_dttm": admit + datetime.timedelta(hours=2),
                "vital_category": "spo2",
                "vital_value": 95.0,
            },
        ],
        collation=minimal_two_prefix_cfg(),
    )
    assert processed.timeline("H0") == [
        "BOS",
        "VTL//heart_rate_Q9",
        "EOS",
        "ZZZ//spo2_Q9",
    ]


# --- determinism -----------------------------------------------------------


def test_retokenizing_learns_the_same_lookup_and_bins(runner, pipeline):
    dirs = [runner.seed_collated(), runner.seed_collated()]
    for d in dirs:
        runner.tokenize(processed=d)
    saved = []
    for d in dirs + [pipeline.path]:
        cfg = OmegaConf.to_container(OmegaConf.load(d / "tokenizer.yaml"))
        cfg.pop("created_dttm")
        saved.append(cfg)
    assert len(dict(saved[0]["lookup"])) > 100
    assert saved[0] == saved[1] == saved[2]


def test_retokenizing_reproduces_times_and_token_counts(runner, pipeline):
    """
    the tokens of a timeline are reproducible as a multiset but *not* as a
    sequence: cotemporaneous tokens of equal priority get shuffled -- see bugs
    """
    d = runner.seed_collated()
    runner.tokenize(processed=d)
    again = pl.read_parquet(d / "tokens_times.parquet").sort("subject_id")
    first = pipeline.tokens_times.sort("subject_id")
    assert again.height == first.height == 48
    assert again["subject_id"].to_list() == first["subject_id"].to_list()
    for a, b in zip(again.iter_rows(named=True), first.iter_rows(named=True)):
        assert list(a["times"]) == list(b["times"]), a["subject_id"]
        assert sorted(a["tokens"]) == sorted(b["tokens"]), a["subject_id"]
