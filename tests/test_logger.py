#!/usr/bin/env python3

"""
the --verbose summary logger: construction, and the summary statistics it
reports for each of the three pipeline stages
"""

import collections
import datetime
import logging
import re
import shutil
import statistics

import polars as pl
import pytest
import synth
from conftest import default_cfg
from rich.logging import RichHandler

from cocoa.logger import Logger
from cocoa.winnower import Winnower

# rich renders each record as "LEVEL <emoji> [time] message", continuation
# lines of a multi-line message being indented instead
HEAD = re.compile(r"^(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+\S+\s+\[[^\]]*\]\s*")
EDGE = "│"  # the box-drawing char bounding a polars table row
CELL = "┆"  # the box-drawing char between cells of a row
ELLIPSIS = "…"  # polars elides long strings with this
WINNOWER_INPUTS = ("tokens_times.parquet", "subject_splits.parquet", "tokenizer.yaml")


@pytest.fixture(autouse=True)
def _stable_tables():
    """pin the table rendering logger.py sets at import, in case a sibling changed it"""
    with pl.Config(tbl_rows=100, tbl_width_chars=500):
        yield


@pytest.fixture
def logger() -> Logger:
    return Logger()


@pytest.fixture(scope="module")
def lookup(pipeline) -> pl.DataFrame:
    """the tokenizer's lookup table, as `Tokenizer.from_yaml` rebuilds it"""
    return pl.DataFrame(
        list(pipeline.vocab.items()),
        schema={"to_tokenize": pl.String, "token": pl.UInt32},
        orient="row",
    )


@pytest.fixture(scope="module")
def outcomes(pipeline) -> list:
    """the outcome tokens the winnower hands to summarize_thresholded"""
    return Winnower(processed_data_home=pipeline.path).grokked_outcome_tokens


def records(captured: str) -> list:
    """one string per log record written to stdout, headers stripped"""
    msgs = []
    for line in captured.splitlines():
        line = line.rstrip()
        head = HEAD.match(line)
        if head is not None:
            msgs.append(line[head.end() :])
        elif msgs:
            msgs[-1] += "\n" + line
    return msgs


def message(captured: str, label: str) -> str:
    """the one record beginning with `label`"""
    hits = [m for m in records(captured) if m.startswith(label)]
    assert len(hits) == 1, f"{label!r} reported {len(hits)} times"
    return hits[0]


def table(msg: str) -> list:
    """the rows of the polars table rendered inside one record, as dicts"""
    cells = [
        [c.strip() for c in line.strip().strip(EDGE).split(CELL)]
        for line in msg.splitlines()
        if CELL in line
    ]
    assert len(cells) > 3, f"no table rows in {msg!r}"
    header, _dashes, _dtypes, *body = cells
    return [dict(zip(header, row)) for row in body]


def shape(msg: str) -> tuple:
    """the (height, width) polars printed above a table"""
    hit = re.search(r"shape: \(([\d_]+), ([\d_]+)\)", msg)
    assert hit is not None, f"no shape in {msg!r}"
    return tuple(int(g.replace("_", "")) for g in hit.groups())


def displayed_as(shown: str, actual: str) -> bool:
    """does `shown` render `actual`, allowing for polars' elision?"""
    return (
        actual.startswith(shown[:-1]) if shown.endswith(ELLIPSIS) else shown == actual
    )


def resolve(shown: str, candidates) -> str:
    """the one candidate that `shown` could be a rendering of"""
    hits = [c for c in candidates if displayed_as(shown, c)]
    assert len(hits) == 1, f"{shown!r} matches {hits}"
    return hits[0]


def captured(capsys, fn, *args) -> str:
    """whatever `fn` alone writes to stdout"""
    capsys.readouterr()
    fn(*args)
    out = capsys.readouterr().out
    assert out, f"{fn} wrote nothing to stdout"
    return out


def split_of(splits: pl.DataFrame) -> dict:
    return dict(zip(splits["subject_id"].to_list(), splits["split"].to_list()))


def seed_winnower(runner, pipeline):
    """a fresh directory holding only what the winnower reads"""
    dest = runner.dir()
    for f in WINNOWER_INPUTS:
        shutil.copy(pipeline.path / f, dest / f)
    return dest


# ------------------------------------------------------------------ construction


def test_logger_is_an_info_level_logging_logger(logger):
    assert isinstance(logger, logging.Logger)
    assert logger.name == "cocoa"
    assert logger.level == logging.INFO
    assert logger.propagate is False


def test_logger_has_exactly_one_rich_handler(logger):
    assert len(logger.handlers) == 1
    handler = logger.handlers[0]
    assert isinstance(handler, RichHandler)
    assert handler.level == logging.INFO


def test_logger_writes_each_record_once_with_a_branded_prefix(capsys):
    Logger()  # a second instance must neither steal nor duplicate output
    logger = Logger()
    out = captured(capsys, logger.info, "marker-quokka")
    assert out.count("marker-quokka") == 1
    assert "☕" in out  # the coffee cup of the shipped format


@pytest.mark.parametrize(
    "level,times", [("debug", 0), ("info", 1), ("warning", 1), ("error", 1)]
)
def test_records_below_info_are_not_emitted(capsys, logger, level, times):
    capsys.readouterr()
    getattr(logger, level)(f"marker-{level}")
    assert capsys.readouterr().out.count(f"marker-{level}") == times


def test_records_bypass_the_root_logger(capsys, caplog, logger):
    """propagate is False, so caplog cannot see cocoa's records -- capsys can"""
    capsys.readouterr()
    with caplog.at_level(logging.DEBUG):
        logger.info("marker-unpropagated")
    assert caplog.records == []
    assert "marker-unpropagated" in capsys.readouterr().out


# -------------------------------------------------------------------- expressions


@pytest.mark.parametrize(
    "code,prefix",
    [
        ("VTL//heart_rate", "VTL"),
        ("LAB-RES//quokka_marker", "LAB-RES"),
        ("BOS", "BOS"),
        ("A//B//C", "A"),
        ("//x", ""),
        ("", ""),
        (None, None),
    ],
)
def test_code_type_is_the_text_before_the_first_double_slash(logger, code, prefix):
    df = pl.DataFrame({"code": [code]}, schema={"code": pl.String})
    assert df.select(logger.code_type)["code"].to_list() == [prefix]


def test_split_order_sorts_train_then_tuning_then_held_out(logger):
    df = pl.DataFrame({"split": ["held_out", "train", "held_out", "tuning"]})
    ordered = df.sort(logger.split_order)["split"].to_list()
    assert ordered == ["train", "tuning", "held_out", "held_out"]
    assert ordered != sorted(ordered)  # alphabetically held_out would come first


# --------------------------------------------------------------- summarize_meds_like


def test_summarize_meds_like_reports_height_and_subject_count(pipeline, logger, capsys):
    meds = pipeline.meds
    out = captured(capsys, logger.summarize_meds_like, meds.lazy(), pipeline.splits)
    n_subjects = len(set(meds["subject_id"].to_list()))
    assert n_subjects == len(pipeline.manifest.subject_ids)
    assert message(out, "total rows:") == f"total rows: {meds.height}"
    assert message(out, "unique subjects:") == f"unique subjects: {n_subjects}"


def test_summarize_meds_like_category_breakdown_matches_prefix_counts(
    pipeline, logger, capsys
):
    out = captured(
        capsys, logger.summarize_meds_like, pipeline.meds.lazy(), pipeline.splits
    )
    codes = pipeline.meds["code"].to_list()
    counts = collections.Counter(c.split("//")[0] for c in codes)
    rows = table(message(out, "by category:"))
    assert len(rows) == len(counts) > 1
    assert {r["code"] for r in rows} == set(counts)
    rates = [float(r["proportion"]) for r in rows]
    assert rates == sorted(rates, reverse=True)  # value_counts(sort=True)
    assert sum(rates) == pytest.approx(1.0, abs=1e-4)
    for r in rows:
        assert float(r["proportion"]) == pytest.approx(
            counts[r["code"]] / len(codes), abs=1e-6
        )


def test_summarize_meds_like_example_rows_come_from_the_frame(pipeline, logger, capsys):
    out = captured(
        capsys, logger.summarize_meds_like, pipeline.meds.lazy(), pipeline.splits
    )
    msg = message(out, "example rows:")
    assert shape(msg) == (10, pipeline.meds.width)
    rows = table(msg)
    assert len(rows) == 10
    real = collections.defaultdict(set)
    for sbj, code in pipeline.meds.select("subject_id", "code").iter_rows():
        real[sbj].add(code)
    for r in rows:
        assert any(displayed_as(r["code"], c) for c in real[r["subject_id"]])


def test_summarize_meds_like_example_subject_is_nearest_to_25_rows(
    pipeline, logger, capsys
):
    out = captured(
        capsys, logger.summarize_meds_like, pipeline.meds.lazy(), pipeline.splits
    )
    msg = message(out, "example subject")
    sbj = re.match(r"example subject \((\S+)\):", msg).group(1)
    counts = collections.Counter(pipeline.meds["subject_id"].to_list())
    assert sbj in counts
    assert abs(counts[sbj] - 25) == min(abs(n - 25) for n in counts.values())
    assert shape(msg) == (counts[sbj], pipeline.meds.width)
    times = [r["time"] for r in table(msg)]
    assert len(times) == counts[sbj]
    assert times == sorted(times)  # the example is sorted by time


def test_summarize_meds_like_split_tables_match_counted_splits(
    pipeline, logger, capsys
):
    out = captured(
        capsys, logger.summarize_meds_like, pipeline.meds.lazy(), pipeline.splits
    )
    assigned = split_of(pipeline.splits)
    by_subject = collections.Counter(assigned.values())
    assert len(by_subject) == 3

    rows = table(message(out, "subjects by split:"))
    assert [r["split"] for r in rows] == ["train", "tuning", "held_out"]
    assert {r["split"]: int(r["count"]) for r in rows} == dict(by_subject)
    for r in rows:
        assert float(r["rate"]) == pytest.approx(
            round(by_subject[r["split"]] / sum(by_subject.values()), 4), abs=1e-9
        )

    by_row = collections.Counter(
        assigned[s] for s in pipeline.meds["subject_id"].to_list()
    )
    assert sum(by_row.values()) == pipeline.meds.height
    rows = table(message(out, "rows by split:"))
    assert [r["split"] for r in rows] == ["train", "tuning", "held_out"]
    assert {r["split"]: int(r["count"]) for r in rows} == dict(by_row)


def test_summarize_meds_like_handles_a_single_subject_frame(pipeline, logger, capsys):
    sbj = sorted(set(pipeline.meds["subject_id"].to_list()))[0]
    meds = pipeline.meds.filter(pl.col("subject_id") == sbj)
    splits = pipeline.splits.filter(pl.col("subject_id") == sbj)
    assert meds.height > 0 and splits.height == 1
    out = captured(capsys, logger.summarize_meds_like, meds.lazy(), splits)
    assert message(out, "total rows:") == f"total rows: {meds.height}"
    assert message(out, "unique subjects:") == "unique subjects: 1"
    assert message(out, "example subject").startswith(f"example subject ({sbj}):")
    assert table(message(out, "subjects by split:")) == [
        {"split": splits["split"].item(), "count": "1", "rate": "1.0"}
    ]
    assert table(message(out, "rows by split:")) == [
        {"split": splits["split"].item(), "count": str(meds.height)}
    ]


def test_summarize_meds_like_leaves_a_scanned_frame_untouched(pipeline, logger, capsys):
    scanned = pl.scan_parquet(pipeline.path / "meds.parquet")
    before, splits_before = scanned.collect(), pipeline.splits.clone()
    out = captured(capsys, logger.summarize_meds_like, scanned, pipeline.splits)
    assert message(out, "total rows:") == f"total rows: {before.height}"
    assert scanned.collect().equals(before)
    assert pipeline.splits.equals(splits_before)


# ------------------------------------------------------------ summarize_tokens_times


def test_summarize_tokens_times_reports_one_row_per_timeline(
    pipeline, logger, lookup, capsys
):
    tokens_times = pipeline.tokens_times
    out = captured(
        capsys,
        logger.summarize_tokens_times,
        tokens_times.lazy(),
        pipeline.splits,
        lookup,
    )
    assert tokens_times.height == len(pipeline.manifest.subject_ids)
    assert message(out, "total rows:") == f"total rows: {tokens_times.height}"

    lengths = [len(t) for t in tokens_times["tokens"].to_list()]
    stats = {
        r["statistic"]: r["lengths"]
        for r in table(message(out, "timeline length stats:"))
    }
    assert float(stats["count"]) == len(lengths)
    assert float(stats["null_count"]) == 0
    assert float(stats["min"]) == min(lengths)
    assert float(stats["max"]) == max(lengths)
    assert float(stats["mean"]) == pytest.approx(sum(lengths) / len(lengths), abs=1e-4)


def test_summarize_tokens_times_duration_stats_span_the_planted_stays(
    pipeline, logger, lookup, capsys
):
    out = captured(
        capsys,
        logger.summarize_tokens_times,
        pipeline.tokens_times.lazy(),
        pipeline.splits,
        lookup,
    )
    stats = {
        r["statistic"]: r["duration"]
        for r in table(message(out, "timeline duration stats:"))
    }
    assert int(stats["count"]) == pipeline.tokens_times.height
    # a timeline runs from admission to discharge
    assert stats["min"] == str(datetime.timedelta(hours=min(synth.LOS_HOURS)))
    assert stats["max"] == str(datetime.timedelta(hours=max(synth.LOS_HOURS)))


def test_summarize_tokens_times_split_table_follows_split_order(
    pipeline, logger, lookup, capsys
):
    out = captured(
        capsys,
        logger.summarize_tokens_times,
        pipeline.tokens_times.lazy(),
        pipeline.splits,
        lookup,
    )
    rows = table(message(out, "split-level info:"))
    assert [r["split"] for r in rows] == ["train", "tuning", "held_out"]
    assert [r["split"] for r in rows] != sorted(r["split"] for r in rows)

    assigned = split_of(pipeline.splits)
    lengths = collections.defaultdict(list)
    starts = collections.defaultdict(list)
    for sbj, tokens, times in pipeline.tokens_times.select(
        "subject_id", "tokens", "times"
    ).iter_rows():
        lengths[assigned[sbj]].append(len(tokens))
        starts[assigned[sbj]].append(min(times))
    for r in rows:
        seen = lengths[r["split"]]
        assert len(seen) > 0
        assert float(r["avg_len"]) == pytest.approx(sum(seen) / len(seen), rel=1e-6)
        assert float(r["median_len"]) == pytest.approx(statistics.median(seen))
        assert datetime.datetime.strptime(
            r["first_event"][:19], "%Y-%m-%d %H:%M:%S"
        ) == min(starts[r["split"]]).replace(tzinfo=None)
    assert (
        datetime.datetime.strptime(rows[0]["first_event"][:19], "%Y-%m-%d %H:%M:%S")
        == synth.BASE
    )


def test_summarize_tokens_times_example_timelines_are_decoded_in_order(
    pipeline, logger, lookup, capsys
):
    out = captured(
        capsys,
        logger.summarize_tokens_times,
        pipeline.tokens_times.lazy(),
        pipeline.splits,
        lookup,
    )
    msgs = [m for m in records(out) if m.startswith("example timeline")]
    assert len(msgs) == 3
    lengths = {
        sbj: len(tokens)
        for sbj, tokens in pipeline.tokens_times.select(
            "subject_id", "tokens"
        ).iter_rows()
    }
    cutoff = sorted(abs(n - 25) for n in lengths.values())[2]
    for msg in msgs:
        sbj = re.match(r"example timeline \((\S+)\):", msg).group(1)
        assert abs(lengths[sbj] - 25) <= cutoff
        assert shape(msg) == (lengths[sbj], 4)  # subject_id, tokens, times, word
        words = [r["to_tokenize"] for r in table(msg)]
        expected = pipeline.timeline(sbj)
        assert len(words) == len(expected) == lengths[sbj]
        assert all(displayed_as(w, e) for w, e in zip(words, expected))


def test_summarize_tokens_times_handles_a_single_subject_frame(
    pipeline, logger, lookup, capsys
):
    sbj = sorted(set(pipeline.tokens_times["subject_id"].to_list()))[0]
    tokens_times = pipeline.tokens_times.filter(pl.col("subject_id") == sbj)
    splits = pipeline.splits.filter(pl.col("subject_id") == sbj)
    assert tokens_times.height == 1
    out = captured(
        capsys, logger.summarize_tokens_times, tokens_times.lazy(), splits, lookup
    )
    assert message(out, "total rows:") == "total rows: 1"
    length = len(tokens_times["tokens"].item())
    rows = table(message(out, "split-level info:"))
    assert len(rows) == 1
    assert rows[0]["split"] == splits["split"].item()
    assert float(rows[0]["avg_len"]) == float(rows[0]["median_len"]) == length
    msgs = [m for m in records(out) if m.startswith("example timeline")]
    assert len(msgs) == 1
    assert shape(msgs[0]) == (length, 4)


def test_summarize_tokens_times_leaves_a_scanned_frame_untouched(
    pipeline, logger, lookup, capsys
):
    scanned = pl.scan_parquet(pipeline.path / "tokens_times.parquet")
    before, splits_before = scanned.collect(), pipeline.splits.clone()
    lookup_before = lookup.clone()
    out = captured(
        capsys, logger.summarize_tokens_times, scanned, pipeline.splits, lookup
    )
    assert message(out, "total rows:") == f"total rows: {before.height}"
    assert scanned.collect().equals(before)
    assert pipeline.splits.equals(splits_before)
    assert lookup.equals(lookup_before)


# ------------------------------------------------------------ summarize_thresholded


def test_summarize_thresholded_rates_match_the_column_means(
    pipeline, logger, outcomes, capsys
):
    df = pipeline.inference("held_out")
    assert df.height > 1 and len(outcomes) == 15
    out = captured(capsys, logger.summarize_thresholded, df.lazy(), outcomes)
    assert len(records(out)) == 1
    rows = table(records(out)[0])
    assert [resolve(r["token"], outcomes) for r in rows] == outcomes
    reported = []
    for r in rows:
        token = resolve(r["token"], outcomes)
        for tense in ("past", "future"):
            values = df[f"{token}_{tense}"]
            assert values.null_count() == 0
            reported.append(float(r[tense]))
            assert float(r[tense]) == pytest.approx(
                sum(values.to_list()) / values.len(), abs=1e-5
            )
    assert max(reported) > 0 and min(reported) == 0  # rates actually vary


def test_summarize_thresholded_on_one_row_reports_zeros_and_ones(
    pipeline, logger, outcomes, capsys
):
    df = pipeline.inference("held_out").head(1)
    assert df.height == 1
    out = captured(capsys, logger.summarize_thresholded, df.lazy(), outcomes)
    rows = table(records(out)[0])
    assert len(rows) == len(outcomes)
    for r in rows:
        token = resolve(r["token"], outcomes)
        for tense in ("past", "future"):
            flag = df[f"{token}_{tense}"].item()
            assert float(r[tense]) == (1.0 if flag else 0.0)


def test_summarize_thresholded_on_an_empty_frame_reports_null_rates(
    pipeline, logger, outcomes, capsys
):
    df = pipeline.inference("held_out").head(0)
    out = captured(capsys, logger.summarize_thresholded, df.lazy(), outcomes[:3])
    rows = table(records(out)[0])
    assert len(rows) == 3
    assert all(r["past"] == "null" and r["future"] == "null" for r in rows)


def test_summarize_thresholded_without_outcome_tokens_raises(pipeline, logger):
    """
    BUG: outcome_tokens matching nothing leaves the transpose without rows,
    so the verbose winnower crashes even though the parquet is written fine
    """
    with pytest.raises(pl.exceptions.ShapeError):
        logger.summarize_thresholded(pipeline.inference("held_out").lazy(), [])


@pytest.mark.parametrize("summary", ["meds_like", "tokens_times", "thresholded"])
def test_summaries_need_a_lazy_frame(pipeline, logger, lookup, outcomes, summary):
    """the summaries collect internally, so an eager frame is not accepted"""
    calls = {
        "meds_like": (logger.summarize_meds_like, (pipeline.meds, pipeline.splits)),
        "tokens_times": (
            logger.summarize_tokens_times,
            (pipeline.tokens_times, pipeline.splits, lookup),
        ),
        "thresholded": (
            logger.summarize_thresholded,
            (pipeline.inference("held_out"), outcomes),
        ),
    }
    fn, args = calls[summary]
    with pytest.raises(AttributeError):
        fn(*args)


# ------------------------------------------------------------------ verbose stages


def test_collator_save_all_verbose_reports_the_collated_frame(runner, capsys):
    dest = runner.dir()
    capsys.readouterr()
    runner.collate(dest=dest, verbose=True)
    out = capsys.readouterr().out
    height = pl.read_parquet(dest / "meds.parquet").height
    assert message(out, "total rows:") == f"total rows: {height}"
    for label in ("unique subjects:", "by category:", "example rows:"):
        assert message(out, label)
    assert len(table(message(out, "subjects by split:"))) == 3
    assert len(table(message(out, "rows by split:"))) == 3


def test_collator_save_all_quietly_skips_the_summary(runner, capsys):
    capsys.readouterr()
    runner.collate()
    out = capsys.readouterr().out
    assert any(m.startswith("Collator initialized") for m in records(out))
    assert not [m for m in records(out) if m.startswith("total rows:")]


def test_tokenizer_save_all_verbose_reports_the_timelines(runner, capsys):
    dest = runner.seed_collated()
    capsys.readouterr()
    runner.tokenize(processed=dest, verbose=True)
    out = capsys.readouterr().out
    height = pl.read_parquet(dest / "tokens_times.parquet").height
    assert message(out, "total rows:") == f"total rows: {height}"
    for label in ("timeline length stats:", "timeline duration stats:"):
        assert message(out, label)
    assert len(table(message(out, "split-level info:"))) == 3
    assert len([m for m in records(out) if m.startswith("example timeline")]) == 3


def test_tokenizer_save_all_quietly_skips_the_summary(runner, capsys):
    dest = runner.seed_collated()
    capsys.readouterr()
    runner.tokenize(processed=dest)
    out = capsys.readouterr().out
    assert any(m.startswith("Loading collated data") for m in records(out))
    assert not [m for m in records(out) if m.startswith("total rows:")]


def test_winnower_save_all_verbose_reports_every_split(
    runner, pipeline, outcomes, capsys
):
    dest = seed_winnower(runner, pipeline)
    capsys.readouterr()
    runner.winnow(processed=dest, verbose=True)
    msgs = records(capsys.readouterr().out)
    headers = [m for m in msgs if m.startswith("Prepared split")]
    assert [
        re.match(r"Prepared split (\S+) for inference:", m).group(1) for m in headers
    ] == ["train", "tuning", "held_out"]
    for split in ("train", "tuning", "held_out"):
        header = f"Prepared split {split} for inference:"
        rows = table(msgs[msgs.index(header) + 1])
        assert len(rows) == len(outcomes)
        df = pl.read_parquet(dest / f"{split}_for_inference.parquet")
        assert df.height > 0
        token = resolve(rows[0]["token"], outcomes)
        values = df[f"{token}_past"].to_list()
        assert float(rows[0]["past"]) == pytest.approx(
            sum(values) / len(values), abs=1e-5
        )


def test_winnower_save_all_quietly_skips_the_summary(runner, pipeline, capsys):
    dest = seed_winnower(runner, pipeline)
    capsys.readouterr()
    runner.winnow(processed=dest)
    out = capsys.readouterr().out
    assert any(m.startswith("Winnower initialized") for m in records(out))
    assert not [m for m in records(out) if m.startswith("Prepared split")]


def test_winnower_save_all_verbose_raises_when_no_outcome_token_matches(
    runner, pipeline, capsys
):
    """
    BUG: the same empty-transpose crash, reached through the shipped stage --
    the frame is written before the summary is attempted, so only --verbose fails
    """
    dest = seed_winnower(runner, pipeline)
    cfg = default_cfg("winnowing")
    cfg["outcome_tokens"] = ["NOSUCHPREFIX//*"]
    cfg["splits"] = ["held_out"]
    with pytest.raises(pl.exceptions.ShapeError):
        runner.winnow(cfg=cfg, processed=dest, verbose=True)
    assert pl.read_parquet(dest / "held_out_for_inference.parquet").height > 0
