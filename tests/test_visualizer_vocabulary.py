#!/usr/bin/env python3

"""
visualization: the vocabulary page, with each binned code's training values
against its bins, every word, the prefixes, and the timelines by split
"""

import datetime
import fnmatch
import json
import math
import pathlib
import re
import tomllib

import polars as pl
import pytest
from conftest import as_written_by_26_6, copy_of, default_cfg, write_cfg
from omegaconf import OmegaConf

from cocoa.visualizer import EPS, EXACT, Visualizer, fmt_setting, round_width

DATA = re.compile(
    r'<script id="cocoa-data" type="application/json">(.*?)</script>', re.S
)
REPO = pathlib.Path(__file__).resolve().parents[1]


def vocabulary_of(path, **kwargs) -> dict:
    """the vocabulary page's payload for a processed directory"""
    return Visualizer(processed_data_home=path, **kwargs).get_vocabulary_payload()


def binned_of(payload: dict) -> dict:
    """the payload's binned codes, by code"""
    return {payload["codes"][b["code"]]["code"]: b for b in payload["binned"]}


def training_values(processed) -> pl.DataFrame:
    """the finite numeric values of the training split's events, by code"""
    train = processed.splits.filter(pl.col("split") == "train")["subject_id"]
    return processed.meds.filter(
        pl.col("subject_id").is_in(train.to_list()), pl.col("numeric_value").is_finite()
    ).select("code", value=pl.col("numeric_value").cast(pl.Float64))


def test_the_page_embeds_its_payload_and_loads_nothing_from_outside(pipeline):
    page = Visualizer(processed_data_home=pipeline.path).render_vocabulary()
    assert "__COCOA_" not in page
    (data,) = DATA.findall(page)
    assert json.loads(data)["meta"]["vocab_size"] == len(pipeline.vocab)
    title = re.search(r"<title>(.*?)</title>", page).group(1)
    assert title == f"Vocabulary · {pipeline.path.name} · cocoa"
    assert "default-src 'none'" in page  # a browser refuses whatever isn't inlined
    refs = re.findall(r"""(?:\s(?:src|href)=|url\()["']?([^"')\s>]+)""", page)
    assert refs and all(r.startswith("data:") for r in refs)
    assert ".hists" in page and "--accent" in page  # its own styles, and the shared


def test_the_page_says_how_to_open_it_where_its_script_does_not_run(pipeline):
    page = Visualizer(processed_data_home=pipeline.path).render_vocabulary()
    shown = re.sub(r"<script\b.*?</script>", "", page, flags=re.S)
    (note,) = re.findall(r'<div id="cocoa-fallback".*?</div>', shown, flags=re.S)
    assert "vocabulary" in note and "web browser" in note
    assert 'getElementById("cocoa-fallback")' in page  # the script clears it


def test_every_page_asset_ships_as_package_data():
    """without a glob for it, a wheel would leave a page's template or script out"""
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())
    globs = pyproject["tool"]["setuptools"]["package-data"]["cocoa.assets"]
    assets = REPO / "src" / "cocoa" / "assets"
    files = {
        f.relative_to(assets).as_posix()
        for f in assets.rglob("*")
        if f.is_file()
        and "__pycache__" not in f.parts
        and not f.name.startswith(".")
        and f.suffix != ".py"
    }
    assert {"vocabulary.html", "vocabulary.js", "vocabulary.css"} <= files
    for f in files:
        assert any(fnmatch.fnmatch(f, g) for g in globs), f


def test_a_rerun_renders_the_same_page(pipeline):
    """but for when it was rendered: sums are taken in one order, whatever order
    the rows come in"""
    v = Visualizer(processed_data_home=pipeline.path)
    pages = {
        re.sub(r'"generated":"[^"]*"', "", v.render_vocabulary()) for _ in range(5)
    }
    assert len(pages) == 1


def test_save_vocabulary_writes_the_page(pipeline, tmp_path):
    v = Visualizer(processed_data_home=pipeline.path)
    out = v.save_vocabulary(tmp_path / "nested" / "vocabulary.html")
    assert out == (tmp_path / "nested" / "vocabulary.html").resolve()
    assert DATA.search(out.read_text(encoding="utf-8"))


def test_save_vocabulary_pdf_writes_the_pdf(pipeline, tmp_path):
    v = Visualizer(processed_data_home=pipeline.path)
    out = v.save_vocabulary_pdf(tmp_path / "nested" / "vocabulary.pdf")
    assert out == (tmp_path / "nested" / "vocabulary.pdf").resolve()
    data = out.read_bytes()
    assert data.startswith(b"%PDF-") and data.endswith(b"%%EOF\n")


def test_every_word_is_listed_once_in_token_order_with_its_training_count(pipeline):
    w = vocabulary_of(pipeline.path)["words"]
    assert w["token"] == sorted(pipeline.decoder)
    assert dict(zip(w["word"], w["token"])) == pipeline.vocab
    assert dict(zip(w["word"], w["count"])) == pipeline.counts  # UNK's is null


def test_fused_words_name_their_code_bin_and_kind(pipeline):
    p = vocabulary_of(pipeline.path)
    w = p["words"]
    rows = {
        word: (kind, p["codes"][c]["code"], b, t)
        for word, kind, c, b, t in zip(
            w["word"], w["kind"], w["code"], w["bin"], w["text"]
        )
    }
    sbp = [x for x in rows if re.fullmatch(r"VTL//sbp_Q\d+", x)]
    assert sbp and all(rows[x] == ("event", "VTL//sbp", int(x[10:]), None) for x in sbp)
    for word, kind in (("BOS", "bos"), ("EOS", "eos"), ("UNK", "unk")):
        assert rows[word] == (kind, word, None, None)
    assert {k for k in w["kind"]} <= {"event", "bos", "eos", "unk", "spacer", "clock"}


def test_unfused_bins_and_text_values_are_words_that_codes_share(runner):
    dest = runner.seed_collated()
    runner.tokenize(cfg={**default_cfg("tokenization"), "fused": False}, processed=dest)
    p = vocabulary_of(dest)
    w = p["words"]
    shared = [
        (word, kind, b, t)
        for word, kind, c, b, t in zip(
            w["word"], w["kind"], w["code"], w["bin"], w["text"]
        )
        if c is None
    ]
    assert {kind for _, kind, _, _ in shared} == {"bin", "text"}
    assert all(word == f"Q{b}" for word, kind, b, _ in shared if kind == "bin")
    assert all(word == t for word, kind, _, t in shared if kind == "text")
    (values,) = [x for x in p["prefixes"] if x["prefix"] == ""]
    assert values["words"] == len(shared)
    assert p["prefixes"][-1] is values  # after every prefix of a code
    # the values of each code are still binned and drawn code by code
    assert p["meta"]["bins_source"] == "meds"
    for b in p["binned"]:
        assert b["n"] > 0 and sum(b["bins"]) == b["n"]


def test_histograms_count_each_binned_code_s_training_values(pipeline):
    p = vocabulary_of(pipeline.path)
    binned = binned_of(p)
    assert list(binned) == list(pipeline.tokenizer_yaml.bins)  # each, as listed
    values = training_values(pipeline)
    expect = values.group_by("code").agg(
        n=pl.len(),
        distinct=pl.col("value").n_unique(),
        min=pl.col("value").min(),
        max=pl.col("value").max(),
    )
    assert expect.height == len(binned)
    for code, n, distinct, lo, hi in expect.iter_rows():
        b = binned[code]
        assert (b["n"], b["distinct"], b["min"], b["max"]) == (n, distinct, lo, hi)
        h = b["hist"]
        assert sum(h["n"]) + h.get("under", 0) + h.get("over", 0) == n, code
    # as the tokenizer learns its bins, values of other splits are left out
    every = pipeline.meds.filter(pl.col("numeric_value").is_finite())
    counts = dict(every.group_by("code").len().iter_rows())
    assert any(b["n"] < counts[code] for code, b in binned.items())


def test_values_fall_in_the_bins_the_tokenizer_puts_them_in(pipeline):
    """under the shipped fused tokenizer, each bin's words count its values"""
    p = vocabulary_of(pipeline.path)
    assert p["meta"]["bins_source"] == "meds"
    for code, b in binned_of(p).items():
        assert len(b["bins"]) == len(b["breaks"]) + 1
        in_words = [
            sum(
                n
                for w, n in pipeline.counts.items()
                if re.fullmatch(re.escape(code) + rf"_Q{k}(_.*)?", w)
            )
            for k in range(len(b["bins"]))
        ]
        assert b["bins"] == in_words, code


def test_a_code_with_few_distinct_values_is_counted_value_by_value(pipeline):
    values = training_values(pipeline)
    exact = {
        c: b
        for c, b in binned_of(vocabulary_of(pipeline.path)).items()
        if "x" in b["hist"]
    }
    assert exact and all(b["distinct"] <= EXACT for b in exact.values())
    for code, b in exact.items():
        counts = (
            values.filter(pl.col("code") == code).group_by("value").len().sort("value")
        )
        assert b["hist"] == {
            "x": counts["value"].to_list(),
            "n": counts["len"].to_list(),
        }


def test_buckets_are_a_round_width_and_span_the_cut_points(pipeline):
    values = training_values(pipeline)
    binned = binned_of(vocabulary_of(pipeline.path))
    bucketed = {c: b for c, b in binned.items() if "w" in b["hist"]}
    assert bucketed and all(b["distinct"] > EXACT for b in bucketed.values())
    for code, b in bucketed.items():
        h = b["hist"]
        w, lo = h["w"], h["lo"]
        assert round(w / 10 ** math.floor(math.log10(w)), 9) in (1, 2, 5)
        assert lo / w == pytest.approx(round(lo / w))  # on the grid of its width
        hi = lo + w * len(h["n"])
        assert all(lo <= x <= hi for x in b["breaks"])
        v = values.filter(pl.col("code") == code)["value"]
        assert h["under"] == (v < lo - EPS * w).sum()
        assert h["over"] == (v >= hi - EPS * w).sum()
        assert h["under"] + h["over"] <= 0.01 * b["n"] + 2  # all but the tails


def readings_of(runner, readings: list) -> dict:
    """the binned heart rate of a stay with `readings`, an hour apart"""
    start = datetime.datetime(2024, 1, 1)
    processed = runner.minimal(
        hospitalizations=[
            {
                "admission_dttm": start,
                "discharge_dttm": start + datetime.timedelta(hours=len(readings)),
            }
        ],
        vitals=[
            {"recorded_dttm": start + datetime.timedelta(hours=k), "vital_value": x}
            for k, x in enumerate(readings)
        ],
    )
    assert processed.meds.schema["numeric_value"] == pl.Float32
    (b,) = vocabulary_of(processed.path)["binned"]
    return b


def test_far_outliers_are_counted_beyond_the_range_drawn(runner):
    """rather than squeezing the rest of the values into a bucket or two"""
    b = readings_of(runner, [-9999.0, *(60.0 + k for k in range(200)), 9999.0])
    h = b["hist"]
    assert (h["under"], h["over"], b["min"], b["max"]) == (1, 1, -9999.0, 9999.0)
    assert 50 < h["lo"] and h["lo"] + h["w"] * len(h["n"]) < 300  # around the rest


def test_values_that_nearly_all_agree_are_drawn_across_all_of_them(runner):
    """their tails' quantiles are the one value, so the range runs to the extremes"""
    tails = [1 + k / 100 for k in range(30)], [9 + k / 100 for k in range(30)]
    b = readings_of(runner, [*tails[0], *[5.0] * 6000, *tails[1]])
    h = b["hist"]
    assert b["distinct"] > EXACT and set(b["breaks"]) == {5.0}
    assert (h["under"], h["over"]) == (0, 0)
    assert h["lo"] <= 1 and h["lo"] + h["w"] * len(h["n"]) > 9.29


def test_a_decimal_read_back_from_float32_lands_in_its_own_bucket(runner):
    """1.4 is 1.39999998 as a float32, which floors to the bucket below 1.4's"""
    h = readings_of(runner, [round(1 + k / 10, 1) for k in range(70)])["hist"]
    assert (h["lo"], h["w"]) == (1.0, 0.2)
    assert h["n"] == [2] * 35  # 1.0 and 1.1, 1.2 and 1.3, and so on to 7.9


def test_a_binned_code_without_training_values_here_counts_none(pipeline, tmp_path):
    """as a code that a tokenizer learned elsewhere may have"""
    dest = copy_of(pipeline.path, tmp_path, without=("meds.parquet",))
    pipeline.meds.filter(pl.col("code") != "VTL//sbp").write_parquet(
        dest / "meds.parquet"
    )
    b = binned_of(vocabulary_of(dest))["VTL//sbp"]
    assert b["n"] == 0 and "hist" not in b
    assert b["bins"] == [0] * (len(b["breaks"]) + 1)


def test_without_meds_parquet_a_fused_vocabulary_counts_bins_by_their_words(
    pipeline, tmp_path
):
    dest = copy_of(pipeline.path, tmp_path, without=("meds.parquet",))
    p = vocabulary_of(dest)
    assert (p["meta"]["value_source"], p["meta"]["bins_source"]) == (None, "vocabulary")
    drawn = binned_of(vocabulary_of(pipeline.path))
    for code, b in binned_of(p).items():
        assert b["n"] is None and "hist" not in b
        assert b["bins"] == drawn[code]["bins"]  # its words count what meds would
    assert DATA.search(Visualizer(processed_data_home=dest).render_vocabulary())


def test_without_meds_parquet_an_unfused_vocabulary_has_only_its_cut_points(
    runner, pipeline
):
    dest = runner.seed_collated()
    runner.tokenize(cfg={**default_cfg("tokenization"), "fused": False}, processed=dest)
    (dest / "meds.parquet").unlink()
    p = vocabulary_of(dest)
    assert (p["meta"]["value_source"], p["meta"]["bins_source"]) == (None, None)
    for code, b in binned_of(p).items():
        assert b.keys() == {"code", "breaks", "n"}
        assert b["breaks"] == list(pipeline.tokenizer_yaml.bins[code])


def test_without_subject_splits_the_timelines_are_one_group(pipeline, tmp_path):
    dest = copy_of(pipeline.path, tmp_path, without=("subject_splits.parquet",))
    p = vocabulary_of(dest)
    t = p["timelines"]
    assert [g["split"] for g in t["splits"]] == [None]
    assert t["splits"][0]["timelines"] == t["all"]["timelines"]
    assert t["all"]["timelines"] == pipeline.tokens_times.height
    # nor can the training split's values be told from the rest
    assert (p["meta"]["value_source"], p["meta"]["bins_source"]) == (None, "vocabulary")


def test_a_tokenizer_yaml_alone_draws_its_words_and_cut_points(pipeline, tmp_path):
    """as one copied beside a model: no values to draw and no timelines to sum"""
    dest = copy_of(
        pipeline.path,
        tmp_path,
        without={f.name for f in pipeline.path.iterdir()} - {"tokenizer.yaml"},
    )
    assert [f.name for f in dest.iterdir()] == ["tokenizer.yaml"]
    p = vocabulary_of(dest)
    assert p["timelines"] is None
    assert p["meta"]["bins_source"] == "vocabulary"  # the fused words still count
    assert dict(zip(p["words"]["word"], p["words"]["token"])) == pipeline.vocab
    assert DATA.search(Visualizer(processed_data_home=dest).render_vocabulary())


def test_timelines_are_summarized_by_split_as_tokenize_verbose_does(pipeline):
    t = vocabulary_of(pipeline.path)["timelines"]
    assert [g["split"] for g in t["splits"]] == ["train", "tuning", "held_out"]
    tt = pipeline.tokens_times.join(
        pipeline.splits.select("subject_id", "split"), on="subject_id"
    )
    for g in [*t["splits"], t["all"]]:
        rows = tt if g is t["all"] else tt.filter(pl.col("split") == g["split"])
        lengths = rows["tokens"].list.len()
        assert g["timelines"] == rows.height
        assert g["tokens"] == lengths.sum()
        assert g["unk"] == rows["tokens"].list.count_matches(0).sum()
        assert g["length"]["min"] == lengths.min()
        assert g["length"]["median"] == lengths.median()
        assert g["length"]["max"] == lengths.max()
        first = rows["times"].list.min().min()
        assert g["first"]["ms"] == first.timestamp() * 1000
        assert sum(g["lengths"]) == rows.height
        assert sum(g["durations"]) == rows.height - g["instant"]
    unk = {g["split"]: g["unk"] for g in t["splits"]}
    assert unk["train"] == 0 < unk["tuning"] + unk["held_out"]  # learned on training


def test_lengths_are_counted_per_decade_from_the_shortest_to_the_longest(pipeline):
    t = vocabulary_of(pipeline.path)["timelines"]
    k, counts = t["length_from"], t["all"]["lengths"]
    lengths = pipeline.tokens_times["tokens"].list.len()
    edge = lambda j: 10 ** (j / t["per_decade"])  # noqa: E731
    assert edge(k) <= lengths.min() < edge(k + 1)
    assert edge(k + len(counts) - 1) <= lengths.max() < edge(k + len(counts))
    assert counts[0] > 0 and counts[-1] > 0
    for j, n in enumerate(counts):
        within = ((lengths >= edge(k + j)) & (lengths < edge(k + j + 1))).sum()
        assert n == within


def test_prefixes_add_up_to_the_vocabulary(pipeline):
    p = vocabulary_of(pipeline.path)
    ps = p["prefixes"]
    assert sum(x["words"] for x in ps) == len(pipeline.vocab)
    trained = [n for n in pipeline.counts.values() if n is not None]
    assert sum(x["count"] for x in ps) == sum(trained)
    assert sum(x["binned"] for x in ps) == len(p["binned"])
    ordering = list(pipeline.tokenizer_yaml.cfg.ordering)
    listed = [x["prefix"] for x in ps if x["prefix"] in ordering]
    assert listed == sorted(listed, key=ordering.index)  # in timeline order
    assert ps[-1]["prefix"] == "UNK"


def test_codes_wear_the_colors_their_lanes_wear_in_a_timeline(pipeline, tmp_path):
    cfg = {
        "lanes": [{"name": "Vitals", "prefixes": ["VTL"], "color": "#A4343A"}],
        "palette": [f"#0000{i:02X}" for i in range(1, 33)],
        "prefixes": {"LAB-RES": "Lab result"},
    }
    v = Visualizer(
        write_cfg(tmp_path / "v.yaml", cfg), processed_data_home=pipeline.path
    )
    vocab = v.get_vocabulary_payload()
    lane = {
        c["prefix"]: vocab["lanes"][c["lane"]]
        for c in vocab["codes"]
        if c["lane"] is not None
    }
    assert lane["VTL"]["name"] == "Vitals" and lane["LAB-RES"]["name"] == "Lab result"
    for sid in pipeline.tokens_times["subject_id"].head(5):
        timeline = v.get_payload(sid)
        for c in timeline["codes"]:
            if c["lane"] is not None:
                assert timeline["lanes"][c["lane"]] == lane[c["prefix"]]


def test_a_transferred_vocabulary_says_so_and_keeps_its_training_counts(
    runner, pipeline
):
    dest = runner.seed_collated()
    runner.tokenize(processed=dest, tokenizer_home=pipeline.path / "tokenizer.yaml")
    p = vocabulary_of(dest)
    assert vocabulary_of(pipeline.path)["meta"]["is_training"] is True
    assert p["meta"]["is_training"] is False
    assert dict(zip(p["words"]["word"], p["words"]["count"])) == pipeline.counts


def test_a_vocabulary_from_26_6_draws_without_training_counts(pipeline, tmp_path):
    """before 26.9.1, tokenizer.yaml had no counts; before 26.9.0, times were naive"""
    legacy = as_written_by_26_6(pipeline.path, tmp_path)
    p, now = vocabulary_of(legacy), vocabulary_of(pipeline.path)
    assert p["meta"]["has_counts"] is False and now["meta"]["has_counts"] is True
    assert set(p["words"]["count"]) == {None}
    assert {x["count"] for x in p["prefixes"]} == {None}
    assert p["timelines"]["timezone"] == "UTC"
    assert p["timelines"] == now["timelines"]  # naive times meant utc
    assert p["binned"] == now["binned"]
    assert DATA.search(Visualizer(processed_data_home=legacy).render_vocabulary())


def test_the_tokenizer_config_is_listed_as_saved(pipeline):
    p = vocabulary_of(pipeline.path)
    cfg = OmegaConf.to_container(pipeline.tokenizer_yaml.cfg)
    assert [k for k, _ in p["config"]] == list(cfg)
    shown = dict(p["config"])
    assert (shown["fused"], shown["n_bins"]) == ("true", "10")
    assert shown["ordering"] == ", ".join(cfg["ordering"])
    assert shown["spacers"].startswith("5m-15m: 5, 15m-1h: 15, ")
    assert shown["clocks"] == "00, 04, 08, 12, 16, 20"


@pytest.mark.parametrize(
    "raw, whole, width",
    [
        (0.138, False, 0.2),
        (0.3, False, 0.5),
        (1.0, False, 1.0),
        (7.0, False, 10.0),
        (1000.0, False, 1000.0),
        (3e-7, False, 5e-7),
        (0.4, True, 1.0),  # a whole number's bucket holds a whole number of them
        (2.2, True, 5.0),
    ],
)
def test_buckets_take_the_least_round_width_at_least_as_wide(raw, whole, width):
    assert round_width(raw, whole=whole) == pytest.approx(width)


@pytest.mark.parametrize(
    "value, shown",
    [
        (True, "true"),
        (False, "false"),
        (None, "null"),
        (10, "10"),
        (0.5, "0.5"),
        ("00", "00"),
        (["BOS", "TIME"], "BOS, TIME"),
        ({"5m-15m": 5, "6mt+": 262980}, "5m-15m: 5, 6mt+: 262980"),
    ],
)
def test_config_values_show_as_yaml_spells_them(value, shown):
    assert fmt_setting(value) == shown
