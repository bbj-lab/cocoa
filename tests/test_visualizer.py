#!/usr/bin/env python3

"""
visualization: rendering one subject's timeline as a self-contained html page
"""

import copy
import datetime
import importlib.resources as resources
import json
import pathlib
import re
import shutil
import socket
import threading
import zoneinfo

import polars as pl
import pytest
from conftest import default_cfg, write_cfg
from omegaconf import OmegaConf

from cocoa.visualizer import (
    SubjectNotFoundError,
    Visualizer,
    fmt_value,
    make_server,
    to_script_json,
)

DATA = re.compile(
    r'<script id="cocoa-data" type="application/json">(.*?)</script>', re.S
)

# the tests' own visualization config, so that none depends on which lanes,
# names, or descriptions the shipped one carries; its prefixes are ones the
# synthetic data collates to. one lane wears a palette color, and other_color
# differs from the visualizer's built-in fallback, so that each can be told apart
PALETTE = [f"#0000{i:02X}" for i in range(1, 33)]
VIZ_CFG = {
    "lanes": [
        {"name": "Time", "prefixes": ["TIME", "CLCK"], "color": "#A6A6A6"},
        {
            "name": "Encounter",
            "prefixes": ["ADMN", "XFR-OUT", "XFR-IN", "CODE", "DSCG"],
            "color": PALETTE[0],
        },
        {"name": "Vitals", "prefixes": ["VTL", "VIT"], "color": "#A4343A"},
    ],
    "palette": PALETTE,
    "other_color": "#565656",
    "prefixes": {"VTL": "Vital sign"},
    "descriptions": {
        "BOS": "Beginning of the timeline",
        "EOS": "End of the timeline",
        "VTL//sbp": "Systolic blood pressure",
    },
}


def viz_cfg(tmp_path, **overrides):
    """the tests' visualization config with keys replaced, as a file"""
    cfg = {**copy.deepcopy(VIZ_CFG), **overrides}
    return write_cfg(tmp_path / "visualization.yaml", cfg), cfg


@pytest.mark.parametrize(
    "value", ["plain", "a\u2028b\u2029c", "</script><!--", "x & y", "", "°C ≥ 6.5"]
)
def test_script_json_round_trips(value):
    s = to_script_json({"v": value})
    assert json.loads(s) == {"v": value}
    assert not re.search(r"[<>&\u2028\u2029]", s)


def test_script_json_does_not_grow_plain_text():
    obj = {"subject_id": "7434407", "tokens": list(range(50))}
    assert to_script_json(obj) == json.dumps(obj, separators=(",", ":"))


@pytest.mark.parametrize(
    "value, shown",
    [
        (None, None),
        (float("nan"), None),
        (True, "yes"),
        (False, "no"),
        (50.0, "50"),  # a whole float reads as an integer
        (26.823, "26.823"),
        (2 / 3, "0.666667"),  # to six significant digits
        (0.1 + 0.2, "0.3"),  # so without float noise
        (7, "7"),
        ("Inpatient", "Inpatient"),
        (
            datetime.datetime(
                2024, 7, 1, 8, 30, tzinfo=zoneinfo.ZoneInfo("America/Chicago")
            ),
            "2024-07-01 08:30:00 CDT",
        ),
        (datetime.datetime(2024, 1, 1, 8, 30, 0, 250000), "2024-01-01 08:30:00.25"),
    ],
)
def test_pass_through_values_show_as_text(value, shown):
    assert fmt_value(value) == shown


def test_rendered_page_embeds_parseable_payload(pipeline):
    sid = pipeline.tokens_times["subject_id"][0]
    page = Visualizer(processed_data_home=pipeline.path).render(sid)
    assert "__COCOA_" not in page
    (data,) = DATA.findall(page)
    payload = json.loads(data)
    assert payload["subject"]["subject_id"] == sid
    assert len(payload["tokens"]["id"]) == len(payload["tokens"]["event"])
    assert payload["tokens"]["id"] == pipeline.tokens_times["tokens"][0].to_list()


def test_page_loads_nothing_from_outside_itself(pipeline):
    sid = pipeline.tokens_times["subject_id"][0]
    page = Visualizer(processed_data_home=pipeline.path).render(sid)
    assert "default-src 'none'" in page  # a browser refuses whatever isn't inlined
    refs = re.findall(r"""(?:\s(?:src|href)=|url\()["']?([^"')\s>]+)""", page)
    assert refs and all(r.startswith("data:") for r in refs)


def test_page_says_how_to_open_it_where_its_script_does_not_run(pipeline):
    """email and file previews often show html without running its script"""
    sid = pipeline.tokens_times["subject_id"][0]
    page = Visualizer(processed_data_home=pipeline.path).render(sid)
    shown = re.sub(r"<script\b.*?</script>", "", page, flags=re.S)
    (note,) = re.findall(r'<div id="cocoa-fallback".*?</div>', shown, flags=re.S)
    assert "web browser" in note and "internet connection" in note
    assert 'getElementById("cocoa-fallback")' in page  # the script clears it


def payload_of(processed, n: int = 0) -> dict:
    """the page payload for the `n`th subject of a processed directory"""
    sid = processed.tokens_times["subject_id"][n]
    return Visualizer(processed_data_home=processed.path).get_payload(sid)


def test_parse_splits_prefix_bin_and_text(pipeline):
    v = Visualizer(processed_data_home=pipeline.path)
    assert v.parse("VTL//heart_rate_Q3") == {
        "prefix": "VTL",
        "code": "VTL//heart_rate",
        "name": "heart_rate",
        "bin": 3,
        "text": None,
    }
    assert v.parse("ASMT//cam_total_yes")["code"] == "ASMT//cam_total_yes"
    for word in ("BOS", "EOS", "UNK"):
        assert v.parse(word)["prefix"] == word
        assert v.parse(word)["code"] == word


def test_structural_words_get_their_descriptions(pipeline, tmp_path):
    path, _ = viz_cfg(tmp_path)
    v = Visualizer(path, processed_data_home=pipeline.path)
    sid = pipeline.tokens_times["subject_id"][0]
    codes = {c["code"]: c for c in v.get_payload(sid)["codes"]}
    assert {"BOS", "EOS"} <= codes.keys()
    assert codes["BOS"]["description"] == "Beginning of the timeline"
    assert codes["EOS"]["description"] == "End of the timeline"
    assert codes["BOS"]["lane"] is None


def test_every_token_belongs_to_one_event(pipeline):
    p = payload_of(pipeline)
    ev = p["events"]
    assert sum(ev["n_tokens"]) == len(p["tokens"]["id"])
    for e, (first, n) in enumerate(zip(ev["first_token"], ev["n_tokens"])):
        assert p["tokens"]["event"][first : first + n] == [e] * n


def test_unfused_bins_attach_to_the_code_before_them(runner):
    dest = runner.seed_collated()
    runner.tokenize(cfg={**default_cfg("tokenization"), "fused": False}, processed=dest)
    v = Visualizer(processed_data_home=dest)
    assert not v.fused
    n_binned = 0
    for sid in pl.read_parquet(dest / "tokens_times.parquet")["subject_id"]:
        p = v.get_payload(sid)
        # a bare bin is never an event of its own, even after an unknown code
        assert not [c["code"] for c in p["codes"] if re.fullmatch(r"Q\d+", c["code"])]
        n_binned += sum(b is not None for b in p["events"]["bin"])
        assert len(p["events"]["kind"]) <= len(p["tokens"]["id"])
    assert n_binned > 0


@pytest.mark.parametrize("fused", [True, False])
def test_a_bare_prefix_is_drawn_with_its_prefix(runner, tmp_path, fused):
    """before 26.6.0, collation wrote an event with a null code as its prefix"""
    dest = runner.seed_collated()
    meds = pl.read_parquet(dest / "meds.parquet")
    bare = meds.filter(pl.col("code").str.starts_with("VTL//")).with_columns(
        code=pl.lit("VTL")
    )
    pl.concat([meds, bare]).write_parquet(dest / "meds.parquet")
    runner.tokenize(cfg={**default_cfg("tokenization"), "fused": fused}, processed=dest)
    path, _ = viz_cfg(tmp_path)
    v = Visualizer(path, processed_data_home=dest)
    assert v.parse("VTL") == {
        "prefix": "VTL",
        "code": "VTL",
        "name": "",
        "bin": None,
        "text": None,
    }
    n_binned = 0
    for sid in bare["subject_id"].unique():
        p = v.get_payload(sid)
        codes = {c["code"]: c for c in p["codes"]}
        assert not [c for c in codes if codes[c]["prefix"] == ""]
        assert codes["VTL"]["prefix"] == "VTL"
        assert codes["VTL"]["lane"] == codes["VTL//sbp"]["lane"] is not None
        assert "VTL" not in p["events"]["text"]  # not glued onto the code before
        n_binned += sum(
            b is not None
            for c, b in zip(p["events"]["code"], p["events"]["bin"])
            if p["codes"][c]["code"] == "VTL"
        )
        assert v.render_pdf(sid).startswith(b"%PDF-")
    assert n_binned > 0


def test_a_tokenizer_cfg_without_fused_is_read_as_the_tokenizer_ran(runner):
    """the visualizer's fallback for `fused` agrees with the tokenizer's"""
    cfg = default_cfg("tokenization")
    del cfg["fused"]
    dest = runner.seed_collated()
    runner.tokenize(cfg=cfg, processed=dest)
    v = Visualizer(processed_data_home=dest)
    assert v.tkzr_cfg["fused"] is True
    assert v.fused
    assert any(re.search(r"_Q\d+$", w) for w in v.decoder.values())


def test_a_legacy_tokenizer_yaml_without_fused_is_read_as_unfused(runner):
    """a tokenizer.yaml lacking `fused` predates recording it, and ran unfused"""
    dest = runner.seed_collated()
    runner.tokenize(cfg={**default_cfg("tokenization"), "fused": False}, processed=dest)
    y = OmegaConf.load(dest / "tokenizer.yaml")
    del y.cfg.fused
    (dest / "tokenizer.yaml").write_text(OmegaConf.to_yaml(y))
    assert not Visualizer(processed_data_home=dest).fused


def test_the_payload_says_whether_the_tokenizer_fused_its_bins(runner, pipeline):
    """as the footers of the page and the pdf do"""
    sid = pipeline.tokens_times["subject_id"][0]
    assert payload_of(pipeline)["meta"]["fused"] is True  # the shipped default
    dest = runner.seed_collated()
    runner.tokenize(cfg={**default_cfg("tokenization"), "fused": False}, processed=dest)
    p = Visualizer(processed_data_home=dest).get_payload(sid)
    assert p["meta"]["fused"] is False


def naive_utc(e: pl.Expr) -> pl.Expr:
    return e.dt.convert_time_zone("UTC").dt.replace_time_zone(None)


def as_written_by_26_6(src: pathlib.Path, dest: pathlib.Path) -> pathlib.Path:
    """
    a copy of processed directory `src` as 26.6.x wrote it: times naive utc, no
    subject start_time or end_time, and a tokenizer.yaml without counts
    """
    for f in src.glob("*.parquet"):
        df = pl.read_parquet(f).drop("start_time", "end_time", strict=False)
        df.with_columns(
            naive_utc(pl.col(c))
            if isinstance(t, pl.Datetime)
            else pl.col(c).list.eval(naive_utc(pl.element()))
            for c, t in df.schema.items()
            if isinstance(t, pl.Datetime)
            or (isinstance(t, pl.List) and isinstance(t.inner, pl.Datetime))
        ).write_parquet(dest / f.name)
    y = OmegaConf.load(src / "tokenizer.yaml")
    del y.counts
    (dest / "tokenizer.yaml").write_text(OmegaConf.to_yaml(y))
    return dest


def test_a_timeline_from_26_6_draws_as_it_does_today(pipeline, tmp_path):
    """before 26.9.0, times were stored naive and meant utc"""
    legacy = as_written_by_26_6(pipeline.path, tmp_path)
    times = pl.read_parquet_schema(legacy / "tokens_times.parquet")["times"]
    assert times.inner.time_zone is None
    sid = pipeline.subjects_in_split("held_out")[0]
    now = Visualizer(processed_data_home=pipeline.path, show_winnowing=True)
    then = Visualizer(processed_data_home=legacy, show_winnowing=True)
    a, b = now.get_payload(sid), then.get_payload(sid)
    assert a["meta"]["timezone"] == "UTC"  # shipped default, so labels agree too
    for k in ("times", "codes", "events", "tokens", "winnowed"):
        assert b[k] == a[k], k
    assert b["meta"]["timezone"] == "UTC"
    assert b["meta"]["value_source"] == a["meta"]["value_source"] == "meds"
    assert b["subject"]["start_time"] is None and b["subject"]["end_time"] is None
    assert b["subject"]["fields"] == a["subject"]["fields"]
    assert DATA.search(then.render(sid))
    assert then.render_pdf(sid).startswith(b"%PDF")


def test_an_empty_tokenizer_yaml_says_why(tmp_path):
    """26.6.0 and 26.6.1 crash writing tokenizer.yaml, leaving it empty"""
    (tmp_path / "tokenizer.yaml").write_text("")
    with pytest.raises(ValueError, match="26.6.0 and 26.6.1"):
        Visualizer(processed_data_home=tmp_path)


def copy_of(src: pathlib.Path, dest: pathlib.Path, without=()) -> pathlib.Path:
    """a copy of processed directory `src` in `dest`, less the files `without`"""
    dest.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.name not in without:
            shutil.copy(f, dest / f.name)
    return dest


def subject_with_unk(pipeline) -> str:
    """the first subject whose timeline holds a word out of the vocabulary"""
    return next(
        s for s in pipeline.tokens_times["subject_id"] if "UNK" in pipeline.timeline(s)
    )


def test_pass_through_columns_are_shown_as_subject_details(pipeline, tmp_path):
    dest = copy_of(pipeline.path, tmp_path, without=("subject_splits.parquet",))
    pipeline.splits.with_columns(
        icu_stay=pl.lit(True), weight_kg=pl.lit(float("nan")), death_dttm="end_time"
    ).write_parquet(dest / "subject_splits.parquet")
    sid = pipeline.tokens_times["subject_id"][0]
    subject = Visualizer(processed_data_home=dest).get_payload(sid)["subject"]
    row = pipeline.splits.filter(pl.col("subject_id") == sid).row(0, named=True)
    shown = {f["name"]: (f["label"], f["value"]) for f in subject["fields"]}
    static = ("subject_id", "split", "start_time", "end_time")
    added = ["icu_stay", "weight_kg", "death_dttm"]
    assert list(shown) == [c for c in row if c not in static] + added
    assert shown["race_category"] == ("Race", row["race_category"])
    assert shown["age_at_admission"][0] == "Age at admission"
    assert shown["icu_stay"] == ("Icu stay", "yes")
    assert shown["weight_kg"] == ("Weight kg", None)  # which the page leaves out
    assert shown["death_dttm"] == ("Death dttm", subject["end_time"]["label"])


@pytest.mark.parametrize("splits", ["missing", "without the subject"])
def test_a_subject_without_a_row_of_subject_splits_has_no_details(
    pipeline, tmp_path, splits
):
    sid = pipeline.inference()["subject_id"][0]  # winnowed, but its split unknown
    dest = copy_of(pipeline.path, tmp_path, without=("subject_splits.parquet",))
    if splits == "without the subject":
        pipeline.splits.filter(pl.col("subject_id") != sid).write_parquet(
            dest / "subject_splits.parquet"
        )
    v = Visualizer(processed_data_home=dest, show_winnowing=True)
    p = v.get_payload(sid)
    assert p["subject"] == {
        "subject_id": sid,
        "split": None,
        "start_time": None,
        "end_time": None,
        "fields": [],
    }
    assert p["winnowed"] is None  # without a split, there's no file to look in
    assert pipeline.decode(p["tokens"]["id"]) == pipeline.timeline(sid)
    assert DATA.search(v.render(sid))


def test_values_come_from_meds_when_tokens_lack_them(pipeline):
    assert "numeric_values" not in pipeline.tokens_times.columns  # shipped default
    v = Visualizer(processed_data_home=pipeline.path)
    n, sources = 0, set()
    for sid in pipeline.tokens_times["subject_id"]:
        p = v.get_payload(sid)
        sources.add(p["meta"]["value_source"])
        ev, codes = p["events"], p["codes"]
        for c, b, x in zip(ev["code"], ev["bin"], ev["value"]):
            if x is not None:  # a recovered value re-bins to its token's bin
                assert b == sum(k <= x for k in codes[c]["breaks"] or ())
                n += 1
    assert n > 0 and "meds" in sources and "tokens" not in sources


def test_values_from_meds_match_values_in_tokens(runner, pipeline):
    tok = default_cfg("tokenization")
    with_values, without = runner.seed_collated(), runner.seed_collated()
    runner.tokenize(cfg={**tok, "include_numeric_values": True}, processed=with_values)
    runner.tokenize(cfg=tok, processed=without)
    a = Visualizer(processed_data_home=with_values)
    b = Visualizer(processed_data_home=without)

    def valued(p):  # (instant, code, bin, value), ignoring order within an instant
        ev = p["events"]
        cols = zip(ev["time"], ev["code"], ev["bin"], ev["value"])
        return sorted(row for row in cols if row[3] is not None)

    n = 0
    for sid in pipeline.tokens_times["subject_id"]:
        pa, pb = a.get_payload(sid), b.get_payload(sid)
        assert pa["tokens"]["id"] == pb["tokens"]["id"]
        assert valued(pa) == valued(pb)
        n += len(valued(pa))
        assert pa["meta"]["value_source"] in ("tokens", None)
    assert n > 0


def test_events_carry_their_hours_to_end_time_where_the_tokens_do(runner, pipeline):
    """as include_hours_to_end_time writes them, for the page's tooltips"""
    assert "hours_to_end_time" not in pipeline.tokens_times.columns  # shipped default
    assert set(payload_of(pipeline)["events"]["hours_to_end"]) == {None}
    dest = runner.seed_collated()
    tok = {**default_cfg("tokenization"), "include_hours_to_end_time": True}
    runner.tokenize(cfg=tok, processed=dest)
    v = Visualizer(processed_data_home=dest)
    tt = pl.read_parquet(dest / "tokens_times.parquet")
    for sid, hours in tt.head(3).select("subject_id", "hours_to_end_time").iter_rows():
        ev = v.get_payload(sid)["events"]
        assert ev["hours_to_end"] == [hours[i] for i in ev["first_token"]]
        assert len(set(ev["hours_to_end"])) > 1


def test_without_meds_parquet_binned_events_have_no_values(pipeline, tmp_path):
    dest = copy_of(pipeline.path, tmp_path, without=("meds.parquet",))
    sid = pipeline.tokens_times["subject_id"][0]
    p = Visualizer(processed_data_home=dest).get_payload(sid)
    assert any(b is not None for b in p["events"]["bin"])
    assert set(p["events"]["value"]) == {None}
    assert p["meta"]["value_source"] is None
    assert p["meta"]["has_numeric_values"] is False


def test_a_value_is_shown_only_where_meds_parquet_holds_it(pipeline, tmp_path):
    """at its event's instant, for its code and bin"""
    sid = pipeline.tokens_times["subject_id"][0]
    full = payload_of(pipeline)
    values = full["events"]["value"]
    codes = [full["codes"][c]["code"] for c in full["events"]["code"]]
    code = next(c for c, x in zip(codes, values) if x is not None)
    less = copy_of(pipeline.path, tmp_path / "less", without=("meds.parquet",))
    pipeline.meds.filter(pl.col("code") != code).write_parquet(less / "meds.parquet")
    p = Visualizer(processed_data_home=less).get_payload(sid)
    assert p["events"]["value"] == [
        None if c == code else x for c, x in zip(codes, values)
    ]
    assert p["meta"]["value_source"] == "meds"
    later = copy_of(pipeline.path, tmp_path / "later", without=("meds.parquet",))
    pipeline.meds.with_columns(pl.col("time") + pl.duration(seconds=1)).write_parquet(
        later / "meds.parquet"
    )
    p = Visualizer(processed_data_home=later).get_payload(sid)
    assert set(p["events"]["value"]) == {None}
    assert p["meta"]["value_source"] is None


def ranked_prefixes(
    pipeline, lane_order, claimed=("TIME", "CLCK"), counts=None
) -> list:
    """unclaimed vocabulary prefixes, most frequent in training (or `counts`) first"""
    weight = {}
    for w, n in (pipeline.counts if counts is None else counts).items():
        p = w.partition("//")[0]
        if "//" in w and p not in claimed:
            weight[p] = weight.get(p, 0) + n
    return sorted(weight, key=lambda p: (-weight[p], lane_order(p)))


@pytest.mark.parametrize(
    "word, description",
    [
        ("LAB//sodium_Q5", "Sodium"),  # a fused bin is not part of the code
        ("OR//open", "Incision"),  # an exact key wins over an earlier pattern
        ("OR//laparotomy", "Primary procedure"),  # the first pattern listed wins
        ("APGAR//total//8", "Apgar score"),  # a pattern spans the separator
        ("APGAR//total_hr//2", "Apgar heart rate score"),
        ("LDA//OUT//picc", "Line, drain, or airway removed"),
        ("LDA//IN//picc", None),
        ("LAB//potassium_Q2", "A metal"),  # a blank description leaves it to patterns
        ("LAB//calcium", "A metal"),
        ("VTL//dbp_Q4", "Blood pressure"),  # as fnmatch reads [ and ?, too
        ("VTL//map", None),
        ("RESP//fio2_set", "Inspired oxygen"),
        ("MED-CTS//propofol", None),  # case-sensitive on every platform, as codes are
    ],
)
def test_descriptions_match_exact_keys_before_patterns(
    pipeline, tmp_path, word, description
):
    descriptions = {
        "OR//*": "Primary procedure",
        "OR//open": "Incision",
        "OR//lap*": "Laparotomy",
        "LAB//sodium": "Sodium",
        "LAB//potassium": None,
        "LAB//calcium": "",
        "LAB//*ium": "A metal",
        "APGAR//total//*": "Apgar score",
        "APGAR//total_hr//*": "Apgar heart rate score",
        "LDA//OUT//*": "Line, drain, or airway removed",
        "VTL//[sd]bp": "Blood pressure",
        "RESP//fio?_set": "Inspired oxygen",
        "med-cts//*": "Continuous medication",
    }
    path, _ = viz_cfg(tmp_path, descriptions=descriptions)
    v = Visualizer(path, processed_data_home=pipeline.path)
    assert v.fused
    assert v.describe(v.parse(word)["code"]) == description


def test_a_code_is_described_whatever_bin_is_fused_onto_it(pipeline, tmp_path):
    """under the shipped fused tokenizer, VTL//sbp_Q2 takes VTL//sbp's description"""
    path, _ = viz_cfg(tmp_path)
    v = Visualizer(path, processed_data_home=pipeline.path)
    p = v.get_payload(pipeline.tokens_times["subject_id"][0])
    words = {w for w in p["tokens"]["vocab"].values() if w.startswith("VTL//sbp")}
    assert words and all(re.fullmatch(r"VTL//sbp_Q\d+", w) for w in words)
    (sbp,) = [c for c in p["codes"] if c["code"] == "VTL//sbp"]
    assert sbp["description"] == "Systolic blood pressure"


def test_the_shipped_palette_is_every_chromatic_uchicago_color_but_maroon():
    path = resources.files("cocoa.config") / "visualization.yaml"
    shipped = OmegaConf.to_container(OmegaConf.load(path))
    palette = [c.upper() for c in shipped["palette"]]
    assert len(palette) == len(set(palette)) == 21
    assert not {"#D9D9D9", "#A6A6A6", "#737373"} & set(palette)  # the greys
    lanes = {
        la["color"].upper() for la in shipped.get("lanes") or () if la.get("color")
    }
    assert "#800000" not in set(palette) | lanes  # maroon is kept for the page


def test_unclaimed_prefixes_take_free_palette_colors(pipeline, tmp_path):
    kept = [lane for lane in VIZ_CFG["lanes"] if lane["name"] in ("Time", "Encounter")]
    path, cfg = viz_cfg(tmp_path, lanes=kept)
    v = Visualizer(path, processed_data_home=pipeline.path)
    worn = next(lane["color"] for lane in kept if lane["name"] == "Encounter")
    assert worn in cfg["palette"]
    free = [c for c in cfg["palette"] if c != worn]  # Encounter is in the vocabulary
    claimed = [p for lane in kept for p in lane["prefixes"]]
    ranked = ranked_prefixes(pipeline, v.lane_order, claimed)
    assert len(ranked) < len(free)  # the whole system has room for every one
    assert v.stray_colors == dict(zip(ranked, free))
    colors = {}
    for sid in pipeline.tokens_times["subject_id"]:
        for lane in v.get_payload(sid)["lanes"]:
            if lane["name"] in ("Time", "Encounter"):
                continue
            (p,) = lane["prefixes"]
            want = v.stray_colors.get(p, cfg["other_color"])
            assert lane["color"] == want
            assert colors.setdefault(p, want) == want  # the same in every subject
    assert colors.get("UNK", cfg["other_color"]) == cfg["other_color"]


def test_a_short_palette_colors_the_most_frequent_prefixes(pipeline, tmp_path):
    time = [lane for lane in VIZ_CFG["lanes"] if lane["name"] == "Time"]
    palette = PALETTE[:3]
    path, _ = viz_cfg(tmp_path, lanes=time, palette=palette)
    v = Visualizer(path, processed_data_home=pipeline.path)
    top = ranked_prefixes(pipeline, v.lane_order)[:3]
    assert v.stray_colors == dict(zip(top, palette))


@pytest.mark.parametrize(
    "edit",
    [
        lambda cfg: cfg.pop("palette"),
        lambda cfg: cfg.update(palette=None),
        lambda cfg: cfg.update(palette=[]),
    ],
)
def test_without_a_palette_unclaimed_prefixes_are_other_color(pipeline, tmp_path, edit):
    kept = [lane for lane in VIZ_CFG["lanes"] if lane["name"] != "Vitals"]
    cfg = {**copy.deepcopy(VIZ_CFG), "lanes": kept}
    edit(cfg)
    path = write_cfg(tmp_path / "no_palette.yaml", cfg)
    v = Visualizer(path, processed_data_home=pipeline.path)
    assert v.stray_colors == {}
    sid = pipeline.tokens_times["subject_id"][0]
    vitals = [la for la in v.get_payload(sid)["lanes"] if la["prefixes"] == ["VTL"]]
    assert [la["color"] for la in vitals] == [VIZ_CFG["other_color"]]


@pytest.mark.parametrize(
    "prefixes, worn",
    [(["ADMN", "DSCG"], True), (["NOPE"], False)],  # NOPE is in no vocabulary
)
def test_a_lane_keeps_its_color_from_strays_where_the_vocabulary_has_it(
    pipeline, tmp_path, prefixes, worn
):
    """a lane's color is matched against the palette's whatever its case"""
    palette = ["#aB12cD", *PALETTE[:30]]
    lane = {"name": "Lane", "prefixes": prefixes, "color": "#Ab12Cd"}
    path, _ = viz_cfg(tmp_path, lanes=[lane], palette=palette)
    v = Visualizer(path, processed_data_home=pipeline.path)
    ranked = ranked_prefixes(pipeline, v.lane_order, claimed=prefixes)
    assert v.stray_colors == dict(zip(ranked, palette[1:] if worn else palette))


@pytest.mark.parametrize("counts", ["absent", "null"])
def test_without_training_counts_prefixes_rank_by_their_words(
    pipeline, tmp_path, counts
):
    """
    before 26.9.1, tokenizer.yaml held no counts, so each word counted once; a
    tokenizer transferred from such a yaml with --tokenizer-home writes them null
    """
    if counts == "absent":
        legacy = as_written_by_26_6(pipeline.path, tmp_path)
    else:
        legacy = copy_of(pipeline.path, tmp_path, without=("tokenizer.yaml",))
        y = OmegaConf.load(pipeline.path / "tokenizer.yaml")
        y.counts = dict.fromkeys(y.counts)
        (legacy / "tokenizer.yaml").write_text(OmegaConf.to_yaml(y))
    time = [lane for lane in VIZ_CFG["lanes"] if lane["name"] == "Time"]
    path, _ = viz_cfg(tmp_path, lanes=time)
    v = Visualizer(path, processed_data_home=legacy)
    words = dict.fromkeys(pipeline.vocab, 1)
    ranked = ranked_prefixes(pipeline, v.lane_order, counts=words)
    assert ranked != ranked_prefixes(pipeline, v.lane_order)  # as counts would rank
    assert v.stray_colors == dict(zip(ranked, PALETTE))


def drop(key):
    return lambda cfg: cfg.pop(key)


def empty(key):  # a heading whose entries are all commented out reads as null
    return lambda cfg: cfg.update({key: None})


def vitals(edit):
    return lambda cfg: edit(next(la for la in cfg["lanes"] if la["name"] == "Vitals"))


@pytest.mark.parametrize(
    "edit, name, color",
    [
        (drop("descriptions"), "Vitals", "#A4343A"),
        (empty("descriptions"), "Vitals", "#A4343A"),
        (lambda c: c["descriptions"].update({"VTL//sbp": None}), "Vitals", "#A4343A"),
        (
            lambda c: c["descriptions"].update({123: "a numeric key"}),
            "Vitals",
            "#A4343A",
        ),
        (empty("prefixes"), "Vitals", "#A4343A"),
        (empty("palette"), "Vitals", "#A4343A"),
        (empty("other_color"), "Vitals", "#A4343A"),
        (vitals(lambda la: la.pop("color")), "Vitals", "#565656"),  # other_color
        (vitals(lambda la: la.update(color=None)), "Vitals", "#565656"),
        (vitals(lambda la: la.update(color="")), "Vitals", "#565656"),
        (vitals(lambda la: la.pop("name")), "VTL, VIT", "#A4343A"),
        (vitals(lambda la: la.update(name=None)), "VTL, VIT", "#A4343A"),
        (vitals(lambda la: la.update(name="")), "VTL, VIT", "#A4343A"),
        (vitals(lambda la: la.pop("prefixes")), "Vital sign", None),  # a stray now
        (empty("lanes"), "Vital sign", None),
        (lambda c: c.clear(), "VTL", "#737373"),  # the built-in other_color
    ],
)
def test_every_config_section_is_optional(pipeline, tmp_path, edit, name, color):
    cfg = copy.deepcopy(VIZ_CFG)
    edit(cfg)
    v = Visualizer(
        write_cfg(tmp_path / "v.yaml", cfg), processed_data_home=pipeline.path
    )
    sid = pipeline.tokens_times["subject_id"][0]
    p = v.get_payload(sid)
    assert "__COCOA_" not in v.render(sid)
    assert all(isinstance(la["color"], str) for la in p["lanes"])
    (lane,) = [la for la in p["lanes"] if "VTL" in la["prefixes"]]
    assert lane["name"] == name
    assert lane["color"] == (color or v.stray_colors["VTL"])
    codes = {c["code"]: c for c in p["codes"]}
    if "descriptions" not in cfg or not cfg["descriptions"]:
        assert all(c["description"] is None for c in codes.values())
    elif "VTL//sbp" in codes and cfg["descriptions"].get("VTL//sbp", "") is None:
        assert codes["VTL//sbp"]["description"] is None


def test_lanes_run_top_to_bottom_as_listed_with_strays_after(pipeline, tmp_path):
    lanes = [
        {"name": "Vitals", "prefixes": ["VTL"]},
        {"name": "Labs", "prefixes": ["LAB-RES", "LAB-ORD"]},
        {
            "name": "Encounter",
            "prefixes": ["ADMN", "XFR-OUT", "XFR-IN", "CODE", "DSCG"],
        },
    ]
    sid = pipeline.tokens_times["subject_id"][0]
    for listed in (lanes, lanes[::-1]):
        path, _ = viz_cfg(tmp_path, lanes=listed)
        v = Visualizer(path, processed_data_home=pipeline.path)
        drawn = v.get_payload(sid)["lanes"]
        assert [la["name"] for la in drawn[:3]] == [la["name"] for la in listed]
        strays = [la["prefixes"] for la in drawn[3:]]
        assert strays and all(len(ps) == 1 for ps in strays)
        assert strays == sorted(strays, key=lambda ps: v.lane_order(ps[0]))


def test_strays_follow_the_tokenizer_ordering_then_their_names_and_unk_last(tmp_path):
    ordering = ["BOS", "SEX", "ETHN", "EOS"]
    write_cfg(tmp_path / "tokenizer.yaml", {"cfg": {"ordering": ordering}})
    path, _ = viz_cfg(tmp_path)
    v = Visualizer(path, processed_data_home=tmp_path)
    lanes = v.get_lanes({"ZZZ", "UNK", "AAA", "ETHN", "SEX", "VTL", "BOS", "EOS"})
    assert [la["prefixes"] for la in lanes] == [
        ["VTL", "VIT"],  # the configured lane holding one of them
        ["SEX"],
        ["ETHN"],
        ["AAA"],
        ["ZZZ"],
        ["UNK"],
    ]


@pytest.mark.parametrize(
    "edit",
    [
        lambda la: la.pop("prefixes"),
        lambda la: la.update(prefixes=None),
        lambda la: la.update(prefixes=[]),
    ],
)
def test_a_lane_without_prefixes_is_left_out(pipeline, tmp_path, edit):
    """checked on `lanes`, since the page draws only lanes holding a subject's codes"""
    cfg = copy.deepcopy(VIZ_CFG)
    vitals(edit)(cfg)
    v = Visualizer(
        write_cfg(tmp_path / "v.yaml", cfg), processed_data_home=pipeline.path
    )
    assert [la["name"] for la in v.lanes] == ["Time", "Encounter"]
    assert "VTL" in v.unclaimed_prefixes()


@pytest.mark.parametrize(
    "edit, other",
    [
        (lambda cfg: None, "#565656"),
        (drop("other_color"), "#737373"),
        (empty("other_color"), "#737373"),
    ],
)
def test_other_color_is_worn_where_no_color_is_given(pipeline, tmp_path, edit, other):
    """by a lane without a color, by prefixes past the palette's end, and by UNK"""
    cfg = copy.deepcopy(VIZ_CFG)
    cfg.update(lanes=[{"name": "Vitals", "prefixes": ["VTL"]}], palette=PALETTE[:1])
    edit(cfg)
    v = Visualizer(
        write_cfg(tmp_path / "v.yaml", cfg), processed_data_home=pipeline.path
    )
    first, *strays = v.get_payload(subject_with_unk(pipeline))["lanes"]
    assert (first["name"], first["color"]) == ("Vitals", other)
    colors = {la["prefixes"][0]: la["color"] for la in strays}
    (top,) = v.stray_colors  # the palette's one color goes to the most frequent
    assert colors.pop(top) == PALETTE[0]
    assert "UNK" in colors and set(colors.values()) == {other}


def test_prefix_names_label_codes_and_the_lanes_of_strays(pipeline, tmp_path):
    names = {
        "VTL": "Vital sign",
        "LAB-RES": "Lab result",
        "BOS": "Beginning of sequence",
        "UNK": "Out of vocabulary",
        "RESP": None,  # a name left blank is no name
        "XFR-IN": "",
    }
    lanes = [{"name": "Vitals", "prefixes": ["VTL"]}]
    path, _ = viz_cfg(tmp_path, lanes=lanes, prefixes=names)
    v = Visualizer(path, processed_data_home=pipeline.path)
    p = v.get_payload(subject_with_unk(pipeline))
    named = {k: name for k, name in names.items() if name}
    assert {c["prefix"] for c in p["codes"]} >= names.keys()
    for c in p["codes"]:
        assert c["category"] == named.get(c["prefix"]), c["code"]
    lane_names = {la["prefixes"][0]: la["name"] for la in p["lanes"]}
    assert lane_names["VTL"] == "Vitals"  # a configured lane goes by its own name
    assert lane_names["LAB-RES"] == "Lab result"
    assert lane_names["UNK"] == "Out of vocabulary"
    assert lane_names["RESP"] == "RESP" and lane_names["XFR-IN"] == "XFR-IN"


def test_winnowing_is_not_shown_by_default(pipeline):
    sid = pipeline.inference()["subject_id"][0]
    v = Visualizer(processed_data_home=pipeline.path)
    assert v.show_winnowing is False
    assert v.get_payload(sid)["winnowed"] is None  # no split, threshold, or flags


@pytest.mark.parametrize("edit", [drop("show_winnowing"), empty("show_winnowing")])
def test_a_config_without_show_winnowing_leaves_it_off(pipeline, tmp_path, edit):
    path = resources.files("cocoa.config") / "visualization.yaml"
    cfg = OmegaConf.to_container(OmegaConf.load(path))
    edit(cfg)
    v = Visualizer(
        write_cfg(tmp_path / "v.yaml", cfg), processed_data_home=pipeline.path
    )
    sid = pipeline.inference()["subject_id"][0]
    assert v.get_payload(sid)["winnowed"] is None


def test_winnowing_is_shown_by_config_or_kwarg(pipeline, tmp_path):
    sid = pipeline.inference()["subject_id"][0]
    path = resources.files("cocoa.config") / "visualization.yaml"
    cfg = OmegaConf.to_container(OmegaConf.load(path))
    cfg["show_winnowing"] = True
    by_cfg = Visualizer(
        write_cfg(tmp_path / "v.yaml", cfg), processed_data_home=pipeline.path
    )
    by_kwarg = Visualizer(processed_data_home=pipeline.path, show_winnowing=True)
    for v in (by_cfg, by_kwarg):
        w = v.get_payload(sid)["winnowed"]
        assert w["last_valid"] > 0 and w["outcomes"]


def test_winnowing_shows_the_split_and_the_flags_the_winnower_wrote(pipeline):
    v = Visualizer(processed_data_home=pipeline.path, show_winnowing=True)
    flipped = 0
    for split in ("train", "tuning", "held_out"):
        df = pipeline.inference(split)
        row = df.row(0, named=True)
        w = v.get_payload(row["subject_id"])["winnowed"]
        assert w["file"] == f"{split}_for_inference.parquet"
        assert w["last_valid"] == row["last_valid"]
        assert w["n_future"] == len(row["tokens_future"])
        flags = [  # tokens_past is a list, not a flag
            c.removesuffix("_past")
            for c, t in df.schema.items()
            if c.endswith("_past") and t == pl.Boolean
        ]
        assert flags and [o["code"] for o in w["outcomes"]] == flags
        for o in w["outcomes"]:
            assert o["past"] == row[f"{o['code']}_past"]
            assert o["future"] == row[f"{o['code']}_future"]
            flipped += o["past"] != o["future"]
    assert flipped  # so that a past and a future can't be mistaken for each other


def test_winnowing_shows_nothing_of_a_timeline_the_winnower_did_not_split(
    pipeline, tmp_path
):
    winnowed = pipeline.inference()["subject_id"].to_list()
    short = next(s for s in pipeline.subjects_in_split("held_out") if s not in winnowed)
    v = Visualizer(processed_data_home=pipeline.path, show_winnowing=True)
    assert v.get_payload(short)["winnowed"] is None  # too short for the threshold
    dest = copy_of(pipeline.path, tmp_path, without=("held_out_for_inference.parquet",))
    v = Visualizer(processed_data_home=dest, show_winnowing=True)
    assert v.get_payload(winnowed[0])["winnowed"] is None


def test_overrides_edit_the_config_and_kwargs_override_both(pipeline, tmp_path):
    sid = pipeline.inference()["subject_id"][0]
    path, _ = viz_cfg(tmp_path, show_winnowing=True)

    def visualizer(**kwargs):
        return Visualizer(path, processed_data_home=pipeline.path, **kwargs)

    v = visualizer(
        overrides=["~lanes", "other_color='#123456'", "show_winnowing=false"]
    )
    assert v.lanes == [] and v.other_color == "#123456"
    assert v.get_payload(sid)["winnowed"] is None
    v = visualizer(
        overrides=["other_color='#123456'"], other_color="#654321", show_winnowing=None
    )
    assert v.other_color == "#654321"  # a kwarg wins over an override
    assert v.get_payload(sid)["winnowed"] is not None  # and a None kwarg is ignored
    assert visualizer(show_winnowing=False).show_winnowing is False
    # an unquoted # starts a yaml comment, so the value is null, as the readme warns
    assert visualizer(overrides=["other_color=#123456"]).other_color == "#737373"


def test_save_pdf_writes_the_subject_s_pdf(pipeline, tmp_path):
    sid = pipeline.tokens_times["subject_id"][0]
    v = Visualizer(processed_data_home=pipeline.path)
    out = v.save_pdf(sid, tmp_path / "nested" / "t.pdf")
    assert out == (tmp_path / "nested" / "t.pdf").resolve()
    data = out.read_bytes()
    assert data.startswith(b"%PDF-") and data.endswith(b"%%EOF\n")
    assert f"Subject {sid}".encode("utf-16-be").hex().upper().encode() in data


def test_unknown_subject_raises(pipeline):
    with pytest.raises(SubjectNotFoundError):
        Visualizer(processed_data_home=pipeline.path).render("no-such-subject")
    with pytest.raises(SubjectNotFoundError):
        Visualizer(processed_data_home=pipeline.path).render_pdf("no-such-subject")


@pytest.fixture
def served():
    """make_server's server on a free port, answering from a thread of its own"""
    page = "<!doctype html><title>Subject 7 · cocoa</title><p>°C ≥ 6.5</p>"
    server = make_server(page, port=0)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield server, page.encode("utf-8")
    finally:
        server.shutdown()
        server.server_close()


def fetch(server, method: str, path: str) -> tuple[int, dict, bytes]:
    """status, headers, and body of a raw request, as the server sent them"""
    with socket.create_connection(server.server_address[:2], timeout=10) as s:
        s.sendall(f"{method} {path} HTTP/1.0\r\n\r\n".encode())
        data = b"".join(iter(lambda: s.recv(1 << 16), b""))
    head, _, body = data.partition(b"\r\n\r\n")
    status, *lines = head.decode("latin-1").split("\r\n")
    return int(status.split()[1]), dict(h.split(": ", 1) for h in lines), body


@pytest.mark.parametrize(
    "path", ["/", "/index.html", "/?subject=7", "/index.html?a=1&b=2"]
)
def test_the_server_answers_with_the_page_at_its_root(served, path):
    server, page = served
    status, headers, body = fetch(server, "GET", path)
    assert (status, body) == (200, page)
    assert headers["Content-Type"] == "text/html; charset=utf-8"
    assert headers["Content-Length"] == str(len(page))  # in bytes, not characters
    assert headers["Cache-Control"] == "no-store"


def test_the_server_answers_head_with_headers_alone(served):
    server, page = served
    status, headers, body = fetch(server, "HEAD", "/")
    assert (status, body) == (200, b"")
    assert headers["Content-Length"] == str(len(page))


@pytest.mark.parametrize("method", ["GET", "HEAD"])
@pytest.mark.parametrize(
    "path", ["/favicon.ico", "/index.htm", "/index.html/", "/timeline.js"]
)
def test_the_server_answers_any_other_path_with_not_found(served, method, path):
    server, page = served
    status, _, body = fetch(server, method, path)
    assert status == 404 and page not in body
    assert method == "GET" or body == b""


def test_the_server_listens_on_localhost_and_logs_nothing(served, capsys):
    server, _ = served
    assert server.server_address[0] == "127.0.0.1" and server.server_port > 0
    for path in ("/", "/favicon.ico"):
        fetch(server, "GET", path)
    assert capsys.readouterr().err == ""


def test_a_stalled_connection_does_not_hold_up_the_page(served):
    """browsers open connections they may never use; each gets a thread of its own"""
    server, page = served
    idle = socket.create_connection(server.server_address[:2], timeout=10)
    try:
        assert fetch(server, "GET", "/")[2] == page
    finally:
        idle.close()
