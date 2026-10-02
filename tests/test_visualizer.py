#!/usr/bin/env python3

"""
visualization: rendering one subject's timeline as a self-contained html page
"""

import copy
import importlib.resources as resources
import json
import re

import polars as pl
import pytest
from conftest import default_cfg, write_cfg
from omegaconf import OmegaConf

from cocoa.visualizer import SubjectNotFoundError, Visualizer, to_script_json

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


def ranked_prefixes(pipeline, lane_order, claimed=("TIME", "CLCK")) -> list:
    """unclaimed vocabulary prefixes, most frequent in training first"""
    weight = {}
    for w, n in pipeline.counts.items():
        p = w.partition("//")[0]
        if "//" in w and p not in claimed:
            weight[p] = weight.get(p, 0) + n
    return sorted(weight, key=lambda p: (-weight[p], lane_order(p)))


@pytest.mark.parametrize(
    "word, description",
    [
        ("LAB//sodium_Q5", "Sodium"),  # a fused bin is not part of the code
        ("OR//open", "Incision"),  # an exact key wins over an earlier pattern
        ("OR//laparotomy", "Primary procedure"),
        ("APGAR//total//8", "Apgar score"),  # a pattern spans the separator
        ("APGAR//total_hr//2", "Apgar heart rate score"),
        ("LDA//OUT//picc", "Line, drain, or airway removed"),
        ("LDA//IN//picc", None),
    ],
)
def test_descriptions_match_exact_keys_before_patterns(
    pipeline, tmp_path, word, description
):
    descriptions = {
        "OR//*": "Primary procedure",
        "OR//open": "Incision",
        "LAB//sodium": "Sodium",
        "APGAR//total//*": "Apgar score",
        "APGAR//total_hr//*": "Apgar heart rate score",
        "LDA//OUT//*": "Line, drain, or airway removed",
    }
    path, _ = viz_cfg(tmp_path, descriptions=descriptions)
    v = Visualizer(path, processed_data_home=pipeline.path)
    assert v.fused
    assert v.describe(v.parse(word)["code"]) == description


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


def test_without_a_palette_unclaimed_prefixes_are_other_color(pipeline, tmp_path):
    kept = [lane for lane in VIZ_CFG["lanes"] if lane["name"] != "Vitals"]
    cfg = {k: v for k, v in VIZ_CFG.items() if k != "palette"}
    path = write_cfg(tmp_path / "no_palette.yaml", {**cfg, "lanes": kept})
    v = Visualizer(path, processed_data_home=pipeline.path)
    assert v.stray_colors == {}
    sid = pipeline.tokens_times["subject_id"][0]
    vitals = [la for la in v.get_payload(sid)["lanes"] if la["prefixes"] == ["VTL"]]
    assert [la["color"] for la in vitals] == [VIZ_CFG["other_color"]]


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
        (vitals(lambda la: la.pop("name")), "VTL, VIT", "#A4343A"),
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
