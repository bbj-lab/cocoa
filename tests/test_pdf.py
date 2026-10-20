#!/usr/bin/env python3

"""the static pdf: a well-formed file, its text, and the timeline it lays out"""

import collections
import datetime
import importlib.resources as resources
import re
import shutil
import struct
import zlib

import polars as pl
import pytest
import synth
from conftest import default_cfg, write_cfg

import cocoa.pdf as pdf
from cocoa.collator import Collator
from cocoa.pdf import (
    Document,
    TimelinePdf,
    clip,
    fmt_dur,
    fmt_num,
    measure,
    read_cmap,
    rgb,
    timeline_pdf,
    wrap,
)
from cocoa.tokenizer import Tokenizer
from cocoa.visualizer import Visualizer

TJ = re.compile(
    rb"/(F\d) [\d.]+ Tf [-\d.]+ [-\d.]+ Td (\((?:\\.|[^\\)])*\)|<[\dA-F]*>) Tj"
)
# each drawing operation is a line of a page's content stream
SHOW = re.compile(
    rb"(?m)^BT ([\d.]+ [\d.]+ [\d.]+) rg /(F\d) ([\d.]+) Tf ([-\d.]+) ([-\d.]+) Td "
    rb"(.*) Tj ET$"
)
FILL = re.compile(
    rb"(?m)^([\d.]+ [\d.]+ [\d.]+) rg ([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+) re f$"
)
STROKE = re.compile(rb"(?m)^([\d.]+ [\d.]+ [\d.]+) RG ([\d.]+) w (\d) J (1 j )?(.*) S$")
Run = collections.namedtuple("Run", "text x y size font color")
Fill = collections.namedtuple("Fill", "color x y w h")
Stroke = collections.namedtuple("Stroke", "color width cap join paths")
# what the winnowing summary states, in the order it states them
SUMMARY = ("Threshold", "Past tokens", "Future tokens", "Within the horizon")

# the tests' own visualization config, so that none depends on the shipped one's
# lanes or names: a lane of one prefix in a color of its own, one of several
# prefixes left to other_color, and a name for a prefix left a lane of its own;
# the prefixes no lane claims take the palette's colors, most frequent first
LANES = {
    "lanes": [
        {"name": "Heart and vessels", "prefixes": ["VTL"], "color": "#A4343A"},
        {"name": "Stay", "prefixes": ["ADMN", "XFR-OUT", "XFR-IN", "CODE", "DSCG"]},
    ],
    "palette": ["#007396", "#275D38", "#CC8A00"],
    "other_color": "#565656",
    "prefixes": {"LAB-RES": "Lab result"},
}


def objects(data: bytes) -> dict[int, bytes]:
    """each object of a pdf, by number"""
    found = re.findall(rb"(?m)^(\d+) 0 obj\n(.*?)\nendobj$", data, re.S)
    return {int(n): obj for n, obj in found}


def inflate(obj: bytes) -> bytes:
    """the data of a stream object, inflated"""
    n = int(re.match(rb"<< /Length (\d+) ", obj).group(1))
    start = obj.index(b">>\nstream\n") + len(b">>\nstream\n")
    return zlib.decompress(obj[start : start + n])


def streams(data: bytes) -> list[bytes]:
    """each page's content stream, inflated"""
    objs = objects(data)
    return [inflate(objs[int(n)]) for n in re.findall(rb"/Contents (\d+) 0 R", data)]


def readers(data: bytes) -> dict:
    """
    how a pdf reader turns the strings shown in each font back into text: through
    the font's ToUnicode map if it has one, else as winansi
    """
    objs, out = objects(data), {}
    for font, n in re.findall(rb"/(F\d) (\d+) 0 R", data):
        if m := re.search(rb"/ToUnicode (\d+) 0 R", objs[int(n)]):
            cmap = inflate(objs[int(m.group(1))])
            pairs = re.findall(
                rb"<([\dA-F]{4})> <([\dA-F]+)>",
                b"".join(re.findall(rb"beginbfchar\n(.*?)endbfchar", cmap, re.S)),
            )
            text = {g: bytes.fromhex(u.decode()).decode("utf-16-be") for g, u in pairs}
            out[font] = lambda s, t=text: "".join(
                t[s[i : i + 4]] for i in range(1, len(s) - 1, 4)
            )
        else:
            out[font] = lambda s: re.sub(rb"\\(.)", rb"\1", s[1:-1]).decode("cp1252")
    return out


def page_texts(data: bytes) -> list[str]:
    """the text set on each page, one string per page, as a reader would copy it"""
    read = readers(data)
    return [
        "\n".join(read[font](shown) for font, shown in TJ.findall(s))
        for s in streams(data)
    ]


def check_structure(data: bytes) -> int:
    """assert the xref table points at every object; return the page count"""
    assert data.startswith(b"%PDF-1.6\n") and data.endswith(b"%%EOF\n")
    start = int(re.search(rb"startxref\n(\d+)\n%%EOF", data).group(1))
    assert data[start:].startswith(b"xref\n")
    n = int(re.match(rb"xref\n0 (\d+)\n", data[start:]).group(1))
    rows = re.findall(rb"(\d{10}) 00000 n \n", data[start:])
    assert len(rows) == n - 1
    for k, off in enumerate(rows, 1):
        assert data[int(off) :].startswith(f"{k} 0 obj\n".encode()), k
    assert re.search(rb"/Size %d /Root 1 0 R" % n, data)
    return int(re.search(rb"/Type /Pages /Kids \[[^]]*\] /Count (\d+)", data).group(1))


def payload_of(pipeline, n: int = 0, **kwargs) -> dict:
    sid = pipeline.tokens_times["subject_id"][n]
    return Visualizer(processed_data_home=pipeline.path, **kwargs).get_payload(sid)


def runs(data: bytes) -> list[list[Run]]:
    """
    the text set on each page, run by run: what a reader copies, where its
    baseline starts (from the top of the page, as the layout measures), its size,
    font, and color
    """
    read = readers(data)
    return [
        [
            Run(
                read[font](s),
                float(x),
                round(pdf.PAGE_H - float(y), 2),
                float(size),
                font.decode(),
                color.decode(),
            )
            for color, font, size, x, y, s in SHOW.findall(page)
        ]
        for page in streams(data)
    ]


def fills(page: bytes) -> list[Fill]:
    """the filled rectangles of a page's content, from the top of the page"""
    out = []
    for color, *box in FILL.findall(page):
        x, y, w, h = map(float, box)
        out.append(Fill(color.decode(), x, round(pdf.PAGE_H - y - h, 2), w, h))
    return out


def strokes(page: bytes) -> list[Stroke]:
    """the stroked paths of a page's content, each a list of its subpaths' points"""
    out = []
    for color, width, cap, join, path in STROKE.findall(page):
        parts, nums = [], path.split()
        for i in range(0, len(nums), 3):
            point = (float(nums[i]), round(pdf.PAGE_H - float(nums[i + 1]), 2))
            if nums[i + 2] == b"m":
                parts.append([point])
            else:
                parts[-1].append(point)
        out.append(Stroke(color.decode(), float(width), int(cap), bool(join), parts))
    return out


def defined(page: list[Run]) -> dict[str, str]:
    """the label-over-value pairs of a page's definitions, by label"""
    values = {(r.x, r.y): r.text for r in page if r.size == 8.5}
    return {
        r.text: values[(r.x, round(r.y + 10, 2))]
        for r in page
        if r.size == 6.5 and (r.x, round(r.y + 10, 2)) in values
    }


def check_layout(data: bytes) -> None:
    """assert that all text sits within the margins, and above the footer's"""
    for k, page in enumerate(runs(data), 1):
        for r in page:
            right = r.x + measure(r.text, r.size, r.font)
            assert pdf.MARGIN - 0.05 <= r.x <= right <= pdf.PAGE_W - pdf.MARGIN + 0.05
            assert pdf.MARGIN <= r.y <= pdf.BOTTOM or r.y == pdf.PAGE_H - 20, (k, r)


def winnowed(runner, pipeline, **cfg) -> dict:
    """
    the payload, winnowing shown, of the first held-out timeline that the winnower
    keeps under `cfg` over the shipped defaults
    """
    out = runner.dir()
    cfg = {**default_cfg("winnowing"), "splits": ["held_out"], **cfg}
    runner.winnow(cfg=cfg, processed=pipeline.path, output=out)
    sid = pl.read_parquet(out / "held_out_for_inference.parquet")["subject_id"][0]
    return Visualizer(processed_data_home=out, show_winnowing=True).get_payload(sid)


def threshold_ms(p: dict) -> float:
    """when a winnowed timeline's past ends: at its last token, or the first if none"""
    n = min(p["winnowed"]["last_valid"], len(p["tokens"]["id"]))
    e = p["tokens"]["event"][max(0, n - 1)]
    return p["times"]["ms"][p["events"]["time"][e]]


def is_outcome(r: Run) -> bool:
    """the code of an outcome that the winnowing summary gives a row"""
    return (r.x, r.font, r.size) == (pdf.MARGIN, "F3", 7)


def is_absent(r: Run) -> bool:
    """a line of the outcomes that the winnowing summary lists as absent"""
    return (r.x, r.font, r.size, r.color) == (pdf.MARGIN, "F1", 7, rgb(pdf.INK))


@pytest.fixture(scope="module")
def lanes_cfg(tmp_path_factory):
    return write_cfg(tmp_path_factory.mktemp("viz") / "visualization.yaml", LANES)


@pytest.fixture(scope="module")
def heart_rates(tmp_path_factory, lanes_cfg) -> Visualizer:
    """
    timelines of heart rates alone, collated in America/Chicago: three spanning
    weeks, months, and years; three ever denser, the last a burst of readings
    within half an hour of a stay of days; and one whose readings are all lower
    than any other's
    """
    root = tmp_path_factory.mktemp("heart_rates")
    at, minute = datetime.datetime, datetime.timedelta(minutes=1)
    stays = {
        "weeks": [at(2024, 1, 3, 8), at(2024, 1, 24, 8)],
        "months": [at(2023, 11, 15, 8), at(2024, 4, 20, 8)],
        "years": [at(2015, 3, 1, 8), at(2024, 6, 1, 8)],
        "few": [at(2024, 1, 1, 8) + k * 60 * minute for k in range(10)],
        "some": [at(2024, 1, 1, 8) + k * 15 * minute for k in range(100)],
        "burst": [
            at(2024, 1, 1, 8),
            *(at(2024, 1, 2, 8) + k * minute / 10 for k in range(298)),
            at(2024, 1, 3, 8),
        ],
        "low": [at(2024, 1, 1, 8) + k * 60 * minute for k in range(4)],
    }
    raw = synth.write_minimal_dataset(
        root / "raw",
        hospitalizations=[
            {"hospitalization_id": h, "admission_dttm": ts[0], "discharge_dttm": ts[-1]}
            for h, ts in stays.items()
        ],
        vitals=[
            {
                "hospitalization_id": h,
                "recorded_dttm": t,
                "vital_value": 30.0 if h == "low" else 40.0 + k * 37 % 120,
            }
            for h, ts in stays.items()
            for k, t in enumerate(ts)
        ],
        tz="America/Chicago",
    )
    cfg = synth.minimal_collation_cfg(default_timezone="America/Chicago")
    dest = root / "processed"
    dest.mkdir()
    Collator(
        collation_cfg=write_cfg(root / "collation.yaml", cfg),
        raw_data_home=raw,
        processed_data_home=dest,
    ).save_all()
    Tokenizer(processed_data_home=dest).save_all()
    return Visualizer(lanes_cfg, processed_data_home=dest)


# ------------------------------------------------------------------ the writer


def test_a_document_is_well_formed():
    doc = Document("a title")
    for k in range(3):
        doc.new_page().text(10, 20, f"page {k}")
    data = doc.to_bytes()
    assert check_structure(data) == 3
    assert page_texts(data) == ["page 0", "page 1", "page 2"]


def test_a_document_references_nothing_outside_itself():
    doc = Document("t")
    doc.new_page().text(10, 20, "see https://example.org")  # text, not a link
    data = doc.to_bytes()
    for key in (b"/URI", b"/Launch", b"/GoToR", b"/EmbeddedFile"):
        assert key not in data
    fonts = set(re.findall(rb"/BaseFont /([\w-]+)", data))
    assert fonts == {b"Gotham-Book", b"Gotham-Medium", b"Courier"}


def test_the_page_fonts_are_embedded_whole():
    doc = Document("t")
    doc.new_page().text(10, 20, "x")
    data = doc.to_bytes()
    programs = [
        inflate(obj) for obj in objects(data).values() if b"/Subtype /OpenType" in obj
    ]
    fonts = resources.files("cocoa.assets") / "fonts"
    packaged = [(fonts / f).read_bytes() for f in pdf.EMBEDDED.values()]
    assert sorted(programs) == sorted(packaged)
    assert len(re.findall(rb"/FontFile3 \d+ 0 R", data)) == len(packaged)


def test_gotham_sets_what_it_has_and_what_it_lacks_becomes_a_question_mark():
    doc = Document("t")
    doc.new_page().text(0, 10, "a (b) °C ≥ 6.5, cmH₂O · Łódź\nok")
    assert page_texts(doc.to_bytes()) == ["a (b) °C ? 6.5, cmH₂O · Łódź ok"]


def test_courier_text_is_escaped_and_what_winansi_lacks_becomes_a_question_mark():
    doc = Document("t")
    doc.new_page().text(0, 10, "a (b) \\ °C ≥ 6.5\nok", font="F3")
    (s,) = streams(doc.to_bytes())
    assert b"(a \\(b\\) \\\\ \xb0C ? 6.5 ok) Tj" in s  # \xb0 is cp1252's °


def test_without_its_font_files_text_is_set_in_standard_fonts(monkeypatch):
    monkeypatch.setattr(pdf, "face", lambda font: None)
    doc = Document("t")
    doc.new_page().text(0, 10, "°C ≥ 6.5, cmH₂O")
    data = doc.to_bytes()
    fonts = set(re.findall(rb"/BaseFont /([\w-]+)", data))
    assert fonts == {b"Helvetica", b"Helvetica-Bold", b"Courier"}  # every reader's
    assert b"/FontFile" not in data
    assert page_texts(data) == ["°C ? 6.5, cmH?O"]
    assert measure("W", 10) == pytest.approx(9.44)  # helvetica's afm width


def test_metadata_holds_any_unicode():
    data = Document("Subject é ≥ · cocoa").to_bytes()
    title = "Subject é ≥ · cocoa".encode("utf-16-be").hex().upper().encode()
    assert b"/Title <FEFF" + title + b">" in data


def test_measure_follows_the_font_metrics():
    assert measure("MMMM", 10, "F3") == pytest.approx(24.0)  # courier is 600 wide
    assert measure("W", 10) > measure("i", 10)
    assert measure("abc", 10, "F2") > measure("abc", 10, "F1")
    assert measure("", 10) == 0


def test_clip_cuts_with_an_ellipsis_to_fit():
    s = "respiratory support, a long lane name"
    assert clip(s, 8, "F1", 1000) == s
    short = clip(s, 8, "F1", 60)
    assert short.endswith("…") and s.startswith(short[:-1])
    assert measure(short, 8) <= 60
    assert clip(s, 8, "F1", 1) == ""


def test_wrap_breaks_at_spaces_within_the_width():
    lines = wrap("one two three four five six seven eight", 8, "F1", 50)
    assert (
        len(lines) > 1 and " ".join(lines) == "one two three four five six seven eight"
    )
    assert all(measure(line, 8) <= 50 for line in lines if " " in line)


@pytest.mark.parametrize(
    "color, want",
    [
        ("#800000", "0.502 0.000 0.000"),
        ("#fff", "1.000 1.000 1.000"),
        ("#A4343A", "0.643 0.204 0.227"),
        ("red", "0.451 0.451 0.451"),  # not hex: the dark greystone of other_color
        ("#12345", "0.451 0.451 0.451"),
    ],
)
def test_rgb_reads_hex_colors(color, want):
    assert rgb(color) == want


def format4(segments: list) -> bytes:
    """
    a format 4 cmap subtable of (first, last, delta, glyph ids or None) segments,
    closed by the 0xFFFF segment that the format requires
    """
    segments = [*segments, (0xFFFF, 0xFFFF, 1, None)]
    n, ids, offsets = len(segments), [], []
    for k, (_, _, _, glyphs) in enumerate(segments):
        # from where this segment's offset is stored to its first glyph id
        offsets.append(2 * (n - k + len(ids)) if glyphs else 0)
        ids += glyphs or []
    body = b"".join(
        struct.pack(f">{n}{t}", *(s[i] for s in segments)) + pad
        for i, t, pad in ((1, "H", b"\0\0"), (0, "H", b""), (2, "h", b""))
    )
    body += struct.pack(f">{n}H{len(ids)}H", *offsets, *ids)
    return struct.pack(">7H", 4, 14 + len(body), 0, 2 * n, 0, 0, 0) + body


def test_the_cmap_reader_leaves_out_what_maps_to_no_glyph():
    """glyph 0 draws as an empty box, so a character mapped to it falls to ?"""
    segments = [
        (ord("0"), ord("0"), -ord("0"), None),  # by delta, to glyph 0
        (ord("A"), ord("C"), 10 - ord("A"), None),
        (ord("a"), ord("c"), 0, [20, 0, 22]),  # by glyph ids, b to glyph 0
        (ord("x"), ord("y"), 100, [5, 6]),  # glyph ids take the delta too
    ]
    want = {ord("A"): 10, ord("B"): 11, ord("C"): 12, ord("a"): 20, ord("c"): 22}
    want |= {ord("x"): 105, ord("y"): 106}
    unicode = format4(segments)
    table = struct.pack(">2H2HI", 0, 1, 0, 3, 12) + unicode  # its one subtable
    assert read_cmap(b"head" + table, 4) == want
    # a windows subtable of the bmp is read before a unicode one
    windows = format4([(ord("A"), ord("A"), 1, None)])
    table = struct.pack(">2H2HI2HI", 0, 2, 0, 3, 20, 3, 1, 20 + len(unicode))
    assert read_cmap(table + unicode + windows, 0) == {ord("A"): ord("A") + 1}


@pytest.mark.parametrize("s", ["", "abc"])
def test_clip_leaves_nothing_where_there_is_no_room(s):
    """as a code's name does beside a prefix wider than the gutter"""
    assert clip(s, 8, "F1", -5) == ""


@pytest.mark.parametrize(
    "v, digits, want",
    [
        (None, 6, ""),
        (0, 6, "0"),
        (0.5, 6, "0.5"),
        (-42.25, 6, "-42.25"),
        (98.6, 6, "98.6"),
        (1234.56789, 6, "1,234.57"),
        (1234.56789, 4, "1,235"),
        (1234567, 6, "1,234,570"),
        (0.001, 6, "0.001"),
        (1.5e10, 6, "1.5e+10"),
    ],
)
def test_numbers_read_to_significant_digits_with_commas(v, digits, want):
    assert fmt_num(v, digits) == want


@pytest.mark.parametrize(
    "v, written, on_the_page",
    [
        (0.000123, "0.000123", "1.23e-4"),
        (-3e-5, "-3e-05", "-3e-5"),
        (2.5e9, "2.5e+09", "2.5e+9"),
    ],
)
def test_tiny_and_huge_numbers_are_written_unlike_the_page(v, written, on_the_page):
    """
    BUG (current behaviour asserted): fmt_num means to write a number as the
    page's fmtNum does, in exponent form below 1e-3 and from 1e9, but python's
    `g` format keeps 1e-4 to 1e-3 in fixed point and pads exponents to two
    digits, where javascript's toExponential does neither
    """
    assert fmt_num(v, 6) == written != on_the_page


@pytest.mark.parametrize(
    "ms, want",
    [
        (0, "0 s"),
        (59_000, "59 s"),
        (59_600, "1 min"),
        (90_000, "2 min"),
        (3_540_000, "59 min"),
        (3_570_000, "1 h"),
        (5_400_000, "1 h 30 min"),
        (47.5 * pdf.HR, "47 h 30 min"),
        (48 * pdf.HR, "2 d"),
        (77 * pdf.HR, "3 d 5 h"),
        (-90_000, "-2 min"),
    ],
)
def test_durations_read_in_their_largest_units(ms, want):
    assert fmt_dur(ms) == want


@pytest.mark.parametrize(
    "ms, written, on_the_page",
    [
        (2_500, "2 s", "3 s"),
        (150_000, "2 min", "3 min"),
        (50.5 * pdf.HR, "2 d 2 h", "2 d 3 h"),
        (-150_000, "-2 min", "−3 min"),
    ],
)
def test_durations_halfway_between_units_read_unlike_the_page(ms, written, on_the_page):
    """
    BUG (current behaviour asserted): fmt_dur means to write a duration as the
    page's fmtDur does, but python's round() takes a half to the even side where
    javascript's Math.round takes it up, so a stay of exactly 2 d 2 h 30 min
    spans 2 d 2 h in the pdf and 2 d 3 h on the page; and the pdf signs with a
    hyphen where the page uses the minus sign, which Gotham has
    """
    assert fmt_dur(ms) == written != on_the_page


# ------------------------------------------------------------------ the layout


def test_a_timeline_pdf_is_well_formed(pipeline):
    data = timeline_pdf(payload_of(pipeline))
    assert check_structure(data) == len(page_texts(data)) >= 3


def test_the_first_page_shows_the_subject_and_every_lane(pipeline):
    p = payload_of(pipeline)
    first = page_texts(timeline_pdf(p))[0]
    assert f"Subject {p['subject']['subject_id']}" in first
    assert f"Times in {p['meta']['timezone']}" in first
    for lane in p["lanes"]:  # names may be cut short to fit the gutter
        assert any(lane["name"].startswith(w.rstrip("…")) for w in first.split("\n"))
    assert "Page 1 of" in first


def test_every_code_and_every_event_is_listed(pipeline):
    p = payload_of(pipeline)
    text = "\n".join(page_texts(timeline_pdf(p)))
    assert f"{len(p['events']['kind']):,} events" in text
    listed = set(text.split("\n"))
    for c in p["codes"]:
        assert c["code"] in listed or c["code"][:30] in text, c["code"]
    for k in range(len(p["events"]["kind"])):  # the table's token column
        assert f"{p['events']['first_token'][k]:,}" in listed


def test_the_events_table_stops_at_its_cap(pipeline, monkeypatch):
    monkeypatch.setattr(pdf, "ROW_CAP", 5)
    p = payload_of(pipeline)
    n = len(p["events"]["kind"])
    text = "\n".join(page_texts(timeline_pdf(p)))
    assert f"the first 5 of {n:,} events; the html page lists them all" in text


@pytest.mark.parametrize("shown", [False, True])
def test_the_winnowing_summary_follows_show_winnowing(pipeline, shown):
    sid = pipeline.inference()["subject_id"][0]
    v = Visualizer(processed_data_home=pipeline.path, show_winnowing=shown)
    data = timeline_pdf(v.get_payload(sid))
    text = "\n".join(page_texts(data))
    assert ("Winnowed for inference" in text) is shown
    marks = [r.text for r in runs(data)[0] if r.size == 6]  # on the overview's axis
    assert ("future" in marks) is shown  # the axis marks the threshold


def test_ticks_fall_on_round_times_in_the_data_zone(pipeline):
    p = payload_of(pipeline)
    p["meta"]["timezone"] = "America/Chicago"
    t = TimelinePdf(p)
    assert t.ticks
    for ms in t.ticks:
        w = t.wall(ms)
        assert w.utcoffset() in (
            datetime.timedelta(hours=-5),
            datetime.timedelta(hours=-6),
        )
        assert (w.minute, w.second) == (0, 0)


def test_an_unknown_zone_falls_back_to_utc(pipeline):
    p = payload_of(pipeline)
    p["meta"]["timezone"] = "Not/AZone"
    assert TimelinePdf(p).zone_name == "UTC"


def test_a_timeline_without_events_still_makes_a_page():
    p = {
        "meta": {"cocoa_version": "0", "timezone": "UTC"},
        "subject": {"subject_id": "s1", "fields": []},
        "lanes": [],
        "codes": [],
        "times": {"ms": [], "label": []},
        "events": {},
        "tokens": {"id": [], "event": [], "vocab": {}},
        "winnowed": None,
    }
    data = timeline_pdf(p)
    assert check_structure(data) == 1
    assert "This subject's timeline has no events." in page_texts(data)[0]


def test_the_header_states_the_subject_s_facts_and_fields(pipeline):
    sid = pipeline.subjects_in_split("held_out")[0]
    p = Visualizer(processed_data_home=pipeline.path).get_payload(sid)
    first = runs(timeline_pdf(p))[0]
    title, badge = first[:2]
    assert (title.text, title.font, title.color) == (
        f"Subject {sid}",
        "F2",
        rgb(pdf.ACCENT),
    )
    assert badge.text == "held_out"
    sub, ev = p["subject"], p["events"]
    start, end = sub["start_time"], sub["end_time"]
    stamps = {t for k, t in zip(ev["kind"], ev["time"]) if k in ("event", "unk")}
    assert defined(first) == {
        "Start": start["label"],
        "End": end["label"],
        "Span": fmt_dur(end["ms"] - start["ms"]),
        "Tokens": f"{len(p['tokens']['id']):,}",
        "Events": f"{len(stamps):,}",  # instants, as the page counts them
        "Distinct codes": f"{len(p['codes']):,}",
        **{f["label"]: f["value"] for f in sub["fields"] if f["value"] is not None},
    }


def test_subject_fields_flow_onto_further_rows_and_pages(pipeline):
    p = payload_of(pipeline)
    fields = [
        {"name": f"f{k}", "label": f"Field {k}", "value": f"category {k}"}
        for k in range(300)
    ]
    p["subject"]["fields"] = [*fields, {"name": "x", "label": "Unknown", "value": None}]
    data = timeline_pdf(p)
    check_layout(data)
    first, second = runs(data)[:2]
    shown = defined(first) | defined(second)
    assert {f["label"]: shown.get(f["label"]) for f in fields} == {
        f["label"]: f["value"] for f in fields
    }
    assert "Unknown" not in shown  # a field without a value is left out
    assert len({r.y for r in first if r.text.startswith("Field ")}) > 1
    assert defined(first).keys() & defined(second).keys() == set()
    assert any(r.text.startswith("Field ") for r in second)  # the rest, overleaf


@pytest.mark.parametrize(
    "stay, labels",
    [
        ("weeks", [("Jan 4", "2024"), ("Jan 11", ""), ("Jan 18", "")]),
        (
            "months",
            [("Dec", "2023"), ("Jan", "2024"), ("Feb", ""), ("Mar", ""), ("Apr", "")],
        ),
        (
            "years",
            [("2016", ""), ("2018", ""), ("2020", ""), ("2022", ""), ("2024", "")],
        ),
    ],
)
def test_ticks_step_by_days_months_or_years_over_longer_timelines(
    heart_rates, stay, labels
):
    p = heart_rates.get_payload(stay)
    t = TimelinePdf(p)
    assert t.zone_name == "America/Chicago"
    assert t.tick_labels == labels
    for ms in t.ticks:  # midnight in the data's zone, on the 1st past weeks
        w = t.wall(ms)
        assert (w.hour, w.minute, w.second) == (0, 0, 0)
        assert w.day == 1 or stay == "weeks"
    shown = page_texts(timeline_pdf(p))[0].split("\n")
    assert all(s in shown for tick in labels for s in tick if s)


def test_a_stay_ending_at_a_sentinel_date_is_ticked_by_decades_up_to_a_cap(runner):
    """
    as records put one down for an end not known: past spans of some 66 years
    the axis steps by decades however long the span, as the page's does, so the
    labels of a span of millennia crowd; both stop after 500 steps
    """
    at = datetime.datetime(2024, 1, 1, 8)
    cfg = synth.minimal_collation_cfg()
    cfg["entries"].append(
        {
            "table": "REFERENCE",
            "prefix": "DSCG",
            "code": "discharge_category",
            "time": "discharge_dttm",
        }
    )
    stay = {"hospitalization_id": "H", "admission_dttm": at}
    processed = runner.minimal(
        hospitalizations=[{**stay, "discharge_dttm": datetime.datetime(9999, 1, 1)}],
        vitals=[{"hospitalization_id": "H", "recorded_dttm": at, "vital_value": 80.0}],
        collation=cfg,
    )
    p = Visualizer(processed_data_home=processed.path).get_payload("H")
    assert p["times"]["label"] == ["2024-01-01 08:00:00 UTC", "9999-01-01 00:00:00 UTC"]
    t = TimelinePdf(p)
    walls = [t.wall(ms) for ms in t.ticks]
    # from 1900, the decade the padded plot opens in, though 1900 itself is off it
    assert [w.year for w in walls] == list(range(1910, 6900, 10))
    assert {(w.month, w.day, w.hour, w.minute) for w in walls} == {(1, 1, 0, 0)}
    assert t.tick_labels == [(str(w.year), "") for w in walls]
    assert check_structure(timeline_pdf(p)) >= 3


def test_lanes_are_drawn_in_their_names_and_colors(pipeline, lanes_cfg):
    """the swatch, marks, and table rows of a lane wear its color"""
    sid = pipeline.tokens_times["subject_id"][0]
    p = Visualizer(lanes_cfg, processed_data_home=pipeline.path).get_payload(sid)
    data = timeline_pdf(p)
    pages, texts = streams(data), runs(data)
    lanes = [(la["name"], rgb(la["color"])) for la in p["lanes"]]
    # the overview: a lane's swatch and name in the gutter, and the marks of its
    # events across the plot, centered on its row
    swatches = [f for f in fills(pages[0]) if (f.x, f.w, f.h) == (pdf.MARGIN, 7, 7)]
    names = [r for r in texts[0] if r.x == pdf.MARGIN + 11 and r.size == 8]
    assert [(r.text, f.color) for r, f in zip(names, swatches, strict=True)] == lanes
    marks = {
        round((s.paths[0][0][1] + s.paths[0][1][1]) / 2, 1): s.color
        for s in strokes(pages[0])
        if (s.width, s.cap) == (1, 0)
    }
    assert [marks.get(round(f.y + 3.5, 1)) for f in swatches] == [c for _, c in lanes]
    drawn = dict(lanes)
    assert drawn["Heart and vessels"] == rgb("#A4343A")
    assert drawn["Stay"] == rgb(LANES["other_color"])  # a lane without a color
    assert "Lab result" in drawn  # a prefix's name, for the lane of its own
    assert {rgb(c) for c in LANES["palette"]} <= set(drawn.values())
    # the events table: a swatch and the name of each event's lane
    lane_of = [p["codes"][c]["lane"] for c in p["events"]["code"]]
    table = next(k for k, t in enumerate(texts) if t[0].text == "Events")
    rows = []
    for page, text in zip(pages[table:], texts[table:]):
        at = {(r.x, r.y): r.text for r in text}
        for f in fills(page):
            if (f.w, f.h) == (5, 5):
                rows.append((at[round(f.x + 8, 2), round(f.y + 4.6, 2)], f.color))
    assert rows == [lanes[j] for j in lane_of if j is not None]


def heading(texts: list[list[Run]], title: str) -> tuple[int, Run]:
    """the page that a section opens on, and its heading there"""
    return next(
        (k, r)
        for k, t in enumerate(texts)
        for r in t
        if (r.text, r.font, r.size) == (title, "F2", 11)
    )


def every_code(data: bytes) -> list[Run]:
    """the text of the pages that give every code a row"""
    texts = runs(data)
    start, end = heading(texts, "Every code")[0], heading(texts, "Events")[0]
    return [r for t in texts[start:end] for r in t]


def test_codes_show_their_prefix_where_their_lane_holds_several(pipeline, lanes_cfg):
    sid = pipeline.tokens_times["subject_id"][0]
    p = Visualizer(lanes_cfg, processed_data_home=pipeline.path).get_payload(sid)
    stay = [la["name"] for la in p["lanes"]].index("Stay")
    several = [c["prefix"] for c in p["codes"] if c["lane"] == stay]
    assert len(set(several)) > 1
    # a bare prefix, as collation wrote for a null code before 26.6.0, has no name
    bare = next(c for c in p["codes"] if c["prefix"] == "VTL")
    bare.update(code="VTL", name="")
    gutter = [r for r in every_code(timeline_pdf(p)) if r.x == pdf.MARGIN + 11]
    shown = [r.text for r in gutter if r.size == 6]
    assert collections.Counter(shown) == collections.Counter([*several, "VTL"])


@pytest.mark.parametrize("stay", ["few", "low"])
def test_binned_values_are_drawn_at_the_height_of_their_quantile(heart_rates, stay):
    """of the tokenizer's n_bins, even where a timeline reaches only the lowest"""
    p = heart_rates.get_payload(stay)
    n_bins = p["meta"]["n_bins"]
    bins = [b for b in p["events"]["bin"] if b is not None]
    assert n_bins == 10 and (max(bins) == 0) is (stay == "low")
    data = timeline_pdf(p)
    text = every_code(data)
    q = [r for r in text if r.size == 5.5]  # the quantiles at the row's top and foot
    assert [r.text for r in q] == [f"Q{n_bins - 1}", "Q0"]
    top, h = q[0].y - 7, pdf.BIN_H
    page = next(s for s, t in zip(streams(data), page_texts(data)) if "Q0" in t)
    (dots,) = [s for s in strokes(page) if s.width == 4.4]
    want = [top + h - 4 - b / (n_bins - 1) * (h - 8) for b in bins]
    assert [a[1] for a, _ in dots.paths] == pytest.approx(want, abs=0.01)


@pytest.mark.parametrize("stay, dots", [("few", 4.4), ("some", 2.6), ("burst", None)])
def test_a_binned_row_draws_dots_then_ranges_as_it_fills(heart_rates, stay, dots):
    """
    a line through its values' dots, haloed while there is room for them; and
    past two values a point of the plot, each half-point column's range instead
    """
    p = heart_rates.get_payload(stay)
    n = sum(b is not None for b in p["events"]["bin"])
    color = rgb(LANES["lanes"][0]["color"])
    data = timeline_pdf(p)
    page = next(s for s, t in zip(streams(data), page_texts(data)) if "Q0" in t)
    drawn = {(s.color, s.width): s for s in strokes(page) if s.cap == 1}
    halo = drawn.get((rgb("#FFFFFF"), 6))
    line = drawn.get((rgb(pdf.mix(LANES["lanes"][0]["color"], 0.45)), 0.8))
    (guides,) = [  # across the row at its top and lowest quantiles
        s
        for s in strokes(page)
        if s.color == rgb(pdf.GRID_SOFT) and len(s.paths) == 2
        if all(a[1] == b[1] for a, b in s.paths)
    ]
    top, foot = sorted(a[1] for a, _ in guides.paths)
    if dots:
        assert line.join and len(line.paths) == 1 and len(line.paths[0]) == n
        marks = drawn[(color, dots)].paths
        assert len(marks) == n and all(a == b for a, b in marks)
        assert (halo is not None) is (dots == 4.4)
    else:
        assert line is None and halo is None
        cols = drawn[(color, 1)].paths
        xs = [a[0] for a, _ in cols]
        assert all(a[0] == b[0] for a, b in cols) and len(set(xs)) == len(xs) < n
        assert all(top <= a[1] <= b[1] <= foot for a, b in cols)
        assert max(b[1] - a[1] for a, b in cols) > 0


@pytest.mark.parametrize("meds", [True, False])
def test_the_events_table_has_a_value_column_only_where_values_are_known(
    pipeline, tmp_path, meds
):
    """
    timelines tokenized without include_numeric_values, as by default, take
    their values from meds.parquet, and without it have none to show
    """
    home = pipeline.path
    if not meds:
        skip = shutil.ignore_patterns("meds.parquet")
        home = shutil.copytree(home, tmp_path / "processed", ignore=skip)
    sid = pipeline.tokens_times["subject_id"][0]
    p = Visualizer(processed_data_home=home).get_payload(sid)
    assert p["meta"]["has_numeric_values"] is meds
    texts = runs(timeline_pdf(p))
    k = heading(texts, "Events")[0]
    heads = [r for r in texts[k] if (r.font, r.size) == ("F2", 6.5)]
    names = ["#", "Time", "Lane", "Code", "Bin", "Value", "Text", "Token"]
    assert [r.text for r in heads] == [s for s in names if meds or s != "Value"]
    if meds:  # set flush right, to six significant digits
        (value,) = [r for r in heads if r.text == "Value"]
        edge = value.x + measure("Value", 6.5, "F2")
        cells = [
            r.text
            for t in texts[k:]
            for r in t
            if (r.font, r.size) == ("F1", 6.5)
            and abs(r.x + measure(r.text, 6.5) - edge) < 0.01
        ]
        assert cells == [fmt_num(v, 6) for v in p["events"]["value"] if v is not None]
        assert cells


# ------------------------------------------------------------------ winnowing


@pytest.mark.parametrize(
    "cfg, stated",
    [
        ({}, SUMMARY[:3]),  # the shipped threshold, a day after the first event
        # a past with nothing in it has no last token to give the time of
        ({"threshold": {"duration_s": 0}}, SUMMARY[1:3]),
        # a horizon is stated where it cuts the future short
        ({"horizon_after_threshold_s": 6 * 3600}, SUMMARY),
    ],
)
def test_the_winnowing_summary_states_where_the_timeline_was_cut(
    runner, pipeline, cfg, stated
):
    p = winnowed(runner, pipeline, **cfg)
    w8, n_t = p["winnowed"], len(p["tokens"]["id"])
    n_past = min(w8["last_valid"], n_t)
    texts = runs(timeline_pdf(p))
    k, head = heading(texts, "Winnowed for inference")
    (note,) = [r for r in texts[k] if r.text == w8["file"]]  # the file it came from
    assert note.y == head.y
    right = note.x + measure(note.text, 7)
    assert right == pytest.approx(pdf.PAGE_W - pdf.MARGIN, abs=0.01)
    facts = {s: v for t in texts for s, v in defined(t).items() if s in SUMMARY}
    assert tuple(facts) == stated
    assert facts["Past tokens"] == f"{n_past:,}"
    assert facts["Future tokens"] == f"{n_t - n_past:,}"
    if n_past:  # the time of the past's last token
        e = p["tokens"]["event"][n_past - 1]
        assert facts["Threshold"] == p["times"]["label"][p["events"]["time"][e]]
    if "Within the horizon" in stated:
        assert facts["Within the horizon"] == f"{w8['n_future']:,}"
        assert w8["n_future"] < n_t - n_past


@pytest.mark.parametrize("outcomes", [None, ["BOS", "EOS"]])
def test_the_winnowing_summary_flags_each_outcome_on_its_side(
    runner, pipeline, outcomes
):
    """
    an outcome found on either side gets a row, its sides set in bold, and the
    rest share a line; this timeline lacks most of the shipped ones, but every
    timeline kept opens before its threshold and ends after it, so BOS and EOS
    are never absent
    """
    cfg = {} if outcomes is None else {"outcome_tokens": outcomes}
    p = winnowed(runner, pipeline, **cfg)
    found = p["winnowed"]["outcomes"]
    flagged = [o for o in found if o["past"] or o["future"]]
    absent = [o["code"] for o in found if not (o["past"] or o["future"])]
    assert flagged and bool(absent) is (outcomes is None)
    texts = runs(timeline_pdf(p))

    def side(on: bool) -> tuple[str, str]:
        return ("F2", rgb(pdf.INK)) if on else ("F1", rgb(pdf.MUTED))

    rows = []
    for t in texts:
        at = {(r.x, r.y): r for r in t}
        for r in filter(is_outcome, t):
            past, future = (at[pdf.MARGIN + dx, r.y] for dx in (240, 270))
            assert (past.text, future.text) == ("past", "future")
            rows.append((r.text, (past.font, past.color), (future.font, future.color)))
    assert rows == [(o["code"], side(o["past"]), side(o["future"])) for o in flagged]
    line = " ".join(r.text for t in texts for r in t if is_absent(r))
    assert line == ("Absent from both: " + ", ".join(absent) if absent else "")


@pytest.mark.parametrize(
    "threshold, marked",
    [
        ({"duration_s": 86400}, ["past", "future", "BOS", "EOS"]),
        # at the first instant, past has no room before the plot, and BOS yields
        ({"duration_s": 0}, ["future", "EOS"]),
        # at the last, future would run off the plot, and EOS yields
        ({"first_occurrence": "EOS"}, ["past", "BOS"]),
    ],
)
def test_the_axis_marks_the_threshold_over_the_ends_of_the_timeline(
    runner, pipeline, threshold, marked
):
    """and the overview washes the future after it"""
    p = winnowed(runner, pipeline, threshold=threshold)
    data = timeline_pdf(p)
    page, first = streams(data)[0], runs(data)[0]
    marks = {r.text: r for r in first if r.size == 6}
    assert list(marks) == marked
    x = TimelinePdf(p).X(threshold_ms(p))
    if "past" in marks:  # set up to the threshold, as future is from it
        assert marks["past"].x + measure("past", 6) == pytest.approx(x - 3, abs=0.01)
    if "future" in marks:
        assert marks["future"].x == pytest.approx(x + 3, abs=0.01)
    rules = [s for s in strokes(page) if (s.color, s.width) == (rgb(pdf.INK2), 0.75)]
    assert len(rules) == 2  # a tick on the axis, and a rule down the lanes
    assert {a[0] for s in rules for a in s.paths[0]} == {round(x, 2)}
    (future,) = [f for f in fills(page) if f.color == rgb(pdf.FUTURE)]
    assert future.x == round(x, 2)
    assert future.x + future.w == pytest.approx(pdf.PLOT_X + pdf.PLOT_W, abs=0.01)


@pytest.mark.parametrize("outcomes", [None, ["*"]])
def test_every_code_follows_a_winnowing_summary_onto_the_page_it_ends(
    runner, pipeline, outcomes
):
    """
    a summary that fits under the overview leaves every code to open the next
    page; one that runs on, as it does with every word an outcome, has every
    code follow it on the page it ends on, rather than leave the rest blank
    """
    cfg = {} if outcomes is None else {"outcome_tokens": outcomes}
    data = timeline_pdf(winnowed(runner, pipeline, **cfg))
    check_layout(data)
    texts = runs(data)
    assert heading(texts, "Winnowed for inference")[0] == 0  # under the overview
    rows = {j for j, t in enumerate(texts) for r in t if is_outcome(r)}
    lines = [(j, r) for j, t in enumerate(texts) for r in t if is_absent(r)]
    k, head = heading(texts, "Every code")
    if outcomes is None:
        assert rows == {j for j, _ in lines} == {0}
        assert (k, head.y) == (1, pdf.MARGIN + 11)
    else:
        assert len(rows) > 1  # the outcomes' rows run over several pages
        last_page, last = lines[-1]
        assert k == last_page >= max(rows) and head.y > last.y
