#!/usr/bin/env python3

"""the static pdf: a well-formed file, its text, and the timeline it lays out"""

import datetime
import re
import zlib

import pytest

import cocoa.pdf as pdf
from cocoa.pdf import Document, TimelinePdf, clip, measure, rgb, timeline_pdf, wrap
from cocoa.visualizer import Visualizer

TJ = re.compile(rb"\(((?:\\.|[^\\)])*)\) Tj")


def streams(data: bytes) -> list[bytes]:
    """each page's content stream, inflated"""
    found = re.findall(rb"stream\n(.*?)\nendstream", data, re.S)
    return [zlib.decompress(s) for s in found]


def page_texts(data: bytes) -> list[str]:
    """the text set on each page, one string per page"""
    out = []
    for s in streams(data):
        words = [re.sub(rb"\\(.)", rb"\1", m) for m in TJ.findall(s)]
        out.append("\n".join(w.decode("cp1252", "replace") for w in words))
    return out


def check_structure(data: bytes) -> int:
    """assert the xref table points at every object; return the page count"""
    assert data.startswith(b"%PDF-1.4\n") and data.endswith(b"%%EOF\n")
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
    for key in (b"/URI", b"/Launch", b"/GoToR", b"/FontFile", b"/EmbeddedFile"):
        assert key not in data
    fonts = set(re.findall(rb"/BaseFont /([\w-]+)", data))
    assert fonts == {b"Helvetica", b"Helvetica-Bold", b"Courier"}  # every reader's


def test_text_is_escaped_and_what_winansi_lacks_becomes_a_question_mark():
    doc = Document("t")
    doc.new_page().text(0, 10, "a (b) \\ °C ≥ 6.5\nok")
    (s,) = streams(doc.to_bytes())
    assert b"(a \\(b\\) \\\\ \xb0C ? 6.5 ok) Tj" in s  # \xb0 is cp1252's °


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
    text = "\n".join(page_texts(timeline_pdf(v.get_payload(sid))))
    assert ("Winnowed for inference" in text) is shown
    assert ("future" in text.split("\n")) is shown  # the axis marks the threshold


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
