#!/usr/bin/env python3

"""
draws a subject's timeline, or the vocabulary, as a static, self-contained pdf,
from the payload the visualizer builds for its html page
"""

import collections
import collections.abc
import datetime
import functools
import importlib.resources as resources
import math
import pathlib
import struct
import zlib
import zoneinfo
from typing import Any, NamedTuple

# a landscape us letter page in points, laid out from its top-left corner
PAGE_W, PAGE_H = 792.0, 612.0
MARGIN = 36.0
BOTTOM = PAGE_H - 40  # where content stops, leaving room for the footer
GUTTER = 150.0  # lane and code labels, left of the plot
PLOT_X = MARGIN + GUTTER + 8
PLOT_W = PAGE_W - MARGIN - PLOT_X
AXIS_H = 36.0
LANE_H, CODE_H, BIN_H = 18.0, 13.0, 28.0
ROW_H = 9.5  # a row of a table
ROW_CAP = 2000  # events, or words, a table lists, as the timeline page's table does
GAP = 6.0  # between table columns
BINNED_CAP = 500  # binned codes the vocabulary draws
HIST_W = 280.0  # a binned code's histogram, left of its bins
BIN_ROW_H = 8.5  # a row of a binned code's bins
METER = 36.0  # the bar of a share
CHART_H = 64.0  # a histogram of the timelines' lengths or durations

# the page's colors, its translucent washes flattened onto white
INK, INK2, MUTED, AXIS = "#000000", "#737373", "#A6A6A6", "#A6A6A6"
GRID, GRID_SOFT, WASH, FUTURE = "#D9D9D9", "#ECECEC", "#FAFAFA", "#F6F6F6"
BAND = "#F6F6F6"  # every other bin of a histogram
ACCENT = "#800000"  # maroon, for the title and footer only

MIN, HR, DAY = 6e4, 36e5, 864e5
MON = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
# axis steps as (ms, months, approximate ms), as the page steps them
STEPS = [
    (ms, None, ms)
    for ms in (1e3, 5e3, 15e3, 3e4, MIN, 5 * MIN, 15 * MIN, 30 * MIN, HR, 2 * HR)
    + (3 * HR, 6 * HR, 12 * HR, DAY, 2 * DAY, 7 * DAY, 14 * DAY)
] + [(None, mo, mo * 30.44 * DAY) for mo in (1, 3, 6, 12, 24, 60, 120)]
EPOCH = datetime.datetime(1970, 1, 1)
UTC_EPOCH = EPOCH.replace(tzinfo=datetime.timezone.utc)
# where a duration histogram ticks, where at least two of these fall on it
HOURS = (
    (1 / 60, "1 min"),
    (1 / 6, "10 min"),
    (1, "1 h"),
    (6, "6 h"),
    (24, "1 d"),
    (168, "1 w"),
    (730.5, "1 mo"),
    (8766, "1 y"),
    (87660, "10 y"),
)
STRUCT = {"BOS", "EOS", "UNK", "TIME", "CLCK"}  # prefixes of no code's

# the page's own fonts, embedded whole so that the pdf is set as the page is; one
# whose file is missing falls back to the pdf standard font beside it, which every
# reader has. codes and tokens are set in courier, as no monospace font is packaged
EMBEDDED = {"F1": "Gotham-Book.otf", "F2": "Gotham-Medium.otf"}
FONTS = {"F1": "Helvetica", "F2": "Helvetica-Bold", "F3": "Courier"}
# the standard fonts' widths are thousandths of the font size, from their afm
# metrics for codes 32 to 126
HELVETICA = tuple(
    map(
        int,
        """
        278 278 355 556 556 889 667 191 333 333 389 584 278 333 278 278 556 556
        556 556 556 556 556 556 556 556 278 278 584 584 584 556 1015 667 667 722
        722 667 611 778 722 278 500 667 556 833 722 778 667 778 722 667 611 722
        667 944 667 667 611 278 278 278 469 556 333 556 556 500 556 556 278 556
        556 222 222 500 222 833 556 556 556 556 333 500 278 556 500 722 500 500
        500 334 260 334 584
        """.split(),
    )
)
HELVETICA_BOLD = tuple(
    map(
        int,
        """
        278 333 474 556 556 889 722 238 333 333 389 584 278 333 278 278 556 556
        556 556 556 556 556 556 556 556 333 333 584 584 584 611 975 722 722 722
        722 667 611 778 722 278 556 722 611 833 722 778 667 778 722 667 611 722
        667 944 667 667 611 333 278 333 584 556 333 556 611 556 611 556 333 611
        611 278 278 556 278 889 611 611 611 611 389 556 333 611 556 778 556 556
        500 389 280 389 584
        """.split(),
    )
)
WIDE = {0x85: 1000, 0x96: 556, 0x97: 1000, 0xB0: 400, 0xB7: 278, 0xD7: 584}
# maps an embedded font's glyphs back to text, so that a reader can copy it
TO_UNICODE = """/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def
/CMapName /Adobe-Identity-UCS def
/CMapType 2 def
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
{}endcmap
CMapName currentdict /CIDInit /ProcSet findresource exch defineresource pop
end
end
"""


def read_cmap(data: bytes, at: int) -> dict[int, int]:
    """code points to glyph ids, from the unicode subtable of the cmap at `at`"""
    n = struct.unpack_from(">H", data, at + 2)[0]
    subtables = {
        (platform, encoding): at + offset
        for platform, encoding, offset in struct.iter_unpack(
            ">HHI", data[at + 4 : at + 4 + 8 * n]
        )
    }
    s = subtables.get((3, 1)) or subtables[(0, 3)]  # format 4, the bmp
    seg = struct.unpack_from(">H", data, s + 6)[0] // 2
    ends = struct.unpack_from(f">{seg}H", data, s + 14)
    starts = struct.unpack_from(f">{seg}H", data, s + 16 + 2 * seg)
    deltas = struct.unpack_from(f">{seg}h", data, s + 16 + 4 * seg)
    ranges = s + 16 + 6 * seg
    offsets = struct.unpack_from(f">{seg}H", data, ranges)
    cmap = {}
    for k, (start, end, delta, offset) in enumerate(zip(starts, ends, deltas, offsets)):
        for c in range(start, min(end, 0xFFFE) + 1):
            if offset:
                at_glyph = ranges + 2 * k + offset + 2 * (c - start)
                g = struct.unpack_from(">H", data, at_glyph)[0]
                g = (g + delta) & 0xFFFF if g else 0
            else:
                g = (c + delta) & 0xFFFF
            if g:
                cmap[c] = g
    return cmap


class Face:
    """an opentype font, read for what setting and embedding text in it takes"""

    def __init__(self, data: bytes, name: str) -> None:
        self.data, self.name = data, name
        n = struct.unpack_from(">H", data, 4)[0]
        tables = {
            tag: offset
            for tag, _, offset, _ in struct.iter_unpack(
                ">4sIII", data[12 : 12 + 16 * n]
            )
        }

        def read(tag: bytes, fmt: str, at: int = 0) -> tuple[int, ...]:
            return struct.unpack_from(">" + fmt, data, tables[tag] + at)

        scale = 1000 / read(b"head", "H", 18)[0]  # font units to thousandths
        self.bbox = [round(v * scale) for v in read(b"head", "4h", 36)]
        self.ascent, self.descent = (round(v * scale) for v in read(b"hhea", "2h", 4))
        n_metrics, n_glyphs = read(b"hhea", "H", 34)[0], read(b"maxp", "H", 4)[0]
        advances = read(b"hmtx", "Hh" * n_metrics)[::2]
        advances += advances[-1:] * (n_glyphs - n_metrics)  # the rest share the last
        self.widths = [round(a * scale) for a in advances]
        version, _, weight = read(b"OS/2", "HhH")
        cap = read(b"OS/2", "h", 88)[0] if version >= 2 else None
        self.cap_height = round(cap * scale) if cap else self.ascent
        self.stem_v = 50 + round((weight / 65) ** 2)  # the usual guess from weight
        self.italic_angle = read(b"post", "i", 4)[0] / 65536
        self.cmap = read_cmap(data, tables[b"cmap"])
        self.missing = self.cmap.get(ord("?"), 0)

    def glyphs(self, s: str) -> list[int]:
        """the glyph of each character of `s`; what the font lacks becomes ?"""
        return [self.cmap.get(ord(c) if c >= " " else 32, self.missing) for c in str(s)]

    @functools.cached_property
    def program(self) -> bytes:
        return zlib.compress(self.data, 9)

    @functools.cached_property
    def to_unicode(self) -> bytes:
        first: dict[int, int]
        first = {}  # a glyph that several code points share copies as the first
        for c, g in sorted(self.cmap.items()):
            first.setdefault(g, c)
        pairs = [
            f"<{g:04X}> <{chr(c).encode('utf-16-be').hex().upper()}>"
            for g, c in sorted(first.items())
        ]
        blocks = "".join(  # a block holds at most 100
            f"{len(pairs[i : i + 100])} beginbfchar\n"
            + "\n".join(pairs[i : i + 100])
            + "\nendbfchar\n"
            for i in range(0, len(pairs), 100)
        )
        return zlib.compress(TO_UNICODE.format(blocks).encode())


@functools.cache
def face(font: str) -> Face | None:
    """the packaged font that `font` embeds, or None to use a standard font"""
    file = EMBEDDED.get(font)
    path = resources.files("cocoa.assets") / "fonts" / file if file else None
    if path is None or not path.is_file():
        return None
    return Face(path.read_bytes(), file.removesuffix(".otf"))


def to_ansi(s: str) -> bytes:
    """text in WinAnsiEncoding, one byte a character; what it lacks becomes ?"""
    return "".join(c if c >= " " else " " for c in str(s)).encode("cp1252", "replace")


def char_widths(s: str, font: str) -> list[int]:
    """the advance of each character of `s` in `font`, in thousandths of its size"""
    if f := face(font):
        return [f.widths[g] for g in f.glyphs(s)]
    b = to_ansi(s)
    if font == "F3":
        return [600] * len(b)
    table = HELVETICA_BOLD if font == "F2" else HELVETICA
    return [table[c - 32] if 32 <= c <= 126 else WIDE.get(c, 556) for c in b]


def encode(s: str, font: str) -> str:
    """`s` as a pdf string in `font`: glyph ids if it is embedded, else winansi"""
    if f := face(font):
        return "<" + "".join(f"{g:04X}" for g in f.glyphs(s)) + ">"
    return literal(to_ansi(s))


def measure(s: str, size: float, font: str = "F1") -> float:
    """the width of `s` set in `font` at `size`, in points"""
    return sum(char_widths(s, font)) * size / 1000


def clip(s: str, size: float, font: str, width: float) -> str:
    """`s`, cut short with an ellipsis if it is wider than `width`"""
    s = str(s)
    ws = char_widths(s, font)
    if sum(ws) * size / 1000 <= width:
        return s
    room, n, acc = width * 1000 / size - char_widths("…", font)[0], 0, 0
    for w in ws:
        if acc + w > room:
            break
        acc, n = acc + w, n + 1
    return s[:n] + "…" if n else ""


def wrap(s: str, size: float, font: str, width: float) -> list[str]:
    """`s` broken at spaces into lines no wider than `width`"""
    lines, line = [], ""
    for word in s.split(" "):
        longer = f"{line} {word}" if line else word
        if line and measure(longer, size, font) > width:
            lines.append(line)
            line = word
        else:
            line = longer
    return lines + [line] if line else lines


def literal(b: bytes) -> str:
    """bytes as a pdf string literal, carried through latin-1"""
    escaped = b.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")
    return "(" + escaped.decode("latin-1") + ")"


def utf16(s: str) -> bytes:
    """a pdf text string that holds any unicode, for document metadata"""
    return b"<FEFF" + s.encode("utf-16-be").hex().upper().encode() + b">"


def stream(data: bytes, entries: str) -> bytes:
    """a pdf stream of `data`, its dictionary holding `entries` too"""
    head = f"<< /Length {len(data)} {entries} >>\nstream\n"
    return head.encode() + data + b"\nendstream"


def add_font(font: str, add: collections.abc.Callable[[bytes | str], int]) -> int:
    """the objects of `font`, each given to `add`; returns the font's number"""
    f = face(font)
    if f is None:
        return add(
            f"<< /Type /Font /Subtype /Type1 /BaseFont /{FONTS[font]} "
            "/Encoding /WinAnsiEncoding >>"
        )
    program = add(stream(f.program, "/Subtype /OpenType /Filter /FlateDecode"))
    descriptor = add(
        f"<< /Type /FontDescriptor /FontName /{f.name} /Flags 4 "
        f"/FontBBox [{' '.join(map(str, f.bbox))}] /ItalicAngle {f.italic_angle:g} "
        f"/Ascent {f.ascent} /Descent {f.descent} /CapHeight {f.cap_height} "
        f"/StemV {f.stem_v} /FontFile3 {program} 0 R >>"
    )
    glyphs = add(  # cids are glyph ids, as its cff is not cid-keyed
        f"<< /Type /Font /Subtype /CIDFontType0 /BaseFont /{f.name} "
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> "
        f"/FontDescriptor {descriptor} 0 R /W [0 [{' '.join(map(str, f.widths))}]] >>"
    )
    to_unicode = add(stream(f.to_unicode, "/Filter /FlateDecode"))
    return add(
        f"<< /Type /Font /Subtype /Type0 /BaseFont /{f.name} /Encoding /Identity-H "
        f"/DescendantFonts [{glyphs} 0 R] /ToUnicode {to_unicode} 0 R >>"
    )


@functools.lru_cache(maxsize=None)
def rgb(color: str) -> str:
    """a #rgb or #rrggbb color as pdf rgb components; anything else is grey"""
    h = str(color).strip().removeprefix("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        if len(h) != 6:
            raise ValueError(color)
        parts = [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    except ValueError:
        parts = [0x73 / 255] * 3
    return " ".join(f"{v:.3f}" for v in parts)


def mix(color: str, alpha: float) -> str:
    """`color` at `alpha` over white, as an opaque color"""
    parts = [float(v) for v in rgb(color).split()]
    return "#" + "".join(f"{round(255 * (1 - alpha + alpha * v)):02X}" for v in parts)


class Page:
    """one page's drawing operations, in points from its top-left corner"""

    def __init__(self) -> None:
        self.ops: list[str] = []

    def rect(self, x: float, y: float, w: float, h: float, color: str) -> None:
        """a filled rectangle"""
        y = PAGE_H - y - h
        self.ops.append(f"{rgb(color)} rg {x:.2f} {y:.2f} {w:.2f} {h:.2f} re f")

    def frame(
        self, x: float, y: float, w: float, h: float, color: str, width: float = 0.5
    ) -> None:
        """an outlined rectangle"""
        y = PAGE_H - y - h
        self.ops.append(
            f"{rgb(color)} RG {width:.2f} w {x:.2f} {y:.2f} {w:.2f} {h:.2f} re S"
        )

    def segments(
        self,
        segs: list[tuple[float, float, float, float]],
        color: str,
        width: float = 1.0,
        round_cap: bool = False,
    ) -> None:
        """
        one stroked path of (x0, y0, x1, y1) segments; with round caps, a
        segment of no length is a dot as wide as the line
        """
        if not segs:
            return
        path = " ".join(
            f"{a:.2f} {PAGE_H - b:.2f} m {c:.2f} {PAGE_H - d:.2f} l"
            for a, b, c, d in segs
        )
        cap = 1 if round_cap else 0
        self.ops.append(f"{rgb(color)} RG {width:.2f} w {cap} J {path} S")

    def polyline(
        self, points: list[tuple[float, float]], color: str, width: float = 1.0
    ) -> None:
        """a stroked line through `points`"""
        if len(points) < 2:
            return
        (x0, y0), *rest = points
        path = f"{x0:.2f} {PAGE_H - y0:.2f} m " + " ".join(
            f"{x:.2f} {PAGE_H - y:.2f} l" for x, y in rest
        )
        self.ops.append(f"{rgb(color)} RG {width:.2f} w 1 J 1 j {path} S")

    def text(
        self,
        x: float,
        y: float,
        s: str,
        size: float = 8.0,
        font: str = "F1",
        color: str = INK,
        align: str = "left",
        width: float | None = None,
    ) -> None:
        """`s` with its baseline at `y`, cut short to fit `width` if given"""
        if width is not None:
            s = clip(s, size, font, width)
        ws = char_widths(s, font)
        if not ws:
            return
        w = sum(ws) * size / 1000
        x -= w if align == "right" else w / 2 if align == "center" else 0
        self.ops.append(
            f"BT {rgb(color)} rg /{font} {size:.2f} Tf "
            f"{x:.2f} {PAGE_H - y:.2f} Td {encode(s, font)} Tj ET"
        )


class Document:
    """pages written out as one pdf, with nothing referenced outside it"""

    def __init__(self, title: str, producer: str = "cocoa") -> None:
        self.title = title
        self.producer = producer
        self.pages: list[Page] = []

    def new_page(self) -> Page:
        self.pages.append(page := Page())
        return page

    def to_bytes(self) -> bytes:
        """the pdf: catalog, page tree, info, fonts, then each page's content and it"""
        objs = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"",  # the page tree, once its pages are numbered
            b"<< /Title "
            + utf16(self.title)
            + b" /Producer "
            + utf16(self.producer)
            + b" >>",
        ]

        def add(obj: bytes | str) -> int:
            """the number of `obj`, added as the next object"""
            objs.append(obj if isinstance(obj, bytes) else obj.encode())
            return len(objs)

        fonts = " ".join(f"/{k} {add_font(k, add)} 0 R" for k in FONTS)
        kids = []
        for page in self.pages:
            data = zlib.compress("\n".join(page.ops).encode("latin-1"))
            contents = add(stream(data, "/Filter /FlateDecode"))
            n = add(
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W:g} {PAGE_H:g}] "
                f"/Resources << /Font << {fonts} >> >> /Contents {contents} 0 R >>"
            )
            kids.append(f"{n} 0 R")
        objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
        objs[1] = objs[1].encode()

        # 1.6, for an embedded opentype font
        out = bytearray(b"%PDF-1.6\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for n, obj in enumerate(objs, 1):
            offsets.append(len(out))
            out += f"{n} 0 obj\n".encode() + obj + b"\nendobj\n"
        xref = len(out)
        out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
        out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
        out += (
            f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R /Info 3 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode()
        return bytes(out)


def fmt_int(n: int) -> str:
    return f"{n:,}"


def fmt_num(v: float | None, digits: int = 4) -> str:
    """a number as the page shows one: to `digits` significant, with commas"""
    if v is None:
        return ""
    r = float(f"{v:.{digits}g}")
    if r and (abs(r) < 1e-3 or abs(r) >= 1e9):
        return f"{r:.{digits}g}"
    return f"{int(r):,}" if r.is_integer() else f"{r:,}"


def fmt_dur(ms: float) -> str:
    """a duration as the page shows one, e.g. 3 d 4 h"""
    sign = "-" if ms < 0 else ""
    s = round(abs(ms) / 1000)
    if s < 60:
        return f"{sign}{s} s"
    m = round(s / 60)
    if m < 60:
        return f"{sign}{m} min"
    if m < 48 * 60:
        return f"{sign}{m // 60} h" + (f" {m % 60} min" if m % 60 else "")
    h = round(m / 60)
    return f"{sign}{h // 24} d" + (f" {h % 24} h" if h % 24 else "")


def fmt_pct(f: float) -> str:
    """a share as the page shows one: to a tenth of a percent below 10%"""
    p = f * 100
    if p == 0:
        return "0%"
    if p < 0.1:
        return "<0.1%"
    return f"{p:.1f}%" if p < 10 else f"{math.floor(p + 0.5)}%"


def fmt_compact(n: float) -> str:
    """a count on an axis, as the page shows one, e.g. 1.5k"""
    for size, unit in ((1e6, "M"), (1e3, "k")):
        if n >= size:
            return f"{float(f'{n / size:.3g}'):g}{unit}"
    return fmt_num(n)


def nice_ceil(v: float) -> float:
    """the least of 1, 2, 2.5, or 5 times a power of ten that is at least `v`"""
    p = 10.0 ** math.floor(math.log10(v))
    return next(m * p for m in (1, 2, 2.5, 5, 10) if m * p >= v)


def nice_ticks(a: float, b: float, n: int) -> list[float]:
    """round values across [a, b], about `n` of them"""
    raw = (b - a) / max(1, n)
    if not raw > 0:
        return []
    p = 10.0 ** math.floor(math.log10(raw))
    step = next(m * p for m in (1, 2, 5, 10) if m * p >= raw)
    out, t = [], math.ceil(a / step) * step
    while t <= b + step * 1e-9:
        out.append(float(f"{t:.12g}"))
        t += step
    return out


def count_ticks(top: float) -> list[float]:
    """ticks up a count axis to `top`, halfway too where that's a whole count"""
    return [0, top] if top % 2 else [0, top / 2, top]


class Cell(NamedTuple):
    """
    a table cell's text, after a swatch of the color `key`, or a meter that
    color and `fill` full; `note` follows it in grey, and `ink` and `font` set
    it in other than its row's and column's
    """

    text: str
    key: str | None = None
    fill: float | None = None
    note: str | None = None
    ink: str | None = None
    font: str | None = None


def lane_cell(lanes: list[dict[str, Any]], j: int | None) -> Cell | str:
    """the name of lane `j` after a swatch of its color, or nothing without one"""
    return Cell(lanes[j]["name"], key=lanes[j]["color"]) if j is not None else ""


def rest_of(cols: list[tuple], width: float = PAGE_W - 2 * MARGIN) -> float:
    """the width left of `width` for a last column after `cols`"""
    return width - sum(c[1] for c in cols) - GAP * len(cols)


class Flow:
    """a document laid out from the top of a page down, onto new pages"""

    def __init__(self, title: str, meta: dict[str, Any]) -> None:
        self.doc = Document(title, f"cocoa {meta.get('cocoa_version', '')}".strip())
        self.page = self.doc.new_page()
        self.y = MARGIN

    def new_page(self) -> None:
        self.page = self.doc.new_page()
        self.y = MARGIN

    def need(self, h: float) -> None:
        """start a new page unless `h` more points fit on this one"""
        if self.y + h > BOTTOM:
            self.new_page()

    def title(self, s: str, badges: collections.abc.Sequence[str] = ()) -> None:
        """the document's title, with a framed badge after it for each of `badges`"""
        p = self.page
        p.text(MARGIN, self.y + 15, s, 17, "F2", ACCENT, width=PAGE_W - 2 * MARGIN)
        x = MARGIN + measure(s, 17, "F2") + 9
        for b in badges:
            w = measure(b, 7.5) + 12
            p.frame(x, self.y + 4, w, 12, AXIS)
            p.text(x + 6, self.y + 13, b, 7.5, color=INK2)
            x += w + 5
        self.y += 26

    def heading(self, title: str, note: str | None = None) -> None:
        self.need(40)
        self.page.text(MARGIN, self.y + 11, title, 11, "F2")
        if note:
            self.page.text(
                PAGE_W - MARGIN, self.y + 11, note, 7, color=INK2, align="right"
            )
        self.y += 20

    def paragraph(self, s: str, size: float = 7.0, color: str = INK2) -> None:
        for line in wrap(s, size, "F1", PAGE_W - 2 * MARGIN):
            self.need(size * 1.5)
            self.page.text(MARGIN, self.y + size, line, size, color=color)
            self.y += size * 1.5

    def definitions(self, pairs: list[tuple[str, str]], bold: bool = True) -> None:
        """label-over-value pairs, flowing in rows across the page"""
        font, x = ("F2" if bold else "F1"), MARGIN
        self.need(24)
        for k, v in pairs:
            w = min(max(measure(k, 6.5), measure(v, 8.5, font)), PAGE_W - 2 * MARGIN)
            if x > MARGIN and x + w > PAGE_W - MARGIN:
                x, self.y = MARGIN, self.y + 22
                self.need(24)
            self.page.text(x, self.y + 7, k, 6.5, color=INK2)
            self.page.text(x, self.y + 17, v, 8.5, font, width=w)
            x += w + 18
        self.y += 26

    def table(
        self,
        cols: list[tuple[str, float, str, str, collections.abc.Callable]],
        items: collections.abc.Iterable[Any],
        left: float = MARGIN,
        row_h: float = ROW_H,
        strong: collections.abc.Callable[[Any], bool] | None = None,
        faint: collections.abc.Callable[[Any], bool] | None = None,
    ) -> None:
        """
        a row per item of `cols`, each (head, width, align, font, and the item's
        cell, as text or a Cell), under their heads, which head every page the
        rows run onto; a `strong` row is set in bold under a rule, and a `faint`
        one in grey
        """
        right = left + sum(c[1] for c in cols) + GAP * (len(cols) - 1)

        def head() -> None:
            x = left
            for name, w, align, *_ in cols:
                at = x + w if align == "right" else x
                self.page.text(at, self.y + 7, name, 6.5, "F2", INK2, align)
                x += w + GAP
            self.y += 10
            self.page.segments([(left, self.y, right, self.y)], GRID, 0.5)

        self.need(10 + row_h)
        head()
        lines: list[tuple[float, float, float, float]] = []
        for item in items:
            if self.y + row_h > BOTTOM:
                self.page.segments(lines, GRID_SOFT, 0.4)
                self.new_page()
                head()
                lines = []
            bold = strong is not None and strong(item)
            if bold:  # its rule, rather than the soft one under the row before
                lines = lines[:-1]
                self.page.segments([(left, self.y, right, self.y)], GRID, 0.5)
            ink = MUTED if faint is not None and faint(item) else INK
            base, x = self.y + row_h - 2.7, left
            for _, w, align, font, get in cols:
                font = "F2" if bold and font == "F1" else font
                self.cell(x, base, w, align, font, ink, get(item))
                x += w + GAP
            self.y += row_h
            lines.append((left, self.y, right, self.y))
        self.page.segments(lines, GRID_SOFT, 0.4)

    def cell(
        self, x: float, base: float, w: float, align: str, font: str, ink: str, c: Any
    ) -> None:
        """a table cell, `w` wide from `x`, with its text's baseline at `base`"""
        if not isinstance(c, Cell):
            at = x + w if align == "right" else x
            self.page.text(at, base, c, 6.5, font, ink, align, width=w)
            return
        if c.key is not None and c.fill is not None:
            self.page.rect(x, base - 4.3, METER, 4, GRID_SOFT)
            self.page.rect(x, base - 4.3, max(0.5, METER * c.fill), 4, c.key)
            x, w = x + METER + 5, w - METER - 5
        elif c.key is not None:
            self.page.rect(x, base - 4.6, 5, 5, c.key)
            x, w = x + 8, w - 8
        font = c.font or font
        self.page.text(x, base, c.text, 6.5, font, c.ink or ink, width=w)
        if c.note:
            at = x + measure(clip(c.text, 6.5, font, w), 6.5, font) + 5
            self.page.text(at, base, c.note, 6.5, color=INK2, width=x + w - at)

    def footers(self, items: list[str | None]) -> None:
        """`items` but those that are None, and its number, at every page's foot"""
        line = " · ".join(s for s in items if s)
        n = len(self.doc.pages)
        for i, page in enumerate(self.doc.pages, 1):
            num = f"Page {i} of {n}"
            page.text(PAGE_W - MARGIN, PAGE_H - 20, num, 6.5, color=INK2, align="right")
            room = PAGE_W - 2 * MARGIN - measure(num, 6.5) - 12
            page.text(MARGIN, PAGE_H - 20, line, 6.5, color=ACCENT, width=room)


class TimelinePdf(Flow):
    """
    lays a visualizer payload out as pages: the subject and its whole timeline
    first, then every code on a row of its own, then the events
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        self.d = d = payload
        self.meta, self.sub = d["meta"], d["subject"]
        self.E, self.C, self.L = d["events"], d["codes"], d["lanes"]
        self.labels = d["times"]["label"]
        self.n_e = len(self.E.get("kind") or ())
        self.n_t = len(d["tokens"]["id"])
        self.zone_name = self.meta.get("timezone") or "UTC"
        try:
            self.zone: datetime.tzinfo = zoneinfo.ZoneInfo(self.zone_name)
        except (zoneinfo.ZoneInfoNotFoundError, ValueError):
            self.zone_name, self.zone = "UTC", datetime.timezone.utc
        super().__init__(f"Subject {self.sub['subject_id']} · cocoa", self.meta)
        if self.n_e:
            self.index()

    def index(self) -> None:
        """what the drawing needs from the events, as the page derives it"""
        E, times = self.E, self.d["times"]["ms"]
        self.ems = [times[i] for i in E["time"]]
        tms = [self.ems[e] for e in self.d["tokens"]["event"]]
        self.ends = [e for e, k in enumerate(E["kind"]) if k in ("bos", "eos")]
        self.code_ev: list[list[int]] = [[] for _ in self.C]
        self.lane_ev: list[list[int]] = [[] for _ in self.L]
        for e, c in enumerate(E["code"]):
            self.code_ev[c].append(e)
            if self.C[c]["lane"] is not None:
                self.lane_ev[self.C[c]["lane"]].append(e)
        self.lane_codes: list[list[int]] = [[] for _ in self.L]
        for c, code in enumerate(self.C):
            if code["lane"] is not None:
                self.lane_codes[code["lane"]].append(c)
        for cs in self.lane_codes:
            cs.sort(key=lambda c: (-len(self.code_ev[c]), self.code_ev[c][0]))
        bins = [b for b in E["bin"] if b is not None]
        self.binned = [any(E["bin"][e] is not None for e in ev) for ev in self.code_ev]
        self.n_bins = max(2, self.meta.get("n_bins") or (max(bins, default=0) + 1))
        self.multi = [len(lane["prefixes"]) > 1 for lane in self.L]
        self.n_stamps, last = 0, None
        for k, t in zip(E["kind"], E["time"]):
            if k in ("event", "unk") and t != last:
                self.n_stamps, last = self.n_stamps + 1, t
        lo, hi = self.ems[0], self.ems[-1]
        pad = 30 * MIN if hi - lo < MIN else (hi - lo) * 0.015
        self.ext0, self.ext1 = lo - pad, hi + pad
        w8 = self.d.get("winnowed")
        lv = w8["last_valid"] if w8 else None
        self.lv = lv
        self.thr = (
            tms[max(0, min(self.n_t, lv) - 1)] if lv is not None and tms else None
        )
        self.make_ticks()

    # ---- time ---------------------------------------------------------------------

    def X(self, t: float) -> float:
        return PLOT_X + (t - self.ext0) / (self.ext1 - self.ext0) * PLOT_W

    def wall(self, ms: float) -> datetime.datetime:
        """an instant's wall-clock reading in the data's zone"""
        return (UTC_EPOCH + datetime.timedelta(milliseconds=ms)).astimezone(self.zone)

    def from_wall(self, naive: datetime.datetime) -> float:
        """a wall-clock reading in the data's zone, as an instant"""
        return naive.replace(tzinfo=self.zone).timestamp() * 1000

    def wall_ms(self, ms: float) -> float:
        return (self.wall(ms).replace(tzinfo=None) - EPOCH) / datetime.timedelta(
            milliseconds=1
        )

    def make_ticks(self) -> None:
        """axis ticks on round wall-clock times, about every 85 points"""
        v0, v1 = self.ext0, self.ext1
        want = (v1 - v0) / max(1, PLOT_W / 85)
        ms, mo, _ = next((s for s in STEPS if s[2] >= want), STEPS[-1])
        out = []
        w: datetime.datetime | float
        if mo:
            w = self.wall(v0)
            m0 = (w.year * 12 + w.month - 1) // mo * mo
            for k in range(500):
                m = m0 + k * mo
                t = self.from_wall(datetime.datetime(m // 12, m % 12 + 1, 1))
                if t > v1:
                    break
                if t >= v0:
                    out.append(t)
        else:
            w = self.wall_ms(v0) // ms * ms
            while len(out) < 500:
                t = self.from_wall(EPOCH + datetime.timedelta(milliseconds=w))
                if t > v1:
                    break
                if t >= v0 and (not out or t != out[-1]):
                    out.append(t)
                w += ms
        self.ticks = out
        self.tick_labels, prev = [], None
        for t in out:
            w = self.wall(t)
            new_year = prev is None or prev.year != w.year
            new_day = new_year or (prev.month, prev.day) != (w.month, w.day)
            b = ""
            if ms and ms < DAY:
                a = f"{w.hour:02d}:{w.minute:02d}" + (
                    f":{w.second:02d}" if ms < MIN else ""
                )
                if new_day:
                    b = f"{MON[w.month - 1]} {w.day}" + (
                        f", {w.year}" if new_year else ""
                    )
            elif ms:
                a = f"{MON[w.month - 1]} {w.day}"
                b = str(w.year) if new_year else ""
            elif mo < 12:
                a, b = MON[w.month - 1], str(w.year) if new_year else ""
            else:
                a = str(w.year)
            self.tick_labels.append((a, b))
            prev = w

    # ---- sections -----------------------------------------------------------------

    def header(self) -> None:
        sub = self.sub
        split = [sub["split"]] if sub.get("split") else []
        self.title(f"Subject {sub['subject_id']}", split)
        start, end = sub.get("start_time"), sub.get("end_time")
        facts = [("Start", start["label"])] if start else []
        facts += [("End", end["label"])] if end else []
        facts += [("Span", fmt_dur(end["ms"] - start["ms"]))] if start and end else []
        if self.n_e:
            facts += [
                ("Tokens", fmt_int(self.n_t)),
                ("Events", fmt_int(self.n_stamps)),
                ("Distinct codes", fmt_int(len(self.C))),
            ]
        self.definitions(facts)
        fields = [(f["label"], f["value"]) for f in sub.get("fields") or ()]
        if fields := [(k, v) for k, v in fields if v is not None]:
            self.y -= 6
            self.definitions(fields, bold=False)
        self.y += 4

    def overview(self) -> None:
        """every lane, squeezed if need be to fit this page under the header"""
        room = BOTTOM - self.y - AXIS_H - 30
        h = max(10.0, min(LANE_H, room / max(1, len(self.L))))
        rows = [
            {"lane": j, "code": None, "ev": self.lane_ev[j], "h": h}
            for j in range(len(self.L))
        ]
        self.figure(rows)
        self.paragraph(
            "Each mark is an event, in the lane of its code's prefix. The pages that "
            "follow give every code a row of its own, with binned values drawn at the "
            "height of their quantile, and then list the events one by one."
        )

    def every_code(self) -> None:
        rows = []
        for j in range(len(self.L)):
            rows.append({"lane": j, "code": None, "ev": self.lane_ev[j], "h": LANE_H})
            for c in self.lane_codes[j]:
                h = BIN_H if self.binned[c] else CODE_H
                rows.append({"lane": j, "code": c, "ev": self.code_ev[c], "h": h})
        if rows:
            if len(self.doc.pages) == 1:  # the overview keeps the first page
                self.new_page()
            else:  # following a winnowing summary that needed a page of its own
                self.y += 10
            self.heading("Every code", f"{fmt_int(len(self.C))} distinct codes")
            self.figure(rows)

    def figure(self, rows: list[dict[str, Any]]) -> None:
        """rows of the timeline under a time axis, carried onto new pages"""
        i = 0
        while i < len(rows):
            if self.y + AXIS_H + rows[i]["h"] > BOTTOM:
                self.new_page()
            top = y = self.y + AXIS_H
            j = i
            while j < len(rows) and (j == i or y + rows[j]["h"] <= BOTTOM):
                rows[j]["y"] = y
                y += rows[j]["h"]
                j += 1
            self.axis(self.y)
            self.plot(rows[i:j], top, y)
            self.y = y + 8
            i = j

    def axis(self, top: float) -> None:
        p = self.page
        p.text(MARGIN, top + 18, f"Times in {self.zone_name}", 6.5, color=INK2)
        marks = []
        for t, (a, b) in zip(self.ticks, self.tick_labels):
            x = self.X(t)
            marks.append((x, top + 21, x, top + 25))
            for s, font, color, y in ((a, "F1", INK2, 9), (b, "F2", INK, 18)):
                half = measure(s, 6.5, font) / 2
                cx = min(max(x, PLOT_X + half), PLOT_X + PLOT_W - half)
                p.text(cx, top + y, s, 6.5, font, color, align="center")
        p.segments(marks, AXIS, 0.5)
        p.segments([(PLOT_X, top + AXIS_H, PLOT_X + PLOT_W, top + AXIS_H)], AXIS, 0.5)
        # markers: the winnowing threshold wins any overlap with BOS and EOS
        taken: list[tuple[float, float]] = []

        def place(s: str, x: float, align: str) -> None:
            w = measure(s, 6)
            x0 = x if align == "left" else x - w
            if x0 < PLOT_X or x0 + w > PLOT_X + PLOT_W:
                return
            if any(x0 < b + 4 and x0 + w > a - 4 for a, b in taken):
                return
            taken.append((x0, x0 + w))
            p.text(x, top + 33.5, s, 6, color=INK2, align=align)

        if self.thr is not None:
            x = self.X(self.thr)
            p.segments([(x, top + 27, x, top + AXIS_H)], INK2, 0.75)
            place("past", x - 3, "right")
            place("future", x + 3, "left")
        for e in self.ends:
            x = self.X(self.ems[e])
            p.segments([(x, top + 27, x, top + AXIS_H)], AXIS, 0.5)
            if self.E["kind"][e] == "bos":
                place("BOS", x + 3, "left")
            else:
                place("EOS", x - 3, "right")

    def plot(self, rows: list[dict[str, Any]], y0: float, y1: float) -> None:
        p, right = self.page, PLOT_X + PLOT_W
        for r in rows:
            if r["code"] is not None:
                p.rect(PLOT_X, r["y"], PLOT_W, r["h"], WASH)
        if self.thr is not None and self.X(self.thr) < right:
            x = max(PLOT_X, self.X(self.thr))
            p.rect(x, y0, right - x, y1 - y0, FUTURE)
        p.segments([(x, y0, x, y1) for x in map(self.X, self.ticks)], GRID_SOFT, 0.5)
        for soft in (False, True):
            p.segments(
                [
                    (PLOT_X, r["y"], right, r["y"])
                    for r in rows
                    if (r["code"] is not None) == soft
                ],
                GRID_SOFT if soft else GRID,
                0.5,
            )
        p.segments([(PLOT_X, y1, right, y1)], GRID, 0.5)
        ends = [self.X(self.ems[e]) for e in self.ends]
        p.segments([(x, y0, x, y1) for x in ends], AXIS, 0.5)
        if self.thr is not None:
            x = self.X(self.thr)
            p.segments([(x, y0, x, y1)], INK2, 0.75)
        for r in rows:
            if r["code"] is not None and self.binned[r["code"]]:
                self.draw_binned(r)
            else:
                self.draw_ticks(r)
            self.label(r)

    def draw_ticks(self, r: dict[str, Any]) -> None:
        pad = max(2.0, r["h"] * 0.2)
        xs = sorted({round(self.X(self.ems[e]) * 4) / 4 for e in r["ev"]})
        segs = [(x, r["y"] + pad, x, r["y"] + r["h"] - pad) for x in xs]
        self.page.segments(segs, self.L[r["lane"]]["color"], 1.0)

    def draw_binned(self, r: dict[str, Any]) -> None:
        p, color, top, h = self.page, self.L[r["lane"]]["color"], r["y"], r["h"]

        def y_bin(b: int) -> float:
            return top + h - 4 - b / (self.n_bins - 1) * (h - 8)

        guides = [y_bin(self.n_bins - 1), y_bin(0)]
        p.segments([(PLOT_X, y, PLOT_X + PLOT_W, y) for y in guides], GRID_SOFT, 0.5)
        pts, bare = [], []
        for e in r["ev"]:
            b = self.E["bin"][e]
            x = self.X(self.ems[e])
            if b is None:
                bare.append(x)
            else:
                pts.append((x, y_bin(b)))
        if len(pts) > PLOT_W / 2:  # too dense for a line: the range per column
            cols: dict[float, tuple[float, float]] = {}
            for x, y in pts:
                k = round(x * 2) / 2
                lo, hi = cols.get(k, (y, y))
                cols[k] = (min(lo, y), max(hi, y))
            segs = [(k, lo, k, hi) for k, (lo, hi) in cols.items()]
            p.segments(segs, color, 1.0, round_cap=True)
        else:
            p.polyline(pts, mix(color, 0.45), 0.8)
            dots = [(x, y, x, y) for x, y in pts]
            if len(pts) <= PLOT_W / 8:
                p.segments(dots, "#FFFFFF", 6.0, round_cap=True)
                p.segments(dots, color, 4.4, round_cap=True)
            else:
                p.segments(dots, color, 2.6, round_cap=True)
        # a text value or unparseable number on a binned code has no height
        mid = top + h / 2
        p.segments([(x, mid - 3, x, mid + 3) for x in bare], color, 1.0)

    def label(self, r: dict[str, Any]) -> None:
        p, lane, top, h = self.page, self.L[r["lane"]], r["y"], r["h"]
        count = fmt_int(len(r["ev"]))
        if r["code"] is None:
            size = 8.0 if h >= 14 else 6.5
            base = top + h / 2 + size * 0.35
            p.rect(MARGIN, top + h / 2 - 3.5, 7, 7, lane["color"])
            p.text(MARGIN + GUTTER, base, count, size - 1, color=INK2, align="right")
            room = GUTTER - 17 - measure(count, size - 1)
            p.text(MARGIN + 11, base, lane["name"], size, width=room)
            return
        c, binned = self.C[r["code"]], self.binned[r["code"]]
        base = top + h / 2 + 2.4
        right = MARGIN + GUTTER - (16 if binned else 0)
        p.text(right, base, count, 6.5, color=INK2, align="right")
        x = MARGIN + 11
        if self.multi[r["lane"]] or not c["name"]:  # a bare prefix has no name
            p.text(x, base, c["prefix"], 6, color=INK2)
            x += measure(c["prefix"], 6) + 4
        p.text(x, base, c["name"], 7, width=right - x - measure(count, 6.5) - 6)
        if binned:
            hi = f"Q{self.n_bins - 1}"
            p.text(MARGIN + GUTTER, top + 7, hi, 5.5, color=MUTED, align="right")
            p.text(
                MARGIN + GUTTER, top + h - 2.5, "Q0", 5.5, color=MUTED, align="right"
            )

    def winnowing(self) -> None:
        w8 = self.d.get("winnowed")
        if not w8:
            return
        self.need(90)
        self.y += 6
        self.heading("Winnowed for inference", w8.get("file"))
        n_past = min(self.lv or 0, self.n_t)
        facts = []
        if n_past > 0:
            e = self.d["tokens"]["event"][n_past - 1]
            facts.append(("Threshold", self.labels[self.E["time"][e]]))
        facts += [
            ("Past tokens", fmt_int(n_past)),
            ("Future tokens", fmt_int(self.n_t - n_past)),
        ]
        if w8.get("n_future") is not None and w8["n_future"] != self.n_t - n_past:
            facts.append(("Within the horizon", fmt_int(w8["n_future"])))
        self.definitions(facts)
        outcomes = w8.get("outcomes") or ()
        for o in [o for o in outcomes if o["past"] or o["future"]]:
            self.need(11)
            base = self.y + 7
            self.page.text(MARGIN, base, o["code"], 7, "F3", width=230)
            for x, tense in ((MARGIN + 240, "past"), (MARGIN + 270, "future")):
                on = o[tense]
                self.page.text(
                    x, base, tense, 7, "F2" if on else "F1", INK if on else MUTED
                )
            self.y += 11
        if absent := [o["code"] for o in outcomes if not o["past"] and not o["future"]]:
            self.y += 4
            self.paragraph("Absent from both: " + ", ".join(absent), 7, INK)

    def events(self) -> None:
        E, toks = self.E, self.d["tokens"]
        vocab = toks["vocab"]

        def tokens_of(e: int) -> str:
            first = E["first_token"][e]
            ids = toks["id"][first : first + E["n_tokens"][e]]
            return "  ".join(f"{t} {vocab.get(str(t), 'UNK')}" for t in ids)

        def lane_of(e: int) -> Cell | str:
            return lane_cell(self.L, self.C[E["code"][e]]["lane"])

        cols = [
            ("#", 34, "right", "F1", lambda e: fmt_int(E["first_token"][e])),
            ("Time", 112, "left", "F1", lambda e: self.labels[E["time"][e]]),
            ("Lane", 96, "left", "F1", lane_of),
            ("Code", 150, "left", "F3", lambda e: self.C[E["code"][e]]["code"]),
            ("Bin", 22, "right", "F1", lambda e: _q(E["bin"][e])),
        ]
        if self.meta.get("has_numeric_values"):
            cols.append(
                ("Value", 48, "right", "F1", lambda e: fmt_num(E["value"][e], 6))
            )
        cols.append(("Text", 70, "left", "F1", lambda e: E["text"][e] or ""))
        cols.append(("Token", rest_of(cols), "left", "F3", tokens_of))

        n = min(self.n_e, ROW_CAP)
        note = (
            f"the first {fmt_int(n)} of {fmt_int(self.n_e)} events; "
            "the html page lists them all"
            if n < self.n_e
            else f"{fmt_int(self.n_e)} events"
        )
        self.new_page()
        self.heading("Events", note)
        self.table(cols, range(n))

    def footer(self) -> list[str | None]:
        m = self.meta
        return [
            f"cocoa {m.get('cocoa_version')}",
            f"rendered {m.get('generated')}",
            f"tokenizer created {m['tokenizer_created']}"
            if m.get("tokenizer_created")
            else None,
            f"vocabulary of {fmt_int(m.get('vocab_size', 0))}",
            "fused bins" if m.get("fused") else "unfused bins",
            "values from meds.parquet" if m.get("value_source") == "meds" else None,
            f"{m['n_bins']} bins" if m.get("n_bins") else None,
            f"times in {m.get('timezone')}, {m.get('time_unit')}",
            m.get("processed_data_home"),
        ]

    def to_bytes(self) -> bytes:
        self.header()
        if not self.n_e:
            self.paragraph("This subject's timeline has no events.")
        else:
            self.overview()
            self.winnowing()
            self.every_code()
            self.events()
        self.footers(self.footer())
        return self.doc.to_bytes()


class VocabularyPdf(Flow):
    """
    lays the vocabulary page's payload out as pages: the vocabulary, its prefixes,
    and its timelines by split first, then each binned code's training values
    against its bins, then every word, then the tokenizer's config
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        self.d = d = payload
        self.meta = m = d["meta"]
        self.C, self.L, self.P = d["codes"], d["lanes"], d["prefixes"]
        self.B, self.V, self.TL = d["binned"], d["words"], d.get("timelines")
        self.n_v = len(self.V["word"])
        home = m.get("processed_data_home")
        name = [pathlib.Path(home).name] if home else []
        super().__init__(" · ".join(["Vocabulary", *name, "cocoa"]), m)
        self.counted = [n for n in self.V["count"] if n is not None]
        self.trained = sum(self.counted)  # tokens in training
        lane_of = {p["prefix"]: p["lane"] for p in self.P}
        self.word_lane = [
            self.C[c]["lane"] if c is not None else lane_of.get("")
            for c in self.V["code"]
        ]
        self.words_of: dict[int, list[int]] = collections.defaultdict(list)
        self.bin_word: dict[int, int] = {}  # each unfused bin's word, codes share
        for i, (c, kind, b) in enumerate(
            zip(self.V["code"], self.V["kind"], self.V["bin"])
        ):
            if c is not None:
                self.words_of[c].append(i)
            elif kind == "bin":
                self.bin_word[b] = i

    def color(self, lane: int | None) -> str:
        return self.L[lane]["color"] if lane is not None else MUTED

    def bin_words(self, code: int, k: int) -> list[int]:
        """the words bin `k` of `code` tokenizes to: fused, its own; unfused, the
        one every code shares"""
        if not self.meta.get("fused"):
            return [self.bin_word[k]] if k in self.bin_word else []
        return [i for i in self.words_of.get(code, ()) if self.V["bin"][i] == k]

    # ---- axes ---------------------------------------------------------------------

    def count_axis(
        self,
        x0: float,
        x1: float,
        top: float,
        Y: collections.abc.Callable[[float], float],
    ) -> None:
        """a count axis's grid up to `top`, labeled left of `x0`"""
        ticks = count_ticks(top)
        self.page.segments([(x0, Y(t), x1, Y(t)) for t in ticks], GRID_SOFT, 0.5)
        for t in ticks:
            label = fmt_compact(t)
            self.page.text(x0 - 3, Y(t) + 2.1, label, 6, color=INK2, align="right")

    def value_axis(
        self, left: float, w: float, base: float, at: float, ticks: list[tuple]
    ) -> None:
        """(x, label) ticks under a plot's baseline, the labels kept on the plot"""
        self.page.segments([(x, base, x, base + 3) for x, _ in ticks], GRID, 0.5)
        for x, s in ticks:
            half = measure(s, 6) / 2 + 2
            x = min(max(x, left + half), left + w - half)
            self.page.text(x, at, s, 6, color=INK2, align="center")

    # ---- sections -----------------------------------------------------------------

    def header(self) -> None:
        m, tl = self.meta, self.TL
        transferred = m.get("is_training") is False
        badges = ["fused" if m.get("fused") else "unfused"]
        self.title("Vocabulary", badges + ["transferred"] if transferred else badges)
        facts = [
            ("Words", fmt_int(m.get("vocab_size", self.n_v))),
            ("Codes", fmt_int(sum(c["prefix"] not in STRUCT for c in self.C))),
            ("Binned codes", fmt_int(len(self.B))),
        ]
        if m.get("n_bins") is not None:
            facts.append(("Bins per code", fmt_int(m["n_bins"])))
        if m.get("has_counts"):
            facts.append(("Tokens in training", fmt_int(self.trained)))
        if tl:
            facts.append(("Timelines", fmt_int(tl["all"]["timelines"])))
            if (median := tl["all"]["length"]["median"]) is not None:
                facts.append(("Median length", fmt_num(median)))
        self.definitions(facts)
        if transferred:
            self.y -= 6
            self.paragraph(
                "This vocabulary was learned on another dataset, so its words and "
                "training counts are that dataset's, while the histograms are of "
                "this directory's training split."
            )
        self.y += 4

    def prefixes(self) -> None:
        note = f"{fmt_int(self.trained)} tokens in training"
        self.heading("Prefixes", note if self.meta.get("has_counts") else None)
        most = max([1, *(p["count"] or 0 for p in self.P)])

        def name(p: dict[str, Any]) -> str:
            if not p["prefix"]:
                return "Unfused bins and text values, shared by codes"
            return p["category"] or ""

        def share(p: dict[str, Any]) -> Cell | str:
            if p["count"] is None:
                return ""
            pct = fmt_pct(p["count"] / self.trained) if self.trained else None
            fill = p["count"] / most
            return Cell(fmt_int(p["count"]), self.color(p["lane"]), fill, pct)

        def binned(p: dict[str, Any]) -> str:
            return fmt_int(p["binned"]) if p["binned"] else ""

        cols = [
            ("Prefix", 60, "left", "F3", lambda p: p["prefix"] or "—"),
            ("Name", 170, "left", "F1", name),
            ("Lane", 120, "left", "F1", lambda p: lane_cell(self.L, p["lane"])),
            ("Words", 36, "right", "F1", lambda p: fmt_int(p["words"])),
            ("Codes", 36, "right", "F1", lambda p: fmt_int(p["codes"])),
            ("Binned", 36, "right", "F1", binned),
        ]
        cols.append(("Training tokens", rest_of(cols), "left", "F1", share))
        self.table(cols, self.P)

    def timelines(self) -> None:
        tl = self.TL
        if not tl:
            return
        everyone = tl["all"]

        def name(g: dict[str, Any]) -> str:
            return "All" if g is everyone else g["split"] or "No split"

        def unk(g: dict[str, Any]) -> str:
            share = f" ({fmt_pct(g['unk'] / g['tokens'])})" if g["tokens"] else ""
            return fmt_int(g["unk"]) + share

        def length(stat: str) -> collections.abc.Callable[[dict[str, Any]], str]:
            return lambda g: fmt_num(g["length"][stat])

        def hours(stat: str) -> collections.abc.Callable[[dict[str, Any]], str]:
            return lambda g: fmt_hours(g["hours"][stat])

        def when(end: str) -> collections.abc.Callable[[dict[str, Any]], str]:
            return lambda g: g[end]["label"] if g[end] else ""

        self.y += 10
        self.heading("Timelines", f"{fmt_int(everyone['timelines'])} timelines")
        cols = [
            ("Split", 56, "left", "F1", name),
            ("Timelines", 44, "right", "F1", lambda g: fmt_int(g["timelines"])),
            ("Tokens", 56, "right", "F1", lambda g: fmt_int(g["tokens"])),
            ("Out of vocabulary", 72, "right", "F1", unk),
            ("Mean length", 48, "right", "F1", length("mean")),
            ("Median length", 52, "right", "F1", length("median")),
            ("Mean duration", 60, "right", "F1", hours("mean")),
            ("Median duration", 60, "right", "F1", hours("median")),
            ("First event", 100, "left", "F1", when("first")),
        ]
        cols.append(("Last event", rest_of(cols), "left", "F1", when("last")))
        self.table(cols, [*tl["splits"], everyone], strong=lambda g: g is everyone)
        self.y += 3
        zone = f"; times in {tl['timezone']}" if tl.get("timezone") else ""
        self.paragraph(
            f"As tokenize --verbose summarizes them, from tokens_times.parquet{zone}."
        )
        self.y += 6
        for g in [everyone, *tl["splits"]]:
            self.charts(g, name(g))

    def charts(self, g: dict[str, Any], name: str) -> None:
        """the lengths and the durations of a group of timelines, side by side"""
        tl, w = self.TL, (PAGE_W - 2 * MARGIN - 24) / 2
        self.need(CHART_H + 42)
        p, top = self.page, self.y
        p.text(MARGIN, top + 8, name, 8, "F2")
        if g["instant"]:
            note = f"{fmt_int(g['instant'])} span no time, so have no duration to draw"
            p.text(PAGE_W - MARGIN, top + 8, note, 6.5, color=INK2, align="right")
        for j, (title, counts, start, kind) in enumerate(
            (
                ("Tokens per timeline", g["lengths"], tl["length_from"], "length"),
                (
                    "Duration, from first token to last",
                    g["durations"],
                    tl["hours_from"],
                    "hours",
                ),
            )
        ):
            x = MARGIN + j * (w + 24)
            p.text(x, top + 21, title, 7, width=w)
            self.log_chart(x, top + 25, w, CHART_H, counts, start, kind)
            if (stats := g[kind]) and stats.get("min") is not None:
                at = top + 25 + CHART_H + 8
                p.text(x, at, spread(stats, kind), 6, color=INK2, width=w)
        self.y = top + CHART_H + 42

    def log_chart(
        self,
        left: float,
        top: float,
        w: float,
        h: float,
        counts: list[int],
        start: int | None,
        kind: str,
    ) -> None:
        """
        a histogram on a log axis, per_decade buckets to a power of ten from
        bucket `start`, ticked at powers of ten of tokens or at familiar spans
        of time
        """
        p, n = self.page, len(counts)
        if not n or start is None:
            return
        x0, x1, y0, base = left + 30, left + w - 6, top + 4, top + h - 14
        slot = (x1 - x0) / n
        top_n = nice_ceil(max([1, *counts]))

        def Y(c: float) -> float:
            return base - c / top_n * (base - y0)

        self.count_axis(x0, x1, top_n, Y)
        gap = 1.5 if slot > 3 else 0
        for k, c in enumerate(counts):
            if c > 0:
                t = min(Y(c), base - 0.75)
                x = x0 + k * slot + gap / 2
                p.rect(x, t, max(0.75, slot - gap), base - t, INK2)
        p.segments([(x0, base, x1, base)], GRID, 0.5)
        per = self.TL["per_decade"]
        lo, hi = 10 ** (start / per), 10 ** ((start + n) / per)
        ticks = [(v, s) for v, s in HOURS if lo <= v <= hi] if kind == "hours" else []
        if len(ticks) < 2:
            ticks, d = [], math.ceil(math.log10(lo))
            while 10.0**d <= hi:
                v = 10.0**d
                ticks.append((v, fmt_hours(v) if kind == "hours" else fmt_compact(v)))
                d += 1
        at = [(x0 + (math.log10(v) * per - start) * slot, s) for v, s in ticks]
        self.value_axis(left, w, base, top + h - 3, at)

    def binned(self) -> None:
        m, n = self.meta, min(len(self.B), BINNED_CAP)
        if not self.B:
            self.y += 10
            self.heading("Binned codes")
            self.paragraph(
                "No code had a numeric value in training, so the tokenizer binned none."
            )
            return
        note = (
            f"the first {fmt_int(n)} of {fmt_int(len(self.B))} binned codes; "
            "the html page draws them all"
            if n < len(self.B)
            else f"{fmt_int(len(self.B))} binned codes"
        )
        self.new_page()
        self.heading("Binned codes", note)
        if m.get("value_source") == "meds":
            source = (
                "Values from this directory's training split, against bins learned "
                "on another dataset."
                if m.get("is_training") is False
                else "Training values against the bins learned on them."
            )
        elif m.get("bins_source") == "vocabulary":
            source = (
                "Each bin's training count, from its words: there are no training "
                "values here to draw."
            )
        else:
            source = "Only cut points: there are no training values here to draw."
        self.paragraph(
            f"{source} A bin holds the values from its lower cut point up to, but "
            "not including, its upper one."
        )
        self.y += 6
        for b in self.B[:n]:
            self.binned_code(b)

    def binned_code(self, b: dict[str, Any]) -> None:
        """a binned code's histogram, beside a row for each of its bins"""
        c, breaks, counts = self.C[b["code"]], b["breaks"], b.get("bins")
        nb, color = len(breaks) + 1, self.color(c["lane"])
        drawn = bool(b.get("hist") or counts)
        body = max(10 + nb * BIN_ROW_H, 80.0 if drawn else 0.0)
        self.need(min(22 + body, BOTTOM - MARGIN))  # one taller runs onto the next
        p, top, right = self.page, self.y, PAGE_W - MARGIN

        # the code, then its description and its prefix's name, each left of a
        # summary set flush right
        p.rect(MARGIN, top + 2.4, 7, 7, color)
        sub = " · ".join(s for s in (c.get("description"), c.get("category")) if s)
        for y, s, size, font, ink, aside in (
            (top + 8.5, c["code"], 8, "F3", INK, facts(b)),
            (top + 17.5, sub, 6.5, "F1", INK2, held(b)),
        ):
            p.text(right, y, aside, 6.5, color=INK2, align="right")
            room = right - MARGIN - 11 - (measure(aside, 6.5) + 12 if aside else 0)
            p.text(MARGIN + 11, y, s, size, font, ink, width=room)
        self.y, left = top + 22, MARGIN + 11
        if drawn:  # as tall as its bins, unless they run onto the next page
            self.histogram(b, MARGIN, self.y, HIST_W, min(body, BOTTOM - self.y))
            left = MARGIN + HIST_W + 18
        total, most = sum(counts or ()), max([1, *(counts or ())])

        def count(k: int) -> str:
            return fmt_int(counts[k]) if counts else "—"

        def share(k: int) -> Cell | str:
            if not counts or not total:
                return ""
            return Cell(fmt_pct(counts[k] / total), color, counts[k] / most)

        def words(k: int) -> Cell | str:
            if ws := self.bin_words(b["code"], k):
                return "  ".join(
                    f"{self.V['token'][i]} {self.V['word'][i]}" for i in ws
                )
            if tied(breaks, k):
                return ""
            return Cell("not in the vocabulary, so UNK", ink=MUTED, font="F1")

        by_words = self.meta.get("bins_source") == "vocabulary"
        cols = [
            ("Bin", 18, "left", "F1", lambda k: f"Q{k}"),
            ("Range", 112, "left", "F1", lambda k: bin_range(breaks, k)),
            (
                "Training count" if by_words else "Training values",
                56,
                "right",
                "F1",
                count,
            ),
            ("Share", 64, "left", "F1", share),
        ]
        cols.append(("Words", rest_of(cols, right - left), "left", "F3", words))
        self.table(
            cols,
            range(nb),
            left=left,
            row_h=BIN_ROW_H,
            faint=lambda k: counts is not None and counts[k] == 0,
        )
        if self.page is p:  # below the histogram, unless the bins ran on
            self.y = max(self.y, top + 22 + body)
        self.y += 14

    def histogram(
        self, b: dict[str, Any], left: float, top: float, w: float, h: float
    ) -> None:
        """
        a binned code's training values, by value or in buckets, or failing those
        each bin's training count side by side, over its bins washed in turn and
        under its cut points, as the page draws an open card
        """
        p, hist = self.page, b.get("hist")
        cuts = [v for v in b["breaks"] if v is not None]
        distinct: list[tuple[float, int]] = []  # each cut point, and the bin it opens
        for v in cuts:
            if not distinct or distinct[-1][0] != v:
                distinct.append((v, sum(c <= v for c in cuts)))
        x0, x1, y0, base = left + 30, left + w - 6, top + 12, top + h - 14
        gap = 1.5
        cut_xs: list[float] | None
        if hist and "x" in hist:  # each value a bar of its own, at its value
            xs = hist["x"]
            step = min((v - u for u, v in zip(xs, xs[1:])), default=math.inf)
            if not math.isfinite(step):
                step = abs(xs[0]) * 0.2 or 1.0
            d0, d1 = min([xs[0], *cuts]) - step / 2, max([xs[-1], *cuts]) + step / 2
            X = axis(d0, d1, x0, x1)
            bw = max(1.0, min(16.0, X(d0 + step) - x0 - gap))
            marks = [(X(v) - bw / 2, X(v) + bw / 2, n) for v, n in zip(xs, hist["n"])]
            index = {v: k for k, v in enumerate(xs)}

            def at(v: float) -> float:
                """a cut point is its bin's lowest value, so just left of its bar"""
                k = index.get(v)
                if k is None:
                    return X(v)
                return (X(xs[k - 1]) + X(v)) / 2 if k else X(v) - bw / 2 - gap

            cut_xs = [at(v) for v, _ in distinct]
        elif hist:  # values in buckets of a round width
            lo, width = hist["lo"], hist["w"]
            d0, d1 = lo, lo + width * len(hist["n"])
            X = axis(d0, d1, x0, x1)
            gp = gap if X(lo + width) - x0 > 3 else 0
            marks = [
                (X(lo + k * width) + gp / 2, X(lo + (k + 1) * width) - gp / 2, n)
                for k, n in enumerate(hist["n"])
            ]
            cut_xs = [X(v) for v, _ in distinct]
        else:  # no values to draw, so each bin's training count
            d0, d1 = 0.0, float(len(b["bins"]))
            X = axis(d0, d1, x0, x1)
            bw = max(1.0, min(16.0, X(1) - x0 - 2 * gap))
            marks = [
                (X(k + 0.5) - bw / 2, X(k + 0.5) + bw / 2, n)
                for k, n in enumerate(b["bins"])
            ]
            cut_xs = None
        top_n = nice_ceil(max([1, *(n for *_, n in marks)]))

        def Y(n: float) -> float:
            return base - n / top_n * (base - y0)

        regions = []
        if cut_xs is not None:
            edges = [x0, *cut_xs, x1]
            qs = [0, *(q for _, q in distinct)]
            regions = [(edges[k], edges[k + 1], q) for k, q in enumerate(qs)]
            for a, z, _ in regions[1::2]:
                p.rect(a, y0, z - a, base - y0, BAND)
        self.count_axis(x0, x1, top_n, Y)
        color = self.color(self.C[b["code"]]["lane"])
        for a, z, n in marks:
            if n > 0:
                t = min(Y(n), base - 0.75)
                p.rect(a, t, max(0.75, z - a), base - t, color)
        p.segments([(x0, base, x1, base)], GRID, 0.5)
        p.segments([(x, y0, x, base) for x in cut_xs or ()], INK2, 0.5)
        for a, z, q in regions:  # which bin each band is, where it has room
            if z - a >= measure(f"Q{q}", 6) + 4:
                p.text((a + z) / 2, y0 - 3, f"Q{q}", 6, color=INK2, align="center")
        if cut_xs is not None:
            ticks = nice_ticks(d0, d1, max(2, int((x1 - x0) // 60)))
            at = [(X(t), fmt_num(t)) for t in ticks]
            self.value_axis(left, w, base, top + h - 3, at)
        else:
            for k, (a, z, _) in enumerate(marks):
                p.text((a + z) / 2, top + h - 3, f"Q{k}", 6, color=INK2, align="center")

    def words(self) -> None:
        V, m = self.V, self.meta
        n = min(self.n_v, ROW_CAP)
        note = (
            f"the first {fmt_int(n)} of {fmt_int(self.n_v)} words; "
            "the html page lists them all"
            if n < self.n_v
            else f"{fmt_int(self.n_v)} words"
        )
        self.new_page()
        self.heading("Words", note)
        if m.get("has_counts"):
            once = sum(c == 1 for c in self.counted)
            few = sum(c < 10 for c in self.counted)
            rare = m.get("min_training_ct") or 0
            self.paragraph(
                f"{fmt_int(once)} seen once in training, {fmt_int(few)} fewer than 10 "
                "times" + (f"; rarer than {fmt_int(rare)} became UNK." if rare else ".")
            )
        else:
            self.paragraph("tokenizer.yaml holds no training counts.")
        self.y += 3

        def count(i: int) -> str:
            return fmt_int(V["count"][i]) if V["count"][i] is not None else ""

        cols = [
            ("Token", 30, "right", "F1", lambda i: fmt_int(V["token"][i])),
            ("Word", 300, "left", "F3", lambda i: V["word"][i]),
            ("Lane", 110, "left", "F1", lambda i: lane_cell(self.L, self.word_lane[i])),
            ("Bin", 20, "right", "F1", lambda i: _q(V["bin"][i])),
        ]
        last = ("Training count", 56, "right", "F1", count)
        text = (
            "Text",
            rest_of([*cols, last]),
            "left",
            "F1",
            lambda i: V["text"][i] or "",
        )
        self.table([*cols, text, last], range(n))

    def config(self) -> None:
        m = self.meta
        self.y += 10
        self.need(80)
        self.heading("Tokenizer config", "as saved in tokenizer.yaml")
        facts = []
        if m.get("is_training") is not None:
            learned = (
                "on this directory's training split"
                if m["is_training"]
                else "on another dataset, applied here"
            )
            facts.append(("Learned", learned))
        if m.get("tokenizer_created"):
            facts.append(("Created", m["tokenizer_created"]))
        if m.get("tokenizer_version"):
            facts.append(("By cocoa", m["tokenizer_version"]))
        if facts:
            self.definitions(facts)
        settings = self.d.get("config") or ()
        kw = min(160.0, max([0.0, *(measure(k, 6.5, "F3") for k, _ in settings)]))
        x = MARGIN + kw + 14
        for k, v in settings:
            for j, line in enumerate(wrap(v, 7, "F1", PAGE_W - MARGIN - x) or [""]):
                self.need(10)
                if j == 0:
                    self.page.text(MARGIN, self.y + 7, k, 6.5, "F3", INK2, width=kw)
                self.page.text(x, self.y + 7, line, 7, width=PAGE_W - MARGIN - x)
                self.y += 10
            self.y += 2

    def footer(self) -> list[str | None]:
        m = self.meta
        return [
            f"cocoa {m.get('cocoa_version')}",
            f"rendered {m.get('generated')}",
            f"tokenizer created {m['tokenizer_created']}"
            if m.get("tokenizer_created")
            else None,
            f"by cocoa {m['tokenizer_version']}"
            if m.get("tokenizer_version")
            else None,
            m.get("processed_data_home"),
        ]

    def to_bytes(self) -> bytes:
        self.header()
        self.prefixes()
        self.timelines()
        self.binned()
        self.words()
        self.config()
        self.footers(self.footer())
        return self.doc.to_bytes()


def _q(b: int | None) -> str:
    return f"Q{b}" if b is not None else ""


def axis(
    d0: float, d1: float, x0: float, x1: float
) -> collections.abc.Callable[[float], float]:
    """where a value falls on an axis from `d0` at `x0` to `d1` at `x1`"""
    k = (x1 - x0) / ((d1 - d0) or 1)
    return lambda v: x0 + (v - d0) * k


def fmt_hours(h: float | None) -> str:
    return fmt_dur(h * HR) if h is not None else ""


def spread(stats: dict[str, Any], kind: str) -> str:
    """a summary of a spread of lengths, or of durations in hours"""
    f = fmt_hours if kind == "hours" else fmt_num
    line = (
        f"min {f(stats['min'])} · quartiles {f(stats['q25'])}, {f(stats['median'])}, "
        f"{f(stats['q75'])} · max {f(stats['max'])} · mean {f(stats['mean'])}"
    )
    return line + (f" (sd {f(stats['std'])})" if stats.get("std") is not None else "")


def facts(b: dict[str, Any]) -> str:
    """what a binned code's training values come to"""
    if b["n"] is None:
        return ""
    if not b["n"]:
        return "No training values in this directory"
    return " · ".join(
        [
            f"{fmt_int(b['n'])} training values",
            f"{fmt_int(b['distinct'])} distinct",
            *(f"{k} {fmt_num(b[k], 6)}" for k in ("min", "median", "mean", "max")),
        ]
    )


def held(b: dict[str, Any]) -> str:
    """how many of a binned code's bins hold values, and what's beyond those drawn"""
    if b["n"] == 0:
        return ""
    nb = len(b["breaks"]) + 1
    if b.get("bins") is not None:
        said = f"{sum(n > 0 for n in b['bins'])} of {nb} bins hold values"
    else:  # cut points that tie leave the bins between them empty
        can = 1 + len({x for x in b["breaks"] if x is not None})
        said = f"{can} of {nb} bins can hold values"
    h = b.get("hist") or {}
    beyond = [
        f"{fmt_int(h[k])} {where}"
        for k, where in (("under", "below"), ("over", "above"))
        if h.get(k)
    ]
    return said + (f" · {', '.join(beyond)} the range drawn" if beyond else "")


def tied(breaks: list[float | None], k: int) -> bool:
    """whether bin `k`'s cut points tie, so that no value can fall in it"""
    lo, hi = (breaks[k - 1], breaks[k]) if 0 < k < len(breaks) else (None, None)
    return lo is not None and hi is not None and lo >= hi


def bin_range(breaks: list[float | None], k: int) -> str:
    """the values bin `k` holds, from its lower cut point up to its upper one"""
    lo = breaks[k - 1] if k > 0 else None
    hi = breaks[k] if k < len(breaks) else None
    if tied(breaks, k):
        return f"none, as its cut points tie at {fmt_num(lo)}"
    if lo is None and hi is None:
        return "every value"
    if lo is None:
        return f"below {fmt_num(hi)}"
    if hi is None:
        return f"{fmt_num(lo)} and above"
    return f"{fmt_num(lo)} to {fmt_num(hi)}"


def timeline_pdf(payload: dict[str, Any]) -> bytes:
    """a visualizer payload as a static, self-contained pdf"""
    return TimelinePdf(payload).to_bytes()


def vocabulary_pdf(payload: dict[str, Any]) -> bytes:
    """a vocabulary page's payload as a static, self-contained pdf"""
    return VocabularyPdf(payload).to_bytes()
