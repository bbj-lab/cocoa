#!/usr/bin/env python3

"""
draws a subject's timeline as a static, self-contained pdf, from the payload the
visualizer builds for its html page
"""

import datetime
import functools
import importlib.resources as resources
import struct
import zlib
import zoneinfo

# a landscape us letter page in points, laid out from its top-left corner
PAGE_W, PAGE_H = 792.0, 612.0
MARGIN = 36.0
BOTTOM = PAGE_H - 40  # where content stops, leaving room for the footer
GUTTER = 150.0  # lane and code labels, left of the plot
PLOT_X = MARGIN + GUTTER + 8
PLOT_W = PAGE_W - MARGIN - PLOT_X
AXIS_H = 36.0
LANE_H, CODE_H, BIN_H = 18.0, 13.0, 28.0
ROW_H = 9.5  # a row of the events table
ROW_CAP = 2000  # events the table lists, as the page's table does
GAP = 6.0  # between table columns

# the page's colors, its translucent washes flattened onto white
INK, INK2, MUTED, AXIS = "#000000", "#737373", "#A6A6A6", "#A6A6A6"
GRID, GRID_SOFT, WASH, FUTURE = "#D9D9D9", "#ECECEC", "#FAFAFA", "#F6F6F6"
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

    def __init__(self, data: bytes, name: str):
        self.data, self.name = data, name
        n = struct.unpack_from(">H", data, 4)[0]
        tables = {
            tag: offset
            for tag, _, offset, _ in struct.iter_unpack(
                ">4sIII", data[12 : 12 + 16 * n]
            )
        }

        def read(tag: bytes, fmt: str, at: int = 0) -> tuple:
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

    def glyphs(self, s) -> list[int]:
        """the glyph of each character of `s`; what the font lacks becomes ?"""
        return [self.cmap.get(ord(c) if c >= " " else 32, self.missing) for c in str(s)]

    @functools.cached_property
    def program(self) -> bytes:
        return zlib.compress(self.data, 9)

    @functools.cached_property
    def to_unicode(self) -> bytes:
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


def to_ansi(s) -> bytes:
    """text in WinAnsiEncoding, one byte a character; what it lacks becomes ?"""
    return "".join(c if c >= " " else " " for c in str(s)).encode("cp1252", "replace")


def char_widths(s, font: str) -> list[int]:
    """the advance of each character of `s` in `font`, in thousandths of its size"""
    if f := face(font):
        return [f.widths[g] for g in f.glyphs(s)]
    b = to_ansi(s)
    if font == "F3":
        return [600] * len(b)
    table = HELVETICA_BOLD if font == "F2" else HELVETICA
    return [table[c - 32] if 32 <= c <= 126 else WIDE.get(c, 556) for c in b]


def encode(s, font: str) -> str:
    """`s` as a pdf string in `font`: glyph ids if it is embedded, else winansi"""
    if f := face(font):
        return "<" + "".join(f"{g:04X}" for g in f.glyphs(s)) + ">"
    return literal(to_ansi(s))


def measure(s, size: float, font: str = "F1") -> float:
    """the width of `s` set in `font` at `size`, in points"""
    return sum(char_widths(s, font)) * size / 1000


def clip(s, size: float, font: str, width: float) -> str:
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


def add_font(font: str, add) -> int:
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

    def __init__(self):
        self.ops = []

    def rect(self, x, y, w, h, color):
        """a filled rectangle"""
        y = PAGE_H - y - h
        self.ops.append(f"{rgb(color)} rg {x:.2f} {y:.2f} {w:.2f} {h:.2f} re f")

    def frame(self, x, y, w, h, color, width=0.5):
        """an outlined rectangle"""
        y = PAGE_H - y - h
        self.ops.append(
            f"{rgb(color)} RG {width:.2f} w {x:.2f} {y:.2f} {w:.2f} {h:.2f} re S"
        )

    def segments(self, segs, color, width=1.0, round_cap=False):
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

    def polyline(self, points, color, width=1.0):
        """a stroked line through `points`"""
        if len(points) < 2:
            return
        (x0, y0), *rest = points
        path = f"{x0:.2f} {PAGE_H - y0:.2f} m " + " ".join(
            f"{x:.2f} {PAGE_H - y:.2f} l" for x, y in rest
        )
        self.ops.append(f"{rgb(color)} RG {width:.2f} w 1 J 1 j {path} S")

    def text(self, x, y, s, size=8.0, font="F1", color=INK, align="left", width=None):
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

    def __init__(self, title: str, producer: str = "cocoa"):
        self.title = title
        self.producer = producer
        self.pages = []

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

        def add(obj) -> int:
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


def fmt_int(n) -> str:
    return f"{n:,}"


def fmt_num(v, digits: int = 4) -> str:
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


class TimelinePdf:
    """
    lays a visualizer payload out as pages: the subject and its whole timeline
    first, then every code on a row of its own, then the events
    """

    def __init__(self, payload: dict):
        self.d = d = payload
        self.meta, self.sub = d["meta"], d["subject"]
        self.E, self.C, self.L = d["events"], d["codes"], d["lanes"]
        self.labels = d["times"]["label"]
        self.n_e = len(self.E.get("kind") or ())
        self.n_t = len(d["tokens"]["id"])
        self.zone_name = self.meta.get("timezone") or "UTC"
        try:
            self.zone = zoneinfo.ZoneInfo(self.zone_name)
        except (zoneinfo.ZoneInfoNotFoundError, ValueError):
            self.zone_name, self.zone = "UTC", datetime.timezone.utc
        self.doc = Document(
            f"Subject {self.sub['subject_id']} · cocoa",
            f"cocoa {self.meta.get('cocoa_version', '')}".strip(),
        )
        self.page = self.doc.new_page()
        self.y = MARGIN
        if self.n_e:
            self.index()

    def index(self):
        """what the drawing needs from the events, as the page derives it"""
        E, times = self.E, self.d["times"]["ms"]
        self.ems = [times[i] for i in E["time"]]
        tms = [self.ems[e] for e in self.d["tokens"]["event"]]
        self.ends = [e for e, k in enumerate(E["kind"]) if k in ("bos", "eos")]
        self.code_ev = [[] for _ in self.C]
        self.lane_ev = [[] for _ in self.L]
        for e, c in enumerate(E["code"]):
            self.code_ev[c].append(e)
            if self.C[c]["lane"] is not None:
                self.lane_ev[self.C[c]["lane"]].append(e)
        self.lane_codes = [[] for _ in self.L]
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

    def make_ticks(self):
        """axis ticks on round wall-clock times, about every 85 points"""
        v0, v1 = self.ext0, self.ext1
        want = (v1 - v0) / max(1, PLOT_W / 85)
        ms, mo, _ = next((s for s in STEPS if s[2] >= want), STEPS[-1])
        out = []
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

    # ---- flow ---------------------------------------------------------------------

    def new_page(self):
        self.page = self.doc.new_page()
        self.y = MARGIN

    def need(self, h: float):
        """start a new page unless `h` more points fit on this one"""
        if self.y + h > BOTTOM:
            self.new_page()

    def heading(self, title: str, note: str = None):
        self.need(40)
        self.page.text(MARGIN, self.y + 11, title, 11, "F2")
        if note:
            self.page.text(
                PAGE_W - MARGIN, self.y + 11, note, 7, color=INK2, align="right"
            )
        self.y += 20

    def paragraph(self, s: str, size: float = 7.0, color: str = INK2):
        for line in wrap(s, size, "F1", PAGE_W - 2 * MARGIN):
            self.need(size * 1.5)
            self.page.text(MARGIN, self.y + size, line, size, color=color)
            self.y += size * 1.5

    def definitions(self, pairs, bold: bool = True):
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

    # ---- sections -----------------------------------------------------------------

    def header(self):
        sub, p = self.sub, self.page
        title = f"Subject {sub['subject_id']}"
        p.text(MARGIN, self.y + 15, title, 17, "F2", ACCENT, width=PAGE_W - 2 * MARGIN)
        if sub.get("split"):
            x = MARGIN + measure(title, 17, "F2") + 9
            w = measure(sub["split"], 7.5) + 12
            p.frame(x, self.y + 4, w, 12, AXIS)
            p.text(x + 6, self.y + 13, sub["split"], 7.5, color=INK2)
        self.y += 26
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

    def overview(self):
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

    def every_code(self):
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

    def figure(self, rows):
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

    def axis(self, top):
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
        taken = []

        def place(s, x, align):
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

    def plot(self, rows, y0, y1):
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

    def draw_ticks(self, r):
        pad = max(2.0, r["h"] * 0.2)
        xs = sorted({round(self.X(self.ems[e]) * 4) / 4 for e in r["ev"]})
        segs = [(x, r["y"] + pad, x, r["y"] + r["h"] - pad) for x in xs]
        self.page.segments(segs, self.L[r["lane"]]["color"], 1.0)

    def draw_binned(self, r):
        p, color, top, h = self.page, self.L[r["lane"]]["color"], r["y"], r["h"]

        def y_bin(b):
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
            cols = {}
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

    def label(self, r):
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

    def winnowing(self):
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

    def table(self):
        E, toks = self.E, self.d["tokens"]
        vocab = toks["vocab"]

        def tokens_of(e):
            first = E["first_token"][e]
            ids = toks["id"][first : first + E["n_tokens"][e]]
            return "  ".join(f"{t} {vocab.get(str(t), 'UNK')}" for t in ids)

        cols = [
            ("#", 34, "right", "F1", lambda e: fmt_int(E["first_token"][e])),
            ("Time", 112, "left", "F1", lambda e: self.labels[E["time"][e]]),
            ("Lane", 96, "left", "F1", None),
            ("Code", 150, "left", "F3", lambda e: self.C[E["code"][e]]["code"]),
            ("Bin", 22, "right", "F1", lambda e: _q(E["bin"][e])),
        ]
        if self.meta.get("has_numeric_values"):
            cols.append(
                ("Value", 48, "right", "F1", lambda e: fmt_num(E["value"][e], 6))
            )
        cols.append(("Text", 70, "left", "F1", lambda e: E["text"][e] or ""))
        used = sum(c[1] for c in cols) + GAP * len(cols)
        cols.append(("Token", PAGE_W - 2 * MARGIN - used, "left", "F3", tokens_of))

        n = min(self.n_e, ROW_CAP)
        note = (
            f"the first {fmt_int(n)} of {fmt_int(self.n_e)} events; "
            "the html page lists them all"
            if n < self.n_e
            else f"{fmt_int(self.n_e)} events"
        )
        self.new_page()
        self.heading("Events", note)

        def col_head():
            x = MARGIN
            for name, w, align, *_ in cols:
                at = x + w if align == "right" else x
                self.page.text(at, self.y + 7, name, 6.5, "F2", INK2, align)
                x += w + GAP
            self.y += 10
            line = (MARGIN, self.y, PAGE_W - MARGIN, self.y)
            self.page.segments([line], GRID, 0.5)

        col_head()
        lines = []
        for e in range(n):
            if self.y + ROW_H > BOTTOM:
                self.page.segments(lines, GRID_SOFT, 0.4)
                self.new_page()
                col_head()
                lines = []
            base, x = self.y + 6.8, MARGIN
            for name, w, align, font, get in cols:
                if get is None:  # the lane, with its color
                    lane = self.C[E["code"][e]]["lane"]
                    if lane is not None:
                        self.page.rect(x, base - 4.6, 5, 5, self.L[lane]["color"])
                        lane_name = self.L[lane]["name"]
                        self.page.text(x + 8, base, lane_name, 6.5, width=w - 8)
                else:
                    at = x + w if align == "right" else x
                    self.page.text(at, base, get(e), 6.5, font, align=align, width=w)
                x += w + GAP
            self.y += ROW_H
            lines.append((MARGIN, self.y, PAGE_W - MARGIN, self.y))
        self.page.segments(lines, GRID_SOFT, 0.4)

    def footers(self):
        m = self.meta
        items = [
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
        line = " · ".join(s for s in items if s)
        n = len(self.doc.pages)
        for i, page in enumerate(self.doc.pages, 1):
            num = f"Page {i} of {n}"
            page.text(PAGE_W - MARGIN, PAGE_H - 20, num, 6.5, color=INK2, align="right")
            room = PAGE_W - 2 * MARGIN - measure(num, 6.5) - 12
            page.text(MARGIN, PAGE_H - 20, line, 6.5, color=ACCENT, width=room)

    def to_bytes(self) -> bytes:
        self.header()
        if not self.n_e:
            self.paragraph("This subject's timeline has no events.")
        else:
            self.overview()
            self.winnowing()
            self.every_code()
            self.table()
        self.footers()
        return self.doc.to_bytes()


def _q(b) -> str:
    return f"Q{b}" if b is not None else ""


def timeline_pdf(payload: dict) -> bytes:
    """a visualizer payload as a static, self-contained pdf"""
    return TimelinePdf(payload).to_bytes()
