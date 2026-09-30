#!/usr/bin/env python3

"""
renders a subject's tokenized timeline as a self-contained html page
"""

import base64
import collections
import datetime
import fnmatch
import html
import http.server
import importlib.metadata as meta
import importlib.resources as resources
import json
import math
import pathlib
import re
import zoneinfo

import polars as pl
import yaml

from cocoa.configurable import Configurable

# a quantile bin fused onto a code, e.g. the Q3 of VTL//heart_rate_Q3; codes are
# lowercased after their prefix, so an uppercase Q can only have come from binning
FUSED_BIN = re.compile(r"_(Q\d+)(?=_|$)")
BIN_WORD = re.compile(r"^Q\d+$")
KINDS = {"BOS": "bos", "EOS": "eos", "TIME": "spacer", "CLCK": "clock", "UNK": "unk"}
FONTS = (("Gotham-Book.otf", "300 400"), ("Gotham-Medium.otf", "500 700"))
MARKER = re.compile(r"__COCOA_(TITLE|STYLE|SCRIPT|DATA|ICON)__")


class SubjectNotFoundError(LookupError):
    """the requested subject has no timeline in the processed data"""


class Visualizer(Configurable):
    """
    gathers one subject's timeline, vocabulary, and pass-through columns
    from a processed data directory and renders them as html
    """

    default_file = "visualization.yaml"

    def __init__(
        self,
        visualization_cfg: pathlib.Path | str = None,
        processed_data_home: pathlib.Path | str = None,
        **kwargs,
    ):
        super().__init__(visualization_cfg, **kwargs)
        self.processed_data_home = (
            pathlib.Path(processed_data_home).expanduser().resolve()
        )
        # the C loader reads a hospital-scale vocabulary several times faster
        # than OmegaConf, and a viewer is run interactively
        with open(self.processed_data_home / "tokenizer.yaml") as f:
            self.tkzr = yaml.load(
                f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader)
            )
        self.tkzr_cfg = self.tkzr.get("cfg") or {}
        self.decoder = {int(t): w for w, t in (self.tkzr.get("lookup") or {}).items()}
        self.bins = {k: list(v) for k, v in (self.tkzr.get("bins") or {}).items()}
        self.fused = bool(self.tkzr_cfg.get("fused", True))
        # every section of the config is optional, and an empty one (a yaml null,
        # as when all its entries are commented out) means none
        self.other_color = str(self.cfg.get("other_color") or "#737373")
        self.palette = [str(c) for c in self.cfg.get("palette") or ()]
        self.prefix_names = {
            str(k): str(v) for k, v in (self.cfg.get("prefixes") or {}).items() if v
        }
        self.lanes = [  # a lane needs prefixes; its name and color have defaults
            {
                "name": str(lane.get("name") or ", ".join(ps)),
                "color": str(lane.get("color") or self.other_color),
                "prefixes": ps,
            }
            for lane in self.cfg.get("lanes") or ()
            if (ps := [str(p) for p in lane.get("prefixes") or ()])
        ]
        descriptions = {
            str(k): str(v) for k, v in (self.cfg.get("descriptions") or {}).items() if v
        }
        self.exact = {
            k: v for k, v in descriptions.items() if not re.search(r"[*?[]", k)
        }
        self.patterns = [(k, v) for k, v in descriptions.items() if k not in self.exact]
        self.prefix_weight = self.get_prefix_weight()
        self.stray_colors = self.get_stray_colors()

    def get_timeline(self, subject_id: str) -> pl.DataFrame:
        """one row per token of the subject's timeline"""
        tt = pl.scan_parquet(self.processed_data_home / "tokens_times.parquet")
        df = tt.filter(pl.col("subject_id") == str(subject_id)).collect()
        if df.height == 0:
            examples = tt.select("subject_id").head(5).collect().to_series().to_list()
            raise SubjectNotFoundError(
                f"No timeline for subject_id={subject_id!r} in "
                f"{self.processed_data_home / 'tokens_times.parquet'}; "
                f"subjects there include {', '.join(map(repr, examples))}"
            )
        return df.head(1).explode(pl.exclude("subject_id"), empty_as_null=False)

    def get_subject(self, subject_id: str) -> dict | None:
        """the subject's row of subject_splits.parquet, if it has one"""
        f = self.processed_data_home / "subject_splits.parquet"
        if not f.exists():
            return None
        df = (
            pl.scan_parquet(f)
            .filter(pl.col("subject_id") == str(subject_id))
            .head(1)
            .collect()
        )
        return df.row(0, named=True) if df.height else None

    def get_values(self, subject_id: str, unit: str) -> dict | None:
        """
        the subject's numeric values from meds.parquet, keyed by (epoch in `unit`,
        code, bin), for timelines tokenized without include_numeric_values
        """
        f = self.processed_data_home / "meds.parquet"
        if not f.exists():
            return None
        df = (
            pl.scan_parquet(f)
            .filter(
                pl.col("subject_id") == str(subject_id),
                pl.col("numeric_value").is_finite(),
            )
            .select(pl.col("time").dt.epoch(unit), "code", "numeric_value")
            .sort(pl.all())
            .collect()
        )
        found = collections.defaultdict(list)
        for t, code, v in df.iter_rows():
            # binned as the tokenizer bins: the number of breaks at or below v
            b = sum(x is not None and x <= v for x in self.bins.get(code) or ())
            found[(t, code, b)].append(v)
        return found

    def get_winnowed(self, subject_id: str, split: str | None) -> dict | None:
        """where the winnower split this subject into past and future, if it did"""
        f = self.processed_data_home / f"{split}_for_inference.parquet"
        if split is None or not f.exists():
            return None
        df = (
            pl.scan_parquet(f)
            .filter(pl.col("subject_id") == str(subject_id))
            .head(1)
            .collect()
        )
        if df.height == 0 or "last_valid" not in df.columns:
            return None
        row = df.row(0, named=True)
        outcomes = [
            {
                "code": c[: -len("_past")],
                "past": row[c],
                "future": row.get(c[:-5] + "_future"),
            }
            for c, t in df.schema.items()
            if c.endswith("_past") and t == pl.Boolean
        ]
        return {
            "file": f.name,
            "last_valid": row["last_valid"],
            "n_future": len(row["tokens_future"])
            if row.get("tokens_future") is not None
            else None,
            "outcomes": outcomes,
        }

    def describe(self, code: str) -> str | None:
        """a configured description of `code`, exact keys before patterns"""
        if code in self.exact:
            return self.exact[code]
        return next((v for k, v in self.patterns if fnmatch.fnmatchcase(code, k)), None)

    def parse(self, word: str) -> dict:
        """split a vocabulary word into its prefix, code, bin, and text value"""
        prefix, sep, rest = word.partition("//")
        if not sep:
            prefix, rest = word if word in KINDS else "", word
        name, bin_, text = rest, None, None
        if self.fused and (m := FUSED_BIN.search(rest)):
            name, bin_ = rest[: m.start()], int(m.group(1)[1:])
            text = rest[m.end() + 1 :] or None
        return {
            "prefix": prefix,
            "code": f"{prefix}{sep}{name}" if sep else name,
            "name": name,
            "bin": bin_,
            "text": text,
        }

    def lane_order(self, prefix: str) -> tuple:
        """where a stray prefix's lane goes: tokenizer ordering, then name, UNK last"""
        ordering = list(self.tkzr_cfg.get("ordering", []))
        return (
            prefix == "UNK",
            ordering.index(prefix) if prefix in ordering else len(ordering),
            prefix,
        )

    def get_prefix_weight(self) -> collections.Counter:
        """training occurrences of each code prefix in the vocabulary"""
        counts = self.tkzr.get("counts") or {}  # before 26.9.1, each word counts 1
        weight = collections.Counter()
        for w in self.decoder.values():
            if "//" in w:
                weight[w.partition("//")[0]] += counts.get(w) or 1
        return weight

    def unclaimed_prefixes(self) -> list[str]:
        """vocabulary prefixes that no configured lane claims, in lane order"""
        claimed = {p for lane in self.lanes for p in lane["prefixes"]}
        return sorted(self.prefix_weight.keys() - claimed, key=self.lane_order)

    def get_stray_colors(self) -> dict:
        """
        palette colors for the vocabulary prefixes no lane claims, from most to
        least frequent in training, skipping colors that lanes present in the
        vocabulary wear; drawn from the whole vocabulary, so a prefix keeps its
        color across subjects
        """
        weight = self.prefix_weight
        worn = {
            lane["color"].lower()
            for lane in self.lanes
            if weight.keys() & set(lane["prefixes"])
        }
        free = [c for c in self.palette if c.lower() not in worn]
        ranked = sorted(
            self.unclaimed_prefixes(), key=lambda p: (-weight[p], self.lane_order(p))
        )
        return dict(zip(ranked, free))

    def get_lanes(self, prefixes: set) -> list[dict]:
        """configured lanes holding any of `prefixes`, then a lane per stray prefix"""
        lanes = [{**lane, "prefixes": list(lane["prefixes"])} for lane in self.lanes]
        configured = {p for lane in lanes for p in lane["prefixes"]}
        strays = sorted(prefixes - configured - {"BOS", "EOS"}, key=self.lane_order)
        lanes += [
            {
                "name": self.prefix_names.get(p) or p or "Other",
                "color": self.stray_colors.get(p) or self.other_color,
                "prefixes": [p],
            }
            for p in strays
        ]
        return [lane for lane in lanes if prefixes & set(lane["prefixes"])]

    def get_payload(self, subject_id: str) -> dict:
        """everything the page draws, as plain json-able data"""
        tl = self.get_timeline(subject_id)
        subject = self.get_subject(subject_id) or {}
        tz = tl.schema["times"].time_zone or "UTC"

        # consecutive tokens at one instant share an entry in `times`
        times = tl.select(
            pl.col("times").alias("t"),
            (pl.col("times").dt.epoch("us") / 1000).alias("ms"),
            pl.col("times").dt.strftime("%Y-%m-%d %H:%M:%S%.f %Z").alias("label"),
            pl.col("times").dt.strftime("%:z").alias("offset"),
        ).with_columns(new=pl.col("t") != pl.col("t").shift(1))
        time_idx = (times["new"].fill_null(True).cast(pl.Int64).cum_sum() - 1).to_list()
        uniq = times.filter(pl.col("new").fill_null(True))

        tokens = tl["tokens"].to_list()
        values = (
            tl["numeric_values"].to_list() if "numeric_values" in tl.columns else None
        )
        to_end = (
            tl["hours_to_end_time"].to_list()
            if "hours_to_end_time" in tl.columns
            else None
        )

        codes, code_idx, events, token_event = [], {}, [], []
        for i, tok in enumerate(tokens):
            word = self.decoder.get(tok, "UNK")
            last = events[-1] if events else None
            if (
                not self.fused
                and last is not None
                and last["kind"] in ("event", "unk")  # an unknown code keeps its bin
                and last["time"] == time_idx[i]
                and "//" not in word
                and word not in KINDS
            ):  # an unfused bin or text value belongs to the code before it
                if BIN_WORD.match(word) and last["bin"] is None:
                    last["bin"] = int(word[1:])
                else:
                    last["text"] = (
                        word if last["text"] is None else f"{last['text']}_{word}"
                    )
                last["n_tokens"] += 1
                token_event.append(len(events) - 1)
                continue
            parsed = self.parse(word)
            kind = KINDS.get(parsed["prefix"], "event")
            if (code := parsed["code"]) not in code_idx:
                code_idx[code] = len(codes)
                codes.append(
                    {
                        "code": code,
                        "prefix": parsed["prefix"],
                        "name": parsed["name"].replace("_", " "),
                        "category": self.prefix_names.get(parsed["prefix"]),
                        "description": self.describe(code),
                        "breaks": self.bins.get(code),
                    }
                )
            events.append(
                {
                    "kind": kind,
                    "time": time_idx[i],
                    "code": code_idx[code],
                    "bin": parsed["bin"],
                    "text": parsed["text"],
                    # a spacer carries the value of the event it precedes
                    "value": finite(values[i]) if values and kind == "event" else None,
                    "hours_to_end": finite(to_end[i]) if to_end else None,
                    "first_token": i,
                    "n_tokens": 1,
                }
            )
            token_event.append(len(events) - 1)

        source = "tokens" if values is not None else None
        unit = tl.schema["times"].time_unit
        if values is None and (found := self.get_values(subject_id, unit)):
            epochs = uniq["t"].dt.epoch(unit).to_list()
            for e in events:  # tokens of one (instant, code, bin) are interchangeable
                if e["kind"] == "event" and e["bin"] is not None:
                    key = (epochs[e["time"]], codes[e["code"]]["code"], e["bin"])
                    if found.get(key):
                        e["value"] = finite(found[key].pop(0))
            if any(e["value"] is not None for e in events):
                source = "meds"

        lanes = self.get_lanes({c["prefix"] for c in codes})
        lane_of = {p: j for j, lane in enumerate(lanes) for p in lane["prefixes"]}
        for c in codes:
            c["lane"] = lane_of.get(c["prefix"])

        split = subject.get("split")
        static = {"subject_id", "split", "start_time", "end_time"}
        return {
            "meta": {
                "cocoa_version": meta.version("cocoa-tokenizer"),
                "generated": datetime.datetime.now(zoneinfo.ZoneInfo("America/Chicago"))
                .replace(microsecond=0)
                .isoformat(),
                "tokenizer_created": self.tkzr.get("created_dttm"),
                "processed_data_home": str(self.processed_data_home),
                "timezone": tz,
                "time_unit": tl.schema["times"].time_unit,
                "fused": self.fused,
                "n_bins": self.tkzr_cfg.get("n_bins"),
                "vocab_size": len(self.decoder),
                "has_numeric_values": source is not None,
                "value_source": source,
                "has_hours_to_end": to_end is not None,
            },
            "subject": {
                "subject_id": str(subject_id),
                "split": split,
                "start_time": fmt_time(subject.get("start_time")),
                "end_time": fmt_time(subject.get("end_time")),
                "fields": [
                    {"name": k, "label": humanize(k), "value": fmt_value(v)}
                    for k, v in subject.items()
                    if k not in static
                ],
            },
            "lanes": lanes,
            "codes": codes,
            "times": {
                "ms": uniq["ms"].to_list(),
                "label": uniq["label"].to_list(),
                "offset": uniq["offset"].to_list(),
            },
            "events": {
                k: [e[k] for e in events] for k in (events[0] if events else {})
            },
            "tokens": {
                "id": tokens,
                "event": token_event,
                "vocab": {
                    str(t): self.decoder.get(t, "UNK") for t in dict.fromkeys(tokens)
                },
            },
            "winnowed": self.get_winnowed(subject_id, split),
        }

    def render(self, subject_id: str) -> str:
        """the subject's timeline as one self-contained html document"""
        payload = self.get_payload(subject_id)
        assets = resources.files("cocoa.assets")
        fill = {
            "TITLE": html.escape(f"Subject {subject_id} · cocoa"),
            "STYLE": font_faces()
            + (assets / "timeline.css").read_text(encoding="utf-8"),
            "SCRIPT": (assets / "timeline.js").read_text(encoding="utf-8"),
            "DATA": to_script_json(payload),
            "ICON": "data:image/svg+xml;base64,"
            + base64.b64encode((assets / "cocoa.svg").read_bytes()).decode(),
        }
        template = (assets / "timeline.html").read_text(encoding="utf-8")
        # one pass, so nothing substituted in is itself scanned for markers
        return MARKER.sub(lambda m: fill[m.group(1)], template)

    def save(self, subject_id: str, path: pathlib.Path | str) -> pathlib.Path:
        """write the rendered page to `path`"""
        to_file = pathlib.Path(path).expanduser().resolve()
        to_file.parent.mkdir(parents=True, exist_ok=True)
        to_file.write_text(self.render(subject_id), encoding="utf-8")
        return to_file


def finite(x) -> float | None:
    """a json-safe float: NaN and infinities become null"""
    return float(x) if x is not None and math.isfinite(x) else None


def humanize(column: str) -> str:
    """a readable label for a pass-through column, e.g. race_category -> Race"""
    label = re.sub(r"_category$", "", column).replace("_", " ").strip()
    return label[:1].upper() + label[1:]


def fmt_time(t) -> dict | None:
    """a datetime as epoch milliseconds plus an exact, zone-labeled string"""
    if not isinstance(t, datetime.datetime):
        return None
    return {
        "ms": t.timestamp() * 1000,
        "label": t.strftime("%Y-%m-%d %H:%M:%S")
        + (f".{t.microsecond:06d}".rstrip("0") if t.microsecond else "")
        + (f" {t.tzname()}" if t.tzinfo else ""),
    }


def fmt_value(v) -> str | None:
    """a pass-through value as display text"""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else f"{v:.6g}"
    if isinstance(v, datetime.datetime):
        return fmt_time(v)["label"]
    return str(v)


def to_script_json(obj) -> str:
    """json that can sit inside a <script> element without closing it early"""
    return (
        json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def font_faces() -> str:
    """@font-face rules embedding whichever packaged fonts are present"""
    fonts = resources.files("cocoa.assets") / "fonts"
    return "".join(
        '@font-face{font-family:"Gotham";src:url(data:font/otf;base64,'
        + base64.b64encode((fonts / f).read_bytes()).decode()
        + f') format("opentype");font-weight:{w};font-display:swap}}\n'
        for f, w in FONTS
        if (fonts / f).is_file()
    )


def make_server(
    page: str, host: str = "127.0.0.1", port: int = 8765
) -> http.server.ThreadingHTTPServer:
    """an http server that answers with `page` at / until shut down"""
    body = page.encode("utf-8")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.respond(include_body=True)

        def do_HEAD(self):
            self.respond(include_body=False)

        def respond(self, include_body: bool):
            if self.path.split("?")[0] not in ("/", "/index.html"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if include_body:
                self.wfile.write(body)

        def log_message(self, *args):
            pass  # keep the console quiet

    return http.server.ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    self = Visualizer(processed_data_home="./processed/mimic/")
    sid = (
        pl.scan_parquet(self.processed_data_home / "tokens_times.parquet")
        .select("subject_id")
        .head(1)
        .collect()
        .item()
    )
    payload = self.get_payload(sid)
    assert len(payload["tokens"]["id"]) == len(payload["tokens"]["event"])
    page = self.render(sid)
    assert not re.search(r"(?:src|href)=[\"']?https?:|url\([\"']?https?:", page)
    print(self.save(sid, self.processed_data_home / f"timeline_{sid}.html"))
    # breakpoint()
