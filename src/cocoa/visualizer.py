#!/usr/bin/env python3

"""
renders a subject's tokenized timeline, or the learned vocabulary with its bins
over the training data, as a self-contained html page
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
from typing import Any, Literal

import polars as pl
import yaml

from cocoa.configurable import Configurable
from cocoa.pdf import timeline_pdf, vocabulary_pdf

# a quantile bin fused onto a code, e.g. the Q3 of VTL//heart_rate_Q3; codes are
# lowercased after their prefix, so an uppercase Q can only have come from binning
FUSED_BIN = re.compile(r"_(Q\d+)(?=_|$)")
BIN_WORD = re.compile(r"^Q\d+$")
KINDS = {"BOS": "bos", "EOS": "eos", "TIME": "spacer", "CLCK": "clock", "UNK": "unk"}
FONTS = (("Gotham-Book.otf", "300 400"), ("Gotham-Medium.otf", "500 700"))
MARKER = re.compile(r"__COCOA_(TITLE|STYLE|SCRIPT|DATA|ICON)__")

# the vocabulary page counts a binned code with at most EXACT distinct training
# values value by value, and one with more in about BUCKETS buckets of a round
# width. these span its cut points and all but the TAIL of its values at either
# end, out to its extreme value where that's within REACH of the span; EPS of a
# width lands a value read from float32, as 1.4 is 1.39999998, in the bucket its
# decimal reading starts
EXACT, BUCKETS, TAIL, REACH, EPS = 50, 50, 0.005, 0.25, 1e-3
PER_DECADE = 8  # buckets per power of ten, for the timelines' lengths and durations
SPLITS = ("train", "tuning", "held_out")


class SubjectNotFoundError(LookupError):
    """the requested subject has no timeline in the processed data"""


class Visualizer(Configurable):
    """
    gathers one subject's timeline, vocabulary, and pass-through columns
    from a processed data directory and renders them as html or pdf;
    `save` and `save_pdf` write them to files. `render_vocabulary` and
    `render_vocabulary_pdf` draw the vocabulary instead, and
    `save_vocabulary` and `save_vocabulary_pdf` write it: its words, each
    binned code's training values against its bins, and the timelines by split
    """

    default_file = "visualization.yaml"

    def __init__(
        self,
        visualization_cfg: pathlib.Path | str | None = None,
        processed_data_home: pathlib.Path | str | None = None,
        **kwargs: Any,
    ) -> None:
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
        if not isinstance(self.tkzr, dict):  # 26.6.0 and 26.6.1 crash writing it
            raise ValueError(
                f"{self.processed_data_home / 'tokenizer.yaml'} holds no tokenizer; "
                "cocoa-tokenizer 26.6.0 and 26.6.1 fail before writing it, so "
                "retokenize with a later version"
            )
        self.tkzr_cfg = self.tkzr.get("cfg") or {}
        self.decoder = {int(t): w for w, t in (self.tkzr.get("lookup") or {}).items()}
        self.bins = {k: list(v) for k, v in (self.tkzr.get("bins") or {}).items()}
        # the tokenizer records `fused`; a yaml without it predates that, when a
        # config omitting it tokenized unfused
        self.fused = bool(self.tkzr_cfg.get("fused", False))
        # every section of the config is optional, and an empty one (a yaml null,
        # as when all its entries are commented out) means none
        self.other_color = str(self.cfg.get("other_color") or "#737373")
        self.palette = [str(c) for c in self.cfg.get("palette") or ()]
        self.prefix_names = {
            str(k): str(v) for k, v in (self.cfg.get("prefixes") or {}).items() if v
        }
        self.lanes: list[dict[str, Any]]
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
        self.show_winnowing = bool(self.cfg.get("show_winnowing") or False)
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
        df = df.head(1).explode(pl.exclude("subject_id"), empty_as_null=False)
        if df.schema["times"].time_zone is None:  # before 26.9.0, times were naive utc
            df = df.with_columns(pl.col("times").dt.replace_time_zone("UTC"))
        return df

    def get_subject(self, subject_id: str) -> dict[str, Any] | None:
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

    def get_values(
        self, subject_id: str, unit: Literal["ms", "us", "ns"]
    ) -> collections.defaultdict[tuple[int, str, int], list[float]] | None:
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

    def get_winnowed(self, subject_id: str, split: str | None) -> dict[str, Any] | None:
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

    def parse(self, word: str) -> dict[str, Any]:
        """split a vocabulary word into its prefix, code, bin, and text value"""
        prefix, sep, rest = word.partition("//")
        if not sep:
            prefix, rest = word if word in KINDS else "", word
        name, bin_, text = rest, None, None
        if self.fused and (m := FUSED_BIN.search(rest)):
            name, bin_ = rest[: m.start()], int(m.group(1)[1:])
            text = rest[m.end() + 1 :] or None
        if not sep and name in self.prefix_weight:
            # a bare prefix, as collation wrote for a null code before 26.6.0
            prefix, name = name, ""
        return {
            "prefix": prefix,
            "code": f"{prefix}{sep}{name}" if sep else name or prefix,
            "name": name,
            "bin": bin_,
            "text": text,
        }

    def lane_order(self, prefix: str) -> tuple[bool, int, str]:
        """where a stray prefix's lane goes: tokenizer ordering, then name, UNK last"""
        ordering = list(self.tkzr_cfg.get("ordering", []))
        return (
            prefix == "UNK",
            ordering.index(prefix) if prefix in ordering else len(ordering),
            prefix,
        )

    def get_prefix_weight(self) -> collections.Counter[str]:
        """training occurrences of each code prefix in the vocabulary"""
        counts = self.tkzr.get("counts") or {}  # before 26.9.1, each word counts 1
        weight: collections.Counter[str] = collections.Counter()
        for w in self.decoder.values():
            if "//" in w:
                weight[w.partition("//")[0]] += counts.get(w) or 1
        return weight

    def unclaimed_prefixes(self) -> list[str]:
        """vocabulary prefixes that no configured lane claims, in lane order"""
        claimed = {p for lane in self.lanes for p in lane["prefixes"]}
        return sorted(self.prefix_weight.keys() - claimed, key=self.lane_order)

    def get_stray_colors(self) -> dict[str, str]:
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

    def get_lanes(self, prefixes: set[str]) -> list[dict[str, Any]]:
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

    def get_payload(self, subject_id: str) -> dict[str, Any]:
        """everything the page draws, as plain json-able data"""
        tl = self.get_timeline(subject_id)
        subject = self.get_subject(subject_id) or {}
        tz = tl.schema["times"].time_zone

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

        codes: list[dict[str, Any]]
        events: list[dict[str, Any]]
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
                and word not in self.prefix_weight  # a bare prefix is a code
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
                "generated": now(),
                "tokenizer_created": self.tkzr.get("created_dttm"),
                "processed_data_home": str(self.processed_data_home),
                "timezone": tz,
                "time_unit": tl.schema["times"].time_unit,
                "fused": self.fused,
                "n_bins": self.tkzr_cfg.get("n_bins"),
                "vocab_size": len(self.decoder),
                "has_numeric_values": source is not None,
                "value_source": source,
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
            "winnowed": self.get_winnowed(subject_id, split)
            if self.show_winnowing
            else None,
        }

    def render(self, subject_id: str) -> str:
        """the subject's timeline as one self-contained html document"""
        return render_page(
            "timeline", f"Subject {subject_id} · cocoa", self.get_payload(subject_id)
        )

    def save(self, subject_id: str, path: pathlib.Path | str) -> pathlib.Path:
        """write the rendered page to `path`"""
        to_file = pathlib.Path(path).expanduser().resolve()
        to_file.parent.mkdir(parents=True, exist_ok=True)
        to_file.write_text(self.render(subject_id), encoding="utf-8")
        return to_file

    def render_pdf(self, subject_id: str) -> bytes:
        """the subject's timeline as a static, self-contained pdf"""
        return timeline_pdf(self.get_payload(subject_id))

    def save_pdf(self, subject_id: str, path: pathlib.Path | str) -> pathlib.Path:
        """write the subject's timeline to `path` as a pdf"""
        to_file = pathlib.Path(path).expanduser().resolve()
        to_file.parent.mkdir(parents=True, exist_ok=True)
        to_file.write_bytes(self.render_pdf(subject_id))
        return to_file

    def get_training_values(self) -> pl.LazyFrame | None:
        """
        the finite numeric values of the training split's events, by code, as the
        tokenizer learned its bins on and bins them; None without meds.parquet or
        subject_splits.parquet
        """
        meds = self.processed_data_home / "meds.parquet"
        splits = self.processed_data_home / "subject_splits.parquet"
        if not (meds.exists() and splits.exists()):
            return None
        train = pl.scan_parquet(splits).filter(pl.col("split") == "train")
        return (
            pl.scan_parquet(meds)
            .join(train.select("subject_id"), on="subject_id", how="semi")
            .select("code", value=pl.col("numeric_value").cast(pl.Float64))
            .filter(pl.col("value").is_finite())
        )

    def get_histograms(self) -> dict[str, dict[str, Any]]:
        """
        each binned code's training values: how many, how many distinct, their
        range, a histogram of them, and how many fall in each of its bins
        """
        values = self.get_training_values()
        if values is None or not self.bins:
            return {}
        values = values.filter(pl.col("code").is_in(list(self.bins)))
        v = pl.col("value")
        stats = values.group_by("code").agg(
            n=pl.len(),
            distinct=v.n_unique(),
            whole=(v == v.round()).all(),
            min=v.min(),
            max=v.max(),
            mean=v.sort().mean(),  # summed in one order, whatever order they come in
            median=v.median(),
            lo=v.quantile(TAIL, "linear"),
            hi=v.quantile(1 - TAIL, "linear"),
        )
        stats = stats.collect(engine="streaming")
        exact, spans = [], []
        for s in stats.iter_rows(named=True):
            if s["distinct"] <= EXACT:
                exact.append(s["code"])
                continue
            cuts = [b for b in map(finite, self.bins[s["code"]]) if b is not None]
            lo, hi = min([s["lo"], *cuts]), max([s["hi"], *cuts])
            if hi <= lo:  # nearly every value is the same, so span them all
                lo, hi = min([s["min"], *cuts]), max([s["max"], *cuts])
            reach = REACH * (hi - lo)
            lo = min(lo, s["min"]) if lo - s["min"] <= reach else lo
            hi = max(hi, s["max"]) if s["max"] - hi <= reach else hi
            w = round_width((hi - lo) / BUCKETS, whole=s["whole"])
            lo = math.floor(lo / w) * w
            spans.append((s["code"], lo, w, math.floor((hi - lo) / w + EPS) + 1))

        n_cuts = max(map(len, self.bins.values()))
        cut_points = pl.LazyFrame(
            [
                [k, *(None if x is None else float(x) for x in b)]
                + [None] * (n_cuts - len(b))
                for k, b in self.bins.items()
            ],
            schema={
                "code": pl.String,
                **{f"break_{i}": pl.Float64 for i in range(1, n_cuts + 1)},
            },
            orient="row",
        )
        # as the tokenizer's bin_data bins: the number of breaks at or below a value
        binned = pl.sum_horizontal(
            pl.lit(0, pl.UInt32),
            *(pl.col(f"break_{i}") <= v for i in range(1, n_cuts + 1)),
        )
        spec = pl.LazyFrame(
            spans,
            schema={
                "code": pl.String,
                "lo": pl.Float64,
                "w": pl.Float64,
                "nb": pl.Int64,
            },
            orient="row",
        )
        # -1 for a value below the histogram's span, and nb for one above it
        bucket = (
            ((v - pl.col("lo")) / pl.col("w") + EPS)
            .clip(-1, pl.col("nb"))
            .floor()
            .cast(pl.Int64)
        )
        by_bin, by_value, by_bucket = pl.collect_all(
            [
                values.join(cut_points, on="code").group_by("code", bin=binned).len(),
                values.filter(pl.col("code").is_in(exact))
                .group_by("code", "value")
                .len()
                .sort("code", "value"),
                values.join(spec, on="code").group_by("code", bucket=bucket).len(),
            ],
            engine="streaming",
        )

        found = {
            s["code"]: {
                "n": s["n"],
                "distinct": s["distinct"],
                **{k: finite(s[k]) for k in ("min", "median", "mean", "max")},
                "bins": [0] * (len(self.bins[s["code"]]) + 1),
            }
            for s in stats.iter_rows(named=True)
        }
        for code, b, n in by_bin.iter_rows():
            found[code]["bins"][b] = n
        for code, x, n in by_value.iter_rows():
            hist = found[code].setdefault("hist", {"x": [], "n": []})
            hist["x"].append(x)
            hist["n"].append(n)
        for code, lo, w, nb in spans:
            found[code]["hist"] = {
                "lo": lo,
                "w": w,
                "n": [0] * nb,
                "under": 0,
                "over": 0,
            }
        for code, i, n in by_bucket.iter_rows():
            hist = found[code]["hist"]
            if i < 0:
                hist["under"] = n
            elif i >= len(hist["n"]):
                hist["over"] = n
            else:
                hist["n"][i] = n
        return found

    def get_timeline_stats(self) -> dict[str, Any] | None:
        """
        the timelines' lengths, durations, and tokens out of the vocabulary,
        overall and by split, as `tokenize --verbose` summarizes them; None
        without tokens_times.parquet, as where a tokenizer.yaml was copied alone
        """
        timelines = self.processed_data_home / "tokens_times.parquet"
        splits = self.processed_data_home / "subject_splits.parquet"
        if not timelines.exists():
            return None
        df = pl.scan_parquet(timelines).select(
            "subject_id",
            length=pl.col("tokens").list.len(),
            unk=pl.col("tokens").list.count_matches(0),
            start=pl.col("times").list.min(),
            end=pl.col("times").list.max(),
        )
        df = (
            df.join(
                pl.scan_parquet(splits).select("subject_id", "split"),
                on="subject_id",
                how="left",
            )
            if splits.exists()
            else df.with_columns(split=pl.lit(None, pl.String))
        )
        # in one order, so that means and deviations sum the same way every run
        df = (
            df.with_columns(
                hours=(pl.col("end") - pl.col("start")).dt.total_microseconds() / 36e8
            )
            .collect(engine="streaming")
            .sort(pl.all())
            .rechunk()
        )
        if df.schema["start"].time_zone is None:  # before 26.9.0, times were naive utc
            df = df.with_columns(pl.col("start", "end").dt.replace_time_zone("UTC"))

        def buckets(col: str) -> pl.Expr:
            return (pl.col(col).log10() * PER_DECADE + 1e-6).floor().cast(pl.Int64)

        spans = {
            col: df.filter(pl.col(col) > 0).select(
                lo=buckets(col).min(), hi=buckets(col).max()
            )
            for col in ("length", "hours")
        }

        def counts(g: pl.DataFrame, col: str) -> list[int]:
            lo, hi = spans[col].row(0)
            found = dict(
                g.filter(pl.col(col) > 0).group_by(k=buckets(col)).len().iter_rows()
            )
            return (
                [found.get(k, 0) for k in range(lo, hi + 1)] if lo is not None else []
            )

        def summary(g: pl.DataFrame) -> dict[str, Any]:
            return {
                "timelines": g.height,
                "tokens": int(g["length"].sum()),
                "unk": int(g["unk"].sum()),
                "length": describe(g["length"]),
                "hours": describe(g["hours"]),
                "instant": int((g["hours"] <= 0).sum()),
                "first": fmt_time(g["start"].min()),
                "last": fmt_time(g["end"].max()),
                "lengths": counts(g, "length"),
                "durations": counts(g, "hours"),
            }

        order = {s: i for i, s in enumerate(SPLITS)}
        splits = sorted(
            df["split"].unique().to_list(),
            key=lambda s: (s is None, order.get(s, len(order)), s or ""),
        )
        return {
            "per_decade": PER_DECADE,
            "length_from": spans["length"].item(0, "lo"),
            "hours_from": spans["hours"].item(0, "lo"),
            "timezone": df.schema["start"].time_zone,
            "all": summary(df),
            "splits": [
                {
                    "split": s,
                    **summary(
                        df.filter(
                            pl.col("split").is_null()
                            if s is None
                            else pl.col("split") == s
                        )
                    ),
                }
                for s in splits
            ],
        }

    def get_vocabulary_payload(self) -> dict[str, Any]:
        """everything the vocabulary page draws, as plain json-able data"""
        counts = self.tkzr.get("counts") or {}  # before 26.9.1, tokenizer.yaml has none
        codes: list[dict[str, Any]] = []
        code_idx: dict[str, int] = {}

        def code_of(parsed: dict[str, Any]) -> int:
            if (code := parsed["code"]) not in code_idx:
                code_idx[code] = len(codes)
                codes.append(
                    {
                        "code": code,
                        "prefix": parsed["prefix"],
                        "name": parsed["name"].replace("_", " "),
                        "category": self.prefix_names.get(parsed["prefix"]),
                        "description": self.describe(code),
                    }
                )
            return code_idx[code]

        columns = ("word", "token", "count", "kind", "code", "bin", "text")
        words: dict[str, list[Any]] = {k: [] for k in columns}
        for token, word in sorted(self.decoder.items()):
            parsed = self.parse(word)
            if parsed["prefix"]:
                kind = KINDS.get(parsed["prefix"], "event")
                code, bin_, text = code_of(parsed), parsed["bin"], parsed["text"]
            elif BIN_WORD.match(word):  # an unfused bin, which every code shares
                kind, code, bin_, text = "bin", None, int(word[1:]), None
            else:  # an unfused text value
                kind, code, bin_, text = "text", None, None, word
            for k, x in zip(
                columns, (word, token, counts.get(word), kind, code, bin_, text)
            ):
                words[k].append(x)
        has_counts = any(c is not None for c in words["count"])

        hists = self.get_histograms()
        in_words: collections.defaultdict[int, collections.Counter[int]]
        in_words = collections.defaultdict(collections.Counter)
        for code, bin_, n in zip(words["code"], words["bin"], words["count"]):
            if code is not None and bin_ is not None:
                in_words[code][bin_] += n or 0
        # without training values, a fused vocabulary still counts each bin's word
        bins_source = (
            "meds" if hists else "vocabulary" if self.fused and has_counts else None
        )
        binned = []
        for code, breaks in self.bins.items():
            c = code_of(self.parse(code))
            entry = {"code": c, "breaks": [finite(b) for b in breaks], "n": None}
            if bins_source == "meds":  # a code transferred from elsewhere may have none
                entry |= hists.get(code) or {"n": 0, "bins": [0] * (len(breaks) + 1)}
            elif bins_source == "vocabulary":
                entry["bins"] = [in_words[c][b] for b in range(len(breaks) + 1)]
            binned.append(entry)

        prefixes = {c["prefix"] for c in codes} | (
            {""} if None in words["code"] else set()
        )
        lanes = self.get_lanes(prefixes)
        lane_of = {p: j for j, lane in enumerate(lanes) for p in lane["prefixes"]}
        for c in codes:
            c["lane"] = lane_of.get(c["prefix"])
        prefix_of = [codes[c]["prefix"] if c is not None else "" for c in words["code"]]
        n_words = collections.Counter(prefix_of)
        n_tokens = collections.Counter()
        for p, n in zip(prefix_of, words["count"]):
            n_tokens[p] += n or 0
        n_codes = collections.Counter(c["prefix"] for c in codes)
        n_binned = collections.Counter(codes[b["code"]]["prefix"] for b in binned)

        is_training = self.tkzr.get("is_training")
        return {
            "meta": {
                "cocoa_version": meta.version("cocoa-tokenizer"),
                "generated": now(),
                "tokenizer_created": fmt_value(self.tkzr.get("created_dttm")),
                "tokenizer_version": fmt_value(self.tkzr.get("cocoa_version")),
                "is_training": None if is_training is None else bool(is_training),
                "processed_data_home": str(self.processed_data_home),
                "fused": self.fused,
                "n_bins": self.tkzr_cfg.get("n_bins"),
                "min_training_ct": self.tkzr_cfg.get("min_training_ct") or 0,
                "vocab_size": len(self.decoder),
                "has_counts": has_counts,
                "value_source": "meds" if hists else None,
                "bins_source": bins_source,
            },
            "config": [[str(k), fmt_setting(x)] for k, x in self.tkzr_cfg.items()],
            "lanes": lanes,
            "prefixes": [
                {
                    "prefix": p,
                    "category": self.prefix_names.get(p),
                    "lane": lane_of.get(p),
                    "words": n_words[p],
                    "count": n_tokens[p] if has_counts else None,
                    "codes": n_codes[p],
                    "binned": n_binned[p],
                }
                for p in sorted(prefixes, key=lambda p: (p == "", self.lane_order(p)))
            ],
            "codes": codes,
            "words": dict(words),
            "binned": binned,
            "timelines": self.get_timeline_stats(),
        }

    def render_vocabulary(self) -> str:
        """the vocabulary as one self-contained html document"""
        return render_page(
            "vocabulary",
            f"Vocabulary · {self.processed_data_home.name} · cocoa",
            self.get_vocabulary_payload(),
        )

    def save_vocabulary(self, path: pathlib.Path | str) -> pathlib.Path:
        """write the rendered vocabulary page to `path`"""
        to_file = pathlib.Path(path).expanduser().resolve()
        to_file.parent.mkdir(parents=True, exist_ok=True)
        to_file.write_text(self.render_vocabulary(), encoding="utf-8")
        return to_file

    def render_vocabulary_pdf(self) -> bytes:
        """the vocabulary as a static, self-contained pdf"""
        return vocabulary_pdf(self.get_vocabulary_payload())

    def save_vocabulary_pdf(self, path: pathlib.Path | str) -> pathlib.Path:
        """write the vocabulary to `path` as a pdf"""
        to_file = pathlib.Path(path).expanduser().resolve()
        to_file.parent.mkdir(parents=True, exist_ok=True)
        to_file.write_bytes(self.render_vocabulary_pdf())
        return to_file


def finite(x: float | None) -> float | None:
    """a json-safe float: NaN and infinities become null"""
    return float(x) if x is not None and math.isfinite(x) else None


def now() -> str:
    """the time a page is rendered, to the second, in chicago"""
    return (
        datetime.datetime.now(zoneinfo.ZoneInfo("America/Chicago"))
        .replace(microsecond=0)
        .isoformat()
    )


def round_width(raw: float, whole: bool = False) -> float:
    """
    the least of 1, 2, or 5 times a power of ten that is at least `raw`, and for
    whole numbers at least 1, so that each bucket holds as many of them
    """
    step = 10.0 ** math.floor(math.log10(raw))
    w = next(m * step for m in (1, 2, 5, 10) if m * step >= raw)
    return max(w, 1.0) if whole else w


def describe(s: pl.Series) -> dict[str, float | None]:
    """a series' mean, standard deviation, and quartiles, its nulls left out"""
    s = s.drop_nulls()
    return {
        "mean": finite(s.mean()),
        "std": finite(s.std()),
        "min": finite(s.min()),
        "q25": finite(s.quantile(0.25, "linear")),
        "median": finite(s.median()),
        "q75": finite(s.quantile(0.75, "linear")),
        "max": finite(s.max()),
    }


def fmt_setting(v: Any) -> str:
    """a tokenizer config value as display text, in yaml's spelling"""
    if isinstance(v, bool) or v is None:
        return json.dumps(v)
    if isinstance(v, dict):
        return ", ".join(f"{k}: {fmt_setting(x)}" for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return ", ".join(map(fmt_setting, v))
    return str(v)


def humanize(column: str) -> str:
    """a readable label for a pass-through column, e.g. race_category -> Race"""
    label = re.sub(r"_category$", "", column).replace("_", " ").strip()
    return label[:1].upper() + label[1:]


def fmt_time(t: datetime.datetime | None) -> dict[str, Any] | None:
    """a datetime as epoch milliseconds plus an exact, zone-labeled string"""
    if not isinstance(t, datetime.datetime):
        return None
    return {
        "ms": t.timestamp() * 1000,
        "label": t.strftime("%Y-%m-%d %H:%M:%S")
        + (f".{t.microsecond:06d}".rstrip("0") if t.microsecond else "")
        + (f" {t.tzname()}" if t.tzinfo else ""),
    }


def fmt_value(v: Any) -> str | None:
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


def to_script_json(obj: dict[str, Any]) -> str:
    """json that can sit inside a <script> element without closing it early"""
    return (
        json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def render_page(name: str, title: str, payload: dict[str, Any]) -> str:
    """
    the page `name` as one self-contained html document: its template, filled
    with `title`, its script, `payload`, and the styles of timeline.css, whose
    tokens and components every page shares, then those of its own `name`.css
    """
    assets = resources.files("cocoa.assets")
    styles = dict.fromkeys(("timeline.css", f"{name}.css"))
    fill = {
        "TITLE": html.escape(title),
        "STYLE": font_faces()
        + "".join((assets / f).read_text(encoding="utf-8") for f in styles),
        "SCRIPT": (assets / f"{name}.js").read_text(encoding="utf-8"),
        "DATA": to_script_json(payload),
        "ICON": "data:image/svg+xml;base64,"
        + base64.b64encode((assets / "cocoa.svg").read_bytes()).decode(),
    }
    template = (assets / f"{name}.html").read_text(encoding="utf-8")
    # one pass, so nothing substituted in is itself scanned for markers
    return MARKER.sub(lambda m: fill[m.group(1)], template)


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
        def do_GET(self) -> None:
            self.respond(include_body=True)

        def do_HEAD(self) -> None:
            self.respond(include_body=False)

        def respond(self, include_body: bool) -> None:
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

        def log_message(self, *args: Any) -> None:
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
    for doc in (self.render(sid), self.render_vocabulary()):
        assert not re.search(r"(?:src|href)=[\"']?https?:|url\([\"']?https?:", doc)
    print(self.save(sid, self.processed_data_home / f"timeline_{sid}.html"))
    print(self.save_vocabulary(self.processed_data_home / "vocabulary.html"))
    print(self.save_vocabulary_pdf(self.processed_data_home / "vocabulary.pdf"))
    # breakpoint()
