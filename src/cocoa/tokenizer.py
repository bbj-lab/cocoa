#!/usr/bin/env python3

"""
tokenizes collated data into integer sequences, creating bins & a lookup table
"""

import datetime
import importlib.metadata as meta
import pathlib
import zoneinfo

import polars as pl
from omegaconf import OmegaConf

from cocoa.configurable import Configurable


class Tokenizer(Configurable):
    """
    converts collated data to tokenized timelines,
    learning bins and lookup table on training data
    """

    default_file = "tokenization.yaml"

    def __init__(
        self,
        tokenization_cfg: pathlib.Path | str = None,
        processed_data_home: pathlib.Path | str = None,
        is_training: bool = True,
        **kwargs,
    ):
        super().__init__(tokenization_cfg, **kwargs)
        self.processed_data_home = (
            pathlib.Path(processed_data_home).expanduser().resolve()
        )

        self.bins = None
        self.rank_cdf = None
        self.zscore_stats = None
        self.subject_splits = None
        self.lookup = None
        self.is_training = is_training
        self.created_dttm = (
            datetime.datetime.now(zoneinfo.ZoneInfo("America/Chicago"))
            .replace(microsecond=0)
            .isoformat()
        )

    def get_data(self) -> pl.LazyFrame:
        self.logger.info(f"Loading collated data with {self.processed_data_home=}")
        self.subject_splits = pl.scan_parquet(
            self.processed_data_home / "subject_splits.parquet"
        )  # allows processed_data_home to be set after initialization
        return pl.scan_parquet(self.processed_data_home / "meds.parquet")

    @staticmethod
    def add_ends(df: pl.LazyFrame) -> pl.LazyFrame:
        """add BOS / EOS codes at appropriate places"""
        return pl.concat(
            [
                df,
                df.group_by("subject_id")
                .agg(pl.col("time").min())
                .with_columns(code=pl.lit("BOS")),
                df.group_by("subject_id")
                .agg(pl.col("time").max())
                .with_columns(code=pl.lit("EOS")),
            ],
            how="diagonal",
        )

    def add_clocks(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """add clock codes if configured"""
        if self.cfg.get("insert_clocks", False):
            return pl.concat(
                [
                    df,
                    df.group_by("subject_id")
                    .agg(
                        start=pl.col("time").min().dt.truncate("1h"),
                        end=pl.col("time").max().dt.truncate("1h"),
                    )
                    .with_columns(
                        pl.datetime_ranges(
                            pl.col("start"),
                            pl.col("end"),
                            interval="1h",
                            closed="right",
                        )
                        .alias("time")
                        .list.filter(
                            pl.element().dt.strftime("%H").is_in(list(self.cfg.clocks))
                        )
                    )
                    .explode("time", keep_nulls=False)
                    .drop_nulls(subset=["time"])
                    .with_columns(HH=pl.col("time").dt.strftime("%H"))
                    .select(
                        "subject_id",
                        "time",
                        pl.concat_str(pl.lit("CLCK//"), pl.col("HH")).alias("code"),
                    ),
                ],
                how="diagonal",
            )
        else:
            return df

    def get_bins(self, df: pl.LazyFrame) -> pl.DataFrame:
        """
        calculate bins for numeric values on training data; when
        include_exact_rank is set, also learns break_0 (min, quantile 0) and
        break_{n_bins} (max, quantile 1) per code. These two are no longer
        used for rank computation (see get_rank_cdf / _add_exact_rank) but
        are kept so this method's output shape is unchanged.
        """
        if self.bins is None:
            assert self.is_training, "Bins must be learned during training"
            include_exact_rank = self.cfg.get("include_exact_rank", False)
            lo = 0 if include_exact_rank else 1
            hi = self.cfg.n_bins + 1 if include_exact_rank else self.cfg.n_bins
            self.bins = (
                df.join(  # restrict to training data
                    self.subject_splits.filter(pl.col("split") == "train"),
                    on="subject_id",
                    validate="m:1",
                )
                .drop_nulls(subset=["numeric_value"])
                .drop_nans(subset=["numeric_value"])
                .group_by("code")
                .agg(
                    [
                        pl.col("numeric_value")
                        .quantile(i / self.cfg.n_bins)
                        .alias(f"break_{i}")
                        for i in range(lo, hi)
                    ]
                )
            ).collect()
        return self.bins

    def get_rank_cdf(self, df: pl.LazyFrame) -> pl.DataFrame:
        """
        empirical CDF of numeric_value per code on training data: for each
        distinct training value, the proportion of training values <= it
        (p2). Recomputed on every run rather than persisted (unlike bins),
        since an exact CDF needs one row per distinct training value and can
        be far larger than n_bins allows; used by _add_exact_rank.
        """
        if self.rank_cdf is None:
            assert self.is_training, "Rank CDF must be learned during training"
            self.rank_cdf = (
                df.join(  # restrict to training data
                    self.subject_splits.filter(pl.col("split") == "train"),
                    on="subject_id",
                    validate="m:1",
                )
                .drop_nulls(subset=["numeric_value"])
                .drop_nans(subset=["numeric_value"])
                .group_by("code", "numeric_value")
                .agg(count=pl.len())
                .sort("code", "numeric_value")
                .with_columns(
                    p2=(
                        pl.col("count").cum_sum().over("code")
                        / pl.col("count").sum().over("code")
                    ).cast(pl.Float32)
                )
                .select("code", "numeric_value", "p2")
            ).collect()
        return self.rank_cdf

    def _add_exact_rank(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        attach a continuous rank in [0, 1] for numeric_value: (p1 + p2) / 2,
        where p1 is the proportion of training values (for that code)
        strictly less than the value and p2 the proportion less than or
        equal to it. Both come from an as-of match against get_rank_cdf's
        sorted per-code training values: p2 allows an exact match, p1
        excludes it (landing on the previous distinct training value). A
        value strictly below a code's smallest training value gets rank 0;
        strictly above the largest gets rank 1.
        """
        rank_cdf = self.get_rank_cdf(df).lazy().sort("code", "numeric_value")
        has_value = df.filter(pl.col("numeric_value").is_not_null()).sort(
            "code", "numeric_value"
        )
        no_value = df.filter(pl.col("numeric_value").is_null()).with_columns(
            exact_rank=pl.lit(None, dtype=pl.Float32)
        )
        ranked = (
            has_value.join_asof(
                rank_cdf, on="numeric_value", by="code", strategy="backward"
            )
            .rename({"p2": "_p2"})
            .join_asof(
                rank_cdf.rename({"p2": "_p1"}),
                on="numeric_value",
                by="code",
                strategy="backward",
                allow_exact_matches=False,
            )
            .with_columns(
                exact_rank=(
                    (pl.col("_p1").fill_null(0.0) + pl.col("_p2").fill_null(0.0)) / 2
                ).cast(pl.Float32)
            )
            .drop("_p1", "_p2")
        )
        return pl.concat([ranked, no_value], how="vertical")

    def get_zscore_stats(self) -> pl.DataFrame:
        """
        per-code median and MAD (median absolute deviation) of
        numeric_value on training data, for the robust z-score used by
        _add_zscore_rank. Two-pass (median, then median of |x - median|)
        since the second aggregation depends on the first's per-code
        result. Recomputed on every run rather than persisted, matching
        get_rank_cdf. Deliberately re-scans meds.parquet fresh via
        get_data() rather than taking the caller's already-elaborate `df`
        (unlike get_bins/get_rank_cdf): chaining this onto the same lazy
        graph as add_ends/bin_data's other branches caused Polars to
        redundantly re-materialize that whole upstream graph per branch,
        ballooning to 100+GB RSS and getting OOM-killed on the full
        dataset, even though this query in isolation runs in ~7s/13GB.
        """
        if self.zscore_stats is None:
            assert self.is_training, "Z-score stats must be learned during training"
            train_values = (
                self.get_data().join(  # restrict to training data
                    self.subject_splits.filter(pl.col("split") == "train"),
                    on="subject_id",
                    validate="m:1",
                )
                .drop_nulls(subset=["numeric_value"])
                .drop_nans(subset=["numeric_value"])
                .select("code", "numeric_value")
            )
            medians = train_values.group_by("code").agg(
                median=pl.col("numeric_value").median()
            )
            self.zscore_stats = (
                train_values.join(medians, on="code")
                .group_by("code")
                .agg(
                    median=pl.col("median").first(),
                    mad=(pl.col("numeric_value") - pl.col("median")).abs().median(),
                )
            ).collect()
        return self.zscore_stats

    def _add_zscore_rank(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        attach a continuous rank in (0, 1) for numeric_value: sigmoid of a
        robust ("modified") z-score, 0.6745 * (x - median) / MAD, using
        each code's training median/MAD from get_zscore_stats. The 0.6745
        constant (Phi^-1(0.75)) puts MAD on the same scale as a normal
        distribution's standard deviation, so this behaves like an
        ordinary z-score under normality but isn't dragged around by the
        extreme outliers/data-entry errors known to be present in this
        dataset -- with plain mean/variance those inflate the spread and
        compress everyone else's z-score, defeating the point of using
        z-scores to keep outliers separated. Unlike _add_exact_rank
        (bounded to [0, 1] by construction, saturating at the training
        data's min/max), this only approaches but never reaches 0/1, and
        extrapolates smoothly for values outside the training range
        rather than clamping.
        A code with MAD == 0 (>=50% of training values tied at the
        median) falls back to a tiny epsilon denominator: rows at the
        median get z=0 (rank 0.5) and any other value saturates toward
        0/1, the right qualitative behavior for an effectively-constant
        code.

        Unlike _add_exact_rank, this is a plain per-code broadcast join
        (no join_asof), so it doesn't need _add_exact_rank's
        filter-into-has/no-value-then-concat structure -- and deliberately
        avoids it: chaining a second such filter/concat on top of
        _add_exact_rank's own broke the final sink_parquet's streaming
        execution (fell back to materializing the whole ~209M-row dataset
        in memory, OOMing a 123GB machine), even though this join +
        with_columns is row-count-preserving and cheap on its own.
        """
        stats = self.get_zscore_stats()
        return (
            df.join(stats.lazy(), on="code", how="left", maintain_order="left")
            .with_columns(
                zscore_rank=pl.when(pl.col("numeric_value").is_not_null())
                .then(
                    (
                        1
                        / (
                            1
                            + (
                                -0.6745
                                * (pl.col("numeric_value") - pl.col("median"))
                                / pl.when(pl.col("mad") > 0)
                                .then(pl.col("mad"))
                                .otherwise(1e-6)
                            ).exp()
                        )
                    ).cast(pl.Float32)
                )
                .otherwise(None)
            )
            .drop("median", "mad")
        )

    def bin_data(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """discretize numeric values with learned cut points"""
        include_exact_rank = self.cfg.get("include_exact_rank", False)
        include_zscore_rank = self.cfg.get("include_zscore_rank", False)
        n = self.cfg.n_bins
        out = df.join(self.get_bins(df).lazy(), on="code", how="left").with_columns(
            pl.when(pl.col("numeric_value").is_not_null())
            .then(
                pl.concat_str(
                    pl.lit("Q"),
                    pl.sum_horizontal(
                        [
                            pl.col(f"break_{i}") <= pl.col("numeric_value")
                            for i in range(1, n)
                        ]
                    ).cast(pl.String),
                )
            )
            .otherwise(None)
            .alias("binned_value")
        )
        if include_exact_rank:
            out = self._add_exact_rank(out)
        if include_zscore_rank:
            out = self._add_zscore_rank(out)
        if include_exact_rank:
            return out.drop([f"break_{i}" for i in range(0, n + 1)])
        return out.drop([f"break_{i}" for i in range(1, n)])

    def insert_time_spacers(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        add time spacing tokens if configured;
        this should be done after clock tokens are inserted
        """
        if self.cfg.get("insert_spacers", False):
            spcrs = dict(self.cfg.spacers)
            return (
                df.sort("subject_id", "time")
                .with_columns(
                    tdiff_mins=(
                        pl.col("time") - pl.col("time").shift(1).over("subject_id")
                    ).dt.total_minutes()
                )
                .with_columns(
                    t_spacer=pl.when(
                        (pl.col("tdiff_mins") < min(spcrs.values()))
                        | pl.col("tdiff_mins").is_null()
                    )
                    .then(None)
                    .otherwise(
                        pl.concat_str(
                            pl.lit("TIME//"),
                            pl.col("tdiff_mins")
                            .cut(
                                breaks=list(spcrs.values())[1:],
                                labels=list(spcrs.keys()),
                            )
                            .cast(pl.String),
                        )
                    )
                    .cast(pl.String)
                )
            )
        else:
            return df.with_columns(t_spacer=pl.lit(None))

    def get_pretokenized(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """prepare codes for tokenization, depending on whether fusion is configured"""
        return df.with_columns(
            pl.concat_list(
                pl.col("t_spacer"),
                pl.concat_str(
                    pl.col("code"),
                    pl.col("binned_value"),
                    pl.col("text_value"),
                    separator="_",
                    ignore_nulls=True,
                )
                if self.cfg.get("fused", False)
                else pl.concat_list("code", "binned_value", "text_value"),
            )
            .list.drop_nulls()
            .alias("to_tokenize")
        )

    def get_lookup(self, pt: pl.LazyFrame) -> pl.DataFrame:
        """create mapping from vocabulary to integer tokens on training data"""
        if self.lookup is None:
            assert self.is_training, "Lookup table must be learned during training"
            lookup = (
                pt.join(  # restrict to training data
                    self.subject_splits.filter(pl.col("split") == "train"),
                    on="subject_id",
                    validate="m:1",
                )
                .explode("to_tokenize")
                .select(pl.col("to_tokenize").unique().sort())
                .filter(pl.col("to_tokenize") != "UNK")  # UNK is 0
                .with_row_index("token", offset=1)
                .select("to_tokenize", "token")
            )
            unk_row = pl.LazyFrame(
                {"to_tokenize": ["UNK"], "token": pl.Series([0], dtype=pl.UInt32)}
            )
            self.lookup = pl.concat([unk_row, lookup]).collect()
        return self.lookup

    def get_priority(self) -> pl.DataFrame:
        """fetch ordering for sorting cotemporaneous codes"""
        return (
            pl.Series("code_type", self.cfg.ordering)
            .to_frame()
            .with_row_index("priority")
        )

    def tokenize_data(self, pt: pl.LazyFrame) -> pl.LazyFrame:
        """apply lookup table to pretokenized data"""
        return (
            pt.with_columns(code_type=pl.col("code").str.split("//").list[0])
            .join(
                self.get_priority().lazy(), on="code_type", how="left", validate="m:1"
            )
            .with_columns(pl.col("priority").fill_null(len(self.cfg.ordering)))
            .sort("time", "priority")
            .explode("to_tokenize")
            .join(
                self.get_lookup(pt).lazy(), on="to_tokenize", validate="m:1", how="left"
            )
            .with_columns(
                pl.col("token").fill_null(pl.lit(0, dtype=pl.UInt32))
            )  # UNK is 0
            .group_by("subject_id", maintain_order=True)
            .agg(
                pl.col("token").alias("tokens"),
                pl.col("time").alias("times"),
                *(
                    [pl.col("numeric_value").alias("numeric_values")]
                    if self.cfg.get("include_numeric_values", False)
                    else []
                ),
                *(
                    # fill_null: non-numeric positions get a 0.0 placeholder
                    # rather than null, since downstream consumers (HF
                    # datasets/torch) can't hold None in a float list; the
                    # placeholder is never read there (those positions are
                    # masked out by token id, not by this array).
                    [pl.col("exact_rank").fill_null(0.0).alias("exact_ranks")]
                    if self.cfg.get("include_exact_rank", False)
                    else []
                ),
                *(
                    [pl.col("zscore_rank").fill_null(0.0).alias("zscore_ranks")]
                    if self.cfg.get("include_zscore_rank", False)
                    else []
                ),
            )
        )

    def get_all(self, verbose: bool = False) -> pl.LazyFrame:
        """run all steps to convert collated data to tokenized timelines"""
        df = self.get_data()  # load data
        df = self.add_ends(df)  # add BOS/EOS tokens
        df = self.add_clocks(df)  # add clock tokens if configured
        df = self.bin_data(df)  # create bins from training data and bin numeric values
        df = self.insert_time_spacers(df)  # insert time spacer codes when configured
        df = self.get_pretokenized(df).cache()  # pretokenize
        df = self.tokenize_data(df)  # collect tokens into timelines

        if verbose:
            self.logger.summarize_tokens_times(df, self.subject_splits, self.lookup)

        return df

    def save_all(self, verbose: bool = False):
        """
        get tokenized timelines and save them to disc,
        along with artifacts created during tokenization
        """
        df = self.get_all(verbose)
        df.sink_parquet(
            self.processed_data_home / "tokens_times.parquet", engine="streaming"
        )
        self.save(self.processed_data_home / "tokenizer.yaml")

    def __call__(self, word: str) -> int:
        """apply tokenizer to a single word"""
        try:
            return (
                self.lookup.filter(pl.col("to_tokenize") == word).select("token").item()
            )
        except (ValueError, AttributeError):
            return 0  # UNK

    def __contains__(self, word: str) -> bool:
        """is word in the tokenizer's vocabulary?"""
        return self.__call__(word) != 0

    def __str__(self):
        return "{sp} of {sz} words {md}".format(
            sp=self.__class__,
            sz=len(self),
            md="in training mode" if self.is_training else "(frozen)",
        )

    def __repr__(self):
        return str(self) + ", created {dttm}".format(dttm=self.created_dttm)

    def __len__(self) -> int:
        """number of tokens in vocabulary"""
        return len(self.lookup) if self.lookup is not None else 0

    def to_yaml(self) -> str:
        """yaml representation of tokenizer; sufficient for reconstruction"""
        return OmegaConf.to_yaml(
            {
                "lookup": dict(self.lookup.rows()) if self.lookup is not None else None,
                "bins": {k: v for k, *v in self.bins.rows()}
                if self.bins is not None
                else None,
                "is_training": self.is_training,
                "cfg": OmegaConf.to_container(self.cfg),
                "created_dttm": self.created_dttm,
                "cocoa_version": meta.version("cocoa-tokenizer"),
            }
        )

    def from_yaml(self, yaml_str: str, done_training=True) -> "Tokenizer":
        """
        construct tokenizer from yaml representation
        places tokenizer into inference mode by default
        """
        data = OmegaConf.create(yaml_str)
        cfg = OmegaConf.to_container(data.cfg)
        tkzr = self.__class__(
            self.config_file,
            processed_data_home=self.processed_data_home,
            is_training=data.is_training,
            **OmegaConf.merge(self.cfg, cfg),
        )
        tkzr.created_dttm = data.created_dttm
        if data.bins is not None:
            break_cols = (
                [f"break_{i}" for i in range(0, cfg["n_bins"] + 1)]
                if cfg.get("include_exact_rank", False)
                else [f"break_{i}" for i in range(1, cfg["n_bins"])]
            )
            tkzr.bins = pl.DataFrame(
                [[k, *v] for k, v in dict(data.bins).items()],
                schema=["code", *break_cols],
                orient="row",
            )
        if data.lookup is not None:
            tkzr.lookup = pl.DataFrame(
                list(dict(data.lookup).items()),
                schema={"to_tokenize": pl.String, "token": pl.UInt32},
                orient="row",
            )
        if done_training:
            tkzr.is_training = False
        return tkzr

    def save(self, path: pathlib.Path):
        """write yaml representation of tokenizer to disc"""
        to_file = pathlib.Path(path).expanduser().resolve()
        to_file.parent.mkdir(parents=True, exist_ok=True)
        with open(to_file, "w") as f:
            f.write(self.to_yaml())

    def load(self, path: pathlib.Path, done_training=True):
        """retrieve tokenizer from saved yaml representation"""
        from_file = pathlib.Path(path).expanduser().resolve()
        with open(from_file, "r") as f:
            yaml_str = f.read()
        return self.from_yaml(yaml_str, done_training=done_training)


if __name__ == "__main__":
    self = Tokenizer(processed_data_home="./processed/mimic/")
    self.save_all(verbose=True)

    tkzr_cp = Tokenizer(processed_data_home="./processed/mimic/").from_yaml(
        self.to_yaml()
    )
    assert self.lookup.equals(tkzr_cp.lookup)
    assert self.bins.equals(tkzr_cp.bins)
    assert self("EOS") != 0
    assert "#$%^&*()" not in self

    # breakpoint()
