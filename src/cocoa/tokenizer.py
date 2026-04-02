#!/usr/bin/env python3

"""
tokenizes collated data into integer sequences, creating bins & a lookup table
"""

import datetime
import pathlib
import zoneinfo

import polars as pl
from omegaconf import OmegaConf

from cocoa.reporter import Logger


class Tokenizer:
    def __init__(self, is_training: bool = True, **kwargs):
        main_cfg = OmegaConf.load(
            pathlib.Path("./config/main.yaml").expanduser().resolve()
        )
        tokenization_cfg = OmegaConf.load(
            pathlib.Path(main_cfg.tokenization_config).expanduser().resolve()
        )
        self.cfg = OmegaConf.merge(main_cfg, tokenization_cfg, kwargs)
        self.data_home = pathlib.Path(self.cfg.data_home).expanduser().resolve()
        self.bins = None
        self.subject_splits = None
        self.lookup = None
        self.is_training = is_training
        self.created_dttm = (
            datetime.datetime.now(zoneinfo.ZoneInfo("America/Chicago"))
            .replace(microsecond=0)
            .isoformat()
        )

    def get_data(self) -> pl.LazyFrame:
        self.subject_splits = pl.scan_parquet(
            pathlib.Path(self.cfg.subject_splits).expanduser().resolve()
        )
        return pl.concat(
            [
                pl.scan_parquet(pathlib.Path(f).expanduser().resolve())
                for f in self.cfg.collated_inputs
            ],
            how="diagonal",
        )

    @staticmethod
    def add_ends(df: pl.LazyFrame) -> pl.LazyFrame:
        # add BOS / EOS codes at appropriate places
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
        # add clock codes if configured
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

    def get_bins(self, df: pl.LazyFrame) -> pl.LazyFrame:
        if self.bins is None and self.is_training:
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
                        for i in range(1, self.cfg.n_bins)
                    ]
                )
            ).collect()
        return self.bins

    def bin_data(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return (
            df.join(self.get_bins(df).lazy(), on="code", how="left")
            .with_columns(
                pl.when(pl.col("numeric_value").is_not_null())
                .then(
                    pl.concat_str(
                        pl.lit("Q"),
                        pl.sum_horizontal(
                            [
                                pl.col(f"break_{i}") <= pl.col("numeric_value")
                                for i in range(1, self.cfg.n_bins)
                            ]
                        ).cast(pl.String),
                    )
                )
                .otherwise(None)
                .alias("binned_value")
            )
            .drop([f"break_{i}" for i in range(1, self.cfg.n_bins)])
        )

    def insert_time_spacers(self, df: pl.LazyFrame) -> pl.LazyFrame:
        if self.cfg.get("insert_spacers", False):
            spcrs = dict(self.cfg.spacers)
            return df.with_columns(
                tdiff_mins=(
                    pl.col("time") - pl.col("time").shift(1).over("subject_id")
                ).dt.total_minutes()
            ).with_columns(
                t_spacer=pl.when(
                    (pl.col("tdiff_mins") < min(spcrs.values()))
                    | pl.col("tdiff_mins").is_null()
                )
                .then(None)
                .otherwise(
                    pl.concat_str(
                        pl.lit("TIME//"),
                        pl.col("tdiff_mins")
                        .cut(breaks=list(spcrs.values())[1:], labels=list(spcrs.keys()))
                        .cast(pl.String),
                    )
                )
                .cast(pl.String)
            )
        else:
            return df.with_columns(t_spacer=pl.lit(None))

    def get_pretokenized(self, df: pl.LazyFrame) -> pl.LazyFrame:
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
                else pl.concat_list("t_spacer", "code", "binned_value", "text_value"),
            )
            .list.drop_nulls()
            .alias("to_tokenize")
        )

    def get_lookup(self, pt: pl.LazyFrame) -> pl.LazyFrame:
        if self.lookup is None and self.is_training:
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

    def get_priority(self) -> pl.LazyFrame:
        return (
            pl.Series("code_type", self.cfg.ordering)
            .to_frame()
            .with_row_index("priority")
        )

    def tokenize_data(self, pt: pl.LazyFrame) -> pl.LazyFrame:
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
            .with_columns(pl.col("token").fill_null(0))  # UNK is 0
            .group_by("subject_id", maintain_order=True)
            .agg(
                pl.col("token").alias("tokens"),
                pl.col("time").alias("times"),
                *(
                    [pl.col("numeric_value").alias("numeric_values")]
                    if self.cfg.get("include_numeric_values", False)
                    else []
                ),
            )
        )

    def get_all(self, verbose: bool = False) -> pl.LazyFrame:
        df = self.get_data()  # load data
        df = self.add_ends(df)  # add BOS/EOS tokens
        df = self.add_clocks(df)  # add clock tokens if configured
        df = self.bin_data(df)  # create bins from training data and bin numeric values
        df = self.insert_time_spacers(df)  # insert time spacer codes when configured
        df = self.get_pretokenized(df).cache()  # pretokenize
        df = self.tokenize_data(df)  # collect tokens into timelines

        if verbose:
            logger = Logger()
            logger.summarize_tokens_times(df, self.subject_splits, self.lookup)

        return df

    def save_all(self, path: pathlib.Path = None, verbose: bool = False):
        df = self.get_all(verbose)

        to_folder = (
            pathlib.Path(path if path is not None else self.cfg.processed_data_home)
            .expanduser()
            .resolve()
        )
        to_folder.mkdir(parents=True, exist_ok=True)
        df.sink_parquet(to_folder / "tokens_times.parquet", engine="streaming")
        self.save(to_folder / "tokenizer.yaml")

    def __call__(self, word: str) -> int:
        try:
            return (
                self.lookup.filter(pl.col("to_tokenize") == word).select("token").item()
            )
        except ValueError or AttributeError:
            return 0  # UNK

    def __contains__(self, word: str) -> bool:
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
        return len(self.lookup) if self.lookup is not None else 0

    def to_yaml(self) -> str:
        return OmegaConf.to_yaml(
            {
                "lookup": dict(self.lookup.rows()) if self.lookup is not None else None,
                "bins": {k: v for k, *v in self.bins.rows()}
                if self.bins is not None
                else None,
                "is_training": self.is_training,
                "cfg": OmegaConf.to_container(self.cfg),
                "created_dttm": self.created_dttm,
            }
        )

    @classmethod
    def from_yaml(cls, yaml_str: str, done_training=True) -> "Tokenizer":
        data = OmegaConf.create(yaml_str)
        cfg = OmegaConf.to_container(data.cfg)
        tkzr = cls(is_training=data.is_training, **cfg)
        tkzr.created_dttm = data.created_dttm
        if data.bins is not None:
            tkzr.bins = pl.DataFrame(
                [[k, *v] for k, v in dict(data.bins).items()],
                schema=["code", *[f"break_{i}" for i in range(1, cfg["n_bins"])]],
                orient="row",
            )
        if data.lookup is not None:
            tkzr.lookup = pl.DataFrame(
                list(dict(data.lookup).items()),
                schema=["to_tokenize", "token"],
                orient="row",
            )
        if done_training:
            tkzr.is_training = False
        return tkzr

    def save(self, path: pathlib.Path):
        to_file = pathlib.Path(path).expanduser().resolve()
        to_file.parent.mkdir(parents=True, exist_ok=True)
        with open(to_file, "w") as f:
            f.write(self.to_yaml())

    def load(self, path: pathlib.Path, done_training=True):
        from_file = pathlib.Path(path).expanduser().resolve()
        with open(from_file, "r") as f:
            yaml_str = f.read()
        return self.from_yaml(yaml_str, done_training=done_training)


if __name__ == "__main__":
    tkzr = Tokenizer()
    tkzr.save_all(verbose=True)

    tkzr_cp = Tokenizer.from_yaml(tkzr.to_yaml())
    assert tkzr.lookup.equals(tkzr_cp.lookup)
    assert tkzr.bins.equals(tkzr_cp.bins)
    assert tkzr("EOS") != 0
    assert "#$%^&*()" not in tkzr

    # breakpoint()
