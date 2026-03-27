#!/usr/bin/env python3

"""
reports summary statistics from collation and tokenization
"""

import logging

import polars as pl
from rich.console import Console
from rich.logging import RichHandler

logging.basicConfig(
    level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("rich")
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_width_chars(500)


class Logger(logging.Logger):
    def __init__(self, name: str = __package__):
        super().__init__(name=name)
        self.setLevel(logging.INFO)
        self.handlers.clear()

        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(message)s", datefmt="%H:%M:%S%Z"
        )
        ch = RichHandler(
            show_path=False, show_time=False, console=Console(width=200, soft_wrap=True)
        )
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.addHandler(ch)
        self.propagate = False

        self.split_order = pl.col("split").replace(
            {v: i for i, v in enumerate(["train", "tuning", "held_out"])}
        )
        self.code_type = pl.col("code").str.split("//").list[0]

    def summarize_meds_like(self, df: pl.LazyFrame, df_splits: pl.DataFrame):
        self.info("total rows: {}".format(df.select(pl.len()).collect().item()))
        self.info(
            "unique subjects: {}".format(
                df.select(pl.col("subject_id").unique().len()).collect().item()
            )
        )
        self.info(
            "by category: {}".format(
                df.select(self.code_type.value_counts(normalize=True, sort=True))
                .unnest("code")
                .collect()
            )
        )
        self.info("example rows: {}".format(df.unique().head(10).collect()))
        sbj_id = (
            df.group_by("subject_id")
            .agg(pl.len())
            .sort((pl.col("len") - pl.lit(25)).abs(), descending=False)
            .collect()
            .head(1)
            .select("subject_id")
            .item()
        )
        self.info(
            "example subject ({}): {}".format(
                sbj_id, df.filter(pl.col("subject_id") == sbj_id).sort("time").collect()
            )
        )
        self.info(
            "subjects by split: {}".format(
                df_splits.group_by("split")
                .agg(pl.len().alias("count"))
                .with_columns(rate=(pl.col("count") / pl.sum("count")).round(4))
                .sort(self.split_order)
            )
        )
        self.info(
            "rows by split: {}".format(
                df.join(df_splits.lazy(), on="subject_id")
                .group_by("split")
                .agg(pl.len().alias("count"))
                .sort(self.split_order)
                .collect()
            )
        )

    def summarize_tokens_times(
        self, df: pl.LazyFrame, df_splits: pl.DataFrame, lookup: pl.DataFrame
    ):
        self.info("total rows: {}".format(df.select(pl.len()).collect().item()))
        self.info(
            "timeline length stats: {}".format(
                df.select(pl.col("tokens").list.len().alias("lengths")).describe()
            )
        )
        self.info(
            "timeline duration stats: {}".format(
                df.select(
                    (pl.col("times").list.max() - pl.col("times").list.min()).alias(
                        "duration"
                    )
                ).describe()
            )
        )

        self.info(
            "split-level info: {}".format(
                df.join(df_splits.lazy(), on="subject_id", validate="m:1")
                .group_by("split")
                .agg(
                    pl.col("tokens").list.len().mean().alias("avg_len"),
                    pl.col("tokens").list.len().median().alias("median_len"),
                    pl.col("times").list.min().min().alias("first_event"),
                    pl.col("times").list.max().max().alias("last_event"),
                    (pl.col("times").list.max() - pl.col("times").list.min())
                    .mean()
                    .alias("avg_duration"),
                )
                .sort(self.split_order)
                .collect()
            )
        )

        sbj_ids = (
            df.with_columns(pl.col("tokens").list.len().alias("len"))
            .sort((pl.col("len") - pl.lit(25)).abs(), descending=False)
            .collect()
            .head(3)
            .select("subject_id")
            .to_series()
            .to_list()
        )

        for sbj_id in sbj_ids:
            self.info(
                "example timeline ({}): {}".format(
                    sbj_id,
                    df.filter(pl.col("subject_id") == sbj_id)
                    .explode("tokens", "times")
                    .join(
                        lookup.lazy(),
                        left_on="tokens",
                        right_on="token",
                        how="left",
                        validate="m:1",
                    )
                    .collect(),
                )
            )


if __name__ == "__main__":
    logger = Logger()
    logger.info("Testing...")
