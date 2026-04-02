#!/usr/bin/env python3

"""
collects and collates different dataframes into a denormalized format
"""

import pathlib

import polars as pl
from omegaconf import OmegaConf

from cocoa.reporter import Logger


class Collator:
    def __init__(self, **kwargs):
        main_cfg = OmegaConf.load(
            pathlib.Path("./config/main.yaml").expanduser().resolve()
        )
        collation_cfg = OmegaConf.load(
            pathlib.Path(main_cfg.collation_config).expanduser().resolve()
        )
        self.cfg = OmegaConf.merge(main_cfg, collation_cfg, kwargs)
        self.data_home = pathlib.Path(self.cfg.data_home).expanduser().resolve()
        self.reference_frame = None

    @staticmethod
    def slightly_safer_eval(expr):
        return eval(
            expr,
            {"__builtins__": {"str": str, "int": int, "float": float, "bool": bool}},
            {"pl": pl},
        )

    def load_table(
        self,
        *,
        table: str = None,
        filter_expr: str | list = None,
        agg_expr: str | list = None,
        with_col_expr: str | list = None,
        key: str = None,
        subject_id_str: str = None,
        **kwargs,
    ) -> pl.LazyFrame:
        """lazy-load the table `table`.parquet and perform some standard ETL
        tasks in the following order if specified:
        1. fix subject_id to `subject_id_str`
        2. perform a filter operation
        3. add columns
        4. perform an aggregation by `key` or self.cfg["subfject_id"]
        """
        if (f := self.data_home / f"{table}.parquet").exists():
            df = pl.scan_parquet(f)
        elif (f := self.data_home / f"{table}.csv").exists():
            df = pl.scan_csv(f)
        else:
            raise FileNotFoundError(
                f"No parquet or csv file found for table '{table}' in {self.data_home}"
            )
        if subject_id_str is not None:
            df = df.with_columns(pl.col(subject_id_str).alias(self.cfg["subject_id"]))
        if filter_expr is not None:
            df = df.filter(
                self.slightly_safer_eval(filter_expr)
                if isinstance(filter_expr, str)
                else [self.slightly_safer_eval(c) for c in filter_expr]
            )
        if with_col_expr is not None:
            df = df.with_columns(
                self.slightly_safer_eval(with_col_expr)
                if isinstance(with_col_expr, str)
                else [self.slightly_safer_eval(c) for c in with_col_expr]
            )
        if agg_expr is not None:
            df = df.group_by(key if key is not None else self.cfg["subject_id"]).agg(
                self.slightly_safer_eval(agg_expr)
                if isinstance(agg_expr, str)
                else [self.slightly_safer_eval(c) for c in agg_expr]
            )
        return df

    def get_reference_frame(self) -> pl.LazyFrame:
        """create the static reference frame as configured"""
        cfg = self.cfg["reference"]
        if self.reference_frame is not None:  # pull from cache if available
            return self.reference_frame
        df = self.load_table(**cfg)
        for tkv in cfg.get("augmentation_tables", []):
            df = df.join(
                self.load_table(**tkv),
                on=tkv["key"],
                validate=tkv["validation"],
                how="left",
                maintain_order="left",
            )
        self.reference_frame = df  # cache result
        return self.reference_frame

    def get_entry(
        self,
        table: str,
        time: str,
        code: str,
        *,
        prefix: str = None,
        numeric_value: str = None,
        text_value: str = None,
        filter_expr: str = None,
        with_col_expr: str = None,
        reference_key: str = None,
        subject_id_str: str = None,
        fix_date_to_time: bool = None,
    ) -> pl.LazyFrame:
        """create tokens corresponding to a configured event"""
        df = (
            self.get_reference_frame()
            if table == "REFERENCE"
            else self.load_table(
                table=table,
                filter_expr=filter_expr,
                with_col_expr=with_col_expr,
                subject_id_str=subject_id_str,
            )
        )
        if fix_date_to_time:
            # if a date was cast to a time,
            # the default of 00:00:00 should be replaced with 23:59:59
            df = df.with_columns(
                pl.col(time).cast(pl.Datetime).dt.replace(hour=23, minute=59, second=59)
            )
        if reference_key is not None:
            df = df.join(self.reference_frame, on=reference_key, how="inner").filter(
                pl.col(time)
                .cast(pl.Datetime)
                .is_between(
                    pl.col(self.cfg["reference"]["start_time"]).cast(pl.Datetime),
                    pl.col(self.cfg["reference"]["end_time"]).cast(pl.Datetime),
                )
            )

        return df.select(
            pl.col(self.cfg["subject_id"]).cast(pl.String).alias("subject_id"),
            pl.col(time)
            .cast(pl.Datetime)
            .dt.replace_time_zone(time_zone=None)
            .alias("time"),
            pl.concat_str(
                [
                    pl.lit(prefix),
                    pl.col(code)
                    .cast(pl.String)
                    .str.to_lowercase()
                    .str.replace_all(r"\s+", "_"),
                ],
                separator="//",
                ignore_nulls=True,
            ).alias("code"),
            (pl.col(numeric_value) if numeric_value else pl.lit(None))
            .cast(pl.Float32)  # dumb
            .alias("numeric_value"),
            (
                pl.col(text_value).str.to_lowercase().str.replace_all(r"\s+", "_")
                if text_value
                else pl.lit(None)
            )
            .cast(pl.String)
            .alias("text_value"),
        ).drop_nulls(subset=["subject_id", "time", "code"])

    def get_all(self) -> pl.LazyFrame:
        """get all tokens for all events as configured"""
        return pl.concat(
            (self.get_entry(**entry) for entry in self.cfg.get("entries", []))
        )

    def get_subject_splits(self, df_all: pl.LazyFrame) -> pl.DataFrame:
        """get the subject splits as configured"""
        sbj = (
            df_all.group_by("subject_id")
            .agg(pl.col("time").min().alias("first_time"))
            .sort("first_time")
            .with_row_index()
        ).collect()
        return sbj.with_columns(
            split=pl.when(
                pl.col("index") < int(len(sbj) * self.cfg.subject_splits.train_frac)
            )
            .then(pl.lit("train"))
            .when(
                pl.col("index")
                < int(
                    len(sbj)
                    * (
                        self.cfg.subject_splits.train_frac
                        + self.cfg.subject_splits.tuning_frac
                    )
                )
            )
            .then(pl.lit("tuning"))
            .otherwise(pl.lit("held_out"))
        ).select("subject_id", "split")

    def save_all(self, path: pathlib.Path = None, verbose: bool = False):
        to_folder = (
            pathlib.Path(path if path is not None else self.cfg.processed_data_home)
            .expanduser()
            .resolve()
        )
        to_folder.mkdir(parents=True, exist_ok=True)
        (df_all := self.get_all()).sink_parquet(
            to_folder / "meds.parquet", engine="streaming"
        )
        (df_splits := self.get_subject_splits(df_all)).write_parquet(
            to_folder / "subject_splits.parquet"
        )

        if verbose:
            logger = Logger()
            logger.summarize_meds_like(df_all, df_splits)


if __name__ == "__main__":
    cltr = Collator()
    cltr.save_all(verbose=True)
    # breakpoint()
