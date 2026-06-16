#!/usr/bin/env python3

"""
combine datasets processed with same configuration
"""

import datetime
import pathlib
import zoneinfo

import polars as pl
from deepdiff import DeepDiff
from omegaconf import OmegaConf

from cocoa.logger import Logger


def combine_processed_data(
    processed_data_homes: list[pathlib.Path | str],
    processed_data_home: pathlib.Path | str,
) -> str:
    """
    combines processed data from multiple sources into a single new version
    """
    processed_data_home = pathlib.Path(processed_data_home).expanduser().resolve()
    processed_data_home.mkdir(parents=True, exist_ok=True)

    input_dirs = [pathlib.Path(p).expanduser().resolve() for p in processed_data_homes]
    parquets = [f.name for f in input_dirs[0].glob("*.parquet")]
    yamls = [f.name for f in input_dirs[0].glob("*.yaml")]

    for f in yamls:
        # configurations of combined data should match
        cfg0 = OmegaConf.load(input_dirs[0] / f)
        if "created_dttm" in cfg0:
            del cfg0.created_dttm
        for d in input_dirs[1:]:
            cfg = OmegaConf.load(d / f)
            if "created_dttm" in cfg:
                del cfg.created_dttm
            if cfg != cfg0:
                logger = Logger()
                logger.warning(
                    f"Configuration mismatch for {f} between {input_dirs[0]} and {d}"
                )
                logger.warning(
                    DeepDiff(
                        OmegaConf.to_container(cfg0, resolve=True),
                        OmegaConf.to_container(cfg, resolve=True),
                    )
                )
        cfg0.created_dttm = (
            datetime.datetime.now(zoneinfo.ZoneInfo("America/Chicago"))
            .replace(microsecond=0)
            .isoformat()
        )
        with open(processed_data_home / f, "w") as f:
            f.write(OmegaConf.to_yaml(cfg0))

    for f in parquets:
        try:
            (
                pl.scan_parquet([d / f for d in input_dirs], missing_columns="insert")
                .with_row_index("_idx")
                .with_columns(
                    pl.col("_idx").hash(seed=42).alias("_rand")  # deterministic shuffle
                )
                .sort("_rand")
                .drop(["_rand", "_idx"])
                .sink_parquet(processed_data_home / f)
            )
        except (
            pl.SchemaError
        ):  # versions <=26.4.0 of tokenizer created Int64 tokens when loaded from json
            schema = dict(pl.scan_parquet(input_dirs[0] / f).collect_schema())
            tk_cols = [k for k in schema if "tokens" in k]
            for k in tk_cols:
                schema[k] = pl.List(pl.Int64)
            pl.scan_parquet(
                [d / f for d in input_dirs],
                schema=schema,
                cast_options=pl.ScanCastOptions(integer_cast="upcast"),
                missing_columns="insert",
            ).with_columns(
                [pl.col(k).cast(pl.List(pl.UInt32)) for k in tk_cols]
            ).with_row_index("_idx").with_columns(
                pl.col("_idx").hash(seed=42).alias("_rand")  # deterministic shuffle
            ).sort("_rand").drop("_rand").sink_parquet(processed_data_home / f)
    return str(processed_data_home)
