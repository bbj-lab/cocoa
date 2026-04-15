#!/usr/bin/env python3

"""
prepares held-out data for evaluation,
adding flags to disqualify certain subjects from evaluation
"""

import pathlib

import polars as pl
from omegaconf import OmegaConf

from cocoa.reporter import Logger


class Winnower:
    """
    filters held-out timelines for evaluation;
    assigns flags to disqualify certain subjects from evaluation,
    e.g. those whose timelines ends prior to the outcome horizon
    """

    def __init__(
        self,
        main_cfg: pathlib.Path | str = None,
        winnowing_cfg: pathlib.Path | str = None,
        **kwargs,
    ):
        main_cfg = OmegaConf.load(
            pathlib.Path(main_cfg if main_cfg is not None else "./config/main.yaml")
            .expanduser()
            .resolve()
        )
        winnowing_cfg = OmegaConf.load(
            pathlib.Path(
                winnowing_cfg
                if winnowing_cfg is not None
                else main_cfg.winnowing_config
            )
            .expanduser()
            .resolve()
        )
        self.cfg = OmegaConf.merge(
            main_cfg, winnowing_cfg, {k: v for k, v in kwargs.items() if v is not None}
        )
        self.processed_data_home = (
            pathlib.Path(self.cfg.processed_data_home).expanduser().resolve()
        )
        self.tkzr_cfg = OmegaConf.load(self.processed_data_home / "tokenizer.yaml")

    def load_frame(self) -> pl.LazyFrame:
        """
        loads held_out timelines, and performs some preliminary calculations;
        these are lazily evaluated, so only completed if used
        """
        return (
            pl.scan_parquet(self.processed_data_home / "tokens_times.parquet")
            .join(
                pl.scan_parquet(self.processed_data_home / "subject_splits.parquet"),
                on="subject_id",
                validate="1:1",
            )
            .filter(pl.col("split") == "held_out")
            .drop("split")
            .with_columns(
                s_elapsed=pl.col("times").list.eval(
                    (pl.element() - pl.element().first()).dt.total_seconds()
                )
            )
            .with_columns(s_total_duration=pl.col("s_elapsed").list.last())
        )

    def run_thresholding(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        evaluates configurable criteria for establishing a cut-point "last_valid";
        drops timelines that do not reach that point
        """

        if "horizon_s" in self.cfg or "duration_s" in self.cfg.get("threshold", {}):
            # run duration-based thresholding
            horizon_s = self.cfg.get("horizon_s", self.cfg.threshold.duration_s)
            return df.filter(pl.col("s_total_duration") > horizon_s).with_columns(
                last_valid=pl.col("s_elapsed")
                .list.eval(pl.element() < horizon_s)
                .list.sum()
            )
        elif "first_occurrence" in self.cfg.get("threshold", {}):
            # run first-occurrence-based thresholding
            toi = self.tkzr_cfg.lookup[self.cfg.threshold.first_occurrence]
            return df.filter(pl.col("tokens").list.contains(toi)).with_columns(
                last_valid=pl.col("tokens")
                .list.eval(pl.element() == toi)
                .list.arg_max()
                + pl.lit(1)
                # place the triggering token into the past; it is known
            )
        else:
            raise NotImplementedError("Please check the thresholding configuration.")

    def add_outcome_flags(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        adds boolean flags for each outcome token and tense,
        e.g. DSCG//expired_past, DSCG//expired_future
        """
        return df.with_columns(
            tokens_past=pl.col("tokens").list.head("last_valid"),
            tokens_future=pl.col("tokens").list.tail(
                pl.col("tokens").list.len() - pl.col("last_valid")
            ),
        ).with_columns(
            **{
                f"{t}_{tense}": pl.col(f"tokens_{tense}").list.contains(
                    self.tkzr_cfg.lookup[t]
                )
                for t in self.cfg.outcome_tokens
                for tense in ("past", "future")
            }
        )

    def prepare_winnowed_frame(self) -> pl.LazyFrame:
        """loads held-out data, splits at time threshold, and prepares labels"""
        return (
            self.load_frame().pipe(self.run_thresholding).pipe(self.add_outcome_flags)
        )

    def save_all(self, verbose: bool = False):
        """grabs winnowed frame, prints summary stats if requested, and saves it"""
        df = self.prepare_winnowed_frame()
        df.sink_parquet(
            self.processed_data_home / "held_out_for_inference.parquet",
            engine="streaming",
        )
        if verbose:
            logger = Logger()
            logger.info(
                (
                    df.select(
                        [
                            pl.col(f"{t}_{s}").mean().alias(f"{t}_{s}")
                            for t in self.cfg.outcome_tokens
                            for s in ("past", "future")
                        ]
                    )
                    .collect()
                    .transpose(
                        include_header=True, header_name="event", column_names=("rate",)
                    )
                    .with_columns(
                        token=pl.col("event").str.replace(r"_(past|future)$", ""),
                        tense=pl.col("event").str.extract(r"(past|future)$"),
                    )
                    .pivot(values="rate", index="token", on="tense")
                )
            )


if __name__ == "__main__":
    self = Winnower()
    self.save_all(verbose=True)
    # breakpoint()
