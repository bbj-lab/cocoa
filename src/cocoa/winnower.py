#!/usr/bin/env python3

"""
prepares held-out data for evaluation,
adding flags to disqualify certain subjects from evaluation
"""

import pathlib

import numpy as np
import polars as pl
from omegaconf import OmegaConf

from cocoa.logger import Logger


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
        self.rng = np.random.default_rng(seed=42)

        self.logger = Logger()
        self.logger.info("Winnower initialized...")
        self.logger.info(f"{self.processed_data_home=}")

    def load_frame(self, split="held_out") -> pl.LazyFrame:
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
            .filter(pl.col("split") == split)
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
        elif "every_s" in self.cfg.get("threshold", {}):
            # rolling decision points: one row per (subject, cutpoint), with
            # cutpoints at multiples of `every_s` up to `max_decision_s` (default 7d)
            # and strictly inside the stay. `decision_s` records each cutpoint so
            # downstream evaluation can stratify performance by time into the stay.
            stride = int(self.cfg.threshold.every_s)
            cap = int(self.cfg.threshold.get("max_decision_s", 7 * 86400))
            return (
                df.with_columns(
                    decision_s=pl.int_ranges(
                        stride,
                        pl.min_horizontal(
                            pl.col("s_total_duration"), pl.lit(cap + 1)
                        ),
                        stride,
                    )
                )
                .filter(pl.col("decision_s").list.len() > 0)
                .explode("decision_s")
                .with_columns(
                    last_valid=pl.struct(["s_elapsed", "decision_s"]).map_elements(
                        lambda r: sum(x < r["decision_s"] for x in r["s_elapsed"]),
                        return_dtype=pl.Int64,
                    )
                )
            )
        elif (
            "uniform_random" in self.cfg.get("threshold", {})
            and self.cfg.threshold.uniform_random
        ):
            # set the threshold uniformly at random over the duration of each stay
            return df.with_columns(
                sampled_duration=pl.col("s_total_duration").map_elements(
                    lambda x: x * self.rng.random()
                )
            ).with_columns(
                last_valid=pl.struct(["s_elapsed", "sampled_duration"]).map_elements(
                    lambda row: sum(
                        x < row["sampled_duration"] for x in row["s_elapsed"]
                    )
                )
            )
        else:
            raise NotImplementedError("Please check the thresholding configuration.")

    def add_outcome_flags(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        adds boolean flags for each outcome token and tense,
        e.g. DSCG//expired_past, DSCG//expired_future
        """
        df = df.with_columns(
            tokens_past=pl.col("tokens").list.head("last_valid"),
            s_elapsed_past=pl.col("s_elapsed").list.head("last_valid"),
            tokens_future=pl.col("tokens").list.tail(
                pl.col("tokens").list.len() - pl.col("last_valid")
            ),
        )  # split into past and future
        if "horizon_after_threshold_s" in self.cfg:
            horizon_s = int(self.cfg.horizon_after_threshold_s)
            if "decision_s" in df.collect_schema().names():
                # rolling mode: window is (decision_s, decision_s + horizon],
                # anchored at the cutpoint rather than the last observed event
                df = (
                    df.with_columns(
                        s_elapsed_future=pl.col("s_elapsed").list.tail(
                            pl.col("s_elapsed").list.len() - pl.col("last_valid")
                        )
                    )
                    .with_columns(
                        valid_future_count=pl.struct(
                            ["s_elapsed_future", "decision_s"]
                        ).map_elements(
                            lambda r: sum(
                                e <= r["decision_s"] + horizon_s
                                for e in r["s_elapsed_future"]
                            ),
                            return_dtype=pl.Int64,
                        )
                    )
                    .with_columns(
                        tokens_future=pl.col("tokens_future").list.head(
                            "valid_future_count"
                        )
                    )
                )
            else:
                df = (
                    df.with_columns(
                        s_elapsed_thresh=pl.col("times")
                        .list.tail(
                            pl.col("tokens_future").list.len() + 1
                        )  # include threshold time
                        .list.eval(
                            (pl.element() - pl.element().first()).dt.total_seconds()
                        )
                    )
                    .with_columns(
                        valid_future_count=pl.col("s_elapsed_thresh")
                        .list.eval(pl.element() <= self.cfg.horizon_after_threshold_s)
                        .list.sum()
                        - pl.lit(1)  # threshold token was counted, drop it
                    )
                    .with_columns(
                        tokens_future=pl.col("tokens_future").list.head(
                            "valid_future_count"
                        )
                    )
                )
        carry = [
            "subject_id",
            "tokens",
            "times",
            "tokens_past",
            "s_elapsed_past",
            "tokens_future",
        ]
        if "decision_s" in df.collect_schema().names():  # rolling (every_s) mode
            carry.append("decision_s")
        return df.select(*carry).with_columns(
            **{
                f"{t}_{tense}": pl.col(f"tokens_{tense}").list.contains(
                    self.tkzr_cfg.lookup[t]
                )
                for t in self.cfg.outcome_tokens
                for tense in ("past", "future")
            }
        )

    def prepare_winnowed_frame(self, split="held_out") -> pl.LazyFrame:
        """loads held-out data, splits at time threshold, and prepares labels"""
        return (
            self.load_frame(split=split)
            .pipe(self.run_thresholding)
            .pipe(self.add_outcome_flags)
        )

    def save_all(self, verbose: bool = False):
        """grabs winnowed frame, prints summary stats if requested, and saves it"""
        for split in self.cfg.get("splits", ["held_out"]):
            df = self.prepare_winnowed_frame(split=split)
            df.sink_parquet(
                self.processed_data_home / f"{split}_for_inference.parquet",
                engine="streaming",
            )
            if verbose:
                self.logger.info(f"Prepared split {split} for inference:")
                self.logger.summarize_thresholded(df, self.cfg.outcome_tokens)


if __name__ == "__main__":
    self = Winnower()
    self.save_all(verbose=True)
    breakpoint()
