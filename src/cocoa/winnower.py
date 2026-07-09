#!/usr/bin/env python3

"""
prepares held-out data for evaluation,
adding flags to disqualify certain subjects from evaluation
"""

import fnmatch
import pathlib

import numpy as np
import polars as pl
from omegaconf import OmegaConf

from cocoa.configurable import Configurable


class Winnower(Configurable):
    """
    filters held-out timelines for evaluation;
    assigns flags to disqualify certain subjects from evaluation,
    e.g. those whose timelines ends prior to the outcome horizon
    """

    default_file = "winnowing.yaml"

    def __init__(
        self,
        winnowing_cfg: pathlib.Path | str = None,
        processed_data_home: pathlib.Path | str = None,
        is_training: bool = True,
        **kwargs,
    ):
        super().__init__(winnowing_cfg, **kwargs)
        self.processed_data_home = (
            pathlib.Path(processed_data_home).expanduser().resolve()
        )
        self.tkzr_cfg = OmegaConf.load(self.processed_data_home / "tokenizer.yaml")
        self.grokked_outcome_tokens = [
            x
            for x in self.tkzr_cfg.lookup.keys()
            if any(fnmatch.fnmatch(x, p) for p in self.cfg.outcome_tokens)
        ]
        self.rng = np.random.default_rng(seed=42)

        self.logger.info("Winnower initialized...")
        self.logger.info(f"{self.processed_data_home=}")
        self.logger.info(
            f"Processed expressions to generate {self.grokked_outcome_tokens=}"
        )

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
            df = (
                df.with_columns(
                    s_elapsed_thresh=pl.col("times")
                    .list.tail(
                        pl.col("tokens_future").list.len() + 1
                    )  # include threshold time
                    .list.eval((pl.element() - pl.element().first()).dt.total_seconds())
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
        return df.with_columns(
            **{
                f"{t}_{tense}": pl.col(f"tokens_{tense}").list.contains(
                    self.tkzr_cfg.lookup[t]
                )
                for t in self.grokked_outcome_tokens
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
                self.logger.summarize_thresholded(df, self.grokked_outcome_tokens)


if __name__ == "__main__":
    self = Winnower(processed_data_home="./processed/mimic/")
    self.save_all(verbose=True)
    # breakpoint()
