#!/usr/bin/env python3

"""
prepares timelines for evaluation,
splitting each into a past and a future and flagging outcomes on either side
"""

import fnmatch
import pathlib
import shutil
from typing import Any

import polars as pl
from omegaconf import OmegaConf

from cocoa.configurable import Configurable


class Winnower(Configurable):
    """
    splits the timelines of the configured splits at a threshold into a past and
    a future, dropping those that never reach it, and flags which outcome tokens
    fall on each side; `save_all` writes {split}_for_inference.parquet
    """

    default_file = "winnowing.yaml"

    def __init__(
        self,
        winnowing_cfg: pathlib.Path | str | None = None,
        processed_data_home: pathlib.Path | str | None = None,
        is_training: bool = True,
        output_home: pathlib.Path | str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(winnowing_cfg, **kwargs)
        self.processed_data_home = (
            pathlib.Path(processed_data_home).expanduser().resolve()
        )
        # where the winnowed frames go, so one dataset can be winnowed many ways;
        # copy_processed_data fills it out into a processed dir of its own
        self.output_home = (
            pathlib.Path(output_home).expanduser().resolve()
            if output_home is not None
            else self.processed_data_home
        )
        self.tkzr_cfg = OmegaConf.load(self.processed_data_home / "tokenizer.yaml")
        self.grokked_outcome_tokens: list[str] = [
            x
            for x in self.tkzr_cfg.lookup.keys()
            if any(fnmatch.fnmatch(x, p) for p in self.cfg.outcome_tokens)
        ]

        self.logger.info("Winnower initialized...")
        self.logger.info(f"{self.processed_data_home=}")
        self.logger.info(f"{self.output_home=}")
        self.logger.info(
            f"Processed expressions to generate {self.grokked_outcome_tokens=}"
        )

    def load_frame(self, split: str = "held_out") -> pl.LazyFrame:
        """
        loads the timelines of `split`, and performs some preliminary calculations;
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
        drops timelines that do not reach that point, and under after_start_time_s,
        those that begin at or after it too, which would have no past
        """
        if "horizon_s" in self.cfg or "duration_s" in self.cfg.get("threshold", {}):
            # run duration-based thresholding
            horizon_s = self.cfg.get("horizon_s", self.cfg.threshold.duration_s)
            return df.filter(pl.col("s_total_duration") > horizon_s).with_columns(
                last_valid=pl.col("s_elapsed")
                .list.eval(pl.element() < horizon_s)
                .list.sum()
            )
        elif "after_start_time_s" in self.cfg.get("threshold", {}):
            # run thresholding at a set time after each subject's start_time, as
            # subject_splits.parquet records it, not after its timeline's first event
            cut = pl.col("start_time") + pl.duration(
                seconds=self.cfg.threshold.after_start_time_s,
                time_unit=df.collect_schema()["start_time"].time_unit,
            )
            return df.filter(
                pl.col("times").list.first() < cut, pl.col("times").list.last() > cut
            ).with_columns(
                last_valid=pl.concat_list(cut, "times")
                .list.eval(pl.element() < pl.element().first())
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
        schema = df.collect_schema()
        # per-token columns the tokenizer writes only when configured to
        optional = [c for c in ("hours_to_end_time",) if c in schema.names()]

        def typed(sliced: pl.Expr, like: str) -> pl.Expr:
            # on an empty frame, polars slices a list by a column to List(Null),
            # whatever the schema says, and list.contains panics on that; cast it
            # back to the type of the list `like`, not strictly, as polars drops a
            # strict cast to the type it plans for
            return sliced.cast(schema[like], strict=False)

        df = df.with_columns(
            tokens_past=typed(pl.col("tokens").list.head("last_valid"), "tokens"),
            s_elapsed_past=typed(
                pl.col("s_elapsed").list.head("last_valid"), "s_elapsed"
            ),
            tokens_future=typed(
                pl.col("tokens").list.tail(
                    pl.col("tokens").list.len() - pl.col("last_valid")
                ),
                "tokens",
            ),
            **{
                f"{c}_past": typed(pl.col(c).list.head("last_valid"), c)
                for c in optional
            },
            **{
                f"{c}_future": typed(
                    pl.col(c).list.tail(pl.col(c).list.len() - pl.col("last_valid")), c
                )
                for c in optional
            },
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
                    tokens_future=typed(
                        pl.col("tokens_future").list.head("valid_future_count"),
                        "tokens",
                    ),
                    **{
                        f"{c}_future": typed(
                            pl.col(f"{c}_future").list.head("valid_future_count"), c
                        )
                        for c in optional
                    },
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

    def prepare_winnowed_frame(self, split: str = "held_out") -> pl.LazyFrame:
        """loads a split's timelines, cuts them at the threshold, and flags outcomes"""
        return (
            self.load_frame(split=split)
            .pipe(self.run_thresholding)
            .pipe(self.add_outcome_flags)
        )

    def copy_processed_data(self) -> list[pathlib.Path]:
        """
        copies the files of the processed data dir into a separate output_home,
        skipping its subdirectories and any frames winnowed there before, which
        may come from another config; returns the copies
        """
        if self.output_home == self.processed_data_home:
            return []
        self.output_home.mkdir(parents=True, exist_ok=True)
        copied = [
            pathlib.Path(shutil.copy2(f, self.output_home / f.name))
            for f in sorted(self.processed_data_home.iterdir())
            if f.is_file() and not f.name.endswith("_for_inference.parquet")
        ]
        self.logger.info(f"Copied {[f.name for f in copied]} to {self.output_home}")
        return copied

    def save_all(self, verbose: bool = False) -> None:
        """grabs winnowed frame, prints summary stats if requested, and saves it"""
        self.copy_processed_data()
        for split in self.cfg.get("splits", ["held_out"]):
            df = self.prepare_winnowed_frame(split=split)
            df.sink_parquet(
                self.output_home / f"{split}_for_inference.parquet", engine="streaming"
            )
            if verbose:
                self.logger.info(f"Prepared split {split} for inference:")
                self.logger.summarize_thresholded(df, self.grokked_outcome_tokens)


if __name__ == "__main__":
    self = Winnower(processed_data_home="./processed/mimic/")
    self.save_all(verbose=True)
    # breakpoint()
