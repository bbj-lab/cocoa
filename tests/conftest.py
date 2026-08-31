#!/usr/bin/env python3

"""
shared fixtures: synthetic raw tables and processed pipeline outputs

the session-scoped `pipeline` fixture runs collate -> tokenize -> winnow once
with the shipped default configs over a synthetic dataset; the `runner` fixture
re-runs individual stages with overridden configs in a per-test directory
"""

import dataclasses
import functools
import importlib.resources as resources
import pathlib
import shutil

import polars as pl
import pytest
import synth
from omegaconf import OmegaConf

from cocoa.collator import Collator
from cocoa.tokenizer import Tokenizer
from cocoa.winnower import Winnower

STAGES = ("collation", "tokenization", "winnowing")


def default_cfg(stage: str) -> dict:
    """a mutable copy of a shipped default config"""
    assert stage in STAGES, f"{stage=} must be one of {STAGES}"
    return OmegaConf.to_container(
        OmegaConf.load(resources.files("cocoa.config") / f"{stage}.yaml"), resolve=True
    )


def write_cfg(path: pathlib.Path, cfg) -> pathlib.Path:
    """write a config mapping to `path` and return it"""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(OmegaConf.create(cfg)))
    return path


@dataclasses.dataclass
class Processed:
    """a processed data directory, with its artifacts read on demand"""

    path: pathlib.Path
    manifest: synth.Manifest | None = None

    @functools.cached_property
    def meds(self) -> pl.DataFrame:
        return pl.read_parquet(self.path / "meds.parquet")

    @functools.cached_property
    def splits(self) -> pl.DataFrame:
        return pl.read_parquet(self.path / "subject_splits.parquet")

    @functools.cached_property
    def tokens_times(self) -> pl.DataFrame:
        return pl.read_parquet(self.path / "tokens_times.parquet")

    @functools.cached_property
    def tokenizer_yaml(self):
        return OmegaConf.load(self.path / "tokenizer.yaml")

    @functools.cached_property
    def vocab(self) -> dict:
        """vocabulary word -> integer token, as recorded in tokenizer.yaml"""
        return dict(self.tokenizer_yaml.lookup)

    @functools.cached_property
    def decoder(self) -> dict:
        """integer token -> vocabulary word"""
        return {v: k for k, v in self.vocab.items()}

    def inference(self, split: str = "held_out") -> pl.DataFrame:
        return pl.read_parquet(self.path / f"{split}_for_inference.parquet")

    def subjects_in_split(self, split: str) -> list:
        return (
            self.splits.filter(pl.col("split") == split)["subject_id"].sort().to_list()
        )

    def decode(self, tokens) -> list:
        """render a token sequence as vocabulary words (UNK for out-of-vocab)"""
        return [self.decoder.get(int(t), "UNK") for t in tokens]

    def timeline(self, subject_id: str) -> list:
        """the decoded timeline of one subject"""
        row = self.tokens_times.filter(pl.col("subject_id") == subject_id)
        return self.decode(row["tokens"].item().to_list())


class Runner:
    """runs individual pipeline stages into directories under one test's tmp dir"""

    def __init__(self, root: pathlib.Path, raw: synth.Manifest, cached: pathlib.Path):
        self.root = pathlib.Path(root)
        self.raw = raw
        self._cached = pathlib.Path(cached)
        self._n = 0

    def dir(self, name: str = None) -> pathlib.Path:
        """a fresh empty directory under the test's tmp dir"""
        if name is None:
            self._n += 1
            name = f"processed{self._n}"
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def cfg_path(self, stage: str, cfg) -> pathlib.Path | None:
        """
        resolve a config argument: None means the shipped default, a path is
        used as-is, and a mapping is written to a yaml file first
        """
        if cfg is None:
            return None
        if isinstance(cfg, (str, pathlib.Path)):
            return pathlib.Path(cfg)
        self._n += 1
        return write_cfg(self.root / f"{stage}{self._n}.yaml", cfg)

    def seed_collated(self, dest: pathlib.Path = None) -> pathlib.Path:
        """copy the cached default collation into a fresh directory"""
        dest = dest or self.dir()
        for f in ("meds.parquet", "subject_splits.parquet"):
            shutil.copy(self._cached / f, dest / f)
        return dest

    def collate(
        self, *, cfg=None, raw: pathlib.Path = None, dest: pathlib.Path = None, **kwargs
    ) -> Collator:
        collator = Collator(
            collation_cfg=self.cfg_path("collation", cfg),
            raw_data_home=raw or self.raw.root,
            processed_data_home=dest or self.dir(),
        )
        collator.save_all(**kwargs)
        return collator

    def tokenize(
        self,
        *,
        cfg=None,
        processed: pathlib.Path = None,
        tokenizer_home: pathlib.Path = None,
        **kwargs,
    ) -> Tokenizer:
        tokenizer = Tokenizer(
            tokenization_cfg=self.cfg_path("tokenization", cfg),
            processed_data_home=processed if processed is not None else self.dir(),
        )
        if tokenizer_home is not None:
            tokenizer = tokenizer.load(tokenizer_home)
        tokenizer.save_all(**kwargs)
        return tokenizer

    def winnow(self, *, cfg=None, processed: pathlib.Path = None, **kwargs) -> Winnower:
        winnower = Winnower(
            winnowing_cfg=self.cfg_path("winnowing", cfg),
            processed_data_home=processed if processed is not None else self.dir(),
        )
        winnower.save_all(**kwargs)
        return winnower

    def full(self, *, collation=None, tokenization=None, winnowing=None, **kwargs):
        """run all three stages into one fresh directory"""
        dest = self.dir()
        self.collate(cfg=collation, dest=dest, **kwargs)
        self.tokenize(cfg=tokenization, processed=dest, **kwargs)
        self.winnow(cfg=winnowing, processed=dest, **kwargs)
        return Processed(dest, self.raw)

    def minimal(
        self,
        *,
        hospitalizations,
        vitals,
        collation=None,
        tokenization=None,
        tz=None,
        naive_times=False,
        winnowing=None,
    ):
        """
        collate and tokenize an explicitly specified minimal dataset;
        `collation` defaults to synth.minimal_collation_cfg()
        """
        self._n += 1
        raw = synth.write_minimal_dataset(
            self.root / f"raw{self._n}",
            hospitalizations=hospitalizations,
            vitals=vitals,
            tz=tz,
            naive_times=naive_times,
        )
        dest = self.dir()
        self.collate(
            cfg=collation if collation is not None else synth.minimal_collation_cfg(),
            raw=raw,
            dest=dest,
        )
        self.tokenize(cfg=tokenization, processed=dest)
        if winnowing is not None:
            self.winnow(cfg=winnowing, processed=dest)
        return Processed(dest, None)


@pytest.fixture(scope="session")
def raw_data(tmp_path_factory) -> synth.Manifest:
    """a synthetic raw dataset covering every table the default config reads"""
    return synth.write_raw_dataset(tmp_path_factory.mktemp("raw"))


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory, raw_data) -> Processed:
    """collate, tokenize, and winnow `raw_data` with the shipped defaults"""
    dest = tmp_path_factory.mktemp("processed")
    Collator(raw_data_home=raw_data.root, processed_data_home=dest).save_all()
    Tokenizer(processed_data_home=dest).save_all()
    Winnower(processed_data_home=dest).save_all()
    return Processed(dest, raw_data)


@pytest.fixture
def runner(tmp_path, raw_data, pipeline) -> Runner:
    """re-run stages with overridden configs, isolated to this test"""
    return Runner(tmp_path, raw_data, pipeline.path)
