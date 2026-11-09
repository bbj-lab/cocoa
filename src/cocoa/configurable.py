#!/usr/bin/env python3

"""
configurable class with overridable defaults
"""

import collections.abc
import importlib.resources as resources
import inspect
import pathlib
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from cocoa.logger import Logger


def apply_overrides(cfg: DictConfig, overrides: collections.abc.Iterable[str]) -> None:
    """
    edits `cfg` in place with `overrides`, in order: `key=value` sets a key,
    adding it if `cfg` lacks it (a leading `+` or `++`, as Hydra writes an
    addition, is ignored), and `~key` deletes one, the only way to switch off a
    key toggled by its presence, since `key=null` leaves it present. Keys are
    dotted paths and values are read as yaml: `0.5` is a float, `[a, b]` a list,
    and `{train_frac: 0.5}` a block, merged into any block already there
    """
    for override in overrides:
        if override.startswith("~"):
            parent, _, leaf = override[1:].rpartition(".")
            del OmegaConf.select(cfg, parent)[leaf]  # `""` selects `cfg` itself
        elif "=" in override:
            cfg.merge_with(OmegaConf.from_dotlist([override.lstrip("+")]))
        else:  # which `from_dotlist` would read as `key=null`
            raise ValueError(f"can't read the override {override!r}: no `=`")


# from 2.4, omegaconf refuses a yaml document of more than 10,000 nodes unless
# told otherwise, and a tokenizer.yaml's lookup and counts soon outgrow that; so
# lift the limit where there is one, since cocoa parses only what it wrote and
# configs, which are as trusted as python anyway
UNLIMITED = (
    {"max_yaml_expanded_nodes": None}
    if "max_yaml_expanded_nodes" in inspect.signature(OmegaConf.load).parameters
    else {}
)


def parse_yaml(text: str) -> DictConfig | ListConfig:
    """yaml `text` as OmegaConf.create reads it, but of any size"""
    return OmegaConf.create(text, **UNLIMITED)


def load_yaml(path: pathlib.Path | str) -> DictConfig | ListConfig:
    """the yaml file at `path` as OmegaConf.load reads it, but of any size"""
    return OmegaConf.load(path, **UNLIMITED)


class Configurable:
    """
    takes a default configuration,
    allows the user to pass a configuration to override that,
    edits it with any command-line `overrides` (see `apply_overrides`), and
    finally considers keyword arguments that override all of these
    """

    default_file: str | None = None
    cfg: DictConfig

    def __init__(
        self,
        config_file: pathlib.Path | str | None = None,
        overrides: collections.abc.Iterable[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.config_file = config_file
        self.cfg = OmegaConf.merge(
            self.load_cfg(self.config_file, overrides),
            {k: v for k, v in kwargs.items() if v is not None},
        )

        self.logger = Logger()

    @classmethod
    def load_cfg(
        cls,
        config_file: pathlib.Path | str | None = None,
        overrides: collections.abc.Iterable[str] | None = None,
    ) -> DictConfig:
        """
        the passed config, or else the shipped default, edited by `overrides`;
        lets `cocoa pipeline` check every stage's overrides before the first runs
        """
        cfg = (
            OmegaConf.load(pathlib.Path(config_file).expanduser().resolve())
            if config_file is not None
            else OmegaConf.load(resources.files("cocoa.config") / cls.default_file)
            if cls.default_file is not None
            else OmegaConf.create()
        )  # options left unspecified in the passed config don't inherit the default
        apply_overrides(cfg, overrides or ())
        return cfg
