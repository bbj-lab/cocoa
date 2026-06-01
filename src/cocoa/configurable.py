#!/usr/bin/env python3

"""
configurable class with overridable defaults
"""

import importlib.resources as resources
import pathlib

from omegaconf import OmegaConf

from cocoa.logger import Logger


class Configurable:
    """
    takes a default configuration,
    allows the user to pass a configuration to override that, and
    finally considers keyword arguments that override both
    """

    default_file: str | None = None

    def __init__(self, config_file: pathlib.Path | str = None, **kwargs):
        self.config_file = config_file
        default_file = self.default_file
        self.cfg = OmegaConf.merge(
            OmegaConf.load(pathlib.Path(self.config_file).expanduser().resolve())
            if self.config_file is not None
            else OmegaConf.load(resources.files("cocoa.config") / default_file)
            if default_file is not None
            else OmegaConf.create(),  # options left unspecified in the passed config
            # should not inherit from default
            {k: v for k, v in kwargs.items() if v is not None},
        )

        self.logger = Logger()
