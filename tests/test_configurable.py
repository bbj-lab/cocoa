#!/usr/bin/env python3

"""config resolution: shipped defaults, user configs, kwargs, nested merging"""

import importlib.resources as resources
import logging
import pathlib
import tomllib

import omegaconf.errors as oc_errors
import pytest
from omegaconf import DictConfig, OmegaConf

from cocoa.collator import Collator
from cocoa.configurable import Configurable
from cocoa.logger import Logger
from cocoa.tokenizer import Tokenizer
from cocoa.winnower import Winnower

REPO = pathlib.Path(__file__).resolve().parents[1]
SHIPPED = {"collation.yaml", "tokenization.yaml", "winnowing.yaml"}

# keys the shipped defaults define, and that a user config is therefore able
# to *lose* by omitting them; see the "config resolution" note in CLAUDE.md
DEFAULT_ONLY_KEYS = {
    "collation": (
        "group_id",
        "default_timezone",
        "subject_splits",
        "reference",
        "pass_through_columns",
        "entries",
    ),
    "tokenization": (
        "fused",
        "include_numeric_values",
        "ordering",
        "insert_spacers",
        "spacers",
        "insert_clocks",
        "clocks",
    ),
    "winnowing": ("threshold", "splits"),
}


class Bare(Configurable):
    """exercises the base class with no shipped default at all"""

    default_file = None


class Phantom(Configurable):
    """names a default that does not exist, to detect if it gets loaded"""

    default_file = "no_such_default.yaml"


class Fake(Configurable):
    """borrows a real shipped default"""

    default_file = "tokenization.yaml"


def write(path: pathlib.Path, text: str) -> pathlib.Path:
    path.write_text(text)
    return path


@pytest.fixture
def tkzr_home(tmp_path) -> pathlib.Path:
    """a processed dir holding just enough tokenizer.yaml for a Winnower"""
    home = tmp_path / "processed"
    home.mkdir()
    lookup = {"UNK": 0, "XFR-IN//icu": 1, "LABEL//death": 2, "VTL//hr_q1": 3}
    write(
        home / "tokenizer.yaml", OmegaConf.to_yaml(OmegaConf.create({"lookup": lookup}))
    )
    return home


def build(stage: str, tmp_path, tkzr_home, cfg=None, **kwargs) -> Configurable:
    """construct the stage for `stage` with `cfg` as its user config"""
    if stage == "collation":
        return Collator(
            collation_cfg=cfg,
            raw_data_home=tmp_path,
            processed_data_home=tmp_path / "out",
            **kwargs,
        )
    if stage == "tokenization":
        return Tokenizer(tokenization_cfg=cfg, processed_data_home=tmp_path, **kwargs)
    return Winnower(winnowing_cfg=cfg, processed_data_home=tkzr_home, **kwargs)


# ---------------------------------------------------------------- shipped defaults


@pytest.mark.parametrize(
    "cls,name",
    [
        (Collator, "collation.yaml"),
        (Tokenizer, "tokenization.yaml"),
        (Winnower, "winnowing.yaml"),
    ],
)
def test_each_stage_declares_its_shipped_default_file(cls, name):
    assert cls.default_file == name
    assert (resources.files("cocoa.config") / name).is_file()


def test_base_class_declares_no_default_file():
    assert Configurable.default_file is None


def test_every_shipped_yaml_is_packaged_and_claimed_by_a_stage():
    on_disc = {p.name for p in (REPO / "src" / "cocoa" / "config").glob("*.yaml")}
    packaged = {
        p.name
        for p in resources.files("cocoa.config").iterdir()
        if p.name.endswith(".yaml")
    }
    assert on_disc == SHIPPED
    assert packaged == SHIPPED
    claimed = {c.default_file for c in (Collator, Tokenizer, Winnower)}
    assert claimed == SHIPPED


@pytest.mark.parametrize("name", sorted(SHIPPED))
def test_shipped_yaml_loads_as_a_nonempty_mapping(name):
    loaded = OmegaConf.load(resources.files("cocoa.config") / name)
    assert isinstance(loaded, DictConfig)
    assert len(loaded) > 0


def test_pyproject_ships_the_config_yamls_as_package_data():
    """without this entry a wheel would carry no default configs"""
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())
    data = pyproject["tool"]["setuptools"]["package-data"]
    assert "*.yaml" in data["cocoa.config"]


def test_collation_default_matches_the_documented_values(tmp_path, tkzr_home):
    cfg = build("collation", tmp_path, tkzr_home).cfg
    assert cfg.subject_id == "hospitalization_id"
    assert cfg.group_id == "patient_id"
    assert cfg.default_timezone == "UTC"
    assert cfg.subject_splits.train_frac == pytest.approx(0.7)
    assert cfg.subject_splits.tuning_frac == pytest.approx(0.1)
    assert cfg.reference.table == "clif_hospitalization"
    assert cfg.reference.start_time == "admission_dttm"
    assert cfg.reference.end_time == "discharge_dttm"
    assert "age_at_admission" in list(cfg.pass_through_columns)
    assert [e.prefix for e in cfg.entries][:3] == ["RACE", "ETHN", "SEX"]


def test_tokenization_default_matches_the_documented_values(tmp_path, tkzr_home):
    cfg = build("tokenization", tmp_path, tkzr_home).cfg
    assert cfg.n_bins == 10 and isinstance(cfg.n_bins, int)
    assert cfg.fused is True
    assert cfg.include_numeric_values is False
    assert cfg.insert_spacers is False
    assert cfg.insert_clocks is False
    ordering = list(cfg.ordering)
    assert ordering[0] == "BOS" and ordering[-1] == "EOS"
    assert len(set(ordering)) == len(ordering)
    assert {"CLCK", "TIME", "VTL", "LABEL"} <= set(ordering)
    # the !!str tags matter: a clock token is CLCK//00, not CLCK//0
    assert list(cfg.clocks) == ["00", "04", "08", "12", "16", "20"]
    assert cfg.spacers["1d-3d"] == 1440
    assert cfg.spacers["5m-15m"] == 5


def test_winnowing_default_matches_the_documented_values(tmp_path, tkzr_home):
    cfg = build("winnowing", tmp_path, tkzr_home).cfg
    assert cfg.threshold.duration_s == 86400
    assert "first_occurrence" not in cfg.threshold  # commented out in the yaml
    assert "horizon_after_threshold_s" not in cfg  # ditto
    assert list(cfg.splits) == ["train", "tuning", "held_out"]
    assert {"XFR-IN//icu", "RESP//imv", "DSCG//expired", "LABEL//*"} <= set(
        cfg.outcome_tokens
    )


@pytest.mark.parametrize("stage", ["collation", "tokenization", "winnowing"])
def test_default_cfg_is_exactly_the_packaged_resource(stage, tmp_path, tkzr_home):
    """the default comes from the installed package, not from the cwd"""
    cfg = build(stage, tmp_path, tkzr_home).cfg
    assert cfg == OmegaConf.load(resources.files("cocoa.config") / f"{stage}.yaml")


def test_default_cfg_is_not_shared_between_instances(tmp_path, tkzr_home):
    first = build("tokenization", tmp_path, tkzr_home)
    first.cfg.n_bins = 99
    second = build("tokenization", tmp_path, tkzr_home)
    assert second.cfg.n_bins == 10


def test_missing_shipped_default_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Phantom()


# ------------------------------------------- headline: a user config replaces default


@pytest.mark.parametrize("stage", ["collation", "tokenization", "winnowing"])
def test_user_config_does_not_inherit_omitted_default_keys(stage, tmp_path, tkzr_home):
    """a -c config replaces the shipped default; omitted keys are simply gone"""
    kept = {
        "collation": "subject_id: hospitalization_id\n",
        "tokenization": "n_bins: 4\n",
        # outcome_tokens is read in Winnower.__init__, so it cannot be dropped
        "winnowing": "outcome_tokens:\n  - LABEL//*\n",
    }[stage]
    cfg_file = write(tmp_path / f"{stage}.yaml", kept)
    cfg = build(stage, tmp_path, tkzr_home, cfg=cfg_file).cfg
    for key in DEFAULT_ONLY_KEYS[stage]:
        assert key not in cfg, f"{key} leaked in from the shipped default"
        assert cfg.get(key) is None
    assert len(cfg) == 1  # only what the user wrote


def test_user_config_short_circuits_loading_the_default(tmp_path):
    """Phantom's default cannot be read, yet a user config constructs fine"""
    cfg_file = write(tmp_path / "u.yaml", "a: 1\n")
    assert Phantom(config_file=cfg_file).cfg == OmegaConf.create({"a": 1})


def test_user_config_omitting_outcome_tokens_breaks_the_winnower(tmp_path, tkzr_home):
    """the documented gotcha, as a user would meet it"""
    cfg_file = write(tmp_path / "w.yaml", "threshold:\n  duration_s: 3600\n")
    with pytest.raises(oc_errors.ConfigAttributeError):
        build("winnowing", tmp_path, tkzr_home, cfg=cfg_file)


def test_user_config_narrows_the_grokked_outcome_tokens(tmp_path, tkzr_home):
    """dropping the default's patterns really does drop matching tokens"""
    full = build("winnowing", tmp_path, tkzr_home).grokked_outcome_tokens
    assert sorted(full) == ["LABEL//death", "XFR-IN//icu"]
    cfg_file = write(tmp_path / "w.yaml", "outcome_tokens:\n  - LABEL//*\n")
    only_labels = build(
        "winnowing", tmp_path, tkzr_home, cfg=cfg_file
    ).grokked_outcome_tokens
    assert only_labels == ["LABEL//death"]


def test_config_file_is_recorded_as_given(tmp_path):
    cfg_file = write(tmp_path / "u.yaml", "a: 1\n")
    assert Bare(config_file=str(cfg_file)).config_file == str(cfg_file)
    assert Bare().config_file is None


# ------------------------------------------------------------------------- kwargs


@pytest.mark.parametrize(
    "kwargs,key,expected",
    [
        ({"n_bins": 3}, "n_bins", 3),
        ({"fused": False}, "fused", False),  # a false override must stick
        ({"include_numeric_values": True}, "include_numeric_values", True),
        ({"insert_clocks": True}, "insert_clocks", True),
        ({"banana": "yellow"}, "banana", "yellow"),  # new keys are added
    ],
)
def test_kwargs_override_the_default(kwargs, key, expected, tmp_path, tkzr_home):
    cfg = build("tokenization", tmp_path, tkzr_home, **kwargs).cfg
    assert cfg[key] == expected


def test_kwargs_are_not_type_checked(tmp_path, tkzr_home):
    """
    the configs carry no schema, so a wrong-typed override is accepted here
    and only surfaces later (as a polars error) downstream
    """
    cfg = build("tokenization", tmp_path, tkzr_home, n_bins="ten", clocks=[0, 4]).cfg
    assert cfg.n_bins == "ten"
    assert list(cfg.clocks) == [0, 4]


@pytest.mark.parametrize("value", [False, 0, "", [], {}])
def test_falsey_kwargs_are_not_dropped(value):
    cfg = Bare(thing=value).cfg
    assert "thing" in cfg
    assert cfg.thing == value


def test_none_kwargs_are_ignored(tmp_path, tkzr_home):
    """None means "unspecified"; it must not blank out a configured key"""
    cfg = build(
        "tokenization",
        tmp_path,
        tkzr_home,
        n_bins=None,
        fused=None,
        ordering=None,
        spacers=None,
    ).cfg
    assert cfg.n_bins == 10
    assert cfg.fused is True
    assert list(cfg.ordering)[0] == "BOS"
    assert len(cfg.spacers) == 13


def test_none_kwargs_do_not_add_keys():
    cfg = Bare(a=1, b=None).cfg
    assert OmegaConf.to_container(cfg) == {"a": 1}


def test_kwargs_override_a_user_config(tmp_path):
    cfg_file = write(tmp_path / "u.yaml", "n_bins: 3\nfused: true\n")
    cfg = Fake(config_file=cfg_file, n_bins=7).cfg
    assert cfg.n_bins == 7
    assert cfg.fused is True  # untouched by the kwarg


def test_none_kwargs_do_not_erase_a_user_config(tmp_path):
    cfg_file = write(tmp_path / "u.yaml", "n_bins: 3\n")
    assert Fake(config_file=cfg_file, n_bins=None).cfg.n_bins == 3


@pytest.mark.parametrize(
    "stage,excluded",
    [
        ("collation", ("raw_data_home", "processed_data_home", "collation_cfg")),
        ("tokenization", ("processed_data_home", "tokenization_cfg", "is_training")),
        ("winnowing", ("processed_data_home", "winnowing_cfg", "is_training")),
    ],
)
def test_named_constructor_arguments_stay_out_of_cfg(
    stage, excluded, tmp_path, tkzr_home
):
    cfg = build(stage, tmp_path, tkzr_home).cfg
    for key in excluded:
        assert key not in cfg


def test_is_training_is_an_attribute_not_a_config_key(tmp_path):
    tokenizer = Tokenizer(processed_data_home=tmp_path, is_training=False)
    assert tokenizer.is_training is False
    assert "is_training" not in tokenizer.cfg


# ------------------------------------------------------------------ nested merging


def test_dict_kwarg_deep_merges_into_a_default_block(tmp_path, tkzr_home):
    cfg = build(
        "collation", tmp_path, tkzr_home, subject_splits={"train_frac": 0.5}
    ).cfg
    assert cfg.subject_splits.train_frac == pytest.approx(0.5)
    assert cfg.subject_splits.tuning_frac == pytest.approx(0.1)  # kept from default
    assert set(cfg.subject_splits.keys()) == {"train_frac", "tuning_frac"}


def test_dict_kwarg_deep_merges_into_the_reference_block(tmp_path, tkzr_home):
    cfg = build(
        "collation",
        tmp_path,
        tkzr_home,
        reference={"table": "my_admissions", "extra": 1},
    ).cfg
    assert cfg.reference.table == "my_admissions"
    assert cfg.reference.start_time == "admission_dttm"  # sibling survives
    assert cfg.reference.end_time == "discharge_dttm"
    assert len(cfg.reference.augmentation_tables) == 1
    assert cfg.reference.extra == 1


def test_dict_kwarg_deep_merges_into_a_user_config(tmp_path):
    cfg_file = write(tmp_path / "u.yaml", "spacers:\n  1h-2h: 60\n")
    cfg = Fake(config_file=cfg_file, spacers={"6mt+": 262980}).cfg
    assert OmegaConf.to_container(cfg.spacers) == {"1h-2h": 60, "6mt+": 262980}


def test_nested_dict_kwarg_creates_a_missing_block():
    cfg = Bare(reference={"table": {"name": "t", "rows": 3}}).cfg
    assert cfg.reference.table.name == "t"
    assert cfg.reference.table.rows == 3


def test_list_inside_a_dict_kwarg_replaces_wholesale(tmp_path, tkzr_home):
    cfg = build(
        "collation",
        tmp_path,
        tkzr_home,
        reference={"augmentation_tables": [{"table": "only_this", "key": "k"}]},
    ).cfg
    # a list value replaces wholesale rather than merging element-wise
    assert [t.table for t in cfg.reference.augmentation_tables] == ["only_this"]
    assert cfg.reference.table == "clif_hospitalization"


@pytest.mark.parametrize(
    "key,value,other,other_len",
    [("clocks", ["00"], "spacers", 13), ("ordering", ["BOS", "EOS"], "clocks", 6)],
)
def test_list_kwarg_replaces_the_default_list(
    key, value, other, other_len, tmp_path, tkzr_home
):
    cfg = build("tokenization", tmp_path, tkzr_home, **{key: value}).cfg
    assert list(cfg[key]) == value
    assert len(cfg[other]) == other_len  # neighbouring blocks untouched


# ------------------------------------------------------------- no default at all


def test_no_default_file_and_no_config_gives_an_empty_cfg():
    cfg = Bare().cfg
    assert isinstance(cfg, DictConfig)
    assert len(cfg) == 0
    assert OmegaConf.to_container(cfg) == {}


def test_no_default_file_with_a_user_config_uses_only_that(tmp_path):
    cfg_file = write(tmp_path / "u.yaml", "a: 1\nb:\n  c: 2\n")
    cfg = Bare(config_file=cfg_file).cfg
    assert OmegaConf.to_container(cfg) == {"a": 1, "b": {"c": 2}}


# ------------------------------------------------------------------- config paths


@pytest.mark.parametrize("as_type", [str, pathlib.Path])
def test_config_file_accepts_str_and_path(as_type, tmp_path):
    cfg_file = write(tmp_path / "u.yaml", "n_bins: 5\n")
    assert Fake(config_file=as_type(cfg_file)).cfg.n_bins == 5


def test_config_file_expands_a_home_relative_path(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    write(home / "mine.yaml", "n_bins: 6\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    assert Fake(config_file="~/mine.yaml").cfg.n_bins == 6


def test_config_file_resolves_relative_to_the_cwd(tmp_path, monkeypatch):
    write(tmp_path / "mine.yaml", "n_bins: 8\n")
    monkeypatch.chdir(tmp_path)
    assert Fake(config_file="mine.yaml").cfg.n_bins == 8


def test_absent_config_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        Fake(config_file=tmp_path / "nope.yaml")


def test_empty_config_file_leaves_only_the_kwargs(tmp_path):
    """an empty -c file is a config that specifies nothing, not a no-op"""
    cfg_file = write(tmp_path / "empty.yaml", "")
    cfg = Fake(config_file=cfg_file, n_bins=3).cfg
    assert OmegaConf.to_container(cfg) == {"n_bins": 3}


def test_config_interpolation_is_resolved(tmp_path):
    cfg_file = write(tmp_path / "u.yaml", "a: 7\nb: ${a}\n")
    assert Bare(config_file=cfg_file).cfg.b == 7


# ------------------------------------------------------------------- cfg and logger


def test_cfg_get_returns_the_fallback_for_a_missing_key(tmp_path, tkzr_home):
    cfg = build("tokenization", tmp_path, tkzr_home).cfg
    assert cfg.get("nonesuch", "fallback") == "fallback"
    assert cfg.get("nonesuch") is None
    assert cfg.get("n_bins", 99) == 10


@pytest.mark.parametrize(
    "access,error",
    [
        (lambda cfg: cfg.nonesuch, oc_errors.ConfigAttributeError),
        (lambda cfg: cfg["nonesuch"], oc_errors.ConfigKeyError),
    ],
)
def test_missing_key_raises_on_direct_access(access, error, tmp_path, tkzr_home):
    cfg = build("tokenization", tmp_path, tkzr_home).cfg
    with pytest.raises(error):
        access(cfg)


def test_missing_key_errors_are_also_ordinary_python_errors():
    """so `except (AttributeError, KeyError)` still catches them"""
    assert issubclass(oc_errors.ConfigAttributeError, AttributeError)
    assert issubclass(oc_errors.ConfigKeyError, KeyError)


@pytest.mark.parametrize("stage", ["collation", "tokenization", "winnowing"])
def test_logger_is_a_cocoa_logger(stage, tmp_path, tkzr_home):
    logger = build(stage, tmp_path, tkzr_home).logger
    assert isinstance(logger, Logger)
    assert isinstance(logger, logging.Logger)
    assert logger.name == "cocoa"
    assert logger.level == logging.INFO
    assert logger.propagate is False


def test_base_class_alone_gets_a_logger():
    assert isinstance(Bare().logger, Logger)
