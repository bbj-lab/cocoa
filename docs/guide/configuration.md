# Configuration

Each stage (collation, tokenization, winnowing, and visualization) is driven by a
YAML config. The package ships a default for each under
[`src/cocoa/config/`](https://github.com/bbj-lab/cocoa/tree/master/src/cocoa/config).
A config file of your own, passed with `-c` (or, to `cocoa pipeline`, with
`--collation-config`, `--tokenization-config`, and `--winnowing-config`),
replaces the default rather than merging into it, and you can change individual
keys on the command line (see [Overriding config keys](#overriding-config-keys)).

## Overriding config keys

To change a few keys without writing a new config file, list them after a
command's options, as dotted paths into its config:

```sh
cocoa tokenize -p processed/demo n_bins=5 insert_spacers=true
```

`cocoa pipeline` takes them too, each key starting with the stage it's for:

```sh
cocoa pipeline -r data-raw/demo -p processed/demo \
    collation.default_timezone=America/New_York tokenization.n_bins=5 \
    '~winnowing.threshold.duration_s' winnowing.threshold.first_occurrence=XFR-IN//icu
```

It checks every stage's overrides before collating, so an unreadable winnowing
override fails at once rather than after the first two stages have run.

The syntax borrows from
[Hydra's](https://hydra.cc/docs/advanced/override_grammar/basic/):

- `key=value` sets a key, adding it if it isn't there, e.g. a
  `horizon_after_threshold_s` for winnowing. Keys aren't checked against the
  config, so a misspelled one is added quietly and does nothing. A leading `+` or
  `++`, as Hydra marks an addition, is accepted and ignored.
- `~key` deletes a key.

Overrides apply in order to the config file passed with `-c` (or to the shipped
default when none is). Values are read as YAML: `0.5` is a float, `[a, b]` a
list, and `{train_frac: 0.5}` a block, merged into any block already there.

A few things to know:

- Setting a key to `null` doesn't switch it off where its presence is what
  counts. The winnower uses whichever of `threshold.duration_s`,
  `threshold.after_start_time_s`, and `threshold.first_occurrence` is there (the
  first in that order, if several are), so switching between them takes `~` on
  one and setting the other, as above. Setting `threshold.after_start_time_s`
  alone changes nothing, since the shipped `threshold.duration_s` outranks it.
- Lists are replaced whole: `entries` can't be edited one item at a time, so a
  change to an entry calls for a config file.
- YAML decides each value's type, so quote what has to stay a string:
  `'clocks=["00", "12"]'`, since an unquoted `00` is the number 0, and
  `'other_color="#737373"'`, since an unquoted `#` starts a comment.
- With `--tokenizer-home`, the loaded tokenizer's saved config is used as
  written: it is frozen, so neither overrides nor `-c` apply, and nothing from
  the shipped default is merged in (`cocoa tokenize` says so when given either).
- Quote anything starting with `~` in zsh, which otherwise reads it as a
  directory name and fails.
