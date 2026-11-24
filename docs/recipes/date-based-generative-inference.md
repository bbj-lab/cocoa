# Date-based generative inference

To prepare data for inference at each timeline's first midnight (here on the
[MIMIC-IV demo](mimic-iv-demo.md), from its working directory), change a few keys
of the shipped configs on the command line, in a single run of the pipeline:

```sh
cocoa pipeline -r data-raw/demo -p processed/demo-midnight \
    collation.default_timezone=America/New_York \
    tokenization.insert_clocks=true 'tokenization.clocks=["00"]' \
    '~winnowing.threshold.duration_s' winnowing.threshold.first_occurrence=CLCK//00
```

Each override starts with the stage it's for (see
[Overriding config keys](../guide/configuration.md#overriding-config-keys)):

- `tokenization.insert_clocks=true 'tokenization.clocks=["00"]'` turns on clock
  tokens and keeps only midnight's, `CLCK//00`, of the shipped `clocks`, which
  mark every fourth hour. The quotes keep `00` a string, which YAML would
  otherwise read as the number 0.
- `collation.default_timezone=America/New_York` puts that midnight in Boston's
  time. Clock hours are read in the collation config's `default_timezone` (`UTC`
  if unset), and the demo's CLIF tables are in UTC, but MIMIC-IV's times are
  Boston's. (For MIMIC-IV's shifted years, polars keeps that zone on standard
  time all year, as the demo's conversion to UTC did, so `CLCK//00` falls at
  midnight as MIMIC-IV records it.)
- `'~winnowing.threshold.duration_s' winnowing.threshold.first_occurrence=CLCK//00`
  cuts each timeline at its first `CLCK//00`. The shipped threshold, 24 hours
  after the timeline's first event, has to be deleted, since the winnower uses
  `duration_s` before `first_occurrence` when given both; the quotes keep zsh
  from reading the `~` as a directory.

Only timelines that reach a midnight are kept: on the demo, 202 of the 208
training hospitalizations, 38 of the 39 tuning, and 62 of the 63 held out. The
held-out one left out, 22380825, is a five-hour stay. Hospitalization 27708593,
admitted at 07:15, is cut at midnight that night, before its transfer to the ICU
and its first tachycardia. Some are cut before they are admitted: 52 of the 302
kept open with an arrival in the emergency department the day before their
admission, so their pasts end at the midnight before it, without the age, sex,
and other tokens timed at admission.

To keep these settings in files instead, copy the shipped tokenization and
winnowing configs, make the same changes, and pass the copies to `cocoa pipeline`
with `--tokenization-config` and `--winnowing-config`.
