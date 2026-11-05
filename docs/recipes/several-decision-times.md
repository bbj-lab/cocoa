# Inference at several decision times

A model put to work in a hospital doesn't predict once per stay, but again and
again as the stay goes on. To evaluate one that way, at several times into each
stay (here a day, three days, and a week after admission, on the
[MIMIC-IV demo](mimic-iv-demo.md), from its working directory), winnow the
tokenized timelines once for each time, each into a directory of its own:

1. In your winnowing config (e.g. `winnowing.yaml`, a copy of the shipped one),
   threshold at a time after each subject's start time, in place of its
   `duration_s`:

    ```yaml
    threshold:
        after_start_time_s: !!int 86400 # each run below sets its own
    ```

    Keep it the config's only threshold: given a `duration_s` too, the winnower
    would cut there and ignore the overrides below (see
    [winnowing](../guide/winnowing.md)).

2. Winnow once for each time, overriding the threshold on the command line and
   sending each run's files to an `--output-home` of its own:

    ```sh
    mkdir -p logs
    for h in 24 72 168; do
        cocoa winnow \
            --processed-data-home processed/demo \
            --output-home processed/demo/${h}h \
            --winnowing-config winnowing.yaml \
            --verbose \
            threshold.after_start_time_s=$((h * 3600)) \
            2>&1 | tee logs/winnowing-${h}h.log
    done
    ```

    Each run cuts every timeline `h` hours after its subject's start time (the
    `admission_dttm`, under the shipped collation config) and writes its
    `*_for_inference.parquet` files to `processed/demo/24h`,
    `processed/demo/72h`, or `processed/demo/168h`, along with copies of the
    processed directory's files, so that each is a processed directory of its
    own. `--verbose` prints how often each outcome falls on each side of the cut,
    which `tee` keeps in `logs/`.

On the demo, the three runs keep 55, 37, and 10 of the 63 held-out
hospitalizations (and 184, 144, and 78 of the 208 training, and 34, 25, and 12 of
the 39 tuning): those whose timelines run past the cut. A stay kept at one time
is kept at every earlier one. A day in, 24 of the 55 have already been
transferred to the ICU (`XFR-IN//icu_past`), and 5 will be later in the stay
(`XFR-IN//icu_future`); three days in, 21 of the 37 have been, and 2 will be; a
week in, 7 of the 10 have been, and none will be.

For decision times at a regular stride, as a model scoring each patient every 8
hours over the first week would need, loop over `$(seq 8 8 168)` instead. Each
run copies the processed directory's files, so on a large dataset, mind the disk
that 21 copies take.

## Evaluating at several decision times

Each directory holds one row per hospitalization, as after any winnowing, so a
metric computed on one of them counts each stay once. Evaluate each directory
separately, and compare them to see how performance changes with time into the
stay. Keep in mind:

- Pooling the directories into one evaluation counts a stay once for each time it
  reaches, which weighs long stays more heavily, and makes their rows correlated.
  To put a confidence interval on a pooled metric, resample hospitalizations, not
  rows.
- Later times keep fewer, longer stays: a week in, only 10 held-out stays remain.
  To separate the effect of time into the stay from that of who is still in
  hospital, compare the runs on just the stays the latest one keeps, which every
  run keeps too.
- An outcome in the past is known at the cut. When predicting one that counts
  only once, such as a first transfer to the ICU, you may want to leave out the
  stays already flagged with it in the past: a day in, that leaves 31 stays, 2 of
  them transferred later.

## Within a horizon

To ask whether an outcome comes within a day of each decision time, rather than
at any later point in the stay, add a horizon to the config:

```yaml
horizon_after_threshold_s: !!int 86400 # 1 day
```

The window counts from the last event before the cut, not from the cut itself, so
it closes early by the time between them: on the demo, by a median of 2 hours a
day in, and of 5 hours three days and a week in. Where the record thins out
before the cut, the window can close before the next event comes: a day in, 5 of
the 55 held-out stays keep no future at all, though each has events within a day
of the cut.
