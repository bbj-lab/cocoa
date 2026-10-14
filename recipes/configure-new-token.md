## Configuring a new event token

Suppose you have a table `clif_medication_admin_intermittent_converted` that you
would like to add to the tokenization process, such as the one clifpy derives in
step 3 of the [MIMIC-IV demo](mimic-iv-demo.md):

```
┌────────────────────┬─────────────────────────┬─────────────────────────┬────────────────────┬─────────────────────────┬─────────────────────┬──────────────────────────────────┐
│ hospitalization_id ┆ admin_dttm              ┆ med_category            ┆ med_dose_converted ┆ med_dose_unit_converted ┆ mar_action_category ┆ _convert_status                  │
│ ---                ┆ ---                     ┆ ---                     ┆ ---                ┆ ---                     ┆ ---                 ┆ ---                              │
│ str                ┆ datetime[μs, UTC]       ┆ str                     ┆ f64                ┆ str                     ┆ str                 ┆ str                              │
╞════════════════════╪═════════════════════════╪═════════════════════════╪════════════════════╪═════════════════════════╪═════════════════════╪══════════════════════════════════╡
│ 20044587           ┆ 2113-08-25 21:29:00 UTC ┆ propofol                ┆ 10.000001          ┆ mg                      ┆ given               ┆ success                          │
│ 20044587           ┆ 2113-08-25 22:13:00 UTC ┆ propofol                ┆ 10.000001          ┆ mg                      ┆ given               ┆ success                          │
│ 20044587           ┆ 2113-08-25 23:08:00 UTC ┆ sodium chloride         ┆ 300.0              ┆ ml                      ┆ given               ┆ success                          │
│ 20044587           ┆ 2113-08-26 00:31:00 UTC ┆ cefazolin               ┆ 1.0                ┆ dose                    ┆ given               ┆ original unit dose is not recog… │
│ 20044587           ┆ 2113-08-26 00:31:00 UTC ┆ dextrose_5_water        ┆ 100.0              ┆ ml                      ┆ given               ┆ success                          │
│ …                  ┆ …                       ┆ …                       ┆ …                  ┆ …                       ┆ …                   ┆ …                                │
│ 29974575           ┆ 2131-03-08 11:00:00 UTC ┆ piperacillin_tazobactam ┆ 1.0                ┆ dose                    ┆ given               ┆ original unit dose is not recog… │
│ 29974575           ┆ 2131-03-08 13:04:00 UTC ┆ heparin                 ┆ 1.0                ┆ dose                    ┆ given               ┆ original unit dose is not recog… │
│ 29974575           ┆ 2131-03-08 19:08:00 UTC ┆ dextrose_5_water        ┆ 100.0              ┆ ml                      ┆ given               ┆ success                          │
│ 29974575           ┆ 2131-03-08 19:08:00 UTC ┆ piperacillin_tazobactam ┆ 1.0                ┆ dose                    ┆ given               ┆ original unit dose is not recog… │
│ 29974575           ┆ 2131-03-08 21:08:00 UTC ┆ heparin                 ┆ 1.0                ┆ dose                    ┆ given               ┆ original unit dose is not recog… │
└────────────────────┴─────────────────────────┴─────────────────────────┴────────────────────┴─────────────────────────┴─────────────────────┴──────────────────────────────────┘
```

(It has other columns too, such as the dose and unit as charted, `med_dose` and
`med_dose_unit` -- they will be ignored unless included in the configuration.)

For each `hospitalization_id`, you want to insert tokens corresponding to
`med_category` with respective `med_dose_converted` (after preprocessing,
`med_dose_unit_converted` is unique within each `med_category`.) However, you
only want to include these tokens if the medication was actually administered
`pl.col("mar_action_category") == "given"` and the conversion during
preprocessing completed successfully (`pl.col("_convert_status") == "success"`).

1. Add an entry to the `entries` list of your collation config (e.g.
   `collation.yaml`, passed via `cocoa collate -c collation.yaml`):

    ```yaml
    entries:
    …
      - table: clif_medication_admin_intermittent_converted
        prefix: MED-INT
        filter_expr:
          - pl.col("mar_action_category") == "given"
          - pl.col("_convert_status") == "success"
        code: med_category
        numeric_value: med_dose_converted
        time: admin_dttm
    ```

    (The shipped collation config already has this entry, along with one that
    keeps the doses clifpy couldn't convert as codes without values.)

2. Now when collation is run, codes such as `MED-INT//propofol`,
   `MED-INT//sodium_chloride`, `MED-INT//dextrose_5_water`, `MED-INT//morphine`
   will be generated and inserted into the `meds.parquet` file along with
   `med_dose_converted` in the `numeric_value` column:

    ```
    ┌────────────┬─────────────────────────┬───────────────────────────┬───────────────┬────────────┐
    │ subject_id ┆ time                    ┆ code                      ┆ numeric_value ┆ text_value │
    │ ---        ┆ ---                     ┆ ---                       ┆ ---           ┆ ---        │
    │ str        ┆ datetime[μs, UTC]       ┆ str                       ┆ f32           ┆ str        │
    ╞════════════╪═════════════════════════╪═══════════════════════════╪═══════════════╪════════════╡
    │ 20044587   ┆ 2113-08-25 21:29:00 UTC ┆ MED-INT//propofol         ┆ 10.000001     ┆ null       │
    │ 20044587   ┆ 2113-08-25 22:13:00 UTC ┆ MED-INT//propofol         ┆ 10.000001     ┆ null       │
    │ 20044587   ┆ 2113-08-25 23:08:00 UTC ┆ MED-INT//sodium_chloride  ┆ 300.0         ┆ null       │
    │ 20044587   ┆ 2113-08-26 00:31:00 UTC ┆ MED-INT//dextrose_5_water ┆ 100.0         ┆ null       │
    │ 20044587   ┆ 2113-08-26 00:53:00 UTC ┆ MED-INT//morphine         ┆ 2.0           ┆ null       │
    │ …          ┆ …                       ┆ …                         ┆ …             ┆ …          │
    │ 29974575   ┆ 2131-03-07 19:08:00 UTC ┆ MED-INT//dextrose_5_water ┆ 100.0         ┆ null       │
    │ 29974575   ┆ 2131-03-08 03:00:00 UTC ┆ MED-INT//dextrose_5_water ┆ 100.0         ┆ null       │
    │ 29974575   ┆ 2131-03-08 03:30:00 UTC ┆ MED-INT//magnesium        ┆ 2.0           ┆ null       │
    │ 29974575   ┆ 2131-03-08 11:00:00 UTC ┆ MED-INT//dextrose_5_water ┆ 100.0         ┆ null       │
    │ 29974575   ┆ 2131-03-08 19:08:00 UTC ┆ MED-INT//dextrose_5_water ┆ 100.0         ┆ null       │
    └────────────┴─────────────────────────┴───────────────────────────┴───────────────┴────────────┘
    ```

    The cefazolin given at 00:31, charted as `1 dose`, is left out, since its
    conversion didn't succeed.

3. When tokenization is run, these codes will be picked up and processed as usual
   — with one caveat if you've raised `min_training_ct` above its default of `0`:
   a code occurring fewer than that many times in the training split doesn't earn
   its own token and tokenizes to `UNK` instead. If your new table is sparse and
   you want each of its codes represented, keep the threshold low.
