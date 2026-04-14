## Configuring a new event token

Suppose you have a table `clif_medication_admin_intermittent_converted` that you
would like to add to the tokenization process:

```
┌────────────────────┬───────────────────────────────┬───────────────────────┬────────────────────┬─────────────────────────┬─────────────────────┬─────────────────────────────────┐
│ hospitalization_id ┆ admin_dttm                    ┆ med_category          ┆ med_dose_converted ┆ med_dose_unit_converted ┆ mar_action_category ┆ _convert_status                 │
│ ---                ┆ ---                           ┆ ---                   ┆ ---                ┆ ---                     ┆ ---                 ┆ ---                             │
│ str                ┆ datetime[μs, America/Chicago] ┆ str                   ┆ f64                ┆ str                     ┆ str                 ┆ str                             │
╞════════════════════╪═══════════════════════════════╪═══════════════════════╪════════════════════╪═════════════════════════╪═════════════════════╪═════════════════════════════════╡
│ 20008851           ┆ 2111-01-19 15:00:00 CST       ┆ morphine              ┆ 2.5                ┆ mg                      ┆ given               ┆ success                         │
│ 20008851           ┆ 2111-01-19 16:00:00 CST       ┆ morphine              ┆ 2.5                ┆ mg                      ┆ given               ┆ success                         │
│ 20008851           ┆ 2111-01-19 18:00:00 CST       ┆ morphine              ┆ 2.5                ┆ mg                      ┆ given               ┆ success                         │
│ 20008851           ┆ 2111-01-19 18:15:00 CST       ┆ morphine              ┆ 2.5                ┆ mg                      ┆ given               ┆ success                         │
│ 20008851           ┆ 2111-01-19 19:45:00 CST       ┆ sodium bicarbonate    ┆ 50.0               ┆ ml                      ┆ given               ┆ success                         │
│ …                  ┆ …                             ┆ …                     ┆ …                  ┆ …                       ┆ …                   ┆ …                               │
│ 29966638           ┆ 2110-09-15 16:00:00 CST       ┆ dextrose              ┆ 100.0              ┆ ml                      ┆ given               ┆ success                         │
│ 29966638           ┆ 2110-09-15 16:00:00 CST       ┆ dextrose_in_water_d5w ┆ 100.0              ┆ ml                      ┆ given               ┆ success                         │
│ 29966638           ┆ 2110-09-15 16:00:00 CST       ┆ metronidazole         ┆ 1.0                ┆ dose                    ┆ given               ┆ original unit dose is not reco… │
│ 29966638           ┆ 2110-09-15 19:57:00 CST       ┆ insulin               ┆ 2.0                ┆ u                       ┆ given               ┆ user-preferred unit units is n… │
│ 29966638           ┆ 2110-09-15 20:51:00 CST       ┆ hydromorphone         ┆ 0.5                ┆ mg                      ┆ given               ┆ success                         │
└────────────────────┴───────────────────────────────┴───────────────────────┴────────────────────┴─────────────────────────┴─────────────────────┴─────────────────────────────────┘
```

(It may have other columns too -- they will be ignored unless included in the
configuration.)

For each `hospitalization_id`, you want to insert tokens corresponding to
`med_category` with respective `med_dose_converted` (after preprocessing,
`med_dose_converted` is unique within each `med_category`.) However, you only
want to include these tokens if the medication was actually administered
`pl.col("mar_action_category") == "given"` and the conversion during
preprocessing completed successfully (`pl.col("_convert_status") == "success"`).

1. Insert an entry `config/collation/xxx.yaml` into the `entries` list:

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

2. Now when collation is run, codes such as `MED-INT//acetaminophen`,
   `MED-INT//acetaminophen`, `MED-INT//adenosine`, `MED-INT//adenosine` will be
   generated and inserted into the `meds.parquet` file along with
   `med_dose_converted` in the `numeric_value` column:

   ```
   ┌────────────┬─────────────────────┬────────────────────────────────┬───────────────┬────────────┐
   │ subject_id ┆ time                ┆ code                           ┆ numeric_value ┆ text_value │
   │ ---        ┆ ---                 ┆ ---                            ┆ ---           ┆ ---        │
   │ str        ┆ datetime[μs]        ┆ str                            ┆ f32           ┆ str        │
   ╞════════════╪═════════════════════╪════════════════════════════════╪═══════════════╪════════════╡
   │ 20008851   ┆ 2111-01-19 21:00:00 ┆ MED-INT//morphine              ┆ 2.5           ┆ null       │
   │ 20008851   ┆ 2111-01-19 22:00:00 ┆ MED-INT//morphine              ┆ 2.5           ┆ null       │
   │ 20008851   ┆ 2111-01-20 00:00:00 ┆ MED-INT//morphine              ┆ 2.5           ┆ null       │
   │ 20008851   ┆ 2111-01-20 00:15:00 ┆ MED-INT//morphine              ┆ 2.5           ┆ null       │
   │ 20008851   ┆ 2111-01-20 01:45:00 ┆ MED-INT//sodium_bicarbonate    ┆ 50.0          ┆ null       │
   │ …          ┆ …                   ┆ …                              ┆ …             ┆ …          │
   │ 29966638   ┆ 2110-09-15 18:54:00 ┆ MED-INT//dextrose              ┆ 100.0         ┆ null       │
   │ 29966638   ┆ 2110-09-15 18:54:00 ┆ MED-INT//dextrose_in_water_d5w ┆ 100.0         ┆ null       │
   │ 29966638   ┆ 2110-09-15 22:00:00 ┆ MED-INT//dextrose              ┆ 100.0         ┆ null       │
   │ 29966638   ┆ 2110-09-15 22:00:00 ┆ MED-INT//dextrose_in_water_d5w ┆ 100.0         ┆ null       │
   │ 29966638   ┆ 2110-09-16 02:51:00 ┆ MED-INT//hydromorphone         ┆ 0.5           ┆ null       │
   └────────────┴─────────────────────┴────────────────────────────────┴───────────────┴────────────┘
   ```

3. When tokenization is run, these codes will be picked up and processed as
   usual.
