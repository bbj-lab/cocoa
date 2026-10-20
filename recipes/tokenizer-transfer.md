## Tokenizer transfer: learn a tokenizer on one dataset and then apply it to a second dataset

In this example, we'll split the [MIMIC-IV demo](mimic-iv-demo.md) in two by when
its patients were treated, learn a tokenizer on the earlier patients, and apply
it to the later ones, as you might apply a model to data newer than any it was
trained on. The second dataset could just as well come from another hospital. The
commands run from the demo's working directory, once its step 3 has made the CLIF
tables.

0. Split the CLIF tables in two. MIMIC-IV shifts each patient's dates by its own
   amount, but the `anchor_year_group` of its `patients` table places each
   patient's care, roughly, in a span of three real years: 2011–2013 for 43 of
   the demo's patients, and 2014–2016 for the other 57. The script on the ETL's
   branch,
   [`CLIF-MIMIC/scripts/split_by_years.py`](https://github.com/bbj-lab/CLIF-MIMIC/blob/demo/scripts/split_by_years.py),
   writes each group's rows of every table to a directory of its own next to the
   demo's, `data-raw/demo-2011-2013` and `data-raw/demo-2014-2016`; run it with
   the Python that cocoa is installed in, which has polars:

    ```sh
    python CLIF-MIMIC/scripts/split_by_years.py data-raw/demo \
        mimic-iv-clinical-database-demo-2.2/hosp/patients.csv.gz
    ```

    The earlier patients have 147 of the demo's hospitalizations, and the
    later 128.

1. Run collation on each dataset separately with the same config. Keeping
   `default_timezone` the same across both matters if you tokenize with clock
   tokens, since `CLCK//HH` refers to an hour of the local day in that zone and
   the learned vocabulary carries no record of it:

    ```sh
    cocoa collate \
        --raw-data-home data-raw/demo-2011-2013 \
        --processed-data-home processed/demo-2011-2013 \
        --verbose

    cocoa collate \
        --raw-data-home data-raw/demo-2014-2016 \
        --processed-data-home processed/demo-2014-2016 \
        --verbose
    ```

2. Learn a tokenizer on the first dataset:

    ```sh
    cocoa tokenize \
      --processed-data-home processed/demo-2011-2013
    ```

    It learns 974 tokens from the 87 hospitalizations of the earlier patients'
    training split.

3. Supply the `--tokenizer-home` argument to the `tokenize` command to load the
   previously learned tokenizer (with fixed vocabulary and binning cutpoints):

    ```sh
    cocoa tokenize \
      --tokenizer-home processed/demo-2011-2013/tokenizer.yaml \
      --processed-data-home processed/demo-2014-2016 \
      --verbose
    ```

    The tokenizer's saved config is used as written, so it tokenizes the later
    patients exactly as it did the earlier ones; a `-c` config or overrides
    passed alongside `--tokenizer-home` don't apply. The vocabulary arrives
    frozen too, so the later patients' own word frequencies never enter into it:
    `min_training_ct` was applied once, to the earlier patients' training split,
    and is not reconsidered here. A code that is common in the later data but was
    too rare in the earlier data to be learned tokenizes to `UNK`, the same as a
    code the earlier data never saw at all. The `counts` in `tokenizer.yaml`
    likewise stay as the earlier data recorded them. Here 553 of the later
    timelines' 152,080 tokens, about 0.4%, are `UNK`, most of them codes that no
    earlier training hospitalization had, such as the infusion
    `MED-CTS//octreotide` and the ventilator mode `RESP//pressure_control`.

4. Proceed as usual:

    ```sh
    cocoa winnow \
      --processed-data-home processed/demo-2011-2013 \
      --verbose

    cocoa winnow \
      --processed-data-home processed/demo-2014-2016
    ```

5. To work with both halves as one dataset, combine them; their tokens mean the
   same thing, since both come from the earlier patients' tokenizer:

    ```sh
    cocoa combine-datasets \
      processed/demo-2011-2013 processed/demo-2014-2016 \
      --output-data-dir processed/demo-combined
    ```

    This writes the events, splits, timelines, and winnowed frames of all 275
    hospitalizations to `processed/demo-combined`, with the earlier patients'
    `tokenizer.yaml`. Each hospitalization keeps the split it was given, so the
    combined dataset has 180 training, 27 tuning, and 68 held-out
    hospitalizations, not the 187, 33, and 55 of the demo collated whole. It
    warns of a `Configuration mismatch` between the two `tokenizer.yaml` files,
    in `is_training` alone, since the later patients were tokenized by a frozen
    copy; any other key would mean the halves were tokenized differently.
