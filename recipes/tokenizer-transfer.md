## Tokenizer transfer: learn a tokenizer on one dataset and then apply it to a second dataset

In this example, we'll train a tokenizer on mimic and apply it to ucmc.

0. Set the paths to your raw datasets:

    ```sh
    raw_mimic="/path/to/data-raw/mimic-2.1.0"
    raw_ucmc="/path/to/data-raw/ucmc-2.1.0"
    ```

1. Run collation on each dataset separately with the same config. Keeping
   `default_timezone` the same across both matters if you tokenize with clock
   tokens, since `CLCK//HH` refers to an hour of the local day in that zone and
   the learned vocabulary carries no record of it:

    ```sh
    cocoa collate \
        --raw-data-home ${raw_mimic} \
        --processed-data-home ./processed/mimic \
        --verbose

    cocoa collate \
        --raw-data-home ${raw_ucmc} \
        --processed-data-home ./processed/ucmc \
        --verbose
    ```

2. Learn a tokenizer on the first dataset:

    ```sh
    cocoa tokenize \
      --processed-data-home ./processed/mimic
    ```

3. Supply the `--tokenizer-home` argument to the `tokenize` command to load the
   previously learned tokenizer (with fixed vocabulary and binning cutpoints):

    ```sh
    cocoa tokenize \
      --tokenizer-home ./processed/mimic/tokenizer.yaml \
      --processed-data-home ./processed/ucmc \
      --verbose
    ```

4. Proceed as usual:

    ```sh
    cocoa winnow \
      --processed-data-home ./processed/mimic \
      --verbose

    cocoa winnow \
      --processed-data-home ./processed/ucmc
    ```
