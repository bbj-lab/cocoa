## Tokenizer transfer: learn a tokenizer on one dataset and then apply it to a second dataset

In this example, we'll train a tokenizer on mimic and apply it to ucmc.

<details>

<summary>Localize filenames by cluster.</summary>

```sh
case "$(uname -n)" in
    cri*)
        hm="/gpfs/data/bbj-lab/users/burkh4rt"
        ;;
    bbj-lab*)
        hm="/mnt/bbj-lab/users/burkh4rt"
        ;;
    *)
        echo "Unknown host $(uname -n)"
        ;;
esac
raw_mimic="${hm}/data-raw/mimic-2.1.0"
raw_ucmc="${hm}/data-raw/ucmc-2.1.0"
```

</details>

1. Run collation on each dataset separately with the same config:

   ```sh
   cocoa collate \
       --raw-data-home ${raw_mimic} \
       --processed-data-home ./processed/mimic

   cocoa collate \
       --raw-data-home ${raw_ucmc} \
       --processed-data-home ./processed/ucmc
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
   --tokenizer-home ./processed/tokenizer-lsolo.yaml \
   --processed-data-home ./processed/ucmc
   ```

4. Proceed as usual:

   ```sh
   cocoa winnow \
   --processed-data-home ./processed/mimic

   cocoa winnow \
   --processed-data-home ./processed/ucmc
   ```
