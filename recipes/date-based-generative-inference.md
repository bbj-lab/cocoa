## Date-based generative inference

To prepare data for inference at first midnight after admission,

1. In your tokenization config (e.g. `tokenization.yaml`), configure midnight
   clock token generation:

   ```yaml
   insert_clocks: !!bool true
   …
   clocks:
     - !!str 00 # produces token CLCK//00
   ```

2. In your winnowing config (e.g. `winnowing.yaml`), threshold at the first
   occurrence of `CLCK//00`:

   ```yaml
   threshold:
     first_occurrence: CLCK//00
   ```

3. Run the pipeline, passing your configs with `-c`/`--*-config` (any stage
   without an override uses the shipped default):

   ```sh
   cocoa pipeline \
     --tokenization-config tokenization.yaml \
     --winnowing-config winnowing.yaml \
     --raw-data-home ./raw-data \
     --processed-data-home ./processed
   ```
