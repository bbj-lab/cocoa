## Date-based generative inference

To prepare data for inference at first midnight after admission,

1. Configure midnight clock token generation in your tokenization config YAML (a
   copy of the packaged default at
   [`src/cocoa/config/tokenization.yaml`](../src/cocoa/config/tokenization.yaml)):

   ```yaml
   insert_clocks: !!bool true
   …
   clocks:
     - !!str 00 # produces token CLCK//00
   ```

2. Configure winnowing to threshold at the first occurrence of `CLCK//00` in your
   winnowing config YAML (a copy of the packaged default at
   [`src/cocoa/config/winnowing.yaml`](../src/cocoa/config/winnowing.yaml)):

   ```yaml
   threshold:
     first_occurrence: CLCK//00
   ```

3. Run the pipeline, passing your custom configs and data paths:

   ```sh
   cocoa pipeline \
     --raw-data-home ~/path/to/raw \
     --processed-data-home ~/path/to/output \
     --tokenization-config ./my-tokenization.yaml \
     --winnowing-config ./my-winnowing.yaml
   ```
