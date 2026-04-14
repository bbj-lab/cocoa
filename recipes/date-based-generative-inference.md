## Date-based generative inference

To prepare data for inference at first midnight after admission,

1. Configure midnight clock token generation in `config/tokenization/xxx.yaml`:

   ```yaml
   insert_clocks: !!bool true
   …
   clocks:
     - !!str 00 # produces token CLCK//00
   ```

2. Configure winnowing to threshold at the first occurrence of `CLCK//00` in
   `config/winnowing/xxx.yaml`:

   ```yaml
   threshold:
     first_occurrence: CLCK//00
   ```

3. Run the pipeline.

   ```sh
   uv run cocoa pipeline
   ```
