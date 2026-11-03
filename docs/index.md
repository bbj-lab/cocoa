---
hide:
  - navigation
---

--8<-- "README.md:intro"

<div class="grid cards" markdown>

-   :material-table-merge-cells:{ .lg .middle } __Collate__

    ---

    Pull raw parquet/csv tables into one denormalized MEDS-like frame, plus
    train / tuning / held-out splits, chronological or random.

    [:octicons-arrow-right-24: Collation](guide/collation.md)

-   :material-tag-multiple:{ .lg .middle } __Tokenize__

    ---

    Convert events into integer token sequences with a vocabulary and quantile
    bins learned on training data, then frozen.

    [:octicons-arrow-right-24: Tokenization](guide/tokenization.md)

-   :material-filter-variant:{ .lg .middle } __Winnow__

    ---

    Split held-out timelines into past / future at a threshold and flag outcome
    tokens for evaluation.

    [:octicons-arrow-right-24: Winnowing](guide/winnowing.md)

-   :material-book-open-variant:{ .lg .middle } __Recipes__

    ---

    Task-oriented guides for common workflows, from an end-to-end run on the
    public MIMIC-IV demo to tokenizer transfer and date-based inference.

    [:octicons-arrow-right-24: Recipes](recipes/index.md)

</div>

--8<-- "README.md:install"

--8<-- "README.md:references"
