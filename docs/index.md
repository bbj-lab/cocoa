---
hide:
  - navigation
---

{%
   include-markdown "../README.md"
   heading-offset=0
   end="<!-- cards-anchor -->"
%}

<div class="grid cards" markdown>

-   :material-table-merge-cells:{ .lg .middle } __Collate__

    ---

    Pull raw parquet/csv tables into one denormalized MEDS-like frame, plus
    chronological train / tuning / held-out splits.

    [:octicons-arrow-right-24: Collation](#1-collation)

-   :material-tag-multiple:{ .lg .middle } __Tokenize__

    ---

    Convert events into integer token sequences with a vocabulary and quantile
    bins learned on training data, then frozen.

    [:octicons-arrow-right-24: Tokenization](#2-tokenization)

-   :material-filter-variant:{ .lg .middle } __Winnow__

    ---

    Split held-out timelines into past / future at a threshold and flag outcome
    tokens for evaluation.

    [:octicons-arrow-right-24: Winnowing](#3-winnowing)

-   :material-book-open-variant:{ .lg .middle } __Recipes__

    ---

    Task-oriented guides for common workflows, from an end-to-end run on the
    public MIMIC-IV demo to tokenizer transfer and date-based inference.

    [:octicons-arrow-right-24: Recipes](recipes/index.md)

</div>

{%
   include-markdown "../README.md"
   heading-offset=0
   start="<!-- cards-anchor -->"
%}
