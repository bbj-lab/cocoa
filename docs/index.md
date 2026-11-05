---
hide:
    - navigation
    - toc
---

<div class="hero" markdown>

--8<-- "README.md:intro"

</div>

<div class="grid cards" markdown>

- :material-table-merge-cells:{ .lg .middle } **Collate**

    ***

    Pull raw parquet/csv tables into one denormalized MEDS-like frame, plus train
    / tuning / held-out splits, chronological or random.

    [:octicons-arrow-right-24: Collation](guide/collation.md)

- :material-tag-multiple:{ .lg .middle } **Tokenize**

    ***

    Convert events into integer token sequences with a vocabulary and quantile
    bins learned on training data, then frozen.

    [:octicons-arrow-right-24: Tokenization](guide/tokenization.md)

- :material-filter-variant:{ .lg .middle } **Winnow**

    ***

    Split held-out timelines into past / future at a threshold and flag outcome
    tokens for evaluation.

    [:octicons-arrow-right-24: Winnowing](guide/winnowing.md)

- :material-chart-timeline-variant:{ .lg .middle } **Visualize**

    ***

    Draw one subject's tokenized timeline, or the learned vocabulary and its
    bins, as a self-contained interactive page or a pdf.

    [:octicons-arrow-right-24: Visualization](guide/visualization.md)

- :material-book-open-variant:{ .lg .middle } **Recipes**

    ***

    Task-oriented guides for common workflows, from an end-to-end run on the
    public MIMIC-IV demo to tokenizer transfer and date-based inference.

    [:octicons-arrow-right-24: Recipes](recipes/index.md)

- :material-console:{ .lg .middle } **Reference**

    ***

    Every `cocoa` command's options, and the Python classes behind each stage,
    from their docstrings.

    [:octicons-arrow-right-24: CLI reference](api/cli.md)

</div>

--8<-- "README.md:install"

--8<-- "README.md:references"
