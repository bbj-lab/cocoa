# Winnower

The `Winnower` is the final stage of the pipeline. It prepares held-out
timelines for evaluation: it splits each timeline at a configurable cut-point
into a **past** portion (the context a model is given) and a **future** portion
(what the model must predict), and it attaches outcome labels. Its behavior is
driven by a winnowing config (`winnowing.yaml`).

The name is apt — the winnower also **filters out** subjects that cannot be
fairly evaluated, for example timelines that end before the outcome horizon is
reached.

## What it produces

Running the winnower (via [`save_all`][cocoa.winnower.Winnower.save_all]) writes
one file per configured split to the processed-data directory:

- `{split}_for_inference.parquet` — the winnowed timelines with past/future
  splits and outcome flags (defaults to `held_out`).

## How it works

[`prepare_winnowed_frame`][cocoa.winnower.Winnower.prepare_winnowed_frame] chains
these steps:

1. [`load_frame`][cocoa.winnower.Winnower.load_frame] — load the tokenized
   timelines for the requested split and compute elapsed times.
2. [`run_thresholding`][cocoa.winnower.Winnower.run_thresholding] — compute
   `last_valid`, the cut-point between past and future. This is derived either
   from an elapsed-time horizon or from the first occurrence of a token of
   interest. Timelines that never reach the cut-point are dropped.
3. [`add_outcome_flags`][cocoa.winnower.Winnower.add_outcome_flags] — for each
   configured outcome token, add `*_past` and `*_future` boolean labels (for
   example `DSCG//expired_past` and `DSCG//expired_future`), so you can tell
   whether an outcome truly falls within the prediction window.

!!! note "Depends on tokenizer artifacts"
    The winnower reads the `tokenizer.yaml` written by the [Tokenizer](tokenizer.md)
    to resolve outcome-token patterns against the learned vocabulary, so
    tokenization must run first.

::: cocoa.winnower
