## Date-based generative inference

To prepare data for inference at each timeline's first midnight (here on the
[MIMIC-IV demo](mimic-iv-demo.md), from its working directory),

1. In your tokenization config (e.g. `tokenization.yaml`, a copy of the shipped
   one), configure midnight clock token generation:

    ```yaml
    insert_clocks: !!bool true
    …
    clocks:
      - !!str 00 # produces token CLCK//00
    ```

    Clock hours are read in the collation config's `default_timezone`, so
    `CLCK//00` marks local midnight in that zone (`UTC` if unset). The demo's
    CLIF tables are in UTC, but MIMIC-IV's times are Boston's, so step 3 collates
    them in `America/New_York`. (For MIMIC-IV's shifted years, polars keeps that
    zone on standard time all year, as the demo's conversion to UTC did, so
    `CLCK//00` falls at midnight as MIMIC-IV records it.)

2. In your winnowing config (e.g. `winnowing.yaml`, a copy of the shipped one),
   threshold at the first occurrence of `CLCK//00`, in place of its `duration_s`:

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
      --raw-data-home data-raw/demo \
      --processed-data-home processed/demo-midnight \
      collation.default_timezone=America/New_York
    ```

    Only timelines that reach a midnight are kept: on the demo, 202 of the 208
    training hospitalizations, 38 of the 39 tuning, and 62 of the 63 held out.
    The held-out one left out, 22380825, is a five-hour stay. Hospitalization
    27708593, admitted at 07:15, is cut at midnight that night, before its
    transfer to the ICU and its first tachycardia. Some are cut before they are
    admitted: 52 of the 302 kept open with an arrival in the emergency department
    the day before their admission, so their pasts end at the midnight before it,
    without the age, sex, and other tokens timed at admission.
