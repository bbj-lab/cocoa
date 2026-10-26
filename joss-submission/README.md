## [JOSS](https://joss.theoj.org)

See the instructions [here](https://joss.readthedocs.io/en/latest/index.html).

This file can be compiled locally with docker:

```sh
docker run --rm --volume $PWD:/data --user $(id -u):$(id -g) --env JOURNAL=joss openjournals/inara
```

On GitHub, the [draft pdf](../.github/workflows/draft-pdf.yml) workflow compiles it
with the same image whenever a push or pull request touches this directory, and
attaches `paper.pdf` to the run as an artifact (or run it by hand from the Actions
tab).
