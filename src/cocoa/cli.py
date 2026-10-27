#!/usr/bin/env python3

"""
CLI for cocoa - configurable collation and tokenization
"""

import errno
import pathlib
import time
import webbrowser
from importlib.metadata import version
from typing import Annotated, Optional

import typer
from rich import print
from rich.console import Console
from rich.markup import escape

from cocoa.collator import Collator
from cocoa.demo import write_demo
from cocoa.tokenizer import Tokenizer
from cocoa.util import combine_processed_data
from cocoa.visualizer import SubjectNotFoundError, Visualizer, make_server
from cocoa.winnower import Winnower

__version__ = version("cocoa-tokenizer")

app = typer.Typer(
    name="cocoa",
    help=f"Configurable collation and tokenization (v{__version__})",
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console()

# trailing a stage's command: edits to its config, applied by `apply_overrides`
Overrides = Annotated[
    Optional[list[str]],
    typer.Argument(
        help="Config overrides: key=value sets a key, adding it if need be, and "
        "~key deletes one",
        show_default=False,
    ),
]

# `pipeline` sends each of its overrides to the stage whose config it names first
STAGES = {"collation": Collator, "tokenization": Tokenizer, "winnowing": Winnower}


def route_overrides(overrides: list[str] | None) -> dict[str, list[str]]:
    """`pipeline`'s overrides by stage, each with its stage's name dropped"""
    routed: dict[str, list[str]] = {stage: [] for stage in STAGES}
    for override in overrides or ():
        op = next((p for p in ("++", "+", "~") if override.startswith(p)), "")
        stage, dot, rest = override.removeprefix(op).partition(".")
        if stage not in routed or not dot:
            raise typer.BadParameter(
                f"{override!r} doesn't say which stage it's for; start its key "
                "with `collation.`, `tokenization.`, or `winnowing.`"
            )
        routed[stage].append(op + rest)
    return routed


def note_unclaimed(visualizer: Visualizer) -> None:
    """name the vocabulary's code prefixes that no configured lane claims"""
    if unclaimed := visualizer.unclaimed_prefixes():
        n = len(unclaimed)
        print(
            f"  {n} code prefix{'es' if n > 1 else ''} without a configured lane: "
            f"{escape(', '.join(unclaimed))}. To name and color their lanes, "
            "list them under `lanes` in a config passed with -c."
        )


def serve(page: str, host: str, port: int, open_browser: bool) -> None:
    """
    serve `page` on `host`, at `port` or at a free one if that's in use, until
    interrupted, opening it in a web browser if asked
    """
    try:
        server = make_server(page, host=host, port=port)
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        print(f"  Port {port} is in use; serving on a free one instead.")
        server = make_server(page, host=host, port=0)
    url = f"http://{host}:{server.server_port}/"
    print(f"  Serving: [cyan]{url}[/cyan] (press Ctrl-C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    print("  Stopped serving.")


@app.command()
def collate(
    collation_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--collation-config",
            "-c",
            help="Collation configuration file (overrides default)",
            show_default=False,
        ),
    ] = None,
    raw_data_home: Annotated[
        str, typer.Option("--raw-data-home", "-r", help="Raw data directory")
    ] = ...,
    processed_data_home: Annotated[
        str,
        typer.Option("--processed-data-home", "-p", help="Processed data directory"),
    ] = ...,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Verbose logging for collate; this may cause "
            "memory issues with large datasets",
        ),
    ] = False,
    overrides: Overrides = None,
) -> None:
    """
    Collate raw data into a denormalized format.

    Reads collation configuration and produces a MEDS-like parquet file
    with collated events.
    """
    with console.status("[bold green]Collating data..."):
        t0 = time.perf_counter()
        collator = Collator(
            collation_cfg=collation_config,
            raw_data_home=raw_data_home,
            processed_data_home=processed_data_home,
            overrides=overrides,
        )
        collator.save_all(verbose=verbose)
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Collation completed in {t1 - t0:.2f}s.")
    out_path = collator.processed_data_home
    print(f"  Output: [cyan]{out_path}/meds.parquet[/cyan]")
    print(f"  Output: [cyan]{out_path}/subject_splits.parquet[/cyan]")


@app.command()
def tokenize(
    tokenization_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--tokenization-config",
            "-c",
            help="Tokenization configuration file (overrides default)",
            show_default=False,
        ),
    ] = None,
    processed_data_home: Annotated[
        str,
        typer.Option("--processed-data-home", "-p", help="Processed data directory"),
    ] = ...,
    tokenizer_home: Annotated[
        Optional[str],
        typer.Option(
            "--tokenizer-home",
            "-t",
            help="Load a previously learned tokenizer from this "
            "tokenizer.yaml file (reuses its frozen config, vocabulary, and bins)",
            show_default=False,
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Verbose logging for tokenize; this may cause "
            "memory issues with large datasets",
        ),
    ] = False,
    overrides: Overrides = None,
) -> None:
    """
    Tokenize collated data into integer sequences.

    Reads collated parquet files and produces tokenized timelines with
    vocabulary and bin information.
    """
    with console.status("[bold green]Tokenizing data..."):
        t0 = time.perf_counter()
        if tokenizer_home is not None:
            print(f"Using pretrained tokenizer from [cyan]{tokenizer_home}[/cyan]...")
            if tokenization_config is not None or overrides:
                print(
                    "  Its saved config is used as written, "
                    "so -c and overrides don't apply."
                )
            tokenizer = Tokenizer(processed_data_home=processed_data_home).load(
                tokenizer_home
            )
        else:
            tokenizer = Tokenizer(
                tokenization_cfg=tokenization_config,
                processed_data_home=processed_data_home,
                overrides=overrides,
            )
        tokenizer.save_all(verbose=verbose)
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Tokenization completed in {t1 - t0:.2f}s.")
    out_path = tokenizer.processed_data_home
    print(f"  Output: [cyan]{out_path}/tokens_times.parquet[/cyan]")
    print(f"  Output: [cyan]{out_path}/tokenizer.yaml[/cyan]")
    print(f"  Vocabulary size: [cyan]{len(tokenizer)}[/cyan] tokens")


@app.command()
def winnow(
    winnowing_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--winnowing-config",
            "-c",
            help="Winnowing configuration file (overrides default)",
            show_default=False,
        ),
    ] = None,
    processed_data_home: Annotated[
        str,
        typer.Option("--processed-data-home", "-p", help="Processed data directory"),
    ] = ...,
    output_home: Annotated[
        Optional[str],
        typer.Option(
            "--output-home",
            "-o",
            help="Directory to write the winnowed files to, along with copies of "
            "the processed data directory's files (defaults to the processed data "
            "directory)",
            show_default=False,
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Verbose logging for winnow; prints summary statistics",
        ),
    ] = False,
    overrides: Overrides = None,
) -> None:
    """
    Winnow timelines for evaluation.

    Splits each timeline of the configured splits into a past and a future
    at a threshold, drops those that never reach it, and flags which outcome
    tokens fall on each side.
    """
    with console.status("[bold green]Winnowing data..."):
        t0 = time.perf_counter()
        winnower = Winnower(
            winnowing_cfg=winnowing_config,
            processed_data_home=processed_data_home,
            output_home=output_home,
            overrides=overrides,
        )
        winnower.save_all(verbose=verbose)
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Winnowing completed in {t1 - t0:.2f}s.")
    out_path = winnower.output_home
    for s in winnower.cfg.get("splits", ["held_out"]):
        print(f"  Output: [cyan]{out_path}/{s}_for_inference.parquet[/cyan]")


@app.command()
def pipeline(
    collation_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--collation-config",
            help="Collation configuration file (overrides default)",
            show_default=False,
        ),
    ] = None,
    tokenization_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--tokenization-config",
            help="Tokenization configuration file (overrides default)",
            show_default=False,
        ),
    ] = None,
    winnowing_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--winnowing-config",
            help="Winnowing configuration file (overrides default)",
            show_default=False,
        ),
    ] = None,
    raw_data_home: Annotated[
        str, typer.Option("--raw-data-home", "-r", help="Raw data directory")
    ] = ...,
    processed_data_home: Annotated[
        str,
        typer.Option("--processed-data-home", "-p", help="Processed data directory"),
    ] = ...,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose logging for pipeline steps")
    ] = False,
    overrides: Annotated[
        Optional[list[str]],
        typer.Argument(
            help="Config overrides, each key starting with the stage it's for: "
            "e.g. tokenization.n_bins=5 or ~winnowing.threshold.duration_s",
            show_default=False,
        ),
    ] = None,
) -> None:
    """
    Run the full pipeline: collate, tokenize, & winnow.
    """
    routed = route_overrides(overrides)
    # check every stage's overrides now, not after the stages before it have run
    for (stage, cls), cfg_file in zip(
        STAGES.items(), (collation_config, tokenization_config, winnowing_config)
    ):
        cls.load_cfg(cfg_file, routed[stage])
    print("[bold]Running full pipeline[/bold]\n")
    t0 = time.perf_counter()
    collate(
        collation_config=collation_config,
        raw_data_home=raw_data_home,
        processed_data_home=processed_data_home,
        verbose=verbose,
        overrides=routed["collation"],
    )
    tokenize(
        tokenization_config=tokenization_config,
        processed_data_home=processed_data_home,
        verbose=verbose,
        overrides=routed["tokenization"],
    )
    winnow(
        winnowing_config=winnowing_config,
        processed_data_home=processed_data_home,
        verbose=verbose,
        overrides=routed["winnowing"],
    )
    t1 = time.perf_counter()
    print(f"\n[bold green]Pipeline completed in {t1 - t0:.2f}s.[/bold green]")


@app.command()
def combine_datasets(
    input_data_dirs: Annotated[
        list[str],
        typer.Argument(
            help="Processed data directories to combine", show_default=False
        ),
    ],
    output_data_dir: Annotated[
        str,
        typer.Option(
            "--output-data-dir", "-o", help="Output directory for combined data"
        ),
    ] = ...,
) -> None:
    """
    Combine multiple processed datasets into one.

    Merges the parquet files of the input directories and keeps the first
    directory's tokenizer.yaml, warning if another's differs.
    """
    with console.status("[bold green]Combining datasets..."):
        t0 = time.perf_counter()
        output = combine_processed_data(
            processed_data_homes=input_data_dirs, processed_data_home=output_data_dir
        )
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Combine completed in {t1 - t0:.2f}s.")
        print(f"  Output at: [cyan]{output}[/cyan]")


@app.command()
def visualize(
    subject_id: Annotated[
        Optional[str],
        typer.Argument(
            help="Subject whose timeline to draw; without one, draws the vocabulary",
            show_default=False,
        ),
    ] = None,
    processed_data_home: Annotated[
        str,
        typer.Option("--processed-data-home", "-p", help="Processed data directory"),
    ] = ...,
    visualization_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--visualization-config",
            "-c",
            help="Visualization configuration file (overrides default)",
            show_default=False,
        ),
    ] = None,
    export_html: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--export-html",
            "-e",
            help="Save the page to this self-contained html file instead of serving it",
            show_default=False,
        ),
    ] = None,
    export_pdf: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--export-pdf",
            help="Save a static pdf of the subject's timeline to this file instead "
            "of serving it; with --export-html, save both",
            show_default=False,
        ),
    ] = None,
    host: Annotated[
        str, typer.Option("--host", help="Address to serve the page on")
    ] = "127.0.0.1",
    port: Annotated[
        int, typer.Option("--port", help="Port to serve the page on; 0 picks any")
    ] = 8765,
    open_browser: Annotated[
        bool, typer.Option("--open", help="Open the page in a web browser")
    ] = False,
    show_winnowing: Annotated[
        bool,
        typer.Option(
            "--show-winnowing",
            help="Show the winnowing split of the subject's timeline into past and "
            "future, and its outcome flags",
        ),
    ] = False,
    # after the subject id, which is read first, so the vocabulary takes none
    overrides: Annotated[
        Optional[list[str]],
        typer.Argument(
            help="Config overrides, after a subject id: key=value sets a key, "
            "adding it if need be, and ~key deletes one",
            show_default=False,
        ),
    ] = None,
) -> None:
    """
    Visualize a subject's timeline, or the vocabulary, as an interactive html page.

    Given a subject id, draws that subject's timeline; without one, draws
    the vocabulary: each binned code's training values against its bins,
    every word with its training count, and the timelines by split, as
    tokenize --verbose summarizes them. Serves the page on localhost until
    interrupted, or saves it as a self-contained html file with
    --export-html, and a timeline as a static pdf with --export-pdf.
    """
    if subject_id is None:
        for flag, given in (
            ("--export-pdf", export_pdf is not None),
            ("--show-winnowing", show_winnowing),
        ):
            if given:
                raise typer.BadParameter(
                    "it's for a subject's timeline, so give a subject id",
                    param_hint=f"'{flag}'",
                )
    drawing = "timeline" if subject_id is not None else "the vocabulary"
    with console.status(f"[bold green]Rendering {drawing}..."):
        t0 = time.perf_counter()
        visualizer = Visualizer(
            visualization_cfg=visualization_config,
            processed_data_home=processed_data_home,
            # without the flag, the config decides
            show_winnowing=True if show_winnowing else None,
            overrides=overrides,
        )
        out_paths = []
        if subject_id is None:
            if export_html is not None:
                out_paths.append(visualizer.save_vocabulary(export_html))
            else:
                page = visualizer.render_vocabulary()
            drawn = f"the vocabulary of {len(visualizer.decoder):,} words"
        else:
            try:
                if export_html is not None:
                    out_paths.append(visualizer.save(subject_id, export_html))
                if export_pdf is not None:
                    out_paths.append(visualizer.save_pdf(subject_id, export_pdf))
                if not out_paths:
                    page = visualizer.render(subject_id)
            except SubjectNotFoundError as e:
                print(f"[red]✗[/red] {escape(str(e))}")
                raise typer.Exit(code=1)
            drawn = f"the timeline of subject {escape(subject_id)}"
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Rendered {drawn} in {t1 - t0:.2f}s.")
        note_unclaimed(visualizer)
    if out_paths:
        for out_path in out_paths:
            print(f"  Output: [cyan]{out_path}[/cyan]")
        return
    serve(page, host=host, port=port, open_browser=open_browser)


@app.command()
def get_demo(
    raw_data_home: Annotated[
        str,
        typer.Option("--raw-data-home", "-r", help="Raw data directory to write to"),
    ] = ...,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", "-v", help="Show what clifpy prints as it derives tables"
        ),
    ] = False,
) -> None:
    """
    Write the MIMIC-IV demo's CLIF tables, from clifpy.

    Copies the tables of the demo that clifpy ships to a raw data
    directory, and derives from them, with clifpy, the four more that the
    shipped collation config reads. Needs clifpy, which cocoa's demo
    extra installs.
    """
    with console.status("[bold green]Writing the demo..."):
        t0 = time.perf_counter()
        try:
            written = write_demo(raw_data_home, verbose=verbose)
        except ModuleNotFoundError as e:
            if e.name != "clifpy":
                raise
            print(f"[red]✗[/red] {escape(str(e))}")
            raise typer.Exit(code=1)
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Demo written in {t1 - t0:.2f}s.")
    print(
        f"  Output: [cyan]{written[0].parent}[/cyan] ({len(written)} tables, "
        f"from clifpy {version('clifpy')})"
    )


def main() -> None:
    """entry point for the `cocoa` command"""
    app()


if __name__ == "__main__":
    # pipeline()
    main()
