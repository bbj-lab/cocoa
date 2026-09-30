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
            is_flag=True,
        ),
    ] = False,
):
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
            "tokenizer.yaml file (reuses its frozen vocabulary and bins)",
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
            is_flag=True,
        ),
    ] = False,
):
    """
    Tokenize collated data into integer sequences.

    Reads collated parquet files and produces tokenized timelines with
    vocabulary and bin information.
    """
    with console.status("[bold green]Tokenizing data..."):
        t0 = time.perf_counter()
        if tokenizer_home is not None:
            print(f"Using pretrained tokenizer from [cyan]{tokenizer_home}[/cyan]...")
            tokenizer = Tokenizer(
                tokenization_cfg=tokenization_config,
                processed_data_home=processed_data_home,
            ).load(tokenizer_home)
        else:
            tokenizer = Tokenizer(
                tokenization_cfg=tokenization_config,
                processed_data_home=processed_data_home,
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
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Verbose logging for winnow; prints summary statistics",
            is_flag=True,
        ),
    ] = False,
):
    """
    Winnow held-out data for evaluation.

    Filters held-out timelines and assigns flags to disqualify certain subjects
    from evaluation based on the configured criteria.
    """
    with console.status("[bold green]Winnowing data..."):
        t0 = time.perf_counter()
        winnower = Winnower(
            winnowing_cfg=winnowing_config, processed_data_home=processed_data_home
        )
        winnower.save_all(verbose=verbose)
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Winnowing completed in {t1 - t0:.2f}s.")
    out_path = winnower.processed_data_home
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
        bool,
        typer.Option(
            "--verbose", "-v", help="Verbose logging for pipeline steps", is_flag=True
        ),
    ] = False,
):
    """
    Run the full pipeline: collate, tokenize, & winnow.
    """
    print("[bold]Running full pipeline[/bold]\n")
    t0 = time.perf_counter()
    collate(
        collation_config=collation_config,
        raw_data_home=raw_data_home,
        processed_data_home=processed_data_home,
        verbose=verbose,
    )
    tokenize(
        tokenization_config=tokenization_config,
        processed_data_home=processed_data_home,
        verbose=verbose,
    )
    winnow(
        winnowing_config=winnowing_config,
        processed_data_home=processed_data_home,
        verbose=verbose,
    )
    t1 = time.perf_counter()
    print(f"\n[bold green]Pipeline completed in {t1 - t0:.2f}s.[/bold green]")


@app.command()
def combine_datasets(
    input_data_dirs: list[str],
    output_data_dir: Annotated[
        str,
        typer.Option(
            "--output-data-dir", "-o", help="Output directory for combined data"
        ),
    ] = ...,
):
    """
    Combine multiple processed datasets into one.

    Merges parquet files and validates that tokenizer configurations match
    across all input directories.
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
        str, typer.Argument(help="Subject whose timeline to show", show_default=False)
    ],
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
            help="Save the timeline to this self-contained html file "
            "instead of serving it",
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
        bool,
        typer.Option("--open", help="Open the page in a web browser", is_flag=True),
    ] = False,
):
    """
    Visualize a subject's timeline as an interactive html page.

    Serves the page on localhost until interrupted, or saves it as a
    self-contained html file with --export-html.
    """
    with console.status("[bold green]Rendering timeline..."):
        t0 = time.perf_counter()
        visualizer = Visualizer(
            visualization_cfg=visualization_config,
            processed_data_home=processed_data_home,
        )
        try:
            if export_html is not None:
                out_path = visualizer.save(subject_id, export_html)
            else:
                page = visualizer.render(subject_id)
        except SubjectNotFoundError as e:
            print(f"[red]✗[/red] {escape(str(e))}")
            raise typer.Exit(code=1)
        t1 = time.perf_counter()
        print(
            f"\n[green]✓[/green] Rendered the timeline of subject "
            f"{escape(subject_id)} in {t1 - t0:.2f}s."
        )
        if unclaimed := visualizer.unclaimed_prefixes():
            n = len(unclaimed)
            print(
                f"  {n} code prefix{'es' if n > 1 else ''} without a configured lane: "
                f"{escape(', '.join(unclaimed))}. To name and color their lanes, "
                "list them under `lanes` in a config passed with -c."
            )
    if export_html is not None:
        print(f"  Output: [cyan]{out_path}[/cyan]")
        return
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


def main():
    app()


if __name__ == "__main__":
    # pipeline()
    main()
