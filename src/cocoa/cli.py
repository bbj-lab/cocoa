#!/usr/bin/env python3

"""
CLI for cocoa - configurable collation and tokenization
"""

import pathlib
import time
from typing import Annotated, Optional

import typer
from rich import print
from rich.console import Console
from rich.table import Table

from cocoa.collator import Collator
from cocoa.tokenizer import Tokenizer
from cocoa.winnower import Winnower

app = typer.Typer(name="cocoa", help="Configurable collation and tokenization")
console = Console()


@app.command()
def collate(
    main_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--main-config", "-m", help="Main configuration file (overrides default)"
        ),
    ] = None,
    collation_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--collation-config",
            "-c",
            help="Collation configuration file (overrides config)",
        ),
    ] = None,
    raw_data_home: Annotated[
        Optional[str],
        typer.Option(
            "--raw-data-home", "-r", help="Raw data directory (overrides config)"
        ),
    ] = None,
    processed_data_home: Annotated[
        Optional[str],
        typer.Option(
            "--processed-data-home",
            "-p",
            help="Processed data directory (overrides config)",
        ),
    ] = None,
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

    Reads configuration from config/main.yaml and produces a MEDS-like
    parquet file with collated events.
    """
    with console.status("[bold green]Collating data..."):
        t0 = time.perf_counter()
        collator = Collator(
            main_cfg=main_config,
            collation_cfg=collation_config,
            raw_data_home=raw_data_home,
            processed_data_home=processed_data_home,
        )
        collator.save_all(verbose=verbose)
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Collation completed in {t1 - t0:.2f}s.")
    out_path = collator.processed_data_home
    print(f"  Output: [cyan]{out_path}/meds.parquet[/cyan]")


@app.command()
def tokenize(
    main_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--main-config", "-m", help="Main configuration file (overrides default)"
        ),
    ] = None,
    tokenization_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--tokenization-config",
            "-c",
            help="Tokenization configuration file (overrides config)",
        ),
    ] = None,
    processed_data_home: Annotated[
        Optional[str],
        typer.Option(
            "--processed-data-home",
            "-p",
            help="Processed data directory (overrides config)",
        ),
    ] = None,
    tokenizer_home: Annotated[
        Optional[str],
        typer.Option(
            "--tokenizer-home",
            "-t",
            help="Use a pretrained tokenizer at this path (overrides config)",
        ),
    ] = None,
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
    Tokenize collated data into integer sequences.

    Reads collated parquet files and produces tokenized timelines with
    vocabulary and bin information.
    """
    with console.status("[bold green]Tokenizing data..."):
        t0 = time.perf_counter()
        if tokenizer_home is not None:
            print(f"Using pretrained tokenizer from [cyan]{tokenizer_home}[/cyan]...")
            tokenizer = Tokenizer().load(tokenizer_home)
            tokenizer.processed_data_home = str(processed_data_home)
        else:
            tokenizer = Tokenizer(
                main_cfg=main_config,
                tokenization_cfg=tokenization_config,
                processed_data_home=processed_data_home,
            )
        tokenizer.save_all(verbose=verbose)
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Tokenization completed in {t1 - t0:.2f}s.")
    out_path = tokenizer.processed_data_home
    print(f"  Output: [cyan]{out_path}/tokens_times.parquet[/cyan]")
    print(f"  Output: [cyan]{out_path}/subject_splits.parquet[/cyan]")
    print(f"  Output: [cyan]{out_path}/tokens_vocab.json[/cyan]")
    print(f"  Vocabulary size: [cyan]{len(tokenizer)}[/cyan] tokens")


@app.command()
def winnow(
    main_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--main-config", "-m", help="Main configuration file (overrides default)"
        ),
    ] = None,
    winnowing_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--winnowing-config",
            "-c",
            help="Winnowing configuration file (overrides config)",
        ),
    ] = None,
    processed_data_home: Annotated[
        Optional[str],
        typer.Option(
            "--processed-data-home",
            "-p",
            help="Processed data directory (overrides config)",
        ),
    ] = None,
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
            main_cfg=main_config,
            winnowing_cfg=winnowing_config,
            processed_data_home=processed_data_home,
        )
        winnower.save_all(verbose=verbose)
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Winnowing completed in {t1 - t0:.2f}s.")
    out_path = winnower.processed_data_home
    print(f"  Output: [cyan]{out_path}/held_out_for_inference.parquet[/cyan]")


@app.command()
def pipeline(
    main_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--main-config", "-m", help="Main configuration file (overrides default)"
        ),
    ] = None,
    collation_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--collation-config", help="Collation configuration file (overrides config)"
        ),
    ] = None,
    tokenization_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--tokenization-config",
            help="Tokenization configuration file (overrides config)",
        ),
    ] = None,
    winnowing_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--winnowing-config", help="Winnowing configuration file (overrides config)"
        ),
    ] = None,
    raw_data_home: Annotated[
        Optional[str],
        typer.Option(
            "--raw-data-home", "-r", help="Raw data directory (overrides config)"
        ),
    ] = None,
    processed_data_home: Annotated[
        Optional[str],
        typer.Option(
            "--processed-data-home",
            "-p",
            help="Processed data directory (overrides config)",
        ),
    ] = None,
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
        main_config=main_config,
        collation_config=collation_config,
        raw_data_home=raw_data_home,
        processed_data_home=processed_data_home,
        verbose=verbose,
    )
    tokenize(
        main_config=main_config,
        tokenization_config=tokenization_config,
        processed_data_home=processed_data_home,
        verbose=verbose,
    )
    winnow(
        main_config=main_config,
        winnowing_config=winnowing_config,
        processed_data_home=processed_data_home,
        verbose=verbose,
    )
    t1 = time.perf_counter()
    print(f"\n[bold green]Pipeline completed in {t1 - t0:.2f}s.[/bold green]")


@app.command()
def info():
    """
    Display configuration information.
    """
    from omegaconf import OmegaConf

    main_cfg = OmegaConf.load(pathlib.Path("./config/main.yaml").expanduser().resolve())

    table = Table(title="Cocoa Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Raw data Home", str(main_cfg.raw_data_home))
    table.add_row("Processed Data Home", str(main_cfg.processed_data_home))
    table.add_row("Collation Config", str(main_cfg.collation_config))
    table.add_row("Tokenization Config", str(main_cfg.tokenization_config))

    console.print(table)


def main():
    app()


if __name__ == "__main__":
    # pipeline()
    main()
