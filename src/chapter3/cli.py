# file: src/chapter3/cli.py
from __future__ import annotations

import logging
import sys

import typer
from rich.console import Console
from rich.table import Table

from src.chapter3.config import PipelineConfig
from src.chapter3.tasks import run_full_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
app = typer.Typer(add_completion=False)
console = Console()


def _strip_ipykernel_args(argv: list[str]) -> list[str]:
    """Jupyter/ipykernel injects `-f <connection_file>` into sys.argv."""
    out = [argv[0]]
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in ("-f", "--f"):
            i += 2  # skip flag + value
            continue
        if a.startswith("--f="):
            i += 1
            continue
        out.append(a)
        i += 1
    return out


@app.command()
def run(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    horizon: int = 24,
    respondent: str = "US48",
    fueltype: str = "NG",
    overwrite: bool = False,
):
    cfg = PipelineConfig(
        start_date=start_date,
        end_date=end_date,
        horizon=horizon,
        respondent=respondent,
        fueltype=fueltype,
        overwrite=overwrite,
    )

    results = run_full_pipeline(cfg)

    table = Table(title="Pipeline Results")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")

    for k, v in results.items():
        table.add_row(str(k), str(v))

    console.print(table)


if __name__ == "__main__":
    sys.argv = _strip_ipykernel_args(sys.argv)
    app(standalone_mode=False)  # <-- prevents SystemExit in Jupyter
