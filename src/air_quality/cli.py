from __future__ import annotations

import logging
import sys

import typer
from rich.console import Console
from rich.table import Table

from .config import AQIPipelineConfig
from .tasks import run_full_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
app = typer.Typer(add_completion=False)
console = Console()


def _strip_ipykernel_args(argv: list[str]) -> list[str]:
    out = [argv[0]]
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in ("-f", "--f"):
            i += 2
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
    end_date: str = "2024-03-31",
    latitude: float = 34.0522,
    longitude: float = -118.2437,
    location_name: str = "los_angeles_ca",
    distance_miles: int = 25,
    horizon: int = 24,
    overwrite: bool = False,
    use_residual: bool = True,
):
    cfg = AQIPipelineConfig(
        start_date=start_date,
        end_date=end_date,
        latitude=latitude,
        longitude=longitude,
        location_name=location_name,
        distance_miles=distance_miles,
        horizon=horizon,
        overwrite=overwrite,
        use_residual=use_residual,
    )

    results = run_full_pipeline(cfg)

    table = Table(title="AQI Pipeline Results")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")

    for k, v in results.items():
        table.add_row(str(k), str(v))

    console.print(table)


if __name__ == "__main__":
    sys.argv = _strip_ipykernel_args(sys.argv)
    app(standalone_mode=False)
