"""Command-line entry points for the minimal research workflow."""

from pathlib import Path
from typing import Annotated

import typer
import yaml
import pandas as pd
from rich.console import Console

from nyiso_reliability.config import load_config
from nyiso_reliability.data_clean import clean_all_dev_data
from nyiso_reliability.data_download import download_dev_sample, download_weather
from nyiso_reliability.features import build_analysis_panels

app = typer.Typer(no_args_is_help=True)
console = Console()
EXCLUDED = {".git", ".pytest_cache", "__pycache__", ".venv"}


@app.command("show-config")
def show_config(
    config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/dev.yaml"),
) -> None:
    """Print the validated project configuration."""
    settings = load_config(config)
    console.print(yaml.safe_dump(settings.to_dict(), sort_keys=False))


@app.command("show-tree")
def show_tree() -> None:
    """Print the repository tree without caches or real data files."""
    root = Path(__file__).resolve().parents[2]
    console.print(root.name)
    for line in _tree_lines(root):
        console.print(line)


@app.command("download-dev-data")
def download_dev_data(
    config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/dev.yaml"),
) -> None:
    """Download monthly archives covering the configured date range."""
    settings = load_config(config)
    summary = download_dev_sample(settings)
    for record in summary.to_dict(orient="records"):
        console.rule(f"{record['dataset']} [{record['status']}]")
        for key in (
            "source",
            "retrieval_time",
            "rows",
            "columns",
            "date_min",
            "date_max",
            "unique_zones",
            "missingness",
            "duplicates",
            "saved_path",
            "blocker",
            "notes",
        ):
            console.print(f"{key}: {record[key]}")
    summary_path = settings.output_dir / "tables/data_download_summary.csv"
    console.print(f"Summary: {summary_path}")


@app.command("clean-data")
def clean_data(
    config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/dev.yaml"),
) -> None:
    """Clean all downloaded development sources into interim Parquet tables."""
    summary = clean_all_dev_data(load_config(config))
    console.print(summary.to_string(index=False))


@app.command("download-weather")
def download_weather_command(
    config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/dev.yaml"),
) -> None:
    """Download the configured Albany weather window with UTC timestamps."""
    settings = load_config(config)
    result = download_weather(settings)
    manifest_path = settings.output_dir / "tables/data_download_summary.csv"
    if manifest_path.is_file() and result["status"] != "BLOCKED":
        manifest = pd.read_csv(manifest_path)
        manifest = manifest[manifest["dataset"] != result["dataset"]]
        pd.concat([manifest, pd.DataFrame([result])], ignore_index=True).to_csv(
            manifest_path, index=False
        )
    for key, value in result.items():
        console.print(f"{key}: {value}")


@app.command("build-panels")
def build_panels(
    config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/dev.yaml"),
) -> None:
    """Build daily covariate and hourly price panels from interim data."""
    summary = build_analysis_panels(load_config(config))
    console.print(summary.to_string(index=False))


@app.command("run-eda")
def run_eda_command(
    config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/dev.yaml"),
) -> None:
    """Run EDA or print a truthful target-feasibility blocker."""
    from nyiso_reliability.eda import run_eda

    summary = run_eda(load_config(config))
    console.print("OUTAGE TARGET SUMMARY")
    console.print("---------------------")
    for key, value in summary.items():
        console.print(f"{key}: {value}")


@app.command("run-model")
def run_model(
    config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/dev.yaml"),
) -> None:
    """Run the gated Poisson/NB comparison or print its blocker."""
    from nyiso_reliability.nb_model import run_initial_models

    summary = run_initial_models(load_config(config))
    console.print("INITIAL COUNT MODEL COMPARISON")
    console.print("------------------------------")
    for key, value in summary.items():
        console.print(f"{key}: {value}")


@app.command("build-report")
def build_report_command(
    config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/dev.yaml"),
) -> None:
    """Build the compact HTML report from saved pipeline outputs."""
    from nyiso_reliability.report import build_report

    summary = build_report(load_config(config))
    for key, value in summary.items():
        console.print(f"{key}: {value}")


def _tree_lines(directory: Path, prefix: str = "") -> list[str]:
    """Build compact tree output recursively."""
    entries = [
        item
        for item in sorted(directory.iterdir(), key=lambda value: value.name)
        if item.name not in EXCLUDED
        and not item.name.endswith(".egg-info")
        and _visible(item)
    ]
    lines: list[str] = []
    for index, item in enumerate(entries):
        last = index == len(entries) - 1
        lines.append(f"{prefix}{'└── ' if last else '├── '}{item.name}")
        if item.is_dir():
            child_prefix = prefix + ("    " if last else "│   ")
            lines.extend(_tree_lines(item, child_prefix))
    return lines


def _visible(path: Path) -> bool:
    """Inside data/output directories, display only folders and placeholders."""
    in_artifact_dir = "data" in path.parts or "outputs" in path.parts
    return not in_artifact_dir or path.is_dir() or path.name == ".gitkeep"


if __name__ == "__main__":
    app()
