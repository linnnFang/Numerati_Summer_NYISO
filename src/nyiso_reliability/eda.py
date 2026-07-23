"""Small exploratory summaries with a safe blocked-target branch."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

_CACHE = Path(tempfile.gettempdir()) / "nyiso-reliability-matplotlib"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nyiso_reliability.config import ProjectConfig  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def summarize_dataset(df: pd.DataFrame, name: str) -> dict[str, Any]:
    """Return basic observed size, coverage, and grain information."""
    time_column = "interval_start" if "interval_start" in df else None
    return {
        "dataset": name,
        "rows": len(df),
        "columns": len(df.columns),
        "date_min": str(df[time_column].min()) if time_column else None,
        "date_max": str(df[time_column].max()) if time_column else None,
        "zones": int(df["zone"].nunique()) if "zone" in df else None,
        "duplicate_rows": int(df.duplicated().sum()),
    }


def summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing counts and shares for every column."""
    return pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().to_numpy(),
            "missing_share": df.isna().mean().to_numpy(),
        }
    )


def summarize_outage_target(df: pd.DataFrame, target: str) -> dict[str, Any]:
    """Validate and summarize a genuine non-negative integer count target."""
    if target not in df:
        raise ValueError(f"Target not available: {target}")
    values = df[target].dropna()
    if (values < 0).any() or not np.allclose(values, np.round(values)):
        raise ValueError(f"{target} is not a non-negative integer count")
    return {
        "target": target,
        "observations": len(values),
        "mean": float(values.mean()),
        "variance": float(values.var(ddof=1)),
        "standard_deviation": float(values.std(ddof=1)),
        "minimum": int(values.min()),
        "maximum": int(values.max()),
        "zero_share": float((values == 0).mean()),
        "quantile_25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "quantile_75": float(values.quantile(0.75)),
    }


def compute_overdispersion_stats(df: pd.DataFrame, target: str) -> dict[str, Any]:
    """Compute descriptive variance/mean evidence without selecting a model."""
    summary = summarize_outage_target(df, target)
    mean = summary["mean"]
    summary["variance_to_mean"] = summary["variance"] / mean if mean else np.nan
    summary["overdispersion_evidence"] = (
        "yes" if mean and summary["variance_to_mean"] > 1 else "no_or_undefined"
    )
    return summary


def _save_figure(path: Path) -> None:
    """Apply common layout, save, and close the active figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_outage_histogram(df: pd.DataFrame, target: str, path: Path) -> None:
    """Plot a histogram for a validated count target."""
    summarize_outage_target(df, target)
    df[target].plot.hist(bins=range(int(df[target].max()) + 2))
    plt.xlabel(target)
    plt.title("Outage count distribution")
    _save_figure(path)


def plot_outage_timeseries(df: pd.DataFrame, target: str, path: Path) -> None:
    """Plot aggregate outage counts through time."""
    summarize_outage_target(df, target)
    df.groupby("interval_start")[target].sum().plot()
    plt.ylabel(target)
    plt.title("Outage count through time")
    _save_figure(path)


def plot_mean_variance_by_zone(df: pd.DataFrame, target: str, path: Path) -> None:
    """Plot zone-level count means against variances."""
    summarize_outage_target(df, target)
    grouped = df.groupby("zone")[target].agg(["mean", "var"])
    plt.scatter(grouped["mean"], grouped["var"])
    plt.xlabel("Mean")
    plt.ylabel("Variance")
    plt.title("Outage mean vs variance by zone")
    _save_figure(path)


def plot_zero_share_by_zone(df: pd.DataFrame, target: str, path: Path) -> None:
    """Plot the fraction of zero-count observations by zone."""
    summarize_outage_target(df, target)
    shares = df.groupby("zone")[target].apply(lambda values: (values == 0).mean())
    shares.plot.bar()
    plt.ylabel("Zero share")
    _save_figure(path)


def plot_outage_vs_load(df: pd.DataFrame, target: str, path: Path) -> None:
    """Plot count against mean load when a valid target exists."""
    summarize_outage_target(df, target)
    plt.scatter(df["daily_mean_load"], df[target], alpha=0.5)
    plt.xlabel("Daily mean load (MW)")
    plt.ylabel(target)
    _save_figure(path)


def plot_outage_vs_weather(df: pd.DataFrame, target: str, path: Path) -> None:
    """Plot count against representative temperature when target exists."""
    summarize_outage_target(df, target)
    plt.scatter(df["daily_mean_temperature"], df[target], alpha=0.5)
    plt.xlabel("Albany daily mean temperature (°C)")
    plt.ylabel(target)
    _save_figure(path)


def plot_load_timeseries(df: pd.DataFrame, path: Path) -> None:
    """Plot total zonal daily mean load for the development window."""
    values = df.groupby("interval_start")["daily_mean_load"].sum()
    values.plot(marker="o")
    plt.ylabel("Sum of zonal daily mean load (MW)")
    plt.title("NYISO development-sample load")
    _save_figure(path)


def plot_p15_timeseries(df: pd.DataFrame, path: Path) -> None:
    """Plot P-15 forecast MW while labeling it explicitly as a snapshot."""
    df.set_index("interval_start")["generation_outage_mw"].plot(marker="o")
    plt.ylabel("Forecasted generation outage (MW)")
    plt.title("P-15 current forecast snapshot — not event count")
    _save_figure(path)


def run_eda(config: ProjectConfig) -> dict[str, Any]:
    """Run valid EDA and save a blocker instead of fabricating outage results."""
    daily = pd.read_parquet(config.processed_data_dir / "daily_analysis_panel.parquet")
    hourly = pd.read_parquet(config.processed_data_dir / "hourly_price_panel.parquet")
    p15 = pd.read_parquet(
        config.interim_data_dir / "p15_generation_outage_forecast.parquet"
    )
    tables = config.output_dir / "tables"
    figures = config.output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    dataset_summary = pd.DataFrame(
        [
            summarize_dataset(daily, "daily_analysis_panel"),
            summarize_dataset(hourly, "hourly_price_panel"),
            summarize_dataset(p15, "p15_generation_outage_forecast"),
        ]
    )
    dataset_summary.to_csv(tables / "eda_dataset_summary.csv", index=False)
    summarize_missingness(daily).to_csv(
        tables / "daily_panel_missingness.csv", index=False
    )
    summarize_missingness(hourly).to_csv(
        tables / "hourly_panel_missingness.csv", index=False
    )

    plot_load_timeseries(daily, figures / "load_timeseries.png")
    plot_p15_timeseries(p15, figures / "p15_generation_outage_mw.png")

    target = "new_outage_count"
    if target in daily:
        target_summary = compute_overdispersion_stats(daily, target)
        target_summary["status"] = "READY"
        target_summary["target_validity"] = "verified integer count"
    else:
        target_summary = {
            "status": "BLOCKED",
            "target": target,
            "observations": None,
            "mean": None,
            "variance": None,
            "variance_to_mean": None,
            "zero_share": None,
            "target_validity": "not available",
            "main_limitations": (
                "P-14B is latest-only and lacks zone; P-54 snapshots lack stable "
                "event identity and complete event timing. P-15 is MW, not count."
            ),
        }
    pd.DataFrame([target_summary]).to_csv(
        tables / "outage_target_feasibility.csv", index=False
    )
    return target_summary
