"""Build simple analysis features and panels from clean interim tables."""

from __future__ import annotations

from typing import Any

import pandas as pd

from nyiso_reliability.config import ProjectConfig

TIMEZONE = "America/New_York"


def _floor_hour(values: pd.Series) -> pd.Series:
    """Floor instants in UTC so repeated New York hours retain their offsets."""
    return values.dt.tz_convert("UTC").dt.floor("h").dt.tz_convert(TIMEZONE)


def _local_day_start(values: pd.Series) -> pd.Series:
    """Return unambiguous New York midnight for each local calendar date."""
    dates = pd.to_datetime(values.dt.date)
    return dates.dt.tz_localize(TIMEZONE, ambiguous="raise", nonexistent="raise")


def add_time_features(df: pd.DataFrame, column: str = "interval_start") -> pd.DataFrame:
    """Add calendar features from a timezone-aware timestamp column."""
    result = df.copy()
    timestamp = result[column]
    if not pd.api.types.is_datetime64_any_dtype(timestamp) or timestamp.dt.tz is None:
        raise ValueError(f"{column} must be timezone-aware")
    result["hour"] = timestamp.dt.hour
    result["weekday"] = timestamp.dt.weekday
    result["month"] = timestamp.dt.month
    result["weekend"] = result["weekday"].isin([5, 6])
    result["season"] = result["month"].map(
        {
            12: "winter",
            1: "winter",
            2: "winter",
            3: "spring",
            4: "spring",
            5: "spring",
            6: "summer",
            7: "summer",
            8: "summer",
            9: "fall",
            10: "fall",
            11: "fall",
        }
    )
    return result


def _select_prior_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """Select the latest forecast vintage strictly before each valid local date."""
    result = df.copy()
    result["valid_date"] = result["interval_start"].dt.date
    result = result[result["forecast_vintage_date"] < result["valid_date"]]
    result = result.sort_values("forecast_vintage_date")
    return result.drop_duplicates(["interval_start", "zone"], keep="last")


def build_load_features(
    actual: pd.DataFrame, forecast: pd.DataFrame
) -> pd.DataFrame:
    """Build hourly load, prior-vintage forecast error, and load ramp features."""
    actual_hourly = (
        actual.assign(interval_start=_floor_hour(actual["interval_start"]))
        .groupby(["interval_start", "zone"], as_index=False)["actual_load_mw"]
        .mean()
    )
    selected = _select_prior_forecast(forecast)[
        ["interval_start", "zone", "forecast_load_mw", "forecast_vintage_date"]
    ]
    result = actual_hourly.merge(selected, on=["interval_start", "zone"], how="left")
    result["load_forecast_error_mw"] = (
        result["actual_load_mw"] - result["forecast_load_mw"]
    )
    result = result.sort_values(["zone", "interval_start"])
    result["load_ramp"] = result.groupby("zone")["actual_load_mw"].diff()
    return result


def build_weather_features(weather: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the one-coordinate weather sample to local calendar days."""
    result = weather.copy()
    result["interval_start"] = _local_day_start(result["interval_start"])
    return (
        result.groupby("interval_start", as_index=False)
        .agg(
            daily_mean_temperature=("temperature", "mean"),
            daily_min_temperature=("temperature", "min"),
            daily_max_temperature=("temperature", "max"),
            precipitation_total=("precipitation", "sum"),
            max_wind_speed=("wind_speed", "max"),
        )
        .sort_values("interval_start")
    )


def build_price_features(da: pd.DataFrame, rt: pd.DataFrame) -> pd.DataFrame:
    """Align hourly DA and mean RT LBMP, then compute DART as RT minus DA."""
    da_hourly = da[["interval_start", "zone", "lbmp"]].rename(
        columns={"lbmp": "da_lbmp"}
    )
    rt_hourly = (
        rt.assign(interval_start=_floor_hour(rt["interval_start"]))
        .groupby(["interval_start", "zone"], as_index=False)["lbmp"]
        .mean()
        .rename(columns={"lbmp": "rt_lbmp"})
    )
    result = da_hourly.merge(rt_hourly, on=["interval_start", "zone"], how="left")
    result["dart"] = result["rt_lbmp"] - result["da_lbmp"]
    return result


def build_outage_count_target(df: pd.DataFrame) -> pd.DataFrame:
    """Build counts only from complete historical event-level records."""
    required = {"event_id", "start_time", "zone", "historical_complete"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Outage count target BLOCKED: missing fields {sorted(missing)}")
    if not df["historical_complete"].fillna(False).all():
        raise ValueError("Outage count target BLOCKED: source is latest-only/incomplete")
    if df[["event_id", "start_time", "zone"]].isna().any().any():
        raise ValueError("Outage count target BLOCKED: event identity/time/zone is missing")
    events = df.drop_duplicates("event_id").copy()
    events["interval_start"] = _local_day_start(events["start_time"])
    return (
        events.groupby(["interval_start", "zone"], as_index=False)
        .size()
        .rename(columns={"size": "new_outage_count"})
    )


def _window(df: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    """Restrict an interval table to the configured inclusive local-date window."""
    local_date = df["interval_start"].dt.date
    return df[(local_date >= config.start_date) & (local_date <= config.end_date)].copy()


def build_daily_analysis_panel(config: ProjectConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a zone-day covariate panel without inventing an outage target."""
    interim = config.interim_data_dir
    actual = _window(pd.read_parquet(interim / "actual_load.parquet"), config)
    forecast = pd.read_parquet(interim / "load_forecast.parquet")
    weather = _window(pd.read_parquet(interim / "weather_albany.parquet"), config)
    p15 = pd.read_parquet(interim / "p15_generation_outage_forecast.parquet")

    actual["interval_start"] = _local_day_start(actual["interval_start"])
    daily = actual.groupby(["interval_start", "zone"], as_index=False).agg(
        daily_mean_load=("actual_load_mw", "mean"),
        daily_max_load=("actual_load_mw", "max"),
    )
    selected_forecast = _select_prior_forecast(forecast)
    selected_forecast["interval_start"] = _local_day_start(
        selected_forecast["interval_start"]
    )
    forecast_daily = selected_forecast.groupby(
        ["interval_start", "zone"], as_index=False
    ).agg(forecast_load_mw=("forecast_load_mw", "mean"))
    daily = daily.merge(forecast_daily, on=["interval_start", "zone"], how="left")
    daily["load_forecast_error_mw"] = (
        daily["daily_mean_load"] - daily["forecast_load_mw"]
    )
    daily = daily.merge(build_weather_features(weather), on="interval_start", how="left")
    p15_daily = p15[["interval_start", "generation_outage_mw"]]
    daily = daily.merge(p15_daily, on="interval_start", how="left")
    daily = add_time_features(daily)

    target_status = "BLOCKED"
    target_definition = (
        "Unavailable: P-14B is latest-only and lacks zone; P-54 snapshots lack "
        "stable event identity and complete event timing."
    )
    output = config.processed_data_dir / "daily_analysis_panel.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output, index=False)
    summary = {
        "panel": "daily_analysis_panel",
        "rows": len(daily),
        "date_min": str(daily["interval_start"].min()),
        "date_max": str(daily["interval_start"].max()),
        "grain": "zone-local_day",
        "zones": int(daily["zone"].nunique()),
        "missingness": str(daily.isna().sum().to_dict()),
        "merge_match_rate": float(daily["forecast_load_mw"].notna().mean()),
        "target_available": "no",
        "target_status": target_status,
        "target_definition": target_definition,
    }
    return daily, summary


def build_hourly_price_panel(config: ProjectConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build an aligned hourly DA/RT/load panel for the later price extension."""
    interim = config.interim_data_dir
    da = _window(pd.read_parquet(interim / "da_lbmp.parquet"), config)
    rt = _window(pd.read_parquet(interim / "rt_lbmp.parquet"), config)
    actual = _window(pd.read_parquet(interim / "actual_load.parquet"), config)
    forecast = pd.read_parquet(interim / "load_forecast.parquet")
    load = build_load_features(actual, forecast)
    load_zones = set(actual["zone"].unique())
    prices = build_price_features(
        da[da["zone"].isin(load_zones)], rt[rt["zone"].isin(load_zones)]
    )
    panel = prices.merge(load, on=["interval_start", "zone"], how="left")
    panel = add_time_features(panel)
    output = config.processed_data_dir / "hourly_price_panel.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output, index=False)
    summary = {
        "panel": "hourly_price_panel",
        "rows": len(panel),
        "date_min": str(panel["interval_start"].min()),
        "date_max": str(panel["interval_start"].max()),
        "grain": "zone-hour",
        "zones": int(panel["zone"].nunique()),
        "missingness": str(panel.isna().sum().to_dict()),
        "merge_match_rate": float(panel["rt_lbmp"].notna().mean()),
        "target_available": "not_applicable",
        "target_status": "PRICE_EXTENSION_ONLY",
        "target_definition": "DART = mean hourly RT LBMP - DA LBMP; not causal.",
    }
    return panel, summary


def build_analysis_panels(config: ProjectConfig) -> pd.DataFrame:
    """Build both valid panels and save a compact panel summary."""
    _, daily_summary = build_daily_analysis_panel(config)
    _, hourly_summary = build_hourly_price_panel(config)
    summary = pd.DataFrame([daily_summary, hourly_summary])
    summary.to_csv(config.output_dir / "tables/panel_summary.csv", index=False)
    return summary
