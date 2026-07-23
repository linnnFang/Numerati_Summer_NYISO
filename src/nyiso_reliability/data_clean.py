"""Clean observed development files into small, timezone-aware tables."""

from __future__ import annotations

import warnings
import re
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from nyiso_reliability.config import ProjectConfig

TIMEZONE = "America/New_York"


def _localize(
    values: pd.Series,
    *,
    timezone_labels: pd.Series | None = None,
    group_keys: list[pd.Series] | None = None,
) -> pd.Series:
    """Parse local times using source evidence and reject unresolved DST clocks."""
    parsed = pd.to_datetime(values, errors="raise")
    try:
        timezone = parsed.dt.tz
    except AttributeError:
        parsed = pd.to_datetime(values, errors="raise", utc=True)
        timezone = parsed.dt.tz
    if timezone is None:
        if timezone_labels is not None:
            labels = timezone_labels.astype("string").str.strip().str.upper()
            unknown = sorted(set(labels.dropna()).difference({"EDT", "EST"}))
            if unknown or labels.isna().any():
                raise ValueError(f"Unknown NYISO Time Zone labels: {unknown}")
            is_daylight = labels.map({"EDT": True, "EST": False}).to_numpy()
            return parsed.dt.tz_localize(
                TIMEZONE, ambiguous=is_daylight, nonexistent="raise"
            )
        if group_keys:
            return _localize_in_source_order(parsed, group_keys)
        return parsed.dt.tz_localize(TIMEZONE, ambiguous="raise", nonexistent="raise")
    return parsed.dt.tz_convert(TIMEZONE)


def _localize_in_source_order(
    parsed: pd.Series, group_keys: list[pd.Series]
) -> pd.Series:
    """Infer repeated-hour order separately within a source file/entity series."""
    work = pd.DataFrame({"_position": range(len(parsed))})
    key_columns = []
    for number, key in enumerate(group_keys):
        column = f"_key_{number}"
        work[column] = key.reset_index(drop=True)
        key_columns.append(column)
    parts = []
    for positions in work.groupby(key_columns, sort=False, dropna=False).indices.values():
        subset = parsed.iloc[positions]
        parts.append(
            subset.dt.tz_localize(
                TIMEZONE, ambiguous="infer", nonexistent="raise"
            )
        )
    return pd.concat(parts).reindex(parsed.index)


def _empty_time(index: pd.Index) -> pd.Series:
    """Create an all-missing timezone-aware timestamp column."""
    return pd.Series(pd.NaT, index=index, dtype=f"datetime64[ns, {TIMEZONE}]")


def _warn_numeric_ranges(df: pd.DataFrame) -> None:
    """Warn about values that require review without redefining market rules."""
    mw_columns = [column for column in df.columns if column.endswith("_mw")]
    for column in mw_columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        if (numeric < 0).any():
            warnings.warn(f"{column} contains negative MW values", stacklevel=2)
        if (numeric > 100_000).any():
            warnings.warn(f"{column} contains values above 100,000 MW", stacklevel=2)
    if "lbmp" in df and (df["lbmp"].abs() > 10_000).any():
        warnings.warn("lbmp contains absolute values above $10,000/MWh", stacklevel=2)


def validate_clean_data(df: pd.DataFrame, dataset_name: str) -> dict[str, Any]:
    """Check required fields, key nulls/duplicates, timestamps, and ranges."""
    rules = {
        "actual_load": (
            ["interval_start", "interval_end", "zone", "actual_load_mw", "source"],
            ["interval_start", "zone"],
        ),
        "load_forecast": (
            [
                "interval_start",
                "interval_end",
                "forecast_vintage_date",
                "zone",
                "forecast_load_mw",
                "source",
            ],
            ["interval_start", "forecast_vintage_date", "zone"],
        ),
        "price": (
            ["interval_start", "interval_end", "market", "zone", "lbmp", "source"],
            ["interval_start", "market", "zone"],
        ),
        "weather": (
            ["interval_start", "location", "temperature", "source"],
            ["interval_start", "location"],
        ),
        "p15": (
            ["interval_start", "retrieval_time", "generation_outage_mw", "source"],
            ["interval_start", "retrieval_time"],
        ),
        "outage_latest": (
            ["event_id", "start_time", "facility_name", "source"],
            ["event_id"],
        ),
        "outage_snapshot": (
            ["snapshot_time", "facility_id", "facility_name", "start_time", "source"],
            ["snapshot_time", "facility_id", "start_time"],
        ),
    }
    if dataset_name not in rules:
        raise ValueError(f"Unknown clean dataset: {dataset_name}")
    required, key = rules[dataset_name]
    if dataset_name == "outage_snapshot":
        key = ["snapshot_time", "facility_id", "facility_name", "start_time"]
        if df["end_time"].notna().any():
            key.append("end_time")
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"{dataset_name} missing required columns: {missing}")
    if df[key].isna().any().any():
        raise ValueError(f"{dataset_name} has missing primary-key values")
    duplicate_count = int(df.duplicated(key).sum())
    if duplicate_count:
        raise ValueError(f"{dataset_name} has {duplicate_count} duplicate keys")
    for column in df.columns:
        if column.endswith("_time") or column.startswith("interval_"):
            if pd.api.types.is_datetime64_any_dtype(df[column]):
                if df[column].dt.tz is None:
                    raise ValueError(f"{dataset_name}.{column} is timezone-naive")
    _warn_numeric_ranges(df)
    return {
        "dataset": dataset_name,
        "rows": len(df),
        "date_min": _date_value(df, "interval_start", "min"),
        "date_max": _date_value(df, "interval_start", "max"),
        "zones": int(df["zone"].nunique()) if "zone" in df else None,
        "missingness": str(df.isna().sum().to_dict()),
        "duplicate_count": duplicate_count,
        "validation_status": "PASS",
    }


def _date_value(df: pd.DataFrame, column: str, operation: str) -> str | None:
    """Return a printable observed endpoint if an interval column exists."""
    if column not in df or not df[column].notna().any():
        return None
    value = getattr(df[column], operation)()
    return str(value)


def clean_load_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean either observed actual load or a wide forecast-vintage table."""
    if {"Time Stamp", "Name", "Load"}.issubset(df.columns):
        timezone_labels = df["Time Zone"] if "Time Zone" in df else None
        result = pd.DataFrame(
            {
                "interval_start": _localize(
                    df["Time Stamp"], timezone_labels=timezone_labels
                ),
                "zone": df["Name"].str.strip().str.upper(),
                "actual_load_mw": pd.to_numeric(df["Load"], errors="raise"),
                "source": "NYISO MIS P-58B",
            }
        )
        result["interval_end"] = result["interval_start"] + pd.Timedelta(minutes=5)
        result = result[
            ["interval_start", "interval_end", "zone", "actual_load_mw", "source"]
        ]
        validate_clean_data(result, "actual_load")
        return result

    if "Time Stamp" not in df or "_source_file" not in df:
        raise ValueError("Unrecognized load schema; actual fields or vintage file missing")
    value_columns = [
        column for column in df.columns if column not in {"Time Stamp", "_source_file"}
    ]
    result = df.melt(
        id_vars=["Time Stamp", "_source_file"],
        value_vars=value_columns,
        var_name="zone",
        value_name="forecast_load_mw",
    )
    result["interval_start"] = _localize(
        result["Time Stamp"],
        group_keys=[result["_source_file"], result["zone"]],
    )
    result["interval_end"] = result["interval_start"] + pd.Timedelta(hours=1)
    result["forecast_vintage_date"] = pd.to_datetime(
        result["_source_file"].str.extract(r"(\d{8})", expand=False),
        format="%Y%m%d",
        errors="raise",
    ).dt.date
    result["zone"] = result["zone"].str.strip().str.upper()
    result["forecast_load_mw"] = pd.to_numeric(
        result["forecast_load_mw"], errors="raise"
    )
    result["source"] = "NYISO MIS P-7"
    result = result[
        [
            "interval_start",
            "interval_end",
            "forecast_vintage_date",
            "zone",
            "forecast_load_mw",
            "source",
        ]
    ]
    validate_clean_data(result, "load_forecast")
    return result


def clean_price_data(df: pd.DataFrame, market: str) -> pd.DataFrame:
    """Standardize NYISO zonal LBMP and its observed components."""
    market_name = market.upper()
    if market_name not in {"DA", "RT"}:
        raise ValueError("market must be DA or RT")
    required = {
        "Time Stamp",
        "Name",
        "LBMP ($/MWHr)",
        "Marginal Cost Losses ($/MWHr)",
        "Marginal Cost Congestion ($/MWHr)",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Price data missing source columns: {sorted(missing)}")
    group_keys = [df["Name"]]
    if "_source_file" in df:
        group_keys.insert(0, df["_source_file"])
    result = pd.DataFrame(
        {
            "interval_start": _localize(
                df["Time Stamp"], group_keys=group_keys
            ),
            "market": market_name,
            "zone": df["Name"].str.strip().str.upper(),
            "lbmp": pd.to_numeric(df["LBMP ($/MWHr)"], errors="raise"),
            "loss_component": pd.to_numeric(
                df["Marginal Cost Losses ($/MWHr)"], errors="raise"
            ),
            "congestion_component": pd.to_numeric(
                df["Marginal Cost Congestion ($/MWHr)"], errors="raise"
            ),
            "source": f"NYISO MIS {'P-2A' if market_name == 'DA' else 'P-24A'}",
        }
    )
    result["energy_component"] = (
        result["lbmp"] - result["loss_component"] - result["congestion_component"]
    )
    minutes = 60 if market_name == "DA" else 5
    result["interval_end"] = result["interval_start"] + pd.Timedelta(minutes=minutes)
    result = result[
        [
            "interval_start",
            "interval_end",
            "market",
            "zone",
            "lbmp",
            "energy_component",
            "congestion_component",
            "loss_component",
            "source",
        ]
    ]
    validate_clean_data(result, "price")
    return result


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the observed Open-Meteo hourly development sample."""
    columns = {
        "time": "interval_start",
        "temperature_2m (°C)": "temperature",
        "precipitation (mm)": "precipitation",
        "wind_speed_10m (km/h)": "wind_speed",
        "relative_humidity_2m (%)": "relative_humidity",
    }
    missing = set(columns).difference(df.columns)
    if missing:
        raise ValueError(f"Weather data missing source columns: {sorted(missing)}")
    result = df.rename(columns=columns)[list(columns.values())].copy()
    source_timezones = (
        set(df["_source_timezone"].dropna().astype(str).str.upper())
        if "_source_timezone" in df
        else set()
    )
    if source_timezones and source_timezones.issubset({"UTC", "GMT"}):
        result["interval_start"] = (
            pd.to_datetime(result["interval_start"], errors="raise")
            .dt.tz_localize("UTC")
            .dt.tz_convert(TIMEZONE)
        )
    else:
        result["interval_start"] = _localize(result["interval_start"])
    result["location"] = "Albany development coordinate"
    result["source"] = "Open-Meteo Historical Weather API"
    result = result[
        [
            "interval_start",
            "location",
            "temperature",
            "precipitation",
            "wind_speed",
            "relative_humidity",
            "source",
        ]
    ]
    validate_clean_data(result, "weather")
    return result


def clean_p15_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the P-15 aggregate forecast MW snapshot, never as a count."""
    required = {"Date", "generation_outage_mw", "retrieval_time"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"P-15 missing source columns: {sorted(missing)}")
    result = pd.DataFrame(
        {
            "interval_start": _localize(df["Date"]),
            "retrieval_time": _localize(df["retrieval_time"]),
            "generation_outage_mw": pd.to_numeric(
                df["generation_outage_mw"], errors="raise"
            ),
            "source": "NYISO MIS P-15 current forecast snapshot",
        }
    )
    result["interval_end"] = result["interval_start"] + pd.Timedelta(days=1)
    result = result[
        [
            "interval_start",
            "interval_end",
            "retrieval_time",
            "generation_outage_mw",
            "source",
        ]
    ]
    validate_clean_data(result, "p15")
    return result


def clean_outage_data(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Clean P-14B latest events or P-54 transmission-outage snapshots."""
    if "Outage ID" in df:
        starts = df["Date Out"].astype(str) + " " + df["Time Out"].astype(str)
        ends = df["Date In"].astype(str) + " " + df["Time In"].astype(str)
        result = pd.DataFrame(
            {
                "event_id": df["Outage ID"].astype(str),
                "start_time": _localize(starts),
                "end_time": _localize(ends),
                "facility_name": df["Equipment Name"].astype(str),
                "planned_forced_status": pd.NA,
                "affected_mw": pd.NA,
                "zone": pd.NA,
                "source": source,
                "historical_complete": False,
            }
        )
        result = result.drop_duplicates()
        validate_clean_data(result, "outage_latest")
        return result

    if "Timestamp" not in df or "PTID" not in df:
        raise ValueError("Unrecognized outage source schema")
    if "Scheduled Out Date/Time" in df:
        starts = _localize(df["Scheduled Out Date/Time"])
        ends = _localize(df["Scheduled In Date/Time"])
    elif "Outage Date/Time" in df:
        starts = _localize(df["Outage Date/Time"])
        ends = _empty_time(df.index)
    else:
        raise ValueError("Outage snapshot has no observed start-time field")
    result = pd.DataFrame(
        {
            "snapshot_time": _localize(df["Timestamp"]),
            "facility_id": df["PTID"].astype(str),
            "facility_name": df["Equipment Name"].astype(str),
            "start_time": starts,
            "end_time": ends,
            "source": source,
            "historical_complete": False,
        }
    ).drop_duplicates()
    validate_clean_data(result, "outage_snapshot")
    return result


def _read_weather(path: Path) -> pd.DataFrame:
    """Read the hourly section below Open-Meteo's metadata preamble."""
    lines = path.read_text(encoding="utf-8").splitlines()
    header = next(index for index, line in enumerate(lines) if line.startswith("time,"))
    frame = pd.read_csv(StringIO("\n".join(lines[header:])))
    if header >= 2:
        metadata = pd.read_csv(StringIO("\n".join(lines[:2])))
        if "timezone" in metadata and not metadata.empty:
            frame["_source_timezone"] = str(metadata.loc[0, "timezone"])
    return frame


def _required_files(directory: Path, pattern: str) -> list[Path]:
    """Return sorted input files or raise a clear missing-raw-data error."""
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No raw files match {directory / pattern}")
    return paths


def _read_parquet_files(
    directory: Path, required_column: str | None = None
) -> pd.DataFrame:
    """Combine monthly Parquet files while ignoring timestamped rerun copies."""
    frames = []
    paths = _prefer_canonical_downloads(_required_files(directory, "*.parquet"))
    for path in paths:
        frame = pd.read_parquet(path)
        if required_column is None or required_column in frame:
            frames.append(frame)
    if not frames:
        raise ValueError(
            f"No raw Parquet in {directory} preserves column {required_column}"
        )
    return pd.concat(frames, ignore_index=True)


def _prefer_canonical_downloads(paths: list[Path]) -> list[Path]:
    """Keep one raw container per archive without deleting any rerun files."""
    suffix = re.compile(r"_\d{8}T\d{6}[+-]\d{4}(?:_\d+)?$")
    groups: dict[str, list[Path]] = {}
    for path in paths:
        key = suffix.sub("", path.stem)
        groups.setdefault(key, []).append(path)
    selected = []
    for key, candidates in groups.items():
        canonical = next((path for path in candidates if path.stem == key), None)
        selected.append(canonical or max(candidates, key=lambda path: path.stat().st_mtime))
    return sorted(selected)


def _read_csv_files(directory: Path) -> pd.DataFrame:
    """Combine all CSV snapshots and remove exact rerun duplicates."""
    frames = [pd.read_csv(path) for path in _required_files(directory, "*.csv")]
    return pd.concat(frames, ignore_index=True).drop_duplicates(ignore_index=True)


def _read_weather_files(directory: Path) -> pd.DataFrame:
    """Combine all Open-Meteo date windows without duplicating overlaps."""
    frames = [_read_weather(path) for path in _required_files(directory, "*.csv")]
    utc_frames = [
        frame
        for frame in frames
        if "_source_timezone" in frame
        and set(frame["_source_timezone"].astype(str).str.upper()).issubset(
            {"UTC", "GMT"}
        )
    ]
    selected = utc_frames or frames
    return pd.concat(selected, ignore_index=True).drop_duplicates(ignore_index=True)


def _read_latest_csv(directory: Path) -> pd.DataFrame:
    """Read only the newest file from a source that is explicitly latest-only."""
    paths = _required_files(directory, "*.csv")
    newest = max(paths, key=lambda path: path.stat().st_mtime)
    return pd.read_csv(newest)


def _save_table(df: pd.DataFrame, path: Path) -> None:
    """Save one clean table to interim Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def clean_all_dev_data(config: ProjectConfig) -> pd.DataFrame:
    """Clean sources used by current panels; leave large P-54 raw partitions intact."""
    raw = config.raw_data_dir
    interim = config.interim_data_dir
    tables: list[tuple[str, str, pd.DataFrame]] = []

    actual = clean_load_data(_read_parquet_files(raw / "nyiso/actual_load"))
    forecast = clean_load_data(
        _read_parquet_files(raw / "nyiso/load_forecast", "_source_file")
    )
    da = clean_price_data(
        _read_parquet_files(raw / "nyiso/da_lbmp"), "DA"
    )
    rt = clean_price_data(
        _read_parquet_files(raw / "nyiso/rt_lbmp"), "RT"
    )
    weather = clean_weather_data(_read_weather_files(raw / "open_meteo/albany"))
    p15 = clean_p15_data(_read_csv_files(raw / "nyiso/p15"))
    p14 = clean_outage_data(
        _read_latest_csv(raw / "nyiso/outage_schedules_latest"),
        "NYISO MIS P-14B latest-only snapshot",
    )
    tables.extend(
        [
            ("actual_load", "actual_load", actual),
            ("load_forecast", "load_forecast", forecast),
            ("da_lbmp", "price", da),
            ("rt_lbmp", "price", rt),
            ("weather_albany", "weather", weather),
            ("p15_generation_outage_forecast", "p15", p15),
            ("outage_schedules_latest", "outage_latest", p14),
        ]
    )
    summaries = []
    for name, schema, frame in tables:
        _save_table(frame, interim / f"{name}.parquet")
        summary = validate_clean_data(frame, schema)
        summary["table"] = name
        summaries.append(summary)
    for name in (
        "p54a_scheduled_snapshots",
        "p54b_actual_snapshots",
        "p54c_scheduled_snapshots",
    ):
        summaries.append(
            {
                "dataset": "outage_snapshot",
                "rows": None,
                "date_min": None,
                "date_max": None,
                "zones": None,
                "missingness": None,
                "duplicate_count": None,
                "validation_status": "SKIPPED",
                "table": name,
                "notes": (
                    "Raw P-54 retained by month; not loaded into the core panel "
                    "and not treated as outage event count."
                ),
            }
        )
    result = pd.DataFrame(summaries)
    result.to_csv(config.output_dir / "tables/clean_data_summary.csv", index=False)
    return result
