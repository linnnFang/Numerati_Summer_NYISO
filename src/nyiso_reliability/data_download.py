"""Download and inspect small raw-data samples from verified public endpoints."""

from __future__ import annotations

import json
import zipfile
from datetime import date, datetime, timedelta
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd

from nyiso_reliability.config import ProjectConfig

NY_TZ = ZoneInfo("America/New_York")
NYISO_ROOT = "https://mis.nyiso.com/public/"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
HTTP_TIMEOUT_SECONDS = 60


class _LinkParser(HTMLParser):
    """Collect href values from a small NYISO report index page."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


def _retrieval_time() -> datetime:
    """Return the current project-local retrieval timestamp."""
    return datetime.now(NY_TZ)


def _fetch(url: str) -> bytes:
    """Fetch one public resource with a bounded timeout."""
    request = Request(url, headers={"User-Agent": "nyiso-reliability-research/0.1"})
    with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        return response.read()


def _month_starts(config: ProjectConfig) -> list[date]:
    """Return each calendar-month start intersecting the configured window."""
    months = pd.period_range(config.start_date, config.end_date, freq="M")
    return [date(period.year, period.month, 1) for period in months]


def _nyiso_archive_links(report_id: str) -> tuple[str, list[str]]:
    """Read one NYISO report index and return its absolute archive links."""
    index_url = urljoin(NYISO_ROOT, f"{report_id}list.htm")
    parser = _LinkParser()
    parser.feed(_fetch(index_url).decode("utf-8", errors="replace"))
    return index_url, [urljoin(index_url, link) for link in parser.links]


def _nyiso_archive_url(
    report_id: str, month_start: date, links: list[str]
) -> str:
    """Select one archive link for a requested report month."""
    month_token = month_start.strftime("%Y%m")
    candidates = [
        link
        for link in links
        if month_token in link
        and link.lower().split("?")[0].endswith((".zip", ".csv", ".txt"))
    ]
    if not candidates:
        raise RuntimeError(
            f"No {month_token} archive link found on NYISO {report_id} index"
        )
    candidates.sort(key=lambda value: (not value.lower().endswith(".zip"), value))
    return candidates[0]


def _nyiso_latest_url(report_id: str) -> str:
    """Select the first current CSV link from a latest-only NYISO index."""
    index_url = urljoin(NYISO_ROOT, f"{report_id}list.htm")
    parser = _LinkParser()
    parser.feed(_fetch(index_url).decode("utf-8", errors="replace"))
    candidates = [
        urljoin(index_url, link)
        for link in parser.links
        if link.lower().split("?")[0].endswith(".csv")
    ]
    if not candidates:
        raise RuntimeError(f"No current CSV link found on NYISO {report_id} index")
    return candidates[0]


def _unique_path(directory: Path, filename: str, retrieved: datetime) -> Path:
    """Choose a raw path without overwriting an existing file."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    if not destination.exists():
        return destination
    stamp = retrieved.strftime("%Y%m%dT%H%M%S%z")
    candidate = directory / f"{destination.stem}_{stamp}{destination.suffix}"
    counter = 1
    while candidate.exists():
        candidate = directory / (
            f"{destination.stem}_{stamp}_{counter}{destination.suffix}"
        )
        counter += 1
    return candidate


def _save_bytes(
    content: bytes, directory: Path, url: str, retrieved: datetime
) -> Path:
    """Save source bytes using their URL filename without overwriting."""
    filename = Path(url.split("?", 1)[0]).name or "download.bin"
    destination = _unique_path(directory, filename, retrieved)
    destination.write_bytes(content)
    return destination


def _save_nyiso_table(
    content: bytes,
    df: pd.DataFrame,
    directory: Path,
    url: str,
    retrieved: datetime,
) -> Path:
    """Save CSV directly or a monthly ZIP's tabular contents as Parquet."""
    if url.lower().split("?", 1)[0].endswith(".zip"):
        stem = Path(url.split("?", 1)[0]).stem
        destination = _unique_path(directory, f"{stem}.parquet", retrieved)
        df.to_parquet(destination, index=False)
        return destination
    return _save_bytes(content, directory, url, retrieved)


def _read_csv(content: bytes) -> pd.DataFrame:
    """Read CSV bytes with a conservative encoding fallback."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(BytesIO(content), encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Unable to decode CSV response")


def _read_tabular(content: bytes, url: str) -> pd.DataFrame:
    """Inspect a CSV or all CSV members of one small monthly ZIP archive."""
    if url.lower().split("?", 1)[0].endswith(".zip"):
        frames: list[pd.DataFrame] = []
        with zipfile.ZipFile(BytesIO(content)) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            for member in members:
                frame = _read_csv(archive.read(member))
                frame["_source_file"] = Path(member).name
                frames.append(frame)
        if not frames:
            raise ValueError("ZIP archive contains no CSV files")
        return pd.concat(frames, ignore_index=True)
    return _read_csv(content)


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Find a column by normalized exact name without claiming source semantics."""
    normalized = {str(column).strip().lower(): str(column) for column in df.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _observed_date_range(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Best-effort date range for inspection; cleaning is deferred to Step 2."""
    column = _find_column(
        df,
        ("time stamp", "timestamp", "datetime", "date/time", "time", "date"),
    )
    if column is not None:
        values = pd.to_datetime(df[column], errors="coerce")
        if values.notna().any():
            return str(values.min()), str(values.max())

    date_out = _find_column(df, ("date out",))
    date_in = _find_column(df, ("date in",))
    if date_out is None or date_in is None:
        return None, None
    time_out = _find_column(df, ("time out",))
    time_in = _find_column(df, ("time in",))
    starts = df[date_out].astype(str)
    ends = df[date_in].astype(str)
    if time_out:
        starts = starts + " " + df[time_out].astype(str)
    if time_in:
        ends = ends + " " + df[time_in].astype(str)
    parsed_starts = pd.to_datetime(starts, errors="coerce")
    parsed_ends = pd.to_datetime(ends, errors="coerce")
    if not parsed_starts.notna().any() or not parsed_ends.notna().any():
        return None, None
    return str(parsed_starts.min()), str(parsed_ends.max())


def _zone_count(df: pd.DataFrame) -> int | None:
    """Count observed location labels when a common exact column is present."""
    column = _find_column(df, ("zone", "zone name", "name"))
    return int(df[column].nunique(dropna=True)) if column else None


def _summary(
    *,
    dataset: str,
    source: str,
    url: str,
    df: pd.DataFrame,
    saved_path: Path,
    retrieved: datetime | None,
    requested_month: str | None = None,
    notes: str = "",
    source_columns: list[str] | None = None,
    status: str = "READY",
) -> dict[str, Any]:
    """Build a truthful inspection record from the downloaded frame."""
    date_min, date_max = _observed_date_range(df)
    return {
        "dataset": dataset,
        "source": source,
        "status": status,
        "requested_month": requested_month,
        "url": url,
        "retrieval_time": retrieved.isoformat() if retrieved else None,
        "rows": len(df),
        "columns": json.dumps([str(value) for value in df.columns]),
        "source_columns": json.dumps(source_columns or [str(value) for value in df.columns]),
        "date_min": date_min,
        "date_max": date_max,
        "unique_zones": _zone_count(df),
        "missingness": json.dumps(
            {str(column): int(count) for column, count in df.isna().sum().items()}
        ),
        "duplicates": int(df.duplicated().sum()),
        "saved_path": str(saved_path),
        "blocker": "",
        "notes": notes,
    }


def _blocked(
    dataset: str,
    source: str,
    url: str,
    exc: Exception,
    requested_month: str | None = None,
) -> dict[str, Any]:
    """Represent a failed source without pretending that data were downloaded."""
    return {
        "dataset": dataset,
        "source": source,
        "status": "BLOCKED",
        "requested_month": requested_month,
        "url": url,
        "retrieval_time": _retrieval_time().isoformat(),
        "rows": None,
        "columns": "[]",
        "source_columns": "[]",
        "date_min": None,
        "date_max": None,
        "unique_zones": None,
        "missingness": "{}",
        "duplicates": None,
        "saved_path": "",
        "blocker": f"{type(exc).__name__}: {exc}",
        "notes": "",
    }


def _existing_nyiso_path(directory: Path, url: str) -> Path | None:
    """Return the canonical saved archive path when it already exists."""
    source_path = Path(url.split("?", 1)[0])
    filename = (
        f"{source_path.stem}.parquet"
        if source_path.suffix.lower() == ".zip"
        else source_path.name
    )
    path = directory / filename
    return path if path.is_file() else None


def _read_saved_nyiso_table(path: Path) -> pd.DataFrame:
    """Read a canonical saved NYISO CSV or Parquet table."""
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _download_nyiso_month(
    dataset: str,
    report_id: str,
    config: ProjectConfig,
    month_start: date,
    index_url: str,
    links: list[str],
) -> dict[str, Any]:
    """Download or reuse one NYISO monthly archive and inspect its contents."""
    month_token = month_start.strftime("%Y-%m")
    try:
        url = _nyiso_archive_url(report_id, month_start, links)
        directory = config.raw_data_dir / "nyiso" / dataset
        existing = _existing_nyiso_path(directory, url)
        if existing is not None:
            df = _read_saved_nyiso_table(existing)
            return _summary(
                dataset=dataset,
                source=f"NYISO MIS {report_id}",
                url=url,
                df=df,
                saved_path=existing,
                retrieved=None,
                requested_month=month_token,
                notes="Reused existing raw file; no archive download performed.",
            )
        retrieved = _retrieval_time()
        content = _fetch(url)
        df = _read_tabular(content, url)
        saved_path = _save_nyiso_table(
            content,
            df,
            config.raw_data_dir / "nyiso" / dataset,
            url,
            retrieved,
        )
        return _summary(
            dataset=dataset,
            source=f"NYISO MIS {report_id}",
            url=url,
            df=df,
            saved_path=saved_path,
            retrieved=retrieved,
            requested_month=month_token,
        )
    except Exception as exc:
        return _blocked(
            dataset,
            f"NYISO MIS {report_id}",
            index_url,
            exc,
            requested_month=month_token,
        )


def _download_nyiso_range(
    dataset: str, report_id: str, config: ProjectConfig
) -> list[dict[str, Any]]:
    """Download each report month intersecting the configured date range."""
    months = _month_starts(config)
    index_url = urljoin(NYISO_ROOT, f"{report_id}list.htm")
    try:
        index_url, links = _nyiso_archive_links(report_id)
    except Exception as exc:
        return [
            _blocked(
                dataset,
                f"NYISO MIS {report_id}",
                index_url,
                exc,
                requested_month=month.strftime("%Y-%m"),
            )
            for month in months
        ]
    return [
        _download_nyiso_month(
            dataset, report_id, config, month, index_url, links
        )
        for month in months
    ]


def download_actual_load(config: ProjectConfig) -> list[dict[str, Any]]:
    """Download configured months of real-time actual zonal load (P-58B)."""
    return _download_nyiso_range("actual_load", "P-58B", config)


def download_load_forecast(config: ProjectConfig) -> list[dict[str, Any]]:
    """Download configured months of NYISO ISO load forecast (P-7)."""
    return _download_nyiso_range("load_forecast", "P-7", config)


def download_da_lbmp(config: ProjectConfig) -> list[dict[str, Any]]:
    """Download configured months of NYISO day-ahead zonal LBMP (P-2A)."""
    return _download_nyiso_range("da_lbmp", "P-2A", config)


def download_rt_lbmp(config: ProjectConfig) -> list[dict[str, Any]]:
    """Download configured months of NYISO real-time zonal LBMP (P-24A)."""
    return _download_nyiso_range("rt_lbmp", "P-24A", config)


def download_p15_snapshot(config: ProjectConfig) -> dict[str, Any]:
    """Download the current P-15 aggregate generation-outage MW forecast."""
    url = urljoin(NYISO_ROOT, "csv/genmaint/gen_maint_report.csv")
    try:
        retrieved = _retrieval_time()
        source_df = _read_csv(_fetch(url))
        source_columns = [str(value) for value in source_df.columns]
        mw_column = "Forecasted Generation Outage (MW)"
        if mw_column not in source_df.columns or "Date" not in source_df.columns:
            raise ValueError(f"P-15 schema changed: observed {source_columns}")
        df = source_df.rename(columns={mw_column: "generation_outage_mw"}).copy()
        df["retrieval_time"] = retrieved.isoformat()
        filename = f"p15_snapshot_{retrieved.strftime('%Y%m%dT%H%M%S%z')}.csv"
        saved_path = _unique_path(
            config.raw_data_dir / "nyiso" / "p15", filename, retrieved
        )
        df.to_csv(saved_path, index=False)
        return _summary(
            dataset="p15_generation_outage_forecast",
            source="NYISO MIS P-15",
            url=url,
            df=df,
            saved_path=saved_path,
            retrieved=retrieved,
            source_columns=source_columns,
            notes=(
                "Current/future aggregate daily forecast MW snapshot; "
                "not outage-event data and not an event count."
            ),
            status="LATEST_ONLY",
        )
    except Exception as exc:
        return _blocked("p15_generation_outage_forecast", "NYISO MIS P-15", url, exc)


def download_p14b_latest(config: ProjectConfig) -> dict[str, Any]:
    """Download the current P-14B outage schedule; no history is published."""
    index_url = urljoin(NYISO_ROOT, "P-14Blist.htm")
    try:
        url = _nyiso_latest_url("P-14B")
        retrieved = _retrieval_time()
        content = _fetch(url)
        df = _read_tabular(content, url)
        saved_path = _save_nyiso_table(
            content,
            df,
            config.raw_data_dir / "nyiso" / "outage_schedules_latest",
            url,
            retrieved,
        )
        return _summary(
            dataset="outage_schedules_latest",
            source="NYISO MIS P-14B",
            url=url,
            df=df,
            saved_path=saved_path,
            retrieved=retrieved,
            notes=(
                "Latest-only snapshot; NYISO P-14B index exposes no historical "
                "archive, so it does not provide the configured historical window."
            ),
            status="LATEST_ONLY",
        )
    except Exception as exc:
        return _blocked("outage_schedules_latest", "NYISO MIS P-14B", index_url, exc)


def download_outage_reports(config: ProjectConfig) -> list[dict[str, Any]]:
    """Download configured P-54 months and one latest-only P-14B snapshot."""
    reports = (
        ("rt_scheduled_transmission_outages", "P-54A"),
        ("rt_actual_transmission_outages", "P-54B"),
        ("da_scheduled_transmission_outages", "P-54C"),
    )
    results: list[dict[str, Any]] = []
    for name, report_id in reports:
        results.extend(_download_nyiso_range(name, report_id, config))
    results.append(download_p14b_latest(config))
    return results


def _open_meteo_frame(content: bytes) -> pd.DataFrame:
    """Read Open-Meteo CSV after its small metadata preamble."""
    lines = content.decode("utf-8").splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if line.startswith("time,")), None
    )
    if header_index is None:
        raise ValueError("Open-Meteo response has no hourly CSV header")
    return pd.read_csv(BytesIO("\n".join(lines[header_index:]).encode("utf-8")))


def download_weather(config: ProjectConfig) -> dict[str, Any]:
    """Download Albany weather in UTC for unambiguous local-time conversion."""
    utc_end_date = config.end_date + timedelta(days=1)
    params = {
        "latitude": 42.6526,
        "longitude": -73.7562,
        "start_date": config.start_date.isoformat(),
        "end_date": utc_end_date.isoformat(),
        "hourly": "temperature_2m,precipitation,wind_speed_10m,relative_humidity_2m",
        "timezone": "UTC",
        "format": "csv",
    }
    url = f"{OPEN_METEO_ARCHIVE}?{urlencode(params)}"
    try:
        retrieved = _retrieval_time()
        content = _fetch(url)
        filename = (
            f"albany_{config.start_date.isoformat()}_"
            f"{config.end_date.isoformat()}_utc_buffered.csv"
        )
        saved_path = _unique_path(
            config.raw_data_dir / "open_meteo" / "albany", filename, retrieved
        )
        saved_path.write_bytes(content)
        df = _open_meteo_frame(content)
        return _summary(
            dataset="weather_albany_dev_sample",
            source="Open-Meteo Historical Weather API",
            url=url,
            df=df,
            saved_path=saved_path,
            retrieved=retrieved,
            notes=(
                "Single Albany coordinate for development only; timestamps requested "
                "in UTC with a one-day end buffer for explicit America/New_York "
                "DST conversion and local-date filtering."
            ),
        )
    except Exception as exc:
        return _blocked(
            "weather_albany_dev_sample", "Open-Meteo Historical Weather API", url, exc
        )


def download_dev_sample(config: ProjectConfig) -> pd.DataFrame:
    """Download configured monthly archives and save a truthful manifest."""
    monthly_functions = (
        download_actual_load,
        download_load_forecast,
        download_da_lbmp,
        download_rt_lbmp,
    )
    records: list[dict[str, Any]] = []
    for function in monthly_functions:
        records.extend(function(config))
    records.extend(download_outage_reports(config))
    records.extend([download_p15_snapshot(config), download_weather(config)])
    summary = pd.DataFrame(records)
    output_dir = config.output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "data_download_summary.csv", index=False)
    return summary
