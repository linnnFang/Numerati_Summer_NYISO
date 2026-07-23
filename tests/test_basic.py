"""Basic tests for the Step 0 project scaffold."""

from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nyiso_reliability.cli import app
from nyiso_reliability.config import ProjectConfig, load_config
from nyiso_reliability.data_clean import (
    _read_parquet_files,
    clean_load_data,
    clean_price_data,
    clean_weather_data,
)
from nyiso_reliability.data_download import (
    _month_starts,
    _observed_date_range,
    download_actual_load,
    download_p15_snapshot,
)
from nyiso_reliability.features import _floor_hour, _local_day_start


def test_load_dev_config() -> None:
    """The development config loads and resolves paths."""
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/dev.yaml")

    assert config.timezone == "America/New_York"
    assert config.start_date <= config.end_date
    assert config.raw_data_dir == root / "data/raw"


def test_month_starts_cover_configured_range(tmp_path: Path) -> None:
    """Every calendar month intersecting the analysis window is requested."""
    config = ProjectConfig(
        project_name="Test",
        timezone="America/New_York",
        start_date=date(2022, 11, 15),
        end_date=date(2024, 1, 7),
        raw_data_dir=tmp_path / "raw",
        interim_data_dir=tmp_path / "interim",
        processed_data_dir=tmp_path / "processed",
        output_dir=tmp_path / "outputs",
        random_seed=1,
    )

    months = _month_starts(config)

    assert months[0] == date(2022, 11, 1)
    assert months[-1] == date(2024, 1, 1)
    assert len(months) == 15


def test_monthly_parquet_files_ignore_rerun_container_copies(
    tmp_path: Path,
) -> None:
    """Cleaning keeps legitimate rows while ignoring a repeated raw container."""
    import pandas as pd

    directory = tmp_path / "monthly"
    directory.mkdir()
    pd.DataFrame({"value": [1, 2]}).to_parquet(directory / "202401.parquet")
    pd.DataFrame({"value": [1, 2]}).to_parquet(
        directory / "202401_20260717T120000-0400.parquet"
    )
    pd.DataFrame({"value": [2, 3]}).to_parquet(directory / "202402.parquet")

    result = _read_parquet_files(directory)

    assert result["value"].tolist() == [1, 2, 2, 3]


def test_explicit_nyiso_dst_labels_disambiguate_repeated_hour() -> None:
    """P-58B EDT/EST labels preserve both fall-back observations."""
    import pandas as pd

    raw = pd.DataFrame(
        {
            "Time Stamp": ["11/05/2023 01:00:00", "11/05/2023 01:00:00"],
            "Time Zone": ["EDT", "EST"],
            "Name": ["CAPITL", "CAPITL"],
            "Load": [1000.0, 1001.0],
        }
    )

    clean = clean_load_data(raw)

    assert clean["interval_start"].nunique() == 2
    assert [value.utcoffset().total_seconds() for value in clean["interval_start"]] == [
        -4 * 3600,
        -5 * 3600,
    ]


def test_price_dst_hour_uses_source_order_within_file_and_zone() -> None:
    """Two ordered NYISO LBMP 01:00 rows become EDT then EST."""
    import pandas as pd

    raw = pd.DataFrame(
        {
            "Time Stamp": ["11/05/2023 01:00", "11/05/2023 01:00"],
            "Name": ["CAPITL", "CAPITL"],
            "LBMP ($/MWHr)": [20.0, 21.0],
            "Marginal Cost Losses ($/MWHr)": [1.0, 1.0],
            "Marginal Cost Congestion ($/MWHr)": [0.0, 0.0],
            "_source_file": ["20231105.csv", "20231105.csv"],
        }
    )

    clean = clean_price_data(raw, "DA")

    assert clean["interval_start"].nunique() == 2


def test_utc_weather_converts_both_fall_back_hours() -> None:
    """UTC weather timestamps map unambiguously to both New York 01:00 hours."""
    import pandas as pd

    raw = pd.DataFrame(
        {
            "time": ["2023-11-05T05:00", "2023-11-05T06:00"],
            "temperature_2m (°C)": [10.0, 9.0],
            "precipitation (mm)": [0.0, 0.0],
            "wind_speed_10m (km/h)": [5.0, 5.0],
            "relative_humidity_2m (%)": [80, 81],
            "_source_timezone": ["UTC", "UTC"],
        }
    )

    clean = clean_weather_data(raw)

    assert clean["interval_start"].dt.hour.tolist() == [1, 1]
    assert clean["interval_start"].nunique() == 2


def test_feature_time_buckets_preserve_repeated_hour_and_local_day() -> None:
    """Hourly flooring keeps both DST folds while day grouping uses local midnight."""
    import pandas as pd

    values = pd.Series(
        pd.to_datetime(["2023-11-05T05:30:00Z", "2023-11-05T06:30:00Z"])
        .tz_convert("America/New_York")
    )

    hours = _floor_hour(values)
    days = _local_day_start(values)

    assert hours.nunique() == 2
    assert hours.dt.hour.tolist() == [1, 1]
    assert days.nunique() == 1
    assert days.dt.hour.tolist() == [0, 0]


def test_monthly_download_covers_range_and_reuses_existing_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Monthly downloads cover both endpoints and do not redownload saved archives."""
    import pandas as pd

    config = ProjectConfig(
        project_name="Test",
        timezone="America/New_York",
        start_date=date(2024, 1, 31),
        end_date=date(2024, 2, 1),
        raw_data_dir=tmp_path / "raw",
        interim_data_dir=tmp_path / "interim",
        processed_data_dir=tmp_path / "processed",
        output_dir=tmp_path / "outputs",
        random_seed=1,
    )
    links = [
        "https://example.test/20240101pal_csv.zip",
        "https://example.test/20240201pal_csv.zip",
    ]
    fetches: list[str] = []
    monkeypatch.setattr(
        "nyiso_reliability.data_download._nyiso_archive_links",
        lambda report_id: ("https://example.test/index", links),
    )
    monkeypatch.setattr(
        "nyiso_reliability.data_download._fetch",
        lambda url: fetches.append(url) or b"archive",
    )
    monkeypatch.setattr(
        "nyiso_reliability.data_download._read_tabular",
        lambda content, url: pd.DataFrame({"month": [Path(url).name[:6]]}),
    )

    first = download_actual_load(config)
    second = download_actual_load(config)

    assert [record["requested_month"] for record in first] == ["2024-01", "2024-02"]
    assert len(fetches) == 2
    assert all(record["status"] == "READY" for record in second)
    assert all("Reused existing raw file" in record["notes"] for record in second)


def test_invalid_date_range(tmp_path: Path) -> None:
    """An inverted analysis window is rejected."""
    path = tmp_path / "invalid.yaml"
    path.write_text(
        """
project_name: Test
timezone: America/New_York
start_date: 2024-02-01
end_date: 2024-01-01
raw_data_dir: data/raw
interim_data_dir: data/interim
processed_data_dir: data/processed
output_dir: outputs
random_seed: 1
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="start_date"):
        load_config(path)


def test_show_config_cli() -> None:
    """The basic CLI prints validated development settings."""
    root = Path(__file__).resolve().parents[1]
    result = CliRunner().invoke(
        app,
        ["show-config", "--config", str(root / "configs/dev.yaml")],
    )

    assert result.exit_code == 0, result.output
    assert "America/New_York" in result.output


def test_p15_snapshot_preserves_retrieval_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P-15 remains aggregate forecast MW and gains snapshot retrieval metadata."""
    response = (
        b'"Date","Forecasted Generation Outage (MW)"\n'
        b'"01/02/2024","1234"\n'
    )
    monkeypatch.setattr(
        "nyiso_reliability.data_download._fetch", lambda url: response
    )
    config = ProjectConfig(
        project_name="Test",
        timezone="America/New_York",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 7),
        raw_data_dir=tmp_path / "raw",
        interim_data_dir=tmp_path / "interim",
        processed_data_dir=tmp_path / "processed",
        output_dir=tmp_path / "outputs",
        random_seed=1,
    )

    result = download_p15_snapshot(config)

    assert result["status"] == "LATEST_ONLY"
    saved = Path(result["saved_path"])
    assert saved.is_file()
    content = saved.read_text(encoding="utf-8")
    assert "generation_outage_mw" in content
    assert "retrieval_time" in content
    assert "event count" in result["notes"]


def test_split_outage_dates_are_inspected() -> None:
    """P-14B split date/time components produce an observed schedule range."""
    import pandas as pd

    frame = pd.DataFrame(
        {
            "Date Out": ["01/02/2024", "01/03/2024"],
            "Time Out": ["01:00", "02:00"],
            "Date In": ["01/04/2024", "01/06/2024"],
            "Time In": ["03:00", "04:00"],
        }
    )

    date_min, date_max = _observed_date_range(frame)

    assert date_min == "2024-01-02 01:00:00"
    assert date_max == "2024-01-06 04:00:00"


def test_clean_load_is_new_york_timezone_aware() -> None:
    """Naive NYISO clock times become timezone-aware New York timestamps."""
    import pandas as pd

    raw = pd.DataFrame(
        {
            "Time Stamp": ["01/01/2024 00:00:00"],
            "Name": ["CAPITL"],
            "Load": [1000.0],
        }
    )

    clean = clean_load_data(raw)

    assert str(clean["interval_start"].dt.tz) == "America/New_York"
    assert clean.loc[0, "interval_end"] - clean.loc[0, "interval_start"] == pd.Timedelta(
        minutes=5
    )


def test_ambiguous_dst_time_is_rejected() -> None:
    """A naive repeated fall-back clock time is never guessed silently."""
    import pandas as pd

    raw = pd.DataFrame(
        {
            "Time Stamp": ["11/03/2024 01:30:00"],
            "Name": ["CAPITL"],
            "Load": [1000.0],
        }
    )

    with pytest.raises(Exception, match="Cannot infer dst time"):
        clean_load_data(raw)


def test_negative_load_emits_warning() -> None:
    """Impossible-looking negative load remains visible and emits a warning."""
    import pandas as pd

    raw = pd.DataFrame(
        {
            "Time Stamp": ["01/01/2024 00:00:00"],
            "Name": ["CAPITL"],
            "Load": [-1.0],
        }
    )

    with pytest.warns(UserWarning, match="negative MW"):
        clean_load_data(raw)


def test_nonexistent_dst_time_is_rejected() -> None:
    """A naive spring-forward nonexistent clock time is rejected."""
    import pandas as pd

    raw = pd.DataFrame(
        {
            "Time Stamp": ["03/10/2024 02:30:00"],
            "Name": ["CAPITL"],
            "Load": [1000.0],
        }
    )

    with pytest.raises(Exception, match="2024-03-10 02:30:00"):
        clean_load_data(raw)


def test_duplicate_load_key_is_rejected() -> None:
    """Two values for the same interval and zone fail clean validation."""
    import pandas as pd

    raw = pd.DataFrame(
        {
            "Time Stamp": ["01/01/2024 00:00:00", "01/01/2024 00:00:00"],
            "Name": ["CAPITL", "CAPITL"],
            "Load": [1000.0, 1001.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate keys"):
        clean_load_data(raw)


def test_missing_required_load_column_is_rejected() -> None:
    """Unknown or incomplete raw load schemas are not guessed."""
    import pandas as pd

    raw = pd.DataFrame({"Time Stamp": ["01/01/2024 00:00:00"]})

    with pytest.raises(ValueError, match="Unrecognized load schema"):
        clean_load_data(raw)
