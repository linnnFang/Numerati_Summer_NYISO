"""Read a YAML project configuration and validate its date range."""

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    """Small set of settings shared by the research steps."""

    project_name: str
    timezone: str
    start_date: date
    end_date: date
    raw_data_dir: Path
    interim_data_dir: Path
    processed_data_dir: Path
    output_dir: Path
    random_seed: int

    def to_dict(self) -> dict[str, Any]:
        """Return JSON/YAML-friendly configuration values."""
        values = asdict(self)
        for key in ("start_date", "end_date"):
            values[key] = values[key].isoformat()
        for key in (
            "raw_data_dir",
            "interim_data_dir",
            "processed_data_dir",
            "output_dir",
        ):
            values[key] = str(values[key])
        return values


def load_config(config_path: str | Path) -> ProjectConfig:
    """Load one complete YAML file and reject an inverted date range."""
    path = Path(config_path).expanduser().resolve()
    try:
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read config {path}: {exc}") from exc
    if not isinstance(values, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")

    root = path.parent.parent
    try:
        start_date = date.fromisoformat(str(values["start_date"]))
        end_date = date.fromisoformat(str(values["end_date"]))
        config = ProjectConfig(
            project_name=str(values["project_name"]),
            timezone=str(values["timezone"]),
            start_date=start_date,
            end_date=end_date,
            raw_data_dir=_resolve_path(values["raw_data_dir"], root),
            interim_data_dir=_resolve_path(values["interim_data_dir"], root),
            processed_data_dir=_resolve_path(values["processed_data_dir"], root),
            output_dir=_resolve_path(values["output_dir"], root),
            random_seed=int(values["random_seed"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid config {path}: {exc}") from exc

    if config.start_date > config.end_date:
        raise ValueError("start_date must be on or before end_date")
    return config


def _resolve_path(value: object, root: Path) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()
