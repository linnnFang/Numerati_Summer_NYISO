"""Build a compact local HTML report from saved, real pipeline outputs."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from nyiso_reliability.config import ProjectConfig


def _read_table(path: Path) -> pd.DataFrame | None:
    """Read a generated CSV if it exists, otherwise return no result."""
    return pd.read_csv(path) if path.is_file() else None


def _table_html(frame: pd.DataFrame | None, columns: list[str] | None = None) -> str:
    """Render a small table or an explicit not-available marker."""
    if frame is None:
        return '<p class="blocked">NOT AVAILABLE</p>'
    selected = frame[[column for column in (columns or list(frame)) if column in frame]]
    return selected.to_html(index=False, border=0, classes="dataframe")


def _figure_html(figures: Path, filename: str, caption: str) -> str:
    """Link one existing local figure relative to the report directory."""
    path = figures / filename
    if not path.is_file():
        return f'<p class="blocked">{escape(caption)}: NOT AVAILABLE</p>'
    return (
        f'<figure><img src="../figures/{escape(filename)}" alt="{escape(caption)}">'
        f"<figcaption>{escape(caption)}</figcaption></figure>"
    )


def build_report(config: ProjectConfig) -> dict[str, Any]:
    """Generate a truthful initial-analysis report from current artifacts."""
    tables = config.output_dir / "tables"
    figures = config.output_dir / "figures"
    reports = config.output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    downloads = _read_table(tables / "data_download_summary.csv")
    clean = _read_table(tables / "clean_data_summary.csv")
    panels = _read_table(tables / "panel_summary.csv")
    feasibility = _read_table(tables / "outage_target_feasibility.csv")
    model = _read_table(tables / "initial_count_model_status.csv")

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NYISO Initial Reliability Analysis</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 2rem auto;
       color: #172033; line-height: 1.55; padding: 0 1rem; }}
h1, h2 {{ color: #123a63; }}
table {{ border-collapse: collapse; width: 100%; font-size: .82rem;
         margin: 1rem 0; }}
th, td {{ border: 1px solid #d7dee8; padding: .4rem; text-align: left; }}
th {{ background: #edf3f8; }}
.blocked {{ border-left: 4px solid #b26a00; background: #fff6e5; padding: .7rem; }}
.note {{ background: #eef5ff; padding: .8rem; }}
img {{ max-width: 100%; }} figure {{ margin: 1rem 0; }}
</style>
</head>
<body>
<h1>NYISO Initial Reliability Analysis</h1>
<p class="note">Development configuration: {config.start_date} through
{config.end_date}, timezone {escape(config.timezone)}.</p>

<h2>1. Project Objective</h2>
<p>Assess whether a verified non-negative integer count of newly occurring
outage events can support an overdispersed count model. The price extension
defines DART as RT LBMP minus DA LBMP; it does not treat real-time price as a
simple causal driver of outages.</p>

<h2>2. Data Sources</h2>
{_table_html(downloads, ['dataset', 'source', 'status', 'rows', 'date_min', 'date_max'])}

<h2>3. Data Coverage and Quality</h2>
{_table_html(downloads, ['dataset', 'unique_zones', 'duplicates', 'blocker', 'notes'])}

<h2>4. Cleaning and Standardization</h2>
<p>Observed timestamps were localized to America/New_York. Ambiguous and
nonexistent naive DST clock times raise errors rather than being guessed.</p>
{_table_html(clean, ['table', 'rows', 'date_min', 'date_max', 'zones',
                     'duplicate_count', 'validation_status'])}
{_table_html(panels, ['panel', 'rows', 'date_min', 'date_max', 'grain', 'zones',
                      'merge_match_rate', 'target_status'])}

<h2>5. Outage Target Feasibility</h2>
<p class="blocked"><strong>BLOCKED.</strong> P-15 is forecasted generation
outage MW, not event count. P-14B is latest-only and lacks zone; inspected P-54
snapshots do not provide stable event identity plus complete timing. A verified
historical new-outage count cannot currently be constructed.</p>
{_table_html(feasibility)}

<h2>6. EDA Results</h2>
{_figure_html(figures, 'load_timeseries.png', 'Development-sample load')}
{_figure_html(figures, 'p15_generation_outage_mw.png',
              'P-15 current forecast MW snapshot — not event count')}

<h2>7. Overdispersion Results</h2>
<p class="blocked">NOT AVAILABLE. Mean, variance, variance-to-mean ratio, and
zero share require a valid outage count target. No substitute target was used.</p>

<h2>8. Initial Poisson vs Negative Binomial Comparison</h2>
{_table_html(model)}
<p>IRR would be interpreted as exp(beta), the multiplicative change in expected
count per one-unit covariate increase, conditional on the model. No IRRs are
reported because the count model is blocked.</p>

<h2>9. Limitations</h2>
<ul>
<li>Forced versus scheduled status is not consistently verified across sources.</li>
<li>Facility-to-zone mapping is unavailable in the inspected outage files.</li>
<li>P-14B is a current snapshot, not historical vintages.</li>
<li>The weather sample is one Albany coordinate, not zone-level exposure.</li>
<li>Market rules and reporting schemas may change over time.</li>
<li>Correlated outages are not fully handled by a simple Negative Binomial model.</li>
<li>RT LBMP and DART have reverse-causality and simultaneity risks.</li>
</ul>

<h2>10. Next Steps</h2>
<p>Seek a versioned event-level outage source with stable event ID, start time,
forced/scheduled definition, affected MW, and effective facility-to-zone mapping.
Until then, retain the valid load, price, weather, and outage-MW pipelines while
keeping the primary count-model branch blocked.</p>
</body>
</html>"""
    report_path = reports / "nyiso_initial_analysis.html"
    report_path.write_text(html, encoding="utf-8")
    return {
        "report_path": str(report_path),
        "tables": len(list(tables.glob("*.csv"))),
        "figures": len(list(figures.glob("*.png"))),
        "blocker": "Verified historical outage event count is unavailable.",
    }
