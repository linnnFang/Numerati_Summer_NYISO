# NYISO Reliability and Outage Research

A minimal, reproducible research prototype for NYISO outage and reliability
analysis.

The primary research track asks whether a verified count of newly occurring
outage events is overdispersed and can be modeled with a Negative Binomial
model. The outcome must be a real non-negative integer event count. NYISO P-15
is forecasted generation outage MW and must never be used as an event count.

A later extension may study `DART = RT_LBMP - DA_LBMP`. Real-time prices and
DART may be consequences of outages or jointly determined with them, so they
are not assumed to cause outages.

All project timestamps use `America/New_York`; DST ambiguity must be handled
explicitly in the data-cleaning step.

## Environment

Use the existing Conda environment. Do not create a project `.venv`.

```bash
conda activate py310
python -m pip install -e .
```

## Basic commands

```bash
python -m nyiso_reliability.cli show-config --config configs/dev.yaml
python -m nyiso_reliability.cli show-tree
python -m nyiso_reliability.cli download-dev-data --config configs/dev.yaml
python -m nyiso_reliability.cli download-weather --config configs/dev.yaml
python -m nyiso_reliability.cli clean-data --config configs/dev.yaml
python -m nyiso_reliability.cli build-panels --config configs/dev.yaml
python -m nyiso_reliability.cli run-eda --config configs/dev.yaml
python -m nyiso_reliability.cli run-model --config configs/dev.yaml
python -m nyiso_reliability.cli build-report --config configs/dev.yaml
python -m compileall src tests
python -m pytest
```

## Notebook

`notebooks/nyiso_workflow.ipynb` imports and runs the same Python functions in
pipeline order, with printed summaries and saved figures. Its download cell is
disabled by default, so opening or rerunning the notebook does not automatically
make network requests.

```bash
conda activate py310
jupyter lab notebooks/nyiso_workflow.ipynb
```

To execute every enabled cell non-interactively and save its outputs:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/nyiso_workflow.ipynb
```

The download command requests every NYISO archive month intersecting the configured
date range, reuses canonical raw files that already exist, prints actual observed
fields and coverage, and writes `outputs/tables/data_download_summary.csv`.
Each monthly manifest row includes `requested_month`. Open-Meteo uses the complete
configured start/end window with UTC timestamps converted to America/New_York;
P-15 and P-14B remain latest-only snapshots. Core cleaning leaves the very large
P-54 monthly snapshots in raw storage because they are not an outage-event count
and are not consumed by the current panels.
The current development run cleans and aligns the available covariates. The
outage-count model remains blocked because no verified historical event-count
target is available. Raw data and generated outputs are ignored by Git.
