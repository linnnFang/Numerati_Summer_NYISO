"""Initial Poisson/NB comparison, gated on a verified count target."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from nyiso_reliability.config import ProjectConfig


def chronological_split(
    df: pd.DataFrame, train_share: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ordered dates without shuffling or leaking future observations."""
    if not 0 < train_share < 1:
        raise ValueError("train_share must be between zero and one")
    dates = sorted(df["interval_start"].drop_duplicates())
    if len(dates) < 2:
        raise ValueError("At least two distinct dates are required")
    split_index = min(max(int(len(dates) * train_share), 1), len(dates) - 1)
    cutoff = dates[split_index]
    return (
        df[df["interval_start"] < cutoff].copy(),
        df[df["interval_start"] >= cutoff].copy(),
    )


def fit_poisson(df: pd.DataFrame, formula: str) -> Any:
    """Fit a Poisson GLM for a verified count response."""
    return smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit()


def fit_negative_binomial(df: pd.DataFrame, formula: str) -> Any:
    """Fit a discrete NB2 model that estimates its dispersion parameter."""
    return smf.negativebinomial(formula=formula, data=df).fit(disp=False)


def _pearson_dispersion(model: Any) -> float:
    """Compute Pearson residual dispersion when residuals are available."""
    residuals = np.asarray(model.resid_pearson)
    return float(np.sum(residuals**2) / model.df_resid)


def compare_models(poisson_result: Any, nb_result: Any) -> pd.DataFrame:
    """Compare convergence, likelihood, AIC, and Pearson dispersion."""
    return pd.DataFrame(
        [
            {
                "model": "Poisson",
                "converged": bool(poisson_result.converged),
                "log_likelihood": float(poisson_result.llf),
                "aic": float(poisson_result.aic),
                "pearson_dispersion": _pearson_dispersion(poisson_result),
                "alpha": None,
            },
            {
                "model": "Negative Binomial",
                "converged": bool(nb_result.mle_retvals.get("converged", False)),
                "log_likelihood": float(nb_result.llf),
                "aic": float(nb_result.aic),
                "pearson_dispersion": _pearson_dispersion(nb_result),
                "alpha": float(nb_result.params.get("alpha", np.nan)),
            },
        ]
    )


def build_irr_table(model: Any) -> pd.DataFrame:
    """Return coefficients and incidence-rate ratios exp(beta)."""
    confidence = model.conf_int()
    return pd.DataFrame(
        {
            "term": model.params.index,
            "coefficient": model.params.values,
            "irr": np.exp(model.params.values),
            "irr_ci_low": np.exp(confidence.iloc[:, 0].values),
            "irr_ci_high": np.exp(confidence.iloc[:, 1].values),
        }
    )


def evaluate_model(model: Any, test_df: pd.DataFrame, target: str) -> dict[str, float]:
    """Compute held-out MAE and RMSE from chronological predictions."""
    prediction = np.asarray(model.predict(test_df))
    actual = test_df[target].to_numpy()
    error = prediction - actual
    return {
        "test_mae": float(np.mean(np.abs(error))),
        "test_rmse": float(np.sqrt(np.mean(error**2))),
    }


def _save_blocked(output_dir: Path, reason: str) -> dict[str, Any]:
    """Persist a model blocker instead of fitting a model to an invalid target."""
    output_dir.mkdir(parents=True, exist_ok=True)
    status = {"model_status": "BLOCKED", "reason": reason}
    pd.DataFrame([status]).to_csv(output_dir / "initial_count_model_status.csv", index=False)
    return status


def run_initial_models(config: ProjectConfig) -> dict[str, Any]:
    """Fit initial count models only when the verified target is present."""
    panel = pd.read_parquet(config.processed_data_dir / "daily_analysis_panel.parquet")
    target = "new_outage_count"
    output = config.output_dir / "tables"
    model_dir = config.output_dir / "reports"
    if target not in panel:
        return _save_blocked(
            output,
            "No verified historical outage event count. P-15 MW is not a count.",
        )
    values = panel[target].dropna()
    if (values < 0).any() or not np.allclose(values, np.round(values)):
        return _save_blocked(output, "Target is not a non-negative integer count.")

    train, test = chronological_split(panel.dropna(subset=[target]))
    formula = f"{target} ~ 1"
    poisson = fit_poisson(train, formula)
    negative_binomial = fit_negative_binomial(train, formula)
    comparison = compare_models(poisson, negative_binomial)
    poisson_metrics = evaluate_model(poisson, test, target)
    nb_metrics = evaluate_model(negative_binomial, test, target)
    comparison.loc[comparison["model"] == "Poisson", list(poisson_metrics)] = list(
        poisson_metrics.values()
    )
    comparison.loc[
        comparison["model"] == "Negative Binomial", list(nb_metrics)
    ] = list(nb_metrics.values())
    comparison.to_csv(output / "initial_count_model_comparison.csv", index=False)
    build_irr_table(negative_binomial).to_csv(output / "initial_nb_irr.csv", index=False)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "poisson_summary.txt").write_text(
        poisson.summary().as_text(), encoding="utf-8"
    )
    (model_dir / "negative_binomial_summary.txt").write_text(
        negative_binomial.summary().as_text(), encoding="utf-8"
    )
    return {"model_status": "READY", "comparison_rows": len(comparison)}
