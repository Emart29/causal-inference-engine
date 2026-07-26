"""Estimators for how the treatment effect varies across individuals.

An average effect can be positive while the treatment actively harms a subset of
the population. These learners predict an effect per unit rather than one number
for everyone, which turns "did it work?" into "who should receive it?" — usually
the question a decision-maker actually has.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from estimation.base import (
    EffectResult,
    Estimator,
    arm_sizes,
    bootstrap_ci,
    prepare_covariates,
)
from estimation.propensity import fit_propensity


def _default_learner(y: np.ndarray):
    """Return a sensible base model for the outcome type.

    Binary outcomes are handled by a classifier so predictions stay in [0, 1];
    everything else uses a regressor.
    """
    is_binary = np.array_equal(np.unique(y), np.array([0, 1]))
    if is_binary:
        return GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    return GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)


def _predict(model, x: pd.DataFrame) -> np.ndarray:
    """Predict on the outcome scale regardless of whether the model classifies."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.predict(x)


class _CATELearner(Estimator):
    """Shared machinery for learners that predict a per-unit treatment effect."""

    estimate_type = "ate"

    def __init__(self, base_learner=None, n_boot: int = 100, seed: int = 42) -> None:
        """
        Args:
            base_learner: Model used to fit outcome surfaces. Defaults to gradient
                boosting chosen by outcome type.
            n_boot: Bootstrap resamples used for the interval around the average.
            seed: Random seed for reproducibility.
        """
        self.base_learner = base_learner
        self.n_boot = n_boot
        self.seed = seed
        self._fitted_cate: np.ndarray | None = None

    def _fit_predict_cate(
        self, df: pd.DataFrame, treatment: str, outcome: str, covariates: list[str]
    ) -> np.ndarray:
        """Fit the learner and return its per-unit effect predictions."""
        raise NotImplementedError

    def predict_cate(
        self, df: pd.DataFrame, treatment: str, outcome: str, covariates: list[str]
    ) -> np.ndarray:
        """Return the predicted treatment effect for every unit in ``df``."""
        return self._fit_predict_cate(df, treatment, outcome, covariates)

    def estimate(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: list[str],
        **kwargs,
    ) -> EffectResult:
        """Estimate the average effect as the mean of the per-unit effects."""
        df = df.reset_index(drop=True)
        cate = self._fit_predict_cate(df, treatment, outcome, covariates)
        self._fitted_cate = cate
        point = float(np.mean(cate))

        def resample_point(sample: pd.DataFrame) -> float:
            return float(np.mean(self._fit_predict_cate(sample, treatment, outcome, covariates)))

        ci_low, ci_high, std_error = bootstrap_ci(
            resample_point, df, n_boot=self.n_boot, seed=self.seed
        )

        n_treated, n_control = arm_sizes(df, treatment)
        return EffectResult(
            method=self.name,
            estimate_type=self.estimate_type,
            point_estimate=point,
            ci_low=ci_low,
            ci_high=ci_high,
            std_error=std_error,
            n_treated=n_treated,
            n_control=n_control,
            diagnostics={
                "cate_mean": point,
                "cate_std": float(np.std(cate)),
                "cate_min": float(np.min(cate)),
                "cate_max": float(np.max(cate)),
                "pct_harmed": float(np.mean(cate < 0) * 100),
                "heterogeneity_note": (
                    f"{np.mean(cate < 0) * 100:.1f}% of units are predicted to be worse "
                    "off under treatment, which the average effect alone would hide."
                    if np.any(cate < 0)
                    else "Every unit is predicted to benefit, though by varying amounts."
                ),
            },
        )


class SLearner(_CATELearner):
    """Fit one model on covariates and treatment together.

    The effect is the difference between predictions with the treatment switched
    on and off. Simple and stable, but the model can under-use the treatment
    variable among many covariates, which shrinks estimated effects toward zero.
    """

    name = "s_learner"

    def _fit_predict_cate(
        self, df: pd.DataFrame, treatment: str, outcome: str, covariates: list[str]
    ) -> np.ndarray:
        x = prepare_covariates(df, covariates)
        y = df[outcome].to_numpy(dtype=float)
        t = df[treatment].to_numpy(dtype=float)

        design = x.copy()
        design[treatment] = t

        model = clone(self.base_learner) if self.base_learner is not None else _default_learner(y)
        model.fit(design, y)

        treated = design.copy()
        treated[treatment] = 1.0
        control = design.copy()
        control[treatment] = 0.0
        return _predict(model, treated) - _predict(model, control)


class TLearner(_CATELearner):
    """Fit separate outcome models for treated and untreated units.

    Letting each arm have its own model captures effects that vary with the
    covariates, at the cost of splitting the data and so fitting each model on
    less of it.
    """

    name = "t_learner"

    def _fit_predict_cate(
        self, df: pd.DataFrame, treatment: str, outcome: str, covariates: list[str]
    ) -> np.ndarray:
        x = prepare_covariates(df, covariates)
        y = df[outcome].to_numpy(dtype=float)
        t = df[treatment].to_numpy()

        treated_mask = t == 1
        control_mask = ~treated_mask
        if treated_mask.sum() < 2 or control_mask.sum() < 2:
            raise ValueError("Both arms need at least two units to fit separate models.")

        model_treated = clone(self.base_learner) if self.base_learner is not None else _default_learner(y)
        model_control = clone(self.base_learner) if self.base_learner is not None else _default_learner(y)

        model_treated.fit(x[treated_mask], y[treated_mask])
        model_control.fit(x[control_mask], y[control_mask])
        return _predict(model_treated, x) - _predict(model_control, x)


class XLearner(_CATELearner):
    """Impute each unit's missing potential outcome, then model the effect directly.

    Effects imputed within each arm are combined using the propensity score, which
    leans on whichever arm has more data for a given unit. This is the most
    reliable of the three when one arm is much smaller than the other.
    """

    name = "x_learner"

    def _fit_predict_cate(
        self, df: pd.DataFrame, treatment: str, outcome: str, covariates: list[str]
    ) -> np.ndarray:
        x = prepare_covariates(df, covariates)
        y = df[outcome].to_numpy(dtype=float)
        t = df[treatment].to_numpy()

        treated_mask = t == 1
        control_mask = ~treated_mask
        if treated_mask.sum() < 2 or control_mask.sum() < 2:
            raise ValueError("Both arms need at least two units to fit separate models.")

        # Stage one: outcome models per arm, as in the T-learner.
        model_treated = clone(self.base_learner) if self.base_learner is not None else _default_learner(y)
        model_control = clone(self.base_learner) if self.base_learner is not None else _default_learner(y)
        model_treated.fit(x[treated_mask], y[treated_mask])
        model_control.fit(x[control_mask], y[control_mask])

        # Stage two: impute the effect each unit would have had, using the other
        # arm's model as its counterfactual.
        imputed_treated = y[treated_mask] - _predict(model_control, x[treated_mask])
        imputed_control = _predict(model_treated, x[control_mask]) - y[control_mask]

        effect_model_treated = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        effect_model_control = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        effect_model_treated.fit(x[treated_mask], imputed_treated)
        effect_model_control.fit(x[control_mask], imputed_control)

        tau_treated = effect_model_treated.predict(x)
        tau_control = effect_model_control.predict(x)

        # Weighting by propensity favours the arm that is better represented for
        # each unit, which is what makes this robust to imbalanced arms.
        scores, _ = fit_propensity(df, treatment, covariates)
        return scores * tau_control + (1 - scores) * tau_treated


def cate_by_segment(
    cate: np.ndarray, df: pd.DataFrame, segment_col: str
) -> pd.DataFrame:
    """Summarise predicted effects within each level of a segment.

    Args:
        cate: Per-unit predicted effects, aligned to ``df``.
        df: Data containing the segment column.
        segment_col: Column identifying the segment each unit belongs to.

    Returns:
        Frame with one row per segment giving the mean effect, its standard
        error, an approximate 95% interval, the unit count, and whether the
        segment appears to be harmed by the treatment.
    """
    frame = pd.DataFrame({"segment": df[segment_col].to_numpy(), "cate": cate})
    rows = []
    for segment, group in frame.groupby("segment", sort=True):
        values = group["cate"].to_numpy()
        mean = float(values.mean())
        stderr = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        rows.append(
            {
                "segment": segment,
                "n": int(len(values)),
                "mean_effect": mean,
                "std_error": stderr,
                "ci_low": mean - 1.96 * stderr,
                "ci_high": mean + 1.96 * stderr,
                "harmed": bool(mean + 1.96 * stderr < 0),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_effect", ascending=False).reset_index(drop=True)


def uplift_curve(cate: np.ndarray, outcome: np.ndarray, treatment: np.ndarray) -> dict:
    """Measure how well the predicted effects rank units by responsiveness.

    Units are sorted by predicted effect and the cumulative gain from treating
    each prefix is compared against treating a random selection of the same size.

    Args:
        cate: Per-unit predicted effects.
        outcome: Observed outcome per unit.
        treatment: Binary treatment indicator per unit.

    Returns:
        Dictionary with the curve points, the area between the model and random
        targeting, and an interpretation of whether the ranking is useful.
    """
    order = np.argsort(-np.asarray(cate))
    y = np.asarray(outcome, dtype=float)[order]
    t = np.asarray(treatment)[order]
    n = len(y)

    fractions, gains = [], []
    for i in range(1, n + 1):
        treated_here = t[:i] == 1
        control_here = ~treated_here
        if treated_here.sum() == 0 or control_here.sum() == 0:
            gains.append(0.0)
        else:
            lift = y[:i][treated_here].mean() - y[:i][control_here].mean()
            gains.append(float(lift * i / n))
        fractions.append(i / n)

    gains_arr = np.asarray(gains)
    # Random targeting traces a straight line to the same endpoint, so the area
    # between the curves measures what the ranking adds over treating at random.
    random_line = np.linspace(0, gains_arr[-1] if len(gains_arr) else 0.0, n)
    auuc = float(np.trapezoid(gains_arr - random_line, dx=1.0 / n)) if n > 1 else 0.0

    return {
        "fractions": fractions,
        "gains": gains,
        "random_line": random_line.tolist(),
        "auuc": auuc,
        "interpretation": (
            "Targeting by predicted effect outperforms treating a random selection "
            "of the same size, so the ranking carries real signal."
            if auuc > 0
            else "Targeting by predicted effect does no better than random selection, "
            "so the model has not found usable heterogeneity."
        ),
    }


def top_k_targeting(cate: np.ndarray, k_pct: float) -> dict:
    """Compare treating only the most responsive units against treating everyone.

    Args:
        cate: Per-unit predicted effects.
        k_pct: Percentage of the population to treat, from 0 to 100.

    Returns:
        Dictionary contrasting the average effect among the selected units with
        the population average, and the share of total benefit captured.
    """
    cate = np.asarray(cate, dtype=float)
    n = len(cate)
    k = max(1, int(round(n * k_pct / 100)))

    ranked = np.sort(cate)[::-1]
    selected = ranked[:k]

    total_benefit = float(cate[cate > 0].sum()) if np.any(cate > 0) else 0.0
    captured = float(selected[selected > 0].sum()) if np.any(selected > 0) else 0.0
    captured_pct = float(captured / total_benefit * 100) if total_benefit > 0 else 0.0

    return {
        "k_pct": k_pct,
        "n_targeted": k,
        "mean_effect_targeted": float(selected.mean()),
        "mean_effect_all": float(cate.mean()),
        "pct_of_benefit_captured": captured_pct,
        "interpretation": (
            f"Treating the top {k_pct:.0f}% by predicted effect delivers an average "
            f"effect of {selected.mean():.2f} against {cate.mean():.2f} for treating "
            f"everyone, capturing {captured_pct:.0f}% of the total available benefit."
        ),
    }
