"""Difference-in-differences estimation for policy-style interventions.

When a treatment switches on at a known moment for one group but not another,
the change in the treated group can be compared against the change in the
untreated group over the same period. Any permanent difference between the two
groups cancels out, which is what makes the design useful when the groups were
never comparable to begin with.

The design rests entirely on the treated group having tracked the untreated
group before the intervention. That assumption is testable on the pre-treatment
periods, and this module tests it rather than assuming it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from estimation.base import EffectResult, Estimator, arm_sizes


class DifferenceInDifferences(Estimator):
    """Estimate an effect from the change in a treated group relative to a control.

    Fits a two-way model with group and period indicators; the coefficient on
    their interaction is the effect. Standard errors are clustered by unit,
    because repeated observations of the same unit are not independent and
    ignoring that overstates precision dramatically.
    """

    name = "did"
    estimate_type = "att"

    def __init__(self, unit_col: str | None = None) -> None:
        """
        Args:
            unit_col: Column identifying the repeated unit, used to cluster
                standard errors. Falls back to conventional errors when absent.
        """
        self.unit_col = unit_col

    def estimate(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: list[str],
        group_col: str | None = None,
        time_col: str | None = None,
        treat_period: int | None = None,
        **kwargs,
    ) -> EffectResult:
        """Estimate the effect of a treatment that begins at a known period.

        Args:
            df: Panel data in long format, one row per unit and period.
            treatment: Name of the column marking treated observations. Rebuilt
                from ``group_col`` and ``treat_period`` when both are supplied.
            outcome: Name of the outcome column.
            covariates: Additional controls to include in the model.
            group_col: Column marking which units are ever treated.
            time_col: Column identifying the period.
            treat_period: First period in which the treatment applies.
            **kwargs: Unused; accepted for interface compatibility.

        Returns:
            The estimated effect on the treated group after the intervention.

        Raises:
            ValueError: If the columns needed to identify the design are missing.
        """
        if group_col is None or time_col is None:
            raise ValueError(
                "Difference-in-differences needs a group column and a time column to "
                "identify who was treated and when."
            )

        data = df.copy()
        if treat_period is not None:
            data["_post"] = (data[time_col] >= treat_period).astype(int)
        elif "post" in data.columns:
            data["_post"] = data["post"].astype(int)
        else:
            raise ValueError(
                "The period in which treatment begins must be supplied, or the data "
                "must already contain a 'post' indicator."
            )

        data["_group"] = data[group_col].astype(int)
        data["_y"] = pd.to_numeric(data[outcome], errors="coerce")
        data = data.dropna(subset=["_y"])

        terms = ["_group", "_post", "_group:_post"]
        # Only numeric controls are added, since categorical handling belongs to
        # the caller's adjustment set rather than the formula.
        numeric_controls = [
            c for c in covariates if c in data.columns and pd.api.types.is_numeric_dtype(data[c])
        ]
        terms.extend(numeric_controls)
        formula = f"_y ~ {' + '.join(terms)}"

        if self.unit_col and self.unit_col in data.columns:
            model = smf.ols(formula, data=data).fit(
                cov_type="cluster", cov_kwds={"groups": data[self.unit_col]}
            )
            se_note = f"standard errors clustered by {self.unit_col}"
        else:
            model = smf.ols(formula, data=data).fit()
            se_note = "conventional standard errors; no unit column was supplied to cluster on"

        interaction = "_group:_post"
        point = float(model.params[interaction])
        std_error = float(model.bse[interaction])
        p_value = float(model.pvalues[interaction])
        ci_low, ci_high = (float(v) for v in model.conf_int().loc[interaction])

        trends = parallel_trends_test(data, outcome, time_col, "_group", treat_period)
        n_treated, n_control = arm_sizes(data.assign(**{treatment: data["_group"] * data["_post"]}), treatment)

        return EffectResult(
            method=self.name,
            estimate_type=self.estimate_type,
            point_estimate=point,
            ci_low=ci_low,
            ci_high=ci_high,
            std_error=std_error,
            p_value=p_value,
            n_treated=n_treated,
            n_control=n_control,
            diagnostics={
                "n_periods": int(data[time_col].nunique()),
                "n_units": int(data[self.unit_col].nunique()) if self.unit_col in data.columns else None,
                "treat_period": treat_period,
                "standard_errors": se_note,
                "parallel_trends": trends,
                "r_squared": float(model.rsquared),
            },
        )


def parallel_trends_test(
    df: pd.DataFrame,
    outcome: str,
    time_col: str,
    group_col: str,
    treat_period: int | None,
) -> dict:
    """Test whether the two groups moved together before the treatment began.

    Fits group-specific time trends on the pre-treatment periods only and tests
    whether they differ. A significant difference means the groups were already
    diverging, so any post-treatment gap cannot be attributed to the treatment.

    Args:
        df: Panel data in long format.
        outcome: Name of the outcome column.
        time_col: Column identifying the period.
        group_col: Column marking which units are ever treated.
        treat_period: First treated period; only earlier periods are used.

    Returns:
        Dictionary with the difference in pre-treatment slopes, its p-value,
        whether the assumption ``passed``, and an ``interpretation``.
    """
    if treat_period is None:
        return {
            "passed": False,
            "pre_trend_diff": float("nan"),
            "p_value": float("nan"),
            "interpretation": "The treatment period is unknown, so pre-treatment trends cannot be compared.",
        }

    pre = df[df[time_col] < treat_period].copy()
    if pre[time_col].nunique() < 3:
        return {
            "passed": False,
            "pre_trend_diff": float("nan"),
            "p_value": float("nan"),
            "interpretation": (
                "Fewer than three pre-treatment periods are available, which is too "
                "few to judge whether the groups were moving together."
            ),
        }

    pre["_y"] = pd.to_numeric(pre[outcome], errors="coerce")
    pre["_g"] = pre[group_col].astype(int)
    pre["_t"] = pre[time_col].astype(float)
    pre = pre.dropna(subset=["_y"])

    # A significant group-by-time interaction before treatment means the groups
    # were already drifting apart, which invalidates the design.
    model = smf.ols("_y ~ _g + _t + _g:_t", data=pre).fit()
    slope_diff = float(model.params["_g:_t"])
    p_value = float(model.pvalues["_g:_t"])
    passed = p_value > 0.05

    if passed:
        interpretation = (
            f"The two groups moved together before the treatment: their trends "
            f"differ by only {slope_diff:.3f} per period (p = {p_value:.2f}), which "
            "supports attributing the later gap to the treatment."
        )
    else:
        interpretation = (
            f"The two groups were already diverging before the treatment, by "
            f"{slope_diff:.3f} per period (p = {p_value:.3f}). The later difference "
            "cannot be attributed to the treatment on this evidence."
        )

    return {
        "passed": passed,
        "pre_trend_diff": slope_diff,
        "p_value": p_value,
        "n_pre_periods": int(pre[time_col].nunique()),
        "interpretation": interpretation,
    }


def event_study(
    df: pd.DataFrame,
    outcome: str,
    time_col: str,
    group_col: str,
    treat_period: int,
) -> pd.DataFrame:
    """Estimate the group gap in every period relative to the one before treatment.

    Flat differences before the treatment and a jump afterwards is the visual
    evidence that the design holds; a drift beforehand is evidence that it does not.

    Args:
        df: Panel data in long format.
        outcome: Name of the outcome column.
        time_col: Column identifying the period.
        group_col: Column marking which units are ever treated.
        treat_period: First treated period.

    Returns:
        Frame with one row per period holding the estimated gap, an approximate
        95% interval, and whether the period precedes the treatment.
    """
    data = df.copy()
    data["_y"] = pd.to_numeric(data[outcome], errors="coerce")
    data["_g"] = data[group_col].astype(int)
    baseline = treat_period - 1

    rows = []
    for period in sorted(data[time_col].unique()):
        window = data[data[time_col] == period]
        treated = window.loc[window["_g"] == 1, "_y"].to_numpy()
        control = window.loc[window["_g"] == 0, "_y"].to_numpy()
        if treated.size == 0 or control.size == 0:
            continue

        gap = float(treated.mean() - control.mean())
        stderr = float(
            np.sqrt(treated.var(ddof=1) / treated.size + control.var(ddof=1) / control.size)
        )
        rows.append(
            {
                "period": period,
                "gap": gap,
                "std_error": stderr,
                "ci_low": gap - 1.96 * stderr,
                "ci_high": gap + 1.96 * stderr,
                "is_pre_treatment": bool(period < treat_period),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    # Expressing every period relative to the last pre-treatment period is the
    # convention that makes a pre-treatment drift visible at a glance.
    baseline_rows = frame.loc[frame["period"] == baseline, "gap"]
    if not baseline_rows.empty:
        offset = float(baseline_rows.iloc[0])
        for column in ("gap", "ci_low", "ci_high"):
            frame[column] = frame[column] - offset

    return frame
