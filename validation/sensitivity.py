"""How much unmeasured confounding it would take to overturn a conclusion.

Adjustment removes the confounding that was measured. Nothing in the data can
reveal what was missed, so the honest question is not whether hidden confounding
exists but how strong it would have to be to matter. A result that only survives
if no unmeasured factor has even a weak association with treatment and outcome is
fragile, whatever its confidence interval says.

These functions convert that question into a number an analyst can argue about
with domain knowledge, which is the only place the answer can come from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from estimation.base import Estimator


def e_value(point_estimate: float, ci_low: float, ci_high: float, outcome_sd: float = 1.0) -> dict:
    """Compute how strong a hidden confounder would need to be to explain the result.

    The E-value is the minimum association a confounder would need with both the
    treatment and the outcome, on the risk ratio scale, to account for the
    observed effect entirely. A value close to one means a weak unmeasured factor
    could overturn the finding; a large value means only an implausibly strong
    one could.

    Continuous effects are first converted to an approximate risk ratio, since
    the E-value is defined on that scale.

    Args:
        point_estimate: The estimated effect.
        ci_low: Lower bound of its confidence interval.
        ci_high: Upper bound of its confidence interval.
        outcome_sd: Standard deviation of the outcome, used to standardise a
            continuous effect before conversion.

    Returns:
        Dictionary with the E-value for the estimate and for the interval bound
        nearest the null, plus a plain-English ``interpretation``.
    """

    def _to_risk_ratio(effect: float) -> float:
        # Standardised mean differences map onto the risk ratio scale through
        # this conventional approximation, which keeps the E-value comparable
        # across continuous and binary outcomes.
        standardized = effect / outcome_sd if outcome_sd > 0 else effect
        return float(np.exp(0.91 * standardized))

    def _e_value_from_rr(rr: float) -> float:
        rr = max(rr, 1e-9)
        # The formula is defined for associations above one, so a protective
        # effect is inverted first and interpreted symmetrically.
        if rr < 1:
            rr = 1.0 / rr
        return float(rr + np.sqrt(rr * (rr - 1)))

    rr_point = _to_risk_ratio(point_estimate)
    e_point = _e_value_from_rr(rr_point)

    # The bound closest to no effect is what a confounder would have to reach to
    # push the interval across zero, so that is the honest figure to quote.
    nearest_bound = ci_low if abs(ci_low) < abs(ci_high) else ci_high
    crosses_null = ci_low <= 0 <= ci_high
    e_bound = 1.0 if crosses_null else _e_value_from_rr(_to_risk_ratio(nearest_bound))

    if crosses_null:
        interpretation = (
            "The confidence interval already includes no effect, so no unmeasured "
            "confounding is needed to explain the result away."
        )
    elif e_bound < 1.25:
        interpretation = (
            f"An unmeasured confounder associated with both treatment and outcome by "
            f"a risk ratio of only {e_bound:.2f} would erase this result. That is a "
            "weak association, so the finding is fragile."
        )
    elif e_bound < 2.0:
        interpretation = (
            f"An unmeasured confounder would need associations of about {e_bound:.2f} "
            "with both treatment and outcome to explain this result away, which is "
            "plausible in many settings."
        )
    else:
        interpretation = (
            f"Explaining this result away would require an unmeasured confounder "
            f"associated with both treatment and outcome by a risk ratio of "
            f"{e_bound:.2f}, which is a strong association and unlikely to have gone "
            "unnoticed."
        )

    return {
        "e_value": e_point,
        "e_value_ci": e_bound,
        "crosses_null": crosses_null,
        "interpretation": interpretation,
    }


def confounding_grid(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: list[str],
    estimator: Estimator,
    original_effect: float,
    strengths: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0),
    seed: int = 42,
) -> pd.DataFrame:
    """Re-estimate the effect under simulated confounders of increasing strength.

    A confounder is injected that influences both the treatment and the outcome,
    and the estimate is recomputed without adjusting for it. Sweeping the
    strength shows where the conclusion changes sign or loses significance.

    Args:
        df: The data the original estimate was produced from.
        treatment: Name of the treatment column.
        outcome: Name of the outcome column.
        covariates: Covariates the estimate adjusted for.
        estimator: The estimator to re-run at each strength.
        original_effect: The estimate being tested.
        strengths: Confounder strengths to evaluate.
        seed: Random seed for reproducibility.

    Returns:
        Frame with one row per strength giving the resulting estimate, how far it
        moved, and whether the direction of the conclusion survived.
    """
    rng = np.random.default_rng(seed)
    treatment_values = df[treatment].to_numpy(dtype=float)
    centred_treatment = treatment_values - treatment_values.mean()
    original_sign = np.sign(original_effect)

    rows = []
    for strength in strengths:
        data = df.copy()
        # The confounder is correlated with treatment and pushes the outcome in
        # the same direction, which is how real confounding inflates an estimate.
        hidden = rng.normal(0, 1, len(df)) + strength * centred_treatment
        data[outcome] = data[outcome].to_numpy(dtype=float) + strength * hidden

        try:
            result = estimator.estimate(data, treatment, outcome, covariates)
            estimate = float(result.point_estimate)
            still_significant = result.significant()
        except Exception:
            estimate = float("nan")
            still_significant = False

        rows.append(
            {
                "strength": strength,
                "estimate": estimate,
                "shift": estimate - original_effect,
                "sign_preserved": bool(np.sign(estimate) == original_sign),
                "still_significant": still_significant,
            }
        )

    return pd.DataFrame(rows)


def sensitivity_summary(
    e_value_result: dict,
    grid: pd.DataFrame | None = None,
) -> dict:
    """Combine the sensitivity evidence into a single verdict.

    Args:
        e_value_result: Output of :func:`e_value`.
        grid: Optional output of :func:`confounding_grid`.

    Returns:
        Dictionary with a ``robustness`` rating of high, moderate, or low, the
        confounder strength at which the conclusion first flips, and a
        plain-English ``interpretation``.
    """
    e_bound = e_value_result.get("e_value_ci", 1.0)
    crosses_null = e_value_result.get("crosses_null", False)

    flip_strength = None
    if grid is not None and not grid.empty:
        flipped = grid.loc[~grid["sign_preserved"]]
        if not flipped.empty:
            flip_strength = float(flipped.iloc[0]["strength"])

    if crosses_null:
        robustness = "low"
        interpretation = (
            "The estimate is not distinguishable from no effect, so there is nothing "
            "for hidden confounding to explain away."
        )
    elif e_bound >= 2.0 and flip_strength is None:
        robustness = "high"
        interpretation = (
            f"Only a strong unmeasured confounder, with associations of about "
            f"{e_bound:.2f} on both sides, could overturn this conclusion, and none "
            "of the simulated confounders reversed it."
        )
    elif e_bound >= 1.25:
        robustness = "moderate"
        detail = (
            f" A simulated confounder of strength {flip_strength:.2f} reverses the "
            "direction."
            if flip_strength is not None
            else ""
        )
        interpretation = (
            f"A moderately strong unmeasured confounder, around {e_bound:.2f}, would "
            f"be enough to explain this result away.{detail}"
        )
    else:
        robustness = "low"
        interpretation = (
            f"A weak unmeasured confounder, around {e_bound:.2f}, would erase this "
            "result, so it should not be relied on without stronger evidence that no "
            "such factor exists."
        )

    return {
        "robustness": robustness,
        "e_value_ci": e_bound,
        "flip_strength": flip_strength,
        "interpretation": interpretation,
    }
