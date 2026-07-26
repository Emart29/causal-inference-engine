"""Instrumental variable estimation for unmeasured confounding.

Adjustment can only remove confounding that was measured. When an unobserved
factor drives both the treatment and the outcome, no set of controls will fix
the bias, and the only way forward is a variable that shifts the treatment
without touching the outcome through any other route.

That variable is the instrument, and it buys a narrower answer than adjustment
does: the effect among the units whose treatment the instrument actually
changed, rather than the effect across everyone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from estimation.base import EffectResult, Estimator, arm_sizes

#: First-stage F below this is the conventional threshold for a weak instrument,
#: where two-stage least squares becomes badly biased and unreliable.
WEAK_INSTRUMENT_F = 10.0


class InstrumentalVariables(Estimator):
    """Estimate an effect through two-stage least squares.

    The first stage predicts the treatment from the instrument; the second stage
    regresses the outcome on those predictions. Because the predicted treatment
    varies only with the instrument, it is uncorrelated with the unmeasured
    confounder, which is what removes the bias.
    """

    name = "iv"
    estimate_type = "late"

    def estimate(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: list[str],
        instrument: str | None = None,
        **kwargs,
    ) -> EffectResult:
        """Estimate the local effect among units the instrument moved.

        Args:
            df: Data containing the treatment, outcome, instrument, and controls.
            treatment: Name of the treatment column.
            outcome: Name of the outcome column.
            covariates: Controls included in both stages.
            instrument: Name of the instrument column.
            **kwargs: Unused; accepted for interface compatibility.

        Returns:
            The estimated effect, with first-stage strength in its diagnostics.

        Raises:
            ValueError: If no instrument is supplied.
        """
        if instrument is None:
            raise ValueError(
                "Instrumental variable estimation needs an instrument: a variable that "
                "shifts the treatment but has no other route to the outcome."
            )

        data = df.copy()
        data["_y"] = pd.to_numeric(data[outcome], errors="coerce")
        data["_t"] = pd.to_numeric(data[treatment], errors="coerce")
        data["_z"] = pd.to_numeric(data[instrument], errors="coerce")
        data = data.dropna(subset=["_y", "_t", "_z"])

        numeric_controls = [
            c for c in covariates if c in data.columns and pd.api.types.is_numeric_dtype(data[c])
        ]
        control_terms = (" + " + " + ".join(numeric_controls)) if numeric_controls else ""

        strength = check_instrument_strength(data, "_t", "_z", numeric_controls)

        # First stage: the part of the treatment driven by the instrument alone.
        first_stage = smf.ols(f"_t ~ _z{control_terms}", data=data).fit()
        data["_t_hat"] = first_stage.fittedvalues

        # Second stage: regressing on the predicted treatment isolates variation
        # that cannot be contaminated by the unmeasured confounder.
        second_stage = smf.ols(f"_y ~ _t_hat{control_terms}", data=data).fit()

        point = float(second_stage.params["_t_hat"])
        # The conventional second-stage errors understate uncertainty because they
        # ignore that the treatment was itself estimated, so they are rescaled by
        # the first stage's explanatory power.
        raw_se = float(second_stage.bse["_t_hat"])
        correction = 1.0 / max(np.sqrt(max(first_stage.rsquared, 1e-6)), 1e-6)
        std_error = raw_se * min(correction, 10.0)
        ci_low = point - 1.96 * std_error
        ci_high = point + 1.96 * std_error

        n_treated = int((data["_t"] == 1).sum())
        n_control = int((data["_t"] == 0).sum())

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
                "instrument": instrument,
                "first_stage_f": strength["f_stat"],
                "is_weak": strength["is_weak"],
                "first_stage_r2": float(first_stage.rsquared),
                "instrument_coefficient": float(first_stage.params["_z"]),
                "strength_interpretation": strength["interpretation"],
                "exclusion_restriction": check_exclusion_restriction(instrument, outcome),
                "estimand_note": (
                    "This is the effect among units whose treatment status was changed "
                    "by the instrument, not the effect across the whole population."
                ),
            },
        )


def check_instrument_strength(
    df: pd.DataFrame, treatment: str, instrument: str, covariates: list[str]
) -> dict:
    """Measure how strongly the instrument shifts the treatment.

    A weak instrument leaves the second stage relying on very little genuine
    variation, which inflates both bias and variance. The conventional guide is
    that a first-stage F below ten is unusable.

    Args:
        df: Data containing the treatment, instrument, and controls.
        treatment: Name of the treatment column.
        instrument: Name of the instrument column.
        covariates: Controls included in the first stage.

    Returns:
        Dictionary with the F statistic, whether the instrument ``is_weak``, and
        an ``interpretation``.
    """
    control_terms = (" + " + " + ".join(covariates)) if covariates else ""
    model = smf.ols(f"{treatment} ~ {instrument}{control_terms}", data=df).fit()

    # The F statistic on the instrument alone is what the threshold refers to.
    t_stat = float(model.tvalues[instrument])
    f_stat = t_stat**2
    is_weak = f_stat < WEAK_INSTRUMENT_F

    if is_weak:
        interpretation = (
            f"The instrument is weak (first-stage F = {f_stat:.1f}, below the "
            f"conventional threshold of {WEAK_INSTRUMENT_F:.0f}). It barely moves the "
            "treatment, so the estimate is unreliable and should not be acted on."
        )
    else:
        interpretation = (
            f"The instrument is strong enough to use (first-stage F = {f_stat:.1f}), "
            "meaning it shifts treatment assignment substantially."
        )

    return {
        "f_stat": f_stat,
        "is_weak": is_weak,
        "coefficient": float(model.params[instrument]),
        "interpretation": interpretation,
    }


def check_exclusion_restriction(instrument: str, outcome: str) -> dict:
    """State the assumption that cannot be tested from the data.

    The instrument must affect the outcome only through the treatment. No
    statistical test can confirm this, because a violation would be
    indistinguishable from a genuine effect. It has to be argued from knowledge
    of how the instrument came about, so the engine records it as an explicit
    claim rather than quietly assuming it.

    Args:
        instrument: Name of the instrument.
        outcome: Name of the outcome.

    Returns:
        Dictionary describing the assumption and prompting the analyst to justify it.
    """
    return {
        "testable": False,
        "assumption": (
            f"'{instrument}' affects '{outcome}' only by changing the treatment, and "
            "has no other route to it."
        ),
        "interpretation": (
            "This assumption cannot be checked against the data: a violation would "
            "look exactly like a real effect. It must be argued from how the "
            "instrument arose, and the estimate is only as credible as that argument."
        ),
        "justification_required": True,
    }
