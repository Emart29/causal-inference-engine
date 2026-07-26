"""Shared contract and utilities for every effect estimator.

Estimators differ in their assumptions and in what they actually estimate, so
each result carries its estimand type explicitly. Reporting an effect on the
treated as though it were an average effect over everyone, or a local effect on
compliers as though it applied to the whole population, is a misstatement rather
than a rounding error.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

#: What each estimand actually refers to, for use in reports and interfaces.
ESTIMAND_MEANING: dict[str, str] = {
    "ate": "the average effect across everyone in the population",
    "att": "the average effect among those who actually received the treatment",
    "cate": "the effect for units with particular characteristics",
    "late": "the effect among units whose treatment was changed by the instrument",
}


@dataclass
class EffectResult:
    """One estimator's answer, with the uncertainty and diagnostics behind it.

    Attributes:
        method: Identifier of the estimator that produced the result.
        estimate_type: Which estimand this is, as a key of :data:`ESTIMAND_MEANING`.
        point_estimate: The estimated effect on the outcome's own scale.
        ci_low: Lower bound of the confidence interval.
        ci_high: Upper bound of the confidence interval.
        std_error: Standard error where the estimator provides one.
        p_value: Two-sided p-value where the estimator provides one.
        n_treated: Number of treated units the estimate is based on.
        n_control: Number of untreated units the estimate is based on.
        diagnostics: Method-specific detail, such as balance or weight statistics.
    """

    method: str
    estimate_type: str
    point_estimate: float
    ci_low: float
    ci_high: float
    n_treated: int
    n_control: int
    std_error: float | None = None
    p_value: float | None = None
    diagnostics: dict = field(default_factory=dict)

    def significant(self, alpha: float = 0.05) -> bool:
        """Return whether the interval excludes zero at the given level.

        Args:
            alpha: Significance level the interval was constructed at.
        """
        return not (self.ci_low <= 0.0 <= self.ci_high)

    @property
    def estimand_meaning(self) -> str:
        """Plain-English description of who this estimate applies to."""
        return ESTIMAND_MEANING.get(self.estimate_type, "an unspecified population")

    def to_dict(self) -> dict:
        """Serialise the result for storage or display."""
        return {
            "method": self.method,
            "estimate_type": self.estimate_type,
            "point_estimate": self.point_estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "std_error": self.std_error,
            "p_value": self.p_value,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "diagnostics": self.diagnostics,
        }


class Estimator(ABC):
    """Base class for anything that turns data into an :class:`EffectResult`."""

    #: Short identifier stored alongside results.
    name: str = "estimator"

    #: Which estimand this class of method targets.
    estimate_type: str = "ate"

    @abstractmethod
    def estimate(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: list[str],
        **kwargs,
    ) -> EffectResult:
        """Estimate the treatment effect.

        Args:
            df: Data containing the treatment, outcome, and covariate columns.
            treatment: Name of the binary treatment column.
            outcome: Name of the outcome column.
            covariates: Covariates to adjust for. These should already have been
                filtered to a safe adjustment set, since adjusting for a mediator
                or collider produces a confidently wrong answer.
            **kwargs: Method-specific options.

        Returns:
            The estimated effect with its uncertainty and diagnostics.
        """


def bootstrap_ci(
    point_fn: Callable[[pd.DataFrame], float],
    df: pd.DataFrame,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Build a percentile confidence interval by resampling the data.

    Used where an estimator has no closed-form standard error, which covers
    matching and most machine-learning-based estimators. Resamples that fail (for
    example when a bootstrap draw contains only one treatment arm) are skipped
    rather than aborting the whole interval.

    Args:
        point_fn: Function returning the point estimate for a given resample.
        df: The observed data to resample from.
        n_boot: Number of bootstrap resamples.
        alpha: Significance level, so 0.05 yields a 95% interval.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of ``(ci_low, ci_high, std_error)``. All three are ``nan`` when too
        few resamples succeeded to form an interval.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    estimates: list[float] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        resample = df.iloc[idx].reset_index(drop=True)
        try:
            value = point_fn(resample)
        except Exception:
            continue
        if value is not None and np.isfinite(value):
            estimates.append(float(value))

    if len(estimates) < max(10, n_boot // 10):
        return float("nan"), float("nan"), float("nan")

    values = np.asarray(estimates)
    ci_low = float(np.percentile(values, 100 * alpha / 2))
    ci_high = float(np.percentile(values, 100 * (1 - alpha / 2)))
    std_error = float(values.std(ddof=1))
    return ci_low, ci_high, std_error


def prepare_covariates(df: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """Convert covariates into a purely numeric design matrix.

    Categorical columns are one-hot encoded with the first level dropped to avoid
    collinearity, and any remaining missing values are filled with the column
    median so a single gap does not discard an entire row.

    Args:
        df: Data containing the covariate columns.
        covariates: Covariate names to encode.

    Returns:
        Numeric frame aligned to ``df``'s index. Empty when no covariates are given.
    """
    if not covariates:
        return pd.DataFrame(index=df.index)

    frame = df[covariates].copy()
    categorical = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if categorical:
        frame = pd.get_dummies(frame, columns=categorical, drop_first=True)

    frame = frame.astype(float)
    return frame.fillna(frame.median(numeric_only=True))


def arm_sizes(df: pd.DataFrame, treatment: str) -> tuple[int, int]:
    """Return the number of treated and untreated units."""
    return int((df[treatment] == 1).sum()), int((df[treatment] == 0).sum())
