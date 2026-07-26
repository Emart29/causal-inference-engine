"""Estimators that adjust for confounding through the propensity score.

The propensity score is the probability of receiving treatment given the observed
covariates. Comparing units with similar scores approximates the comparison an
experiment would have made, provided every relevant confounder was measured.
That proviso is the method's entire weakness: nothing here can correct for a
confounder that is absent from the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from estimation.base import (
    EffectResult,
    Estimator,
    arm_sizes,
    bootstrap_ci,
    prepare_covariates,
)
from discovery.assumption_check import check_balance

#: Above this AUC the treatment is almost perfectly predictable from covariates,
#: which means treated and untreated units barely overlap.
SEPARATION_AUC = 0.95


def fit_propensity(
    df: pd.DataFrame, treatment: str, covariates: list[str]
) -> tuple[np.ndarray, dict]:
    """Estimate each unit's probability of receiving treatment.

    Args:
        df: Data containing the treatment and covariate columns.
        treatment: Name of the binary treatment column.
        covariates: Covariates believed to drive treatment assignment.

    Returns:
        Tuple of the fitted scores and a diagnostics dictionary. A high ``auc``
        is a warning rather than a success: it means treatment is nearly
        deterministic given the covariates, leaving no comparable units.
    """
    y = df[treatment].to_numpy()
    x = prepare_covariates(df, covariates)

    if x.shape[1] == 0:
        # With no covariates the best estimate is the overall treatment rate,
        # which is the randomised-experiment case.
        rate = float(y.mean())
        scores = np.full(len(df), rate)
        return scores, {"auc": 0.5, "n_features": 0, "near_separation": False}

    scaled = StandardScaler().fit_transform(x)
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(scaled, y)
    scores = model.predict_proba(scaled)[:, 1]

    auc = float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else 0.5
    return scores, {
        "auc": auc,
        "n_features": int(x.shape[1]),
        "near_separation": auc > SEPARATION_AUC,
        "note": (
            "Treatment is almost perfectly predictable from the covariates, so "
            "treated and untreated units barely overlap and the estimate leans "
            "heavily on extrapolation."
            if auc > SEPARATION_AUC
            else "Treatment assignment is not fully determined by the covariates, "
            "so comparable units exist in both arms."
        ),
    }


class PropensityScoreMatching(Estimator):
    """Match each treated unit to the nearest untreated unit by propensity score.

    Matching estimates the effect among the treated, because it builds a
    comparison group for the units that actually received the treatment. Treated
    units with no acceptable match are dropped, which is honest but narrows the
    population the answer applies to.
    """

    name = "psm"
    estimate_type = "att"

    def __init__(self, caliper_sd: float = 0.2, n_boot: int = 200, seed: int = 42) -> None:
        """
        Args:
            caliper_sd: Maximum allowed distance between matched units, in
                standard deviations of the logit propensity score.
            n_boot: Bootstrap resamples used for the confidence interval.
            seed: Random seed for reproducibility.
        """
        self.caliper_sd = caliper_sd
        self.n_boot = n_boot
        self.seed = seed

    def _match(
        self, df: pd.DataFrame, treatment: str, scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        """Pair treated units with their nearest untreated match within the caliper.

        Returns:
            Tuple of matched treated indices, matched control indices, the caliper
            used, and how many treated units found no match.
        """
        # Matching on the logit scale spaces out scores near 0 and 1, where raw
        # probabilities compress genuinely different units together.
        clipped = np.clip(scores, 1e-6, 1 - 1e-6)
        logit = np.log(clipped / (1 - clipped))
        caliper = self.caliper_sd * float(np.std(logit))

        treated_idx = np.flatnonzero(df[treatment].to_numpy() == 1)
        control_idx = np.flatnonzero(df[treatment].to_numpy() == 0)
        control_logit = logit[control_idx]

        matched_treated: list[int] = []
        matched_control: list[int] = []
        unmatched = 0

        for i in treated_idx:
            distances = np.abs(control_logit - logit[i])
            nearest = int(np.argmin(distances))
            if caliper > 0 and distances[nearest] > caliper:
                unmatched += 1
                continue
            matched_treated.append(int(i))
            matched_control.append(int(control_idx[nearest]))

        return np.asarray(matched_treated), np.asarray(matched_control), caliper, unmatched

    def estimate(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: list[str],
        **kwargs,
    ) -> EffectResult:
        """Estimate the effect on the treated by nearest-neighbour matching."""
        df = df.reset_index(drop=True)
        scores, ps_diag = fit_propensity(df, treatment, covariates)
        treated_idx, control_idx, caliper, unmatched = self._match(df, treatment, scores)

        if treated_idx.size == 0:
            raise ValueError(
                "No treated unit had a comparable untreated match within the caliper, "
                "so an effect cannot be estimated from this data."
            )

        y = df[outcome].to_numpy(dtype=float)
        point = float(np.mean(y[treated_idx] - y[control_idx]))

        def resample_point(sample: pd.DataFrame) -> float:
            sample_scores, _ = fit_propensity(sample, treatment, covariates)
            t_idx, c_idx, _, _ = self._match(sample, treatment, sample_scores)
            if t_idx.size == 0:
                return float("nan")
            sample_y = sample[outcome].to_numpy(dtype=float)
            return float(np.mean(sample_y[t_idx] - sample_y[c_idx]))

        ci_low, ci_high, std_error = bootstrap_ci(
            resample_point, df, n_boot=self.n_boot, seed=self.seed
        )

        matched = pd.concat(
            [df.iloc[treated_idx], df.iloc[control_idx]], ignore_index=True
        )
        balance_after = check_balance(matched, treatment, covariates)
        balance_before = check_balance(df, treatment, covariates)

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
                "n_matched_pairs": int(treated_idx.size),
                "n_unmatched_treated": int(unmatched),
                "caliper": float(caliper),
                "propensity": ps_diag,
                "n_imbalanced_before": balance_before["n_imbalanced"],
                "n_imbalanced_after": balance_after["n_imbalanced"],
                "balance_interpretation": balance_after["interpretation"],
            },
        )


class InverseProbabilityWeighting(Estimator):
    """Reweight units so the covariate distributions of both arms coincide.

    Each unit is weighted by the inverse of its probability of receiving the arm
    it actually received, which reconstructs the population that a randomised
    experiment would have produced. Weights are stabilised and trimmed, because a
    single unit with a near-zero propensity can otherwise dominate the estimate.
    """

    name = "ipw"
    estimate_type = "ate"

    def __init__(
        self,
        stabilized: bool = True,
        trim: tuple[float, float] = (0.01, 0.99),
        n_boot: int = 200,
        seed: int = 42,
    ) -> None:
        """
        Args:
            stabilized: Multiply weights by the marginal treatment probability,
                which reduces variance without changing what is estimated.
            trim: Propensity bounds outside which scores are clipped.
            n_boot: Bootstrap resamples used for the confidence interval.
            seed: Random seed for reproducibility.
        """
        self.stabilized = stabilized
        self.trim = trim
        self.n_boot = n_boot
        self.seed = seed

    def _weights(self, treatment_values: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Compute per-unit weights from propensity scores."""
        low, high = self.trim
        clipped = np.clip(scores, low, high)
        weights = np.where(treatment_values == 1, 1.0 / clipped, 1.0 / (1.0 - clipped))
        if self.stabilized:
            rate = float(treatment_values.mean())
            weights = np.where(treatment_values == 1, weights * rate, weights * (1 - rate))
        return weights

    def _weighted_effect(
        self, df: pd.DataFrame, treatment: str, outcome: str, covariates: list[str]
    ) -> tuple[float, np.ndarray, dict]:
        """Return the weighted mean difference along with the weights used."""
        scores, ps_diag = fit_propensity(df, treatment, covariates)
        t = df[treatment].to_numpy()
        y = df[outcome].to_numpy(dtype=float)
        weights = self._weights(t, scores)

        treated_mask = t == 1
        control_mask = ~treated_mask
        if not treated_mask.any() or not control_mask.any():
            return float("nan"), weights, ps_diag

        mean_treated = np.average(y[treated_mask], weights=weights[treated_mask])
        mean_control = np.average(y[control_mask], weights=weights[control_mask])
        return float(mean_treated - mean_control), weights, ps_diag

    def estimate(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: list[str],
        **kwargs,
    ) -> EffectResult:
        """Estimate the average effect by inverse probability weighting."""
        df = df.reset_index(drop=True)
        point, weights, ps_diag = self._weighted_effect(df, treatment, outcome, covariates)

        def resample_point(sample: pd.DataFrame) -> float:
            value, _, _ = self._weighted_effect(sample, treatment, outcome, covariates)
            return value

        ci_low, ci_high, std_error = bootstrap_ci(
            resample_point, df, n_boot=self.n_boot, seed=self.seed
        )

        # The effective sample size falls as weights become uneven; a small value
        # means the estimate rests on relatively few influential units.
        effective_n = float(weights.sum() ** 2 / np.sum(weights**2))
        balance_after = check_balance(df, treatment, covariates, weights=weights)
        balance_before = check_balance(df, treatment, covariates)

        scores, _ = fit_propensity(df, treatment, covariates)
        low, high = self.trim
        n_trimmed = int(np.sum((scores < low) | (scores > high)))

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
                "effective_sample_size": effective_n,
                "effective_sample_pct": float(effective_n / len(df) * 100),
                "max_weight": float(weights.max()),
                "n_trimmed": n_trimmed,
                "stabilized": self.stabilized,
                "propensity": ps_diag,
                "n_imbalanced_before": balance_before["n_imbalanced"],
                "n_imbalanced_after": balance_after["n_imbalanced"],
                "balance_interpretation": balance_after["interpretation"],
            },
        )
