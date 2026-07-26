"""Checks on the assumptions an adjustment-based causal estimate depends on.

An effect size means nothing without them. If treated and untreated units do not
overlap, there is no comparison to make; if covariates remain imbalanced after
adjustment, the confounding was never removed. These checks run before any
estimate is reported so a reader always sees whether the number is trustworthy
before they see the number itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Conventional threshold above which a standardised mean difference is treated
#: as meaningful imbalance.
BALANCE_THRESHOLD = 0.1

#: Minimum units per arm below which estimates are too unstable to act on.
MIN_ARM_SIZE = 30


def check_overlap(treatment: np.ndarray, propensity_scores: np.ndarray) -> dict:
    """Check that treated and untreated units occupy the same propensity range.

    Where the two groups do not overlap there is no comparable unit to compare
    against, so the effect is not identifiable for those units no matter which
    estimator is used.

    Args:
        treatment: Binary treatment indicator per unit.
        propensity_scores: Estimated probability of treatment per unit.

    Returns:
        Dictionary describing the common support region, the share of units
        outside it, whether the check ``passed``, and an ``interpretation``.
    """
    treatment = np.asarray(treatment)
    propensity_scores = np.asarray(propensity_scores)

    treated_ps = propensity_scores[treatment == 1]
    control_ps = propensity_scores[treatment == 0]

    if treated_ps.size == 0 or control_ps.size == 0:
        return {
            "overlap_low": 0.0,
            "overlap_high": 0.0,
            "pct_off_support": 100.0,
            "passed": False,
            "interpretation": "One of the two groups is empty, so no comparison is possible.",
        }

    # The common support is where both groups are actually observed.
    overlap_low = max(treated_ps.min(), control_ps.min())
    overlap_high = min(treated_ps.max(), control_ps.max())

    off_support = np.mean((propensity_scores < overlap_low) | (propensity_scores > overlap_high))
    pct_off = float(off_support * 100)
    passed = overlap_high > overlap_low and pct_off < 10.0

    if not passed and overlap_high <= overlap_low:
        interpretation = (
            "Treated and untreated units do not overlap at all, so there is no "
            "comparable group and the effect cannot be estimated from this data."
        )
    elif not passed:
        interpretation = (
            f"{pct_off:.1f}% of units fall outside the region where both groups "
            "are observed, so the estimate relies on extrapolation for a "
            "substantial share of the sample."
        )
    else:
        interpretation = (
            f"Treated and untreated units overlap across propensity scores "
            f"{overlap_low:.2f} to {overlap_high:.2f}, with only {pct_off:.1f}% of "
            "units outside that range."
        )

    return {
        "overlap_low": float(overlap_low),
        "overlap_high": float(overlap_high),
        "pct_off_support": pct_off,
        "passed": bool(passed),
        "interpretation": interpretation,
    }


def check_positivity(propensity_scores: np.ndarray, eps: float = 0.05) -> dict:
    """Check that every unit had a realistic chance of either treatment arm.

    Units with a propensity near zero or one were effectively destined for one
    arm. Their counterfactual is unobservable, and in weighting estimators they
    receive extreme weights that destabilise the whole estimate.

    Args:
        propensity_scores: Estimated probability of treatment per unit.
        eps: Boundary below/above which a unit is treated as deterministic.

    Returns:
        Dictionary with counts of extreme units, whether the check ``passed``,
        and an ``interpretation``.
    """
    propensity_scores = np.asarray(propensity_scores)
    n_low = int(np.sum(propensity_scores < eps))
    n_high = int(np.sum(propensity_scores > 1 - eps))
    n_extreme = n_low + n_high
    pct = float(n_extreme / len(propensity_scores) * 100) if len(propensity_scores) else 0.0
    passed = pct < 10.0

    if passed:
        interpretation = (
            f"{pct:.1f}% of units have a near-certain treatment assignment, which "
            "is low enough not to destabilise the estimate."
        )
    else:
        interpretation = (
            f"{pct:.1f}% of units were almost certain to end up in one arm "
            f"(propensity below {eps} or above {1 - eps}). Their counterfactual is "
            "not observed, so the estimate for them is extrapolation."
        )

    return {
        "n_below_eps": n_low,
        "n_above_eps": n_high,
        "pct_extreme": pct,
        "eps": eps,
        "passed": bool(passed),
        "interpretation": interpretation,
    }


def standardized_mean_diff(
    df: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute the standardised mean difference for each covariate.

    The standardised mean difference expresses how far apart the treated and
    untreated groups are on a covariate, in pooled standard deviations, which
    makes it comparable across variables measured in different units.

    Args:
        df: Data containing the treatment and covariate columns.
        treatment: Name of the binary treatment column.
        covariates: Covariate names to evaluate.
        weights: Optional per-unit weights, as produced by a weighting estimator.

    Returns:
        Frame with one row per covariate holding the group means, the
        standardised difference, and whether it exceeds the balance threshold.
    """
    is_treated = df[treatment].to_numpy() == 1
    w = np.ones(len(df)) if weights is None else np.asarray(weights, dtype=float)

    rows = []
    for covariate in covariates:
        values = pd.to_numeric(df[covariate], errors="coerce").to_numpy(dtype=float)
        valid = ~np.isnan(values)

        treated_mask = is_treated & valid
        control_mask = (~is_treated) & valid
        if not treated_mask.any() or not control_mask.any():
            continue

        mean_treated = np.average(values[treated_mask], weights=w[treated_mask])
        mean_control = np.average(values[control_mask], weights=w[control_mask])

        var_treated = np.average((values[treated_mask] - mean_treated) ** 2, weights=w[treated_mask])
        var_control = np.average((values[control_mask] - mean_control) ** 2, weights=w[control_mask])
        pooled_sd = np.sqrt((var_treated + var_control) / 2)

        smd = 0.0 if pooled_sd == 0 else (mean_treated - mean_control) / pooled_sd
        rows.append(
            {
                "covariate": covariate,
                "mean_treated": float(mean_treated),
                "mean_control": float(mean_control),
                "smd": float(smd),
                "abs_smd": float(abs(smd)),
                "imbalanced": bool(abs(smd) > BALANCE_THRESHOLD),
            }
        )

    return pd.DataFrame(rows).sort_values("abs_smd", ascending=False).reset_index(drop=True)


def check_balance(
    df: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    weights: np.ndarray | None = None,
) -> dict:
    """Check whether adjustment has made the two groups comparable.

    Args:
        df: Data containing the treatment and covariate columns.
        treatment: Name of the binary treatment column.
        covariates: Covariate names to evaluate.
        weights: Optional per-unit weights from a weighting estimator.

    Returns:
        Dictionary with the per-covariate ``smd_table``, how many covariates
        remain imbalanced, whether the check ``passed``, and an
        ``interpretation``.
    """
    if not covariates:
        return {
            "smd_table": pd.DataFrame(),
            "n_imbalanced": 0,
            "worst_covariate": None,
            "worst_smd": 0.0,
            "passed": True,
            "interpretation": "No covariates were supplied, so there is nothing to balance.",
        }

    table = standardized_mean_diff(df, treatment, covariates, weights)
    if table.empty:
        return {
            "smd_table": table,
            "n_imbalanced": 0,
            "worst_covariate": None,
            "worst_smd": 0.0,
            "passed": True,
            "interpretation": "No numeric covariates were available to evaluate.",
        }

    n_imbalanced = int(table["imbalanced"].sum())
    worst = table.iloc[0]
    passed = n_imbalanced == 0

    stage = "after adjustment" if weights is not None else "before adjustment"
    if passed:
        interpretation = (
            f"All {len(table)} covariates are balanced {stage}, so the two groups "
            "are comparable on the measured variables."
        )
    else:
        interpretation = (
            f"{n_imbalanced} of {len(table)} covariates remain imbalanced {stage}; "
            f"'{worst['covariate']}' differs most, at {worst['smd']:.2f} pooled "
            "standard deviations."
        )

    return {
        "smd_table": table,
        "n_imbalanced": n_imbalanced,
        "worst_covariate": str(worst["covariate"]),
        "worst_smd": float(worst["smd"]),
        "passed": passed,
        "interpretation": interpretation,
    }


def check_sample_size(df: pd.DataFrame, treatment: str) -> dict:
    """Check that both treatment arms are large enough for a stable estimate.

    Args:
        df: Data containing the treatment column.
        treatment: Name of the binary treatment column.

    Returns:
        Dictionary with the size of each arm, whether the check ``passed``, and
        an ``interpretation``.
    """
    n_treated = int((df[treatment] == 1).sum())
    n_control = int((df[treatment] == 0).sum())
    smallest = min(n_treated, n_control)
    passed = smallest >= MIN_ARM_SIZE

    if passed:
        interpretation = (
            f"{n_treated} treated and {n_control} untreated units provide enough "
            "data for a stable estimate."
        )
    else:
        interpretation = (
            f"The smaller group has only {smallest} units, which is too few for a "
            "reliable estimate; the confidence interval will be very wide."
        )

    return {
        "n_treated": n_treated,
        "n_control": n_control,
        "smallest_arm": smallest,
        "passed": passed,
        "interpretation": interpretation,
    }


def run_all_checks(
    df: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    propensity_scores: np.ndarray | None = None,
) -> dict:
    """Run every assumption check and return a single verdict.

    Args:
        df: Data containing the treatment and covariate columns.
        treatment: Name of the binary treatment column.
        covariates: Covariates the estimate will adjust for.
        propensity_scores: Optional fitted propensity scores. Overlap and
            positivity are skipped when they are not supplied.

    Returns:
        Dictionary with each individual check under ``checks``, whether
        ``all_passed``, the ``blocking_failures`` that make an estimate
        untrustworthy, any ``warnings``, and a plain-English ``summary``.
    """
    checks: dict[str, dict] = {
        "sample_size": check_sample_size(df, treatment),
        "balance": check_balance(df, treatment, covariates),
    }
    if propensity_scores is not None:
        checks["overlap"] = check_overlap(df[treatment].to_numpy(), propensity_scores)
        checks["positivity"] = check_positivity(propensity_scores)

    # Overlap and sample size are structural: without them there is no comparison
    # to make. Imbalance before adjustment is expected and is only a warning,
    # since removing it is precisely what the estimators are for.
    blocking_names = {"overlap", "sample_size"}
    blocking_failures = [
        name for name, result in checks.items() if name in blocking_names and not result["passed"]
    ]
    warnings = [
        name
        for name, result in checks.items()
        if name not in blocking_names and not result["passed"]
    ]

    all_passed = not blocking_failures and not warnings
    if blocking_failures:
        summary = (
            "This data cannot support a credible causal estimate: "
            + " ".join(checks[name]["interpretation"] for name in blocking_failures)
        )
    elif warnings:
        summary = (
            "An estimate can be produced, but with caveats: "
            + " ".join(checks[name]["interpretation"] for name in warnings)
        )
    else:
        summary = "All assumption checks passed; the estimate can be interpreted as intended."

    return {
        "checks": checks,
        "all_passed": all_passed,
        "blocking_failures": blocking_failures,
        "warnings": warnings,
        "summary": summary,
    }
