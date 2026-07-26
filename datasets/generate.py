"""Synthetic datasets with known causal effects.

Every generator returns the data *and* the ground truth that produced it. Because
the true effect is known by construction, each estimator can be scored on whether
it actually recovers the right answer — which is the only way to demonstrate that
a causal pipeline works, since real observational data never comes with a label
saying what the true effect was.

Each generator returns ``(df, ground_truth)`` where ``ground_truth`` always
contains ``true_ate``, the biased ``naive_estimate`` a difference-in-means would
produce, and a short explanation of why the naive number misleads.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def _naive_difference(df: pd.DataFrame, treatment: str, outcome: str) -> float:
    """Difference in mean outcome between treated and untreated units.

    This is the number a naive analysis reports. It equals the causal effect only
    when treatment is independent of every cause of the outcome (i.e. randomised).
    """
    treated = df.loc[df[treatment] == 1, outcome].mean()
    control = df.loc[df[treatment] == 0, outcome].mean()
    return float(treated - control)


def make_confounded(
    n: int = 5000,
    true_ate: float = 2.0,
    confounding_strength: float = 1.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Generate data where a common cause biases the naive estimate.

    Structure: ``income`` and ``prior_spend`` cause both treatment assignment and
    the outcome, so treated units differ systematically from untreated ones even
    before treatment. Adjusting for those covariates recovers ``true_ate``;
    comparing raw group means does not.

    Args:
        n: Number of units to generate.
        true_ate: The true average treatment effect built into the outcome.
        confounding_strength: Multiplier on how strongly the confounders push
            both treatment assignment and the outcome. Zero reproduces a
            randomised experiment.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of the generated frame and its ground-truth description.
    """
    rng = np.random.default_rng(seed)

    income = rng.normal(50_000, 15_000, n)
    prior_spend = rng.gamma(shape=2.0, scale=250.0, size=n)
    age = rng.integers(18, 75, n).astype(float)

    # Standardised confounders keep the logit on a sane scale regardless of units.
    income_z = (income - income.mean()) / income.std()
    prior_z = (prior_spend - prior_spend.mean()) / prior_spend.std()

    # Customers who already spend more are more likely to receive the treatment.
    logit = confounding_strength * (0.9 * income_z + 1.1 * prior_z) - 0.2
    propensity = 1.0 / (1.0 + np.exp(-logit))
    treatment = rng.binomial(1, propensity)

    # The same confounders also raise the outcome directly, independent of treatment.
    noise = rng.normal(0, 1.5, n)
    outcome = (
        10.0
        + true_ate * treatment
        + confounding_strength * (2.5 * income_z + 3.0 * prior_z)
        + 0.02 * age
        + noise
    )

    df = pd.DataFrame(
        {
            "income": income,
            "prior_spend": prior_spend,
            "age": age,
            "treatment": treatment,
            "outcome": outcome,
        }
    )

    naive = _naive_difference(df, "treatment", "outcome")
    return df, {
        "true_ate": true_ate,
        "naive_estimate": naive,
        "bias": naive - true_ate,
        "treatment": "treatment",
        "outcome": "outcome",
        "confounders": ["income", "prior_spend"],
        "covariates": ["income", "prior_spend", "age"],
        "why": (
            "Customers with higher income and prior spend were more likely to be "
            "treated and would have spent more anyway, so the raw difference in "
            "means credits the treatment with their pre-existing behaviour."
        ),
    }


def make_heterogeneous(n: int = 5000, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Generate data where the treatment effect differs sharply by segment.

    Segment ``A`` responds strongly, ``B`` responds mildly, and ``C`` is harmed by
    the treatment. The average effect is positive, which hides the fact that one
    group should never be treated — the failure mode that conditional effect
    estimation exists to catch.

    Args:
        n: Number of units to generate.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of the generated frame and its ground truth, including the true
        per-segment effects and the per-unit effect column ``true_effect``.
    """
    rng = np.random.default_rng(seed)

    segment = rng.choice(["A", "B", "C"], size=n, p=[0.35, 0.4, 0.25])
    engagement = rng.beta(2, 5, n)
    tenure_months = rng.integers(1, 60, n).astype(float)

    segment_effect = {"A": 5.0, "B": 1.5, "C": -2.0}
    true_effect = np.array([segment_effect[s] for s in segment])
    # Effect also grows with engagement, so it varies continuously within segments.
    true_effect = true_effect + 2.0 * (engagement - engagement.mean())

    # Treatment is randomised here: the challenge is heterogeneity, not confounding.
    treatment = rng.binomial(1, 0.5, n)

    baseline = 20.0 + 0.05 * tenure_months + 6.0 * engagement
    outcome = baseline + true_effect * treatment + rng.normal(0, 1.0, n)

    df = pd.DataFrame(
        {
            "segment": segment,
            "engagement": engagement,
            "tenure_months": tenure_months,
            "treatment": treatment,
            "outcome": outcome,
            "true_effect": true_effect,
        }
    )

    true_ate = float(true_effect.mean())
    naive = _naive_difference(df, "treatment", "outcome")
    return df, {
        "true_ate": true_ate,
        "naive_estimate": naive,
        "bias": naive - true_ate,
        "treatment": "treatment",
        "outcome": "outcome",
        "covariates": ["segment", "engagement", "tenure_months"],
        "segment_col": "segment",
        "true_segment_effects": segment_effect,
        "true_effect_col": "true_effect",
        "why": (
            "Treatment was randomised, so the average effect is unbiased — but the "
            "average is positive while segment C is actively harmed, so acting on "
            "the average alone would damage a quarter of the population."
        ),
    }


def make_did_panel(
    n_units: int = 300,
    n_periods: int = 8,
    treat_period: int = 4,
    true_effect: float = 3.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Generate panel data suitable for a difference-in-differences analysis.

    Treated and control units have different baseline levels but parallel trends
    before ``treat_period``, which is exactly the assumption difference-in-
    differences relies on. The treatment effect switches on for treated units from
    ``treat_period`` onwards.

    Args:
        n_units: Number of distinct units observed over time.
        n_periods: Number of time periods per unit.
        treat_period: First period in which treated units receive the treatment.
        true_effect: The true effect applied to treated units post-treatment.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of the long-format panel and its ground truth.
    """
    rng = np.random.default_rng(seed)

    unit_ids = np.arange(n_units)
    is_treated_unit = (unit_ids % 2 == 0).astype(int)
    # Level differences between groups are fine for DiD; trend differences are not.
    unit_baseline = rng.normal(20, 4, n_units) + 3.0 * is_treated_unit

    rows = []
    for period in range(n_periods):
        common_trend = 0.8 * period
        post = int(period >= treat_period)
        outcome = (
            unit_baseline
            + common_trend
            + true_effect * post * is_treated_unit
            + rng.normal(0, 1.0, n_units)
        )
        rows.append(
            pd.DataFrame(
                {
                    "unit": unit_ids,
                    "period": period,
                    "treated_unit": is_treated_unit,
                    "post": post,
                    "outcome": outcome,
                }
            )
        )

    df = pd.concat(rows, ignore_index=True)
    df["treatment"] = df["treated_unit"] * df["post"]

    naive = _naive_difference(df, "treatment", "outcome")
    return df, {
        "true_ate": true_effect,
        "naive_estimate": naive,
        "bias": naive - true_effect,
        "treatment": "treatment",
        "outcome": "outcome",
        "time_col": "period",
        "group_col": "treated_unit",
        "treat_period": treat_period,
        "covariates": [],
        "why": (
            "Treated units start from a higher baseline, so comparing treated and "
            "untreated observations directly conflates that permanent gap with the "
            "effect of the treatment."
        ),
    }


def make_iv(
    n: int = 5000,
    true_effect: float = 1.5,
    confounding: float = 2.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Generate data with an endogenous treatment and a valid instrument.

    An unobserved variable (``ability``, deliberately excluded from the returned
    frame) drives both treatment and outcome, so no amount of adjustment on the
    observed covariates can remove the bias. The instrument affects the outcome
    only through the treatment, which is what makes two-stage least squares work.

    Args:
        n: Number of units to generate.
        true_effect: True causal effect of the treatment on the outcome.
        confounding: Strength of the unobserved confounder.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of the generated frame (without the hidden confounder) and its
        ground truth.
    """
    rng = np.random.default_rng(seed)

    # Never included in the returned frame: this is the whole point of the design.
    ability = rng.normal(0, 1, n)

    instrument = rng.binomial(1, 0.5, n)
    experience = rng.integers(0, 30, n).astype(float)

    # The instrument shifts treatment, but has no direct path to the outcome.
    treatment_latent = 0.4 + 1.2 * instrument + confounding * 0.6 * ability + rng.normal(0, 0.5, n)
    treatment = (treatment_latent > treatment_latent.mean()).astype(int)

    outcome = (
        5.0
        + true_effect * treatment
        + confounding * ability
        + 0.03 * experience
        + rng.normal(0, 1.0, n)
    )

    df = pd.DataFrame(
        {
            "instrument": instrument,
            "experience": experience,
            "treatment": treatment,
            "outcome": outcome,
        }
    )

    naive = _naive_difference(df, "treatment", "outcome")
    return df, {
        "true_ate": true_effect,
        "naive_estimate": naive,
        "bias": naive - true_effect,
        "treatment": "treatment",
        "outcome": "outcome",
        "instrument": "instrument",
        "covariates": ["experience"],
        "why": (
            "An unmeasured trait raises both the chance of treatment and the "
            "outcome, and it is not in the data, so covariate adjustment cannot "
            "fix the bias — only an instrument can."
        ),
    }


def make_no_effect(n: int = 5000, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Generate data where the treatment genuinely does nothing.

    Treatment and outcome are still correlated through shared causes, so a naive
    analysis reports a confident non-zero effect. A trustworthy pipeline must
    report approximately zero here and survive a placebo test. A tool that always
    finds an effect is worse than no tool at all, which is why this case is part
    of the validation suite rather than an afterthought.

    Args:
        n: Number of units to generate.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of the generated frame and its ground truth, with ``true_ate`` of 0.
    """
    rng = np.random.default_rng(seed)

    risk_score = rng.normal(0, 1, n)
    region = rng.choice([0, 1, 2], size=n)

    logit = 1.4 * risk_score + 0.3 * region - 0.5
    treatment = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))

    # Treatment does not appear in the outcome equation at all.
    outcome = 12.0 + 4.0 * risk_score + 0.8 * region + rng.normal(0, 1.0, n)

    df = pd.DataFrame(
        {
            "risk_score": risk_score,
            "region": region.astype(float),
            "treatment": treatment,
            "outcome": outcome,
        }
    )

    naive = _naive_difference(df, "treatment", "outcome")
    return df, {
        "true_ate": 0.0,
        "naive_estimate": naive,
        "bias": naive,
        "treatment": "treatment",
        "outcome": "outcome",
        "confounders": ["risk_score"],
        "covariates": ["risk_score", "region"],
        "why": (
            "The treatment has no effect whatsoever, but high-risk units were far "
            "more likely to receive it and have worse outcomes regardless, so the "
            "raw comparison invents an effect that does not exist."
        ),
    }


#: Registry of every generator, keyed by the name shown in the interface.
DATASET_REGISTRY: dict[str, Callable[..., tuple[pd.DataFrame, dict]]] = {
    "confounded": make_confounded,
    "heterogeneous": make_heterogeneous,
    "did_panel": make_did_panel,
    "instrumental": make_iv,
    "no_effect": make_no_effect,
}

#: Short human-readable description of what each dataset demonstrates.
DATASET_DESCRIPTIONS: dict[str, str] = {
    "confounded": "Common causes bias the naive comparison; adjustment recovers the truth.",
    "heterogeneous": "A positive average effect hides a segment the treatment harms.",
    "did_panel": "Panel data with parallel pre-treatment trends for difference-in-differences.",
    "instrumental": "Unobserved confounding that only an instrument can overcome.",
    "no_effect": "The treatment does nothing, but the raw comparison suggests otherwise.",
}


def list_datasets() -> dict[str, str]:
    """Return the available dataset names mapped to their descriptions."""
    return dict(DATASET_DESCRIPTIONS)


def load_dataset(name: str, **kwargs) -> tuple[pd.DataFrame, dict]:
    """Generate a registered dataset by name.

    Args:
        name: Key from :data:`DATASET_REGISTRY`.
        **kwargs: Forwarded to the underlying generator.

    Returns:
        Tuple of the generated frame and its ground truth.

    Raises:
        ValueError: If ``name`` is not a registered dataset.
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name](**kwargs)
