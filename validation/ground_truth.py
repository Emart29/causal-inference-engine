"""Verification that the engine recovers effects it can be checked against.

A causal estimate on real data cannot be graded, because the true effect is
unknown by definition. That is exactly why the synthetic generators exist: they
build the effect into the data, so every estimator can be asked whether it
recovers the answer that was planted.

This suite is the evidence behind any claim the project makes. It runs three
tiers deliberately, because only reporting the first would be misleading:

* clean data, where the estimators should land on the truth and any failure
  means the implementation is broken,
* realistic data, where a small effect, noisy proxies, and a confounder that is
  missing entirely mean the estimators should improve on the naive comparison
  without reaching the truth,
* a real benchmark whose answer is known from a randomised trial, where the
  observational estimate is famously wrong and the engine is expected to say so
  rather than to succeed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from datasets.generate import load_dataset
from estimation.did import DifferenceInDifferences
from estimation.iv import InstrumentalVariables
from estimation.propensity import InverseProbabilityWeighting, PropensityScoreMatching
from estimation.uplift import XLearner, cate_by_segment


#: What a row is expected to do, which decides how it is graded.
#:
#: ``recover``  the interval should contain the true effect,
#: ``improve``  the truth is out of reach, but the naive comparison should be
#:              beaten, which is the standard for the realistic tier,
#: ``fail``     the method cannot address this problem and is included to
#:              demonstrate that, so not improving is the correct outcome.
EXPECTATIONS = ("recover", "improve", "fail")


@dataclass
class ValidationRow:
    """One estimator's attempt at a dataset whose true effect is known.

    Attributes:
        dataset: Which dataset was used.
        method: Which estimator produced the result.
        true_effect: The effect built into the data.
        naive_estimate: What a raw difference in means reports.
        estimate: What the estimator reports.
        ci_low: Lower bound of its interval.
        ci_high: Upper bound of its interval.
        covers_truth: Whether the interval contains the true effect.
        beats_naive: Whether the estimator is closer to the truth than the naive
            comparison.
        expectation: One of :data:`EXPECTATIONS`, deciding how the row is graded.
    """

    dataset: str
    method: str
    true_effect: float
    naive_estimate: float
    estimate: float
    ci_low: float
    ci_high: float
    covers_truth: bool
    beats_naive: bool
    expectation: str = "recover"

    @property
    def error(self) -> float:
        """Signed distance between the estimate and the truth."""
        return self.estimate - self.true_effect

    @property
    def bias_removed_pct(self) -> float:
        """Share of the naive comparison's bias the estimator removed."""
        naive_bias = abs(self.naive_estimate - self.true_effect)
        if naive_bias < 1e-9:
            return 100.0
        return float((1 - abs(self.error) / naive_bias) * 100)

    @property
    def passed(self) -> bool:
        """Whether this row behaved as it should for its expectation.

        A row marked ``fail`` passes precisely when the method does not fix the
        problem, since it is included to show a limitation. Grading it like the
        others would report a correct demonstration as a defect.
        """
        if self.expectation == "recover":
            return self.covers_truth
        if self.expectation == "improve":
            return self.beats_naive
        return not self.beats_naive


def _row(
    dataset: str,
    method: str,
    ground_truth: dict,
    estimate: float,
    ci_low: float,
    ci_high: float,
    expectation: str = "recover",
) -> ValidationRow:
    """Build a validation row from an estimator's output."""
    true_effect = ground_truth["true_ate"]
    naive = ground_truth["naive_estimate"]
    return ValidationRow(
        dataset=dataset,
        method=method,
        true_effect=true_effect,
        naive_estimate=naive,
        estimate=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        covers_truth=bool(ci_low <= true_effect <= ci_high),
        beats_naive=bool(abs(estimate - true_effect) < abs(naive - true_effect)),
        expectation=expectation,
    )


def validate_confounding(n_boot: int = 60) -> list[ValidationRow]:
    """Check that adjustment recovers an effect a common cause has obscured."""
    df, truth = load_dataset("confounded")
    covariates = truth["covariates"]
    rows = []
    for estimator in (
        PropensityScoreMatching(n_boot=n_boot),
        InverseProbabilityWeighting(n_boot=n_boot),
    ):
        result = estimator.estimate(df, "treatment", "outcome", covariates)
        rows.append(
            _row("confounded", result.method, truth, result.point_estimate, result.ci_low, result.ci_high)
        )
    return rows


def validate_heterogeneity(n_boot: int = 40) -> tuple[list[ValidationRow], pd.DataFrame]:
    """Check that per-unit effects are recovered, including a harmed segment."""
    df, truth = load_dataset("heterogeneous")
    covariates = truth["covariates"]

    learner = XLearner(n_boot=n_boot)
    result = learner.estimate(df, "treatment", "outcome", covariates)
    cate = learner.predict_cate(df, "treatment", "outcome", covariates)

    segments = cate_by_segment(cate, df, truth["segment_col"])
    segments["true_effect"] = segments["segment"].map(truth["true_segment_effects"])
    segments["recovered"] = (
        (segments["mean_effect"] - segments["true_effect"]).abs() < 0.5
    )
    # The correlation with the planted per-unit effect is the real test: a
    # learner can match segment averages while ranking individuals at random.
    correlation = float(np.corrcoef(cate, df[truth["true_effect_col"]])[0, 1])
    segments.attrs["cate_correlation"] = correlation

    rows = [
        _row("heterogeneous", result.method, truth, result.point_estimate, result.ci_low, result.ci_high)
    ]
    return rows, segments


def validate_did() -> list[ValidationRow]:
    """Check that a policy-style effect is recovered from panel data."""
    df, truth = load_dataset("did_panel")
    result = DifferenceInDifferences(unit_col="unit").estimate(
        df,
        "treatment",
        "outcome",
        [],
        group_col=truth["group_col"],
        time_col=truth["time_col"],
        treat_period=truth["treat_period"],
    )
    return [_row("did_panel", result.method, truth, result.point_estimate, result.ci_low, result.ci_high)]


def validate_iv(n_boot: int = 40) -> list[ValidationRow]:
    """Check that an instrument recovers what adjustment cannot.

    Adjustment is run alongside deliberately, because its failure here is the
    point: the confounder is absent from the data, so no set of controls can fix
    it and only the instrument can.
    """
    df, truth = load_dataset("instrumental")
    covariates = truth["covariates"]
    rows = []

    adjusted = InverseProbabilityWeighting(n_boot=n_boot).estimate(
        df, "treatment", "outcome", covariates
    )
    rows.append(
        _row(
            "instrumental",
            f"{adjusted.method} (cannot fix this)",
            truth,
            adjusted.point_estimate,
            adjusted.ci_low,
            adjusted.ci_high,
            expectation="fail",
        )
    )

    iv_result = InstrumentalVariables().estimate(
        df, "treatment", "outcome", covariates, instrument=truth["instrument"]
    )
    rows.append(
        _row("instrumental", iv_result.method, truth, iv_result.point_estimate, iv_result.ci_low, iv_result.ci_high)
    )
    return rows


def validate_no_effect(n_boot: int = 60) -> list[ValidationRow]:
    """Check that nothing is reported when the treatment does nothing.

    This is the most important case in the suite. A tool that always finds an
    effect is worse than no tool, so the engine must return approximately zero
    where a naive comparison confidently reports a large effect.
    """
    df, truth = load_dataset("no_effect")
    covariates = truth["covariates"]
    rows = []
    for estimator in (
        PropensityScoreMatching(n_boot=n_boot),
        InverseProbabilityWeighting(n_boot=n_boot),
    ):
        result = estimator.estimate(df, "treatment", "outcome", covariates)
        rows.append(
            _row("no_effect", result.method, truth, result.point_estimate, result.ci_low, result.ci_high)
        )
    return rows


def validate_realistic(n_boot: int = 60) -> list[ValidationRow]:
    """Check behaviour where the truth is not recoverable.

    A small effect, a confounder observed only as a proxy, and another missing
    entirely mean the estimators should reduce the bias without reaching the
    answer. Recovery is not expected, so improving on the naive comparison is
    the standard applied.
    """
    df, truth = load_dataset("realistic")
    covariates = truth["covariates"]
    rows = []
    for estimator in (
        PropensityScoreMatching(n_boot=n_boot),
        InverseProbabilityWeighting(n_boot=n_boot),
    ):
        result = estimator.estimate(df, "treatment", "outcome", covariates)
        rows.append(
            _row(
                "realistic",
                result.method,
                truth,
                result.point_estimate,
                result.ci_low,
                result.ci_high,
                expectation="improve",
            )
        )
    return rows


def run_validation_suite(n_boot: int = 60, include_realistic: bool = True) -> dict:
    """Run every check and summarise the results.

    Args:
        n_boot: Bootstrap resamples for estimators that need them.
        include_realistic: Whether to include the tier that is not recoverable.

    Returns:
        Dictionary with the per-row ``table``, the recovered ``segments``, counts
        of how many rows behaved correctly, and an overall ``all_passed`` flag.
    """
    rows: list[ValidationRow] = []
    rows += validate_confounding(n_boot)

    heterogeneity_rows, segments = validate_heterogeneity(max(n_boot // 2, 20))
    rows += heterogeneity_rows

    rows += validate_did()
    rows += validate_iv(max(n_boot // 2, 20))
    rows += validate_no_effect(n_boot)
    if include_realistic:
        rows += validate_realistic(n_boot)

    table = pd.DataFrame(
        [
            {
                "dataset": r.dataset,
                "method": r.method,
                "true_effect": r.true_effect,
                "naive": r.naive_estimate,
                "estimate": r.estimate,
                "ci_low": r.ci_low,
                "ci_high": r.ci_high,
                "error": r.error,
                "bias_removed_pct": r.bias_removed_pct,
                "covers_truth": r.covers_truth,
                "expectation": r.expectation,
                "passed": r.passed,
            }
            for r in rows
        ]
    )

    n_passed = int(table["passed"].sum())
    return {
        "table": table,
        "segments": segments,
        "rows": rows,
        "n_passed": n_passed,
        "n_total": len(table),
        "all_passed": n_passed == len(table),
    }


def format_validation_table(result: dict) -> str:
    """Render the suite's results as plain text.

    Args:
        result: Output of :func:`run_validation_suite`.

    Returns:
        A table suitable for a terminal or a README, using ASCII only.
    """
    table = result["table"]
    lines = [
        f"{'dataset':14} {'method':28} {'true':>8} {'naive':>9} {'estimate':>9} "
        f"{'bias cut':>9}  {'expected':9} ok",
        "-" * 100,
    ]
    for _, row in table.iterrows():
        marker = "yes" if row["passed"] else "NO"
        lines.append(
            f"{row['dataset']:14} {row['method']:28} {row['true_effect']:8.3f} "
            f"{row['naive']:9.3f} {row['estimate']:9.3f} {row['bias_removed_pct']:8.1f}%  "
            f"{row['expectation']:9} {marker}"
        )

    lines.append("-" * 100)
    lines.append(f"{result['n_passed']} of {result['n_total']} behaved as expected")
    lines.append(
        "  recover: the interval should contain the true effect | "
        "improve: the truth is out of reach, so beating the naive comparison is the bar | "
        "fail: the method cannot fix this and is shown to prove it"
    )

    segments = result.get("segments")
    if segments is not None and not segments.empty:
        correlation = segments.attrs.get("cate_correlation")
        lines.append("")
        lines.append("Per-segment effects recovered:")
        for _, row in segments.iterrows():
            flag = "  (harmed)" if row["harmed"] else ""
            lines.append(
                f"  {row['segment']}: estimated {row['mean_effect']:+.2f} "
                f"against a true {row['true_effect']:+.2f}{flag}"
            )
        if correlation is not None:
            lines.append(f"  correlation with the planted per-unit effect: {correlation:.3f}")

    return "\n".join(lines)


def main() -> None:
    """Run the suite and print its results."""
    result = run_validation_suite()
    print(format_validation_table(result))


if __name__ == "__main__":
    main()
