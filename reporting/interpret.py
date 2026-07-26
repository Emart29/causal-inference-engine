"""Translation of statistical output into careful, non-overclaiming English.

Most of the damage done with causal estimates happens in the sentence, not the
arithmetic: an effect on the treated described as an effect on everyone, a wide
interval reported as a point, an absence of evidence reported as evidence of
absence. This module owns that sentence.

Four rules are enforced throughout:

* an estimate is never stated without its uncertainty,
* the estimand is always named, so an effect measured on one group is never
  presented as applying to another,
* a failed robustness check leads the summary rather than trailing it,
* a non-significant result is described as no detectable effect, never as proof
  that no effect exists.
"""

from __future__ import annotations

import pandas as pd

from estimation.base import EffectResult

#: How each estimand should be described to a non-technical reader.
ESTIMAND_PHRASING: dict[str, str] = {
    "ate": "across everyone in the population",
    "att": "among those who actually received the treatment",
    "cate": "for units with the given characteristics",
    "late": "among the units whose treatment was changed by the instrument",
}


def _format_effect(value: float, units: str = "") -> str:
    """Render an effect size with optional units."""
    suffix = f" {units}" if units else ""
    return f"{value:,.3f}{suffix}" if abs(value) < 1000 else f"{value:,.0f}{suffix}"


def interpret_effect(
    result: EffectResult,
    treatment: str,
    outcome: str,
    units: str = "",
) -> str:
    """Describe a single estimate, its uncertainty, and who it applies to.

    Args:
        result: The estimate to describe.
        treatment: Human-readable name of the treatment.
        outcome: Human-readable name of the outcome.
        units: Optional unit label for the outcome, such as "dollars".

    Returns:
        Two or three sentences stating the effect, its interval, and its scope.
    """
    direction = "increased" if result.point_estimate >= 0 else "reduced"
    magnitude = _format_effect(abs(result.point_estimate), units)
    scope = ESTIMAND_PHRASING.get(result.estimate_type, "for an unspecified group")

    sentences = [
        f"{treatment} {direction} {outcome} by {magnitude} on average {scope} "
        f"(95% confidence interval: {_format_effect(result.ci_low, units)} to "
        f"{_format_effect(result.ci_high, units)})."
    ]

    if result.significant():
        sentences.append(
            "The interval excludes zero, so an effect of this direction is "
            "distinguishable from no effect at the 5% level."
        )
    else:
        # The distinction between "no effect" and "no detectable effect" is the
        # difference between a claim the data supports and one it does not.
        sentences.append(
            "The interval includes zero, so this analysis did not detect an effect. "
            "That is not the same as showing the treatment does nothing: an effect "
            "smaller than the data can resolve would look the same."
        )

    if result.estimate_type == "late":
        sentences.append(
            "This applies only to units the instrument moved, which may not "
            "represent the wider population."
        )
    elif result.estimate_type == "att":
        sentences.append(
            "This describes those who were treated, and may differ from the effect "
            "treatment would have on those who were not."
        )

    return " ".join(sentences)


def interpret_assumptions(checks: dict) -> str:
    """Describe whether the analysis rests on ground the data can support.

    Args:
        checks: Output of the assumption check runner.

    Returns:
        A summary leading with any blocking failure, since an estimate produced
        on unsupported data should not be read at all.
    """
    if checks.get("blocking_failures"):
        failed = checks["checks"]
        details = " ".join(failed[name]["interpretation"] for name in checks["blocking_failures"])
        return (
            "The assumptions required for a causal estimate do not hold on this "
            f"data, so the numbers below should not be acted on. {details}"
        )

    if checks.get("warnings"):
        failed = checks["checks"]
        details = " ".join(failed[name]["interpretation"] for name in checks["warnings"])
        return f"The analysis can proceed, with caveats. {details}"

    return (
        "Every assumption check passed: the two groups overlap, both arms are "
        "large enough, and the measured covariates are balanced."
    )


def interpret_refutations(summary: dict) -> str:
    """Describe how the estimate behaved under attempts to discredit it.

    Args:
        summary: Output of the refutation runner.

    Returns:
        A statement leading with failure where one occurred.
    """
    verdict = summary.get("verdict", "untested")
    n_passed = summary.get("n_passed", 0)
    n_total = summary.get("n_total", 0)

    if verdict == "failed":
        return (
            "This estimate failed the placebo check: effects of a similar size appear "
            "even when the treatment is assigned at random. The result cannot be "
            "interpreted as causal, whatever its confidence interval suggests."
        )
    if verdict == "no_effect":
        return (
            "The estimate is no larger than what random treatment assignment "
            "produces, so the analysis found no detectable effect. This is a "
            "conclusion, not a failure of the method."
        )
    if verdict == "robust":
        return (
            f"The estimate survived all {n_total} robustness checks, including the "
            "placebo check where the treatment was replaced with noise."
        )
    return (
        f"The estimate passed {n_passed} of {n_total} robustness checks. It survived "
        "the placebo check but was unstable under others, so treat it as provisional."
    )


def interpret_sensitivity(sensitivity: dict) -> str:
    """Describe how vulnerable the conclusion is to something unmeasured.

    Args:
        sensitivity: Output of the sensitivity summary.

    Returns:
        A statement of what strength of hidden confounder would overturn the result.
    """
    return sensitivity.get("interpretation", "Sensitivity to unmeasured confounding was not assessed.")


def interpret_heterogeneity(segment_table: pd.DataFrame, outcome: str = "the outcome") -> str:
    """Describe how the effect differs across groups.

    Args:
        segment_table: Per-segment effects, as produced by the CATE summary.
        outcome: Human-readable name of the outcome.

    Returns:
        A statement naming the strongest and weakest groups, and flagging any
        group the treatment appears to harm.
    """
    if segment_table is None or segment_table.empty:
        return "Effect variation across groups was not assessed."

    best = segment_table.iloc[0]
    worst = segment_table.iloc[-1]
    harmed = segment_table[segment_table["harmed"]]

    parts = [
        f"The effect is strongest for {best['segment']} at "
        f"{best['mean_effect']:+.2f} and weakest for {worst['segment']} at "
        f"{worst['mean_effect']:+.2f}."
    ]

    if not harmed.empty:
        names = ", ".join(str(s) for s in harmed["segment"])
        # A harmed group is the single most actionable finding available, so it is
        # stated plainly rather than left for the reader to infer from a table.
        parts.append(
            f"Treatment appears to make {names} worse off, so applying it "
            "universally would harm part of the population even though the average "
            "effect is positive."
        )
    else:
        parts.append("No group appears to be made worse off by the treatment.")

    return " ".join(parts)


def interpret_covariate_roles(classification: dict) -> str:
    """Describe which variables were adjusted for and which were deliberately not.

    Args:
        classification: Output of the covariate classifier.

    Returns:
        A statement of the adjustment set and any variable excluded to avoid
        damaging the estimate.
    """
    adjusted = classification.get("safe_adjustment_set", [])
    warnings = classification.get("warnings", [])

    if adjusted:
        parts = [f"The analysis adjusts for {', '.join(adjusted)}."]
    else:
        parts = ["No covariates were adjusted for."]

    if warnings:
        parts.append(
            "Some variables were deliberately excluded because controlling for them "
            "would distort the result: " + " ".join(warnings)
        )

    return " ".join(parts)


def headline(
    result: EffectResult,
    treatment: str,
    outcome: str,
    checks: dict | None = None,
    refutations: dict | None = None,
    sensitivity: dict | None = None,
    units: str = "",
) -> str:
    """Produce the one sentence a decision-maker should read first.

    The sentence earns its confidence rather than assuming it. If the assumptions
    fail, or the placebo check fails, that comes first and no effect size is
    offered, because quoting a number alongside a warning invites the reader to
    remember only the number.

    Args:
        result: The estimate being summarised.
        treatment: Human-readable name of the treatment.
        outcome: Human-readable name of the outcome.
        checks: Optional assumption check output.
        refutations: Optional refutation summary.
        sensitivity: Optional sensitivity summary.
        units: Optional unit label for the outcome.

    Returns:
        A single sentence carrying the appropriate level of confidence.
    """
    if checks and checks.get("blocking_failures"):
        return (
            f"No reliable conclusion can be drawn about the effect of {treatment} on "
            f"{outcome}: the data does not support a causal comparison."
        )

    if refutations and refutations.get("verdict") == "failed":
        return (
            f"The apparent effect of {treatment} on {outcome} does not survive a "
            "placebo check and should not be treated as real."
        )

    if refutations and refutations.get("verdict") == "no_effect":
        return (
            f"This analysis found no detectable effect of {treatment} on {outcome}."
        )

    if not result.significant():
        return (
            f"This analysis did not detect an effect of {treatment} on {outcome}; the "
            "range of plausible values includes no effect at all."
        )

    direction = "increases" if result.point_estimate >= 0 else "reduces"
    magnitude = _format_effect(abs(result.point_estimate), units)
    scope = ESTIMAND_PHRASING.get(result.estimate_type, "")

    confidence = ""
    if sensitivity:
        robustness = sensitivity.get("robustness")
        if robustness == "low":
            confidence = ", though a weak unmeasured factor could account for it"
        elif robustness == "moderate":
            confidence = ", though a moderately strong unmeasured factor could account for it"

    return (
        f"{treatment} {direction} {outcome} by about {magnitude} {scope} "
        f"(95% CI {_format_effect(result.ci_low, units)} to "
        f"{_format_effect(result.ci_high, units)}){confidence}."
    )


def compare_methods(results: list[EffectResult], units: str = "") -> str:
    """Describe whether different methods reached the same conclusion.

    Agreement across methods that rely on different assumptions is stronger
    evidence than any single estimate, and disagreement is a signal that the
    assumptions are doing more work than the data.

    Args:
        results: Estimates from the methods that were run.
        units: Optional unit label for the outcome.

    Returns:
        A statement about agreement, or the lack of it.
    """
    if not results:
        return "No estimates were produced."
    if len(results) == 1:
        return f"Only one method was applicable, so its result cannot be cross-checked."

    estimates = [r.point_estimate for r in results]
    lowest, highest = min(estimates), max(estimates)
    all_same_sign = all(e >= 0 for e in estimates) or all(e < 0 for e in estimates)
    spread = highest - lowest
    typical = max(abs(sum(estimates) / len(estimates)), 1e-9)

    names = ", ".join(r.method for r in results)
    if all_same_sign and spread <= abs(typical):
        return (
            f"All {len(results)} methods ({names}) agree on the direction and give "
            f"estimates between {_format_effect(lowest, units)} and "
            f"{_format_effect(highest, units)}, which is reassuring because they rely "
            "on different assumptions."
        )
    if all_same_sign:
        return (
            f"The {len(results)} methods agree on direction but differ in size, from "
            f"{_format_effect(lowest, units)} to {_format_effect(highest, units)}. The "
            "direction is more trustworthy than the magnitude."
        )
    return (
        f"The methods disagree on direction, ranging from {_format_effect(lowest, units)} "
        f"to {_format_effect(highest, units)}. Their assumptions are driving the answer "
        "more than the data is, so no conclusion should be drawn."
    )
