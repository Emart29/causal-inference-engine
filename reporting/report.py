"""Assembly of a complete causal analysis into a single readable document.

The ordering here is deliberate and is the main thing this module contributes.
Sections run question, then assumptions, then estimates, then the attempts to
discredit them, then sensitivity, then heterogeneity, and only afterwards the
headline. A reader who meets the effect size first tends to keep it regardless of
what follows, so the conditions under which the number is meaningful are put in
front of it rather than appended as caveats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from estimation.base import EffectResult
from reporting import interpret


@dataclass
class CausalReport:
    """Everything needed to present one analysis, in the order it should be read.

    Attributes:
        question: The causal question in plain language.
        treatment: Human-readable name of the treatment.
        outcome: Human-readable name of the outcome.
        dataset_summary: Row and column counts and other framing detail.
        assumption_checks: Output of the assumption checks, or ``None``.
        covariate_roles: Output of the covariate classifier, or ``None``.
        estimates: One entry per method that produced a result.
        refutations: Robustness summary keyed by method name.
        sensitivity: Sensitivity summary keyed by method name.
        segments: Per-segment effects, where heterogeneity was assessed.
        headline: The single sentence a decision-maker should read first.
        narrative: Section name mapped to its plain-English text.
        units: Optional unit label for the outcome.
        generated_at: When the report was assembled.
    """

    question: str
    treatment: str
    outcome: str
    dataset_summary: dict = field(default_factory=dict)
    assumption_checks: dict | None = None
    covariate_roles: dict | None = None
    estimates: list[EffectResult] = field(default_factory=list)
    refutations: dict = field(default_factory=dict)
    sensitivity: dict = field(default_factory=dict)
    segments: pd.DataFrame | None = None
    headline: str = ""
    narrative: dict = field(default_factory=dict)
    units: str = ""
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def primary_estimate(self) -> EffectResult | None:
        """The estimate the headline is based on, or ``None`` if there are none."""
        return self.estimates[0] if self.estimates else None

    @property
    def is_trustworthy(self) -> bool:
        """Whether the analysis rests on assumptions the data supports.

        False when a blocking assumption failed or the primary estimate did not
        survive the placebo check, which are the two conditions under which the
        effect size should not be quoted at all.
        """
        if self.assumption_checks and self.assumption_checks.get("blocking_failures"):
            return False
        primary = self.primary_estimate
        if primary and self.refutations.get(primary.method, {}).get("verdict") == "failed":
            return False
        return True


class ReportBuilder:
    """Turn the pieces of an analysis into an assembled report."""

    def build(
        self,
        treatment: str,
        outcome: str,
        estimates: list[EffectResult],
        dataset_summary: dict | None = None,
        assumption_checks: dict | None = None,
        covariate_roles: dict | None = None,
        refutations: dict | None = None,
        sensitivity: dict | None = None,
        segments: pd.DataFrame | None = None,
        units: str = "",
    ) -> CausalReport:
        """Assemble a report and write its narrative.

        Args:
            treatment: Human-readable name of the treatment.
            outcome: Human-readable name of the outcome.
            estimates: Results from each method that ran, most trusted first.
            dataset_summary: Row and column counts and other framing detail.
            assumption_checks: Output of the assumption checks.
            covariate_roles: Output of the covariate classifier.
            refutations: Robustness summaries keyed by method name.
            sensitivity: Sensitivity summaries keyed by method name.
            segments: Per-segment effects, where heterogeneity was assessed.
            units: Optional unit label for the outcome.

        Returns:
            The assembled report, with every narrative section written.
        """
        refutations = refutations or {}
        sensitivity = sensitivity or {}
        primary = estimates[0] if estimates else None

        question = (
            f"What effect does {treatment} have on {outcome}?"
        )

        narrative: dict[str, str] = {}
        if assumption_checks:
            narrative["assumptions"] = interpret.interpret_assumptions(assumption_checks)
        if covariate_roles:
            narrative["adjustment"] = interpret.interpret_covariate_roles(covariate_roles)
        if estimates:
            narrative["estimate"] = interpret.interpret_effect(primary, treatment, outcome, units)
            narrative["agreement"] = interpret.compare_methods(estimates, units)
        if primary and primary.method in refutations:
            narrative["robustness"] = interpret.interpret_refutations(refutations[primary.method])
        if primary and primary.method in sensitivity:
            narrative["sensitivity"] = interpret.interpret_sensitivity(sensitivity[primary.method])
        if segments is not None and not segments.empty:
            narrative["heterogeneity"] = interpret.interpret_heterogeneity(segments, outcome)

        headline = ""
        if primary:
            headline = interpret.headline(
                primary,
                treatment,
                outcome,
                checks=assumption_checks,
                refutations=refutations.get(primary.method),
                sensitivity=sensitivity.get(primary.method),
                units=units,
            )

        return CausalReport(
            question=question,
            treatment=treatment,
            outcome=outcome,
            dataset_summary=dataset_summary or {},
            assumption_checks=assumption_checks,
            covariate_roles=covariate_roles,
            estimates=estimates,
            refutations=refutations,
            sensitivity=sensitivity,
            segments=segments,
            headline=headline,
            narrative=narrative,
            units=units,
        )

    def to_markdown(self, report: CausalReport) -> str:
        """Render a report as Markdown, in the order it should be read.

        Args:
            report: The assembled report.

        Returns:
            A Markdown document.
        """
        lines: list[str] = [f"# {report.question}", ""]

        if not report.is_trustworthy:
            lines += [
                "> **This analysis does not support a causal conclusion.** "
                "The sections below explain why.",
                "",
            ]

        lines += ["## What the data can support", "", report.narrative.get("assumptions", "Not assessed."), ""]
        if "adjustment" in report.narrative:
            lines += [report.narrative["adjustment"], ""]

        lines += ["## Estimated effect", ""]
        if report.estimates:
            lines += [report.narrative.get("estimate", ""), ""]
            lines += ["| Method | Estimand | Estimate | 95% interval |", "| --- | --- | --- | --- |"]
            for result in report.estimates:
                lines.append(
                    f"| {result.method} | {result.estimate_type.upper()} | "
                    f"{result.point_estimate:,.3f} | "
                    f"{result.ci_low:,.3f} to {result.ci_high:,.3f} |"
                )
            lines += ["", report.narrative.get("agreement", ""), ""]
        else:
            lines += ["No estimate could be produced.", ""]

        if "robustness" in report.narrative:
            lines += ["## Does the result survive scrutiny", "", report.narrative["robustness"], ""]
        if "sensitivity" in report.narrative:
            lines += ["## What could overturn it", "", report.narrative["sensitivity"], ""]
        if "heterogeneity" in report.narrative:
            lines += ["## Who it affects", "", report.narrative["heterogeneity"], ""]
            if report.segments is not None and not report.segments.empty:
                lines += ["| Segment | Units | Effect | 95% interval |", "| --- | --- | --- | --- |"]
                for _, row in report.segments.iterrows():
                    flag = "  (harmed)" if row["harmed"] else ""
                    lines.append(
                        f"| {row['segment']}{flag} | {int(row['n'])} | "
                        f"{row['mean_effect']:+.3f} | "
                        f"{row['ci_low']:+.3f} to {row['ci_high']:+.3f} |"
                    )
                lines.append("")

        # The headline sits last in the document but is surfaced first in the
        # interface, so a reader who works through the evidence and one who wants
        # the summary both arrive at the same sentence.
        lines += ["## In one sentence", "", report.headline or "No conclusion could be drawn.", ""]
        lines += [
            "---",
            "",
            f"*Generated {report.generated_at:%Y-%m-%d %H:%M} UTC. Every estimate here is "
            "conditional on the assumptions stated above; no statistical method can "
            "rescue a comparison the data cannot support.*",
        ]
        return "\n".join(lines)
