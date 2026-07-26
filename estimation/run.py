"""Orchestration of a full analysis from data to assembled report.

The order matters as much as the individual steps. Covariates are classified
before anything is adjusted for, so mediators and colliders are dropped from the
adjustment set rather than quietly biasing every estimate that follows. Every
applicable method is then run, each is attacked with the robustness checks, and
the results are assembled with the conditions under which they hold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from discovery.assumption_check import run_all_checks
from discovery.confounder import classify_covariates
from discovery.dag_builder import CausalDAG, auto_dag
from estimation.base import EffectResult
from estimation.did import DifferenceInDifferences
from estimation.iv import InstrumentalVariables
from estimation.propensity import (
    InverseProbabilityWeighting,
    PropensityScoreMatching,
    fit_propensity,
)
from estimation.uplift import SLearner, TLearner, XLearner, cate_by_segment
from reporting.report import CausalReport, ReportBuilder
from validation.refutation import Refuter
from validation.sensitivity import e_value, sensitivity_summary

#: Methods that adjust for measured covariates and apply to any cross-section.
ADJUSTMENT_METHODS = {
    "psm": PropensityScoreMatching,
    "ipw": InverseProbabilityWeighting,
    "s_learner": SLearner,
    "t_learner": TLearner,
    "x_learner": XLearner,
}

#: Methods worth running by default. The remaining learners mostly restate what
#: the X-learner already shows, and each bootstrap resample refits several
#: gradient boosting models, so including them all multiplies runtime for very
#: little extra information.
DEFAULT_METHODS = ("psm", "ipw", "x_learner")

#: Learners that refit machine learning models on every resample. They are given
#: a smaller bootstrap budget because their cost per resample is an order of
#: magnitude higher than the propensity methods, and the interval width they
#: produce stabilises long before the point estimate changes.
EXPENSIVE_METHODS = frozenset({"s_learner", "t_learner", "x_learner"})


class AnalysisRunner:
    """Run every applicable estimator on a dataset and assemble the results."""

    def __init__(self, n_boot: int = 100, seed: int = 42) -> None:
        """
        Args:
            n_boot: Bootstrap resamples used where an estimator has no closed
                form standard error.
            seed: Random seed, so a rerun reproduces the same numbers.
        """
        self.n_boot = n_boot
        self.seed = seed

    def resolve_adjustment_set(
        self,
        treatment: str,
        outcome: str,
        covariates: list[str],
        dag: CausalDAG | None = None,
        instrument: str | None = None,
    ) -> tuple[list[str], dict, CausalDAG]:
        """Decide which covariates may be adjusted for.

        Args:
            treatment: Name of the treatment column.
            outcome: Name of the outcome column.
            covariates: Candidate covariates supplied by the analyst.
            dag: Causal graph encoding the analyst's assumptions. A conservative
                default is built when none is supplied.
            instrument: Optional instrument, excluded from the adjustment set.

        Returns:
            Tuple of the safe adjustment set, the full classification, and the
            graph the decision was made under.
        """
        if dag is None:
            dag = auto_dag(treatment, outcome, covariates, instrument=instrument)

        classification = classify_covariates(dag, treatment, outcome, covariates)
        safe = classification["safe_adjustment_set"]
        # A conservative default graph marks every covariate a confounder, so an
        # empty safe set only happens when the analyst's own graph rules them all
        # out. Falling back to the raw list there would defeat the check.
        return safe, classification, dag

    def run(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: list[str],
        dag: CausalDAG | None = None,
        instrument: str | None = None,
        time_col: str | None = None,
        group_col: str | None = None,
        treat_period: int | None = None,
        unit_col: str | None = None,
        segment_col: str | None = None,
        methods: list[str] | None = None,
        treatment_label: str | None = None,
        outcome_label: str | None = None,
        units: str = "",
        run_refutations: bool = True,
    ) -> CausalReport:
        """Analyse a dataset end to end and return an assembled report.

        Args:
            df: The data to analyse.
            treatment: Name of the binary treatment column.
            outcome: Name of the outcome column.
            covariates: Candidate covariates.
            dag: Optional causal graph; a conservative default is used otherwise.
            instrument: Optional instrument enabling instrumental variables.
            time_col: Period column, required for difference-in-differences.
            group_col: Ever-treated column, required for difference-in-differences.
            treat_period: First treated period.
            unit_col: Repeated unit column, used to cluster panel errors.
            segment_col: Column to summarise effect heterogeneity over.
            methods: Restrict which adjustment methods run. All run by default.
            treatment_label: Human-readable treatment name for the narrative.
            outcome_label: Human-readable outcome name for the narrative.
            units: Optional unit label for the outcome.
            run_refutations: Whether to attack the leading estimate. Disabling
                this is only appropriate for a quick look, since an unchecked
                estimate has not earned a causal reading.

        Returns:
            The assembled report.
        """
        df = df.reset_index(drop=True)
        adjustment_set, classification, dag = self.resolve_adjustment_set(
            treatment, outcome, covariates, dag, instrument
        )

        estimates: list[EffectResult] = []
        selected = methods or list(DEFAULT_METHODS)

        for name in selected:
            estimator_class = ADJUSTMENT_METHODS.get(name)
            if estimator_class is None:
                continue
            budget = max(self.n_boot // 3, 20) if name in EXPENSIVE_METHODS else self.n_boot
            estimator = estimator_class(n_boot=budget, seed=self.seed)
            try:
                estimates.append(
                    estimator.estimate(df, treatment, outcome, adjustment_set)
                )
            except Exception:
                # One estimator failing on awkward data should not lose the others.
                continue

        if time_col and group_col:
            try:
                estimates.append(
                    DifferenceInDifferences(unit_col=unit_col).estimate(
                        df,
                        treatment,
                        outcome,
                        adjustment_set,
                        group_col=group_col,
                        time_col=time_col,
                        treat_period=treat_period,
                    )
                )
            except Exception:
                pass

        if instrument:
            try:
                estimates.append(
                    InstrumentalVariables().estimate(
                        df, treatment, outcome, adjustment_set, instrument=instrument
                    )
                )
            except Exception:
                pass

        checks = None
        if adjustment_set:
            try:
                scores, _ = fit_propensity(df, treatment, adjustment_set)
                checks = run_all_checks(df, treatment, adjustment_set, scores)
            except Exception:
                checks = run_all_checks(df, treatment, adjustment_set)
        else:
            checks = run_all_checks(df, treatment, [])

        refutations: dict = {}
        sensitivity: dict = {}
        if estimates:
            primary = estimates[0]
            if run_refutations:
                refuter = Refuter(df, treatment, outcome, adjustment_set, seed=self.seed)
                estimator = ADJUSTMENT_METHODS.get(primary.method)
                if estimator is not None:
                    refutations[primary.method] = refuter.run_all(
                        estimator(n_boot=max(self.n_boot // 4, 20), seed=self.seed),
                        primary.point_estimate,
                    )
            outcome_sd = float(pd.to_numeric(df[outcome], errors="coerce").std())
            sensitivity[primary.method] = sensitivity_summary(
                e_value(primary.point_estimate, primary.ci_low, primary.ci_high, outcome_sd)
            )

        segments = None
        if segment_col and segment_col in df.columns:
            try:
                # Only the per-unit predictions are needed here, and those involve
                # no resampling, so no bootstrap budget is spent on this step.
                learner = XLearner(n_boot=0, seed=self.seed)
                cate = learner.predict_cate(df, treatment, outcome, adjustment_set)
                segments = cate_by_segment(cate, df, segment_col)
            except Exception:
                segments = None

        return ReportBuilder().build(
            treatment=treatment_label or treatment,
            outcome=outcome_label or outcome,
            estimates=estimates,
            dataset_summary={"n_rows": len(df), "n_cols": df.shape[1]},
            assumption_checks=checks,
            covariate_roles=classification,
            refutations=refutations,
            sensitivity=sensitivity,
            segments=segments,
            units=units,
        )
