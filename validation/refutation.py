"""Robustness checks that try to break an estimate rather than confirm it.

A causal estimate is a claim, and the useful question is not whether the number
is large but whether it survives attempts to discredit it. Each check here
manipulates the data in a way that should have a predictable effect: replacing
the treatment with noise should destroy the estimate, adding an irrelevant
variable should leave it alone, and dropping data at random should barely move
it. An estimate that behaves otherwise is measuring something other than the
treatment.

The placebo test is the one that matters most. If an effect appears when the
treatment is pure noise, the pipeline is manufacturing signal and no amount of
statistical significance rescues the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from estimation.base import Estimator

#: How far an estimate may move when an irrelevant variable is added, relative to
#: the original estimate, before the result is considered unstable.
PERTURBATION_TOLERANCE = 0.30

#: Minimum movement treated as meaningful, so checks on a near-zero estimate are
#: judged against an absolute floor rather than a fraction of almost nothing.
MIN_TOLERANCE = 0.05


@dataclass
class RefutationOutcome:
    """The result of one attempt to discredit an estimate.

    Attributes:
        test_name: Identifier of the check that was run.
        original_effect: The estimate being tested.
        new_effect: The estimate produced under the manipulation.
        passed: Whether the estimate behaved as a genuine effect should.
        interpretation: What the outcome means, in plain language.
        p_value: Where the check produces one.
    """

    test_name: str
    original_effect: float
    new_effect: float
    passed: bool
    interpretation: str
    p_value: float | None = None

    def to_dict(self) -> dict:
        """Serialise the outcome for storage or display."""
        return {
            "test_name": self.test_name,
            "original_effect": self.original_effect,
            "new_effect": self.new_effect,
            "passed": self.passed,
            "interpretation": self.interpretation,
            "p_value": self.p_value,
        }


class Refuter:
    """Run robustness checks against an estimator on a given dataset."""

    def __init__(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: list[str],
        seed: int = 42,
    ) -> None:
        """
        Args:
            df: The data the original estimate was produced from.
            treatment: Name of the treatment column.
            outcome: Name of the outcome column.
            covariates: Covariates the estimate adjusted for.
            seed: Random seed, so every check is reproducible.
        """
        self.df = df.reset_index(drop=True)
        self.treatment = treatment
        self.outcome = outcome
        self.covariates = covariates
        self.seed = seed

    def _estimate_on(self, estimator: Estimator, data: pd.DataFrame, covariates: list[str]) -> float:
        """Run the estimator and return only its point estimate."""
        result = estimator.estimate(data, self.treatment, self.outcome, covariates)
        return float(result.point_estimate)

    def placebo_treatment(
        self, estimator: Estimator, original_effect: float, n_trials: int = 5
    ) -> RefutationOutcome:
        """Replace the treatment with noise and confirm the effect disappears.

        The replacement preserves the original treatment rate but assigns it at
        random, so nothing about it can influence the outcome. Any effect that
        survives is coming from the estimation procedure rather than the data.

        The test is repeated so the placebo estimates themselves describe how
        much movement is attributable to chance. Judging a single placebo against
        a fraction of the original estimate would fail every correct null result,
        where that fraction is near zero by definition.

        Args:
            estimator: The estimator to re-run.
            original_effect: The estimate being tested.
            n_trials: Number of random treatment assignments to try.

        Returns:
            The outcome, which fails when the original estimate is not clearly
            larger than what random assignment produces.
        """
        rng = np.random.default_rng(self.seed)
        rate = float(self.df[self.treatment].mean())
        placebo_estimates: list[float] = []

        for _ in range(n_trials):
            data = self.df.copy()
            data[self.treatment] = rng.binomial(1, rate, len(data))
            try:
                placebo_estimates.append(self._estimate_on(estimator, data, self.covariates))
            except Exception:
                continue

        if not placebo_estimates:
            return RefutationOutcome(
                test_name="placebo_treatment",
                original_effect=original_effect,
                new_effect=float("nan"),
                passed=False,
                interpretation="The placebo test could not be completed on any resample.",
            )

        values = np.asarray(placebo_estimates)
        new_effect = float(np.mean(np.abs(values)))
        placebo_spread = float(values.std(ddof=1)) if len(values) > 1 else abs(new_effect)

        # Random assignment sets the scale of movement that means nothing. A real
        # effect must stand clearly above it; a true null sits inside it, which is
        # the correct outcome rather than a failure.
        noise_band = max(new_effect + 2 * placebo_spread, 1e-9)
        original_is_null = abs(original_effect) <= noise_band
        passed = original_is_null or abs(original_effect) > 2 * noise_band

        if original_is_null:
            interpretation = (
                f"Randomly assigned treatments produce effects around "
                f"{new_effect:.3f}, and the estimate of {original_effect:.3f} sits "
                "within that range. This is consistent with there being no effect, "
                "which the estimate correctly reports."
            )
        elif passed:
            interpretation = (
                f"Randomly assigned treatments produce effects of about "
                f"{new_effect:.3f}, far below the estimate of {original_effect:.3f}, "
                "so the result is not an artefact of the procedure."
            )
        else:
            interpretation = (
                f"A randomly assigned treatment produces effects of about "
                f"{new_effect:.3f}, close to the estimate of {original_effect:.3f}. "
                "The procedure finds comparable effects where none can exist, so "
                "this result should not be treated as causal."
            )

        return RefutationOutcome(
            test_name="placebo_treatment",
            original_effect=original_effect,
            new_effect=new_effect,
            passed=passed,
            interpretation=interpretation,
        )

    def random_common_cause(self, estimator: Estimator, original_effect: float) -> RefutationOutcome:
        """Add an irrelevant variable and confirm the estimate barely moves.

        A variable unrelated to everything carries no confounding, so adjusting
        for it should change nothing. A large shift means the estimate is
        sensitive to arbitrary choices about model specification.

        Args:
            estimator: The estimator to re-run.
            original_effect: The estimate being tested.

        Returns:
            The outcome, which fails when the estimate moves materially.
        """
        rng = np.random.default_rng(self.seed + 1)
        data = self.df.copy()
        data["_random_covariate"] = rng.normal(0, 1, len(data))

        try:
            new_effect = self._estimate_on(
                estimator, data, [*self.covariates, "_random_covariate"]
            )
        except Exception as exc:
            return RefutationOutcome(
                test_name="random_common_cause",
                original_effect=original_effect,
                new_effect=float("nan"),
                passed=False,
                interpretation=f"The check could not be completed: {exc}",
            )

        tolerance = max(abs(original_effect) * PERTURBATION_TOLERANCE, MIN_TOLERANCE)
        shift = abs(new_effect - original_effect)
        passed = shift <= tolerance

        interpretation = (
            f"Adding an irrelevant variable moves the estimate by {shift:.3f}, "
            + (
                "which is small enough that the result does not hinge on model choice."
                if passed
                else "which is larger than it should be for a variable that carries no "
                "information, so the estimate is sensitive to specification."
            )
        )

        return RefutationOutcome(
            test_name="random_common_cause",
            original_effect=original_effect,
            new_effect=new_effect,
            passed=passed,
            interpretation=interpretation,
        )

    def data_subset(
        self,
        estimator: Estimator,
        original_effect: float,
        fraction: float = 0.8,
        n_trials: int = 10,
    ) -> RefutationOutcome:
        """Re-estimate on random subsets and confirm the answer is stable.

        An estimate driven by a genuine relationship reappears in most of the
        data. One that depends on a handful of unusual units swings wildly when
        those units are dropped.

        Args:
            estimator: The estimator to re-run.
            original_effect: The estimate being tested.
            fraction: Share of rows kept in each subset.
            n_trials: Number of subsets to draw.

        Returns:
            The outcome, which fails when subset estimates scatter widely.
        """
        rng = np.random.default_rng(self.seed + 2)
        estimates: list[float] = []

        for _ in range(n_trials):
            sample = self.df.sample(frac=fraction, random_state=int(rng.integers(0, 1_000_000)))
            try:
                estimates.append(self._estimate_on(estimator, sample.reset_index(drop=True), self.covariates))
            except Exception:
                continue

        if len(estimates) < 3:
            return RefutationOutcome(
                test_name="data_subset",
                original_effect=original_effect,
                new_effect=float("nan"),
                passed=False,
                interpretation="Too few subsets could be estimated to judge stability.",
            )

        values = np.asarray(estimates)
        mean_estimate = float(values.mean())
        spread = float(values.std(ddof=1))
        tolerance = max(abs(original_effect) * PERTURBATION_TOLERANCE, MIN_TOLERANCE)
        passed = abs(mean_estimate - original_effect) <= tolerance

        interpretation = (
            f"Across {len(values)} random subsets the estimate averages "
            f"{mean_estimate:.3f} with a spread of {spread:.3f}, "
            + (
                "so the result does not depend on any particular part of the data."
                if passed
                else "which drifts from the full-sample estimate and suggests the result "
                "is driven by a subset of the data rather than a stable relationship."
            )
        )

        return RefutationOutcome(
            test_name="data_subset",
            original_effect=original_effect,
            new_effect=mean_estimate,
            passed=passed,
            interpretation=interpretation,
        )

    def add_unobserved_confounder(
        self,
        estimator: Estimator,
        original_effect: float,
        strength: float = 1.0,
    ) -> RefutationOutcome:
        """Simulate a missing confounder and report how far the estimate moves.

        A confounder is injected that influences both treatment and outcome, then
        the estimate is recomputed without adjusting for it. This does not test
        whether such a variable exists; it quantifies how much damage one of that
        strength would do.

        Args:
            estimator: The estimator to re-run.
            original_effect: The estimate being tested.
            strength: How strongly the simulated confounder acts on both sides.

        Returns:
            The outcome, which fails when a confounder of this strength would
            overturn the conclusion.
        """
        rng = np.random.default_rng(self.seed + 3)
        data = self.df.copy()

        hidden = rng.normal(0, 1, len(data))
        treatment_values = data[self.treatment].to_numpy()
        # Correlate the hidden variable with treatment, then let it push the
        # outcome in the same direction, which is how real confounding operates.
        hidden = hidden + strength * (treatment_values - treatment_values.mean())
        data[self.outcome] = data[self.outcome].to_numpy() + strength * hidden

        try:
            new_effect = self._estimate_on(estimator, data, self.covariates)
        except Exception as exc:
            return RefutationOutcome(
                test_name="add_unobserved_confounder",
                original_effect=original_effect,
                new_effect=float("nan"),
                passed=False,
                interpretation=f"The check could not be completed: {exc}",
            )

        sign_flipped = np.sign(new_effect) != np.sign(original_effect)
        passed = not sign_flipped

        if passed:
            interpretation = (
                f"An unmeasured confounder of strength {strength:.1f} would move the "
                f"estimate to {new_effect:.3f} without reversing its direction."
            )
        else:
            interpretation = (
                f"An unmeasured confounder of strength {strength:.1f} would reverse "
                f"the conclusion, moving the estimate from {original_effect:.3f} to "
                f"{new_effect:.3f}. The finding is fragile to confounding of a "
                "plausible size."
            )

        return RefutationOutcome(
            test_name="add_unobserved_confounder",
            original_effect=original_effect,
            new_effect=new_effect,
            passed=passed,
            interpretation=interpretation,
        )

    def run_all(self, estimator: Estimator, original_effect: float) -> dict:
        """Run every check and summarise how well the estimate held up.

        Args:
            estimator: The estimator to re-run under each manipulation.
            original_effect: The estimate being tested.

        Returns:
            Dictionary with each ``RefutationOutcome``, counts of how many
            passed, an overall ``verdict``, and a plain-English ``summary``. The
            verdict is ``failed`` whenever the placebo test fails, regardless of
            the other results.
        """
        results = [
            self.placebo_treatment(estimator, original_effect),
            self.random_common_cause(estimator, original_effect),
            self.data_subset(estimator, original_effect),
            self.add_unobserved_confounder(estimator, original_effect),
        ]

        n_passed = sum(1 for r in results if r.passed)
        placebo = next(r for r in results if r.test_name == "placebo_treatment")

        # An estimate indistinguishable from what random assignment produces is a
        # null result, not a robust finding. Reporting it as "robust" would invite
        # the reader to act on an effect the analysis did not detect.
        is_null = abs(original_effect) <= abs(placebo.new_effect) + 1e-9

        if not placebo.passed:
            verdict = "failed"
            summary = (
                "This estimate failed the placebo test: effects of a similar size "
                "appear even when the treatment is randomly assigned, so it cannot "
                "be interpreted causally regardless of the other checks."
            )
        elif is_null:
            verdict = "no_effect"
            summary = (
                "The estimate is no larger than what random treatment assignment "
                "produces, so this analysis found no detectable effect. That is a "
                "conclusion in its own right, not a failure of the method."
            )
        elif n_passed == len(results):
            verdict = "robust"
            summary = (
                f"The estimate survived all {len(results)} robustness checks, "
                "including the placebo test."
            )
        else:
            verdict = "fragile"
            failed = ", ".join(r.test_name for r in results if not r.passed)
            summary = (
                f"The estimate passed the placebo test but failed {failed}, so it "
                "should be treated as provisional."
            )

        return {
            "results": results,
            "n_passed": n_passed,
            "n_total": len(results),
            "verdict": verdict,
            "summary": summary,
        }
