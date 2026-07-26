"""What an A/B test still leaves on the table, and what it protects you from.

A randomised experiment already gives an unbiased average effect, so nothing
here improves on it. What it adds is the question the average cannot answer:
whether the treatment works the same way for everyone. It then runs the opposite
case, where the treatment does nothing at all, because a tool that always finds
an effect is worse than no tool.

Run with: python -m examples.ab_extension
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from datasets.generate import load_dataset
from estimation.propensity import PropensityScoreMatching
from estimation.uplift import XLearner, cate_by_segment
from reporting.interpret import headline, interpret_refutations
from validation.refutation import Refuter


def main() -> None:
    """Run the experiment extension and the no-effect check."""
    print("=" * 78)
    print("Beyond the average an A/B test reports")
    print("=" * 78)

    df, truth = load_dataset("heterogeneous")
    covariates = truth["covariates"]

    print()
    print("  Treatment here was randomised, so the raw difference of "
          f"{truth['naive_estimate']:.2f}")
    print(f"  is already an unbiased estimate of the true {truth['true_ate']:.2f}.")
    print("  There is no confounding to remove, and nothing to correct.")

    learner = XLearner(n_boot=0)
    cate = learner.predict_cate(df, "treatment", "outcome", covariates)
    segments = cate_by_segment(cate, df, truth["segment_col"])

    print()
    print("  What the average conceals")
    print("  " + "-" * 74)
    for _, row in segments.iterrows():
        true_segment = truth["true_segment_effects"][row["segment"]]
        flag = "   <- made worse off" if row["harmed"] else ""
        print(
            f"    Segment {row['segment']} ({int(row['n']):,} units): "
            f"{row['mean_effect']:+.2f} (true {true_segment:+.1f}){flag}"
        )

    harmed = segments[segments["harmed"]]
    share_harmed = float(np.mean(cate < 0) * 100)
    print()
    print(f"    The experiment's headline is a positive {truth['true_ate']:.2f}.")
    print(f"    Roughly {share_harmed:.0f}% of units are predicted to be worse off, "
          "concentrated")
    print(f"    in segment {', '.join(harmed['segment'].astype(str))}. Rolling the "
          "treatment out to")
    print("    everyone on the strength of the average would harm them.")

    print()
    print("=" * 78)
    print("The opposite case: a treatment that does nothing")
    print("=" * 78)

    df_null, truth_null = load_dataset("no_effect")
    print()
    print(f"  The true effect is exactly {truth_null['true_ate']:.1f}, but a raw comparison")
    print(f"  reports {truth_null['naive_estimate']:.2f}.")
    print(f"  {truth_null['why']}")

    estimator = PropensityScoreMatching(n_boot=60)
    result = estimator.estimate(df_null, "treatment", "outcome", truth_null["covariates"])

    print()
    print("  After adjustment")
    print("  " + "-" * 74)
    print(
        f"    {result.point_estimate:.3f}  [{result.ci_low:.3f}, {result.ci_high:.3f}]  "
        f"significant: {result.significant()}"
    )
    print()
    print(f"  {headline(result, 'The programme', 'the outcome')}")

    refuter = Refuter(df_null, "treatment", "outcome", truth_null["covariates"])
    summary = refuter.run_all(PropensityScoreMatching(n_boot=25), result.point_estimate)

    print()
    print("  Under scrutiny")
    print("  " + "-" * 74)
    for outcome_row in summary["results"]:
        state = "passed" if outcome_row.passed else "not met"
        print(f"    {outcome_row.test_name:26} {state}")
    print()
    print(f"  {interpret_refutations(summary)}")

    print()
    print("=" * 78)
    print("What this changes")
    print("=" * 78)
    print("  An experiment answers whether the treatment worked on average. It does")
    print("  not say who it worked for, and the average can be positive while a")
    print("  quarter of the population is harmed. The second case matters just as")
    print("  much: an engine that reports an effect on data where none exists")
    print("  cannot be trusted on data where the answer is unknown.")


if __name__ == "__main__":
    main()
