"""Did the discount actually cause the extra spend, and who should receive it?

The discount was sent to the customers most likely to respond, who were already
the highest spenders. A raw comparison therefore credits the discount with
behaviour those customers would have shown anyway. This example recovers the
real effect, then goes further and asks which customers are worth treating,
because an average effect hides that some customers respond far more than others
and some are made worse off.

Run with: python -m examples.marketing_uplift
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.generate import load_dataset
from estimation.propensity import InverseProbabilityWeighting, PropensityScoreMatching
from estimation.uplift import XLearner, cate_by_segment, top_k_targeting, uplift_curve
from reporting.interpret import headline, interpret_heterogeneity


def main() -> None:
    """Run the marketing example and print its findings."""
    print("=" * 78)
    print("Did the discount cause the extra spend?")
    print("=" * 78)

    df, truth = load_dataset("confounded")
    covariates = truth["covariates"]
    true_effect = truth["true_ate"]

    print()
    print("The setup")
    print("-" * 78)
    print(f"  {len(df):,} customers, of whom {int(df['treatment'].sum()):,} received the discount.")
    print(f"  {truth['why']}")

    print()
    print("What a raw comparison says")
    print("-" * 78)
    print(f"  Treated customers spent {truth['naive_estimate']:.2f} more on average.")
    print(f"  The true effect built into this data is {true_effect:.2f},")
    print(f"  so the raw comparison overstates it by {truth['naive_estimate'] - true_effect:.2f}.")

    print()
    print("After adjusting for who was targeted")
    print("-" * 78)
    results = []
    for estimator in (
        PropensityScoreMatching(n_boot=60),
        InverseProbabilityWeighting(n_boot=60),
    ):
        result = estimator.estimate(df, "treatment", "outcome", covariates)
        results.append(result)
        removed = (
            1 - abs(result.point_estimate - true_effect) / abs(truth["naive_estimate"] - true_effect)
        ) * 100
        print(
            f"  {result.method:4} {result.point_estimate:6.3f}  "
            f"[{result.ci_low:.3f}, {result.ci_high:.3f}]  "
            f"removes {removed:.0f}% of the bias"
        )

    print()
    print(f"  {headline(results[0], 'The discount', 'spend')}")

    # The average effect answers "did it work". The next question, and the one
    # that changes what a marketing team actually does, is "for whom".
    print()
    print("=" * 78)
    print("Who is worth treating?")
    print("=" * 78)

    df_seg, truth_seg = load_dataset("heterogeneous")
    learner = XLearner(n_boot=0)
    cate = learner.predict_cate(df_seg, "treatment", "outcome", truth_seg["covariates"])
    segments = cate_by_segment(cate, df_seg, truth_seg["segment_col"])

    print()
    print("  Effect by customer segment")
    print("  " + "-" * 74)
    for _, row in segments.iterrows():
        true_segment = truth_seg["true_segment_effects"][row["segment"]]
        flag = "   <- made worse off" if row["harmed"] else ""
        print(
            f"    {row['segment']}: estimated {row['mean_effect']:+.2f} "
            f"(true {true_segment:+.1f}){flag}"
        )

    print()
    print(f"  {interpret_heterogeneity(segments, 'spend')}")

    print()
    print("  Targeting")
    print("  " + "-" * 74)
    for share in (10, 30, 50):
        targeting = top_k_targeting(cate, share)
        print(
            f"    Top {share:2}%: average effect {targeting['mean_effect_targeted']:+.2f} "
            f"against {targeting['mean_effect_all']:+.2f} for treating everyone, "
            f"capturing {targeting['pct_of_benefit_captured']:.0f}% of the benefit"
        )

    curve = uplift_curve(
        cate,
        df_seg["outcome"].to_numpy(),
        df_seg["treatment"].to_numpy(),
    )
    print()
    print(f"  {curve['interpretation']}")

    print()
    print("=" * 78)
    print("What this changes")
    print("=" * 78)
    print("  The raw comparison would have credited the discount with roughly three")
    print("  times its real effect. Correcting that matters, but the segment view")
    print("  matters more: sending the discount to everyone would harm a quarter of")
    print("  customers, which no average effect would ever reveal.")


if __name__ == "__main__":
    main()
