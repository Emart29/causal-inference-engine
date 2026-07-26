"""Did the policy change work, and how would we know if it had not?

Policies arrive at a known moment for one group and not another, which allows
the change in the treated group to be compared against the change in the
untreated one. The comparison only means something if the two groups were moving
together beforehand, so this example checks that first, then repeats the analysis
on data where they were not, to show the engine declining to claim an effect.

Run with: python -m examples.policy_effect
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.generate import load_dataset
from estimation.did import DifferenceInDifferences, event_study, parallel_trends_test
from reporting.interpret import interpret_effect


def _report_case(label: str, df, truth, note: str) -> None:
    """Run the analysis on one panel and print what it supports."""
    print()
    print("=" * 78)
    print(label)
    print("=" * 78)
    print(f"  {note}")

    time_col = truth["time_col"]
    group_col = truth["group_col"]
    treat_period = truth["treat_period"]

    trends = parallel_trends_test(df, "outcome", time_col, group_col, treat_period)
    print()
    print("  Before anything else: were the groups moving together?")
    print("  " + "-" * 74)
    print(f"    {trends['interpretation']}")

    if not trends["passed"]:
        # The estimate is still computed, but reporting it as the effect of the
        # policy would be the mistake this check exists to prevent.
        print()
        print("    The comparison this design relies on does not hold here, so no")
        print("    effect is reported. A number could be produced, but it would")
        print("    measure the pre-existing divergence rather than the policy.")
        return

    result = DifferenceInDifferences(unit_col="unit").estimate(
        df,
        "treatment",
        "outcome",
        [],
        group_col=group_col,
        time_col=time_col,
        treat_period=treat_period,
    )

    print()
    print("  The estimate")
    print("  " + "-" * 74)
    print(
        f"    {result.point_estimate:.3f}  [{result.ci_low:.3f}, {result.ci_high:.3f}]  "
        f"against a true effect of {truth['true_ate']:.2f}"
    )
    print(f"    A raw comparison would have reported {truth['naive_estimate']:.2f}.")
    print(f"    Standard errors: {result.diagnostics['standard_errors']}.")

    print()
    print(f"  {interpret_effect(result, 'The policy', 'the outcome')}")

    events = event_study(df, "outcome", time_col, group_col, treat_period)
    print()
    print("  Gap between the groups, period by period")
    print("  " + "-" * 74)
    for _, row in events.iterrows():
        marker = "before" if row["is_pre_treatment"] else "after "
        bar = "#" * max(int(abs(row["gap"]) * 6), 0)
        print(f"    period {int(row['period'])} ({marker}): {row['gap']:+6.2f}  {bar}")
    print()
    print("    Flat before the policy and stepping up afterwards is the evidence")
    print("    that the design holds; a drift beforehand would undermine it.")


def main() -> None:
    """Run the policy example on a sound design and on a broken one."""
    df, truth = load_dataset("did_panel")
    _report_case(
        "A policy change where the design holds",
        df,
        truth,
        "Treated units start from a higher baseline, so levels cannot be compared "
        "directly, but their trends can.",
    )

    # The same data with a trend added to the treated group only. Nothing about
    # the policy changed; only the assumption the design rests on is now false.
    broken = df.copy()
    broken["outcome"] = broken["outcome"] + broken["treated_unit"] * broken["period"] * 1.2
    _report_case(
        "The same policy where the design does not hold",
        broken,
        truth,
        "The treated group was already pulling away before the policy arrived, "
        "for reasons that have nothing to do with it.",
    )

    print()
    print("=" * 78)
    print("What this changes")
    print("=" * 78)
    print("  Both panels produce a number. Only one of them supports a claim about")
    print("  the policy. The difference is not visible in the estimate or its")
    print("  confidence interval, which is why the assumption is tested rather")
    print("  than assumed.")


if __name__ == "__main__":
    main()
