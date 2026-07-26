"""Interface for running a causal analysis and reading its results.

The page order is the argument this project makes. A reader who meets an effect
size before the conditions under which it holds tends to keep the number and
forget the caveats, so results stay behind the assumptions page until those have
been seen. The graph page comes earlier still, because which variables may be
adjusted for is decided there and a wrong choice invalidates everything after it.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from datasets.generate import list_datasets, load_dataset
from discovery.dag_builder import CausalDAG, auto_dag
from discovery.confounder import classify_covariates, identify_backdoor_set
from estimation.run import ADJUSTMENT_METHODS, AnalysisRunner
from estimation.uplift import XLearner, cate_by_segment, top_k_targeting, uplift_curve
from reporting import visualizer as viz
from reporting.pdf_export import export_pdf
from reporting.report import ReportBuilder

PAGES = [
    "1. Data and question",
    "2. Causal graph",
    "3. Assumptions",
    "4. Results",
    "5. Report",
]

ROLE_BADGES = {
    "confounder": ("Adjust for it", "normal"),
    "mediator": ("Do not adjust", "inverse"),
    "collider": ("Do not adjust", "inverse"),
    "instrument": ("Use as instrument", "off"),
    "outcome_predictor": ("Optional", "off"),
    "irrelevant": ("Ignore", "off"),
}


def _init_state() -> None:
    """Seed the session with the keys every page expects."""
    defaults = {
        "df": None,
        "ground_truth": None,
        "dataset_name": None,
        "treatment": None,
        "outcome": None,
        "covariates": [],
        "instrument": None,
        "segment_col": None,
        "dag": None,
        "report": None,
        "assumptions_seen": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _require(*keys: str) -> bool:
    """Return whether the session holds everything a page needs."""
    return all(st.session_state.get(key) is not None for key in keys)


def page_data() -> None:
    """Choose a dataset and name the treatment, outcome, and covariates."""
    st.header("Data and question", anchor="data-and-question")
    st.write(
        "Start from a built-in dataset whose true effect is known, or upload your "
        "own. The built-in ones exist so the engine's answers can be checked "
        "against an answer that is known by construction."
    )

    source = st.radio("Source", ["Built-in dataset", "Upload a CSV"], horizontal=True)

    if source == "Built-in dataset":
        options = list_datasets()
        name = st.selectbox(
            "Dataset",
            list(options),
            format_func=lambda key: f"{key} — {options[key]}",
        )
        if st.button("Load dataset", type="primary"):
            df, truth = load_dataset(name)
            st.session_state.update(
                df=df,
                ground_truth=truth,
                dataset_name=name,
                treatment=truth["treatment"],
                outcome=truth["outcome"],
                covariates=truth.get("covariates", []),
                instrument=truth.get("instrument"),
                segment_col=truth.get("segment_col"),
                dag=None,
                report=None,
                assumptions_seen=False,
            )
            st.success(f"Loaded {name}: {len(df):,} rows.")
    else:
        upload = st.file_uploader("CSV file", type=["csv"])
        if upload is not None:
            df = pd.read_csv(upload)
            st.session_state.update(
                df=df,
                ground_truth=None,
                dataset_name=upload.name,
                dag=None,
                report=None,
                assumptions_seen=False,
            )
            st.success(f"Loaded {upload.name}: {len(df):,} rows.")

    df = st.session_state.df
    if df is None:
        return

    st.subheader("Preview")
    st.dataframe(df.head(10), width="stretch")

    st.subheader("Define the question")
    columns = list(df.columns)
    col_left, col_right = st.columns(2)
    with col_left:
        treatment = st.selectbox(
            "Treatment", columns, index=columns.index(st.session_state.treatment)
            if st.session_state.treatment in columns else 0,
            help="The intervention whose effect you want to measure. Must be 0 or 1.",
        )
        outcome = st.selectbox(
            "Outcome", columns, index=columns.index(st.session_state.outcome)
            if st.session_state.outcome in columns else len(columns) - 1,
            help="What the treatment is supposed to change.",
        )
    with col_right:
        covariates = st.multiselect(
            "Covariates",
            [c for c in columns if c not in (treatment, outcome)],
            default=[c for c in st.session_state.covariates if c in columns],
            help="Variables that may explain who got treated. The next page decides "
                 "which of these are safe to adjust for.",
        )
        instrument = st.selectbox(
            "Instrument (optional)",
            ["None"] + [c for c in columns if c not in (treatment, outcome)],
            index=0 if not st.session_state.instrument
            else (["None"] + columns).index(st.session_state.instrument),
            help="A variable that shifts the treatment but has no other route to "
                 "the outcome. Only needed when confounders are unmeasured.",
        )

    unique_treatment = df[treatment].dropna().unique()
    if len(unique_treatment) != 2:
        st.error(
            f"'{treatment}' takes {len(unique_treatment)} distinct values. The "
            "treatment must be binary for these methods to apply."
        )
        return

    st.session_state.update(
        treatment=treatment,
        outcome=outcome,
        covariates=covariates,
        instrument=None if instrument == "None" else instrument,
    )
    st.info("Question set. Move to the causal graph to decide what may be adjusted for.")


def page_graph() -> None:
    """Review the assumed causal structure and the resulting adjustment set."""
    st.header("Causal graph", anchor="causal-graph")
    if not _require("df", "treatment", "outcome"):
        st.warning("Load a dataset and define the question first.")
        return

    st.write(
        "This graph is an assumption, not a discovery. Causal structure cannot be "
        "learned from observational data alone, so the default assumes every "
        "covariate causes both the treatment and the outcome. Correct it where you "
        "know better, because what may be adjusted for is decided here."
    )

    treatment = st.session_state.treatment
    outcome = st.session_state.outcome
    covariates = st.session_state.covariates

    if st.session_state.dag is None:
        st.session_state.dag = auto_dag(
            treatment, outcome, covariates, instrument=st.session_state.instrument
        )
    dag: CausalDAG = st.session_state.dag

    with st.expander("Edit the graph"):
        nodes = dag.nodes
        edit_left, edit_right, edit_action = st.columns([2, 2, 1])
        with edit_left:
            cause = st.selectbox("Cause", nodes, key="edge_cause")
        with edit_right:
            effect = st.selectbox("Effect", nodes, key="edge_effect")
        with edit_action:
            st.write("")
            if st.button("Add"):
                try:
                    dag.add_edge(cause, effect)
                    st.session_state.report = None
                except ValueError as exc:
                    st.error(str(exc))
            if st.button("Remove"):
                dag.remove_edge(cause, effect)
                st.session_state.report = None

    validation = dag.validate()
    if not validation["is_dag"]:
        st.error(validation["message"])
        return

    classification = classify_covariates(dag, treatment, outcome, covariates)
    backdoor = identify_backdoor_set(dag, treatment, outcome)

    st.subheader("What each variable does")
    for name, info in classification["roles"].items():
        label, _ = ROLE_BADGES.get(info["role"], ("", "off"))
        with st.container(border=True):
            st.markdown(f"**{name}** — {info['role'].replace('_', ' ')} · _{label}_")
            st.caption(info["guidance"])

    for warning in classification["warnings"]:
        st.warning(warning)

    st.subheader("Adjustment set")
    safe = classification["safe_adjustment_set"]
    st.success(
        f"Adjusting for: {', '.join(safe)}" if safe else "No covariates will be adjusted for."
    )
    st.caption(backdoor["reason"])

    st.subheader("Graph")
    st.components.v1.html(
        viz.plot_dag(dag, treatment, outcome, classification["roles"]), height=540
    )


def page_assumptions() -> None:
    """Run the analysis and show whether the data can support a causal estimate."""
    st.header("Assumptions", anchor="assumptions")
    if not _require("df", "treatment", "outcome"):
        st.warning("Load a dataset and define the question first.")
        return

    st.write(
        "These checks decide whether an effect estimate means anything. They run "
        "before any result is shown, because a number produced on data that cannot "
        "support a comparison is worse than no number at all."
    )

    thorough = st.checkbox(
        "Run every method and use more bootstrap resamples",
        value=False,
        help="Slower, and narrows the confidence intervals slightly. The point "
             "estimates do not change materially.",
    )

    # Only an explicit click starts the work. Running whenever no report exists
    # restarts the analysis on every rerun, so any interaction during the minute
    # it takes would begin it again from scratch.
    if st.button("Run the analysis", type="primary"):
        with st.spinner("Estimating, then attacking the estimate..."):
            runner = AnalysisRunner(n_boot=60 if thorough else 30)
            st.session_state.report = runner.run(
                st.session_state.df,
                st.session_state.treatment,
                st.session_state.outcome,
                st.session_state.covariates,
                dag=st.session_state.dag,
                instrument=st.session_state.instrument,
                segment_col=st.session_state.segment_col,
                methods=list(ADJUSTMENT_METHODS) if thorough else None,
            )

    report = st.session_state.report
    if report is None or report.assumption_checks is None:
        st.info(
            "Press **Run the analysis** to estimate the effect and test the "
            "assumptions behind it. This takes about a minute, because every "
            "estimate is bootstrapped and then attacked with robustness checks."
        )
        return

    checks = report.assumption_checks
    if checks["blocking_failures"]:
        st.error(report.narrative.get("assumptions", ""))
    elif checks["warnings"]:
        st.warning(report.narrative.get("assumptions", ""))
    else:
        st.success(report.narrative.get("assumptions", ""))

    st.subheader("Individual checks")
    for name, check in checks["checks"].items():
        with st.container(border=True):
            state = "passed" if check["passed"] else "not met"
            st.markdown(f"**{name.replace('_', ' ').title()}** — {state}")
            st.caption(check["interpretation"])

    balance = checks["checks"].get("balance", {})
    table = balance.get("smd_table")
    if table is not None and not table.empty:
        st.subheader("Covariate balance")
        st.pyplot(viz.plot_love(table), width="content")

    st.session_state.assumptions_seen = True
    st.info("Assumptions reviewed. The results page is now available.")


def page_results() -> None:
    """Show the estimates, robustness, sensitivity, and effect variation."""
    st.header("Results", anchor="results")
    if not st.session_state.assumptions_seen:
        st.warning(
            "Review the assumptions page first. Effect sizes are withheld until "
            "the conditions they depend on have been seen."
        )
        return

    report = st.session_state.report
    if report is None or not report.estimates:
        st.info("No estimates are available. Run the analysis on the assumptions page.")
        return

    if report.is_trustworthy:
        st.success(report.headline)
    else:
        st.error(report.headline)

    truth = st.session_state.ground_truth
    true_effect = truth.get("true_ate") if truth else None
    if true_effect is not None:
        st.caption(
            f"This dataset has a known true effect of {true_effect:,.3f}, marked on "
            "the chart so each method can be checked against it."
        )

    st.subheader("Estimates by method")
    st.pyplot(viz.plot_forest(report.estimates, true_effect=true_effect), width="content")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "method": r.method,
                    "applies to": r.estimate_type.upper(),
                    "estimate": round(r.point_estimate, 4),
                    "95% low": round(r.ci_low, 4),
                    "95% high": round(r.ci_high, 4),
                }
                for r in report.estimates
            ]
        ),
        width="stretch",
        hide_index=True,
        # Sized to the row count so every method is visible at once. The default
        # height clips the last row behind a scrollbar, which hides a result.
        height=(len(report.estimates) + 1) * 35 + 3,
    )
    st.caption(report.narrative.get("agreement", ""))

    if "robustness" in report.narrative:
        st.subheader("Does it survive scrutiny")
        primary = report.primary_estimate
        summary = report.refutations.get(primary.method, {})
        verdict = summary.get("verdict")
        if verdict == "failed":
            st.error(report.narrative["robustness"])
        elif verdict == "no_effect":
            st.info(report.narrative["robustness"])
        else:
            st.success(report.narrative["robustness"])
        for outcome_row in summary.get("results", []):
            with st.container(border=True):
                state = "passed" if outcome_row.passed else "not met"
                st.markdown(f"**{outcome_row.test_name.replace('_', ' ')}** — {state}")
                st.caption(outcome_row.interpretation)

    if "sensitivity" in report.narrative:
        st.subheader("What could overturn it")
        st.write(report.narrative["sensitivity"])

    if report.segments is not None and not report.segments.empty:
        st.subheader("Who it affects")
        st.write(report.narrative.get("heterogeneity", ""))
        st.pyplot(viz.plot_segment_effects(report.segments), width="content")

        learner = XLearner(n_boot=0)
        cate = learner.predict_cate(
            st.session_state.df,
            st.session_state.treatment,
            st.session_state.outcome,
            st.session_state.covariates,
        )
        st.pyplot(viz.plot_cate_distribution(cate), width="content")

        curve = uplift_curve(
            cate,
            st.session_state.df[st.session_state.outcome].to_numpy(),
            st.session_state.df[st.session_state.treatment].to_numpy(),
        )
        st.pyplot(viz.plot_uplift_curve(curve), width="content")
        st.caption(top_k_targeting(cate, 30)["interpretation"])


def page_report() -> None:
    """Preview the assembled report and export it as a PDF."""
    st.header("Report", anchor="report")
    report = st.session_state.report
    if report is None:
        st.info("Run the analysis first.")
        return

    st.markdown(ReportBuilder().to_markdown(report))

    st.subheader("Export")
    if st.button("Generate PDF", type="primary"):
        with st.spinner("Building the document..."):
            truth = st.session_state.ground_truth
            figures = {"forest": viz.plot_forest(
                report.estimates, true_effect=truth.get("true_ate") if truth else None
            )}
            balance = (report.assumption_checks or {}).get("checks", {}).get("balance", {})
            table = balance.get("smd_table")
            if table is not None and not table.empty:
                figures["balance"] = viz.plot_love(table)
            if report.segments is not None and not report.segments.empty:
                figures["segments"] = viz.plot_segment_effects(report.segments)

            path = os.path.join(os.getcwd(), "causal_report.pdf")
            export_pdf(report, path, figures)
            with open(path, "rb") as handle:
                st.download_button(
                    "Download the report",
                    data=handle.read(),
                    file_name="causal_report.pdf",
                    mime="application/pdf",
                )


def main() -> None:
    """Render the application."""
    st.set_page_config(page_title="Causal Inference Engine", layout="wide")
    _init_state()

    st.sidebar.title("Causal Inference Engine")
    st.sidebar.caption(
        "Estimate what actually caused what, with the assumptions and robustness "
        "checks attached."
    )
    page = st.sidebar.radio("Step", PAGES)

    if st.session_state.dataset_name:
        st.sidebar.divider()
        st.sidebar.write(f"**Dataset:** {st.session_state.dataset_name}")
        if st.session_state.treatment:
            st.sidebar.write(f"**Treatment:** {st.session_state.treatment}")
            st.sidebar.write(f"**Outcome:** {st.session_state.outcome}")

    if page == PAGES[0]:
        page_data()
    elif page == PAGES[1]:
        page_graph()
    elif page == PAGES[2]:
        page_assumptions()
    elif page == PAGES[3]:
        page_results()
    else:
        page_report()


if __name__ == "__main__":
    main()
