"""Charts for reading a causal analysis.

Every figure answers one question and is built to be legible in both the web
interface and a printed report, so the palette is fixed, the chrome is recessive,
and identity is never carried by colour alone.

Colour follows meaning rather than magnitude: the two-hue categorical pair marks
identity (before against after, treated against untreated), while the diverging
pair is reserved for polarity, where a negative effect genuinely means the
opposite of a positive one. Charts return figures rather than writing files, so
the same code serves the interface and the PDF exporter.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

# Categorical slots for identity, validated for colour-vision deficiency
# separation and contrast against the light surface.
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"

# Diverging poles, used only where sign carries meaning.
POSITIVE = "#2a78d6"
NEGATIVE = "#e34948"

# Status colours, always paired with a text label since two of them fall below
# the contrast floor on a light surface by design.
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e5e4e0"


def _style_axes(ax: plt.Axes, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    """Apply the shared chart styling: recessive chrome, text in ink colours."""
    ax.set_facecolor(SURFACE)
    ax.figure.patch.set_facecolor(SURFACE)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
        ax.spines[spine].set_linewidth(1.0)

    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)

    if title:
        ax.set_title(title, color=INK_PRIMARY, fontsize=12, fontweight="600", loc="left", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)


def plot_forest(
    results: list,
    true_effect: float | None = None,
    title: str = "Estimated effect by method",
    units: str = "",
) -> Figure:
    """Compare every method's estimate and interval on one scale.

    This is the chart that shows whether methods relying on different assumptions
    reached the same answer. Agreement is evidence; disagreement means the
    assumptions are doing the work.

    Args:
        results: Estimates to compare, each exposing ``method``,
            ``point_estimate``, ``ci_low``, ``ci_high``, and ``estimate_type``.
        true_effect: Known effect to mark, where the data has one.
        title: Chart title.
        units: Optional unit label for the axis.

    Returns:
        The assembled figure.
    """
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.75 * len(results) + 1.5)))
    positions = np.arange(len(results))[::-1]

    for pos, result in zip(positions, results):
        ax.plot(
            [result.ci_low, result.ci_high],
            [pos, pos],
            color=SERIES_1,
            linewidth=2.0,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(
            result.point_estimate,
            pos,
            marker="o",
            markersize=9,
            color=SERIES_1,
            markeredgecolor=SURFACE,
            markeredgewidth=2.0,
            zorder=3,
        )
        # Direct labels remove the need to read values off the axis.
        ax.text(
            result.ci_high,
            pos + 0.22,
            f"  {result.point_estimate:,.3f}",
            color=INK_PRIMARY,
            fontsize=9,
            va="center",
        )

    ax.axvline(0, color=INK_MUTED, linewidth=1.2, linestyle="-", zorder=1)
    if true_effect is not None:
        ax.axvline(true_effect, color=NEGATIVE, linewidth=1.6, linestyle="--", zorder=1)
        ax.text(
            true_effect,
            len(results) - 0.4,
            " true effect",
            color=NEGATIVE,
            fontsize=9,
            va="bottom",
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(
        [f"{r.method}  ({r.estimate_type.upper()})" for r in results], fontsize=10
    )
    ax.set_ylim(-0.7, len(results) - 0.1)
    _style_axes(ax, xlabel=f"Effect on the outcome{f' ({units})' if units else ''}", title=title)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_love(
    smd_before: pd.DataFrame,
    smd_after: pd.DataFrame | None = None,
    threshold: float = 0.1,
    title: str = "Covariate balance",
) -> Figure:
    """Show how far apart the groups were on each covariate, before and after.

    Args:
        smd_before: Standardised mean differences before adjustment.
        smd_after: The same after adjustment, where available.
        threshold: Conventional imbalance threshold to mark.
        title: Chart title.

    Returns:
        The assembled figure.
    """
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.45 * len(smd_before) + 1.8)))
    order = smd_before.sort_values("abs_smd")
    positions = np.arange(len(order))

    ax.scatter(
        order["abs_smd"],
        positions,
        s=70,
        color=SERIES_1,
        edgecolor=SURFACE,
        linewidth=1.5,
        label="Before adjustment",
        zorder=3,
    )
    if smd_after is not None and not smd_after.empty:
        aligned = smd_after.set_index("covariate").reindex(order["covariate"])
        ax.scatter(
            aligned["abs_smd"].to_numpy(),
            positions,
            s=70,
            color=SERIES_2,
            edgecolor=SURFACE,
            linewidth=1.5,
            label="After adjustment",
            zorder=3,
        )

    ax.axvline(threshold, color=INK_MUTED, linewidth=1.2, linestyle="--", zorder=1)
    # Anchored in axes coordinates so the note stays inside the plot area rather
    # than drifting above the title as the number of covariates changes.
    ax.text(
        threshold,
        0.97,
        f"  balanced below {threshold}",
        transform=ax.get_xaxis_transform(),
        color=INK_SECONDARY,
        fontsize=9,
        va="top",
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(order["covariate"], fontsize=10)
    _style_axes(ax, xlabel="Absolute standardised difference", title=title)
    ax.grid(axis="y", visible=False)
    legend = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    fig.tight_layout()
    return fig


def plot_propensity_overlap(
    propensity_scores: np.ndarray,
    treatment: np.ndarray,
    title: str = "Overlap between the groups",
) -> Figure:
    """Show whether comparable units exist in both arms.

    Where the two distributions do not overlap there is no one to compare
    against, so no method can estimate an effect for those units.

    Args:
        propensity_scores: Estimated probability of treatment per unit.
        treatment: Binary treatment indicator per unit.
        title: Chart title.

    Returns:
        The assembled figure.
    """
    fig, ax = plt.subplots(figsize=(8, 3.6))
    scores = np.asarray(propensity_scores)
    treated = np.asarray(treatment) == 1
    bins = np.linspace(0, 1, 41)

    ax.hist(scores[~treated], bins=bins, color=SERIES_1, alpha=0.75, label="Untreated", zorder=2)
    ax.hist(scores[treated], bins=bins, color=SERIES_2, alpha=0.75, label="Treated", zorder=2)

    _style_axes(ax, xlabel="Probability of receiving treatment", ylabel="Units", title=title)
    legend = ax.legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    fig.tight_layout()
    return fig


def plot_segment_effects(
    segment_table: pd.DataFrame,
    title: str = "Effect by segment",
) -> Figure:
    """Show how the effect differs across groups, marking any that are harmed.

    Colour encodes sign rather than size, because a negative effect means the
    opposite of a positive one rather than merely less of it.

    Args:
        segment_table: Per-segment effects with confidence bounds and a
            ``harmed`` flag.
        title: Chart title.

    Returns:
        The assembled figure.
    """
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.6 * len(segment_table) + 1.6)))
    order = segment_table.sort_values("mean_effect")
    positions = np.arange(len(order))
    colors = [NEGATIVE if value < 0 else POSITIVE for value in order["mean_effect"]]

    ax.barh(positions, order["mean_effect"], color=colors, height=0.55, zorder=2)
    ax.errorbar(
        order["mean_effect"],
        positions,
        xerr=[
            order["mean_effect"] - order["ci_low"],
            order["ci_high"] - order["mean_effect"],
        ],
        fmt="none",
        ecolor=INK_SECONDARY,
        elinewidth=1.4,
        capsize=3,
        zorder=3,
    )
    ax.axvline(0, color=INK_MUTED, linewidth=1.2, zorder=1)

    for pos, (_, row) in zip(positions, order.iterrows()):
        offset = 0.02 * max(abs(order["mean_effect"]).max(), 1e-9)
        align = "left" if row["mean_effect"] >= 0 else "right"
        # Harmed segments are named in text as well as coloured, so the warning
        # never depends on the reader distinguishing two hues.
        label = f"{row['mean_effect']:+.2f}" + ("  harmed" if row["harmed"] else "")
        ax.text(
            row["mean_effect"] + (offset if row["mean_effect"] >= 0 else -offset),
            pos,
            label,
            color=INK_PRIMARY,
            fontsize=9,
            va="center",
            ha=align,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(order["segment"], fontsize=10)
    _style_axes(ax, xlabel="Estimated effect", title=title)
    ax.grid(axis="y", visible=False)
    # Generous horizontal margin so the value labels sitting outside each bar end
    # never collide with the axis or the segment names.
    ax.margins(x=0.30)
    fig.tight_layout()
    return fig


def plot_cate_distribution(
    cate: np.ndarray,
    title: str = "Distribution of individual effects",
) -> Figure:
    """Show the spread of predicted effects and how many units are harmed.

    Args:
        cate: Per-unit predicted effects.
        title: Chart title.

    Returns:
        The assembled figure.
    """
    fig, ax = plt.subplots(figsize=(8, 3.6))
    values = np.asarray(cate)

    counts, bins, patches = ax.hist(values, bins=40, color=POSITIVE, zorder=2)
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < 0:
            patch.set_facecolor(NEGATIVE)

    ax.axvline(0, color=INK_MUTED, linewidth=1.2, zorder=3)
    pct_harmed = float(np.mean(values < 0) * 100)
    if pct_harmed > 0:
        ax.text(
            0.02,
            0.94,
            f"{pct_harmed:.0f}% of units are predicted to be worse off",
            transform=ax.transAxes,
            color=INK_PRIMARY,
            fontsize=9,
            va="top",
        )

    _style_axes(ax, xlabel="Predicted effect for an individual unit", ylabel="Units", title=title)
    fig.tight_layout()
    return fig


def plot_uplift_curve(curve: dict, title: str = "Value of targeting by predicted effect") -> Figure:
    """Compare targeting the most responsive units against treating at random.

    Args:
        curve: Output of the uplift curve calculation.
        title: Chart title.

    Returns:
        The assembled figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    fractions = np.asarray(curve["fractions"]) * 100

    ax.plot(fractions, curve["gains"], color=SERIES_1, linewidth=2.0, label="Targeted by predicted effect", zorder=3)
    ax.plot(fractions, curve["random_line"], color=SERIES_2, linewidth=2.0, linestyle="--", label="Random selection", zorder=2)

    _style_axes(ax, xlabel="Share of population treated (%)", ylabel="Cumulative gain", title=title)
    legend = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    fig.tight_layout()
    return fig


def plot_event_study(event_table: pd.DataFrame, treat_period: int, title: str = "Group gap over time") -> Figure:
    """Show the gap between groups in every period relative to treatment.

    A flat line before the intervention is the visual evidence that the design
    holds; a drift beforehand is evidence that it does not.

    Args:
        event_table: Per-period gaps with confidence bounds.
        treat_period: First treated period.
        title: Chart title.

    Returns:
        The assembled figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    periods = event_table["period"].to_numpy()

    ax.fill_between(
        periods,
        event_table["ci_low"],
        event_table["ci_high"],
        color=SERIES_1,
        alpha=0.18,
        zorder=1,
    )
    ax.plot(periods, event_table["gap"], color=SERIES_1, linewidth=2.0, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)

    ax.axhline(0, color=INK_MUTED, linewidth=1.2, zorder=2)
    ax.axvline(treat_period - 0.5, color=NEGATIVE, linewidth=1.6, linestyle="--", zorder=2)
    ax.text(
        treat_period - 0.45,
        0.97,
        " treatment begins",
        transform=ax.get_xaxis_transform(),
        color=NEGATIVE,
        fontsize=9,
        va="top",
    )

    _style_axes(ax, xlabel="Period", ylabel="Gap between groups", title=title)
    fig.tight_layout()
    return fig


def plot_sensitivity(grid: pd.DataFrame, original_effect: float, title: str = "Sensitivity to hidden confounding") -> Figure:
    """Show how the estimate moves as a simulated hidden confounder strengthens.

    Args:
        grid: Output of the confounding grid sweep.
        original_effect: The estimate being tested.
        title: Chart title.

    Returns:
        The assembled figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [POSITIVE if preserved else NEGATIVE for preserved in grid["sign_preserved"]]

    ax.plot(grid["strength"], grid["estimate"], color=SERIES_1, linewidth=2.0, zorder=2)
    ax.scatter(grid["strength"], grid["estimate"], s=80, c=colors, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.axhline(0, color=INK_MUTED, linewidth=1.2, zorder=1)
    ax.axhline(original_effect, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=1)
    ax.text(grid["strength"].max(), original_effect, " original estimate", color=INK_SECONDARY, fontsize=9, va="bottom", ha="right")

    flipped = grid.loc[~grid["sign_preserved"]]
    if not flipped.empty:
        ax.text(
            0.02, 0.06,
            f"The conclusion reverses once a hidden confounder reaches strength {flipped.iloc[0]['strength']:.2f}",
            transform=ax.transAxes, color=INK_PRIMARY, fontsize=9,
        )

    _style_axes(ax, xlabel="Strength of simulated hidden confounder", ylabel="Estimated effect", title=title)
    fig.tight_layout()
    return fig


def plot_dag(dag, treatment: str, outcome: str, roles: dict | None = None) -> str:
    """Render the causal graph as an interactive page.

    Nodes are coloured by the role each variable plays, since that role decides
    whether it may be adjusted for.

    Args:
        dag: The causal graph to draw.
        treatment: Name of the treatment variable.
        outcome: Name of the outcome variable.
        roles: Optional mapping of variable to its classified role.

    Returns:
        A self-contained HTML document.
    """
    from pyvis.network import Network

    role_colors = {
        "confounder": SERIES_2,
        "mediator": STATUS["critical"],
        "collider": STATUS["serious"],
        "instrument": "#1baf7a",
        "outcome_predictor": INK_MUTED,
        "irrelevant": GRID,
    }

    net = Network(height="520px", width="100%", directed=True, bgcolor=SURFACE, font_color=INK_PRIMARY)
    net.set_options('{"physics": {"solver": "forceAtlas2Based", "stabilization": {"iterations": 150}}}')

    for node in dag.nodes:
        if node == treatment:
            color, shape, label = SERIES_1, "box", f"{node}\n(treatment)"
        elif node == outcome:
            color, shape, label = "#008300", "box", f"{node}\n(outcome)"
        else:
            role = (roles or {}).get(node, {}).get("role", "irrelevant")
            color, shape = role_colors.get(role, GRID), "dot"
            label = f"{node}\n({role})" if roles else node
        net.add_node(node, label=label, color=color, shape=shape, size=22)

    for cause, effect in dag.edges:
        net.add_edge(cause, effect, color=INK_MUTED, width=2)

    return net.generate_html()
