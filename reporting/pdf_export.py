"""Export of an assembled analysis as a PDF for people who did not run it.

The document is written for a reader who will not open the code and may not read
past the first page, so the cover states whether the analysis can be trusted
before it states any number, and every section leads with a sentence rather than
a table. Charts are embedded from the same functions the interface uses, so the
printed and on-screen versions cannot drift apart.
"""

from __future__ import annotations

import io

from matplotlib.figure import Figure
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from reporting.report import CausalReport

INK_PRIMARY = colors.HexColor("#0b0b0b")
INK_SECONDARY = colors.HexColor("#52514e")
RULE = colors.HexColor("#e5e4e0")
GOOD = colors.HexColor("#0ca30c")
CRITICAL = colors.HexColor("#d03b3b")


def _styles() -> dict:
    """Build the paragraph styles used throughout the document."""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontSize=20,
            leading=25,
            textColor=INK_PRIMARY,
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "heading": ParagraphStyle(
            "ReportHeading",
            parent=base["Heading2"],
            fontSize=13,
            leading=17,
            textColor=INK_PRIMARY,
            spaceBefore=16,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=base["BodyText"],
            fontSize=10.5,
            leading=15,
            textColor=INK_PRIMARY,
            spaceAfter=8,
        ),
        "muted": ParagraphStyle(
            "ReportMuted",
            parent=base["BodyText"],
            fontSize=9,
            leading=13,
            textColor=INK_SECONDARY,
        ),
        "headline": ParagraphStyle(
            "ReportHeadline",
            parent=base["BodyText"],
            fontSize=13,
            leading=18,
            textColor=INK_PRIMARY,
            spaceBefore=6,
            spaceAfter=12,
        ),
    }


def _figure_to_image(figure: Figure, width: float = 16 * cm) -> Image:
    """Embed a matplotlib figure at a width that fits the page margins."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    aspect = figure.get_figheight() / figure.get_figwidth()
    return Image(buffer, width=width, height=width * aspect)


def _verdict_banner(report: CausalReport, styles: dict) -> Table:
    """Build the banner stating whether the analysis can be trusted.

    This sits above the headline because a reader who sees an effect size first
    tends to carry it away regardless of any warning that follows.
    """
    if report.is_trustworthy:
        text = "This analysis rests on assumptions the data supports."
        accent = GOOD
    else:
        text = "This analysis does not support a causal conclusion. Read the assumptions before any number below."
        accent = CRITICAL

    table = Table([[Paragraph(text, styles["body"])]], colWidths=[16 * cm])
    table.setStyle(
        TableStyle(
            [
                ("LINEBEFORE", (0, 0), (0, -1), 3, accent),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


def _estimates_table(report: CausalReport, styles: dict) -> Table:
    """Build the table comparing every method's estimate."""
    rows = [["Method", "Applies to", "Estimate", "95% interval"]]
    for result in report.estimates:
        rows.append(
            [
                result.method,
                result.estimate_type.upper(),
                f"{result.point_estimate:,.3f}",
                f"{result.ci_low:,.3f} to {result.ci_high:,.3f}",
            ]
        )

    table = Table(rows, colWidths=[4 * cm, 3 * cm, 3.5 * cm, 5.5 * cm])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("TEXTCOLOR", (0, 0), (-1, -1), INK_PRIMARY),
                ("LINEBELOW", (0, 0), (-1, 0), 1, RULE),
                ("LINEBELOW", (0, 1), (-1, -2), 0.5, RULE),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
            ]
        )
    )
    return table


def _segments_table(report: CausalReport) -> Table:
    """Build the table of per-segment effects, marking any harmed group."""
    rows = [["Segment", "Units", "Effect", "95% interval"]]
    harmed_rows: list[int] = []
    for index, (_, row) in enumerate(report.segments.iterrows(), start=1):
        if row["harmed"]:
            harmed_rows.append(index)
        rows.append(
            [
                str(row["segment"]) + ("  (harmed)" if row["harmed"] else ""),
                str(int(row["n"])),
                f"{row['mean_effect']:+.3f}",
                f"{row['ci_low']:+.3f} to {row['ci_high']:+.3f}",
            ]
        )

    table = Table(rows, colWidths=[5 * cm, 2.5 * cm, 3 * cm, 5.5 * cm])
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK_PRIMARY),
        ("LINEBELOW", (0, 0), (-1, 0), 1, RULE),
        ("LINEBELOW", (0, 1), (-1, -2), 0.5, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ]
    # A harmed segment is marked in text as well as colour, so the warning
    # survives printing in greyscale.
    for row_index in harmed_rows:
        style.append(("TEXTCOLOR", (0, row_index), (-1, row_index), CRITICAL))
    table.setStyle(TableStyle(style))
    return table


def export_pdf(
    report: CausalReport,
    path: str,
    figures: dict[str, Figure] | None = None,
) -> str:
    """Write an assembled report to a PDF file.

    Args:
        report: The assembled analysis.
        path: Destination path for the PDF.
        figures: Optional charts to embed, keyed by ``forest``, ``balance``,
            ``overlap``, ``segments``, or ``sensitivity``. Missing entries are
            skipped, so a partial analysis still produces a document.

    Returns:
        The path the document was written to.
    """
    figures = figures or {}
    styles = _styles()
    document = SimpleDocTemplate(
        path,
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2.2 * cm,
        title=report.question,
    )

    story: list = [
        Paragraph(report.question, styles["title"]),
        Paragraph(
            f"Effect of {report.treatment} on {report.outcome}"
            + (f" | {report.dataset_summary.get('n_rows', 0):,} units" if report.dataset_summary else ""),
            styles["muted"],
        ),
        Spacer(1, 14),
        _verdict_banner(report, styles),
        Spacer(1, 14),
        Paragraph(report.headline or "No conclusion could be drawn.", styles["headline"]),
    ]

    story += [Paragraph("What the data can support", styles["heading"])]
    story += [Paragraph(report.narrative.get("assumptions", "Not assessed."), styles["body"])]
    if "adjustment" in report.narrative:
        story += [Paragraph(report.narrative["adjustment"], styles["body"])]
    if "balance" in figures:
        story += [Spacer(1, 6), _figure_to_image(figures["balance"])]
    if "overlap" in figures:
        story += [Spacer(1, 6), _figure_to_image(figures["overlap"])]

    story += [PageBreak(), Paragraph("Estimated effect", styles["heading"])]
    if report.estimates:
        story += [Paragraph(report.narrative.get("estimate", ""), styles["body"])]
        story += [Spacer(1, 6), _estimates_table(report, styles), Spacer(1, 10)]
        if "agreement" in report.narrative:
            story += [Paragraph(report.narrative["agreement"], styles["body"])]
        if "forest" in figures:
            story += [Spacer(1, 6), _figure_to_image(figures["forest"])]
    else:
        story += [Paragraph("No estimate could be produced from this data.", styles["body"])]

    if "robustness" in report.narrative:
        story += [Paragraph("Does the result survive scrutiny", styles["heading"])]
        story += [Paragraph(report.narrative["robustness"], styles["body"])]

    if "sensitivity" in report.narrative:
        story += [Paragraph("What could overturn it", styles["heading"])]
        story += [Paragraph(report.narrative["sensitivity"], styles["body"])]
        if "sensitivity" in figures:
            story += [Spacer(1, 6), _figure_to_image(figures["sensitivity"])]

    if "heterogeneity" in report.narrative:
        story += [PageBreak(), Paragraph("Who it affects", styles["heading"])]
        story += [Paragraph(report.narrative["heterogeneity"], styles["body"])]
        if report.segments is not None and not report.segments.empty:
            story += [Spacer(1, 6), _segments_table(report), Spacer(1, 10)]
        if "segments" in figures:
            story += [_figure_to_image(figures["segments"])]

    story += [
        Spacer(1, 20),
        Paragraph(
            f"Generated {report.generated_at:%Y-%m-%d %H:%M} UTC. Every estimate here is "
            "conditional on the assumptions stated above. No statistical method can "
            "rescue a comparison the data cannot support, and none can rule out a "
            "confounder that was never measured.",
            styles["muted"],
        ),
    ]

    document.build(story)
    return path
