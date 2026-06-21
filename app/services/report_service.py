from __future__ import annotations

import csv
import io
from typing import Any

from app.algorithms.assignment import assign_population
from app.algorithms.geometry import build_voronoi_geojson
from app.algorithms.metrics import compute_metrics
from app.algorithms.scoring import compute_scores
from app.algorithms.types import ScenarioInput
from app.services.serialization import facilities_from_dict, scenario_to_input


METRIC_ROWS = [
    ("load_variance", "Load variance", 0),
    ("max_load", "Max load", 0),
    ("load_gini", "Load Gini", 3),
    ("outside_radius_percent", "Outside-radius percentage", 3),
    ("mean_weighted_distance", "Mean distance", 5),
]


def build_report_context(scenario: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    scenario_input = scenario_to_input(scenario)
    scenario_input.parameters = run.get("parameters", scenario_input.parameters)
    latest_facilities = facilities_from_dict(run["latest"]["facilities"])
    optimized_input = ScenarioInput(
        population=scenario_input.population,
        facilities=latest_facilities,
        bounds=scenario_input.bounds,
        parameters=run.get("parameters", scenario_input.parameters),
    )
    baseline = _evaluate(scenario_input)
    optimized = _evaluate(optimized_input)
    variance_reduction = percent_reduction(
        baseline["metrics"]["load_variance"],
        optimized["metrics"]["load_variance"],
    )
    outside_reduction = percent_reduction(
        baseline["metrics"]["outside_radius_population"],
        optimized["metrics"]["outside_radius_population"],
    )

    return {
        "scenario": scenario,
        "run": run,
        "parameters": run.get("parameters", {}),
        "baseline": baseline,
        "optimized": optimized,
        "variance_reduction": variance_reduction,
        "outside_reduction": outside_reduction,
        "interpretation": (
            f"Optimization reduced load variance by {variance_reduction:.1f}% "
            f"and outside-radius population by {outside_reduction:.1f}%."
        ),
    }


def report_csv_bytes(context: dict[str, Any]) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Section", "Field", "Baseline", "Optimized"])
    writer.writerow(["Scenario", "Name", context["scenario"].get("name", "Untitled scenario"), ""])
    writer.writerow(["Method", "Algorithm", context["run"].get("algorithm", "weighted_lloyd"), ""])
    writer.writerow(["Method", "Iteration", "", context["run"]["latest"].get("iteration", 0)])
    writer.writerow([])

    writer.writerow(["Parameters", "Name", "Value", ""])
    for key, value in sorted(context["parameters"].items()):
        writer.writerow(["Parameters", key, value, ""])
    writer.writerow([])

    writer.writerow(["Results", "Metric", "Baseline", "Optimized"])
    for key, label, _digits in METRIC_ROWS:
        writer.writerow(["Results", label, context["baseline"]["metrics"][key], context["optimized"]["metrics"][key]])
    writer.writerow(["Results", "Variance reduction", "", f"{context['variance_reduction']:.1f}%"])
    writer.writerow(["Results", "Outside-radius population reduction", "", f"{context['outside_reduction']:.1f}%"])
    writer.writerow([])

    writer.writerow(["Facility Scores", "Facility", "Regular population", "Outside radius population", "External fuzzy population", "Risk factor", "Score"])
    for row in context["optimized"]["facility_scores"]:
        writer.writerow(
            [
                "Facility Scores",
                row["name"],
                row["regular_population"],
                row["outside_radius_population"],
                row["external_fuzzy_population"],
                row["risk_factor"],
                row["score"],
            ]
        )
    writer.writerow([])
    writer.writerow(["Interpretation", context["interpretation"], "", ""])
    return output.getvalue().encode("utf-8")


def report_pdf_bytes(context: dict[str, Any]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output,
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="Voronoi Facility Optimization Report",
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Voronoi Facility Optimization Report", styles["Title"]))
    story.append(Spacer(1, 0.15 * inch))

    _section(story, styles, "Scenario")
    story.append(Paragraph(context["scenario"].get("name", "Untitled scenario"), styles["BodyText"]))
    story.append(Spacer(1, 0.12 * inch))

    _section(story, styles, "Method")
    method_text = (
        "Weighted Lloyd/K-means optimization assigns population points to the nearest facility, "
        "computes weighted centroids using population demand and score penalties, then moves "
        "eligible facilities toward those centroids over iterative steps."
    )
    story.append(Paragraph(method_text, styles["BodyText"]))
    story.append(Spacer(1, 0.12 * inch))

    _section(story, styles, "Parameters")
    story.append(_parameters_table(context["parameters"]))
    story.append(Spacer(1, 0.16 * inch))

    _section(story, styles, "Results")
    story.append(_results_table(context["baseline"]["metrics"], context["optimized"]["metrics"]))
    story.append(Spacer(1, 0.16 * inch))

    map_bytes = static_map_png_bytes(context)
    story.append(Image(map_bytes, width=7.1 * inch, height=4.2 * inch))
    story.append(PageBreak())

    _section(story, styles, "Facility Scores")
    story.append(_facility_score_table(context["optimized"]["facility_scores"]))
    story.append(Spacer(1, 0.16 * inch))

    _section(story, styles, "Interpretation")
    story.append(Paragraph(context["interpretation"], styles["BodyText"]))

    doc.build(story)
    return output.getvalue()


def static_map_png_bytes(context: dict[str, Any]) -> io.BytesIO:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as PlotPolygon

    scenario = context["scenario"]
    population = scenario_to_input(scenario).population
    facilities = facilities_from_dict(context["run"]["latest"]["facilities"])
    values = [row["score"] for row in context["optimized"]["facility_scores"]]
    voronoi = build_voronoi_geojson(facilities, scenario["bounds"], values, "score")
    output = io.BytesIO()

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    ax.set_facecolor("#f7f9fb")

    for feature in voronoi["features"]:
        color = feature["properties"].get("color", "#dfe7ed")
        for polygon in _polygon_rings(feature["geometry"]):
            ax.add_patch(PlotPolygon(polygon, closed=True, facecolor=color, edgecolor="#33434f", alpha=0.32, linewidth=0.8))

    point_sizes = 12 + 80 * (population.weight / max(float(population.weight.max()), 1.0)) ** 0.5
    ax.scatter(population.lon, population.lat, s=point_sizes, c="#3f8ecb", alpha=0.45, edgecolors="#275d8c", linewidths=0.4, label="Population")
    ax.scatter(facilities.lon, facilities.lat, s=70, c="#d64b3a", marker="^", edgecolors="#82271d", linewidths=0.7, label="Facilities")

    for idx, name in enumerate(facilities.names):
        ax.annotate(str(name)[:18], (facilities.lon[idx], facilities.lat[idx]), fontsize=6, xytext=(3, 3), textcoords="offset points")

    bounds = scenario["bounds"]
    ax.set_xlim(bounds["min_lon"], bounds["max_lon"])
    ax.set_ylim(bounds["min_lat"], bounds["max_lat"])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Optimized Facility Placement and Voronoi Regions")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, format="png")
    plt.close(fig)
    output.seek(0)
    return output


def percent_reduction(before: float, after: float) -> float:
    if before == 0:
        return 0.0 if after == 0 else -100.0
    return ((before - after) / before) * 100


def _evaluate(scenario_input: ScenarioInput) -> dict[str, Any]:
    params = scenario_input.parameters
    acceptable_radius = float(params.get("acceptable_radius", 0.05))
    assignments = assign_population(scenario_input.population, scenario_input.facilities)
    return {
        "metrics": compute_metrics(scenario_input.population, assignments, acceptable_radius),
        "facility_scores": compute_scores(
            scenario_input.population,
            scenario_input.facilities,
            assignments,
            acceptable_radius=acceptable_radius,
            fuzzy_radius=float(params.get("fuzzy_radius", 0.0)),
            omega=float(params.get("omega", 0.2)),
            outside_radius_weight=float(params.get("outside_radius_weight", 1.0)),
        ),
    }


def _section(story, styles, title: str) -> None:
    from reportlab.platypus import Paragraph

    story.append(Paragraph(title, styles["Heading2"]))


def _parameters_table(parameters: dict[str, Any]):
    rows = [["Parameter", "Value"]] + [[key, str(value)] for key, value in sorted(parameters.items())]
    return _styled_table(rows, widths=[220, 260])


def _results_table(baseline: dict[str, Any], optimized: dict[str, Any]):
    rows = [["Metric", "Baseline", "Optimized"]]
    for key, label, digits in METRIC_ROWS:
        rows.append([label, _format_metric(baseline[key], digits), _format_metric(optimized[key], digits)])
    return _styled_table(rows, widths=[220, 130, 130])


def _facility_score_table(scores: list[dict[str, Any]]):
    rows = [["Facility", "Regular", "Outside", "External", "Risk", "Score"]]
    for row in scores:
        rows.append(
            [
                row["name"][:28],
                f"{row['regular_population']:.0f}",
                f"{row['outside_radius_population']:.0f}",
                f"{row['external_fuzzy_population']:.0f}",
                f"{row['risk_factor']:.2f}",
                f"{row['score']:.0f}",
            ]
        )
    return _styled_table(rows, widths=[190, 70, 70, 70, 55, 70], font_size=7)


def _styled_table(rows: list[list[Any]], widths: list[int], font_size: int = 8):
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b6a5c")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8d2db")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fb")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _format_metric(value: float, digits: int) -> str:
    if digits == 0:
        return f"{value:,.0f}"
    return f"{value:,.{digits}f}"


def _polygon_rings(geometry: dict[str, Any]) -> list[list[tuple[float, float]]]:
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"][0]]
    if geometry["type"] == "MultiPolygon":
        return [polygon[0] for polygon in geometry["coordinates"]]
    return []
