from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from flask import Blueprint, Response, current_app, jsonify, request
from werkzeug.utils import secure_filename

from app.services.data_import import compute_bounds, guess_mapping, read_facilities_csv, read_population_csv
from app.services.scenario_service import (
    create_scenario_payload,
    scenario_response,
    update_facilities_from_payload,
)
from app.services.serialization import facilities_from_dict, scenario_to_input
from app.services.store import JsonStore
from app.services.run_service import create_run_payload, run_response, run_to_convergence, step_run
from app.services.report_service import build_report_context, report_csv_bytes, report_pdf_bytes


api_bp = Blueprint("api", __name__)


def _store() -> JsonStore:
    return JsonStore(current_app.config["STORE_PATH"])


@api_bp.get("/datasets/guess")
def guess_dataset_mapping():
    columns = request.args.get("columns", "")
    dataset_type = request.args.get("dataset_type", "population")
    return jsonify({"mapping": guess_mapping([column.strip() for column in columns.split(",") if column.strip()], dataset_type)})


@api_bp.post("/datasets/upload")
def upload_dataset_pair():
    population_file = request.files.get("population")
    facilities_file = request.files.get("facilities")
    if not population_file or not facilities_file:
        return jsonify({"error": "Upload both population and facility CSV files."}), 400

    upload_dir = Path(current_app.config["UPLOAD_FOLDER"])
    population_path = upload_dir / secure_filename(population_file.filename or "population.csv")
    facilities_path = upload_dir / secure_filename(facilities_file.filename or "facilities.csv")
    population_file.save(population_path)
    facilities_file.save(facilities_path)

    try:
        population_mapping = json.loads(request.form.get("population_mapping") or "{}") or None
        facilities_mapping = json.loads(request.form.get("facilities_mapping") or "{}") or None
        population = read_population_csv(population_path, population_mapping)
        facilities = read_facilities_csv(facilities_path, facilities_mapping)
        bounds = compute_bounds(population, facilities)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    scenario = create_scenario_payload(
        population,
        facilities,
        bounds,
        name=request.form.get("name") or "Uploaded scenario",
        parameters={"acceptable_radius": float(request.form.get("acceptable_radius") or 0.05)},
    )
    stored = _store().create_scenario(scenario)
    return jsonify({"scenario_id": stored["id"], "workspace_url": f"/workspace/{stored['id']}"})


@api_bp.get("/scenarios")
def list_scenarios():
    rows = _store().list_scenarios()
    return jsonify([{"id": row["id"], "name": row.get("name", "Untitled scenario")} for row in rows])


@api_bp.get("/scenarios/<scenario_id>")
def get_scenario(scenario_id):
    try:
        scenario = _store().get_scenario(scenario_id)
    except KeyError:
        return jsonify({"error": "Scenario not found."}), 404
    scenario_input = scenario_to_input(scenario)
    return jsonify(scenario_response(scenario, scenario_input))


@api_bp.put("/scenarios/<scenario_id>/facilities")
def update_facilities(scenario_id):
    try:
        scenario = _store().get_scenario(scenario_id)
    except KeyError:
        return jsonify({"error": "Scenario not found."}), 404
    payload = request.get_json(silent=True) or {}
    scenario = update_facilities_from_payload(scenario, payload.get("facilities", []))
    _store().update_scenario(scenario_id, scenario)
    scenario_input = scenario_to_input(scenario)
    return jsonify(scenario_response(scenario, scenario_input))


@api_bp.post("/scenarios/<scenario_id>/runs")
def create_run(scenario_id):
    try:
        scenario = _store().get_scenario(scenario_id)
    except KeyError:
        return jsonify({"error": "Scenario not found."}), 404
    payload = request.get_json(silent=True) or {}
    algorithm = payload.get("algorithm", "weighted_lloyd")
    if algorithm != "weighted_lloyd":
        return jsonify({"error": "Only weighted_lloyd is implemented in the MVP."}), 400
    run = create_run_payload(scenario, algorithm, payload.get("parameters", {}))
    stored = _store().create_run(run)
    return jsonify(run_response(stored, scenario))


@api_bp.get("/runs/<run_id>")
def get_run(run_id):
    try:
        run = _store().get_run(run_id)
        scenario = _store().get_scenario(run["scenario_id"])
    except KeyError:
        return jsonify({"error": "Run not found."}), 404
    return jsonify(run_response(run, scenario))


@api_bp.post("/runs/<run_id>/step")
def step(run_id):
    try:
        run = _store().get_run(run_id)
        scenario = _store().get_scenario(run["scenario_id"])
    except KeyError:
        return jsonify({"error": "Run not found."}), 404
    if run.get("latest", {}).get("converged"):
        return jsonify(run_response(run, scenario))
    run = step_run(scenario, run)
    _store().update_run(run_id, run)
    return jsonify(run_response(run, scenario))


@api_bp.post("/runs/<run_id>/run-to-convergence")
def converge(run_id):
    try:
        run = _store().get_run(run_id)
        scenario = _store().get_scenario(run["scenario_id"])
    except KeyError:
        return jsonify({"error": "Run not found."}), 404
    payload = request.get_json(silent=True) or {}
    run = run_to_convergence(scenario, run, payload.get("max_steps"))
    _store().update_run(run_id, run)
    return jsonify(run_response(run, scenario))


@api_bp.get("/runs/<run_id>/iterations/<int:index>")
def iteration(run_id, index):
    try:
        run = _store().get_run(run_id)
    except KeyError:
        return jsonify({"error": "Run not found."}), 404
    iterations = run.get("iterations", [])
    if index < 0 or index >= len(iterations):
        return jsonify({"error": "Iteration not found."}), 404
    return jsonify(iterations[index])


@api_bp.get("/runs/<run_id>/export/facilities.csv")
def export_facilities(run_id):
    try:
        run = _store().get_run(run_id)
    except KeyError:
        return jsonify({"error": "Run not found."}), 404
    facilities = facilities_from_dict(run["latest"]["facilities"])
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "name", "latitude", "longitude", "risk_factor", "fixed"])
    for idx, facility_id in enumerate(facilities.ids):
        writer.writerow(
            [
                facility_id,
                facilities.names[idx],
                float(facilities.lat[idx]),
                float(facilities.lon[idx]),
                float(facilities.risk_factor[idx]),
                bool(facilities.fixed[idx]),
            ]
        )
    return Response(output.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=optimized_facilities.csv"})


@api_bp.get("/runs/<run_id>/export/metrics.csv")
def export_metrics(run_id):
    try:
        run = _store().get_run(run_id)
    except KeyError:
        return jsonify({"error": "Run not found."}), 404
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["facility_id", "name", "regular_population", "outside_radius_population", "external_fuzzy_population", "risk_factor", "score"])
    for row in run["latest"]["facility_scores"]:
        writer.writerow(
            [
                row["facility_id"],
                row["name"],
                row["regular_population"],
                row["outside_radius_population"],
                row["external_fuzzy_population"],
                row["risk_factor"],
                row["score"],
            ]
        )
    return Response(output.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=facility_metrics.csv"})


@api_bp.get("/runs/<run_id>/export/scenario.json")
def export_scenario(run_id):
    try:
        run = _store().get_run(run_id)
        scenario = _store().get_scenario(run["scenario_id"])
    except KeyError:
        return jsonify({"error": "Run not found."}), 404
    return jsonify({"scenario": scenario, "run": run})


@api_bp.get("/runs/<run_id>/export/report.csv")
def export_report_csv(run_id):
    try:
        run = _store().get_run(run_id)
        scenario = _store().get_scenario(run["scenario_id"])
    except KeyError:
        return jsonify({"error": "Run not found."}), 404
    report = report_csv_bytes(build_report_context(scenario, run))
    return Response(
        report,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=optimization_report.csv"},
    )


@api_bp.get("/runs/<run_id>/export/report.pdf")
def export_report_pdf(run_id):
    try:
        run = _store().get_run(run_id)
        scenario = _store().get_scenario(run["scenario_id"])
    except KeyError:
        return jsonify({"error": "Run not found."}), 404
    report = report_pdf_bytes(build_report_context(scenario, run))
    return Response(
        report,
        mimetype="application/pdf",
        headers={"Content-Disposition": "attachment; filename=optimization_report.pdf"},
    )
