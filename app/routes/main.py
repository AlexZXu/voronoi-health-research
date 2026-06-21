from flask import Blueprint, current_app, redirect, render_template, url_for

from app.services.data_import import load_atlanta_sample
from app.services.scenario_service import create_scenario_payload
from app.services.store import JsonStore


main_bp = Blueprint("main", __name__)


@main_bp.get("/")
def index():
    return render_template("index.html")


@main_bp.post("/sample")
def sample():
    population, facilities, bounds = load_atlanta_sample(current_app.config["DATA_DIR"])
    scenario = create_scenario_payload(
        population,
        facilities,
        bounds,
        name="Atlanta public health sample",
        parameters={"acceptable_radius": 0.05, "fuzzy_radius": 0.05, "omega": 0.2},
    )
    stored = JsonStore(current_app.config["STORE_PATH"]).create_scenario(scenario)
    return redirect(url_for("main.workspace", scenario_id=stored["id"]))


@main_bp.get("/workspace/<scenario_id>")
def workspace(scenario_id):
    return render_template("workspace.html", scenario_id=scenario_id)
