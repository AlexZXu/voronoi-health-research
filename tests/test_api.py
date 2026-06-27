import io


def test_sample_scenario_api_flow(client):
    response = client.post("/sample", follow_redirects=False)
    assert response.status_code == 302
    scenario_id = response.headers["Location"].rstrip("/").split("/")[-1]

    scenario_response = client.get(f"/api/scenarios/{scenario_id}")
    assert scenario_response.status_code == 200
    scenario = scenario_response.get_json()
    assert scenario["baseline"]["metrics"]["total_population"] > 0
    assert len(scenario["facilities"]) == 10

    run_response = client.post(
        f"/api/scenarios/{scenario_id}/runs",
        json={"algorithm": "weighted_lloyd", "parameters": {"acceptable_radius": 0.05, "fuzzy_radius": 0.05}},
    )
    assert run_response.status_code == 200
    run = run_response.get_json()

    step_response = client.post(f"/api/runs/{run['id']}/step", json={})
    assert step_response.status_code == 200
    stepped = step_response.get_json()
    assert stepped["latest"]["iteration"] == 1
    assert stepped["latest"]["metrics"]["total_population"] == scenario["baseline"]["metrics"]["total_population"]


def test_workspace_contains_before_after_comparison(client):
    response = client.post("/sample", follow_redirects=False)
    scenario_id = response.headers["Location"].rstrip("/").split("/")[-1]

    workspace = client.get(f"/workspace/{scenario_id}")

    assert workspace.status_code == 200
    assert b"Before vs. After Optimization" in workspace.data
    assert b"Baseline Placement" in workspace.data
    assert b"Optimized Placement" in workspace.data
    assert b"Explain Results" in workspace.data
    assert b"Metric Guide" in workspace.data
    assert b"Current Interpretation" in workspace.data


def test_workspace_map_interpretability_assets(client):
    response = client.get("/static/js/workspace.js")

    assert response.status_code == 200
    assert b"Map Legend" in response.data
    assert b"Overloaded facility" in response.data
    assert b"Outside-radius demand" in response.data
    assert b"Assigned population" in response.data


def test_report_exports_after_optimization(client):
    response = client.post("/sample", follow_redirects=False)
    scenario_id = response.headers["Location"].rstrip("/").split("/")[-1]
    run_response = client.post(
        f"/api/scenarios/{scenario_id}/runs",
        json={"algorithm": "weighted_lloyd", "parameters": {"acceptable_radius": 0.05, "fuzzy_radius": 0.05}},
    )
    run_id = run_response.get_json()["id"]
    client.post(f"/api/runs/{run_id}/step", json={})

    csv_response = client.get(f"/api/runs/{run_id}/export/report.csv")
    pdf_response = client.get(f"/api/runs/{run_id}/export/report.pdf")

    assert csv_response.status_code == 200
    assert csv_response.mimetype == "text/csv"
    assert b"Scenario" in csv_response.data
    assert b"Facility Scores" in csv_response.data
    assert pdf_response.status_code == 200
    assert pdf_response.mimetype == "application/pdf"
    assert pdf_response.data.startswith(b"%PDF")


def test_dataset_preview_detects_columns(client):
    response = client.post(
        "/api/datasets/preview",
        data={
            "population": (io.BytesIO(b"Latitude,Longitude,Demand\n33.1,-84.1,100\n33.2,-84.2,200\n"), "population.csv"),
            "facilities": (io.BytesIO(b"Name,Latitude,Longitude,Risk\nClinic A,33.1,-84.1,1.2\n"), "facilities.csv"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["valid"] is True
    assert payload["population"]["detected_mapping"]["weight"] == "Demand"
    assert payload["facilities"]["detected_mapping"]["name"] == "Name"
    assert len(payload["population"]["preview"]) == 2


def test_dataset_preview_lists_exact_missing_columns(client):
    response = client.post(
        "/api/datasets/preview",
        data={
            "population": (io.BytesIO(b"Latitude,Demand\n33.1,100\n"), "population.csv"),
            "facilities": (io.BytesIO(b"Name,Latitude\nClinic A,33.1\n"), "facilities.csv"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["valid"] is False
    assert "Population CSV is missing required column(s): longitude." in payload["errors"]
    assert "Facility CSV is missing required column(s): longitude." in payload["errors"]


def test_upload_requires_column_confirmation(client):
    response = client.post(
        "/api/datasets/upload",
        data={
            "population": (io.BytesIO(b"Latitude,Longitude,Population\n33.1,-84.1,100\n"), "population.csv"),
            "facilities": (io.BytesIO(b"Name,Latitude,Longitude\nClinic A,33.1,-84.1\n"), "facilities.csv"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "confirm the detected columns" in response.get_json()["error"]


def test_upload_reports_missing_confirmed_mapping_column(client):
    response = client.post(
        "/api/datasets/upload",
        data={
            "columns_confirmed": "true",
            "population_mapping": '{"lat":"Latitude","lon":"Missing longitude","weight":"Population"}',
            "facilities_mapping": '{"lat":"Latitude","lon":"Longitude"}',
            "population": (io.BytesIO(b"Latitude,Longitude,Population\n33.1,-84.1,100\n"), "population.csv"),
            "facilities": (io.BytesIO(b"Name,Latitude,Longitude\nClinic A,33.1,-84.1\n"), "facilities.csv"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "Missing column for longitude" in response.get_json()["error"]


def test_upload_allows_blank_optional_numeric_columns(client):
    response = client.post(
        "/api/datasets/upload",
        data={
            "columns_confirmed": "true",
            "population_mapping": '{"lat":"Latitude","lon":"Longitude","weight":"Population"}',
            "facilities_mapping": '{"lat":"Latitude","lon":"Longitude","name":"Name","risk_factor":"Risk","capacity":"Capacity"}',
            "population": (io.BytesIO(b"Latitude,Longitude,Population\n33.1,-84.1,100\n"), "population.csv"),
            "facilities": (io.BytesIO(b"Name,Latitude,Longitude,Risk,Capacity\nClinic A,33.1,-84.1,,\n"), "facilities.csv"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.get_json()["workspace_url"].startswith("/workspace/")
