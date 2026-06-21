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
