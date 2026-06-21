from __future__ import annotations

import json
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any


class JsonStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"scenarios": {}, "runs": {}}
        with self.path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save(self, state: dict[str, Any]) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)

    def create_scenario(self, scenario: dict[str, Any]) -> dict[str, Any]:
        state = self.load()
        scenario = deepcopy(scenario)
        scenario["id"] = scenario.get("id") or uuid.uuid4().hex
        state["scenarios"][scenario["id"]] = scenario
        self.save(state)
        return scenario

    def get_scenario(self, scenario_id: str) -> dict[str, Any]:
        state = self.load()
        return state["scenarios"][scenario_id]

    def update_scenario(self, scenario_id: str, scenario: dict[str, Any]) -> dict[str, Any]:
        state = self.load()
        scenario = deepcopy(scenario)
        scenario["id"] = scenario_id
        state["scenarios"][scenario_id] = scenario
        self.save(state)
        return scenario

    def create_run(self, run: dict[str, Any]) -> dict[str, Any]:
        state = self.load()
        run = deepcopy(run)
        run["id"] = run.get("id") or uuid.uuid4().hex
        state["runs"][run["id"]] = run
        self.save(state)
        return run

    def get_run(self, run_id: str) -> dict[str, Any]:
        state = self.load()
        return state["runs"][run_id]

    def update_run(self, run_id: str, run: dict[str, Any]) -> dict[str, Any]:
        state = self.load()
        run = deepcopy(run)
        run["id"] = run_id
        state["runs"][run_id] = run
        self.save(state)
        return run

    def list_scenarios(self) -> list[dict[str, Any]]:
        state = self.load()
        return list(state["scenarios"].values())
