from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from videogen.config import Settings
from videogen.models import RunStatus, StageName, StageStatus

T = TypeVar("T", bound=BaseModel)


class RunStore:
    def __init__(self, settings: Settings, run_id: str | None = None):
        self.settings = settings
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.root = settings.runs_path() / self.run_id
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in ("audio", "frames", "frames/drafts", "frames/heroes", "clips", "stems"):
            (self.root / sub).mkdir(exist_ok=True)

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def write_json(self, name: str, model: BaseModel | dict) -> Path:
        p = self.path(name)
        if isinstance(model, BaseModel):
            data = model.model_dump()
        else:
            data = model
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return p

    def read_json(self, name: str, model_type: type[T]) -> T:
        data = json.loads(self.path(name).read_text(encoding="utf-8"))
        return model_type.model_validate(data)

    def write_text(self, name: str, text: str) -> Path:
        p = self.path(name)
        p.write_text(text, encoding="utf-8")
        return p

    def read_text(self, name: str) -> str:
        return self.path(name).read_text(encoding="utf-8")

    def status(self) -> RunStatus:
        p = self.path("status.json")
        if not p.exists():
            return RunStatus.fresh(self.run_id)
        return RunStatus.model_validate_json(p.read_text(encoding="utf-8"))

    def save_status(self, status: RunStatus) -> None:
        self.write_json("status.json", status)

    def mark(self, stage: StageName, state: StageStatus, *, error: str | None = None) -> RunStatus:
        st = self.status()
        st.stages[stage.value] = state
        if state == StageStatus.RUNNING:
            st.current_stage = stage.value
        elif state == StageStatus.DONE and stage == StageName.ASSEMBLE:
            st.current_stage = "done"
        elif state == StageStatus.STOPPED:
            st.current_stage = "stopped"
        elif state == StageStatus.DONE and st.current_stage == stage.value:
            st.current_stage = None
        if error:
            st.error = error
        elif state == StageStatus.DONE:
            st.error = None
        if state == StageStatus.DONE and stage == StageName.ASSEMBLE:
            final = self.path("final.mp4")
            if final.exists():
                st.final_video = str(final)
        self.save_status(st)
        return st

    def stage_done(self, stage: StageName) -> bool:
        return self.status().stages.get(stage.value) == StageStatus.DONE
