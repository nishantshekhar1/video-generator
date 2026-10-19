from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class StageName(str, Enum):
    TOPIC = "topic"
    RESEARCH = "research"
    SCRIPT = "script"
    NARRATION = "narration"
    STORYBOARD = "storyboard"
    BIBLE = "bible"
    STILLS = "stills"
    QA = "qa"
    MOTION = "motion"
    ASSEMBLE = "assemble"


STAGE_ORDER: list[StageName] = [
    StageName.TOPIC,
    StageName.RESEARCH,
    StageName.SCRIPT,
    StageName.NARRATION,
    StageName.STORYBOARD,
    StageName.BIBLE,
    StageName.STILLS,
    StageName.QA,
    StageName.MOTION,
    StageName.ASSEMBLE,
]


class ApprovalNeeded(Exception):
    """Draft frames are ready. Final stills wait for an approval."""


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    STOPPED = "stopped"


class Source(BaseModel):
    title: str = ""
    url: str
    snippet: str = ""


class Brief(BaseModel):
    topic: str
    angle: str = ""
    teaching_point: str = ""
    duration_s: int = 60
    style_preset: str = "sketch"
    sources: list[Source] = Field(default_factory=list)
    notes: str = ""


class ScriptBeat(BaseModel):
    role: Literal["hook", "beat", "close"] = "beat"
    text: str


class ScriptSentence(BaseModel):
    id: str
    beat_index: int
    role: str
    text: str
    visual_hint: str = ""


class Script(BaseModel):
    title: str
    teaching_point: str
    beats: list[ScriptBeat]
    sentences: list[ScriptSentence]
    word_count: int = 0
    target_duration_s: int = 60


class NarrationItem(BaseModel):
    sentence_id: str
    audio_path: str
    duration_s: float
    text: str


class NarrationManifest(BaseModel):
    items: list[NarrationItem]
    total_duration_s: float = 0.0


CameraMove = Literal["push", "pan_left", "pan_right", "hold", "pull"]


class Shot(BaseModel):
    id: str
    sentence_id: str
    sentence_ids: list[str] = Field(default_factory=list)
    narration_text: str
    visual_prompt: str
    overlay_text: str = ""
    camera: CameraMove = "hold"
    duration_s: float
    role: str = "beat"
    coverage: str = "action"
    hero: bool = False
    frame_path: str | None = None
    draft_path: str | None = None
    clip_path: str | None = None
    audio_path: str | None = None
    qa_notes: str = ""
    qa_ok: bool | None = None


class CharacterBible(BaseModel):
    anchors: list[str] = Field(default_factory=list)
    body: str = ""
    sheet_prompt: str = ""
    style_prompt: str = ""
    sheet_path: str = ""
    style_path: str = ""


class Storyboard(BaseModel):
    shots: list[Shot]
    style_prefix: str = ""
    bible: CharacterBible | None = None


class RunStatus(BaseModel):
    run_id: str
    topic: str = ""
    stages: dict[str, StageStatus] = Field(default_factory=dict)
    current_stage: str | None = None
    error: str | None = None
    final_video: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def fresh(cls, run_id: str, topic: str = "") -> RunStatus:
        return cls(
            run_id=run_id,
            topic=topic,
            stages={s.value: StageStatus.PENDING for s in STAGE_ORDER},
        )
