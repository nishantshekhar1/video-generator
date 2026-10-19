from __future__ import annotations

import logging
from typing import Callable

from videogen.cancel import Stopped, raise_if_stopped
from videogen.config import Settings, get_settings
from videogen.models import STAGE_ORDER, ApprovalNeeded, StageName, StageStatus
from videogen.providers.comfy import ComfyClient
from videogen.providers.llm import OllamaLLM
from videogen.providers.research import WebResearcher
from videogen.providers.tts import TTSProvider
from videogen.providers.vision import OllamaVision
from videogen.run_store import RunStore
from videogen.stages import (
    assemble as assemble_stage,
    bible as bible_stage,
    motion as motion_stage,
    narration as narration_stage,
    qa as qa_stage,
    research as research_stage,
    script as script_stage,
    stills as stills_stage,
    storyboard as storyboard_stage,
    topic as topic_stage,
)

log = logging.getLogger(__name__)

ProgressCb = Callable[[str, str], None]


class Pipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.llm = OllamaLLM(self.settings)
        self.vision = OllamaVision(self.settings)
        self.comfy = ComfyClient(self.settings)
        self.tts = TTSProvider(self.settings)
        self.researcher = WebResearcher(self.settings)

    def health_check(self) -> dict[str, bool]:
        return {
            "ollama": self.llm.health(),
            "comfyui": self.comfy.health(),
        }

    def require_healthy(self, *, need_comfy: bool = True) -> None:
        h = self.health_check()
        missing = []
        if not h["ollama"]:
            missing.append("Ollama (http://127.0.0.1:11434)")
        if need_comfy and not h["comfyui"]:
            missing.append("ComfyUI (http://127.0.0.1:8188)")
        if missing:
            raise RuntimeError(
                "Health check failed; start local services: " + ", ".join(missing)
            )

    def run(
        self,
        *,
        topic: str | None = None,
        duration_s: int | None = None,
        style_preset: str = "sketch",
        pick_topic: bool = False,
        run_id: str | None = None,
        resume: bool = True,
        skip_vision_qa: bool = False,
        await_approval: bool = False,
        motion: bool | None = None,
        still_density: str | None = None,
        on_progress: ProgressCb | None = None,
    ) -> RunStore:
        duration = duration_s or self.settings.pipeline.default_duration_s
        store = RunStore(self.settings, run_id=run_id)
        status = store.status()
        if not status.stages:
            status = type(status).fresh(store.run_id, topic or "")
            store.save_status(status)
        st = store.status()
        if "skip_vision_qa" not in st.meta:
            st.meta["skip_vision_qa"] = skip_vision_qa
        if motion is not None:
            st.meta["motion"] = motion
        if still_density in ("fewer", "standard", "more"):
            st.meta["still_density"] = still_density
        store.save_status(st)

        def emit(stage: str, state: str) -> None:
            log.info("[%s] %s", stage, state)
            if on_progress:
                on_progress(stage, state)

        # Health: need comfy from stills onward
        self.require_healthy(need_comfy=False)

        topic_text = topic

        stages_to_run: list[tuple[StageName, Callable[[], None]]] = [
            (
                StageName.TOPIC,
                lambda: topic_stage.run_topic(
                    store,
                    self.settings,
                    self.llm,
                    self.researcher,
                    topic=topic_text,
                    duration_s=duration,
                    style_preset=style_preset,
                    pick_topic=pick_topic,
                ),
            ),
            (
                StageName.RESEARCH,
                lambda: research_stage.run_research(
                    store, self.settings, self.llm, self.researcher
                ),
            ),
            (
                StageName.SCRIPT,
                lambda: script_stage.run_script(store, self.settings, self.llm),
            ),
            (
                StageName.NARRATION,
                lambda: narration_stage.run_narration(store, self.settings, self.tts),
            ),
            (
                StageName.STORYBOARD,
                lambda: storyboard_stage.run_storyboard(store, self.settings, self.llm),
            ),
            (
                StageName.BIBLE,
                lambda: bible_stage.run_bible(
                    store, self.settings, self.llm, self.comfy
                ),
            ),
            (
                StageName.STILLS,
                lambda: self._run_stills(store, await_approval=await_approval),
            ),
            (
                StageName.QA,
                lambda: qa_stage.run_qa(
                    store, self.settings, self.vision, self.comfy
                ),
            ),
            (
                StageName.MOTION,
                lambda: motion_stage.run_motion(store, self.settings, self.comfy),
            ),
            (
                StageName.ASSEMBLE,
                lambda: assemble_stage.run_assemble(store, self.settings),
            ),
        ]

        for stage_name, fn in stages_to_run:
            raise_if_stopped()
            if resume and store.stage_done(stage_name):
                emit(stage_name.value, "skipped")
                continue
            if stage_name in (StageName.BIBLE, StageName.STILLS, StageName.QA, StageName.MOTION):
                self.require_healthy(need_comfy=True)
            if stage_name == StageName.QA and skip_vision_qa:
                store.mark(stage_name, StageStatus.SKIPPED)
                emit(stage_name.value, "skipped")
                continue
            store.mark(stage_name, StageStatus.RUNNING)
            emit(stage_name.value, "running")
            try:
                fn()
                store.mark(stage_name, StageStatus.DONE)
                emit(stage_name.value, "done")
            except Stopped:
                store.mark(stage_name, StageStatus.STOPPED, error="Stopped")
                emit(stage_name.value, "stopped")
                raise
            except ApprovalNeeded:
                st = store.status()
                st.stages[stage_name.value] = StageStatus.PENDING
                st.meta["awaiting_approval"] = True
                st.current_stage = "awaiting_approval"
                st.error = None
                store.save_status(st)
                emit(stage_name.value, "awaiting_approval")
                return store
            except Exception as e:  # noqa: BLE001
                store.mark(stage_name, StageStatus.FAILED, error=str(e))
                emit(stage_name.value, "failed")
                raise

        return store

    def _run_stills(self, store: RunStore, *, await_approval: bool) -> None:
        approved = bool(store.status().meta.get("storyboard_approved"))
        if await_approval and not approved:
            stills_stage.run_stills(store, self.settings, self.comfy, finals=False)
            raise ApprovalNeeded()
        st = store.status()
        st.meta["awaiting_approval"] = False
        st.meta["storyboard_approved"] = True
        store.save_status(st)
        stills_stage.run_stills(store, self.settings, self.comfy, finals=True)
