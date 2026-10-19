from __future__ import annotations

import re
import shutil
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from videogen.cancel import Stopped, bind, clear_stop, request_stop
from videogen.config import ROOT, get_settings
from videogen.models import StageStatus
from videogen.pipeline import Pipeline
from videogen.providers.comfy import ComfyClient
from videogen.providers.tts import TTSProvider
from videogen.run_store import RunStore
from videogen.stages.revise import revise_shot

app = FastAPI(title="Local Video Studio")
settings = get_settings()
static_dir = ROOT / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")


class StartRequest(BaseModel):
    topic: str | None = None
    pick_topic: bool = False
    duration_s: int = Field(default=60, ge=20, le=360)
    style_preset: str = "sketch"
    skip_qa: bool = False
    await_approval: bool = True
    motion: bool = True
    still_density: str = "standard"


class OptionsRequest(BaseModel):
    motion: bool | None = None
    still_density: str | None = None
    replan: bool = False


class ReviseRequest(BaseModel):
    visual_prompt: str = ""
    narration_text: str = ""
    overlay_text: str = ""


_tts: TTSProvider | None = None


def _tts_provider() -> TTSProvider:
    global _tts
    if _tts is None:
        _tts = TTSProvider(settings)
    return _tts


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_path = static_dir / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Missing static/index.html</h1>", status_code=500)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/health")
def api_health() -> dict[str, bool]:
    return Pipeline(settings).health_check()


@app.post("/api/runs")
def start_run(req: StartRequest) -> dict[str, Any]:
    if not req.topic and not req.pick_topic:
        raise HTTPException(400, "Provide topic or pick_topic")
    store = RunStore(settings)
    run_id = store.run_id
    with _lock:
        _jobs[run_id] = {"state": "queued", "error": None}

    clear_stop(run_id)

    def worker() -> None:
        pipe = Pipeline(settings)
        bind(run_id)
        try:
            with _lock:
                _jobs[run_id]["state"] = "running"

            def on_progress(stage: str, state: str) -> None:
                with _lock:
                    _jobs[run_id]["stage"] = stage
                    _jobs[run_id]["stage_state"] = state

            pipe.run(
                topic=req.topic,
                duration_s=req.duration_s,
                style_preset=req.style_preset,
                pick_topic=req.pick_topic,
                run_id=run_id,
                skip_vision_qa=req.skip_qa,
                await_approval=req.await_approval,
                motion=req.motion,
                still_density=req.still_density,
                on_progress=on_progress,
            )
            waiting = RunStore(settings, run_id=run_id).status().meta.get("awaiting_approval")
            with _lock:
                _jobs[run_id]["state"] = "awaiting_approval" if waiting else "done"
        except Stopped:
            with _lock:
                _jobs[run_id]["state"] = "stopped"
                _jobs[run_id]["error"] = "Stopped"
        except Exception as e:  # noqa: BLE001
            with _lock:
                _jobs[run_id]["state"] = "failed"
                _jobs[run_id]["error"] = str(e)

    threading.Thread(target=worker, daemon=True).start()
    return {"run_id": run_id}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    root = settings.runs_path() / run_id
    if not root.exists():
        raise HTTPException(404, "unknown run")
    store = RunStore(settings, run_id=run_id)
    status = store.status()
    job = _jobs.get(run_id, {})
    return {
        "run_id": run_id,
        "status": status.model_dump(),
        "job": job,
        "final_ready": (root / "final.mp4").exists(),
    }


def _safe_asset(run_id: str, rel: str) -> Path:
    root = (settings.runs_path() / run_id).resolve()
    if not root.exists():
        raise HTTPException(404, "unknown run")
    path = (root / rel).resolve()
    try:
        path.relative_to(root)
    except ValueError as e:
        raise HTTPException(400, "bad path") from e
    if not path.is_file():
        raise HTTPException(404, "missing file")
    return path


def _busy(run_id: str) -> bool:
    with _lock:
        return _jobs.get(run_id, {}).get("state") == "running"


def _drop_clips(store: RunStore) -> None:
    clip_dir = store.root / "clips"
    if not clip_dir.exists():
        return
    for path in clip_dir.glob("*.mp4"):
        path.unlink()


def _drop_pictures(store: RunStore) -> None:
    frames = store.root / "frames"
    for path in frames.glob("shot_*.png"):
        path.unlink()
    for folder in ("drafts", "heroes"):
        directory = frames / folder
        if directory.exists():
            for path in directory.glob("*.png"):
                path.unlink()
    _drop_clips(store)


def _resume(run_id: str, *, await_approval: bool = False) -> None:
    clear_stop(run_id)
    with _lock:
        _jobs[run_id] = {"state": "running", "error": None}

    def worker() -> None:
        pipe = Pipeline(settings)
        store = RunStore(settings, run_id=run_id)
        bind(run_id)
        skip = bool(store.status().meta.get("skip_vision_qa"))

        def on_progress(stage: str, state: str) -> None:
            with _lock:
                _jobs[run_id]["stage"] = stage
                _jobs[run_id]["stage_state"] = state

        try:
            pipe.run(
                run_id=run_id,
                resume=True,
                skip_vision_qa=skip,
                await_approval=await_approval,
                on_progress=on_progress,
            )
            waiting = RunStore(settings, run_id=run_id).status().meta.get("awaiting_approval")
            with _lock:
                _jobs[run_id]["state"] = "awaiting_approval" if waiting else "done"
        except Stopped:
            with _lock:
                _jobs[run_id]["state"] = "stopped"
                _jobs[run_id]["error"] = "Stopped"
        except Exception as e:  # noqa: BLE001
            with _lock:
                _jobs[run_id]["state"] = "failed"
                _jobs[run_id]["error"] = str(e)

    threading.Thread(target=worker, daemon=True).start()


@app.get("/api/runs/{run_id}/storyboard")
def get_storyboard(run_id: str) -> dict[str, Any]:
    root = settings.runs_path() / run_id
    path = root / "storyboard.json"
    if not path.exists():
        raise HTTPException(404, "storyboard not ready")
    import json

    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/runs/{run_id}/asset")
def get_asset(run_id: str, rel: str) -> FileResponse:
    path = _safe_asset(run_id, rel)
    return FileResponse(path)


@app.post("/api/runs/{run_id}/shots/{shot_id}")
def revise_run_shot(run_id: str, shot_id: str, req: ReviseRequest) -> dict[str, bool]:
    root = settings.runs_path() / run_id
    if not root.exists():
        raise HTTPException(404, "unknown run")
    store = RunStore(settings, run_id=run_id)
    try:
        revise_shot(
            store,
            settings,
            ComfyClient(settings),
            _tts_provider(),
            shot_id,
            visual_prompt=req.visual_prompt,
            narration_text=req.narration_text,
            overlay_text=req.overlay_text,
        )
    except KeyError as e:
        raise HTTPException(404, "unknown shot") from e
    return {"ok": True}


@app.post("/api/runs/{run_id}/options")
def set_run_options(run_id: str, req: OptionsRequest) -> dict[str, Any]:
    root = settings.runs_path() / run_id
    if not root.exists():
        raise HTTPException(404, "unknown run")
    if _busy(run_id):
        raise HTTPException(409, "This run is still working")
    if req.still_density not in (None, "fewer", "standard", "more"):
        raise HTTPException(400, "still_density must be fewer, standard, or more")
    store = RunStore(settings, run_id=run_id)
    st = store.status()
    if req.motion is not None and st.meta.get("motion", True) != req.motion:
        st.meta["motion"] = req.motion
        _drop_clips(store)
        st.stages["motion"] = "pending"
        st.stages["assemble"] = "pending"
        st.meta["needs_rebuild"] = True
        st.current_stage = "motion"
    elif req.motion is not None:
        st.meta["motion"] = req.motion
    if req.still_density is not None:
        st.meta["still_density"] = req.still_density
    store.save_status(st)
    if req.replan:
        if req.still_density is None:
            raise HTTPException(400, "Choose fewer, standard, or more stills")
        _drop_pictures(store)
        for stage in ("storyboard", "bible", "stills", "qa", "motion", "assemble"):
            st.stages[stage] = "pending"
        st.meta["needs_rebuild"] = False
        st.meta["storyboard_approved"] = False
        st.meta["awaiting_approval"] = False
        st.current_stage = "storyboard"
        st.error = None
        store.save_status(st)
        _resume(run_id, await_approval=True)
    return {"ok": True, "meta": st.meta}


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str) -> dict[str, bool]:
    root = settings.runs_path() / run_id
    if not root.exists():
        raise HTTPException(404, "unknown run")
    request_stop(run_id)
    try:
        ComfyClient(settings).interrupt()
    except Exception:  # noqa: BLE001
        pass
    with _lock:
        live = _jobs.get(run_id, {}).get("state") in {"queued", "running", "stopping"}
        if live:
            _jobs[run_id]["state"] = "stopping"
    if not live:
        store = RunStore(settings, run_id=run_id)
        st = store.status()
        stage = st.current_stage
        if stage and stage not in {"done", "stopped", "awaiting_approval"}:
            st.stages[stage] = StageStatus.STOPPED
            st.current_stage = "stopped"
            st.error = "Stopped"
            store.save_status(st)
    return {"ok": True}


@app.post("/api/runs/{run_id}/resume")
def resume_run(run_id: str) -> dict[str, str]:
    root = settings.runs_path() / run_id
    if not root.exists():
        raise HTTPException(404, "unknown run")
    st = RunStore(settings, run_id=run_id).status()
    st.meta["needs_rebuild"] = False
    st.meta["storyboard_approved"] = True
    st.meta["awaiting_approval"] = False
    st.error = None
    RunStore(settings, run_id=run_id).save_status(st)
    _resume(run_id)
    return {"run_id": run_id, "state": "running"}


@app.post("/api/runs/{run_id}/approve")
def approve_run(run_id: str) -> dict[str, str]:
    root = settings.runs_path() / run_id
    if not root.exists():
        raise HTTPException(404, "unknown run")
    store = RunStore(settings, run_id=run_id)
    st = store.status()
    st.meta["storyboard_approved"] = True
    st.meta["awaiting_approval"] = False
    st.error = None
    store.save_status(st)
    _resume(run_id)
    return {"run_id": run_id, "state": "running"}


@app.get("/api/runs/{run_id}/video")
def get_video(run_id: str) -> FileResponse:
    path = settings.runs_path() / run_id / "final.mp4"
    if not path.exists():
        raise HTTPException(404, "video not ready")
    return FileResponse(path, media_type="video/mp4", filename=f"{run_id}.mp4")


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str) -> dict[str, bool]:
    if not _RUN_ID.fullmatch(run_id):
        raise HTTPException(400, "bad run id")
    if _busy(run_id):
        raise HTTPException(409, "This run is still working")
    runs = settings.runs_path().resolve()
    root = (runs / run_id).resolve()
    if root.parent != runs or not root.is_dir():
        raise HTTPException(404, "unknown run")
    shutil.rmtree(root)
    with _lock:
        _jobs.pop(run_id, None)
    return {"ok": True}


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    runs_dir = settings.runs_path()
    out = []
    for p in sorted(runs_dir.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        final = p / "final.mp4"
        status_path = p / "status.json"
        topic = ""
        if status_path.exists():
            try:
                topic = RunStore(settings, run_id=p.name).status().topic
            except Exception:  # noqa: BLE001
                topic = ""
        out.append(
            {
                "run_id": p.name,
                "topic": topic,
                "final_ready": final.exists(),
            }
        )
    return out[:50]
