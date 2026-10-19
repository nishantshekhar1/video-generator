from __future__ import annotations

import logging
from pathlib import Path

from videogen.cancel import raise_if_stopped
from videogen.config import Settings
from videogen.models import Storyboard
from videogen.overlay import apply_overlay, image_prompt
from videogen.providers.comfy import ComfyClient
from videogen.run_store import RunStore

log = logging.getLogger(__name__)


def _sheet(store: RunStore, board: Storyboard) -> Path | None:
    if board.bible and board.bible.sheet_path:
        path = store.path(board.bible.sheet_path)
        if path.exists():
            return path
    return None


def run_stills(
    store: RunStore,
    settings: Settings,
    comfy: ComfyClient,
    *,
    finals: bool,
) -> Storyboard:
    board = store.read_json("storyboard.json", Storyboard)
    reference = _sheet(store, board) if finals else None
    quality = "final" if finals else "draft"
    for shot in board.shots:
        raise_if_stopped()
        if finals:
            dest = store.path("frames", f"{shot.id}.png")
            rel_attr = "frame_path"
        else:
            dest = store.path("frames", "drafts", f"{shot.id}.png")
            rel_attr = "draft_path"
        if dest.exists() and dest.stat().st_size > 1000:
            setattr(shot, rel_attr, str(dest.relative_to(store.root)))
            continue
        log.info("Generating %s still %s", quality, shot.id)
        comfy.generate_still(
            image_prompt(shot),
            dest,
            quality=quality,
            reference=reference,
            filename_prefix=f"videogen_{store.run_id}_{shot.id}_{quality}",
        )
        apply_overlay(dest, shot.overlay_text)
        setattr(shot, rel_attr, str(dest.relative_to(store.root)))
        store.write_json("storyboard.json", board)
    store.write_json("storyboard.json", board)
    return board
