from __future__ import annotations

import logging
import re

from videogen.config import Settings
from videogen.models import Storyboard
from videogen.overlay import apply_overlay, image_prompt
from videogen.providers.comfy import ComfyClient
from videogen.providers.vision import OllamaVision
from videogen.run_store import RunStore

log = logging.getLogger(__name__)


def _expected_subject(visual_prompt: str, narration: str) -> str:
    m = re.search(r"Scene:\s*(.+?)(?:\.\s*This still must|\Z)", visual_prompt, re.DOTALL)
    scene = (m.group(1).strip() if m else visual_prompt)[:280]
    return f"Narration: {narration} | Scene: {scene}"


def _sheet(store: RunStore, board: Storyboard):
    if board.bible and board.bible.sheet_path:
        path = store.path(board.bible.sheet_path)
        if path.exists():
            return path
    return None


def run_qa(
    store: RunStore,
    settings: Settings,
    vision: OllamaVision,
    comfy: ComfyClient,
) -> Storyboard:
    board = store.read_json("storyboard.json", Storyboard)
    sheet = _sheet(store, board)
    for shot in board.shots:
        if not shot.frame_path:
            continue
        frame = store.path(shot.frame_path)
        expected = _expected_subject(shot.visual_prompt, shot.narration_text)
        try:
            result = vision.review_frame(
                frame,
                expected_subject=expected,
                style_notes=board.style_prefix,
                reference_path=sheet,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("Vision QA failed for %s: %s", shot.id, e)
            shot.qa_ok = True
            shot.qa_notes = f"vision_error:{e}"
            continue

        shot.qa_ok = bool(result.get("ok"))
        shot.qa_notes = str(result.get("reasons") or "")
        if (
            not shot.qa_ok
            and settings.pipeline.vision_retry_once
            and result.get("retry_prompt")
        ):
            retry_prompt = str(result["retry_prompt"])
            full = f"{board.style_prefix}. {retry_prompt}"
            shot.visual_prompt = full
            log.info("Retrying still %s after vision reject", shot.id)
            comfy.generate_still(
                image_prompt(shot),
                frame,
                quality="final",
                reference=sheet,
                filename_prefix=f"videogen_{store.run_id}_{shot.id}_retry",
            )
            apply_overlay(frame, shot.overlay_text)
            try:
                result2 = vision.review_frame(
                    frame,
                    expected_subject=retry_prompt[:300],
                    style_notes=board.style_prefix,
                    reference_path=sheet,
                )
                shot.qa_ok = bool(result2.get("ok"))
                shot.qa_notes = (
                    shot.qa_notes + " | retry: " + str(result2.get("reasons") or "")
                )
            except Exception as e:  # noqa: BLE001
                shot.qa_notes += f" | retry_vision_error:{e}"
                shot.qa_ok = True
    store.write_json("storyboard.json", board)
    return board
