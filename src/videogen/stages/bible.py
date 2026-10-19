"""Character sheet, style plate, and the anchors repeated on every shot."""

from __future__ import annotations

import logging
import re

from videogen.config import Settings
from videogen.models import CharacterBible, Storyboard
from videogen.providers.comfy import ComfyClient
from videogen.providers.llm import OllamaLLM
from videogen.run_store import RunStore

log = logging.getLogger(__name__)


_BODY_RULE = (
    "Same torso width, hip width, and limb thickness in every frame. "
    "Do not redraw the character thinner or heavier."
)
_PLACE = re.compile(
    r"\b(room|study|landscape|scenery|filled with|background|street|square)\b",
    re.IGNORECASE,
)


def _sheet_prompt(style_prefix: str, body: str, outfit: str) -> str:
    return (
        f"{style_prefix}. Character proportion sheet on a plain warm cream background. "
        "Two full-body standing views side by side, front and exact side, same height, "
        "feet visible, no scenery, no extra people. "
        f"Body, identical in both views: {body}. "
        f"Outfit, identical in both views: {outfit}. "
        "Do not slim or widen either view. No text, no watermark."
    )


def _fallback_bible(topic: str, style_prefix: str) -> CharacterBible:
    body = (
        "one adult guide, medium shoulders, soft middle, "
        "hip width close to shoulder width, never slim and never heavy"
    )
    outfit = "one repeating outfit with the same colors in every view"
    return CharacterBible(
        anchors=[
            "one recurring guide character",
            "same face shape and hair",
            "same outfit colors",
            body,
            "one signature prop",
        ],
        body=body,
        sheet_prompt=_sheet_prompt(style_prefix, body, outfit),
        style_prompt=(
            f"{style_prefix}. Style reference plate for a video about {topic}. "
            "Empty environment, same medium, palette, and lighting, no character, no text."
        ),
    )


def _identity_anchors(anchors: str) -> str:
    parts = [part.strip() for part in anchors.split(",") if part.strip()]
    kept = [part for part in parts if not _PLACE.search(part)]
    return ", ".join(kept[:4])


def _apply_anchors(prompt: str, anchors: str, body: str) -> str:
    anchors = _identity_anchors(anchors)
    if anchors and "Character anchors:" not in prompt:
        needle = "Scene:"
        line = f"Character anchors: {anchors}."
        prompt = (
            prompt.replace(needle, f"{line} {needle}", 1)
            if needle in prompt
            else f"{line} {prompt}"
        )
    lock = f"Body lock: {body}."
    if body and lock not in prompt:
        prompt = f"{lock} {_BODY_RULE} {prompt}"
    return prompt


def run_bible(
    store: RunStore,
    settings: Settings,
    llm: OllamaLLM,
    comfy: ComfyClient,
) -> Storyboard:
    board = store.read_json("storyboard.json", Storyboard)
    brief_topic = ""
    try:
        from videogen.models import Brief

        brief_topic = store.read_json("brief.json", Brief).topic
    except Exception:  # noqa: BLE001
        brief_topic = board.shots[0].narration_text if board.shots else "the topic"

    bible = _fallback_bible(brief_topic, board.style_prefix)
    try:
        data = llm.json_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You design one recurring illustrated character for an educational video. "
                        "Respond ONLY as JSON: "
                        '{"body":"one sentence of build only: shoulders, waist, hips, limb thickness",'
                        '"outfit":"one sentence of clothes and colors",'
                        '"anchors":["5 to 7 short visual anchors"],'
                        '"style_prompt":"empty style-plate prompt"}. '
                        "Pick one body and keep it. Do not say slim in one place and plump in another. "
                        "Anchors cover hair, face, outfit, and one prop. "
                        "Do not put a room, a landscape, or a list of extra objects in the anchors. "
                        "No readable text, logos, or photorealism."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Topic: {brief_topic}\n"
                        f"Style: {board.style_prefix}\n"
                        "Keep the character appropriate to the chosen illustration style."
                    ),
                },
            ],
            temperature=0.3,
        )
        anchors = [str(a).strip() for a in (data.get("anchors") or []) if str(a).strip()]
        if 3 <= len(anchors) <= 8:
            bible.anchors = anchors[:7]
        body = " ".join(str(data.get("body") or "").split())
        outfit = " ".join(str(data.get("outfit") or "").split())
        if len(body.split()) >= 4:
            bible.body = body[:240]
        if outfit:
            bible.sheet_prompt = _sheet_prompt(board.style_prefix, bible.body, outfit[:240])
        style = str(data.get("style_prompt") or "").strip()
        if style:
            bible.style_prompt = f"{board.style_prefix}. {style}"[:700]
    except Exception as e:  # noqa: BLE001
        log.warning("Bible planning failed, using fallbacks: %s", e)

    sheet_path = store.path("frames", "character_sheet.png")
    style_path = store.path("frames", "style_plate.png")
    if not (sheet_path.exists() and sheet_path.stat().st_size > 1000):
        log.info("Generating character sheet")
        comfy.generate_still(
            bible.sheet_prompt,
            sheet_path,
            quality="reference",
            width=1024,
            height=1024,
            filename_prefix=f"videogen_{store.run_id}_sheet",
        )
    if not (style_path.exists() and style_path.stat().st_size > 1000):
        log.info("Generating style plate")
        comfy.generate_still(
            bible.style_prompt,
            style_path,
            quality="reference",
            width=1024,
            height=1024,
            filename_prefix=f"videogen_{store.run_id}_style",
        )
    bible.sheet_path = str(sheet_path.relative_to(store.root))
    bible.style_path = str(style_path.relative_to(store.root))
    anchor_line = ", ".join(bible.anchors)
    for shot in board.shots:
        shot.visual_prompt = _apply_anchors(shot.visual_prompt, anchor_line, bible.body)
    board.bible = bible
    store.write_json("storyboard.json", board)
    return board
