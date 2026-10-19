from __future__ import annotations

import logging

from videogen.config import Settings
from videogen.delivery import shot_count_for, split_duration
from videogen.models import Brief, CameraMove, NarrationManifest, Script, Shot, Storyboard
from videogen.providers.llm import OllamaLLM
from videogen.run_store import RunStore

log = logging.getLogger(__name__)

_ROLES = ("establishing", "action", "insert")
_CAMERAS: dict[str, CameraMove] = {
    "establishing": "pull",
    "action": "push",
    "insert": "hold",
}


def _style_prefix(settings: Settings, style_preset: str) -> str:
    preset = style_preset or "sketch"
    style_extra = settings.pipeline.style_presets.get(
        preset, settings.pipeline.style_presets.get("sketch", "")
    )
    base = settings.pipeline.style_prefix.strip().rstrip(",")
    extra = style_extra.strip().rstrip(",")
    if extra and extra not in base:
        return f"{base}, {extra}"
    return base or extra


def _continuity(preset: str, topic: str) -> tuple[str, str]:
    if preset == "retro_digital":
        return (
            f"Keep one consistent pixel-art sprite for topic '{topic}' "
            "across frames; same palette, silhouette, and neon mood when it returns.",
            "Retro digital pixel-art still that illustrates this mature spoken line "
            "(do not write the words on the image)",
        )
    if preset == "classic_cartoon":
        return (
            f"Keep one consistent classic cartoon character for topic '{topic}' "
            "across frames; same ink line, cel colors, and proportions when it returns.",
            "Classic vintage cartoon still that illustrates this mature spoken line "
            "(do not write the words on the image)",
        )
    if preset == "storybook":
        return (
            f"Keep one consistent picture-book character for topic '{topic}' "
            "across frames; same ink line, painted colors, and proportions when it returns.",
            "Storybook illustration still that illustrates this mature spoken line "
            "(do not write the words on the image)",
        )
    if preset == "sketch":
        return (
            f"Keep one consistent cute cartoon character for topic '{topic}' "
            "across frames; same proportions, face, and colored-pencil look when it returns.",
            "Whimsical toddler cartoon still that illustrates this mature spoken line "
            "(do not write the words on the image)",
        )
    return (
        f"Keep consistent subject for topic '{topic}' across frames; "
        "same character design when the subject returns.",
        "Still that illustrates this mature spoken line (do not write the words on the image)",
    )


def _fallback_scenes(hints: list[str]) -> dict[str, str]:
    first = hints[0]
    mid = hints[len(hints) // 2]
    last = hints[-1]
    return {
        "establishing": f"Wide establishing view of the setting. {first}",
        "action": mid,
        "insert": f"Close insert of the key object or gesture. {last}",
    }


def _clean_overlay(value: object) -> str:
    text = " ".join(str(value or "").split())
    words = text.split()
    if not words or len(words) > 8 or len(text) > 48:
        return ""
    return text


def _plan_scenes(
    llm: OllamaLLM,
    *,
    topic: str,
    style_prefix: str,
    role: str,
    narration: str,
    hints: list[str],
) -> tuple[dict[str, str], dict[str, str]]:
    try:
        data = llm.json_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You break one narration beat into three shots for an illustrated video. "
                        "establishing = wide place, action = the deed, insert = one close detail. "
                        "Concrete subject and action. Keep the same body build in every shot. "
                        "The image model must not paint letters. "
                        "If a shot needs words (a sign, a label, a title card), put the exact "
                        "spelling in overlay_establishing, overlay_action, or overlay_insert. "
                        "Leave an overlay empty when the picture needs no words. "
                        "An overlay is at most 8 words, never the narration sentence. "
                        "Respond ONLY as JSON: "
                        '{"establishing":"...","action":"...","insert":"...",'
                        '"overlay_establishing":"","overlay_action":"","overlay_insert":""}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Topic: {topic}\nRole: {role}\nStyle: {style_prefix}\n"
                        f"Narration: {narration}\nExisting hints: {hints}"
                    ),
                },
            ],
            temperature=0.3,
        )
        out = {k: str(data.get(k) or "").strip() for k in _ROLES}
        if all(out.values()):
            overlays = {k: _clean_overlay(data.get(f"overlay_{k}")) for k in _ROLES}
            return {k: v[:400] for k, v in out.items()}, overlays
    except Exception as e:  # noqa: BLE001
        log.warning("Coverage planning failed: %s", e)
    return _fallback_scenes(hints or [narration]), {k: "" for k in _ROLES}


# Fewer holds each picture longer. More cuts sooner, still above a short flash.
_DENSITY = {
    "fewer": (10.0, 20.0),
    "standard": (5.0, 12.0),
    "more": (3.0, 6.0),
}


def _hold_bounds(density: str) -> tuple[float, float]:
    return _DENSITY.get(density, _DENSITY["standard"])


def _groups(
    sentences: list,
    dur_map: dict[str, float],
    *,
    min_hold: float,
    max_hold: float,
) -> list[list]:
    """Hold one picture until the speech has lasted at least min_hold seconds."""
    groups: list[list] = []
    current: list = []
    for sent in sentences:
        if not current:
            current = [sent]
            continue
        have = sum(dur_map.get(s.id, 2.0) for s in current)
        nxt = dur_map.get(sent.id, 2.0)
        if have < min_hold and have + nxt <= max_hold:
            current.append(sent)
        else:
            groups.append(current)
            current = [sent]
    if current:
        groups.append(current)
    if len(groups) >= 2:
        last = sum(dur_map.get(s.id, 2.0) for s in groups[-1])
        prev = sum(dur_map.get(s.id, 2.0) for s in groups[-2])
        if last < min_hold and prev + last <= max_hold:
            groups[-2].extend(groups[-1])
            groups.pop()
    return groups


def _plan_one_scene(
    llm: OllamaLLM,
    *,
    topic: str,
    style_prefix: str,
    narration: str,
    hints: list[str],
) -> tuple[str, str]:
    """One picture that shows the things the spoken lines name."""
    try:
        data = llm.json_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You choose one illustration for a stretch of narration. "
                        "The picture must show the objects and action the words name. "
                        "If the words mention lipstick, coffee, a receipt, or a shop, those are in the frame. "
                        "Do not invent a different subject. "
                        "The image model must not paint letters. "
                        "overlay is the exact words only when the picture needs a sign or label, "
                        "at most 8 words, else empty. "
                        "Respond ONLY as JSON: {\"scene\":\"...\",\"overlay\":\"\"}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Topic: {topic}\nStyle: {style_prefix}\n"
                        f"Spoken words: {narration}\n"
                        f"Sentence hints: {hints}"
                    ),
                },
            ],
            temperature=0.2,
        )
        scene = str(data.get("scene") or "").strip()
        if scene:
            return scene[:500], _clean_overlay(data.get("overlay"))
    except Exception as e:  # noqa: BLE001
        log.warning("Scene planning failed: %s", e)
    scene = " ".join(h for h in hints if h) or narration
    return scene[:500], ""


def _mark_heroes(shots: list[Shot], limit: int) -> None:
    chosen = 0
    for shot in shots:
        if shot.role == "hook":
            shot.hero = True
            chosen += 1
            break
    if chosen >= limit or not shots:
        return
    longest = max((s for s in shots if not s.hero), key=lambda s: s.duration_s, default=None)
    if longest is not None:
        longest.hero = True


def run_storyboard(store: RunStore, settings: Settings, llm: OllamaLLM) -> Storyboard:
    brief = store.read_json("brief.json", Brief)
    script = store.read_json("script.json", Script)
    narration = store.read_json("narration.json", NarrationManifest)
    dur_map = {i.sentence_id: i.duration_s for i in narration.items}

    preset = brief.style_preset or "sketch"
    style_prefix = _style_prefix(settings, preset)
    continuity, sync_line = _continuity(preset, brief.topic)

    density = str(store.status().meta.get("still_density") or "standard")
    min_hold, max_hold = _hold_bounds(density)
    shots: list[Shot] = []
    for index, sentences in enumerate(
        _groups(script.sentences, dur_map, min_hold=min_hold, max_hold=max_hold)
    ):
        role = sentences[0].role or "beat"
        spoken = " ".join(s.text.strip() for s in sentences)
        hints = [(s.visual_hint or s.text).strip() for s in sentences]
        scene, overlay = _plan_one_scene(
            llm,
            topic=brief.topic,
            style_prefix=style_prefix,
            narration=spoken,
            hints=hints,
        )
        duration = round(sum(dur_map.get(s.id, 2.0) for s in sentences), 3)
        prompt = (
            f"{style_prefix}. {continuity} "
            f"Scene: {scene}. "
            f"{sync_line}: {spoken}"
        )
        shots.append(
            Shot(
                id=f"shot_{index:02d}",
                sentence_id=sentences[0].id,
                sentence_ids=[s.id for s in sentences],
                narration_text=spoken,
                visual_prompt=prompt,
                overlay_text=overlay,
                camera="hold",
                duration_s=duration,
                role=role,
                coverage="sentence",
            )
        )

    _mark_heroes(shots, settings.pipeline.hero_count)
    board = Storyboard(shots=shots, style_prefix=style_prefix)
    store.write_json("storyboard.json", board)
    return board
