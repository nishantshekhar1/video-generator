"""Edit one shot's picture prompt, on-image words, and spoken line."""

from __future__ import annotations

import logging

from videogen.config import Settings
from videogen.models import NarrationManifest, Script, StageName, StageStatus, Storyboard
from videogen.overlay import apply_overlay, image_prompt, still_seed
from videogen.providers.comfy import ComfyClient
from videogen.providers.tts import TTSProvider
from videogen.run_store import RunStore

log = logging.getLogger(__name__)

_ORDER = ("establishing", "action", "insert")


def _beat_shots(board: Storyboard, script: Script, sentence_id: str):
    beat = next(s.beat_index for s in script.sentences if s.id == sentence_id)
    ids = {s.id for s in script.sentences if s.beat_index == beat}
    shots = [s for s in board.shots if s.sentence_id in ids]
    shots.sort(key=lambda s: _ORDER.index(s.coverage) if s.coverage in _ORDER else 9)
    return ids, shots


def _invalidate(store: RunStore, board: Storyboard, *, all_clips: bool, shot_id: str) -> None:
    targets = board.shots if all_clips else [s for s in board.shots if s.id == shot_id]
    for shot in targets:
        clip = store.path("clips", f"{shot.id}.mp4")
        if clip.exists():
            clip.unlink()
        shot.clip_path = None
        for name in (f"{shot.id}_mid.png", f"{shot.id}_end.png"):
            hero = store.path("frames", "heroes", name)
            if hero.exists():
                hero.unlink()
    st = store.status()
    for stage in (StageName.MOTION, StageName.ASSEMBLE):
        if st.stages.get(stage.value) in (StageStatus.DONE, StageStatus.FAILED, StageStatus.RUNNING):
            st.stages[stage.value] = StageStatus.PENDING
    st.meta["needs_rebuild"] = True
    st.error = None
    store.save_status(st)


def revise_shot(
    store: RunStore,
    settings: Settings,
    comfy: ComfyClient,
    tts: TTSProvider,
    shot_id: str,
    *,
    visual_prompt: str,
    narration_text: str,
    overlay_text: str,
) -> Storyboard:
    board = store.read_json("storyboard.json", Storyboard)
    script = store.read_json("script.json", Script)
    narration = store.read_json("narration.json", NarrationManifest)
    shot = next((s for s in board.shots if s.id == shot_id), None)
    if shot is None:
        raise KeyError(shot_id)

    visual_prompt = visual_prompt.strip() or shot.visual_prompt
    narration_text = " ".join(narration_text.split()) or shot.narration_text
    overlay_text = " ".join(overlay_text.split())

    speech_changed = narration_text != shot.narration_text
    picture_changed = visual_prompt != shot.visual_prompt or overlay_text != shot.overlay_text

    shot.visual_prompt = visual_prompt
    shot.overlay_text = overlay_text
    if speech_changed:
        shot.narration_text = narration_text
        for other in board.shots:
            if other.sentence_id == shot.sentence_id:
                other.narration_text = narration_text
        for sentence in script.sentences:
            if sentence.id == shot.sentence_id:
                sentence.text = narration_text
        audio = store.path("audio", f"{shot.sentence_id}.wav")
        dur = max(tts.synthesize(narration_text, audio), 0.8)
        for item in narration.items:
            if item.sentence_id == shot.sentence_id:
                item.text = narration_text
                item.duration_s = round(dur, 3)
                item.audio_path = str(audio.relative_to(store.root))
        for board_shot in board.shots:
            ids = board_shot.sentence_ids or [board_shot.sentence_id]
            board_shot.duration_s = round(
                sum(i.duration_s for i in narration.items if i.sentence_id in set(ids)),
                3,
            )
        narration.total_duration_s = round(sum(i.duration_s for i in narration.items), 3)
        script.word_count = sum(len(s.text.split()) for s in script.sentences)
        store.write_json("script.json", script)
        store.write_json("narration.json", narration)

    if picture_changed or speech_changed:
        _invalidate(store, board, all_clips=speech_changed, shot_id=shot.id)

    if picture_changed:
        draft = store.path("frames", "drafts", f"{shot.id}.png")
        final = store.path("frames", f"{shot.id}.png")
        had_final = final.exists()
        for path in (draft, final):
            if path.exists():
                path.unlink()
        shot.draft_path = None
        shot.frame_path = None
        reference = None
        if had_final and board.bible and board.bible.sheet_path:
            sheet = store.path(board.bible.sheet_path)
            if sheet.exists():
                reference = sheet
        dest = final if had_final else draft
        quality = "final" if had_final else "draft"
        log.info("Regenerating %s for %s", quality, shot.id)
        comfy.generate_still(
            image_prompt(shot),
            dest,
            seed=still_seed(store.run_id, shot.id, bump=3),
            quality=quality,
            reference=reference,
            filename_prefix=f"videogen_{store.run_id}_{shot.id}_edit",
        )
        apply_overlay(dest, shot.overlay_text)
        rel = str(dest.relative_to(store.root))
        if had_final:
            shot.frame_path = rel
            draft.parent.mkdir(parents=True, exist_ok=True)
            draft.write_bytes(dest.read_bytes())
            shot.draft_path = str(draft.relative_to(store.root))
        else:
            shot.draft_path = rel

    store.write_json("storyboard.json", board)
    return board
