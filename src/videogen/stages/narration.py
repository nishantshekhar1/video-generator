from __future__ import annotations

from videogen.config import Settings
from videogen.models import NarrationItem, NarrationManifest, Script
from videogen.providers.tts import TTSProvider
from videogen.run_store import RunStore


def run_narration(
    store: RunStore, settings: Settings, tts: TTSProvider
) -> NarrationManifest:
    script = store.read_json("script.json", Script)
    items: list[NarrationItem] = []
    for sent in script.sentences:
        dest = store.path("audio", f"{sent.id}.wav")
        if dest.exists() and dest.stat().st_size > 44:
            # resume
            from videogen.providers.tts import _wav_duration

            dur = _wav_duration(dest)
        else:
            dur = tts.synthesize(sent.text, dest)
        # pad very short clips for readability
        dur = max(dur, 0.8)
        items.append(
            NarrationItem(
                sentence_id=sent.id,
                audio_path=str(dest.relative_to(store.root)),
                duration_s=round(dur, 3),
                text=sent.text,
            )
        )
    manifest = NarrationManifest(
        items=items,
        total_duration_s=round(sum(i.duration_s for i in items), 3),
    )
    store.write_json("narration.json", manifest)
    return manifest
