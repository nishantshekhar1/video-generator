from __future__ import annotations

import json
import logging
import re

from videogen.config import Settings
from videogen.models import Brief, Script, ScriptBeat, ScriptSentence
from videogen.providers.llm import OllamaLLM
from videogen.run_store import RunStore

log = logging.getLogger(__name__)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


_STYLE_PLANNER: dict[str, tuple[str, str]] = {
    # system guidance, user-style blurb
    "sketch": (
        "VISUALS: funny toddler-friendly cartoon scenes as if a cartoonist "
        "hand-drew them with colored pencils. Write scenes a toddler could enjoy "
        "while an adult hears the mature line. Use cute characters, silly "
        "expressions, playful exaggeration; keep one continuous cute character design.",
        "funny colored-pencil cartoon for toddlers; narration stays adult/educational",
    ),
    "retro_digital": (
        "VISUALS: retro digital pixel-art scenes like 8-bit/16-bit video game assets "
        "with synthwave neon mood. Describe chunky pixel sprites, limited palettes, "
        "arcade key-art compositions, neon magenta/cyan glow. Keep one continuous "
        "sprite/character design across shots. Still show the real idea (subject + action).",
        "pixel art / 8-bit–16-bit game asset + synthwave neon; narration stays adult",
    ),
    "classic_cartoon": (
        "VISUALS: classic theatrical cartoon scenes (1930s–1950s rubber-hose / cel "
        "animation vibe). Clean ink outlines, flat colors, squash-and-stretch poses, "
        "expressive faces. Keep one continuous cartoon character design across shots. "
        "Still show the real idea (subject + action).",
        "classic vintage cartoon / cel animation; narration stays adult/educational",
    ),
    "storybook": (
        "VISUALS: children's picture-book plates painted in gouache and watercolor "
        "on warm paper, with a delicate ink line and richly painted scenery. "
        "Each shot is a narrative illustration: one clear character acting in a "
        "storybook setting. Keep one continuous picture-book character design "
        "across shots. Still show the real idea (subject + action).",
        "picture-book gouache and watercolor illustration; narration stays adult/educational",
    ),
    "painterly": (
        "VISUALS: evocative painterly graphic stills with bold shapes and cinematic light. "
        "Concrete subject and action; keep one continuous character/design across shots.",
        "painterly graphic essay still; narration stays adult/educational",
    ),
    "graphic": (
        "VISUALS: flat graphic poster illustration with bold color blocks. "
        "Concrete subject and action; keep one continuous design across shots.",
        "flat graphic poster illustration; narration stays adult/educational",
    ),
}


def _fallback_visual_hint(text: str, topic: str, style_name: str = "sketch") -> str:
    """Concrete-ish scene when the planner call fails."""
    cleaned = re.sub(r"[\"']", "", text).strip()
    if style_name == "retro_digital":
        return (
            f"16-bit pixel-art game scene about {topic}: chunky sprite acting out "
            f"the idea behind '{cleaned}', limited palette, synthwave neon glow, "
            "simple arcade backdrop, one focal subject, no text"
        )
    if style_name == "classic_cartoon":
        return (
            f"Classic vintage cartoon still about {topic}: lively cel-animated character "
            f"acting out the idea behind '{cleaned}', ink outlines, flat colors, "
            "squash-and-stretch pose, painted backdrop, one focal subject, no text"
        )
    if style_name == "storybook":
        return (
            f"Children's picture-book plate about {topic}: painted character acting out "
            f"the idea behind '{cleaned}', gouache and watercolor, delicate ink line, "
            "warm cream paper, rich scenery, one focal subject, no text"
        )
    if style_name == "graphic":
        return (
            f"Bold flat graphic poster about {topic}: clear subject acting out "
            f"'{cleaned}', simple shapes, strong color blocks, no text"
        )
    if style_name == "painterly":
        return (
            f"Painterly educational still about {topic}: concrete subject acting out "
            f"'{cleaned}', bold shapes, cinematic light, no text"
        )
    return (
        f"Funny toddler cartoon about {topic}: cute simplified character acting out "
        f"the idea behind '{cleaned}', big expressive eyes, silly pose, "
        "colored pencil on paper, one focal subject, simple cozy background, no text"
    )


def _plan_visual_hints(
    llm: OllamaLLM,
    *,
    brief: Brief,
    title: str,
    sentences: list[ScriptSentence],
) -> None:
    """Replace narration-echo hints with concrete scene descriptions."""
    if not sentences:
        return

    style_name = brief.style_preset or "sketch"
    style_sys, style_user = _STYLE_PLANNER.get(
        style_name, _STYLE_PLANNER["sketch"]
    )
    payload = [{"id": s.id, "narration": s.text} for s in sentences]
    try:
        data = llm.json_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a storyboard artist for an educational video. "
                        "AUDIO/narration is mature and factual; visuals follow the art brief. "
                        f"{style_sys} "
                        "For each narration line, write a CONCRETE visual scene of what that line is about. "
                        "The objects and actions named in the line must be in the picture. "
                        "If the line mentions lipstick, coffee, a shop, or a person doing something, show that, "
                        "not a different metaphor. "
                        "No text, labels, logos, UI, watermarks, or photorealism. "
                        "No chain-of-thought. Respond ONLY as JSON: "
                        '{"visuals":[{"id":"s00_00","visual_hint":"..."}]} '
                        "with one entry per input sentence id."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Topic: {brief.topic}\n"
                        f"Title: {title}\n"
                        f"Teaching point: {brief.teaching_point}\n"
                        f"Art style: {style_name} — {style_user}\n"
                        f"Sentences:\n{json.dumps(payload, ensure_ascii=False)}"
                    ),
                },
            ],
            temperature=0.35,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("Visual planning failed, using fallbacks: %s", e)
        for s in sentences:
            s.visual_hint = _fallback_visual_hint(s.text, brief.topic, style_name)
        return

    by_id: dict[str, str] = {}
    for item in data.get("visuals") or []:
        sid = str(item.get("id") or "").strip()
        hint = str(item.get("visual_hint") or "").strip()
        if sid and hint:
            by_id[sid] = hint

    for s in sentences:
        hint = by_id.get(s.id, "").strip()
        # Reject hints that are just the narration repeated.
        if not hint or hint.lower().rstrip(".") == s.text.lower().rstrip("."):
            s.visual_hint = _fallback_visual_hint(s.text, brief.topic, style_name)
        else:
            s.visual_hint = hint[:400]


def run_script(store: RunStore, settings: Settings, llm: OllamaLLM) -> Script:
    brief = store.read_json("brief.json", Brief)
    research_raw = store.read_text("research.md")
    if "# Synthesis" in research_raw:
        research = research_raw.split("# Synthesis", 1)[-1].strip()
    else:
        research = research_raw
    research = research[:4500]
    if brief.notes:
        research = (brief.notes[:2500] + "\n\n" + research)[:4500]

    word_budget = max(
        40, int(brief.duration_s * settings.pipeline.words_per_minute / 60)
    )

    data = llm.json_chat(
        [
            {
                "role": "system",
                "content": (
                    "You write evocative educational narration scripts for adults. "
                    "Tone: mature, curious, clear — never baby-talk or toddler wording. "
                    "Only use claims supported by the research synthesis. "
                    "Visuals may be playful cartoons later; your job is grown-up voiceover. "
                    "The hook's first sentence states the payoff the title promises, "
                    "in plain speech. No throat-clearing and no 'in this video'. "
                    "No chain-of-thought; output JSON only. "
                    "Respond as JSON with keys: title, teaching_point, beats "
                    "(list of {role: hook|beat|close, text}). "
                    "Structure: one hook, 2-3 beats, one close. "
                    "Keep total words near the budget. Short punchy sentences."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Topic: {brief.topic}\nAngle: {brief.angle}\n"
                    f"Teaching point: {brief.teaching_point}\n"
                    f"Word budget: ~{word_budget}\n"
                    f"Duration: {brief.duration_s}s\n\n"
                    f"RESEARCH:\n{research}"
                ),
            },
        ]
    )

    beats_raw = data.get("beats") or []
    beats: list[ScriptBeat] = []
    for b in beats_raw:
        role = str(b.get("role") or "beat").lower()
        if role in ("hook", "close"):
            pass
        elif role in ("beat", "reveal", "explanation", "body", "middle"):
            role = "beat"
        else:
            role = "beat"
        text = str(b.get("text") or "").strip()
        if text:
            beats.append(ScriptBeat(role=role, text=text))
    if not beats:
        raise RuntimeError("Script model returned no beats")

    sentences: list[ScriptSentence] = []
    for bi, beat in enumerate(beats):
        for si, sent in enumerate(_split_sentences(beat.text)):
            sid = f"s{bi:02d}_{si:02d}"
            sentences.append(
                ScriptSentence(
                    id=sid,
                    beat_index=bi,
                    role=beat.role,
                    text=sent,
                )
            )

    title = str(data.get("title") or brief.topic)
    _plan_visual_hints(llm, brief=brief, title=title, sentences=sentences)

    words = sum(len(s.text.split()) for s in sentences)
    script = Script(
        title=title,
        teaching_point=str(data.get("teaching_point") or brief.teaching_point),
        beats=beats,
        sentences=sentences,
        word_count=words,
        target_duration_s=brief.duration_s,
    )
    store.write_json("script.json", script)
    return script
