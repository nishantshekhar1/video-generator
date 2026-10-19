from __future__ import annotations

from videogen.config import Settings
from videogen.models import Brief
from videogen.providers.llm import OllamaLLM
from videogen.providers.research import WebResearcher
from videogen.run_store import RunStore


def run_topic(
    store: RunStore,
    settings: Settings,
    llm: OllamaLLM,
    researcher: WebResearcher,
    *,
    topic: str | None,
    duration_s: int,
    style_preset: str,
    pick_topic: bool = False,
) -> Brief:
    if pick_topic or not topic:
        interests = settings.interests or ["curious science that teaches a clear idea"]
        # Light search to seed ideas
        seed_hits = []
        for interest in interests[:3]:
            try:
                seed_hits.extend(researcher.search(interest)[:2])
            except Exception:  # noqa: BLE001
                continue
        seed_text = "\n".join(
            f"- {s.title}: {s.snippet}" for s in seed_hits
        ) or "(no web results)"
        data = llm.json_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You pick one teachable, evocative short-video topic. "
                        "Respond as JSON with keys: topic, angle, teaching_point."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Interest list:\n" + "\n".join(f"- {i}" for i in interests)
                        + f"\n\nWeb sparks:\n{seed_text}\n\n"
                        "Choose one focused topic that can teach something substantial "
                        f"in about {duration_s} seconds."
                    ),
                },
            ]
        )
        topic = str(data.get("topic") or interests[0])
        angle = str(data.get("angle") or "")
        teaching_point = str(data.get("teaching_point") or "")
    else:
        data = llm.json_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Clarify the topic into a short-video brief. "
                        "JSON keys: topic, angle, teaching_point."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Topic: {topic}\nTarget length: {duration_s}s\n"
                        "Return a sharpened topic, a clear angle, and one teaching point."
                    ),
                },
            ]
        )
        topic = str(data.get("topic") or topic)
        angle = str(data.get("angle") or "")
        teaching_point = str(data.get("teaching_point") or "")

    brief = Brief(
        topic=topic,
        angle=angle,
        teaching_point=teaching_point,
        duration_s=duration_s,
        style_preset=style_preset,
    )
    store.write_json("brief.json", brief)
    st = store.status()
    st.topic = topic
    store.save_status(st)
    return brief
