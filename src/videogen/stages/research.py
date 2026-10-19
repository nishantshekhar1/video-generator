from __future__ import annotations

from videogen.config import Settings
from videogen.models import Brief
from videogen.providers.llm import OllamaLLM
from videogen.providers.research import WebResearcher
from videogen.run_store import RunStore


def run_research(
    store: RunStore,
    settings: Settings,
    llm: OllamaLLM,
    researcher: WebResearcher,
) -> Brief:
    brief = store.read_json("brief.json", Brief)
    query = f"{brief.topic} {brief.angle}".strip()
    sources, notes = researcher.research(query)
    brief.sources = sources

    synthesis = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "You synthesize research notes for a short educational video. "
                    "Only use claims present in the notes. Flag uncertainty. "
                    "Write a concise sourced brief in markdown with: "
                    "Key facts, Surprises, Teaching spine, Open questions."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Topic: {brief.topic}\nAngle: {brief.angle}\n"
                    f"Teaching point: {brief.teaching_point}\n\nNOTES:\n{notes}"
                ),
            },
        ],
        model=settings.ollama.writer_model_prose,
        temperature=0.3,
    )
    brief.notes = synthesis
    store.write_text("research.md", notes + "\n\n---\n\n# Synthesis\n\n" + synthesis)
    store.write_json("brief.json", brief)
    return brief
