from __future__ import annotations

import json
import re
from typing import Any

import httpx

from videogen.config import Settings


class OllamaLLM:
    """Only language-model entry point. Talks to local Ollama."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.ollama.base_url.rstrip("/")
        self.timeout = settings.ollama.timeout_s
        self.model = (
            settings.ollama.writer_model_quality
            if settings.use_quality_writer
            else settings.ollama.writer_model
        )

    def health(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.4,
        format_json: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                # Cap context — default model ctx can be 128k+ and thrash VRAM/time.
                "num_ctx": self.settings.ollama.num_ctx,
            },
        }
        if format_json:
            payload["format"] = "json"
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.base_url}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
        msg = data.get("message", {})
        content = msg.get("content") or msg.get("thinking") or ""
        return _strip_think(str(content)).strip()

    def json_chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        structured = model or self.settings.ollama.writer_model_structured
        raw = self.chat(
            messages,
            model=structured,
            temperature=temperature,
            format_json=True,
        )
        if not raw.strip():
            raw = self.chat(
                messages,
                model=structured,
                temperature=temperature,
                format_json=False,
            )
        return _parse_json(raw)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"Could not parse JSON from model output: {text[:400]}")
