from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

import httpx

from videogen.config import Settings
from videogen.providers.llm import _strip_think


class OllamaVision:
    """Only vision-model entry point. Talks to local Ollama VLM."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.ollama.base_url.rstrip("/")
        self.timeout = settings.ollama.timeout_s
        self.model = settings.ollama.vision_model

    def health(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def review_frame(
        self,
        image_path: Path,
        *,
        expected_subject: str,
        style_notes: str,
        reference_path: Path | None = None,
    ) -> dict[str, Any]:
        """Return {ok: bool, reasons: str, retry_prompt: str|None}."""
        images = [base64.b64encode(image_path.read_bytes()).decode("ascii")]
        identity = ""
        if reference_path is not None and reference_path.exists():
            images.append(base64.b64encode(reference_path.read_bytes()).decode("ascii"))
            identity = (
                "Image 2 is the approved character sheet. The character in image 1 "
                "must match that sheet's face, hair, outfit, proportions, and prop. "
            )
        prompt = (
            "You are reviewing a keyframe for an illustrated educational video.\n"
            f"Expected subject: {expected_subject}\n"
            f"Style notes: {style_notes}\n"
            f"{identity}"
            "Check: (1) subject clearly matches the expected scene (not unrelated imagery), "
            "(2) style coherence with the notes (e.g. toddler colored-pencil, classic "
            "cel cartoon, picture-book gouache/watercolor, or retro pixel/synthwave "
            "when requested; not photoreal photos "
            "or UI screenshots), "
            "(3) garbled letters painted into the scene are a failure. "
            "A clean typeset caption in the lower band is intentional and is OK, "
            "(4) if a character sheet was provided, the face, outfit, and body width must match it. "
            "A torso or hips that look clearly thinner or heavier than the sheet is a failure.\n"
            "Respond ONLY as JSON: "
            '{"ok": true|false, "reasons": "...", "retry_prompt": "improved image prompt or empty"}'
        )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": images,
                }
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "num_ctx": 4096},
        }
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.base_url}/api/chat", json=payload)
            r.raise_for_status()
            content = r.json().get("message", {}).get("content", "")
        content = _strip_think(content)
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", content, re.DOTALL)
            data = json.loads(m.group(0)) if m else {"ok": True, "reasons": "parse_fallback"}
        return {
            "ok": bool(data.get("ok", True)),
            "reasons": str(data.get("reasons", "")),
            "retry_prompt": (data.get("retry_prompt") or None),
        }
