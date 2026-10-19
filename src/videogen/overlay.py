"""Typeset exact words onto a still. The image model is not asked to paint letters."""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat

from videogen.models import Shot

_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
_SPACE_NOTE = (
    "Leave the lower fifth of the frame empty. Do not paint letters, numbers, or writing."
)
# Naming the character again for every prop made the model paint a second copy.
_ONCE = (
    "Exactly one person, shown once, in a single setting. "
    "Do not draw a second copy, a twin, a split screen, side-by-side panels, or a turnaround."
)
_SETTING = re.compile(
    r"\b(room|study|landscape|scenery|filled with|background|street|square)\b",
    re.IGNORECASE,
)


def image_is_static(path: Path) -> bool:
    """True when a frame is color noise rather than an illustration."""
    image = Image.open(path).convert("RGB").resize((160, 90))
    edges = image.filter(ImageFilter.FIND_EDGES)
    return sum(ImageStat.Stat(edges).mean) / 3 > 70


def single_character(prompt: str) -> str:
    """Keep identity cues, and drop the lines that were painting the person twice."""
    prompt = re.sub(r"Keep one consistent .+?when it returns\.\s*", "", prompt, count=1)

    def _anchors(match: re.Match[str]) -> str:
        parts = [part.strip() for part in match.group(1).split(",") if part.strip()]
        kept = [part for part in parts if not _SETTING.search(part)][:4]
        if not kept:
            return ""
        return "Character anchors: " + ", ".join(kept) + ". "

    prompt = re.sub(r"Character anchors: ([^.]+)\.\s*", _anchors, prompt, count=1)
    if _ONCE not in prompt:
        prompt = f"{_ONCE} {prompt}"
    return prompt


def image_prompt(shot: Shot) -> str:
    prompt = single_character(shot.visual_prompt.strip())
    if shot.overlay_text.strip() and _SPACE_NOTE not in prompt:
        prompt = f"{prompt} {_SPACE_NOTE}"
    return prompt


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=font) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if len(lines) <= 2:
        return lines
    return [lines[0], " ".join(lines[1:])]


def apply_overlay(path: Path, text: str) -> None:
    text = " ".join((text or "").split())
    if not text or not path.exists():
        return
    im = Image.open(path).convert("RGBA")
    w, h = im.size
    size = max(28, w // 18)
    font = ImageFont.truetype(str(_FONT), size=size)
    draw = ImageDraw.Draw(im)
    lines = _wrap(draw, text, font, int(w * 0.86))
    line_h = int(size * 1.25)
    pad = int(size * 0.55)
    bar_h = line_h * len(lines) + pad * 2
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    pen = ImageDraw.Draw(layer)
    margin = int(w * 0.04)
    top = h - bar_h - int(h * 0.035)
    pen.rounded_rectangle(
        (margin, top, w - margin, top + bar_h),
        radius=size // 3,
        fill=(12, 16, 28, 196),
    )
    y = top + pad
    for line in lines:
        tw = pen.textlength(line, font=font)
        x = (w - tw) / 2
        pen.text(
            (x, y),
            line,
            font=font,
            fill=(255, 252, 245, 255),
            stroke_width=max(1, size // 18),
            stroke_fill=(0, 0, 0, 230),
        )
        y += line_h
    Image.alpha_composite(im, layer).convert("RGB").save(path)
