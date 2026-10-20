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
# A wide frame was being painted as two copies of the same page.
_ONCE = (
    "One scene filling the whole frame. "
    "Not a diptych, not a two-page spread, not a second copy of the same picture."
)
_SETTING = re.compile(
    r"\b(room|study|landscape|scenery|filled with|background|street|square)\b",
    re.IGNORECASE,
)


def collapse_diptych(path: Path) -> bool:
    """Keep one half when the two sides are copies of the same picture.

    A front-and-side character sheet is a mirror pair, not a copy, so it stays.
    Two different people in one shop stay as well.
    """
    import numpy as np

    image = Image.open(path).convert("RGB")
    width, height = image.size
    if width < int(height * 1.3):
        return False
    left_im = image.crop((0, 0, width // 2, height))
    right_im = image.crop((width // 2, 0, width, height))
    left = np.asarray(left_im.resize((32, 32)), dtype=np.float32).reshape(-1)
    right = np.asarray(right_im.resize((32, 32)), dtype=np.float32).reshape(-1)
    mirror = np.asarray(
        right_im.transpose(Image.Transpose.FLIP_LEFT_RIGHT).resize((32, 32)),
        dtype=np.float32,
    ).reshape(-1)

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if float(a.std()) < 1 or float(b.std()) < 1:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    copied = _corr(left, right)
    mirrored = _corr(left, mirror)
    if not (copied > 0.42 and copied > mirrored + 0.05):
        return False
    pixels = np.asarray(image, dtype=np.float32)
    dark = pixels.mean(axis=(0, 2))
    jump = np.abs(pixels[:, 1:] - pixels[:, :-1]).mean(axis=(0, 2))
    lo, hi = int(width * 0.40), int(width * 0.58)
    dark_cols = np.where(dark[lo:hi] < 30)[0]
    if len(dark_cols):
        cut = lo + int(dark_cols[0])
    else:
        cut = lo + int(jump[lo:hi].argmax())
    cut = max(width // 3, min(cut, width // 2))
    page = image.crop((0, 0, cut, height)).resize((width, height), Image.Resampling.LANCZOS)
    page.save(path)
    return True


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


def scene_text(prompt: str) -> str:
    """The action and place, without the locked style and character sentences."""
    match = re.search(r"Scene:\s*(.+)", prompt, re.DOTALL)
    text = match.group(1) if match else prompt
    text = re.split(
        r"(?:Storybook illustration still|Children's picture-book still|"
        r"Whimsical toddler|Classic vintage|Retro digital|Still that illustrates)",
        text,
        maxsplit=1,
    )[0]
    return " ".join(text.split())[:500]


def compose_shot_prompt(style: str, identity: str, scene: str) -> str:
    """One verbatim style phrase and one verbatim character sentence, then the new scene."""
    parts = []
    if style:
        parts.append(f"Style, use these words exactly: {style}.")
    if identity:
        parts.append(
            f"The same character in every frame: {identity}. "
            "Keep this face, hair, outfit, and body. Change only the action and the place."
        )
    parts.append(f"Scene: {scene}.")
    return " ".join(parts)


def still_seed(run_id: str, shot_id: str, bump: int = 0) -> int:
    """Same shot keeps the same noise, so a retry is a choice rather than a new person."""
    import zlib

    base = zlib.adler32(f"{run_id}:{shot_id}".encode()) % (2**31 - 1)
    return base + bump


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
