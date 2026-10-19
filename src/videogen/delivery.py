"""Pure helpers for the YouTube master: captions, chapters, loudness."""

from __future__ import annotations

import json


def split_duration(total: float, n: int) -> list[float]:
    """Split a beat across establishing / action / insert."""
    n = max(1, n)
    if n == 1:
        return [round(total, 3)]
    weights = [0.30, 0.45, 0.25][:n]
    scale = sum(weights)
    parts = [round(total * w / scale, 3) for w in weights]
    parts[-1] = round(total - sum(parts[:-1]), 3)
    return parts


def shot_count_for(duration: float) -> int:
    if duration >= 6:
        return 3
    if duration >= 3.2:
        return 2
    return 1


def wrap_caption(text: str, width: int = 40) -> str:
    words = text.split()
    if not words:
        return text
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if len(trial) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if len(lines) <= 2:
        return "\n".join(lines)
    return lines[0] + "\n" + " ".join(lines[1:])


def srt_timestamp(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"


def vtt_timestamp(seconds: float) -> str:
    return srt_timestamp(seconds).replace(",", ".")


def chapter_clock(seconds: float) -> str:
    whole = int(max(0, round(seconds)))
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def pack_chapters(
    segments: list[tuple[float, float, str]],
    *,
    min_len: float = 10.0,
    min_count: int = 3,
) -> list[tuple[float, str]]:
    """YouTube chapters: start at 0, at least three, each at least 10 seconds.

    Returns an empty list when the cut is too short to satisfy those rules.
    """
    if not segments:
        return []
    groups: list[tuple[float, float, str]] = []
    i = 0
    while i < len(segments):
        start, end, label = segments[i]
        i += 1
        while end - start < min_len and i < len(segments):
            end = segments[i][1]
            i += 1
        groups.append((start, end, label))
    if len(groups) >= 2 and groups[-1][1] - groups[-1][0] < min_len:
        prev_s, _prev_e, prev_label = groups[-2]
        groups[-2] = (prev_s, groups[-1][1], prev_label)
        groups.pop()
    if len(groups) < min_count:
        return []
    if any(end - start < min_len - 0.05 for start, end, _ in groups):
        return []
    groups[0] = (0.0, groups[0][1], groups[0][2])
    return [(start, label) for start, _end, label in groups]


def format_chapters(chapters: list[tuple[float, str]]) -> str:
    if not chapters:
        return (
            "# This cut is shorter than 30 seconds, so a YouTube chapter list\n"
            "# would be rejected (needs 00:00, at least three chapters, each >= 10s).\n"
        )
    return "".join(f"{chapter_clock(t)} {label}\n" for t, label in chapters)


def parse_loudnorm(stderr: str) -> dict[str, str]:
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError("loudnorm measurement did not return JSON")
    data = json.loads(stderr[start : end + 1])
    needed = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    missing = [k for k in needed if k not in data]
    if missing:
        raise RuntimeError(f"loudnorm JSON missing {missing}")
    return {k: str(data[k]).strip() for k in needed}


def loudnorm_filter(
    measured: dict[str, str],
    *,
    integrated: float = -14.0,
    true_peak: float = -1.5,
    lra: float = 11.0,
) -> str:
    return (
        f"loudnorm=I={integrated}:TP={true_peak}:LRA={lra}:"
        f"measured_I={measured['input_i']}:"
        f"measured_TP={measured['input_tp']}:"
        f"measured_LRA={measured['input_lra']}:"
        f"measured_thresh={measured['input_thresh']}:"
        f"offset={measured['target_offset']}:"
        "linear=true"
    )
