from __future__ import annotations

import logging
from pathlib import Path

from videogen.cancel import Stopped, run_command
from videogen.config import Settings
from videogen.models import Shot, Storyboard
from videogen.providers.comfy import ComfyClient
from videogen.run_store import RunStore

log = logging.getLogger(__name__)


def _zoompan_filter(camera: str, frames: int, w: int, h: int) -> str:
    # Ken Burns style on a still. z starts at 1.0.
    if camera == "push":
        z_expr = f"min(1.15,1+0.15*on/{max(frames - 1, 1)})"
        x_expr = f"(iw-iw/zoom)/2"
        y_expr = f"(ih-ih/zoom)/2"
    elif camera == "pull":
        z_expr = f"max(1.0,1.15-0.15*on/{max(frames - 1, 1)})"
        x_expr = f"(iw-iw/zoom)/2"
        y_expr = f"(ih-ih/zoom)/2"
    elif camera == "pan_left":
        z_expr = "1.1"
        x_expr = f"(iw-iw/zoom)*on/{max(frames - 1, 1)}"
        y_expr = "(ih-ih/zoom)/2"
    elif camera == "pan_right":
        z_expr = "1.1"
        x_expr = f"(iw-iw/zoom)*(1-on/{max(frames - 1, 1)})"
        y_expr = "(ih-ih/zoom)/2"
    else:  # hold
        z_expr = "1.05"
        x_expr = "(iw-iw/zoom)/2"
        y_expr = "(ih-ih/zoom)/2"
    return (
        f"scale=8000:-1,"
        f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':"
        f"d={frames}:s={w}x{h}:fps={24},"
        f"format=yuv420p"
    )


def _run_ffmpeg(cmd: list[str]) -> None:
    proc = run_command(cmd)
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-2000:]
        raise RuntimeError(tail or "ffmpeg failed")


def _sheet(store: RunStore, board: Storyboard):
    if board.bible and board.bible.sheet_path:
        path = store.path(board.bible.sheet_path)
        if path.exists():
            return path
    return None


def _hero_frames(
    store: RunStore,
    comfy: ComfyClient | None,
    shot: Shot,
    start: Path,
    reference,
) -> tuple[Path, Path] | None:
    """Two later moments of the same illustrated shot, for a short dissolve."""
    if comfy is None:
        return None
    mid = store.path("frames", "heroes", f"{shot.id}_mid.png")
    end = store.path("frames", "heroes", f"{shot.id}_end.png")
    prompts = {
        mid: (
            shot.visual_prompt
            + " Mid-action: the same character has begun the motion, "
            "same wardrobe, same side of the frame."
        ),
        end: (
            shot.visual_prompt
            + " End of the action: the same character has finished the motion, "
            "same wardrobe, same side of the frame."
        ),
    }
    try:
        for dest, prompt in prompts.items():
            if dest.exists() and dest.stat().st_size > 1000:
                continue
            comfy.generate_still(
                prompt,
                dest,
                quality="final",
                reference=reference,
                filename_prefix=f"videogen_hero_{shot.id}_{dest.stem}",
            )
    except Stopped:
        raise
    except Exception as e:  # noqa: BLE001
        log.warning("Hero frames failed for %s, using camera move: %s", shot.id, e)
        return None
    if start.exists() and mid.exists() and end.exists():
        return mid, end
    return None


def _hero_clip(
    start: Path,
    mid: Path,
    end: Path,
    dest: Path,
    *,
    duration: float,
    fps: int,
    w: int,
    h: int,
) -> None:
    fade = 0.35
    hold = max((duration + 2 * fade) / 3, fade + 0.2)
    offset1 = hold - fade
    offset2 = offset1 + hold - fade
    scale = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={fps}"
    filt = (
        f"[0:v]{scale}[a];[1:v]{scale}[b];[2:v]{scale}[c];"
        f"[a][b]xfade=transition=fade:duration={fade}:offset={offset1:.3f}[ab];"
        f"[ab][c]xfade=transition=fade:duration={fade}:offset={offset2:.3f}[v]"
    )
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-t",
            f"{hold:.3f}",
            "-i",
            str(start),
            "-loop",
            "1",
            "-t",
            f"{hold:.3f}",
            "-i",
            str(mid),
            "-loop",
            "1",
            "-t",
            f"{hold:.3f}",
            "-i",
            str(end),
            "-filter_complex",
            filt,
            "-map",
            "[v]",
            "-frames:v",
            str(max(int(round(duration * fps)), fps)),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(dest),
        ]
    )


def _fit_clip(src: Path, dest: Path, shot: Shot, settings: Settings) -> None:
    fps = settings.pipeline.camera_fps
    w = settings.pipeline.output_width
    h = settings.pipeline.output_height
    frames = max(int(round(shot.duration_s * fps)), fps)
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-vf",
            (
                f"scale={w}:{h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{h},fps={fps},tpad=stop_mode=clone:stop=-1"
            ),
            "-frames:v",
            str(frames),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(dest),
        ]
    )


def _ken_burns(frame: Path, dest: Path, shot: Shot, settings: Settings) -> None:
    fps = settings.pipeline.camera_fps
    w = settings.pipeline.output_width
    h = settings.pipeline.output_height
    n_frames = max(int(round(shot.duration_s * fps)), fps)
    vf = _zoompan_filter(shot.camera, n_frames, w, h).replace("fps=24", f"fps={fps}")
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            str(frame),
            "-vf",
            vf,
            "-frames:v",
            str(n_frames),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(dest),
        ]
    )


def _hold_still(frame: Path, dest: Path, shot: Shot, settings: Settings) -> None:
    """A still picture for the whole spoken stretch, with no camera move."""
    fps = settings.pipeline.camera_fps
    w = settings.pipeline.output_width
    h = settings.pipeline.output_height
    n_frames = max(int(round(shot.duration_s * fps)), fps)
    vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},fps={fps},format=yuv420p"
    )
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            str(frame),
            "-vf",
            vf,
            "-frames:v",
            str(n_frames),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(dest),
        ]
    )


def run_motion(
    store: RunStore,
    settings: Settings,
    comfy: ComfyClient | None = None,
) -> Storyboard:
    board = store.read_json("storyboard.json", Storyboard)
    motion_on = store.status().meta.get("motion", True) is not False
    reference = _sheet(store, board)
    fps = settings.pipeline.camera_fps
    w = settings.pipeline.output_width
    h = settings.pipeline.output_height

    for shot in board.shots:
        if not shot.frame_path:
            raise RuntimeError(f"Missing frame for {shot.id}")
        dest = store.path("clips", f"{shot.id}.mp4")
        if dest.exists() and dest.stat().st_size > 1000:
            shot.clip_path = str(dest.relative_to(store.root))
            store.write_json("storyboard.json", board)
            continue
        frame = store.path(shot.frame_path)
        if not motion_on:
            log.info("Still hold %s", shot.id)
            _hold_still(frame, dest, shot, settings)
            shot.clip_path = str(dest.relative_to(store.root))
            store.write_json("storyboard.json", board)
            continue
        made_hero = False
        if shot.hero and shot.duration_s >= 1.8 and comfy is not None:
            raw = store.path("clips", f"{shot.id}_wan.mp4")
            try:
                log.info("Wan 14B hero %s", shot.id)
                comfy.generate_clip(
                    shot.visual_prompt,
                    frame,
                    raw,
                    duration_s=shot.duration_s,
                    filename_prefix=f"videogen_{store.run_id}_{shot.id}_wan",
                )
                _fit_clip(raw, dest, shot, settings)
                made_hero = True
            except Stopped:
                raise
            except Exception as e:  # noqa: BLE001
                log.warning("Wan hero failed for %s, trying illustrated dissolve: %s", shot.id, e)
                if dest.exists():
                    dest.unlink()
        if shot.hero and shot.duration_s >= 1.8 and not made_hero:
            pair = _hero_frames(store, comfy, shot, frame, reference)
            if pair is not None:
                log.info("Hero dissolve %s", shot.id)
                try:
                    _hero_clip(
                        frame,
                        pair[0],
                        pair[1],
                        dest,
                        duration=shot.duration_s,
                        fps=fps,
                        w=w,
                        h=h,
                    )
                    made_hero = True
                except Stopped:
                    raise
                except Exception as e:  # noqa: BLE001
                    log.warning("Hero dissolve failed for %s: %s", shot.id, e)
                    if dest.exists():
                        dest.unlink()
        if not made_hero:
            log.info("Camera move %s (%s)", shot.id, shot.camera)
            _ken_burns(frame, dest, shot, settings)
        shot.clip_path = str(dest.relative_to(store.root))
        store.write_json("storyboard.json", board)
    store.write_json("storyboard.json", board)
    return board
