from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from videogen.cancel import CommandResult, run_command
from videogen.config import Settings
from videogen.delivery import (
    format_chapters,
    loudnorm_filter,
    pack_chapters,
    parse_loudnorm,
    srt_timestamp,
    vtt_timestamp,
    wrap_caption,
)
from videogen.models import NarrationManifest, Script, ScriptBeat, Storyboard
from videogen.run_store import RunStore

log = logging.getLogger(__name__)


def _run(cmd: list[str]) -> CommandResult:
    proc = run_command(cmd)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2500:]
        raise RuntimeError(tail or "ffmpeg failed")
    return proc


def _write_captions(store: RunStore, script: Script, narration: NarrationManifest) -> Path:
    dur_map = {i.sentence_id: i.duration_s for i in narration.items}
    srt_path = store.path("captions.srt")
    vtt_path = store.path("captions.vtt")
    t = 0.0
    srt: list[str] = []
    vtt = ["WEBVTT", ""]
    idx = 1
    for sent in script.sentences:
        dur = dur_map.get(sent.id, 2.0)
        if not sent.text.strip():
            t += dur
            continue
        start, end = t, t + dur
        body = wrap_caption(sent.text)
        srt.extend([str(idx), f"{srt_timestamp(start)} --> {srt_timestamp(end)}", body, ""])
        vtt.extend([str(idx), f"{vtt_timestamp(start)} --> {vtt_timestamp(end)}", body, ""])
        t = end
        idx += 1
    srt_path.write_text("\n".join(srt), encoding="utf-8")
    vtt_path.write_text("\n".join(vtt), encoding="utf-8")
    return srt_path


def _write_chapters(store: RunStore, script: Script, narration: NarrationManifest) -> Path:
    dur_map = {i.sentence_id: i.duration_s for i in narration.items}
    segments: list[tuple[float, float, str]] = []
    t = 0.0
    beats: list[ScriptBeat] = script.beats
    by_beat: dict[int, list] = {}
    for sent in script.sentences:
        by_beat.setdefault(sent.beat_index, []).append(sent)
    for beat_index in sorted(by_beat):
        sentences = by_beat[beat_index]
        dur = sum(dur_map.get(s.id, 2.0) for s in sentences)
        role = sentences[0].role
        if beat_index < len(beats) and beats[beat_index].role:
            role = beats[beat_index].role
        if role == "hook":
            label = "Hook"
        elif role == "close":
            label = "Close"
        else:
            label = " ".join(sentences[0].text.split()[:6])
        segments.append((t, t + dur, label))
        t += dur
    text = format_chapters(pack_chapters(segments))
    path = store.path("chapters.txt")
    path.write_text(text, encoding="utf-8")
    return path


def _fade_wav(src: Path, dest: Path) -> None:
    # 15 ms edges so concatenated sentences do not click.
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-af",
            "afade=t=in:st=0:d=0.015,areverse,afade=t=in:st=0:d=0.015,areverse",
            str(dest),
        ]
    )


def _concat_voice(store: RunStore, narration: NarrationManifest, dest: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="videogen_voice_") as tmp:
        tmp_path = Path(tmp)
        listing = tmp_path / "audio.txt"
        with listing.open("w") as handle:
            for i, item in enumerate(narration.items):
                src = store.path(item.audio_path).resolve()
                faded = tmp_path / f"{i:03d}.wav"
                _fade_wav(src, faded)
                handle.write(f"file '{faded.resolve()}'\n")
        _run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(listing),
                "-c",
                "pcm_s16le",
                str(dest),
            ]
        )


def _synth_music(dest: Path, duration: float) -> None:
    """Original low pad. One lavfi source: this ffmpeg rejects extra sine inputs."""
    dur = max(duration, 1.0)
    # G major triad. aevalsrc is a single source, which lavfi -i accepts.
    expr = "0.33*sin(2*PI*196*t)+0.28*sin(2*PI*247*t)+0.22*sin(2*PI*294*t)"
    filt = (
        f"aevalsrc={expr}:duration={dur:.3f}:sample_rate=48000,"
        "lowpass=f=700,tremolo=f=0.12:d=0.4,volume=-20dB,"
        "aformat=channel_layouts=stereo"
    )
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            filt,
            "-c:a",
            "pcm_s16le",
            str(dest),
        ]
    )


def _mix(voice: Path, music: Path, dest: Path, duration: float) -> None:
    dur = max(duration, 1.0)
    fade_out = max(dur - 1.5, 0.2)
    filt = (
        "[0:a]aformat=sample_rates=48000,pan=stereo|c0=c0|c1=c0,highpass=f=80,asplit=2[sc][dry];"
        f"[1:a]aformat=sample_rates=48000:channel_layouts=stereo,"
        f"afade=t=in:d=1.2,afade=t=out:st={fade_out:.3f}:d=1.2,volume=0.45[bed];"
        f"anoisesrc=color=brown:duration={dur:.3f}:sample_rate=48000:amplitude=0.02,"
        "aformat=sample_rates=48000,pan=stereo|c0=c0|c1=c0,volume=-28dB[room];"
        "[bed][sc]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=350[ducked];"
        "[dry][ducked][room]amix=inputs=3:duration=first:normalize=0[mix]"
    )
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(voice),
            "-i",
            str(music),
            "-filter_complex",
            filt,
            "-map",
            "[mix]",
            "-c:a",
            "pcm_s16le",
            str(dest),
        ]
    )


def _thumbnail(store: RunStore, board: Storyboard, settings: Settings) -> None:
    frame = None
    for shot in board.shots:
        if shot.frame_path:
            frame = store.path(shot.frame_path)
            break
    if frame is None or not frame.exists():
        return
    dest = store.path("thumbnail.jpg")
    w = settings.pipeline.output_width
    h = settings.pipeline.output_height
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(frame),
            "-vf",
            f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}",
            "-frames:v",
            "1",
            str(dest),
        ]
    )


def run_assemble(store: RunStore, settings: Settings) -> Path:
    board = store.read_json("storyboard.json", Storyboard)
    script = store.read_json("script.json", Script)
    narration = store.read_json("narration.json", NarrationManifest)
    srt = _write_captions(store, script, narration)
    _write_chapters(store, script, narration)
    pipe = settings.pipeline
    stems = store.path("stems")
    stems.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="videogen_asm_") as tmp:
        tmp_path = Path(tmp)
        concat_list = tmp_path / "clips.txt"
        with concat_list.open("w") as handle:
            for shot in board.shots:
                if not shot.clip_path:
                    raise RuntimeError(f"Missing clip for {shot.id}")
                clip = store.path(shot.clip_path).resolve()
                handle.write(f"file '{clip}'\n")
        picture = stems / "picture.mp4"
        _run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-c",
                "copy",
                str(picture),
            ]
        )

        voice = stems / "voice.wav"
        _concat_voice(store, narration, voice)
        probe = _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(voice),
            ]
        )
        duration = float(probe.stdout.strip() or narration.total_duration_s or 1.0)
        music = stems / "music.wav"
        _synth_music(music, duration)
        mix = tmp_path / "mix.wav"
        _mix(voice, music, mix, duration)

        measure = run_command(
            [
                "ffmpeg",
                "-i",
                str(mix),
                "-af",
                f"loudnorm=I={pipe.loudness_i}:TP={pipe.loudness_tp}:LRA={pipe.loudness_lra}:print_format=json",
                "-f",
                "null",
                "-",
            ]
        )
        measured = parse_loudnorm(measure.stderr or "")
        audio_filter = loudnorm_filter(
            measured,
            integrated=pipe.loudness_i,
            true_peak=pipe.loudness_tp,
            lra=pipe.loudness_lra,
        )

        # Publish final.mp4 only after the encode exits. ffmpeg creates the
        # output path immediately, and the UI treats that path as "video ready".
        partial = store.path("final.encoding.mp4")
        final = store.path("final.mp4")
        gop = max(pipe.camera_fps // 2, 1)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(picture),
            "-i",
            str(mix),
        ]
        if pipe.burn_captions:
            srt_escaped = (
                str(srt.resolve()).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
            )
            cmd.extend(["-vf", f"subtitles='{srt_escaped}':force_style='FontSize=18,Outline=1'"])
        cmd.extend(
            [
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-profile:v",
                "high",
                "-pix_fmt",
                "yuv420p",
                "-b:v",
                pipe.video_bitrate,
                "-maxrate",
                pipe.video_bitrate,
                "-bufsize",
                "32M",
                "-g",
                str(gop),
                "-keyint_min",
                str(gop),
                "-bf",
                "2",
                "-x264-params",
                "open-gop=0:scenecut=0",
                "-c:a",
                "aac",
                "-b:a",
                pipe.audio_bitrate,
                "-ar",
                "48000",
                "-ac",
                "2",
                "-af",
                audio_filter,
                "-movflags",
                "+faststart",
                "-shortest",
                str(partial),
            ]
        )
        _run(cmd)
    partial.replace(final)
    _thumbnail(store, board, settings)
    log.info("Wrote %s", final)
    return final
