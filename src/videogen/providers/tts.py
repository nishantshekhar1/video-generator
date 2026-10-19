from __future__ import annotations

import logging
import shutil
import subprocess
import wave
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from videogen.config import ROOT, Settings

log = logging.getLogger(__name__)


class TTSBackend(ABC):
    @abstractmethod
    def synthesize(self, text: str, dest: Path) -> float:
        """Write WAV to dest; return duration seconds."""


class KokoroOnnxTTS(TTSBackend):
    def __init__(self, settings: Settings):
        self.settings = settings
        model = ROOT / "models" / "kokoro" / "kokoro-v1.0.onnx"
        voices = ROOT / "models" / "kokoro" / "voices-v1.0.bin"
        if not model.exists() or not voices.exists():
            raise FileNotFoundError(
                f"Kokoro ONNX assets missing under {model.parent}"
            )
        from kokoro_onnx import Kokoro  # type: ignore

        self._kokoro = Kokoro(str(model), str(voices))
        self._voice = settings.tts.voice
        # Map common kokoro voices; fall back if unknown
        self._sample_rate = settings.tts.sample_rate

    def synthesize(self, text: str, dest: Path) -> float:
        dest.parent.mkdir(parents=True, exist_ok=True)
        samples, sr = self._kokoro.create(
            text, voice=self._voice, speed=1.0, lang="en-us"
        )
        audio = np.asarray(samples, dtype=np.float32)
        import soundfile as sf

        sf.write(str(dest), audio, int(sr))
        return float(len(audio) / float(sr))


class KokoroTorchTTS(TTSBackend):
    def __init__(self, settings: Settings):
        from kokoro import KPipeline  # type: ignore

        self.settings = settings
        self._pipeline = KPipeline(lang_code=settings.tts.lang_code)

    def synthesize(self, text: str, dest: Path) -> float:
        dest.parent.mkdir(parents=True, exist_ok=True)
        chunks: list[np.ndarray] = []
        for _, _, audio in self._pipeline(
            text, voice=self.settings.tts.voice, speed=1.0
        ):
            chunks.append(np.asarray(audio, dtype=np.float32))
        if not chunks:
            raise RuntimeError("Kokoro produced no audio")
        audio = np.concatenate(chunks)
        import soundfile as sf

        sf.write(str(dest), audio, self.settings.tts.sample_rate)
        return float(len(audio) / self.settings.tts.sample_rate)


class EspeakTTS(TTSBackend):
    """CPU fallback using espeak-ng/espeak + ffmpeg to WAV."""

    def __init__(self, settings: Settings):
        self.settings = settings
        if not shutil.which("espeak-ng") and not shutil.which("espeak"):
            raise RuntimeError("espeak-ng/espeak not found")

    def synthesize(self, text: str, dest: Path) -> float:
        dest.parent.mkdir(parents=True, exist_ok=True)
        bin_name = "espeak-ng" if shutil.which("espeak-ng") else "espeak"
        wav_tmp = dest.with_suffix(".espeak.wav")
        subprocess.run(
            [bin_name, "-w", str(wav_tmp), text],
            check=True,
            capture_output=True,
        )
        if shutil.which("ffmpeg"):
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(wav_tmp),
                    "-ar",
                    str(self.settings.tts.sample_rate),
                    "-ac",
                    "1",
                    str(dest),
                ],
                check=True,
                capture_output=True,
            )
            wav_tmp.unlink(missing_ok=True)
        else:
            wav_tmp.rename(dest)
        return _wav_duration(dest)


class RoughTTS(TTSBackend):
    """Last-resort local placeholder: soft tone bursts timed to word count."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def synthesize(self, text: str, dest: Path) -> float:
        dest.parent.mkdir(parents=True, exist_ok=True)
        words = max(1, len(text.split()))
        duration = max(0.9, words / 2.6)
        sr = self.settings.tts.sample_rate
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Gentle amplitude-modulated tone so assembly still has audible speech bed
        env = 0.15 * (0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t))
        audio = (env * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
        import soundfile as sf

        sf.write(str(dest), audio, sr)
        log.warning("RoughTTS used (install kokoro-onnx models for real narration)")
        return float(duration)


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


class TTSProvider:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._backend = self._build()

    def _build(self) -> TTSBackend:
        preferred = self.settings.tts.backend.lower()
        errors: list[str] = []
        order: list[str]
        if preferred == "espeak":
            order = ["espeak", "kokoro_onnx", "kokoro", "rough"]
        elif preferred == "rough":
            order = ["rough"]
        else:
            order = ["kokoro_onnx", "kokoro", "espeak", "rough"]

        builders = {
            "kokoro_onnx": lambda: KokoroOnnxTTS(self.settings),
            "kokoro": lambda: KokoroTorchTTS(self.settings),
            "espeak": lambda: EspeakTTS(self.settings),
            "rough": lambda: RoughTTS(self.settings),
        }
        for name in order:
            try:
                backend = builders[name]()
                log.info("TTS backend: %s", name)
                return backend
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name}: {e}")
        raise RuntimeError("No TTS backend available: " + "; ".join(errors))

    def synthesize(self, text: str, dest: Path) -> float:
        text = " ".join(text.split())
        if not text:
            raise ValueError("empty text for TTS")
        return self._backend.synthesize(text, dest)
