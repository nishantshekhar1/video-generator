from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"


class OllamaConfig(BaseModel):
    base_url: str = "http://127.0.0.1:11434"
    writer_model: str = "gemma4:latest"
    writer_model_structured: str = "gemma4:latest"
    writer_model_prose: str = "qwen3.5:latest"
    writer_model_quality: str = "qwen3.6:35b"
    vision_model: str = "qwen3-vl:8b"
    embed_model: str = "nomic-embed-text:latest"
    timeout_s: float = 600
    num_ctx: int = 8192


class ComfyConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8188"
    workflow: str = "workflows/flux_schnell_api.json"
    unet_name: str = "flux1-schnell.safetensors"
    unet_weight_dtype: str = "default"
    clip_name1: str = "t5xxl_fp8_e4m3fn.safetensors"
    clip_name2: str = "clip_l.safetensors"
    vae_name: str = "ae.safetensors"
    width: int = 768
    height: int = 768
    steps: int = 4
    guidance: float = 3.5
    timeout_s: float = 3600
    final_unet_name: str = "flux1-dev-fp8.safetensors"
    final_lora_name: str = "fluxdev1-turbo-8steps-lora.safetensors"
    final_steps: int = 8
    final_guidance: float = 3.5
    draft_width: int = 768
    draft_height: int = 432
    redux_model: str = "flux1-redux-dev.safetensors"
    clip_vision_name: str = "sigclip_vision_patch14_384.safetensors"
    redux_strength: float = 0.55
    # Published-channel stills. Drafts are Z-Image-Turbo. The character sheet
    # and text-only fallback are Qwen-Image-2512. Finals are Qwen-Image-Edit-2511.
    zimage_unet: str = "z_image_turbo_bf16.safetensors"
    zimage_clip: str = "qwen_3_4b.safetensors"
    zimage_vae: str = "ae.safetensors"
    zimage_steps: int = 8
    zimage_shift: float = 3.0
    qwen_unet: str = "qwen_image_2512_fp8_e4m3fn.safetensors"
    qwen_edit_unet: str = "qwen_image_edit_2511_fp8mixed.safetensors"
    qwen_clip: str = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
    qwen_vae: str = "qwen_image_vae.safetensors"
    qwen_lora: str = "Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors"
    qwen_edit_lora: str = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
    qwen_steps: int = 4
    qwen_cfg: float = 1.0
    qwen_shift: float = 3.1
    wan_high_unet: str = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
    wan_low_unet: str = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
    wan_clip: str = "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    wan_vae: str = "split_files/vae/wan_2.1_vae.safetensors"
    wan_steps: int = 20
    wan_cfg: float = 5.0
    wan_shift: float = 5.0
    wan_width: int = 1280
    wan_height: int = 720
    wan_high_device: int = 0
    wan_low_device: int = 1


class TTSConfig(BaseModel):
    backend: str = "kokoro"
    voice: str = "af_heart"
    sample_rate: int = 24000
    lang_code: str = "a"


class PipelineConfig(BaseModel):
    words_per_minute: int = 150
    default_duration_s: int = 60
    style_prefix: str = (
        "illustrated educational video still, no photorealism, no watermark, "
        "no UI chrome, no readable text overlays"
    )
    style_presets: dict[str, str] = Field(default_factory=dict)
    vision_retry_once: bool = True
    camera_fps: int = 24
    output_width: int = 1920
    output_height: int = 1080
    video_bitrate: str = "16M"
    audio_bitrate: str = "384k"
    loudness_i: float = -14.0
    loudness_tp: float = -1.5
    loudness_lra: float = 11.0
    burn_captions: bool = False
    hero_count: int = 2


class ResearchConfig(BaseModel):
    max_results: int = 6
    max_pages: int = 4
    max_chars_per_page: int = 4000


class PathsConfig(BaseModel):
    runs_dir: str = "runs"
    comfy_output_dir: str = "/home/nishant/Desktop/ComfyUI/output"


class GPUConfig(BaseModel):
    ollama_cuda_visible_devices: str = "1"
    comfy_cuda_visible_devices: str = "0"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VIDEOGEN_", extra="ignore")

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    comfyui: ComfyConfig = Field(default_factory=ComfyConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    interests: list[str] = Field(default_factory=list)
    use_quality_writer: bool = False

    def runs_path(self) -> Path:
        p = Path(self.paths.runs_dir)
        if not p.is_absolute():
            p = ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    def workflow_path(self) -> Path:
        p = Path(self.comfyui.workflow)
        if not p.is_absolute():
            p = ROOT / p
        return p


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml_settings(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or DEFAULT_CONFIG
    if not cfg_path.exists():
        return {}
    with cfg_path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {cfg_path}")
    return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    raw = load_yaml_settings()
    return Settings(**raw)


def reload_settings(path: Path | None = None) -> Settings:
    get_settings.cache_clear()
    raw = load_yaml_settings(path)
    settings = Settings(**raw)
    # Re-seed cache with this instance.
    get_settings.cache_clear()

    @lru_cache(maxsize=1)
    def _cached() -> Settings:
        return settings

    # Replace the cached function body by calling once after clearing is awkward;
    # simply return and let callers use the returned object. For CLI we pass settings.
    return settings
