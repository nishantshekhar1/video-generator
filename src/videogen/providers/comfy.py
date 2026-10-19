from __future__ import annotations

import copy
import json
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from videogen.cancel import Stopped, raise_if_stopped
from videogen.config import Settings

log = logging.getLogger(__name__)


class ComfyClient:
    """Submit API-format workflows to local ComfyUI and retrieve images."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.comfyui.base_url.rstrip("/")
        self.timeout = settings.comfyui.timeout_s
        self.client_id = str(uuid.uuid4())

    def health(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/system_stats", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def load_workflow_template(self) -> dict[str, Any]:
        path = self.settings.workflow_path()
        with path.open() as f:
            return json.load(f)

    def build_flux_workflow(
        self,
        prompt: str,
        *,
        seed: int | None = None,
        filename_prefix: str = "videogen",
        quality: str = "draft",
        width: int | None = None,
        height: int | None = None,
        reference_name: str | None = None,
    ) -> dict[str, Any]:
        cfg = self.settings.comfyui
        wf = copy.deepcopy(self.load_workflow_template())
        seed = int(seed if seed is not None else uuid.uuid4().int % (2**32))
        final = quality == "final"
        w = width if width is not None else (cfg.width if final else cfg.draft_width)
        h = height if height is not None else (cfg.height if final else cfg.draft_height)
        unet = cfg.final_unet_name if final else cfg.unet_name
        steps = cfg.final_steps if final else cfg.steps
        guidance = cfg.final_guidance if final else cfg.guidance

        wf["6"]["inputs"]["text"] = prompt
        wf["9"]["inputs"]["filename_prefix"] = filename_prefix
        wf["10"]["inputs"]["vae_name"] = cfg.vae_name
        wf["11"]["inputs"]["clip_name1"] = cfg.clip_name1
        wf["11"]["inputs"]["clip_name2"] = cfg.clip_name2
        wf["12"]["inputs"]["unet_name"] = unet
        wf["12"]["inputs"]["weight_dtype"] = cfg.unet_weight_dtype
        model_src: list[Any] = ["12", 0]
        if final and cfg.final_lora_name:
            wf["40"] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": ["12", 0],
                    "lora_name": cfg.final_lora_name,
                    "strength_model": 1.0,
                },
            }
            model_src = ["40", 0]
        cond_src: list[Any] = ["6", 0]
        if reference_name:
            wf["32"] = {
                "class_type": "StyleModelLoader",
                "inputs": {"style_model_name": cfg.redux_model},
            }
            wf["34"] = {
                "class_type": "CLIPVisionLoader",
                "inputs": {"clip_name": cfg.clip_vision_name},
            }
            wf["35"] = {
                "class_type": "LoadImage",
                "inputs": {"image": reference_name},
            }
            wf["33"] = {
                "class_type": "CLIPVisionEncode",
                "inputs": {
                    "clip_vision": ["34", 0],
                    "image": ["35", 0],
                    "crop": "none",
                },
            }
            wf["31"] = {
                "class_type": "StyleModelApply",
                "inputs": {
                    "conditioning": ["6", 0],
                    "style_model": ["32", 0],
                    "clip_vision_output": ["33", 0],
                    "strength": cfg.redux_strength,
                    "strength_type": "multiply",
                },
            }
            cond_src = ["31", 0]
        wf["17"]["inputs"]["steps"] = steps
        wf["25"]["inputs"]["noise_seed"] = seed
        wf["26"]["inputs"]["guidance"] = guidance
        wf["26"]["inputs"]["conditioning"] = cond_src
        wf["27"]["inputs"]["width"] = w
        wf["27"]["inputs"]["height"] = h
        wf["30"]["inputs"]["model"] = model_src
        wf["30"]["inputs"]["width"] = w
        wf["30"]["inputs"]["height"] = h
        return wf

    def upload_image(self, path: Path) -> str:
        """Copy a reference into ComfyUI's input folder. Returns the filename."""
        name = f"videogen_{path.stem}_{uuid.uuid4().hex[:6]}{path.suffix.lower() or '.png'}"
        with path.open("rb") as handle:
            files = {"image": (name, handle, "image/png")}
            data = {"overwrite": "true"}
            with httpx.Client(timeout=120.0) as client:
                r = client.post(f"{self.base_url}/upload/image", data=data, files=files)
                r.raise_for_status()
                body = r.json()
        uploaded = body.get("name") or name
        sub = body.get("subfolder") or ""
        return f"{sub}/{uploaded}" if sub else uploaded

    def _weights_path(self, folder: str, name: str) -> Path:
        return Path(self.settings.paths.comfy_output_dir).parent / "models" / folder / name

    def _weights_ready(self, folder: str, name: str) -> bool:
        path = self._weights_path(folder, name)
        return path.is_file() and path.stat().st_size > 1_000_000

    def _comfy_packed_unet(self, name: str) -> bool:
        """Comfy can sample a unet only when the keys use its diffusion_model prefix.

        The Edit-2511 fp8mixed file is a diffusers pack. Loading it samples static.
        """
        path = self._weights_path("diffusion_models", name)
        if not path.is_file():
            return False
        try:
            from safetensors import safe_open

            with safe_open(path, framework="pt") as handle:
                return any(key.startswith("model.diffusion_model.") for key in handle.keys())
        except Exception as e:  # noqa: BLE001
            log.warning("Could not inspect %s: %s", name, e)
            return False

    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        payload = {"prompt": workflow, "client_id": self.client_id}
        with httpx.Client(timeout=60.0) as client:
            r = client.post(f"{self.base_url}/prompt", json=payload)
            if r.status_code >= 400:
                raise RuntimeError(f"ComfyUI rejected the workflow ({r.status_code}): {r.text[:1500]}")
            data = r.json()
        if "error" in data:
            raise RuntimeError(f"ComfyUI prompt error: {data}")
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI did not return prompt_id: {data}")
        return prompt_id

    def wait_for_images(self, prompt_id: str) -> list[dict[str, Any]]:
        deadline = time.time() + self.timeout
        with httpx.Client(timeout=30.0) as client:
            while time.time() < deadline:
                raise_if_stopped()
                r = client.get(f"{self.base_url}/history/{prompt_id}")
                r.raise_for_status()
                hist = r.json()
                if prompt_id in hist:
                    entry = hist[prompt_id]
                    status = entry.get("status", {})
                    if status.get("status_str") == "error" or status.get("completed") is False:
                        msgs = status.get("messages") or entry.get("status", {})
                        if status.get("status_str") == "error":
                            raise RuntimeError(f"ComfyUI job failed: {msgs}")
                    outputs = entry.get("outputs") or {}
                    images: list[dict[str, Any]] = []
                    for node_out in outputs.values():
                        for img in node_out.get("images", []):
                            images.append(img)
                    if images:
                        return images
                    # completed with no images is still a failure
                    if status.get("completed"):
                        raise RuntimeError(f"ComfyUI completed with no images: {entry}")
                time.sleep(0.5)
        raise TimeoutError(f"ComfyUI timed out waiting for prompt {prompt_id}")

    def interrupt(self) -> None:
        """Ask ComfyUI to drop the picture it is drawing now."""
        with httpx.Client(timeout=10.0) as client:
            client.post(f"{self.base_url}/interrupt")
            client.post(f"{self.base_url}/queue", json={"clear": True})

    def download_image(self, image_meta: dict[str, Any], dest: Path) -> Path:
        params = {
            "filename": image_meta["filename"],
            "subfolder": image_meta.get("subfolder", ""),
            "type": image_meta.get("type", "output"),
        }
        dest.parent.mkdir(parents=True, exist_ok=True)
        with httpx.Client(timeout=120.0) as client:
            r = client.get(f"{self.base_url}/view", params=params)
            r.raise_for_status()
            dest.write_bytes(r.content)
        return dest

    def _seed(self, seed: int | None) -> int:
        return int(seed if seed is not None else uuid.uuid4().int % (2**32))

    def _with_lora(self, wf: dict[str, Any], node: str, model_src: list[Any], lora_name: str) -> list[Any]:
        if not lora_name:
            return model_src
        wf[node] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": model_src, "lora_name": lora_name, "strength_model": 1.0},
        }
        return [node, 0]

    def build_zimage_workflow(
        self,
        prompt: str,
        *,
        seed: int,
        filename_prefix: str,
        width: int,
        height: int,
    ) -> dict[str, Any]:
        cfg = self.settings.comfyui
        return {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": cfg.zimage_unet, "weight_dtype": "default"}},
            "2": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": cfg.zimage_shift}},
            "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": cfg.zimage_clip, "type": "lumina2"}},
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": cfg.zimage_vae}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 0]}},
            "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
            "7": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
            "8": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["2", 0],
                    "seed": seed,
                    "steps": cfg.zimage_steps,
                    "cfg": 1.0,
                    "sampler_name": "res_multistep",
                    "scheduler": "simple",
                    "positive": ["5", 0],
                    "negative": ["6", 0],
                    "latent_image": ["7", 0],
                    "denoise": 1.0,
                },
            },
            "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["4", 0]}},
            "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": filename_prefix}},
        }

    def build_qwen_workflow(
        self,
        prompt: str,
        *,
        seed: int,
        filename_prefix: str,
        width: int,
        height: int,
        reference_name: str | None,
    ) -> dict[str, Any]:
        cfg = self.settings.comfyui
        edit = reference_name is not None
        unet = cfg.qwen_edit_unet if edit else cfg.qwen_unet
        lora_name = cfg.qwen_edit_lora if edit else cfg.qwen_lora
        lora = lora_name if self._weights_ready("loras", lora_name) else ""
        text = prompt
        if edit:
            text = (
                "Match the reference character's body width exactly. "
                "Do not make the torso, hips, or limbs thinner or heavier than the reference. "
                "Keep the same face, hair, and outfit. Change only the pose and the setting. "
                + prompt
            )
        wf: dict[str, Any] = {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
            "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": cfg.qwen_clip, "type": "qwen_image"}},
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": cfg.qwen_vae}},
            "7": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
            "11": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["4", 0]}},
            "12": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": filename_prefix}},
        }
        model_src = self._with_lora(wf, "2", ["1", 0], lora)
        wf["13"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": model_src, "shift": cfg.qwen_shift}}
        if edit:
            wf["14"] = {"class_type": "LoadImage", "inputs": {"image": reference_name}}
            wf["5"] = {
                "class_type": "TextEncodeQwenImageEdit",
                "inputs": {"clip": ["3", 0], "prompt": text, "vae": ["4", 0], "image": ["14", 0]},
            }
        else:
            wf["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": ["3", 0]}}
        wf["6"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}}
        wf["8"] = {
            "class_type": "KSampler",
            "inputs": {
                "model": ["13", 0],
                "seed": seed,
                "steps": cfg.qwen_steps,
                "cfg": cfg.qwen_cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["7", 0],
                "denoise": 1.0,
            },
        }
        return wf

    def build_wan_workflow(
        self,
        prompt: str,
        *,
        seed: int,
        filename_prefix: str,
        image_name: str,
        width: int,
        height: int,
        length: int,
        fps: int,
    ) -> dict[str, Any]:
        cfg = self.settings.comfyui
        split = max(cfg.wan_steps // 2, 1)
        positive = (
            prompt
            + " Gentle illustrated motion. Keep the drawing style, line work, palette, "
            "face, and outfit. No photorealism."
        )
        negative = "photorealistic, live action, blurry, watermark, extra limbs, morphing face, text"
        return {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": cfg.wan_high_unet, "weight_dtype": "default"}},
            "2": {"class_type": "UNETLoader", "inputs": {"unet_name": cfg.wan_low_unet, "weight_dtype": "default"}},
            "3": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": cfg.wan_shift}},
            "4": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["2", 0], "shift": cfg.wan_shift}},
            "5": {"class_type": "PinModelDevice", "inputs": {"model": ["3", 0], "device": cfg.wan_high_device}},
            "6": {"class_type": "PinModelDevice", "inputs": {"model": ["4", 0], "device": cfg.wan_low_device}},
            "7": {"class_type": "CLIPLoader", "inputs": {"clip_name": cfg.wan_clip, "type": "wan"}},
            "8": {"class_type": "VAELoader", "inputs": {"vae_name": cfg.wan_vae}},
            "9": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["7", 0]}},
            "10": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["7", 0]}},
            "11": {"class_type": "LoadImage", "inputs": {"image": image_name}},
            "12": {
                "class_type": "WanImageToVideo",
                "inputs": {
                    "positive": ["9", 0],
                    "negative": ["10", 0],
                    "vae": ["8", 0],
                    "width": width,
                    "height": height,
                    "length": length,
                    "batch_size": 1,
                    "start_image": ["11", 0],
                },
            },
            "13": {
                "class_type": "KSamplerAdvanced",
                "inputs": {
                    "model": ["5", 0],
                    "add_noise": "enable",
                    "noise_seed": seed,
                    "steps": cfg.wan_steps,
                    "cfg": cfg.wan_cfg,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "positive": ["12", 0],
                    "negative": ["12", 1],
                    "latent_image": ["12", 2],
                    "start_at_step": 0,
                    "end_at_step": split,
                    "return_with_leftover_noise": "enable",
                },
            },
            "14": {
                "class_type": "KSamplerAdvanced",
                "inputs": {
                    "model": ["6", 0],
                    "add_noise": "disable",
                    "noise_seed": seed,
                    "steps": cfg.wan_steps,
                    "cfg": cfg.wan_cfg,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "positive": ["12", 0],
                    "negative": ["12", 1],
                    "latent_image": ["13", 0],
                    "start_at_step": split,
                    "end_at_step": cfg.wan_steps,
                    "return_with_leftover_noise": "disable",
                },
            },
            "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["8", 0]}},
            "16": {"class_type": "CreateVideo", "inputs": {"images": ["15", 0], "fps": fps}},
            "17": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["16", 0],
                    "filename_prefix": filename_prefix,
                    "format": "mp4",
                    "codec": "h264",
                },
            },
        }

    def _unload_ollama(self) -> None:
        base = self.settings.ollama.base_url.rstrip("/")
        try:
            with httpx.Client(timeout=10.0) as client:
                listed = client.get(f"{base}/api/ps")
                listed.raise_for_status()
                for item in listed.json().get("models") or []:
                    name = item.get("name") or item.get("model")
                    if name:
                        client.post(f"{base}/api/generate", json={"model": name, "keep_alive": 0})
        except httpx.HTTPError as e:
            log.warning("Could not free Ollama before hero motion: %s", e)

    def generate_still(
        self,
        prompt: str,
        dest: Path,
        *,
        seed: int | None = None,
        filename_prefix: str = "videogen",
        quality: str = "draft",
        width: int | None = None,
        height: int | None = None,
        reference: Path | None = None,
    ) -> Path:
        cfg = self.settings.comfyui
        final = quality == "final"
        w = width if width is not None else (cfg.width if final else cfg.draft_width)
        h = height if height is not None else (cfg.height if final else cfg.draft_height)
        reference_name = None
        if final and reference is not None and reference.exists():
            try:
                reference_name = self.upload_image(reference)
            except Exception as e:  # noqa: BLE001
                log.warning("Character sheet upload failed: %s", e)
        chosen = self._seed(seed)
        qwen_ready = (
            self._weights_ready("diffusion_models", cfg.qwen_unet)
            and self._weights_ready("text_encoders", cfg.qwen_clip)
            and self._weights_ready("vae", cfg.qwen_vae)
        )
        edit_ready = (
            qwen_ready
            and self._weights_ready("diffusion_models", cfg.qwen_edit_unet)
            and self._comfy_packed_unet(cfg.qwen_edit_unet)
        )
        if reference_name and not edit_ready:
            log.info("Character sheet stays a reference for review; finals use Qwen-Image-2512")
            reference_name = None
        zimage_ready = self._weights_ready("diffusion_models", cfg.zimage_unet) and self._weights_ready(
            "text_encoders", cfg.zimage_clip
        )
        use_qwen = qwen_ready and quality != "draft"
        use_zimage = quality == "draft" and zimage_ready

        def accept(path: Path) -> Path:
            from videogen.overlay import image_is_static

            if image_is_static(path):
                path.unlink(missing_ok=True)
                raise RuntimeError("image model returned static")
            return path

        def schnell(prefix: str) -> Path:
            log.warning("Using Flux Schnell for %s until the Apache weights are on disk", prefix)
            wf = self.build_flux_workflow(
                prompt, seed=chosen, filename_prefix=prefix, quality="draft", width=w, height=h
            )
            return accept(self._run_image(wf, dest))

        try:
            if use_zimage:
                wf = self.build_zimage_workflow(
                    prompt, seed=chosen, filename_prefix=filename_prefix, width=w, height=h
                )
            elif use_qwen and quality != "draft":
                wf = self.build_qwen_workflow(
                    prompt,
                    seed=chosen,
                    filename_prefix=filename_prefix,
                    width=w,
                    height=h,
                    reference_name=reference_name if edit_ready else None,
                )
            else:
                return schnell(filename_prefix)
            return accept(self._run_image(wf, dest))
        except Stopped:
            raise
        except Exception:
            if quality == "draft" or not (reference_name and qwen_ready):
                log.warning("Still failed for %s; falling back to Flux Schnell", filename_prefix)
                return schnell(filename_prefix + "_schnell")
            log.warning("Qwen edit failed for %s; retrying Qwen text-only", filename_prefix)
            wf = self.build_qwen_workflow(
                prompt,
                seed=chosen,
                filename_prefix=filename_prefix + "_text",
                width=w,
                height=h,
                reference_name=None,
            )
            return self._run_image(wf, dest)

    def generate_clip(
        self,
        prompt: str,
        image: Path,
        dest: Path,
        *,
        duration_s: float,
        filename_prefix: str = "videogen_hero",
        seed: int | None = None,
    ) -> Path:
        """Wan 2.2 14B image-to-video. High-noise expert on GPU 0, low-noise on GPU 1."""
        cfg = self.settings.comfyui
        self._unload_ollama()
        image_name = self.upload_image(image)
        fps = 16
        length = int(round(duration_s * fps))
        length = max(17, ((length - 1) // 4) * 4 + 1)
        length = min(length, 81)
        chosen = self._seed(seed)
        sizes = [(cfg.wan_width, cfg.wan_height), (832, 480)]
        last_error: Exception | None = None
        for width, height in sizes:
            wf = self.build_wan_workflow(
                prompt,
                seed=chosen,
                filename_prefix=filename_prefix,
                image_name=image_name,
                width=width,
                height=height,
                length=length,
                fps=fps,
            )
            try:
                return self._run_image(wf, dest)
            except Stopped:
                raise
            except Exception as e:  # noqa: BLE001
                last_error = e
                log.warning("Wan %sx%s failed: %s", width, height, e)
        raise RuntimeError(f"Wan hero clip failed: {last_error}")

    def _run_image(self, workflow: dict[str, Any], dest: Path) -> Path:
        prompt_id = self.queue_prompt(workflow)
        images = self.wait_for_images(prompt_id)
        return self.download_image(images[0], dest)

    def copy_from_comfy_output(self, filename: str, dest: Path) -> Path:
        src = Path(self.settings.paths.comfy_output_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return dest
