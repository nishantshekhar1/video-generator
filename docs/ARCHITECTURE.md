# Local evocative video studio — architecture

Living design note. Update this when stages, providers, or GPU tradeoffs change.

## Goal

Produce a narrated, illustrated essay video from a topic (user-provided or locally scouted).
Language and vision stay on Ollama. Images go through the existing ComfyUI install (Flux).
Motion is ffmpeg camera moves on stills, with Wan 2.2 14B image-to-video on the hero shots.

## Hardware split

| Role | Device | Process |
|------|--------|---------|
| LLM / VLM | GPU 1 | Ollama (`CUDA_VISIBLE_DEVICES=1`) |
| Diffusion | GPU 0 | ComfyUI (`CUDA_VISIBLE_DEVICES=0`) |
| TTS | CPU | Kokoro (fallback: espeak) |
| Assemble | CPU | ffmpeg |

Never load `qwen3.6:35b` and Flux on the same card at once. Stages are sequential.

## Pipeline stages

```
topic → research → script → narration → storyboard → bible → stills → qa → motion → assemble
```

The character sheet is a full-body front-and-side proportion plate. Every shot copies one identity sentence and one style phrase verbatim, and only the action and place change. That is the consistency method that works on the illustrated models we sample: a fixed seed does not keep a face once the scene changes, and chaining each frame from the previous one drifts. Flux Redux and PuLID can pin a reference photo, but those Flux Dev weights are non-commercial, so a published run does not call them. A wide frame that comes back as two copies of the same page is folded down to one page; a front-and-side character sheet is left as two views.

On-image words are not painted by Flux. When a shot needs a sign, label, or title, `overlay_text` is typeset with DejaVu Sans after the still. The review screen shows the spoken line under each frame and can rewrite that line, the overlay, and the image prompt, then regenerate the still and the sentence audio.

The web UI can pause after draft stills (`await_approval`). The stills-and-voiceover section can be collapsed. Motion and still count are chosen on the start form and can be changed on that section; a new still count replans the board, and a motion change waits for a rebuild. Drafts are Z-Image-Turbo. The character sheet and style plate are Qwen-Image-2512. Approving resumes finals on Qwen-Image-2512. The Edit-2511 file is a diffusers fp8 pack that this ComfyUI samples as static, so it is not called until its keys use the `model.diffusion_model` prefix. The sheet stays the body reference for review. Flux Dev and Redux stay on disk and are not called. Hero shots are Wan 2.2 I2V 14B fp8: the high-noise expert stays on GPU 0 and the low-noise expert on GPU 1. The two experts still run one after the other, so the second card removes the reload, and it does not turn the pair into a 48 GB model. If Wan leaves the drawing style or runs out of memory, the shot falls back to a three-frame dissolve and then to Ken Burns. Assemble writes a 1920×1080 H.264 master, AAC at 384 kbps, two-pass loudness to −14 LUFS, plus `captions.srt`, `captions.vtt`, `chapters.txt`, `thumbnail.jpg`, and stems (`voice.wav`, `music.wav`, `picture.mp4`). PuLID is installed but unused: these presets are illustrated, and PuLID needs a photographic face.

| Stage | Module | Inputs | Outputs |
|-------|--------|--------|---------|
| topic | `stages/topic.py` | user text or interests | `brief.json` |
| research | `stages/research.py` | topic | `research.md`, sources in brief |
| script | `stages/script.py` | brief + research | `script.json` (+ per-sentence `visual_hint`) |
| narration | `stages/narration.py` | script sentences | `audio/*.wav`, durations |
| storyboard | `stages/storyboard.py` | script + durations + still count | `storyboard.json` (one held picture per group) |
| bible | `stages/bible.py` | storyboard | character sheet, style plate, one identity sentence copied onto every shot |
| stills | `stages/stills.py` | storyboard + sheet | draft frames, then final frames |
| qa | `stages/qa.py` | frames + prompts | maybe regenerated frames |
| motion | `stages/motion.py` | frames + camera | `clips/*.mp4` |
| assemble | `stages/assemble.py` | clips + audio + captions | `final.mp4` |

Each run lives under `runs/{id}/` with `status.json` for resumability.

### Image ↔ voiceover alignment

Voiceover and stills used to diverge because `visual_hint` was a truncated copy of the narration
(e.g. “It's a superpower of touch”) and Flux treated that abstract line as the scene.

**Fix:** after beats are split into sentences, `script._plan_visual_hints()` asks the structured
writer for a concrete scene per sentence id (subject, action, setting). Storyboard prompts then
lead with style + that scene, and treat narration only as sync context (“illustrate this spoken
line; do not write the words”).

### Default look: toddler color-pencil cartoon (mature audio)

Default `style_preset` is `sketch`. Shared `style_prefix` holds constraints (no photoreal /
watermarks / UI text). Presets supply the look:

- `sketch` — funny hand-drawn colored-pencil cartoons for toddlers (wobbly outlines,
  soft crayon fills, cute exaggeration). **Only the stills** use this look;
  script/TTS stay mature adult educational voiceover.
- `classic_cartoon` — vintage theatrical / cel animation (rubber-hose charm, ink
  outlines, flat colors, squash-and-stretch). Same mature narration split.
- `storybook` — storybook illustration plates (gouache and watercolor on warm
  paper, delicate ink line, painted scenery). Same mature narration split.
- `picture_book` — children's picture-book pages, one painted scene per still.
  Same mature narration split.
- `retro_digital` — 8-bit/16-bit pixel-art game assets + synthwave neon (magenta/cyan
  glow, limited palette, arcade key art). Same mature narration split.
- `painterly` / `graphic` — optional alternate looks from UI/CLI

Visual planner (`script._STYLE_PLANNER`) and storyboard sync lines are style-aware.

## Important classes and functions

- `videogen.config.Settings` — loads `config/default.yaml` + env overrides.
- `videogen.providers.llm.OllamaLLM` — `chat()`, `json_chat()`; only LLM entry point.
- `videogen.providers.vision.OllamaVision` — `review_frame()`; only VLM entry point.
- `videogen.providers.comfy.ComfyClient` — queue Flux workflow, wait, copy image.
- `videogen.providers.tts.TTSProvider` — Kokoro or espeak; sentence-level WAV.
- `videogen.providers.research.WebResearcher` — DuckDuckGo + trafilatura extract.
- `videogen.pipeline.Pipeline` — ordered stages, health checks, resume from status.
- `videogen.cli` / `videogen.webapp` — CLI and FastAPI API over the same pipeline. The screen is a React app in `web/` (Vite build served from `web/dist`). It follows a simple editor layout: projects on the left, the picture in the center, the selected still’s text and prompt on the right, and a filmstrip along the bottom. A rail lists every stage and its state. Stills, picture check, and motion also show how many items in that step are finished. A recent run can be deleted from the list; that removes only its `runs/{id}/` folder, and a run that is still working is left in place. Stop ends the run in the current stage: ComfyUI is interrupted, and an ffmpeg encode is killed. Resume continues from the last finished stage.
  `videogen serve` binds `0.0.0.0:7860` by default so other devices on the LAN can open the UI.
  Ollama and ComfyUI stay on localhost; only the studio UI is LAN-exposed.

## Trade-offs

1. **Illustrated stills + camera > full video diffusion** for 3–5 min pieces on 24 GB cards.
   Style stays coherent; GPU time stays tens of minutes, not hours.
2. **One picture stays up with the lines it covers** and shows the objects those lines name. Standard holds are 5–12 seconds. Fewer stills holds 10–20 seconds. More stills holds 3–6 seconds. Motion can be turned off, and then each picture is a static hold with no Wan clip and no camera move.
3. **Extra LLM pass for visuals** (after script) costs one JSON call but stops Flux from
   illustrating abstract narration phrases; falls back to a topic-anchored heuristic on failure.
4. **Local-only LLM/VLM** means quality tracks Ollama models; no cloud fallback by design.
5. **Published stills are Apache-2.0.** Z-Image-Turbo draws the drafts. Qwen-Image-2512 draws the sheet and the finals. The Edit-2511 fp8mixed file is not sampled until it is packed for Comfy, because the diffusers pack decodes as static. Flux Schnell remains the fallback. Flux Dev is non-commercial and is not in the path.
6. **Wan 2.2 I2V 14B fp8 is the hero model.** Each expert is about 14 GB and fits one 4090; only one is sampled at a time. GPU 0 holds the high-noise expert and GPU 1 the low-noise expert, after Ollama is unloaded for that step. The installed Wan 2.1 14B fp16 file is 30 GB and is not used. Ken Burns remains the fallback when the clip leaves the illustration.

## Model review (2026-10-17)

Two RTX 4090s (24 GB). ComfyUI sees both. GPU 0 renders stills and the Wan high-noise expert. GPU 1 holds Ollama, and holds the Wan low-noise expert only while a hero clip is sampling. Kokoro-82M remains the narration default. Gemma 4 stays the JSON writer; Qwen3-VL 8B stays the frame reviewer. On-image words are typeset, not painted.

## Deferred

- 35B final-script pass as a UI button
- Rich topic scout beyond interest list + search
- Indic Parler TTS behind the same TTS interface

## Prove log

**2026-10-09 — first end-to-end short run**

- Run: `runs/20261009-012728-3bbb2f/`
- Topic: Octopus Arm Taste Superpower (user topic + local research/script)
- Writer for structured JSON: `gemma4:latest` (qwen3.5 returned empty `content` under `format=json`)
- Prose synthesis: `qwen3.5:latest` with `num_ctx=8192`
- TTS: Kokoro ONNX (`af_heart`) on CPU
- Stills: Flux Schnell via ComfyUI — 7 frames
- Motion: ffmpeg Ken Burns camera moves
- QA: skipped for this prove (`--skip-qa`)
- Output: `final.mp4` ~21.6s, ~2.6 MB, 7 sentences / 59 words, narration ~21.7s
- Lessons: cap Ollama `num_ctx`; prefer a JSON-reliable local model for script stage; resume works via `status.json`
