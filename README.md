# Local evocative video studio

Researches a topic, writes a narrated script with a local LLM, generates illustrated frames via ComfyUI (Flux), speaks narration locally, and assembles a finished video with ffmpeg.

**Language and vision stay on Ollama.** No cloud LLM/VLM.

## Requirements

- Dual NVIDIA GPUs recommended (this machine: 2× RTX 4090)
- [Ollama](https://ollama.com) with `qwen3.5`, `qwen3-vl:8b`
- ComfyUI with Flux Schnell + VAE + CLIP weights
- `ffmpeg`
- Python 3.11+

Suggested GPU split (also documented in `docs/ARCHITECTURE.md`):

```bash
# Terminal A — Ollama on GPU 1
CUDA_VISIBLE_DEVICES=1 ollama serve

# Terminal B — ComfyUI on GPU 0
cd ~/Desktop/ComfyUI
CUDA_VISIBLE_DEVICES=0 .venv/bin/python main.py --listen 127.0.0.1 --port 8188
```

## Install

```bash
cd ~/Desktop/video-generator
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
# Optional better TTS:
pip install -e ".[tts]"
```

## CLI

```bash
videogen health
videogen run --topic "Why octopus arms can taste" --duration 45
videogen run --pick-topic --duration 60
videogen serve   # http://0.0.0.0:7860 — reachable on your LAN
# optional: videogen serve --host 127.0.0.1   # localhost only
```

Find this machine’s LAN IP with `hostname -I` and open `http://<ip>:7860` from other devices.

Runs land in `runs/{id}/` with resumable `status.json` and `final.mp4`.

## Design

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Production path for this studio

The stage order already matches how 2026 explainer pipelines are built: lock a brief, write for the ear, plan one visual job per shot, approve cheap frames, then render, mix, and export. What keeps current runs at a demo level is character drift, a thin mix, and a 720p master. Fix those in the order below. Full video diffusion stays deferred until a quantized model fits one 24 GB card.

Sources: YouTube upload encoding help, creator loudness practice (about −14 LUFS), and 2026 explainer / Flux continuity writeups. Read against this repo’s assemble, motion, and Flux Schnell path.

| | |
| --- | --- |
| **9** | Stages already in the right order |
| **720p** | Master today (1280×720) |
| **−16 LUFS** | Loudness target today |
| **Prompt** | Character lock today |

### Practices that actually move quality

Shops that ship illustrated explainers treat generation as the expensive middle. The decisions that survive YouTube compression are made before the first final still and after the last clip.

| Practice | What it protects | This app today |
| --- | --- | --- |
| Lock audience, payoff, length, 16:9, and one style before any final frame | Runtime and look stay inside the brief | Brief and style presets exist. No approval stop before stills. |
| One visual job per shot: subject, action, camera, duration | Each cut explains one idea | Sentence equals shot. Visual hints are concrete. Coverage is still one frame per line. |
| Character bible first: sheet plus 5–7 fixed anchors (hair, palette, outfit, prop) | The same person returns after a cut | Continuity is a sentence in the prompt. No reference image is passed to Flux. |
| Draft on a fast model, approve the board, then render finals | Bad plans are cheap to throw away | Flux Schnell, 4 steps, 768px, straight into the cut. Vision QA retries once. |
| Condition later shots on the approved sheet, and stop using a drifted frame as the next reference | Identity error does not compound | Each still is an independent text-to-image call. |
| Separate voice, music, and room tone. Master near −14 LUFS, true peak at or under −1 dBTP | YouTube playback matches the mix. Quiet masters stay quiet; hot masters get turned down. | Voice only. loudnorm is −16 LUFS, AAC 192 kbps. No music bed. |
| 1080p H.264 High, 4:2:0, progressive, closed GOP, AAC-LC 48 kHz at 384 kbps stereo | YouTube’s re-encode keeps edges and speech | 1280×720, default x264 rate, burned-in captions, no chapter list, no thumbnail. |
| Script-accurate SRT (about 40 characters, two lines) plus chapters: 00:00, at least three, each at least 10 seconds | Sound-off viewing and the chapter bar | `captions.srt` is written from the script, then burned in. Chapters are not emitted. |
| Three review passes: story, then identity and text, then audio with the picture hidden | Different failures get caught | One VLM check for subject, style, and garbled text. Often skipped. |

Encoding numbers are YouTube’s published upload settings for 1080p at 24–30 fps (8 Mbps is their floor recommendation; a 15–20 Mbps master gives the re-encode more detail). −14 LUFS is the creator target YouTube normalizes toward, from loudness practice, and is absent from the encoding help page itself.

### Build this, in this order

Each step raises what a viewer hears or sees on YouTube. Later steps depend on the earlier ones staying stable.

| Order | Change | YouTube result | Cost |
| --- | --- | --- | --- |
| 1. Master | Render 1920×1080. x264 High, yuv420p, about 16 Mbps, closed GOP of half the frame rate. AAC-LC 48 kHz, 384 kbps. Two-pass loudnorm to −14 LUFS, true peak −1.5 dBTP. Write a sidecar SRT and a chapters file from hook, beats, and close. Pick the hook frame as `thumbnail.jpg`. | A file YouTube can ingest without looking soft, quiet, or uncaptioned. | CPU. Change `assemble.py` and output size. Keep burn-in optional. |
| 2. Mix | Add a music bed ducked under speech, a short room tone in the gaps, and export stems: `voice.wav`, `music.wav`, `picture.mp4`. Keep Kokoro for drafts. | The video feels finished in the first five seconds. Speech stays intelligible. | CPU plus a small licensed or generated music library. A recorded or higher local voice can replace Kokoro on the final pass. |
| 3. Bible | Before the shot loop, generate one character sheet and one style frame. Repeat the same anchors in every prompt. Run finals on Flux Dev with PuLID (identity) or Redux (style) from that sheet. Schnell stays the draft model. Vision QA scores the frame against the sheet. | Storybook, cartoon, and pencil cuts look like one film. | GPU 0. Dev is slower than Schnell. PuLID wants a clean, front-lit sheet. Flux IP-Adapter is a style nudge, and it will not hold a face. |
| 4. Coverage | Inside a beat, allow an establishing shot, an action, and an insert. Give each shot one camera move that matches the action. Put the payoff in the opening line so the title is paid off immediately. | Retention through the first 30 seconds, which is where YouTube classifies a strong intro. | More stills per minute. The storyboard UI needs an approve step before stills run. |
| 5. Hero motion | Animate one or two approved keyframes. Leave the rest on the existing Ken Burns moves. Review those clips beside their neighbors for screen direction and wardrobe. | A few shots move with purpose. The rest stay sharp. | Only after a quantized image-to-video model fits GPU 0. Wan 2.1 14B fp16 does not. |

### Keep

Local Ollama for script and vision, ComfyUI for stills, sentence-level audio timing, style presets, and resume via `status.json`. Those are the production skeleton. A human gate belongs between storyboard and final stills, in the UI you already have.

### Hand off

Ship the master plus stems, SRT, chapters, and thumbnail. A last pass in Resolve is for a bad hand, a claim, or a music swell. The app should be able to publish without that pass once phases 1–3 are in.

**Where to spend the next GPU hours.** A character sheet plus Flux Dev and PuLID will change how a three-minute video looks more than turning every still into a video clip. Do the 1080p mix and the bible before any image-to-video experiment.

### Sources

YouTube recommended upload encoding (H.264 High, 4:2:0, 1080p at 8 Mbps for 24–30 fps, AAC stereo 384 kbps). Creator loudness practice aims the integrated level near −14 LUFS with true peak at or under −1 dBTP, because YouTube turns louder uploads down and leaves quieter ones quiet. Chapter lists need a 00:00 start, at least three timestamps, and at least 10 seconds each. Explainer workflows from Astorie, Media.io, Oakgen, and Pixo (2026) all lock a reference pack and approve a storyboard before final renders. Flux-PuLID is the identity path on Flux Dev; Flux IP-Adapter is documented for broad appearance, and its own notes say it is the wrong tool for a locked face.
