# Text to image: Qwen-Image 2.1 Turbo as its card says

Decided by the user on 2026-10-17: where the official model card or README says a setting is
better, it is applied as written, without a comparison trial. Text to image (the studio's
Create and `scripts/text_image.py`) therefore runs with these, all from sources read on
2026-10-17:

| Setting | Value | Source |
|---|---|---|
| Size | 2048x2048 by default; the presets 2400x1792, 1792x2400, 2528x1696, 1696x2528, 2752x1536, 1536x2752 | [Qwen-Image-2.1-Turbo model card](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) ("Use the same resolution presets as Qwen-Image-2.1") and the [Qwen-Image-2.1 README](https://github.com/QwenLM/Qwen-Image-2.1) |
| Steps, guidance | 8 steps with the checkpoint's own sigmas, cfg 1 | Turbo model card ("CFG=1 by default"); `--sigmas qwen-image-2.1-turbo` in stable-diffusion.cpp |
| Transparent background | `This is an RGBA image with transparency. <description>. The image has alpha channel and the background is transparent.` | Qwen-Image-2.1 README, "Transparent image generation (RGBA)"; stable-diffusion.cpp `docs/qwen_image_2.1.md`, "Alpha channel" |
| Prompt rewriting | QwenLM's t2i system prompt and production sampling (temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5) | Qwen-Image-2.1 README ("we recommend using the official prompt rewriting models"); `prompt_rewrite/prompts/system_prompt_t2i.txt` and `pe_core.py` at commit `ab96caa` |

The rewriting runs here, never through an API: the official system prompt on Qwen3-VL-8B,
the text encoder Qwen-Image already installs, with llama.cpp from `vendor/llama-cpp`. The
fine-tuned rewriters the README names (Qwen-Image-2.1-PE-T2I and -PE-I2I, Qwen3.5-VL 9B)
are not installed. Without the system prompt or llama.cpp the run says so and draws from
the prompt as written. The brief sent to the rewriter already asks for a transparent
background: given a bare "a knight's helmet" it invented a castle courtyard behind it.

## Measured on this PC (RTX 5070 Ti 16 GB, Q8_0, sd-cli master-961-60423ab)

| Run | Time | Peak VRAM |
|---|---|---|
| 1024x1024, text only | 14 s | 15.1 GB |
| 2048x2048, text only | 72 s (59 s sampling; the VAE decode runs out of memory and sd-cli retries it tiled, 10 s) | 14.7 GB |
| 2048x2048, rewriting + transparency, end to end | 83 s (rewriting 9 s) | |
| 1024x1024, three reference pictures | 50 s | 14.9 GB |
| 2048x2048, three reference pictures | fails: the vision encoder cannot prepare its weights | |
| RGBA template at 1024 | same time; 62 to 75% of the picture fully transparent | |

So runs that draw from reference pictures (the reference concept, the pieces of a
character) stay at 1024 and keep their prompt, and the batches start from the step
catalogue's own range until it takes the card's sizes.
