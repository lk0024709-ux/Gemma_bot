# 🎨 IRA Image Studio — Flux Model Demo

A standalone playground to **test the IRA Flux image model** (Hugging Face
`FLUX.1-dev` → `stable-diffusion-3.5-large` fallback) outside of Telegram.

It exercises the *exact same* code path the bot uses
(`ai_router.generate_image_with_meta`), so if it works here, it works in the bot.

---

## 1. Prerequisites

You need at least one Hugging Face token in your environment:

```bash
export HF_TOKEN_1=hf_xxxxxxxxxxxxxxxxxxxx   # or HF_TOKEN_2 / HF_TOKEN
```

Get one at <https://huggingface.co/settings/tokens> (a free **read** token is enough).

Install deps once (if you haven't):

```bash
pip install -r requirements.txt
```

---

## 2. Interactive web demo (recommended)

Start the server:

```bash
export TELEGRAM_BOT_TOKEN=any-non-empty-value   # only so the app boots; not used by the demo
python3 main.py            # serves on http://localhost:8000
#   or: PORT=8000 uvicorn main:app --reload
```

Open the studio in your browser:

```
http://localhost:8000/demo/
```

- The status pill turns green when the HF token is detected.
- Type a prompt (or click a suggestion / 🎲 Surprise me) and hit **Generate**.
- Each result card shows **which model** produced it, the **time taken**, the
  **file size**, and a **Download** button.

> The studio calls two endpoints:
> `GET  /api/image/status` → `{ token_configured, models }`
> `POST /api/image`         → `{ ok, image_b64, model, endpoint, elapsed_ms, size_bytes, … }`

---

## 3. Command-line tester

Quick smoke test from the terminal — saves the image to disk with a timing report:

```bash
# custom prompt
python3 test_image.py "a neon koi fish swimming through the stars"

# built-in example prompt
python3 test_image.py --example

# save to a specific path
python3 test_image.py "a samurai cat" -o result.png
```

Output goes to `demo_output/` by default (git-ignored). Exit codes:
`0` = success, `1` = config error (no token/deps), `2` = provider error.

---

## 4. How it maps to the bot

| Demo component | Bot equivalent |
|---|---|
| `POST /api/image` | `bot_handler._generate_image_and_reply` |
| `generate_image_with_meta(prompt)` | `generate_image_router(prompt)` (same engine) |
| FLUX → SD 3.5 fallback | same chain in `ai_router.py` |

The demo is **additive** — it does not change any bot behavior.
