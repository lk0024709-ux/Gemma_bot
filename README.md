# Gemma 3 Neuro-System — Backend

Production-ready FastAPI backend powering an AI "neuro-system" built on Google's
**Gemma 3** model family. One codebase serves **both** a Telegram bot and a web
dashboard, with automatic multi-provider fallback and a private Telegram channel
used as a JSON memory store.

```
gemma3-neuro-backend/
├── main.py            # FastAPI application & entry point
├── ai_router.py       # Multi-API fallback logic for Gemma 3
├── bot_handler.py     # Telegram bot logic (background thread)
├── tg_db.py           # Telegram channel database logic
├── requirements.txt   # Python dependencies
├── .env.example       # Environment variable template
├── .gitignore         # Python gitignore (excludes .env)
└── README.md
```

## Features

- **4-tier AI fallback** — Google AI Studio → Groq → OpenRouter → Hugging Face.
  If one provider is down, rate-limited or missing a key, the next one takes over
  transparently.
- **Correct Gemma 3 chat template** — `<start_of_turn>user … <end_of_turn>` with the
  system instruction folded into the first user turn (Gemma has no system role).
- **Telegram bot** — pyTelegramBotAPI polling in a daemon thread, so it never blocks
  Uvicorn. Rolling per-chat memory, `/start`, `/help`, `/reset`, `/status`, auto message
  chunking above 4096 chars.
- **Telegram-channel database** — every exchange is archived as a JSON record; the
  returned `message_id` is your primary key (read / edit / delete supported).
- **Robust error handling** — per-provider retries with backoff, flood-control awareness,
  graceful degradation, and a global exception handler.
- **CORS open to all origins** so any web dashboard can call the API.

## Fallback chain

| Order | Provider | Default model | Env key |
|-------|----------|---------------|---------|
| 1 | Google AI Studio (Generative Language API) | `gemma-3-27b-it` | `GOOGLE_API_KEY` |
| 2 | Groq | `gemma-3-12b-it` | `GROQ_API_KEY` |
| 3 | OpenRouter | `google/gemma-3-27b-it:free` | `OPENROUTER_API_KEY` |
| 4 | Hugging Face Inference API | `google/gemma-3-27b-it` | `HF_TOKEN` |

Every model ID is overridable via env vars (`GOOGLE_MODEL`, `GROQ_MODEL`,
`OPENROUTER_MODEL`, `HF_MODEL`) so you can point at whichever Gemma 3 endpoint a
provider currently exposes.

## Setup

```bash
git clone <your-repo-url>
cd gemma3-neuro-backend

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env           # then fill in your keys
```

You need **at least one** AI provider key. The Telegram parts are optional —
without `TELEGRAM_BOT_TOKEN` the API still runs, just with the bot disabled.

### Telegram setup

1. Create a bot with [@BotFather](https://t.me/BotFather) → copy the token into
   `TELEGRAM_BOT_TOKEN`.
2. Create a **private channel**, add your bot as an **administrator** with
   "Post messages" permission.
3. Get the channel ID (forward a channel post to
   [@userinfobot](https://t.me/userinfobot), or use `getUpdates`). It looks like
   `-1001234567890` → put it in `TELEGRAM_CHANNEL_ID`.

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# or simply:
python main.py
```

Interactive docs: <http://localhost:8000/docs>

Run the bot alone (no API):

```bash
python bot_handler.py
```

## API

### `POST /api/chat`

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain quantum entanglement in two sentences."}'
```

```json
{
  "success": true,
  "response": "Quantum entanglement is …",
  "provider": "google",
  "model": "gemma-3-27b-it",
  "latency_ms": 842,
  "memory_id": 137,
  "timestamp": "2026-07-27T10:15:00+00:00"
}
```

Optional fields: `history` (`[{"role":"user","content":"…"}]`), `system_prompt`,
`temperature`, `max_tokens`, `user_id`.

### Other endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Service banner |
| `GET`  | `/health` | Providers + bot + channel health |
| `GET`  | `/api/providers` | Fallback chain and key configuration |
| `POST` | `/api/memory` | Store an arbitrary JSON record in the channel |
| `GET`  | `/api/bot/status` | Telegram thread diagnostics |
| `POST` | `/api/bot/restart` | Restart the polling thread |

## Using the modules directly

```python
from ai_router import smart_gemma_router
print(smart_gemma_router("Hello!"))

from tg_db import save_memory_to_channel
message_id = save_memory_to_channel(CHANNEL_ID, BOT_TOKEN, {"event": "boot", "ok": True})
```

## Deployment

Any container/PaaS host works (Railway, Render, Fly.io, VPS):

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
```

> Keep `--workers 1`, or set `ENABLE_TELEGRAM_BOT=false` on extra workers — multiple
> polling threads on the same bot token cause Telegram `409 Conflict` errors.

## Security notes

- `.env` is git-ignored; only `.env.example` is committed.
- Keys are read exclusively from the environment — nothing is hard-coded.
- CORS is wide open by design for public dashboards; put the service behind an API
  gateway or add auth before exposing it in production.

## License

MIT
