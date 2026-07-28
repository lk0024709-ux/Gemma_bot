# Gemma 3 Neuro-System — Monolithic Fullstack

Production-ready FastAPI app powering an AI "neuro-system" built on Google's
**Gemma 3** model family. **One single service** serves the web dashboard, the
JSON API and the Telegram bot — deployable on Render's free tier as one web
service, with automatic multi-provider fallback and a private Telegram channel
used as a JSON memory store.

```
gemma3-neuro-system/
├── main.py            # FastAPI server (serves API + static web UI)
├── ai_router.py       # Multi-API fallback logic for Gemma 3
├── bot_handler.py     # Telegram bot logic (background thread)
├── tg_db.py           # Telegram channel database logic
├── requirements.txt   # Python dependencies
├── .env.example       # Environment variable template
├── .gitignore         # Python gitignore (excludes .env)
├── README.md
└── frontend/          # Static web dashboard (served by FastAPI)
    ├── index.html
    ├── style.css
    └── app.js
```

## Features

- **Monolithic fullstack** — FastAPI mounts `frontend/` via `StaticFiles`, so `/`
  returns the dashboard while `/api/*` keeps serving JSON. One port, one deploy,
  no CORS headaches, no separate frontend host.
- **Telegram Mini App + Force Subscribe** — the dashboard doubles as a TMA that
  follows the user's Telegram theme. Both the bot and `/api/chat` verify channel
  membership with `get_chat_member()` before spending a single AI token.
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

| URL | What |
|-----|------|
| <http://localhost:8000/> | Web dashboard |
| <http://localhost:8000/docs> | Interactive API docs |
| <http://localhost:8000/health> | Health / keep-alive |

Editing files in `frontend/` takes effect on refresh — no build step, no bundler.
If the folder is missing the app still boots in API-only mode.

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
| `GET`  | `/` | The web dashboard (`frontend/index.html`) |
| `GET`  | `/api` | Service banner (JSON) |
| `GET`  | `/api/config` | Public client config (force-subscribe, invite link) |
| `POST` | `/api/membership` | Re-check the authenticated user's membership |
| `GET`  | `/health` | Providers + bot + channel health |
| `GET`  | `/api/providers` | Fallback chain and key configuration |
| `POST` | `/api/memory` | Store an arbitrary JSON record in the channel |
| `GET`  | `/api/bot/status` | Telegram thread diagnostics |
| `POST` | `/api/bot/restart` | Restart the polling thread |

## Force Subscribe (channel gate)

Require users to join a channel before the AI answers — in **both** the bot and
the Mini App.

```bash
REQUIRED_CHANNEL_ID=@mychannel          # @handle or -100...  (empty = disabled)
CHANNEL_INVITE_LINK=https://t.me/mychannel   # auto-derived from an @handle
```

**The bot must be an administrator of that channel**, otherwise `get_chat_member()`
fails. When the check errors the request is denied (fail-closed); set
`FORCE_SUB_FAIL_OPEN=true` to invert that.

Behaviour:

- **Bot** — non-members get `🔒 Access Denied! Please join our channel to use this AI.`
  plus a *Join Channel* button and an *I've Joined* button that re-verifies instantly.
  The AI router is never called for them.
- **API** — `POST /api/chat` requires a signed `tg_init_data` (see below).
  Non-members get **403**:
  ```json
  { "error": "FORBIDDEN_NOT_MEMBER", "invite_link": "https://t.me/mychannel" }
  ```
- **Caching** — verdicts are cached (`MEMBERSHIP_CACHE_TTL`, default 300s; negative
  results only 20s) so joining takes effect almost immediately without hammering
  Telegram.

## Telegram Mini App

`frontend/` loads the official `telegram-web-app.js` SDK, calls `tg.ready()`, and
sends the **signed** `tg.initData` string as `tg_init_data` on every request.
The CSS maps `--tg-theme-*` variables onto the palette, so the UI matches the
user's light/dark theme automatically — and falls back to the built-in dark theme
when opened in a plain browser.

To register it: **@BotFather → /newapp** (or *Bot Settings → Menu Button*) and point
it at your Render URL.

### Anti-spoofing: signed `initData` (HMAC-SHA256)

The API never trusts a client-supplied user id. The Mini App sends the **raw signed**
`window.Telegram.WebApp.initData` string as `tg_init_data`, and
`verify_tg_web_app_data()` in `main.py` validates it using Telegram's official
algorithm:

```
secret_key = HMAC_SHA256(key="WebAppData", msg=bot_token)
signature  = HMAC_SHA256(key=secret_key, msg=data_check_string)
```

where `data_check_string` is every field except `hash`, sorted alphabetically and
joined with `\n`. The comparison uses `hmac.compare_digest` (constant-time).
Only the `user.id` **inside the verified payload** is passed to the channel gate,
so `curl -d '{"telegram_user_id": 12345}'` no longer gets you anything.

| Condition | Response |
|-----------|----------|
| Missing / forged / tampered `tg_init_data` | `401 {"error": "UNAUTHORIZED_SPOOFING_DETECTED"}` |
| Valid signature, but not in the channel | `403 {"error": "FORBIDDEN_NOT_MEMBER", ...}` |
| Valid signature + channel member | `200` with the AI answer |

Replay protection: payloads older than `INIT_DATA_MAX_AGE` (default 24h) are rejected.
`initDataUnsafe` is still read for cosmetics but is **never** sent to the server.

> For local UI work outside Telegram set `ALLOW_UNVERIFIED_TMA=true` **with no bot
> token configured**. Never enable it in production — it disables anti-spoofing.

## Using the modules directly

```python
from ai_router import smart_gemma_router
print(smart_gemma_router("Hello!"))

from tg_db import save_memory_to_channel
message_id = save_memory_to_channel(CHANNEL_ID, BOT_TOKEN, {"event": "boot", "ok": True})
```

## Deploy to Render (free tier)

Create a new **Web Service** from this repo:

| Setting | Value |
|---------|-------|
| Environment | Python 3 |
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1` |

Add your `GOOGLE_API_KEY` / `GROQ_API_KEY` / `TELEGRAM_BOT_TOKEN` / etc. under
**Environment → Environment Variables** (never commit `.env`). Render injects
`$PORT` automatically. Once live, `https://<your-app>.onrender.com/` shows the
dashboard and `/api/chat` serves the API — same origin, one service.

> Keep `--workers 1`, or set `ENABLE_TELEGRAM_BOT=false` on extra workers — multiple
> polling threads on the same bot token cause Telegram `409 Conflict` errors.

### Keep it awake (free tier sleeps after 15 min)

Point an external cron at `/health` every 10 minutes:

- [cron-job.org](https://cron-job.org) or [UptimeRobot](https://uptimerobot.com) →
  monitor `https://<your-app>.onrender.com/health` at a 10-minute interval.
- Or a GitHub Action:

```yaml
name: keep-alive
on:
  schedule:
    - cron: "*/10 * * * *"
jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - run: curl -fsS https://<your-app>.onrender.com/health
```

`/health` makes **no outbound calls** by default, so pinging it is cheap. It also
answers `HEAD`. Use `/health?deep=true` when you want the Telegram token/channel
verified too.

Other hosts (Railway, Fly.io, a VPS) work with the same start command.

## Security notes

- `.env` is git-ignored; only `.env.example` is committed.
- Keys are read exclusively from the environment — nothing is hard-coded.
- CORS is **not** wide open: the dashboard is same-origin so it needs no grant.
  Only localhost dev origins are allowed by default; add more via `ALLOWED_ORIGINS`
  (comma-separated) or `ALLOWED_ORIGIN_REGEX`, or set `ALLOWED_ORIGINS=*` to opt back
  into the permissive behaviour.
- There is no authentication on `/api/chat` — anyone who finds your URL can spend
  your API quota. Add an API key or rate limiting before sharing it publicly.

## License

MIT
