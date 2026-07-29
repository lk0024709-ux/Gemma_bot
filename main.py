"""
main.py - FastAPI Server + Threaded Telegram Bot for IRA

Architecture:
  - FastAPI runs as the main server (uvicorn).
  - Telegram bot polling runs in a background daemon thread.
  - POST /api/chat accepts TMA (Telegram Mini App) requests with HMAC validation.

Environment Variables Required:
  - TELEGRAM_BOT_TOKEN: Bot token from @BotFather
  - GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, GOOGLE_API_KEY_3 (Gemma fallback chain)
  - GROQ_API_KEY_1 (Flash/Groq mode)
  - GITHUB_TOKEN_1 (DeepSeek), GITHUB_TOKEN_2 (Maverick), GITHUB_TOKEN_3 (Scout)
  - HF_TOKEN_1, HF_TOKEN_2 (Flux image generation)
"""

import os
import hashlib
import hmac
import json
import logging
import threading
import urllib.parse
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ai_router import generate_api_response
from bot_handler import register_handlers

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ira.main")

# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(title="IRA Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Incoming request from Telegram Mini App (TMA)."""
    tg_init_data: str  # Raw initData string from Telegram WebApp
    message: str       # User's message text


class ChatResponse(BaseModel):
    """Response sent back to TMA client."""
    reply: str
    user_id: int


# ---------------------------------------------------------------------------
# HMAC-SHA256 Validation for Telegram initData
# ---------------------------------------------------------------------------

def validate_telegram_init_data(init_data: str, bot_token: str) -> Optional[dict]:
    """
    Validate Telegram Mini App initData using the official HMAC-SHA256 algorithm.

    Algorithm:
      1. Parse initData as URL-encoded query string.
      2. Extract 'hash' field and remove it from the data.
      3. Sort remaining key-value pairs alphabetically.
      4. Create data_check_string: "key1=val1\\nkey2=val2\\n..."
      5. secret_key = HMAC-SHA256(key="WebAppData", msg=bot_token)
      6. computed_hash = HMAC-SHA256(key=secret_key, msg=data_check_string).hexdigest()
      7. Compare computed_hash with the provided hash.

    Returns parsed user dict if valid, None if invalid.
    """
    try:
        # Parse URL-encoded initData
        parsed = urllib.parse.parse_qs(init_data, keep_blank_values=True)

        # Extract hash
        provided_hash = parsed.get("hash", [None])[0]
        if not provided_hash:
            logger.warning("initData missing 'hash' field")
            return None

        # Build sorted data-check string (exclude 'hash')
        data_pairs = []
        for key, values in sorted(parsed.items()):
            if key == "hash":
                continue
            data_pairs.append(f"{key}={values[0]}")

        data_check_string = "\n".join(data_pairs)

        # Compute secret key: HMAC-SHA256(key="WebAppData", msg=bot_token)
        secret_key = hmac.new(
            b"WebAppData",
            bot_token.encode("utf-8"),
            hashlib.sha256,
        ).digest()

        # Compute hash
        computed_hash = hmac.new(
            secret_key,
            data_check_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        # Constant-time comparison
        if not hmac.compare_digest(computed_hash, provided_hash):
            logger.warning("initData HMAC validation failed")
            return None

        # Parse user object from validated data
        user_json = parsed.get("user", [None])[0]
        if user_json:
            return json.loads(user_json)

        # No user data but hash is valid
        return {"id": 0}

    except Exception:
        logger.exception("Error validating initData")
        return None


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(request: ChatRequest):
    """
    POST /api/chat - TMA Backend endpoint.

    Accepts:
      - tg_init_data: Raw initData from Telegram WebApp
      - message: User's message text

    Validates the initData HMAC, then routes through AI engine.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not bot_token:
        raise HTTPException(status_code=500, detail="Server misconfigured: bot token missing")

    # Validate Telegram initData
    user_data = validate_telegram_init_data(request.tg_init_data, bot_token)
    if user_data is None:
        raise HTTPException(status_code=403, detail="Invalid or expired Telegram session")

    user_id = user_data.get("id", 0)

    # Generate AI response (uses Gemma/Core mode by default)
    try:
        reply = generate_api_response(request.message, user_id)
    except Exception:
        logger.exception("AI generation failed for /api/chat")
        reply = "Sorry, the AI service is temporarily unavailable. Please try again."

    return ChatResponse(reply=reply, user_id=user_id)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "IRA is online",
        "service": "Telegram Bot + FastAPI Backend",
        "creator": "Aditya Upadhyay",
    }


@app.get("/health")
async def health():
    """Simple health check for deployment platforms."""
    return {"ok": True}


# ---------------------------------------------------------------------------
# Telegram Bot - Background Thread
# ---------------------------------------------------------------------------

def _run_telegram_bot() -> None:
    """
    Initialize and run the Telegram bot with polling.
    This function runs inside a daemon thread.
    """
    import asyncio

    # Create and set event loop for this background thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not bot_token:
        logger.error("TELEGRAM_BOT_TOKEN not set. Bot cannot start.")
        return

    try:
        from telegram.ext import ApplicationBuilder

        application = (
            ApplicationBuilder()
            .token(bot_token)
            .build()
        )

        # Register all handlers
        register_handlers(application)

        logger.info("Starting Telegram Bot polling...")
        # stop_signals=None: this runs in a background (non-main) thread, where
        # asyncio signal handlers cannot be registered ("set_wakeup_fd only
        # works in main thread"). Uvicorn handles shutdown signals in the main
        # thread; this daemon thread exits with the process.
        application.run_polling(
            drop_pending_updates=True,
            stop_signals=None,
        )

    except Exception:
        logger.exception("Telegram bot crashed")
    finally:
        # Clean up the event loop
        try:
            loop.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Startup: Launch Bot in Background Thread
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    """Launch the Telegram bot in a background daemon thread on FastAPI startup."""
    logger.info("FastAPI starting — launching Telegram Bot in background thread.")
    bot_thread = threading.Thread(target=_run_telegram_bot, daemon=True)
    bot_thread.start()
    logger.info("Telegram Bot thread started.")


# ---------------------------------------------------------------------------
# Entry point for local development
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting IRA server on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
