"""
Telegram Mini App (TMA) Backend — FastAPI
=========================================
Secured against spoofing via Telegram's official HMAC-SHA256 WebApp data
validation.  Every request to /api/chat must carry a valid initData string
signed by Telegram.
"""
import hmac
import hashlib
import json
import os
import urllib.parse
import logging
import threading
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from a .env file during local development only.
# In production (Render) environment variables are injected by the platform.
try:
    from dotenv import load_dotenv

    # Only attempt to load .env if it exists to avoid masking production env vars.
    if os.path.exists(".env"):
        load_dotenv()
except Exception:
    # If python-dotenv is not installed or loading fails, continue — env vars may be set by the platform.
    pass

# Import AI router (keeps existing app logic)
from ai_router import smart_gemma_router
# Import fallback helper
from ai_fallback import generate_response, FallbackError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Configuration — prefer environment variables, avoid insecure defaults
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # do not provide a secret literal here
CHANNEL_USERNAME = os.getenv("CHANNEL_USERNAME", "@yourchannel")  # used by gatekeeper

if not TELEGRAM_BOT_TOKEN:
    logger.warning("TELEGRAM_BOT_TOKEN is not set. Telegram bot polling will run using a fallback token placeholder.")

# ===========================================================================
# 1.  Telegram WebApp data validation (HMAC-SHA256)
# ===========================================================================
def verify_tg_web_app_data(init_data: str, bot_token: str) -> Optional[Dict]:
    """
    Validate the integrity and authenticity of Telegram Mini App initData.

    Parameters
    ----------
    init_data : str
        The raw URL-encoded initData string supplied by
        ``window.Telegram.WebApp.initData`` on the frontend.
    bot_token : str
        The bot token obtained from @BotFather.

    Returns
    -------
    dict | None
        The parsed ``user`` object (as a Python dict) if the signature is
        valid, otherwise ``None``.
    """
    if not init_data:
        return None

    # Parse the URL-encoded string into a dict of key/value pairs.
    parsed = urllib.parse.parse_qs(init_data, keep_blank_values=True)

    # Flatten single-element lists produced by parse_qs.
    data: dict[str, str] = {k: v[0] for k, v in parsed.items()}

    # 1. Extract the `hash` and remove it from data so it is not included in the
    #    data-check-string.
    received_hash = data.pop("hash", None)
    if received_hash is None:
        return None

    # 2. Sort remaining key/value pairs and build the data-check-string.
    sorted_items = sorted(data.items(), key=lambda kv: kv[0])
    data_check_string = "\n".join(f"{k}={v}" for k, v in sorted_items)

    # 3. Create the secret key
    secret_key = hmac.new(
        key=b"WebAppData",
        msg=bot_token.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()

    # 4. Compute the expected signature
    expected_hash = hmac.new(
        key=secret_key,
        msg=data_check_string.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()

    # 5. Constant-time compare
    if not hmac.compare_digest(received_hash, expected_hash):
        return None

    # 6. Parse and return the user object, if present
    user_json = data.get("user", "{}")
    try:
        user_obj: dict = json.loads(user_json)
    except json.JSONDecodeError:
        return None

    return user_obj


# ===========================================================================
# 2.  Pydantic models
# ===========================================================================
class ChatRequest(BaseModel):
    """
    Request body for the /api/chat endpoint.

    The frontend MUST send the raw ``initData`` string (obtained from
    ``window.Telegram.WebApp.initData``) so the backend can verify it.
    """
    tg_init_data: str = Field(..., description="Raw initData from Telegram.WebApp")
    message: Optional[str] = Field(None, description="The user's prompt or message")


class ChatResponse(BaseModel):
    reply: str
    user_id: int


# ===========================================================================
# 3.  FastAPI application
# ===========================================================================
app = FastAPI(title="Gemma Bot TMA Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Channel / membership gatekeeper (stub — replace with real Telegram API call)
# ---------------------------------------------------------------------------
async def is_user_member(user_id: int) -> bool:
    """
    Check whether the given *user_id* is a member of the configured channel.

    In production, call the Telegram Bot API:
        GET https://api.telegram.org/bot<TOKEN>/getChatMember
            ?chat_id=@channelusername&user_id=<user_id>

    For now this is a stub that returns True — **replace with real logic**.
    """
    # TODO: replace with actual Bot API call
    return True


# ===========================================================================
# 4.  /api/chat  —  the main conversational endpoint
# ===========================================================================
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Authenticated chat endpoint.

    Flow
    ----
    1.  Validate the ``tg_init_data`` string using Telegram's HMAC scheme.
    2.  If invalid → 401 UNAUTHORIZED_SPOOFING_DETECTED.
    3.  If valid  → extract ``user_id`` and run channel gatekeeper.
    4.  If the user is a member → run the AI/LLM logic and return the reply.
    """
    # --- Step 1 & 2: Validate initData ------------------------------------
    user_obj = verify_tg_web_app_data(request.tg_init_data, TELEGRAM_BOT_TOKEN or "")
    if user_obj is None:
        raise HTTPException(
            status_code=401,
            detail="UNAUTHORIZED_SPOOFING_DETECTED",
        )

    # --- Step 3: Extract verified user_id --------------------------------
    verified_user_id: int = user_obj.get("id")
    if verified_user_id is None:
        raise HTTPException(status_code=400, detail="User ID missing from initData")

    # --- Step 4: Channel gatekeeper ---------------------------------------
    if not await is_user_member(verified_user_id):
        raise HTTPException(
            status_code=403,
            detail="NOT_A_CHANNEL_MEMBER",
        )

    # --- Step 5: AI / LLM logic -----------------------------------
    prompt = request.message if request.message else "Hello!"

    # Primary: existing smart_gemma_router. If it fails for any reason, fall
    # back to the multi-provider generator implemented in ai_fallback.generate_response.
    try:
        ai_reply = await smart_gemma_router(
            prompt,
            mode="normal",
            chat_id=verified_user_id,
            user_id=verified_user_id,
        )
    except Exception as e:
        logger.exception("Primary LLM (smart_gemma_router) failed: %s", e)
        try:
            ai_reply = await generate_response(prompt)
        except FallbackError as fe:
            logger.exception("All fallback providers failed: %s", fe)
            ai_reply = f"Hello, Telegram user {verified_user_id}! (All AI providers failed) This is a stub reply."

    return ChatResponse(reply=ai_reply, user_id=verified_user_id)


# ===========================================================================
# 5.  Health-check (optional)
# ===========================================================================
@app.get("/health")
async def health():
    return {"status": "ok"}


# ===========================================================================
# 6.  Start Telegram bot polling in background (non-blocking)
# ===========================================================================
def _start_bot_thread():
    """
    Import the bot handler and start polling in a daemon thread so it doesn't
    block the FastAPI event loop. Import is done inside the function to avoid
    side-effects during module import in contexts that don't want polling.
    """
    try:
        # Importing the module registers handlers; start_bot_polling runs a
        # blocking polling loop, so run it in a separate thread.
        import bot_handler  # local module that defines start_bot_polling()
        t = threading.Thread(target=bot_handler.start_bot_polling, name="tg-poller", daemon=True)
        t.start()
        logger.info("Telegram bot polling started in background thread.")
    except Exception as e:
        logger.exception("Failed to start Telegram bot polling: %s", e)


@app.on_event("startup")
async def on_startup():
    # Start the background bot polling thread on startup.
    _start_bot_thread()


# Note:
# - The container runtime / command (uvicorn) must bind to 0.0.0.0 and use the
#   $PORT environment variable provided by Render. That is handled by the Dockerfile/CMD.
