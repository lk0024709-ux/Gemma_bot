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
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from ai_router import smart_gemma_router

# ---------------------------------------------------------------------------
# Configuration — set these environment variables or replace with literals.
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "<YOUR_BOT_TOKEN_HERE>")
CHANNEL_USERNAME = os.getenv("CHANNEL_USERNAME", "@yourchannel")   # used by gatekeeper


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

    # ------------------------------------------------------------------
    # 1.  Extract the `hash` value and remove it from the data dict so
    #     it is NOT included in the data-check-string.
    # ------------------------------------------------------------------
    received_hash = data.pop("hash", None)
    if received_hash is None:
        return None

    # ------------------------------------------------------------------
    # 2.  Sort the remaining key-value pairs alphabetically by key and
    #     build the data-check-string (key=value, newline-separated).
    # ------------------------------------------------------------------
    sorted_items = sorted(data.items(), key=lambda kv: kv[0])
    data_check_string = "\n".join(f"{k}={v}" for k, v in sorted_items)

    # ------------------------------------------------------------------
    # 3.  Create the secret key:
    #     HMAC-SHA-256(bot_token, "WebAppData")
    # ------------------------------------------------------------------
    secret_key = hmac.new(
        key=b"WebAppData",
        msg=bot_token.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()

    # ------------------------------------------------------------------
    # 4.  Compute the expected signature:
    #     HMAC-SHA-256(secret_key, data_check_string)
    # ------------------------------------------------------------------
    expected_hash = hmac.new(
        key=secret_key,
        msg=data_check_string.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()

    # ------------------------------------------------------------------
    # 5.  Compare (constant-time) the received and expected hashes.
    # ------------------------------------------------------------------
    if not hmac.compare_digest(received_hash, expected_hash):
        return None

    # ------------------------------------------------------------------
    # 6.  Validation succeeded — parse and return the `user` object.
    # ------------------------------------------------------------------
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
    allow_origins=["*"],   # tighten this in production!
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
    user_obj = verify_tg_web_app_data(request.tg_init_data, TELEGRAM_BOT_TOKEN)
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
    try:
        ai_reply = await smart_gemma_router(
            prompt,
            mode="normal",
            chat_id=verified_user_id,
            user_id=verified_user_id
        )
    except Exception as e:
        ai_reply = f"Hello, Telegram user {verified_user_id}! (API Fallback) This is a stub reply."

    return ChatResponse(reply=ai_reply, user_id=verified_user_id)


# ===========================================================================
# 5.  Health-check (optional)
# ===========================================================================

@app.get("/health")
async def health():
    return {"status": "ok"}
