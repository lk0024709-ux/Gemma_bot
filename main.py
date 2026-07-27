"""
main.py
========
FastAPI entry point for the **Gemma 3 Neuro-System** — monolithic fullstack app.

A single Render web service that simultaneously:

* serves the static frontend (``frontend/``) as a dashboard UI,
* exposes the JSON chat API consumed by that dashboard,
* runs the Telegram bot polling loop in a background thread,
* answers ``/health`` so an external cron job can keep the free tier awake.

Run locally::

    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
import urllib.parse
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from ai_router import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODE,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    available_modes,
    available_providers,
    capability_router,
    smart_gemma_router,
    smart_gemma_router_verbose,
)
from bot_handler import (
    force_sub_enabled,
    get_bot,
    get_bot_status,
    is_user_member,
    start_bot_thread,
    stop_bot,
)
from tg_db import health_check as tg_health_check
from tg_db import log_interaction, save_memory_to_channel

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("neuro-system")

APP_NAME = os.getenv("APP_NAME", "Gemma 3 Neuro-System")
APP_VERSION = "1.0.0"
LOG_WEB_TO_CHANNEL = os.getenv("LOG_WEB_TO_CHANNEL", "true").lower() in ("1", "true", "yes")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID", "").strip()

# --------------------------------------------------------------------------- #
# Force-subscribe (Telegram Mini App gate)                                     #
# --------------------------------------------------------------------------- #
# Channel the user must join before the AI answers. Empty = gate disabled.
REQUIRED_CHANNEL_ID = os.getenv("REQUIRED_CHANNEL_ID", "").strip()
CHANNEL_INVITE_LINK = os.getenv("CHANNEL_INVITE_LINK", "").strip()

# Derive a usable invite link from an @handle when none was supplied.
if not CHANNEL_INVITE_LINK and REQUIRED_CHANNEL_ID.startswith("@"):
    CHANNEL_INVITE_LINK = f"https://t.me/{REQUIRED_CHANNEL_ID.lstrip('@')}"

# --------------------------------------------------------------------------- #
# Telegram Mini App initData validation                                        #
# --------------------------------------------------------------------------- #
# Reject initData older than this many seconds, which stops a leaked payload
# from being replayed forever. Set to 0 to disable the freshness check.
INIT_DATA_MAX_AGE = int(os.getenv("INIT_DATA_MAX_AGE", "86400"))  # 24h

# When no bot token is configured we cannot verify signatures at all. Refusing
# to run the gate open is the safe choice, but it would break local frontend
# development, so it is opt-in.
ALLOW_UNVERIFIED_TMA = os.getenv("ALLOW_UNVERIFIED_TMA", "false").lower() in ("1", "true", "yes")

_STARTED_AT = time.time()

# --------------------------------------------------------------------------- #
# Frontend (monolithic fullstack: the API also serves the dashboard)           #
# --------------------------------------------------------------------------- #

# Resolved relative to this file, not the CWD, so the app works no matter where
# Render (or you) launches uvicorn from.
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", BASE_DIR / "frontend")).resolve()
INDEX_FILE = FRONTEND_DIR / "index.html"
FRONTEND_AVAILABLE = INDEX_FILE.is_file()

# --------------------------------------------------------------------------- #
# CORS                                                                         #
# --------------------------------------------------------------------------- #
# The dashboard is same-origin now, so it needs no CORS grant at all. We only
# allow the local dev origins (Vite/Live Server/etc.) plus anything explicitly
# listed in ALLOWED_ORIGINS, e.g.
#     ALLOWED_ORIGINS=https://my-dashboard.vercel.app,https://app.example.com
# Set ALLOWED_ORIGINS=* to restore the old wide-open behaviour.

DEFAULT_DEV_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
]
_raw_origins = os.getenv("ALLOWED_ORIGINS", "").strip()
if _raw_origins == "*":
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS = False  # browsers reject "*" together with credentials
else:
    _extra = [o.strip().rstrip("/") for o in _raw_origins.split(",") if o.strip()]
    CORS_ORIGINS = list(dict.fromkeys(DEFAULT_DEV_ORIGINS + _extra))
    CORS_ALLOW_CREDENTIALS = True


# --------------------------------------------------------------------------- #
# Telegram WebApp HMAC-SHA256 validation                                       #
# --------------------------------------------------------------------------- #


def verify_tg_web_app_data(init_data: str, bot_token: str) -> Optional[Dict[str, Any]]:
    """Validate a Telegram Mini App ``initData`` string and return its payload.

    Implements Telegram's official algorithm
    (https://core.telegram.org/bots/webapps#validating-data-received-via-the-mini-app):

    1. Parse the URL-encoded query string.
    2. Pop the ``hash`` field.
    3. Build ``data_check_string``: remaining ``key=value`` pairs sorted
       alphabetically by key and joined with ``\\n``.
    4. ``secret_key = HMAC_SHA256(key="WebAppData", msg=bot_token)``.
    5. ``signature  = HMAC_SHA256(key=secret_key, msg=data_check_string)``.
    6. Constant-time compare against the received ``hash``.

    Returns
    -------
    dict | None
        The parsed fields on success, with the ``user`` JSON string already
        decoded into a dict (plus a convenience ``user_id`` key). ``None`` if
        the signature is missing, malformed, forged or expired.

    Notes
    -----
    Never raises - any malformed input simply yields ``None``.
    """
    if not init_data or not isinstance(init_data, str):
        logger.debug("initData validation failed: empty payload")
        return None
    if not bot_token:
        logger.error("initData validation failed: TELEGRAM_BOT_TOKEN is not configured")
        return None

    try:
        # keep_blank_values so empty fields still take part in the check string.
        parsed = urllib.parse.parse_qsl(init_data, keep_blank_values=True, strict_parsing=True)
    except ValueError as exc:
        logger.debug("initData validation failed: unparsable query string (%s)", exc)
        return None

    # Duplicate keys would make the check string ambiguous -> reject.
    keys = [k for k, _ in parsed]
    if len(keys) != len(set(keys)):
        logger.warning("initData validation failed: duplicate keys in payload")
        return None

    data = dict(parsed)
    received_hash = data.pop("hash", "")
    if not received_hash:
        logger.debug("initData validation failed: no hash field")
        return None

    # Step 3 - alphabetically sorted "key=value" lines.
    data_check_string = "\n".join(f"{key}={data[key]}" for key in sorted(data))

    # Step 4/5 - derive the secret key, then sign the check string.
    secret_key = hmac.new(b"WebAppData", bot_token.encode("utf-8"), hashlib.sha256).digest()
    computed_hash = hmac.new(
        secret_key, data_check_string.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    # Step 6 - constant-time comparison defeats timing attacks.
    if not hmac.compare_digest(computed_hash, received_hash):
        logger.warning("initData validation failed: signature mismatch (possible spoofing)")
        return None

    # Optional replay protection: reject stale payloads.
    if INIT_DATA_MAX_AGE > 0:
        try:
            auth_date = int(data.get("auth_date", "0"))
        except (TypeError, ValueError):
            auth_date = 0
        if auth_date <= 0:
            logger.warning("initData validation failed: missing/invalid auth_date")
            return None
        age = time.time() - auth_date
        if age > INIT_DATA_MAX_AGE:
            logger.warning("initData validation failed: payload is %.0fs old (expired)", age)
            return None

    # Signature is valid - decode the nested JSON fields.
    result: Dict[str, Any] = dict(data)
    for field in ("user", "receiver", "chat"):
        raw = result.get(field)
        if isinstance(raw, str) and raw:
            try:
                result[field] = json.loads(raw)
            except ValueError:
                logger.warning("initData: %s field is not valid JSON", field)
                if field == "user":
                    return None

    user = result.get("user")
    if not isinstance(user, dict) or user.get("id") is None:
        logger.warning("initData validation failed: no usable user object")
        return None

    try:
        result["user_id"] = int(user["id"])
    except (TypeError, ValueError):
        logger.warning("initData validation failed: user.id is not an integer")
        return None

    return result


# --------------------------------------------------------------------------- #
# Schemas                                                                      #
# --------------------------------------------------------------------------- #


class ChatMessage(BaseModel):
    role: str = Field(default="user", description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's prompt", max_length=16000)
    history: Optional[List[ChatMessage]] = Field(
        default=None, description="Optional prior conversation turns"
    )
    system_prompt: Optional[str] = Field(default=None, description="Override the system prompt")
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, le=8192)
    user_id: Optional[str] = Field(default=None, description="Client identifier for logging")
    tg_init_data: Optional[str] = Field(
        default=None,
        description=(
            "Raw, signed window.Telegram.WebApp.initData string. Verified server-side "
            "via HMAC-SHA256; required when force-subscribe is enabled."
        ),
    )
    mode: Optional[str] = Field(
        default=None,
        description="Capability: flash | reasoning | pro | core | vision. Defaults to core.",
    )
    images: Optional[List[str]] = Field(
        default=None,
        description="Base64 image strings (or data: URLs). Forces the vision engine.",
    )

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("message must not be empty")
        return value.strip()


class ChatResponse(BaseModel):
    success: bool
    response: str
    provider: str
    model: str
    latency_ms: int
    memory_id: Optional[int] = None
    timestamp: str


class MembershipRequest(BaseModel):
    tg_init_data: Optional[str] = Field(
        default=None, description="Raw signed window.Telegram.WebApp.initData string"
    )


class MemoryRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Arbitrary JSON record to persist")
    tags: Optional[List[str]] = Field(default=None, description="Hashtags for the record")


# --------------------------------------------------------------------------- #
# Lifespan: start / stop the Telegram bot alongside the API                    #
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s v%s", APP_NAME, APP_VERSION)
    for provider in available_providers():
        logger.info(
            "  provider %-12s %-28s %s",
            provider["name"],
            provider["model"],
            "configured" if provider["configured"] else "MISSING KEY",
        )
    if FRONTEND_AVAILABLE:
        logger.info("  frontend     serving %s", FRONTEND_DIR)
    else:
        logger.warning("  frontend     %s not found - UI disabled, API still up", INDEX_FILE)
    logger.info("  cors         %s", ", ".join(CORS_ORIGINS))
    if force_sub_enabled():
        logger.info("  force-sub    ON - users must join %s", REQUIRED_CHANNEL_ID)
        if not CHANNEL_INVITE_LINK:
            logger.warning(
                "  force-sub    CHANNEL_INVITE_LINK is empty - blocked users get no join button"
            )
    else:
        logger.info("  force-sub    off (set REQUIRED_CHANNEL_ID to enable)")

    try:
        thread = start_bot_thread()
        logger.info("Telegram bot thread: %s", "started" if thread else "not started")
    except Exception as exc:  # noqa: BLE001 - the API must boot regardless
        logger.exception("Could not start the Telegram bot: %s", exc)

    yield

    logger.info("Shutting down %s", APP_NAME)
    try:
        stop_bot()
    except Exception as exc:  # noqa: BLE001
        logger.error("Error while stopping the Telegram bot: %s", exc)


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Multi-provider Gemma 3 backend serving a Telegram bot and a web dashboard.",
    lifespan=lifespan,
)

# CORS: same-origin dashboard needs none; this covers local dev + extra origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=os.getenv("ALLOWED_ORIGIN_REGEX") or None,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Error handling                                                               #
# --------------------------------------------------------------------------- #


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"success": False, "error": "Internal server error", "detail": str(exc)},
    )


# --------------------------------------------------------------------------- #
# Routes                                                                       #
# --------------------------------------------------------------------------- #


@app.get("/", include_in_schema=False)
async def serve_index() -> Any:
    """Serve the dashboard's ``index.html`` (monolithic fullstack root)."""
    if not FRONTEND_AVAILABLE:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "name": APP_NAME,
                "version": APP_VERSION,
                "status": "online",
                "detail": "Frontend not found; API endpoints are available.",
                "endpoints": ["/api/chat", "/api/providers", "/health", "/docs"],
            },
        )
    # no-cache so a redeploy never leaves users on a stale dashboard shell
    return FileResponse(INDEX_FILE, headers={"Cache-Control": "no-cache"})


@app.get("/api", tags=["meta"])
async def api_root() -> Dict[str, Any]:
    """Service banner (the JSON that used to live at ``/``)."""
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "status": "online",
        "uptime_seconds": int(time.time() - _STARTED_AT),
        "frontend": FRONTEND_AVAILABLE,
        "endpoints": ["/api/chat", "/api/providers", "/api/memory", "/health", "/docs"],
    }


@app.get("/health", tags=["meta"])
async def health(deep: bool = False) -> Dict[str, Any]:
    """Health check — also the keep-alive target for an external cron job.

    Ping ``GET /health`` every 10 minutes (UptimeRobot, cron-job.org, GitHub
    Actions, ...) to stop Render's free tier from spinning the service down
    after 15 minutes of inactivity.

    The default response is intentionally cheap: no outbound network calls, so
    it stays fast and never burns Telegram API quota. Pass ``?deep=true`` to
    additionally verify the bot token and channel access.
    """
    providers = available_providers()
    payload: Dict[str, Any] = {
        "status": "healthy" if any(p["configured"] for p in providers) else "degraded",
        "uptime_seconds": int(time.time() - _STARTED_AT),
        "providers": providers,
        "frontend": FRONTEND_AVAILABLE,
        "telegram_bot": get_bot_status(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if deep:
        try:
            payload["telegram_db"] = tg_health_check()
        except Exception as exc:  # noqa: BLE001 - health must never 500
            payload["telegram_db"] = {"error": str(exc)}
    return payload


@app.head("/health", include_in_schema=False)
async def health_head() -> JSONResponse:
    """Cheapest possible keep-alive: uptime monitors often send HEAD."""
    return JSONResponse(content=None, status_code=status.HTTP_200_OK)


@app.get("/api/config", tags=["meta"])
async def client_config() -> Dict[str, Any]:
    """Public config the Mini App needs at boot (no secrets)."""
    return {
        "success": True,
        "app_name": APP_NAME,
        "force_subscribe": force_sub_enabled(),
        "invite_link": CHANNEL_INVITE_LINK,
        "channel": REQUIRED_CHANNEL_ID or None,
    }


@app.post("/api/membership", tags=["telegram"])
async def check_membership(payload: MembershipRequest) -> Any:
    """Check whether the *authenticated* user has joined the required channel.

    Used by the Mini App's "I've joined, re-check" button so the user doesn't
    have to send a throwaway message to find out.

    Takes the signed ``initData`` rather than a bare user id - otherwise anyone
    could enumerate the membership status of arbitrary Telegram accounts.
    """
    if not force_sub_enabled():
        return {"success": True, "is_member": True, "force_subscribe": False}

    if not TELEGRAM_BOT_TOKEN and ALLOW_UNVERIFIED_TMA:
        return {"success": True, "is_member": True, "force_subscribe": True, "unverified": True}

    verified = await run_in_threadpool(
        verify_tg_web_app_data, payload.tg_init_data or "", TELEGRAM_BOT_TOKEN
    )
    if not verified:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": "UNAUTHORIZED_SPOOFING_DETECTED",
                "message": (
                    "Could not verify your Telegram identity. "
                    "Please open this app from inside Telegram."
                ),
            },
        )

    is_member = await run_in_threadpool(
        is_user_member, get_bot(), verified["user_id"], REQUIRED_CHANNEL_ID, False
    )
    return {
        "success": True,
        "is_member": bool(is_member),
        "force_subscribe": True,
        "invite_link": CHANNEL_INVITE_LINK,
    }


@app.get("/api/modes", tags=["ai"])
async def modes() -> Dict[str, Any]:
    """List the capability map (flash / reasoning / pro / core / vision)."""
    return {"success": True, "default": DEFAULT_MODE, "modes": available_modes()}


@app.get("/api/providers", tags=["ai"])
async def providers() -> Dict[str, Any]:
    """List the Gemma 3 fallback chain in priority order."""
    return {"success": True, "fallback_order": available_providers()}


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    responses={403: {"description": "User is not a member of the required channel"}},
    tags=["ai"],
)
async def chat(payload: ChatRequest) -> Any:
    """Main chat endpoint.

    Accepts ``{"message": "..."}`` (plus optional ``history``, ``system_prompt``,
    ``temperature``, ``max_tokens``) and returns Gemma 3's answer together with
    the provider that served it.
    """
    # --- Gate 1: cryptographic identity (anti-spoofing) ------------------- #
    # The client can claim anything, so the only trustworthy user id is the one
    # inside a payload signed with our bot token.
    verified_user_id: Optional[int] = None

    if force_sub_enabled():
        unauthorized = JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": "UNAUTHORIZED_SPOOFING_DETECTED",
                "message": (
                    "Could not verify your Telegram identity. "
                    "Please open this app from inside Telegram."
                ),
            },
        )

        if not TELEGRAM_BOT_TOKEN:
            if not ALLOW_UNVERIFIED_TMA:
                logger.error("Cannot verify initData: TELEGRAM_BOT_TOKEN is not set")
                return unauthorized
            logger.warning("ALLOW_UNVERIFIED_TMA is on - skipping initData verification")
        else:
            if not payload.tg_init_data:
                logger.info("Chat request rejected: no tg_init_data supplied")
                return unauthorized

            # HMAC over a short string is fast, but keep the event loop clean.
            verified = await run_in_threadpool(
                verify_tg_web_app_data, payload.tg_init_data, TELEGRAM_BOT_TOKEN
            )
            if not verified:
                logger.warning("Chat request rejected: initData failed HMAC validation")
                return unauthorized

            verified_user_id = verified["user_id"]

    # --- Gate 2: force-subscribe channel membership ----------------------- #
    # Runs before any AI call so non-members never consume provider quota.
    if force_sub_enabled() and verified_user_id is not None:
        # get_chat_member() is a blocking HTTP call - run it off the event loop
        # so it can never stall other requests.
        allowed = await run_in_threadpool(
            is_user_member, get_bot(), verified_user_id, REQUIRED_CHANNEL_ID
        )
        if not allowed:
            logger.info("Chat request rejected: user %s is not a member", verified_user_id)
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": "FORBIDDEN_NOT_MEMBER",
                    "invite_link": CHANNEL_INVITE_LINK,
                    "message": "🔒 Access Denied! Please join our channel to use this AI.",
                },
            )

    history = [m.model_dump() for m in payload.history] if payload.history else None

    try:
        result = await run_in_threadpool(
            capability_router,
            payload.message,
            payload.mode or DEFAULT_MODE,
            payload.images,
            payload.system_prompt or DEFAULT_SYSTEM_PROMPT,
            history,
            payload.temperature,
            payload.max_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat generation failed: %s", exc)
        raise HTTPException(status_code=502, detail="AI providers unavailable") from exc

    if result.provider == "none":
        logger.error("All providers failed for a /api/chat request: %s", result.errors)
        raise HTTPException(status_code=503, detail=result.text)

    memory_id: Optional[int] = None
    if LOG_WEB_TO_CHANNEL:
        try:
            memory_id = log_interaction(
                user_id=payload.user_id,
                username=None,
                prompt=payload.message,
                response=result.text,
                source="web",
                provider=result.provider,
            )
        except Exception as exc:  # noqa: BLE001 - logging must not break the response
            logger.error("Failed to archive the web interaction: %s", exc)

    return ChatResponse(
        success=True,
        response=result.text,
        provider=result.provider,
        model=result.model,
        latency_ms=result.latency_ms,
        memory_id=memory_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/api/memory", tags=["memory"])
async def create_memory(payload: MemoryRequest) -> Dict[str, Any]:
    """Persist an arbitrary JSON record in the private Telegram channel."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHANNEL_ID:
        raise HTTPException(
            status_code=503,
            detail="Telegram channel storage is not configured "
            "(set TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID)",
        )

    message_id = save_memory_to_channel(
        channel_id=TELEGRAM_CHANNEL_ID,
        bot_token=TELEGRAM_BOT_TOKEN,
        json_data=payload.data,
        tags=payload.tags,
    )
    if message_id is None:
        raise HTTPException(status_code=502, detail="Could not write to the Telegram channel")

    return {"success": True, "message_id": message_id}


@app.get("/api/bot/status", tags=["telegram"])
async def bot_status() -> Dict[str, Any]:
    """Telegram bot thread diagnostics."""
    return {"success": True, "bot": get_bot_status()}


@app.post("/api/bot/restart", tags=["telegram"])
async def bot_restart() -> Dict[str, Any]:
    """Stop and restart the Telegram polling thread."""
    stop_bot()
    thread = start_bot_thread()
    return {"success": bool(thread), "bot": get_bot_status()}


# --------------------------------------------------------------------------- #
# Static frontend mounts                                                       #
# --------------------------------------------------------------------------- #
# IMPORTANT: these are registered *after* every API route. Starlette matches
# routes in registration order, so /api/*, /health and /docs always win and the
# catch-all mount below only handles what's left (CSS, JS, images, favicon...).

if FRONTEND_AVAILABLE:
    # Explicit prefix — handy for absolute asset URLs like /static/style.css
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    # Root mount: serves /style.css, /app.js, /favicon.ico natively, and
    # html=True makes directory requests fall back to index.html.
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    logger.warning(
        "Frontend directory %s is missing - running in API-only mode", FRONTEND_DIR
    )


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() in ("1", "true", "yes"),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
