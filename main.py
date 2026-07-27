"""
main.py
========
FastAPI entry point for the **Gemma 3 Neuro-System** backend.

Responsibilities
----------------
* Expose a JSON chat API consumed by the web dashboard (CORS open to all origins).
* Start the Telegram bot polling thread on application startup.
* Provide health / provider introspection endpoints for monitoring.

Run locally::

    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from ai_router import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    available_providers,
    smart_gemma_router,
    smart_gemma_router_verbose,
)
from bot_handler import get_bot_status, start_bot_thread, stop_bot
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

_STARTED_AT = time.time()


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

# CORS: open to all origins so any web dashboard can call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # must be False when allow_origins is "*"
    allow_methods=["*"],
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


@app.get("/", tags=["meta"])
async def root() -> Dict[str, Any]:
    """Service banner."""
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "status": "online",
        "uptime_seconds": int(time.time() - _STARTED_AT),
        "endpoints": ["/api/chat", "/api/providers", "/api/memory", "/health", "/docs"],
    }


@app.get("/health", tags=["meta"])
async def health() -> Dict[str, Any]:
    """Deep health check: API, providers and Telegram integration."""
    providers = available_providers()
    return {
        "status": "healthy" if any(p["configured"] for p in providers) else "degraded",
        "uptime_seconds": int(time.time() - _STARTED_AT),
        "providers": providers,
        "telegram_bot": get_bot_status(),
        "telegram_db": tg_health_check(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/providers", tags=["ai"])
async def providers() -> Dict[str, Any]:
    """List the Gemma 3 fallback chain in priority order."""
    return {"success": True, "fallback_order": available_providers()}


@app.post("/api/chat", response_model=ChatResponse, tags=["ai"])
async def chat(payload: ChatRequest) -> ChatResponse:
    """Main chat endpoint.

    Accepts ``{"message": "..."}`` (plus optional ``history``, ``system_prompt``,
    ``temperature``, ``max_tokens``) and returns Gemma 3's answer together with
    the provider that served it.
    """
    history = [m.model_dump() for m in payload.history] if payload.history else None

    try:
        result = smart_gemma_router_verbose(
            payload.message,
            system_prompt=payload.system_prompt or DEFAULT_SYSTEM_PROMPT,
            history=history,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
            raise_on_failure=False,
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
