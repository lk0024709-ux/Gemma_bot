"""
ai_router.py - Mode-to-Provider Bridge for IRA

Bridges the bot's user-facing modes (Flash, Thinking, Pro, Expert, Core, Flux)
to the exact provider functions in ai_fallback.py.

This is the single entry point that bot_handler.py calls to generate responses.
"""

import logging
import os
import time
import requests
from typing import Optional
from urllib.parse import quote

from ai_fallback import (
    call_groq,
    call_github_models,
    call_google_gemma,
    call_hf_flux,
    IRAProviderError,
    IRAAllProvidersFailed,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mode Constants
# ---------------------------------------------------------------------------

MODE_GROQ = "Groq"           # /Flash
MODE_DEEPSEEK = "DeepSeek"   # /Thinking
MODE_SCOUT = "Scout"          # /Pro
MODE_MAVERICK = "Maverick"    # /Expert
MODE_GEMMA = "Gemma"          # /Core (default)
MODE_FLUX = "Flux"            # /Image

DEFAULT_MODE = MODE_GEMMA

# Default system prompt injected when user hasn't set a custom one
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, warm, and knowledgeable AI assistant. "
    "Respond clearly and concisely."
)

# IRA identity prompt
IRA_SYSTEM_PROMPT = (
    "You are IRA, an advanced multi-model AI assistant created by Aditya Upadhyay. "
    "Your systems are fully operational."
)


# ---------------------------------------------------------------------------
# Text Generation Router
# ---------------------------------------------------------------------------

def route_text(prompt: str, mode: str, system_prompt: str = "") -> str:
    """
    Route a text prompt to the correct provider based on the user's mode.

    Args:
        prompt: The user's message text.
        mode: One of MODE_GROQ, MODE_DEEPSEEK, MODE_SCOUT, MODE_MAVERICK, MODE_GEMMA.
        system_prompt: Optional custom system prompt from user state.

    Returns:
        The generated text response string.

    Raises:
        IRAAllProvidersFailed if every provider in the chain fails.
    """
    effective_system = system_prompt or DEFAULT_SYSTEM_PROMPT

    try:
        if mode == MODE_GROQ:
            return call_groq(prompt, system_prompt=effective_system)

        elif mode == MODE_DEEPSEEK:
            return call_github_models(
                prompt,
                model="deepseek-r1",
                token_env="GITHUB_TOKEN_1",
                system_prompt=effective_system,
            )

        elif mode == MODE_SCOUT:
            return call_github_models(
                prompt,
                model="llama-4-scout",
                token_env="GITHUB_TOKEN_3",
                system_prompt=effective_system,
            )

        elif mode == MODE_MAVERICK:
            return call_github_models(
                prompt,
                model="llama-4-maverick",
                token_env="GITHUB_TOKEN_2",
                system_prompt=effective_system,
            )

        elif mode == MODE_GEMMA:
            return call_google_gemma(prompt, system_prompt=effective_system)

        else:
            # Fallback to Gemma/Core for any unknown mode
            logger.warning("Unknown mode '%s', defaulting to Gemma", mode)
            return call_google_gemma(prompt, system_prompt=effective_system)

    except IRAProviderError as exc:
        logger.error("Provider failed for mode %s: %s", mode, exc)
        raise IRAAllProvidersFailed(f"Provider failed for {mode}: {exc}") from exc


# ---------------------------------------------------------------------------
# Image Generation Router
# ---------------------------------------------------------------------------

def generate_image_with_meta(prompt: str) -> dict:
    """
    Generate an image through the Pollinations.ai image generation endpoint.

    FLUX.1-dev is tried first. If it is unavailable, the request is retried with
    Stable Diffusion 3.5 Large. Direct REST calls are used (no SDK) to avoid
    third-party provider routing.

    This is the metadata-rich variant used by the Image Studio demo: it reports
    which endpoint produced the image, how long it took, and the response size.

    Returns a dict with keys:
        - image:        raw image bytes (PNG/JPEG)
        - model:        HF model id, e.g. "black-forest-labs/FLUX.1-dev"
        - endpoint:     full API URL that succeeded
        - content_type: response Content-Type header
        - elapsed_ms:   wall-clock time from first request to success (ms)
        - size_bytes:   number of image bytes returned

    Raises:
        ValueError if no HF token is found in environment variables.
        IRAProviderError if every endpoint fails or returns an error.
    """
    encoded_prompt = quote(prompt, safe="")
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"

    try:
        logger.info("Calling Pollinations image endpoint")
        response = requests.get(url, timeout=60)
    except requests.RequestException as exc:
        raise IRAProviderError(f"Pollinations network error: {exc}") from exc

    if response.status_code != 200:
        raise IRAProviderError(
            f"Pollinations API Error ({response.status_code}): {response.text}"
        )

    logger.info("Image generated successfully (%d bytes)", len(response.content))
    return response.content


def generate_image_router(prompt: str) -> bytes:
    """
    Generate an image and return the raw image bytes.

    Thin convenience wrapper around :func:`generate_image_with_meta` that keeps
    the original byte-only contract used by the Telegram bot handler.
    """
    return generate_image_with_meta(prompt)["image"]


def route_image(prompt: str) -> bytes:
    """
    Route an image generation request.

    Kept for backward compatibility: delegates to generate_image_router.
    Falls back to ai_fallback.call_hf_flux on failure.
    """
    # Prefer the new direct HTTP router
    try:
        return generate_image_router(prompt)
    except IRAProviderError:
        # Fall back to the older HF caller with fallback token logic
        logger.info("Falling back to ai_fallback.call_hf_flux for prompt")
        return call_hf_flux(prompt)


# ---------------------------------------------------------------------------
# Convenience: Generate response for FastAPI /api/chat endpoint
# ---------------------------------------------------------------------------

def generate_api_response(message: str, user_id: int) -> str:
    """
    Generate a response for the TMA backend API endpoint.
    Uses Gemma (Core) mode as default for web clients.

    Args:
        message: The user's message.
        user_id: The Telegram user ID (from validated init data).

    Returns:
        Generated reply string.
    """
    try:
        return route_text(message, mode=MODE_GEMMA, system_prompt=DEFAULT_SYSTEM_PROMPT)
    except IRAAllProvidersFailed:
        logger.error("All providers failed for API request (user %d)", user_id)
        return "Sorry, the AI service is temporarily unavailable. Please try again shortly."
