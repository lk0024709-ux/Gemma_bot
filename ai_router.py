"""
ai_router.py - Mode-to-Provider Bridge for IRA

Bridges the bot's user-facing modes (Flash, Thinking, Pro, Expert, Core, Flux)
to the exact provider functions in ai_fallback.py.

This is the single entry point that bot_handler.py calls to generate responses.
"""

import logging
import os
import requests
from typing import Optional

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

def generate_image_router(prompt: str) -> bytes:
    """
    Direct HTTP POST to Hugging Face Serverless Endpoint for FLUX.1-schnell.

    Uses direct REST API calls (no SDK) to avoid third-party provider routing.
    Returns image bytes directly from the API response.

    Args:
        prompt: The user's image description.

    Returns:
        Image bytes (binary content, typically PNG or JPEG).

    Raises:
        ValueError if HF_TOKEN environment variable is missing.
        IRAProviderError if API request fails or returns an error.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is missing!")

    api_url = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": prompt}

    try:
        logger.info(f"Calling HF Serverless API for prompt: {prompt[:50]}...")
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            error_msg = f"HF API Error ({response.status_code}): {response.text}"
            logger.error(error_msg)
            raise IRAProviderError(error_msg)

        logger.info(f"✅ Image generated successfully ({len(response.content)} bytes)")
        return response.content

    except requests.RequestException as exc:
        logger.exception("Network error during HF API call")
        raise IRAProviderError(f"Network error: {str(exc)}") from exc
    except Exception as exc:
        logger.exception("Unexpected error during image generation")
        raise IRAProviderError(f"Unexpected error: {str(exc)}") from exc


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
