"""
ai_router.py - Mode-to-Provider Bridge for IRA

Bridges the bot's user-facing modes (Flash, Thinking, Pro, Expert, Core, Flux)
to the exact provider functions in ai_fallback.py.

This is the single entry point that bot_handler.py calls to generate responses.
"""

import logging
from typing import Optional

# New imports for fast in-memory image generation
from huggingface_hub import InferenceClient
from PIL import Image
import io

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
    Fast in-memory image generation using Hugging Face InferenceClient.

    Uses provider="auto" and model "black-forest-labs/FLUX.1-schnell:preferred".

    Important: This function does NOT write any files to disk.
    It converts the returned image (PIL.Image or raw bytes) into PNG bytes
    via io.BytesIO and returns the bytes.

    Args:
        prompt: The user's image description.

    Returns:
        PNG image bytes.

    Raises:
        IRAProviderError if inference returns an unexpected result or fails.
    """
    model_ref = "black-forest-labs/FLUX.1-schnell:preferred"

    try:
        client = InferenceClient(model=model_ref, provider="auto")
    except Exception as exc:
        logger.exception("Failed to create InferenceClient")
        raise IRAProviderError(f"InferenceClient initialization failed: {exc}") from exc

    try:
        # Use the text_to_image convenience method which typically returns raw bytes.
        # Different HF backends may return bytes, a PIL.Image, or structured dicts.
        response = client.text_to_image(prompt=prompt)

        # If we received raw bytes already (PNG/JPEG), try to load via PIL and re-encode to PNG bytes.
        if isinstance(response, (bytes, bytearray)):
            try:
                img = Image.open(io.BytesIO(response))
                out = io.BytesIO()
                img.save(out, format="PNG")
                return out.getvalue()
            except Exception:
                # As a fallback, return raw bytes if PIL cannot parse (still in-memory, no disk usage).
                return bytes(response)

        # If a PIL image object was returned
        if isinstance(response, Image.Image):
            out = io.BytesIO()
            response.save(out, format="PNG")
            return out.getvalue()

        # If it's a dict or other structure, attempt to extract base64 payloads
        if isinstance(response, dict):
            # Common keys: "image" (base64), "data" (list with b64_json), etc.
            import base64

            if "image" in response and isinstance(response["image"], str):
                data = base64.b64decode(response["image"])
                img = Image.open(io.BytesIO(data))
                out = io.BytesIO()
                img.save(out, format="PNG")
                return out.getvalue()

            if "data" in response and isinstance(response["data"], list):
                for item in response["data"]:
                    if isinstance(item, dict) and "b64_json" in item:
                        data = base64.b64decode(item["b64_json"])
                        img = Image.open(io.BytesIO(data))
                        out = io.BytesIO()
                        img.save(out, format="PNG")
                        return out.getvalue()

        # If none of the above matched, raise an error
        raise IRAProviderError(f"Unexpected HF InferenceClient response type: {type(response)}")

    except Exception as exc:
        logger.exception("Hugging Face inference failed for prompt: %s", prompt)
        # Wrap in IRAProviderError so callers can handle provider-specific failures.
        raise IRAProviderError(str(exc)) from exc


def route_image(prompt: str) -> bytes:
    """
    Route an image generation request.

    Kept for backward compatibility: delegates to generate_image_router.
    """
    # Prefer the new fast in-memory router
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
