"""Multi-tier AI provider fallback helper

Attempts providers in order:
  1. Gemini
  2. Groq
  3. GitHub Models

This module provides a single async entrypoint `generate_response(prompt)` which
tries each provider in turn and returns the first successful response. If no
provider succeeds it raises FallbackError.

Note: The actual HTTP calls to provider APIs are intentionally minimal/stubbed
here — replace with real SDK calls or HTTP client code and add proper error
handling, timeouts, retries and rate-limit handling as appropriate for your
production environment.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger("ai_fallback")


class FallbackError(Exception):
    """Raised when all fallback providers fail or none are configured."""


async def _call_gemini(prompt: str) -> str:
    # Replace with real Gemini API call. We use a simple environment-key check
    # to decide whether Gemini is configured and to simulate failures.
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Gemini API key not configured")

    # TODO: implement real async HTTP request to Gemini here.
    logger.info("Calling Gemini for prompt (truncated): %s", prompt[:80])
    # Simulate a successful response body — replace with real response parsing
    return f"(Gemini) Echo: {prompt}"


async def _call_groq(prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Groq API key not configured")

    logger.info("Calling Groq for prompt (truncated): %s", prompt[:80])
    return f"(Groq) Echo: {prompt}"


async def _call_github_models(prompt: str) -> str:
    api_token = os.getenv("GITHUB_MODELS_TOKEN")
    if not api_token:
        raise RuntimeError("GitHub Models token not configured")

    logger.info("Calling GitHub Models for prompt (truncated): %s", prompt[:80])
    return f"(GitHub Models) Echo: {prompt}"


async def generate_response(prompt: str) -> str:
    """Try multiple AI providers in order and return the first successful reply.

    Raises
    ------
    FallbackError
        If none of the providers could produce a response.
    """
    last_exc: Optional[Exception] = None

    # Provider order: Gemini -> Groq -> GitHub Models
    for provider_name, provider_call in (
        ("Gemini", _call_gemini),
        ("Groq", _call_groq),
        ("GitHub Models", _call_github_models),
    ):
        try:
            logger.info("Attempting AI provider: %s", provider_name)
            resp = await provider_call(prompt)
            logger.info("Provider %s succeeded", provider_name)
            return resp
        except Exception as e:
            last_exc = e
            logger.warning("Provider %s failed: %s", provider_name, e)
            # try next provider

    # If we reach here, all providers failed
    raise FallbackError(f"All AI providers failed (last error: {last_exc})")
