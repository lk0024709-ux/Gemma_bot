"""
ai_fallback.py - Model-Aware Fallback Engine for IRA

Implements REST API calls with exact model-to-provider mappings,
retry logic, 429 handling, and fallback chains.

Text Modes:
  - Groq (Flash):    GROQ_API_KEY_1 -> llama3-8b-8192
  - DeepSeek (Thinking): GITHUB_TOKEN_1 -> deepseek-r1
  - Scout (Pro):     GITHUB_TOKEN_3 -> llama-4-scout
  - Maverick (Expert): GITHUB_TOKEN_2 -> llama-4-maverick
  - Gemma (Core):    GOOGLE_API_KEY_1 -> gemma-3 (fallback: KEY_2 -> KEY_3)

Image Mode:
  - Flux:            HF_TOKEN_1 -> black-forest-labs/FLUX.1-schnell (fallback: HF_TOKEN_2)
"""

import os
import logging
import time
from typing import Optional, List

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class IRAProviderError(Exception):
    """Raised when a single AI provider call fails."""


class IRAAllProvidersFailed(Exception):
    """Raised when every provider in a fallback chain has failed."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_env(name: str) -> Optional[str]:
    """Fetch an environment variable, returning None if empty or missing."""
    val = os.getenv(name, "").strip()
    return val or None


def _is_rate_limit(exc: Exception) -> bool:
    """Check if an httpx exception represents a 429 Too Many Requests."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429
    return False


def _backoff(attempt: int, base: float = 1.0, cap: float = 10.0) -> None:
    """Simple exponential backoff sleep."""
    wait = min(base * (2 ** attempt), cap)
    logger.info("Backing off %.1f seconds (attempt %d)", wait, attempt)
    time.sleep(wait)


# ---------------------------------------------------------------------------
# Groq Provider (Flash Mode)
# ---------------------------------------------------------------------------

def call_groq(prompt: str, system_prompt: str = "") -> str:
    """
    Flash Mode -> Groq API -> llama3-8b-8192
    Uses GROQ_API_KEY_1
    """
    api_key = _get_env("GROQ_API_KEY_1")
    if not api_key:
        raise IRAProviderError("GROQ_API_KEY_1 not configured")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages: List[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "llama3-8b-8192",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2048,
    }

    for attempt in range(3):
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                return _extract_openai_content(data)
        except httpx.HTTPStatusError as exc:
            if _is_rate_limit(exc):
                logger.warning("Groq 429 rate limited (attempt %d)", attempt + 1)
                _backoff(attempt)
                continue
            logger.exception("Groq HTTP error")
            raise IRAProviderError(f"Groq HTTP {exc.response.status_code}") from exc
        except Exception as exc:
            logger.exception("Groq call failed")
            raise IRAProviderError(str(exc)) from exc

    raise IRAProviderError("Groq rate-limited on all retries")


# ---------------------------------------------------------------------------
# GitHub Models Provider (DeepSeek / Scout / Maverick)
# ---------------------------------------------------------------------------

def call_github_models(prompt: str, model: str, token_env: str, system_prompt: str = "") -> str:
    """
    Generic GitHub Models caller (Azure-compatible OpenAI chat endpoint).

    Mapping:
      - DeepSeek (Thinking): model='deepseek-r1', token_env='GITHUB_TOKEN_1'
      - Scout (Pro):         model='llama-4-scout', token_env='GITHUB_TOKEN_3'
      - Maverick (Expert):   model='llama-4-maverick', token_env='GITHUB_TOKEN_2'
    """
    token = _get_env(token_env)
    if not token:
        raise IRAProviderError(f"{token_env} not configured")

    url = os.getenv(
        "GITHUB_MODELS_ENDPOINT",
        "https://models.inference.ai.azure.com/chat/completions",
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    messages: List[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 4096,
    }

    for attempt in range(3):
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                return _extract_openai_content(data)
        except httpx.HTTPStatusError as exc:
            if _is_rate_limit(exc):
                logger.warning("GitHub Models 429 (attempt %d) for %s", attempt + 1, model)
                _backoff(attempt)
                continue
            logger.exception("GitHub Models HTTP error for %s", model)
            raise IRAProviderError(f"GitHub Models ({model}) HTTP {exc.response.status_code}") from exc
        except Exception as exc:
            logger.exception("GitHub Models call failed for %s", model)
            raise IRAProviderError(str(exc)) from exc

    raise IRAProviderError(f"GitHub Models ({model}) rate-limited on all retries")


# ---------------------------------------------------------------------------
# Google AI Studio Provider (Core / Gemma Mode with fallback chain)
# ---------------------------------------------------------------------------

def call_google_gemma(prompt: str, system_prompt: str = "") -> str:
    """
    Core Mode -> Google AI Studio -> gemma-3

    Fallback chain:
      GOOGLE_API_KEY_1 -> GOOGLE_API_KEY_2 -> GOOGLE_API_KEY_3
    """
    keys = []
    for i in range(1, 4):
        key = _get_env(f"GOOGLE_API_KEY_{i}")
        if key:
            keys.append((f"GOOGLE_API_KEY_{i}", key))

    if not keys:
        raise IRAAllProvidersFailed("No GOOGLE_API_KEY configured for Gemma mode")

    last_exc: Optional[Exception] = None
    model_name = os.getenv("GEMMA_MODEL", "gemma-3-27b-it")

    for key_label, api_key in keys:
        try:
            return _call_google_single(api_key, model_name, prompt, system_prompt)
        except httpx.HTTPStatusError as exc:
            if _is_rate_limit(exc):
                logger.warning("Google 429 with %s, trying next key", key_label)
                last_exc = exc
                continue
            logger.exception("Google HTTP error with %s", key_label)
            last_exc = exc
            continue
        except Exception as exc:
            logger.exception("Google call failed with %s", key_label)
            last_exc = exc
            continue

    raise IRAAllProvidersFailed(
        f"All Google API keys exhausted for Gemma mode (last error: {last_exc})"
    )


def _call_google_single(api_key: str, model: str, prompt: str, system_prompt: str) -> str:
    """Single attempt against Google Generative Language API."""
    base = "https://generativelanguage.googleapis.com/v1beta/models"
    url = f"{base}/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    contents = []
    # System instruction via systemInstruction field (v1beta supports this)
    payload: dict = {}

    if system_prompt:
        payload["systemInstruction"] = {
            "parts": [{"text": system_prompt}]
        }

    contents.append({"role": "user", "parts": [{"text": prompt}]})
    payload["contents"] = contents

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    # Extract text from response
    candidates = data.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        texts = [p.get("text", "") for p in parts if "text" in p]
        if texts:
            return "\n".join(t.strip() for t in texts if t).strip()

    if "output" in data and isinstance(data["output"], str):
        return data["output"].strip()

    return str(data).strip()


# ---------------------------------------------------------------------------
# Hugging Face Provider (Flux / Image Mode with fallback chain)
# ---------------------------------------------------------------------------

def call_hf_flux(prompt: str) -> bytes:
    """
    Flux Mode -> Hugging Face Inference -> black-forest-labs/FLUX.1-schnell

    Fallback chain:
      HF_TOKEN_1 -> HF_TOKEN_2

    Returns PNG image bytes.
    """
    tokens = []
    for i in range(1, 3):
        token = _get_env(f"HF_TOKEN_{i}")
        if token:
            tokens.append((f"HF_TOKEN_{i}", token))

    if not tokens:
        raise IRAAllProvidersFailed("No HF_TOKEN configured for Flux mode")

    model = "black-forest-labs/FLUX.1-schnell"
    last_exc: Optional[Exception] = None

    for token_label, token in tokens:
        try:
            return _call_hf_single(token, model, prompt)
        except httpx.HTTPStatusError as exc:
            if _is_rate_limit(exc):
                logger.warning("HF 429 with %s, trying next token", token_label)
                last_exc = exc
                continue
            logger.exception("HF HTTP error with %s", token_label)
            last_exc = exc
            continue
        except Exception as exc:
            logger.exception("HF Flux call failed with %s", token_label)
            last_exc = exc
            continue

    raise IRAAllProvidersFailed(
        f"All Hugging Face tokens exhausted for Flux mode (last error: {last_exc})"
    )


def _call_hf_single(token: str, model: str, prompt: str) -> bytes:
    """Single attempt at Hugging Face Inference API for image generation."""
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": prompt,
        "options": {"wait_for_model": True},
    }

    for attempt in range(3):
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")

            # If JSON response (some endpoints return base64)
            if "application/json" in content_type:
                data = resp.json()
                if isinstance(data, dict):
                    # Check for base64 encoded image
                    if "image" in data and isinstance(data["image"], str):
                        import base64
                        return base64.b64decode(data["image"])
                    if "data" in data and isinstance(data["data"], list):
                        for item in data["data"]:
                            if isinstance(item, dict) and "b64_json" in item:
                                import base64
                                return base64.b64decode(item["b64_json"])
                # If it's a loading/error JSON
                raise IRAProviderError(f"HF returned JSON instead of image: {data}")

            # Binary image response (PNG)
            return resp.content

    raise IRAProviderError("HF Flux retries exhausted")


# ---------------------------------------------------------------------------
# Shared extraction helpers
# ---------------------------------------------------------------------------

def _extract_openai_content(data: dict) -> str:
    """Extract text content from OpenAI-compatible chat completion response."""
    choices = data.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        content = msg.get("content") or msg.get("text", "")
        if isinstance(content, str):
            return content.strip()
    return str(data).strip()
