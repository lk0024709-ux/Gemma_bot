import os
import time
import random
import logging
import requests
from typing import List, Optional

logger = logging.getLogger("ai_fallback")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FallbackError(RuntimeError):
    """Raised when all providers fail."""

def _get_env_vars(keys: List[str]) -> List[str]:
    return [os.getenv(k) for k in keys if os.getenv(k)]

def is_rate_limit_error(exc: Exception, resp: Optional[requests.Response] = None) -> bool:
    """
    Heuristics to detect rate limits / quota exhaustion:
    - HTTP 429 status
    - Retry-After header
    - google.api_core.exceptions.ResourceExhausted
    - Textual hints like 'rate limit', 'too many requests', 'quota'
    """
    try:
        if resp is not None:
            status = getattr(resp, "status_code", None)
            if status == 429:
                return True
            headers = getattr(resp, "headers", {}) or {}
            if headers.get("Retry-After") or headers.get("retry-after"):
                return True
    except Exception:
        pass

    # If exc is requests.HTTPError, look at its response
    try:
        import requests as _req
        if isinstance(exc, _req.exceptions.HTTPError):
            e_resp = getattr(exc, "response", None)
            if e_resp is not None:
                if getattr(e_resp, "status_code", None) == 429:
                    return True
                headers = getattr(e_resp, "headers", {}) or {}
                if headers.get("Retry-After") or headers.get("retry-after"):
                    return True
    except Exception:
        pass

    # If google library is installed, check ResourceExhausted
    try:
        import google.api_core.exceptions as google_exceptions  # type: ignore
        if isinstance(exc, getattr(google_exceptions, "ResourceExhausted", (Exception,))):
            return True
    except Exception:
        pass

    s = str(exc).lower()
    keywords = ["rate limit", "resourceexhausted", "quota", "too many requests", "429", "rate-limited"]
    return any(k in s for k in keywords)


def _sleep_backoff(attempt: int):
    sleep_time = min(30.0, (2.0 ** (attempt - 1))) * (1 + 0.2 * (2 * random.random() - 1))
    logger.info("Backoff: sleeping %.2f seconds before retry...", sleep_time)
    time.sleep(sleep_time)


# --- PROVIDER CALLS ---
# Note: adapt the payload/endpoint to your exact provider usage if needed.

def call_gemini(prompt: str, api_key: str, timeout: float = 12.0) -> str:
    """
    Call Google Generative Language REST endpoint with API key.
    Adapt to google.generativeai SDK if you prefer.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    resp = requests.post(url, json=payload, timeout=timeout)
    try:
        resp.raise_for_status()
        data = resp.json()
        # typical response shape used in examples:
        if "candidates" in data:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        # fallback: try other common shapes
        if isinstance(data, dict):
            for k in ("output", "result", "text"):
                if k in data:
                    return str(data[k])
        return str(data)
    except requests.exceptions.RequestException as e:
        # Attach response on HTTP errors when available to help detection
        raise e

def call_groq(prompt: str, api_key: str, timeout: float = 12.0) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "llama3-8b-8192", "messages": [{"role": "user", "content": prompt}]}
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        resp.raise_for_status()
        j = resp.json()
        return j["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        raise e

def call_github_models(prompt: str, token: str, timeout: float = 12.0) -> str:
    """
    Uses your GitHub Models endpoint. The URL below was previously used in this repo.
    If you have a different URL for GitHub models, replace it here.
    """
    url = "https://models.inference.ai.azure.com/chat/completions"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"model": "meta-llama-3-70b-instruct", "messages": [{"role": "user", "content": prompt}]}
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        resp.raise_for_status()
        j = resp.json()
        return j["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        raise e


# --- MULTI-TIER FALLBACK FLOW ---

def generate_response(prompt: str) -> str:
    """
    Tries:
      1) Gemini keys: GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, GOOGLE_API_KEY_3
      2) Groq key: GROQ_API_KEY_1
      3) GitHub tokens: GITHUB_TOKEN_1, GITHUB_TOKEN_2, GITHUB_TOKEN_3
    Returns model text on first success. Raises FallbackError if all fail.
    """
    google_keys = _get_env_vars(["GOOGLE_API_KEY_1", "GOOGLE_API_KEY_2", "GOOGLE_API_KEY_3"])
    groq_keys = _get_env_vars(["GROQ_API_KEY_1"])
    github_tokens = _get_env_vars(["GITHUB_TOKEN_1", "GITHUB_TOKEN_2", "GITHUB_TOKEN_3"])

    last_exc: Optional[Exception] = None

    # 1) Gemini keys
    for idx, key in enumerate(google_keys, start=1):
        provider_label = f"Gemini (GOOGLE_API_KEY_{idx})"
        logger.info("Trying %s", provider_label)
        for attempt in range(1, 3):  # attempts per key
            try:
                resp_text = call_gemini(prompt, key)
                logger.info("%s succeeded on attempt %d", provider_label, attempt)
                return resp_text
            except Exception as e:
                last_exc = e
                resp = getattr(e, "response", None) if isinstance(e, requests.exceptions.RequestException) else None
                if is_rate_limit_error(e, resp):
                    logger.warning("%s rate-limited on attempt %d: %s. Switching to next key/provider...", provider_label, attempt, e)
                    # break attempts for this key and try next provider/key
                    break
                logger.warning("%s failed on attempt %d: %s", provider_label, attempt, e, exc_info=True)
                if attempt < 2:
                    _sleep_backoff(attempt)

    # 2) Groq
    for idx, key in enumerate(groq_keys, start=1):
        provider_label = f"Groq (GROQ_API_KEY_{idx})"
        logger.info("Switching to %s", provider_label)
        for attempt in range(1, 3):
            try:
                resp_text = call_groq(prompt, key)
                logger.info("%s succeeded on attempt %d", provider_label, attempt)
                return resp_text
            except Exception as e:
                last_exc = e
                resp = getattr(e, "response", None) if isinstance(e, requests.exceptions.RequestException) else None
                if is_rate_limit_error(e, resp):
                    logger.warning("%s rate-limited on attempt %d: %s. Switching to next provider...", provider_label, attempt, e)
                    break
                logger.warning("%s failed on attempt %d: %s", provider_label, attempt, e, exc_info=True)
                if attempt < 2:
                    _sleep_backoff(attempt)

    # 3) GitHub Models
    for idx, token in enumerate(github_tokens, start=1):
        provider_label = f"GitHub Models (GITHUB_TOKEN_{idx})"
        logger.info("Switching to %s", provider_label)
        for attempt in range(1, 3):
            try:
                resp_text = call_github_models(prompt, token)
                logger.info("%s succeeded on attempt %d", provider_label, attempt)
                return resp_text
            except Exception as e:
                last_exc = e
                resp = getattr(e, "response", None) if isinstance(e, requests.exceptions.RequestException) else None
                if is_rate_limit_error(e, resp):
                    logger.warning("%s rate-limited on attempt %d: %s. Moving on...", provider_label, attempt, e)
                    break
                logger.warning("%s failed on attempt %d: %s", provider_label, attempt, e, exc_info=True)
                if attempt < 2:
                    _sleep_backoff(attempt)

    # All providers exhausted
    raise FallbackError("All AI Providers and Keys failed!") from last_exc
