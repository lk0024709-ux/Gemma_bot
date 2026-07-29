"""
ai_router.py - AI routing, key rotation, and system prompt management

This file implements a multi-provider routing strategy with dynamic key/token
rotation, explicit fallback chains, and watermarking for responses.

Security notes:
- NO API keys or secrets are hardcoded. Keys/tokens are loaded from environment
  variables and rotated via round-robin pools.
- All provider call blocks log full exception tracebacks using logger.exception.

Model mapping (environment variables expected):
- GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, GOOGLE_API_KEY_3 (fallback: GOOGLE_API_KEY)
- GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3 (fallback: GROQ_API_KEY)
- GITHUB_TOKEN_1, GITHUB_TOKEN_2, GITHUB_TOKEN_3 (fallback: GITHUB_TOKEN)
- HF_TOKEN_1, HF_TOKEN_2 (fallback: HF_TOKEN)

Primary default model: gemma-3-27b-it (Google AI Studio tier)

"""

import os
import logging
import traceback
import threading
import time
from typing import List, Optional, Tuple
import requests

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

# ---- Key pools and rotation state ----
_lock = threading.Lock()
_rotation_idx = {
    "google": 0,
    "groq": 0,
    "github": 0,
    "hf": 0,
}


def _load_pool(prefix: str, count: int, fallback: Optional[str] = None) -> List[str]:
    """Load a list of keys from environment using the prefix and numeric suffix.
    e.g. prefix=GOOGLE_API_KEY_, count=3 -> GOOGLE_API_KEY_1..3
    If no numbered keys found, a single fallback env var name may be used.
    """
    keys = []
    for i in range(1, count + 1):
        name = f"{prefix}{i}"
        val = os.getenv(name)
        if val:
            keys.append(val)
    if not keys and fallback:
        fb = os.getenv(fallback)
        if fb:
            keys = [fb]
    return keys


# Pools configuration
_GOOGLE_KEYS = _load_pool("GOOGLE_API_KEY_", 3, fallback="GOOGLE_API_KEY")
_GROQ_KEYS = _load_pool("GROQ_API_KEY_", 3, fallback="GROQ_API_KEY")
_GITHUB_TOKENS = _load_pool("GITHUB_TOKEN_", 3, fallback="GITHUB_TOKEN")
_HF_TOKENS = _load_pool("HF_TOKEN_", 2, fallback="HF_TOKEN")

# Default model name
DEFAULT_MODEL = "gemma-3-27b-it"

# Image enhancer configuration
IMAGE_ENHANCER_ENABLED = os.getenv("IMAGE_ENHANCER_ENABLED", "true").lower() in ("1", "true", "yes")
IMAGE_ENHANCER_TEMPLATE = os.getenv(
    "IMAGE_ENHANCER_TEMPLATE",
    "{prompt}, studio-grade photorealism, true-to-life color science, realistic skin textures and material physics, cinematic lighting with natural shadows, sharp focus, highly detailed textures, shot on 35mm lens, f/1.8 aperture, 8k resolution, clean composition, zero AI artifacts."
)
# Backoff / retry config for HF
HF_MAX_RETRIES = int(os.getenv("HF_MAX_RETRIES", "3"))
HF_BACKOFF_BASE = float(os.getenv("HF_BACKOFF_BASE", "1.0"))  # seconds


def _pick_key(pool: List[str], pool_name: str) -> Optional[str]:
    """Round-robin pick from a pool thread-safely. Returns None if pool empty."""
    if not pool:
        return None
    with _lock:
        idx = _rotation_idx.get(pool_name, 0) % len(pool)
        _rotation_idx[pool_name] = (_rotation_idx.get(pool_name, 0) + 1) % len(pool)
    return pool[idx]


# ---------------- Provider wrappers ----------------

def _call_google_ai_studio(prompt: str, model: str, api_key: str, **kwargs) -> str:
    """Placeholder HTTP call to Google AI Studio. Replace endpoint or client
    calls if you have a dedicated SDK. This function is written defensively
    and logs full tracebacks on exceptions.

    Important: Do NOT log the api_key itself.
    """
    try:
        # Example endpoint - the exact path/params depend on your Google AI Studio setup.
        url = os.getenv("GOOGLE_AI_ENDPOINT", "https://api.googleaistudio.example/v1/generate")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "prompt": prompt, "max_tokens": 512}
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Adapt extracting logic depending on real response format
        return data.get("output", data.get("text", "")).strip()
    except Exception:
        logger.exception("AI Generation Error - Google AI Studio call failed")
        raise


def _call_github_models(prompt: str, model: str, token: str, **kwargs) -> str:
    """Placeholder HTTP call to a GitHub hosted model endpoint. The concrete
    endpoint or SDK may differ; adapt as needed. This is defensive and logs
    exceptions.
    """
    try:
        url = os.getenv("GITHUB_MODELS_ENDPOINT", "https://api.githubmodels.example/v1/generate")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"model": model, "input": prompt, "max_tokens": 1024}
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("output", data.get("text", "")).strip()
    except Exception:
        logger.exception("AI Generation Error - GitHub Models call failed")
        raise


def _call_groq(prompt: str, model: str, api_key: str, **kwargs) -> str:
    """Placeholder HTTP call to Groq inference endpoint. Adapt as required.
    """
    try:
        url = os.getenv("GROQ_ENDPOINT", "https://api.groq.example/v1/generate")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "prompt": prompt, "max_tokens": 512}
        resp = requests.post(url, json=payload, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return data.get("output", data.get("text", "")).strip()
    except Exception:
        logger.exception("AI Generation Error - Groq call failed")
        raise


def _call_hf_image_model(prompt: str, model: str, token: str, **kwargs) -> bytes:
    """Call Hugging Face Inference API to generate an image. Returns raw bytes
    suitable for sending as a photo. Uses the official Inference endpoint.
    """
    try:
        url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {token}"}
        # include wait_for_model to handle cold starts gracefully
        payload = {"inputs": prompt, "options": {"wait_for_model": True}}
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        # HF may return JSON with base64 or binary like image data depending on model.
        # If it's JSON with a base64 string under 'image', decode it. Otherwise return content.
        content_type = resp.headers.get("Content-Type", "")
        if "application/json" in content_type:
            data = resp.json()
            # Look for common fields
            if isinstance(data, dict) and "image" in data:
                import base64
                return base64.b64decode(data["image"])
            # Some models return {'data': [{'b64_json': '...'}]}
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                for item in data["data"]:
                    if isinstance(item, dict) and "b64_json" in item:
                        import base64
                        return base64.b64decode(item["b64_json"])
            # Fallback - convert the JSON str
            return str(data).encode("utf-8")
        else:
            return resp.content
    except Exception:
        # Per requirement, log with this specific message for flux failures
        logger.exception("Flux Image Generation Failed")
        raise


# ---------------- Public routing functions ----------------


def generate_image_router(prompt: str, hf_model: str = "black-forest-labs/FLUX.1-schnell") -> bytes:
    """Route image generation requests to Hugging Face Flux.1 models using
    rotating HF tokens. Returns image bytes.

    This function applies a lightweight GPT Image 2-style prompt enhancer
    via string templating only (no extra LLM calls) to improve photorealism
    while adding zero server load.

    It also implements a lightweight exponential backoff to handle transient
    HF errors (429/503) and rotates HF tokens on persistent failures.
    """
    # Determine enhanced prompt based on env toggle and template
    if IMAGE_ENHANCER_ENABLED:
        tpl = IMAGE_ENHANCER_TEMPLATE
        # Allow templates that contain a {prompt} placeholder, otherwise append
        if "{prompt}" in tpl:
            try:
                enhanced_prompt = tpl.format(prompt=prompt)
            except Exception:
                # Fallback to safe concatenation if template formatting fails
                logger.exception("IMAGE_ENHANCER_TEMPLATE formatting failed, falling back to concatenation")
                enhanced_prompt = f"{prompt}, {tpl}"
        else:
            enhanced_prompt = f"{prompt}, {tpl}"
    else:
        enhanced_prompt = prompt

    # Cycle through HF tokens (round-robin). For each token, attempt retries with exponential backoff
    tried_tokens = 0
    total_tokens = max(1, len(_HF_TOKENS))

    while tried_tokens < total_tokens:
        token = _pick_key(_HF_TOKENS, "hf")
        if not token:
            break

        attempt = 0
        while attempt < HF_MAX_RETRIES:
            attempt += 1
            try:
                return _call_hf_image_model(enhanced_prompt, hf_model, token)
            except requests.exceptions.HTTPError as http_err:
                # Try to inspect status code for transient errors
                status = None
                try:
                    status = http_err.response.status_code  # type: ignore[attr-defined]
                except Exception:
                    status = None
                if status in (429, 502, 503):
                    backoff = HF_BACKOFF_BASE * (2 ** (attempt - 1))
                    logger.warning(f"Transient HF HTTP {status} error on attempt {attempt}. Backing off {backoff}s and retrying.")
                    logger.exception("Flux Image Generation Failed")
                    time.sleep(backoff)
                    continue
                else:
                    # Non-transient HTTP error - log and break to try next token
                    logger.exception("Flux Image Generation Failed")
                    break
            except Exception:
                # Non-HTTP exceptions could be network errors, decode errors, etc.
                logger.exception("Flux Image Generation Failed")
                # For generic exceptions, treat as transient and backoff a bit
                backoff = HF_BACKOFF_BASE * (2 ** (attempt - 1))
                time.sleep(backoff)
                continue

        # This token exhausted; try next
        tried_tokens += 1
        logger.warning("HF token exhausted or failed repeatedly; rotating to next token if available.")

    # If we reach here, all tokens/retries failed
    logger.error("All Hugging Face Flux tokens failed to generate the image")
    raise RuntimeError("All Hugging Face Flux tokens failed to generate the image")


def is_secret_request(user_text: str) -> bool:
    """Same heuristics as before to detect secret-extraction attempts."""
    import re
    _SECRETS_PATTERNS = [
        re.compile(r"(api[_-]?key|secret|token|password|env|environment variable)", re.IGNORECASE),
        re.compile(r"(show me your keys|give (me|us) your secrets|what is your api key)", re.IGNORECASE),
        re.compile(r"(server structure|routing logic|internal architecture|source code for .*server)", re.IGNORECASE),
    ]
    for pat in _SECRETS_PATTERNS:
        if pat.search(user_text or ""):
            return True
    return False


def get_system_prompt(model_name: str) -> str:
    guardrails = (
        "You are a warm, polite, and friendly AI assistant."
        " Always be helpful, concise, and kind.\n\n"
        "Security guardrails (MUST follow):\n"
        "- Under NO circumstances reveal API keys, environment variables, secrets, or private server configuration.\n"
        "- If asked for secrets, keys, or internal implementation details, refuse politely with the message: \"I’m sorry, but I can’t share secrets, API keys, or private configuration.\"\n"
    )
    model_info = f"Current runtime model name: {model_name}\n"
    persona = "Persona: Respond in a warm, polite, and friendly tone.\n"
    return guardrails + model_info + persona


def _append_watermark(text: str, model_name: str) -> str:
    return f"{text}\n\n---\n⚡ *Generated via {model_name}*"


def generate_ai_response(user_text: str, model_name: Optional[str] = None) -> str:
    """Top-level routing for text responses.

    Routing & fallback order:
      1. Google AI Studio (gemma-3-27b-it)
      2. GitHub Models (deepseek-r1, llama-4-maverick, llama-4-scout)
      3. Groq (llama-3.1-8b-instant)

    The function will attempt providers in order and rotate keys/tokens.
    All provider exceptions are logged with logger.exception and then the
    next provider is attempted.
    """
    model_name = model_name or DEFAULT_MODEL

    if is_secret_request(user_text):
        return "I’m sorry, but I can’t share secrets, API keys, or private configuration."

    system_prompt = get_system_prompt(model_name)
    full_prompt = system_prompt + "\nUser: " + user_text

    # 1) Try Google AI Studio primary
    google_key = _pick_key(_GOOGLE_KEYS, "google")
    if google_key:
        try:
            reply = _call_google_ai_studio(full_prompt, "gemma-3-27b-it", google_key)
            return _append_watermark(reply, "Gemma 3 (gemma-3-27b-it)")
        except Exception:
            logger.exception("Google AI Studio failed; will try GitHub Models as fallback")

    # 2) Try GitHub Models tier
    for gh_model in ["deepseek-r1", "llama-4-maverick", "llama-4-scout"]:
        token = _pick_key(_GITHUB_TOKENS, "github")
        if not token:
            break
        try:
            reply = _call_github_models(full_prompt, gh_model, token)
            return _append_watermark(reply, f"GitHub Models ({gh_model})")
        except Exception:
            logger.exception(f"GitHub model {gh_model} failed; trying next GitHub model or Groq")

    # 3) Try Groq tier
    groq_key = _pick_key(_GROQ_KEYS, "groq")
    if groq_key:
        try:
            reply = _call_groq(full_prompt, "llama-3.1-8b-instant", groq_key)
            return _append_watermark(reply, "Groq (llama-3.1-8b-instant)")
        except Exception:
            logger.exception("Groq generation failed; exhausting fallbacks")

    # Final fallback - friendly failure message
    logger.error("All AI provider attempts failed for the request")
    return "Sorry — I had trouble contacting the AI service. Please try again later."


# Convenience smart router for legacy callers in the repo
async def smart_gemma_router(prompt: str, mode: str = "normal", image_base64: Optional[str] = None, chat_id: Optional[int] = None, user_id: Optional[int] = None, model: Optional[str] = None) -> str:
    """Async wrapper used by bot handlers. Routes multimodal requests to the
    appropriate provider and returns string reply (watermarked).
    """
    # If an image is included, prefer the Google multimodal gemma route where possible
    if image_base64:
        # For now, reuse generate_ai_response shorthand with a caption that indicates multimodal input
        multimodal_prompt = f"[Image attached]. Caption/Query: {prompt}\n[Image base64 length={len(image_base64)}]"
        return generate_ai_response(multimodal_prompt, model_name=model)
    else:
        return generate_ai_response(prompt, model_name=model)


def get_current_model() -> str:
    return os.environ.get("CURRENT_MODEL", DEFAULT_MODEL)
