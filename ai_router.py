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
import io
from huggingface_hub import InferenceClient
from PIL import Image

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
    "{prompt}, studio-grade photorealism, true-to-life color science, realistic skin textures and material physics, cinematic lighting with natural shadows, sharp focus, highly detailed textures, high resolution"
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
    """Call Google Gemini / Generative Language API (generateContent).

    Uses the REST endpoint:
    https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}

    Payload format: {"contents": [{"parts": [{"text": prompt}]}]}
    """
    try:
        base = "https://generativelanguage.googleapis.com/v1beta/models"
        url = f"{base}/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Robust extraction across possible response shapes
        if isinstance(data, dict):
            candidates = data.get("candidates")
            if isinstance(candidates, list) and candidates:
                texts = []
                for c in candidates:
                    if not isinstance(c, dict):
                        continue
                    content = c.get("content") or c.get("output") or []
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                texts.append(item["text"])
                    if isinstance(c.get("output"), str):
                        texts.append(c.get("output"))
                if texts:
                    return "\n".join([t.strip() for t in texts if t]).strip()

            if "output" in data and isinstance(data["output"], str):
                return data["output"].strip()

        return str(data).strip()
    except Exception:
        logger.exception("AI Generation Error - Google AI Studio call failed")
        raise


def _call_github_models(prompt: str, model: str, token: str, **kwargs) -> str:
    """Call GitHub-hosted models via the Azure-compatible chat/completions endpoint.

    Uses the OpenAI chat payload format: {"model": model, "messages": [{"role":"user","content":prompt}]}
    Endpoint default: https://models.inference.ai.azure.com/chat/completions
    """
    try:
        url = os.getenv("GITHUB_MODELS_ENDPOINT", "https://models.inference.ai.azure.com/chat/completions")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict):
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                msg = first.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content") or msg.get("text")
                    if isinstance(content, str):
                        return content.strip()
                text = first.get("text")
                if isinstance(text, str):
                    return text.strip()
        return str(data).strip()
    except Exception:
        logger.exception("AI Generation Error - GitHub Models call failed")
        raise


def _call_groq(prompt: str, model: str, api_key: str, **kwargs) -> str:
    """Call Groq's OpenAI-compatible chat completions endpoint.

    Uses the OpenAI chat payload format: {"model": model, "messages": [{"role": "user", "content": prompt}]}
    Endpoint: https://api.groq.com/openai/v1/chat/completions
    """
    try:
        url = os.getenv("GROQ_ENDPOINT", "https://api.groq.com/openai/v1/chat/completions")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        resp = requests.post(url, json=payload, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict):
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                msg = first.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content") or msg.get("text")
                    if isinstance(content, str):
                        return content.strip()
                text = first.get("text")
                if isinstance(text, str):
                    return text.strip()
        return str(data).strip()
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
        payload = {"inputs": prompt, "options": {"wait_for_model": True}}
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "application/json" in content_type:
            data = resp.json()
            if isinstance(data, dict) and "image" in data:
                import base64
                return base64.b64decode(data["image"])
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                for item in data["data"]:
                    if isinstance(item, dict) and "b64_json" in item:
                        import base64
                        return base64.b64decode(item["b64_json"])
            return str(data).encode("utf-8")
        else:
            return resp.content
    except Exception:
        logger.exception("Flux Image Generation Failed")
        raise


# ---------------- Public routing functions ----------------


def generate_image_router(prompt: str, hf_model: str = "black-forest-labs/FLUX.1-schnell:preferred") -> bytes:
    """Generate an image using Hugging Face InferenceClient (preferred FLUX.1-schnell).

    This implementation prefers rotating HF tokens when available. It uses the
    huggingface_hub.InferenceClient.text_to_image convenience method which may
    return a PIL.Image.Image or raw bytes depending on the provider/model.
    """
    # Enhance prompt if enabled
    if IMAGE_ENHANCER_ENABLED:
        tpl = IMAGE_ENHANCER_TEMPLATE
        if "{prompt}" in tpl:
            try:
                enhanced_prompt = tpl.format(prompt=prompt)
            except Exception:
                logger.exception("IMAGE_ENHANCER_TEMPLATE formatting failed, falling back to concatenation")
                enhanced_prompt = f"{prompt}, {tpl}"
        else:
            enhanced_prompt = f"{prompt}, {tpl}"
    else:
        enhanced_prompt = prompt

    # Try rotating HF tokens if set, otherwise fall back to environment single token
    tried = 0
    total = max(1, len(_HF_TOKENS))
    last_exc = None

    while tried < total:
        token = _pick_key(_HF_TOKENS, "hf") or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("Hugging Face API Token (HF_TOKEN) is missing.")

        try:
            client = InferenceClient(provider="auto", api_key=token)

            # Use the text_to_image helper; different HF providers may return
            # either a PIL.Image.Image or bytes. Handle both.
            image_obj = client.text_to_image(enhanced_prompt, model=hf_model)

            # If it's bytes, return directly
            if isinstance(image_obj, (bytes, bytearray)):
                return bytes(image_obj)

            # If it's a PIL Image, convert to PNG bytes
            if isinstance(image_obj, Image.Image):
                buf = io.BytesIO()
                image_obj.save(buf, format="PNG")
                return buf.getvalue()

            # If it's a dict with base64 data
            if isinstance(image_obj, dict):
                # common key 'image' or nested data
                import base64
                if "image" in image_obj and isinstance(image_obj["image"], str):
                    return base64.b64decode(image_obj["image"])
                if "data" in image_obj and isinstance(image_obj["data"], list):
                    for item in image_obj["data"]:
                        if isinstance(item, dict) and "b64_json" in item:
                            return base64.b64decode(item["b64_json"])
                # fallback to stringified payload
                return str(image_obj).encode("utf-8")

            # Last resort: stringify
            return str(image_obj).encode("utf-8")

        except Exception as e:
            last_exc = e
            # Inspect for transient HTTP errors if requests exception available
            if isinstance(e, requests.exceptions.HTTPError):
                status = None
                try:
                    status = e.response.status_code  # type: ignore[attr-defined]
                except Exception:
                    status = None
                if status in (429, 502, 503):
                    backoff = HF_BACKOFF_BASE * (2 ** tried)
                    logger.warning(f"Transient HF HTTP {status} error. Backing off {backoff}s and retrying.")
                    time.sleep(backoff)
                    tried += 1
                    continue
            logger.exception("Flux Image Generation Failed")
            # rotate token and try next
            tried += 1
            continue

    # If we reach here, all tokens failed
    logger.error("All Hugging Face Flux tokens failed to generate the image: %s", last_exc)
    raise RuntimeError(f"All Hugging Face Flux tokens failed to generate the image: {last_exc}")


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
