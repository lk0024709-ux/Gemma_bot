"""
ai_router.py
=============
Multi-provider fallback router for Google's **Gemma 3** model family.

Providers are tried in a strict priority order. The first provider that returns
a usable completion wins; any failure (missing key, HTTP error, rate limit,
timeout, malformed payload) is logged and the router moves on to the next one.

Fallback order
--------------
1. Google AI Studio (Generative Language API) -> ``gemma-3-27b-it``
2. Groq                                       -> ``gemma-3-12b-it`` (configurable)
3. OpenRouter                                 -> ``google/gemma-3-27b-it:free``
4. Hugging Face Inference API                 -> ``google/gemma-3-27b-it``

Public API
----------
``smart_gemma_router(prompt, system_prompt=None, history=None, ...) -> str``
``smart_gemma_router_verbose(...) -> RouterResult``
``build_gemma_prompt(...) -> str``   (raw Gemma 3 chat template)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

# Model IDs (overridable via env so you can chase whatever endpoint is live).
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemma-3-27b-it")
GROQ_MODEL = os.getenv("GROQ_MODEL", "gemma-3-12b-it")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-3-27b-it:free")
HF_MODEL = os.getenv("HF_MODEL", "google/gemma-3-27b-it")

# --------------------------------------------------------------------------- #
# Capability map                                                               #
# --------------------------------------------------------------------------- #
# Each capability ("mode") pins a task type to the provider + model best suited
# to it. Gemma 3 on Google AI Studio is the core/default engine and the only
# multimodal one, so every mode ultimately falls back to `core`.

CORE_MODE = "core"

MODEL_MAP: Dict[str, Dict[str, Any]] = {
    # High speed, low parameter count - snappy replies.
    "flash": {
        "provider": "groq",
        "model": os.getenv("FLASH_MODEL", "openai/gpt-oss-20b"),
        "label": "⚡ Flash",
        "description": "Groq · GPT-OSS 20B — fastest replies (~1000 t/s)",
        "vision": False,
    },
    # Step-by-step logic / chain-of-thought.
    "reasoning": {
        "provider": "openrouter",
        "model": os.getenv("REASONING_MODEL", "deepseek/deepseek-r1:free"),
        "label": "🧩 Reasoning",
        "description": "OpenRouter · DeepSeek-R1 — step-by-step logic",
        "vision": False,
    },
    # Heavy coding / expert tasks.
    "pro": {
        "provider": "openrouter",
        "model": os.getenv("PRO_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
        "label": "🎓 Pro",
        "description": "OpenRouter · Llama 3.3 70B — expert coding",
        "vision": False,
    },
    # Balanced everyday engine - the default.
    CORE_MODE: {
        "provider": "google",
        "model": os.getenv("CORE_MODEL", GOOGLE_MODEL),
        "label": "🌐 Core",
        "description": "Google AI Studio · Gemma 3 27B — balanced engine",
        "vision": False,
    },
    # Multimodal: handles all image inputs.
    "vision": {
        "provider": "google",
        "model": os.getenv("VISION_MODEL", "gemma-3-27b-it"),
        "label": "👁 Vision",
        "description": "Google AI Studio · Gemma 3 multimodal — image understanding",
        "vision": True,
    },
}

DEFAULT_MODE = CORE_MODE
VISION_MODE = "vision"


def normalise_mode(mode: Optional[str]) -> str:
    """Coerce arbitrary user input to a known capability, defaulting to core."""
    key = str(mode or "").strip().lower().lstrip("/")
    return key if key in MODEL_MAP else DEFAULT_MODE


# Model ids that providers have retired. Warned about at startup so a dead
# capability surfaces at boot instead of mid-conversation.
KNOWN_DEPRECATED: Dict[str, str] = {
    "meta-llama/llama-4-scout-17b-16e-instruct": "Groq shut this down 2026-07-17; use openai/gpt-oss-20b",
    "llama-4-scout": "not a valid Groq model id; use openai/gpt-oss-20b",
    "meta-llama/llama-4-maverick-17b-128e-instruct": "Groq shut this down 2026-03-09",
    "qwen/qwen3-32b": "Groq shut this down 2026-07-17",
    "llama-3.1-8b-instant": "Groq deprecation: shutdown 2026-08-16; use openai/gpt-oss-20b",
    "llama-3.3-70b-versatile": "Groq deprecation: shutdown 2026-08-16",
    "deepseek/deepseek-r1:free": "OpenRouter retired the DeepSeek :free tier; drop ':free'",
}


def check_model_health() -> List[str]:
    """Return human-readable warnings for retired/invalid model ids."""
    warnings: List[str] = []
    for mode, spec in MODEL_MAP.items():
        note = KNOWN_DEPRECATED.get(spec["model"])
        if note:
            warnings.append(f"'{mode}' -> {spec['model']}: {note}")
    return warnings


def available_modes() -> List[Dict[str, Any]]:
    """Describe every capability, including whether its provider key is set."""
    return [
        {
            "mode": name,
            "provider": spec["provider"],
            "model": spec["model"],
            "label": spec["label"],
            "description": spec["description"],
            "vision": spec["vision"],
            "configured": bool(_provider_key(spec["provider"])),
        }
        for name, spec in MODEL_MAP.items()
    ]

# Endpoints
GOOGLE_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
HF_ENDPOINT = "https://api-inference.huggingface.co/models/{model}"

# Generation defaults
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are Gemma, a helpful, concise and friendly AI assistant powering a "
    "neuro-system that serves a Telegram bot and a web dashboard. "
    "Answer clearly and avoid unnecessary preamble.",
)
DEFAULT_TEMPERATURE = float(os.getenv("AI_TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.getenv("AI_MAX_TOKENS", "1024"))
REQUEST_TIMEOUT = int(os.getenv("AI_REQUEST_TIMEOUT", "60"))
# Images make the request much heavier, so vision gets a longer budget.
VISION_TIMEOUT = int(os.getenv("AI_VISION_TIMEOUT", "120"))
MAX_RETRIES_PER_PROVIDER = int(os.getenv("AI_MAX_RETRIES", "2"))
RETRY_BACKOFF_SECONDS = float(os.getenv("AI_RETRY_BACKOFF", "1.5"))

# HTTP status codes worth retrying on the *same* provider before falling back.
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504, 529}

# Optional OpenRouter attribution headers
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "https://localhost")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "gemma3-neuro-backend")


# --------------------------------------------------------------------------- #
# Errors & result objects                                                      #
# --------------------------------------------------------------------------- #


class ProviderError(RuntimeError):
    """Raised when a single provider fails; caught by the router."""

    def __init__(self, provider: str, message: str, status: Optional[int] = None):
        self.provider = provider
        self.status = status
        super().__init__(f"[{provider}] {message}")


class AllProvidersFailedError(RuntimeError):
    """Raised when every configured provider failed."""

    def __init__(self, errors: Dict[str, str]):
        self.errors = errors
        detail = " | ".join(f"{k}: {v}" for k, v in errors.items()) or "no providers configured"
        super().__init__(f"All Gemma 3 providers failed -> {detail}")


@dataclass
class RouterResult:
    """Structured result returned by :func:`smart_gemma_router_verbose`."""

    text: str
    provider: str
    model: str
    latency_ms: int
    errors: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "errors": self.errors,
        }


# --------------------------------------------------------------------------- #
# Gemma 3 chat template helpers                                                #
# --------------------------------------------------------------------------- #

Message = Dict[str, str]  # {"role": "user" | "assistant" | "model", "content": "..."}


def _normalise_history(history: Optional[Sequence[Message]]) -> List[Message]:
    """Coerce arbitrary history payloads into ``[{"role", "content"}]``."""
    clean: List[Message] = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user")).lower().strip()
        content = item.get("content") or item.get("text") or item.get("message") or ""
        content = str(content).strip()
        if not content:
            continue
        if role in ("assistant", "model", "bot", "ai"):
            role = "assistant"
        elif role in ("system", "developer"):
            # Gemma 3 has no system role; such turns are merged into the prompt.
            role = "system"
        else:
            role = "user"
        clean.append({"role": role, "content": content})
    return clean


def build_gemma_prompt(
    prompt: str,
    system_prompt: Optional[str] = None,
    history: Optional[Sequence[Message]] = None,
) -> str:
    """Render the official **Gemma 3** chat template as a raw string.

    Gemma 3 has no dedicated ``system`` role: the system instruction is
    prepended to the first user turn, exactly as the reference chat template
    does.

    Example output::

        <bos><start_of_turn>user
        {system}

        {user}<end_of_turn>
        <start_of_turn>model
    """
    system_prompt = (system_prompt or "").strip()
    turns = _normalise_history(history)

    # Absorb any system turns coming from history into the system instruction.
    inline_system = [t["content"] for t in turns if t["role"] == "system"]
    turns = [t for t in turns if t["role"] != "system"]
    if inline_system:
        system_prompt = "\n\n".join(filter(None, [system_prompt, *inline_system]))

    turns.append({"role": "user", "content": str(prompt).strip()})

    parts: List[str] = ["<bos>"]
    first_user_done = False
    for turn in turns:
        role = "model" if turn["role"] == "assistant" else "user"
        content = turn["content"]
        if role == "user" and not first_user_done and system_prompt:
            content = f"{system_prompt}\n\n{content}"
            first_user_done = True
        elif role == "user":
            first_user_done = True
        parts.append(f"<start_of_turn>{role}\n{content}<end_of_turn>\n")
    parts.append("<start_of_turn>model\n")
    return "".join(parts)


def _build_openai_messages(
    prompt: str,
    system_prompt: Optional[str],
    history: Optional[Sequence[Message]],
) -> List[Message]:
    """Build an OpenAI-style message list.

    Gemma 3 does not support a system role, so the system instruction is folded
    into the first user message for maximum provider compatibility.
    """
    turns = [t for t in _normalise_history(history) if t["role"] != "system"]
    turns.append({"role": "user", "content": str(prompt).strip()})

    system_prompt = (system_prompt or "").strip()
    if system_prompt:
        for turn in turns:
            if turn["role"] == "user":
                turn["content"] = f"{system_prompt}\n\n{turn['content']}"
                break
    return turns


def _build_google_contents(
    prompt: str,
    history: Optional[Sequence[Message]],
) -> List[Dict[str, Any]]:
    """Build the ``contents`` array for the Generative Language API."""
    contents: List[Dict[str, Any]] = []
    for turn in _normalise_history(history):
        if turn["role"] == "system":
            continue
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": turn["content"]}]})
    contents.append({"role": "user", "parts": [{"text": str(prompt).strip()}]})
    return contents


def _strip_data_url(data: str) -> str:
    """Accept either a bare base64 string or a full ``data:`` URL."""
    data = (data or "").strip()
    if data.startswith("data:") and "," in data:
        data = data.split(",", 1)[1]
    return data.replace("\n", "").replace("\r", "")


def _provider_key(provider: str) -> str:
    """Return the configured API key for a provider name ('' when missing)."""
    return {
        "google": GOOGLE_API_KEY,
        "google-vision": GOOGLE_API_KEY,
        "groq": GROQ_API_KEY,
        "openrouter": OPENROUTER_API_KEY,
        "huggingface": HF_TOKEN,
    }.get(provider, "")


def _clean_output(text: str) -> str:
    """Strip stray Gemma control tokens that some providers echo back."""
    if not text:
        return ""
    for token in ("<end_of_turn>", "<start_of_turn>model", "<start_of_turn>user", "<bos>", "<eos>"):
        text = text.replace(token, "")
    return text.strip()


# --------------------------------------------------------------------------- #
# HTTP helper                                                                  #
# --------------------------------------------------------------------------- #


def _post_json(
    provider: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    payload: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, str]] = None,
    timeout: int = REQUEST_TIMEOUT,
) -> Any:
    """POST JSON with same-provider retries; raise :class:`ProviderError` on failure."""
    last_error: Optional[ProviderError] = None

    for attempt in range(1, MAX_RETRIES_PER_PROVIDER + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                params=params,
                timeout=timeout,
            )
        except requests.exceptions.Timeout:
            last_error = ProviderError(provider, f"timeout after {timeout}s")
        except requests.exceptions.RequestException as exc:
            last_error = ProviderError(provider, f"network error: {exc}")
        else:
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError:
                    raise ProviderError(provider, "response was not valid JSON", 200)

            snippet = (response.text or "")[:300].replace("\n", " ")
            last_error = ProviderError(
                provider, f"HTTP {response.status_code}: {snippet}", response.status_code
            )
            if response.status_code not in RETRYABLE_STATUS:
                raise last_error

        if attempt < MAX_RETRIES_PER_PROVIDER:
            sleep_for = RETRY_BACKOFF_SECONDS * attempt
            logger.warning(
                "%s attempt %s/%s failed (%s) - retrying in %.1fs",
                provider, attempt, MAX_RETRIES_PER_PROVIDER, last_error, sleep_for,
            )
            time.sleep(sleep_for)

    raise last_error or ProviderError(provider, "unknown failure")


# --------------------------------------------------------------------------- #
# Provider 1: Google AI Studio (Generative Language API)                       #
# --------------------------------------------------------------------------- #


def call_google_gemma(
    prompt: str,
    system_prompt: Optional[str] = None,
    history: Optional[Sequence[Message]] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Call Google AI Studio's Generative Language API (``gemma-3-27b-it``)."""
    return _call_google_model(
        GOOGLE_MODEL, prompt, system_prompt, history, temperature, max_tokens
    )


def _call_google_model(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history: Optional[Sequence[Message]] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Text-only Google AI Studio call against an explicit Gemma 3 model id."""
    if not GOOGLE_API_KEY:
        raise ProviderError("google", "GOOGLE_API_KEY is not set")

    url = GOOGLE_ENDPOINT.format(model=model)
    system_prompt = DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt

    # Gemma models on this API reject `systemInstruction`, so we inline it.
    contents = _build_google_contents(prompt, history)
    if system_prompt and contents:
        first = contents[0]["parts"][0]
        first["text"] = f"{system_prompt}\n\n{first['text']}"

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "topP": 0.95,
            "topK": 64,
        },
    }

    data = _post_json(
        "google",
        url,
        headers={"Content-Type": "application/json", "x-goog-api-key": GOOGLE_API_KEY},
        payload=payload,
    )

    if isinstance(data, dict) and data.get("promptFeedback", {}).get("blockReason"):
        raise ProviderError("google", f"blocked: {data['promptFeedback']['blockReason']}")

    try:
        candidates = data["candidates"]
        parts = candidates[0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError("google", f"unexpected response shape: {exc}") from exc

    text = _clean_output(text)
    if not text:
        raise ProviderError("google", "empty completion")
    return text


def call_google_vision(
    prompt: str,
    images: Sequence[Any],
    system_prompt: Optional[str] = None,
    history: Optional[Sequence[Message]] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    model: Optional[str] = None,
) -> str:
    """Call Gemma 3 multimodal on Google AI Studio with one or more images.

    ``images`` accepts either raw base64 strings or dicts shaped like
    ``{"data": <base64>, "mime_type": "image/jpeg"}``. Each is emitted into the
    ``parts`` array using Google's ``inlineData`` format::

        {"inlineData": {"data": <base64>, "mimeType": "image/jpeg"}}
    """
    if not GOOGLE_API_KEY:
        raise ProviderError("google-vision", "GOOGLE_API_KEY is not set")
    if not images:
        raise ProviderError("google-vision", "no image supplied")

    target_model = model or MODEL_MAP[VISION_MODE]["model"]
    system_prompt = DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt

    # Prior turns stay text-only; the image rides on the final user turn.
    contents = _build_google_contents(prompt or "Describe this image.", history)
    final_parts = contents[-1]["parts"]

    if system_prompt:
        first = contents[0]["parts"][0]
        first["text"] = f"{system_prompt}\n\n{first['text']}"

    for image in images:
        if isinstance(image, dict):
            data = image.get("data") or image.get("base64") or ""
            mime = image.get("mime_type") or image.get("mimeType") or "image/jpeg"
        else:
            data, mime = str(image), "image/jpeg"

        data = _strip_data_url(data)
        if not data:
            continue
        # Image first, then the question - Gemma attends better in this order.
        final_parts.insert(0, {"inlineData": {"data": data, "mimeType": mime}})

    if not any("inlineData" in part for part in final_parts):
        raise ProviderError("google-vision", "no decodable image data")

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "topP": 0.95,
            "topK": 64,
        },
    }

    data = _post_json(
        "google-vision",
        GOOGLE_ENDPOINT.format(model=target_model),
        headers={"Content-Type": "application/json", "x-goog-api-key": GOOGLE_API_KEY},
        payload=payload,
        timeout=VISION_TIMEOUT,
    )

    if isinstance(data, dict) and data.get("promptFeedback", {}).get("blockReason"):
        raise ProviderError("google-vision", f"blocked: {data['promptFeedback']['blockReason']}")

    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError("google-vision", f"unexpected response shape: {exc}") from exc

    text = _clean_output(text)
    if not text:
        raise ProviderError("google-vision", "empty completion")
    return text


# --------------------------------------------------------------------------- #
# Provider 2: Groq (OpenAI-compatible)                                         #
# --------------------------------------------------------------------------- #


def _call_openai_compatible(
    provider: str,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    system_prompt: Optional[str],
    history: Optional[Sequence[Message]],
    temperature: float,
    max_tokens: int,
    extra_headers: Optional[Dict[str, str]] = None,
) -> str:
    """Shared implementation for Groq / OpenRouter style chat-completions APIs."""
    if not api_key:
        raise ProviderError(provider, f"API key for {provider} is not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    headers.update(extra_headers or {})

    payload = {
        "model": model,
        "messages": _build_openai_messages(prompt, system_prompt, history),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    data = _post_json(provider, url, headers=headers, payload=payload)

    if isinstance(data, dict) and data.get("error"):
        err = data["error"]
        message = err.get("message") if isinstance(err, dict) else str(err)
        raise ProviderError(provider, f"api error: {message}")

    try:
        choice = data["choices"][0]
        text = choice.get("message", {}).get("content") or choice.get("text", "")
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(provider, f"unexpected response shape: {exc}") from exc

    text = _clean_output(text)
    if not text:
        raise ProviderError(provider, "empty completion")
    return text


def call_groq_gemma(
    prompt: str,
    system_prompt: Optional[str] = None,
    history: Optional[Sequence[Message]] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Call the Groq API with a Gemma 3 model."""
    return _call_openai_compatible(
        provider="groq",
        url=GROQ_ENDPOINT,
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        prompt=prompt,
        system_prompt=DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt,
        history=history,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# --------------------------------------------------------------------------- #
# Provider 3: OpenRouter                                                       #
# --------------------------------------------------------------------------- #


def call_openrouter_gemma(
    prompt: str,
    system_prompt: Optional[str] = None,
    history: Optional[Sequence[Message]] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Call OpenRouter with a Gemma 3 model (free tier by default)."""
    return _call_openai_compatible(
        provider="openrouter",
        url=OPENROUTER_ENDPOINT,
        api_key=OPENROUTER_API_KEY,
        model=OPENROUTER_MODEL,
        prompt=prompt,
        system_prompt=DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt,
        history=history,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_headers={
            "HTTP-Referer": OPENROUTER_SITE_URL,
            "X-Title": OPENROUTER_APP_NAME,
        },
    )


# --------------------------------------------------------------------------- #
# Provider 4: Hugging Face Inference API                                       #
# --------------------------------------------------------------------------- #


def call_huggingface_gemma(
    prompt: str,
    system_prompt: Optional[str] = None,
    history: Optional[Sequence[Message]] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Call the Hugging Face Inference API using the raw Gemma 3 chat template."""
    if not HF_TOKEN:
        raise ProviderError("huggingface", "HF_TOKEN is not set")

    templated = build_gemma_prompt(
        prompt,
        DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt,
        history,
    )

    payload = {
        "inputs": templated,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": max(temperature, 0.01),  # HF rejects temperature == 0
            "top_p": 0.95,
            "return_full_text": False,
            "stop": ["<end_of_turn>"],
        },
        "options": {"wait_for_model": True, "use_cache": False},
    }

    data = _post_json(
        "huggingface",
        HF_ENDPOINT.format(model=HF_MODEL),
        headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
        payload=payload,
    )

    text = ""
    if isinstance(data, list) and data:
        first = data[0]
        text = first.get("generated_text", "") if isinstance(first, dict) else str(first)
    elif isinstance(data, dict):
        if data.get("error"):
            raise ProviderError("huggingface", f"api error: {data['error']}")
        text = data.get("generated_text", "")

    # Some deployments echo the prompt back regardless of `return_full_text`.
    if text.startswith(templated):
        text = text[len(templated):]

    text = _clean_output(text)
    if not text:
        raise ProviderError("huggingface", "empty completion")
    return text


# --------------------------------------------------------------------------- #
# The router                                                                   #
# --------------------------------------------------------------------------- #

ProviderFn = Callable[..., str]

PROVIDER_CHAIN: List[Dict[str, Any]] = [
    {"name": "google", "model": GOOGLE_MODEL, "fn": call_google_gemma, "key": GOOGLE_API_KEY},
    {"name": "groq", "model": GROQ_MODEL, "fn": call_groq_gemma, "key": GROQ_API_KEY},
    {"name": "openrouter", "model": OPENROUTER_MODEL, "fn": call_openrouter_gemma, "key": OPENROUTER_API_KEY},
    {"name": "huggingface", "model": HF_MODEL, "fn": call_huggingface_gemma, "key": HF_TOKEN},
]

FALLBACK_MESSAGE = (
    "⚠️ All Gemma 3 providers are currently unavailable. "
    "Please check your API keys or try again in a moment."
)

VISION_FALLBACK_MESSAGE = (
    "⚠️ The Gemma 3 vision engine is unavailable right now. "
    "Please try sending the image again in a moment."
)


def available_providers() -> List[Dict[str, Any]]:
    """Return the provider chain with a boolean flag for configured API keys."""
    return [
        {"name": p["name"], "model": p["model"], "configured": bool(p["key"])}
        for p in PROVIDER_CHAIN
    ]


def smart_gemma_router_verbose(
    prompt: str,
    system_prompt: Optional[str] = None,
    history: Optional[Sequence[Message]] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    raise_on_failure: bool = True,
) -> RouterResult:
    """Run the fallback chain and return a :class:`RouterResult` with metadata."""
    if not prompt or not str(prompt).strip():
        raise ValueError("prompt must be a non-empty string")

    errors: Dict[str, str] = {}
    started = time.perf_counter()

    for provider in PROVIDER_CHAIN:
        name, model, fn, key = provider["name"], provider["model"], provider["fn"], provider["key"]

        if not key:
            errors[name] = "skipped: API key not configured"
            logger.debug("Skipping %s - no API key configured", name)
            continue

        logger.info("Routing prompt to %s (%s)", name, model)
        try:
            text = fn(
                prompt,
                system_prompt=system_prompt,
                history=history,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except ProviderError as exc:
            errors[name] = str(exc)
            logger.warning("Provider %s failed: %s", name, exc)
        except Exception as exc:  # noqa: BLE001 - never let one provider kill the chain
            errors[name] = f"unexpected error: {exc}"
            logger.exception("Unexpected error from provider %s", name)
        else:
            latency_ms = int((time.perf_counter() - started) * 1000)
            logger.info("Provider %s succeeded in %sms", name, latency_ms)
            return RouterResult(
                text=text, provider=name, model=model, latency_ms=latency_ms, errors=errors
            )

    logger.error("All Gemma 3 providers failed: %s", errors)
    if raise_on_failure:
        raise AllProvidersFailedError(errors)

    return RouterResult(
        text=FALLBACK_MESSAGE,
        provider="none",
        model="none",
        latency_ms=int((time.perf_counter() - started) * 1000),
        errors=errors,
    )


# --------------------------------------------------------------------------- #
# Capability-based router                                                      #
# --------------------------------------------------------------------------- #


def _dispatch(
    mode: str,
    prompt: str,
    system_prompt: Optional[str],
    history: Optional[Sequence[Message]],
    temperature: float,
    max_tokens: int,
    images: Optional[Sequence[Any]] = None,
) -> str:
    """Invoke the provider bound to ``mode``. Raises :class:`ProviderError`."""
    spec = MODEL_MAP[mode]
    provider, model = spec["provider"], spec["model"]

    if not _provider_key(provider):
        raise ProviderError(provider, f"API key for '{mode}' mode is not configured")

    # Gemma 3 on Google AI Studio handles both core and vision.
    if provider == "google":
        if images:
            return call_google_vision(
                prompt, images, system_prompt, history, temperature, max_tokens, model=model
            )
        return _call_google_model(
            model, prompt, system_prompt, history, temperature, max_tokens
        )

    if images:
        raise ProviderError(provider, f"'{mode}' mode cannot process images")

    if provider == "groq":
        url, extra = GROQ_ENDPOINT, None
    elif provider == "openrouter":
        url = OPENROUTER_ENDPOINT
        extra = {"HTTP-Referer": OPENROUTER_SITE_URL, "X-Title": OPENROUTER_APP_NAME}
    else:
        raise ProviderError(provider, f"unsupported provider for mode '{mode}'")

    return _call_openai_compatible(
        provider=provider,
        url=url,
        api_key=_provider_key(provider),
        model=model,
        prompt=prompt,
        system_prompt=DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt,
        history=history,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_headers=extra,
    )


def capability_router(
    prompt: str,
    mode: Optional[str] = None,
    images: Optional[Sequence[Any]] = None,
    system_prompt: Optional[str] = None,
    history: Optional[Sequence[Message]] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> RouterResult:
    """Route a request to the capability that fits the task.

    Resolution order:

    1. Any request carrying ``images`` is forced to ``vision`` (Gemma 3
       multimodal) regardless of the requested mode.
    2. Otherwise the requested mode is used, defaulting to ``core``.
    3. If that provider times out or errors, fall back to ``core`` (Gemma 3),
       then to the legacy provider chain as a final safety net.

    Never raises for provider outages - always returns a :class:`RouterResult`.
    """
    if not prompt and not images:
        raise ValueError("prompt must be a non-empty string")

    requested = normalise_mode(mode)
    # Images can only be understood by the multimodal engine.
    if images:
        if requested != VISION_MODE:
            logger.info("Images present - overriding mode '%s' with vision", requested)
        requested = VISION_MODE

    errors: Dict[str, str] = {}
    started = time.perf_counter()

    # Try the requested mode, then core as the universal fallback.
    attempts = [requested] if requested == CORE_MODE else [requested, CORE_MODE]

    for attempt_mode in attempts:
        spec = MODEL_MAP[attempt_mode]
        # Falling back to a text-only engine is pointless when images are present.
        if images and not spec["vision"]:
            continue

        logger.info("Routing to '%s' (%s · %s)", attempt_mode, spec["provider"], spec["model"])
        try:
            text = _dispatch(
                attempt_mode, prompt, system_prompt, history, temperature, max_tokens, images
            )
        except ProviderError as exc:
            errors[attempt_mode] = str(exc)
            logger.warning("Mode '%s' failed: %s", attempt_mode, exc)
        except Exception as exc:  # noqa: BLE001 - one mode must not kill the request
            errors[attempt_mode] = f"unexpected error: {exc}"
            logger.exception("Unexpected error in mode '%s'", attempt_mode)
        else:
            return RouterResult(
                text=text,
                provider=f"{attempt_mode}:{spec['provider']}",
                model=spec["model"],
                latency_ms=int((time.perf_counter() - started) * 1000),
                errors=errors,
            )

    # Text-only last resort: walk the original multi-provider chain.
    if not images:
        logger.warning("All capability modes failed - trying the legacy provider chain")
        try:
            result = smart_gemma_router_verbose(
                prompt,
                system_prompt=system_prompt,
                history=history,
                temperature=temperature,
                max_tokens=max_tokens,
                raise_on_failure=False,
            )
        except Exception as exc:  # noqa: BLE001
            errors["fallback_chain"] = str(exc)
        else:
            if result.provider != "none":
                result.errors = {**errors, **result.errors}
                return result
            errors.update(result.errors)

    logger.error("Capability routing failed for mode '%s': %s", requested, errors)
    return RouterResult(
        text=VISION_FALLBACK_MESSAGE if images else FALLBACK_MESSAGE,
        provider="none",
        model="none",
        latency_ms=int((time.perf_counter() - started) * 1000),
        errors=errors,
    )


def smart_gemma_router(
    prompt: str,
    system_prompt: Optional[str] = None,
    history: Optional[Sequence[Message]] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Main entry point: return Gemma 3's answer, trying every provider in order.

    Never raises for provider outages - returns :data:`FALLBACK_MESSAGE` instead,
    so callers (bot / API) always have something to show the user.
    """
    try:
        result = smart_gemma_router_verbose(
            prompt,
            system_prompt=system_prompt,
            history=history,
            temperature=temperature,
            max_tokens=max_tokens,
            raise_on_failure=False,
        )
        return result.text
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 - defensive last line of protection
        logger.exception("smart_gemma_router crashed: %s", exc)
        return FALLBACK_MESSAGE


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    print("Providers:", available_providers())
    print(smart_gemma_router("Say hello in one short sentence."))
