"""
ai_router.py - AI routing and system prompt management
This file centralizes how we call the AI model, inject system prompt, and enforce guardrails.
Adapt the generate_ai_response function to your preferred model client if you don't use OpenAI.
Environment:
  - OPENAI_API_KEY
  - CURRENT_MODEL (optional; default used if missing)
"""

import os
import re
import openai  # If you use a different client, adapt this code.

# Configure OpenAI key from environment; do not ever print or log it.
openai.api_key = os.environ.get("OPENAI_API_KEY")

DEFAULT_MODEL = os.environ.get("CURRENT_MODEL", "gpt-4o-mini")

# Regex to detect attempts to extract secrets or environment details
_SECRETS_PATTERNS = [
    re.compile(r"(api[_-]?key|secret|token|password|env|environment variable)", re.IGNORECASE),
    re.compile(r"(show me your keys|give (me|us) your secrets|what is your api key)", re.IGNORECASE),
    re.compile(r"(server structure|routing logic|internal architecture|source code for .*server)", re.IGNORECASE),
]

def is_secret_request(user_text: str) -> bool:
    """
    Heuristic: detect if the user is asking for secrets, keys, or internal details.
    """
    for pat in _SECRETS_PATTERNS:
        if pat.search(user_text or ""):
            return True
    return False


def get_system_prompt(model_name: str) -> str:
    """
    Compose a system prompt that:
      - Sets a warm, polite, friendly persona.
      - Enforces strict security guardrails (never reveal keys, env vars, server internals).
      - Makes the model aware of which model name is currently in use.
    """
    guardrails = (
        "You are a warm, polite, and friendly AI assistant. "
        "Always be helpful, concise, and kind.\n\n"
        "Security guardrails (MUST follow):\n"
        "- Under NO circumstances reveal API keys, environment variables, secrets, or private server configuration.\n"
        "- If asked for secrets, keys, or internal implementation details (API keys, deployment tokens, environment variables, server routing logic, etc.), refuse politely with the message: "
        "\"I’m sorry, but I can’t share secrets, API keys, or private configuration.\"\n"
        "- Do not output private file contents unless explicitly allowed by the server owner. Always prioritize safety and privacy.\n\n"
    )
    model_info = f"Current runtime model name: {model_name}\n"
    persona = "Persona: Respond in a warm, polite, and friendly tone.\n"
    return guardrails + model_info + persona


def generate_ai_response(user_text: str, model_name: str = None) -> str:
    """
    Send message(s) to the AI model with the injected system prompt.
    Returns a string reply.

    If you use a different API, replace the openai.ChatCompletion code with the appropriate call.
    """
    if model_name is None:
        model_name = DEFAULT_MODEL

    # If the user is explicitly trying to extract secrets, refuse immediately.
    if is_secret_request(user_text):
        return "I’m sorry, but I can’t share secrets, API keys, or private configuration."

    system_prompt = get_system_prompt(model_name)

    # Build chat messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text}
    ]

    # Call the model (OpenAI example). Wrap in try-except to avoid leaking error details.
    try:
        resp = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        )
        # Extract text safely
        return resp.choices[0].message.content.strip()
    except Exception:
        # On failure, give a user-friendly message (do NOT leak exceptions, API keys, or internals)
        return "Sorry — I had trouble contacting the AI service. Please try again later."


def get_current_model() -> str:
    return os.environ.get("CURRENT_MODEL", DEFAULT_MODEL)
