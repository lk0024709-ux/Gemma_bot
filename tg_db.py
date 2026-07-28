"""
tg_db.py
=========
Use a **private Telegram channel** as an append-only JSON database / log store.

Why: a private channel is free, replicated, searchable and gives you a
permanent ``message_id`` primary key for every record - perfect for a
lightweight "neuro-system" memory layer without provisioning a real DB.

Records are written as::

    #memory #chat
    ```json
    { ... }
    ```

Public API
----------
``save_memory_to_channel(channel_id, bot_token, json_data) -> int | None``
``save_memory(json_data, tags=None) -> int | None``          (env-configured)
``read_memory(message_id) -> dict | None``
``edit_memory(message_id, json_data) -> bool``
``delete_memory(message_id) -> bool``
``log_interaction(user_id, username, prompt, response, ...) -> int | None``
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID", "").strip()

TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}/{method}"
REQUEST_TIMEOUT = int(os.getenv("TG_DB_TIMEOUT", "20"))
MAX_RETRIES = int(os.getenv("TG_DB_MAX_RETRIES", "3"))
RETRY_BACKOFF_SECONDS = float(os.getenv("TG_DB_RETRY_BACKOFF", "1.5"))

# Telegram hard-caps a text message at 4096 UTF-16 code units.
TELEGRAM_MAX_MESSAGE_LEN = 4096
# Room for the fenced code block, tags and the truncation notice.
JSON_PAYLOAD_BUDGET = 3500


class TelegramDBError(RuntimeError):
    """Raised for unrecoverable Telegram Bot API errors."""


# --------------------------------------------------------------------------- #
# Internals                                                                    #
# --------------------------------------------------------------------------- #


def _api_call(
    method: str,
    payload: Dict[str, Any],
    bot_token: Optional[str] = None,
    timeout: int = REQUEST_TIMEOUT,
) -> Dict[str, Any]:
    """Call a Telegram Bot API method with retries and unified error handling."""
    token = (bot_token or TELEGRAM_BOT_TOKEN).strip()
    if not token:
        raise TelegramDBError("TELEGRAM_BOT_TOKEN is not configured")

    url = TELEGRAM_API_BASE.format(token=token, method=method)
    last_error = "unknown error"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            last_error = f"network error: {exc}"
        else:
            try:
                data = response.json()
            except ValueError:
                data = {}

            if response.status_code == 200 and data.get("ok"):
                return data.get("result", {})

            last_error = (
                f"HTTP {response.status_code}: "
                f"{data.get('description') or (response.text or '')[:200]}"
            )

            # Honour Telegram's flood-control hint.
            retry_after = (data.get("parameters") or {}).get("retry_after")
            if retry_after:
                logger.warning("Telegram flood control: sleeping %ss", retry_after)
                time.sleep(float(retry_after) + 0.5)
                continue

            # 4xx other than 429 will never succeed on retry.
            if 400 <= response.status_code < 500 and response.status_code != 429:
                raise TelegramDBError(f"{method} failed -> {last_error}")

        if attempt < MAX_RETRIES:
            sleep_for = RETRY_BACKOFF_SECONDS * attempt
            logger.warning(
                "Telegram %s attempt %s/%s failed (%s) - retrying in %.1fs",
                method, attempt, MAX_RETRIES, last_error, sleep_for,
            )
            time.sleep(sleep_for)

    raise TelegramDBError(f"{method} failed after {MAX_RETRIES} attempts -> {last_error}")


def _serialise(json_data: Any) -> str:
    """Serialise any payload to pretty JSON, tolerating non-serialisable values."""
    if isinstance(json_data, str):
        # Already a JSON string? Normalise it; otherwise wrap it.
        try:
            json_data = json.loads(json_data)
        except (ValueError, TypeError):
            json_data = {"raw": json_data}
    try:
        return json.dumps(json_data, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError) as exc:
        logger.warning("Falling back to repr() for unserialisable payload: %s", exc)
        return json.dumps({"unserialisable": repr(json_data)}, ensure_ascii=False, indent=2)


def _format_record(json_data: Any, tags: Optional[Iterable[str]] = None) -> str:
    """Render a Telegram message body containing the JSON record."""
    body = _serialise(json_data)
    truncated = False
    if len(body) > JSON_PAYLOAD_BUDGET:
        body = body[:JSON_PAYLOAD_BUDGET]
        truncated = True

    tag_line = " ".join(f"#{str(t).lstrip('#')}" for t in (tags or ["memory"]))
    message = f"{tag_line}\n```json\n{body}\n```"
    if truncated:
        message += "\n_(payload truncated to fit Telegram's message limit)_"
    return message[:TELEGRAM_MAX_MESSAGE_LEN]


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #


def save_memory_to_channel(
    channel_id: str,
    bot_token: str,
    json_data: Any,
    tags: Optional[Iterable[str]] = None,
    silent: bool = True,
    raise_on_error: bool = False,
) -> Optional[int]:
    """Save a JSON record to a private Telegram channel.

    Parameters
    ----------
    channel_id:
        Target channel, e.g. ``-1001234567890`` or ``@my_private_log``.
    bot_token:
        Bot token; the bot must be an administrator of the channel.
    json_data:
        Any JSON-serialisable object (dict, list, str, ...).
    tags:
        Hashtags prepended to the message for Telegram-side search.
        Defaults to ``["memory"]``.
    silent:
        Send without a notification sound.
    raise_on_error:
        Re-raise :class:`TelegramDBError` instead of returning ``None``.

    Returns
    -------
    int | None
        The Telegram ``message_id`` (your record's primary key), or ``None``
        on failure when ``raise_on_error`` is ``False``.
    """
    channel_id = str(channel_id or TELEGRAM_CHANNEL_ID).strip()
    if not channel_id:
        message = "TELEGRAM_CHANNEL_ID is not configured - skipping memory write"
        logger.warning(message)
        if raise_on_error:
            raise TelegramDBError(message)
        return None

    payload = {
        "chat_id": channel_id,
        "text": _format_record(json_data, tags),
        "parse_mode": "Markdown",
        "disable_notification": silent,
        "disable_web_page_preview": True,
    }

    try:
        result = _api_call("sendMessage", payload, bot_token=bot_token)
    except TelegramDBError as exc:
        logger.error("Failed to save memory to channel %s: %s", channel_id, exc)
        # Markdown parse failures are common with user text - retry as plain text.
        if "parse" in str(exc).lower() or "entity" in str(exc).lower():
            try:
                payload.pop("parse_mode", None)
                result = _api_call("sendMessage", payload, bot_token=bot_token)
            except TelegramDBError as retry_exc:
                logger.error("Plain-text retry also failed: %s", retry_exc)
                if raise_on_error:
                    raise
                return None
        elif raise_on_error:
            raise
        else:
            return None

    message_id = result.get("message_id")
    logger.info("Saved memory to channel %s as message_id=%s", channel_id, message_id)
    return message_id


def save_memory(
    json_data: Any,
    tags: Optional[Iterable[str]] = None,
    raise_on_error: bool = False,
) -> Optional[int]:
    """Convenience wrapper using ``TELEGRAM_CHANNEL_ID`` / ``TELEGRAM_BOT_TOKEN``."""
    return save_memory_to_channel(
        channel_id=TELEGRAM_CHANNEL_ID,
        bot_token=TELEGRAM_BOT_TOKEN,
        json_data=json_data,
        tags=tags,
        raise_on_error=raise_on_error,
    )


def log_interaction(
    user_id: Any,
    username: Optional[str],
    prompt: str,
    response: str,
    source: str = "telegram",
    provider: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Persist one prompt/response exchange as a structured memory record."""
    record: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "user_id": user_id,
        "username": username,
        "provider": provider,
        "prompt": prompt,
        "response": response,
    }
    if extra:
        record.update(extra)
    return save_memory(record, tags=["memory", source])


def read_memory(message_id: int, channel_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Best-effort read of a stored record by ``message_id``.

    Implemented via ``forwardMessage`` into the same channel (the Bot API has no
    "get message by id" method), then parsing the JSON code block. Returns
    ``None`` if the record cannot be retrieved or parsed.
    """
    chat_id = str(channel_id or TELEGRAM_CHANNEL_ID).strip()
    if not chat_id:
        logger.warning("read_memory: no channel configured")
        return None

    try:
        result = _api_call(
            "forwardMessage",
            {
                "chat_id": chat_id,
                "from_chat_id": chat_id,
                "message_id": int(message_id),
                "disable_notification": True,
            },
        )
    except (TelegramDBError, ValueError) as exc:
        logger.error("read_memory(%s) failed: %s", message_id, exc)
        return None

    text = result.get("text", "")
    # Clean up the temporary forwarded copy.
    try:
        _api_call("deleteMessage", {"chat_id": chat_id, "message_id": result.get("message_id")})
    except TelegramDBError:
        logger.debug("Could not delete temporary forwarded message")

    return _extract_json(text)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull the JSON object out of a stored record's message text."""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except ValueError as exc:
        logger.warning("Could not parse stored JSON record: %s", exc)
        return None


def edit_memory(
    message_id: int,
    json_data: Any,
    channel_id: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
) -> bool:
    """Overwrite an existing record in place. Returns ``True`` on success."""
    chat_id = str(channel_id or TELEGRAM_CHANNEL_ID).strip()
    if not chat_id:
        return False
    try:
        _api_call(
            "editMessageText",
            {
                "chat_id": chat_id,
                "message_id": int(message_id),
                "text": _format_record(json_data, tags),
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
        )
        return True
    except (TelegramDBError, ValueError) as exc:
        logger.error("edit_memory(%s) failed: %s", message_id, exc)
        return False


def delete_memory(message_id: int, channel_id: Optional[str] = None) -> bool:
    """Delete a record by ``message_id``. Returns ``True`` on success."""
    chat_id = str(channel_id or TELEGRAM_CHANNEL_ID).strip()
    if not chat_id:
        return False
    try:
        _api_call("deleteMessage", {"chat_id": chat_id, "message_id": int(message_id)})
        return True
    except (TelegramDBError, ValueError) as exc:
        logger.error("delete_memory(%s) failed: %s", message_id, exc)
        return False


def health_check() -> Dict[str, Any]:
    """Verify the bot token and channel access. Used by ``/health``."""
    status: Dict[str, Any] = {
        "bot_token_configured": bool(TELEGRAM_BOT_TOKEN),
        "channel_configured": bool(TELEGRAM_CHANNEL_ID),
        "bot_username": None,
        "channel_reachable": False,
        "error": None,
    }
    if not TELEGRAM_BOT_TOKEN:
        status["error"] = "TELEGRAM_BOT_TOKEN missing"
        return status

    try:
        me = _api_call("getMe", {})
        status["bot_username"] = me.get("username")
    except TelegramDBError as exc:
        status["error"] = str(exc)
        return status

    if TELEGRAM_CHANNEL_ID:
        try:
            _api_call("getChat", {"chat_id": TELEGRAM_CHANNEL_ID})
            status["channel_reachable"] = True
        except TelegramDBError as exc:
            status["error"] = str(exc)

    return status


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    print("Health:", health_check())
    mid = save_memory({"event": "smoke_test", "ok": True}, tags=["memory", "test"])
    print("Saved message_id:", mid)
