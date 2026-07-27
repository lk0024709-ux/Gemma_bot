"""
bot_handler.py
===============
Telegram bot layer built on **pyTelegramBotAPI** (``telebot``).

* Every text message is routed through :func:`ai_router.smart_gemma_router`.
* Short-term per-chat conversation memory is kept in RAM.
* Every exchange is archived to the private Telegram channel via ``tg_db``.
* Polling runs in a **daemon background thread** so it never blocks FastAPI.

Public API
----------
``start_bot_thread() -> threading.Thread | None``
``stop_bot() -> None``
``get_bot_status() -> dict``
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

from dotenv import load_dotenv

from ai_router import DEFAULT_SYSTEM_PROMPT, smart_gemma_router_verbose
from tg_db import log_interaction

load_dotenv()

logger = logging.getLogger(__name__)

try:
    import telebot
    from telebot.apihelper import ApiTelegramException

    TELEBOT_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency not installed
    telebot = None  # type: ignore[assignment]
    ApiTelegramException = Exception  # type: ignore[assignment, misc]
    TELEBOT_AVAILABLE = False
    logger.warning("pyTelegramBotAPI is not installed - the Telegram bot is disabled")

# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ENABLE_TELEGRAM_BOT = os.getenv("ENABLE_TELEGRAM_BOT", "true").lower() in ("1", "true", "yes")
LOG_TO_CHANNEL = os.getenv("LOG_TO_CHANNEL", "true").lower() in ("1", "true", "yes")

HISTORY_TURNS = int(os.getenv("BOT_HISTORY_TURNS", "6"))  # user+assistant messages kept
POLLING_TIMEOUT = int(os.getenv("BOT_POLLING_TIMEOUT", "30"))
POLLING_RESTART_DELAY = int(os.getenv("BOT_RESTART_DELAY", "10"))
TELEGRAM_MSG_LIMIT = 4096

WELCOME_TEXT = (
    "👋 *Gemma 3 Neuro-System online*\n\n"
    "Send me any message and I'll answer using the Gemma 3 model family "
    "with automatic multi-provider fallback.\n\n"
    "*Commands*\n"
    "/start - show this message\n"
    "/help - usage help\n"
    "/reset - clear our conversation memory\n"
    "/status - show provider & bot status"
)

HELP_TEXT = (
    "🧠 *How to use*\n\n"
    "Just type a question. I keep a short rolling memory of our chat so "
    "follow-ups work naturally.\n\n"
    "Use /reset to wipe that memory, /status to see which AI provider is live."
)

# --------------------------------------------------------------------------- #
# State                                                                        #
# --------------------------------------------------------------------------- #

bot: Optional["telebot.TeleBot"] = None
_bot_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_lock = threading.Lock()

# chat_id -> rolling deque of {"role", "content"}
_conversations: Dict[int, Deque[Dict[str, str]]] = defaultdict(
    lambda: deque(maxlen=max(HISTORY_TURNS, 2))
)

_stats: Dict[str, Any] = {
    "started_at": None,
    "messages_handled": 0,
    "errors": 0,
    "last_provider": None,
    "running": False,
}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _chunk(text: str, limit: int = TELEGRAM_MSG_LIMIT) -> List[str]:
    """Split a long reply into Telegram-sized chunks on paragraph/line breaks."""
    text = text or ""
    if len(text) <= limit:
        return [text]

    chunks: List[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        window = text[:limit]
        split_at = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "))
        if split_at < limit // 2:
            split_at = limit
        chunks.append(text[:split_at].rstrip())
        text = text[split_at:].lstrip()
    return chunks


def _safe_reply(message: Any, text: str) -> None:
    """Reply to a Telegram message, chunking and degrading gracefully."""
    if bot is None:
        return
    for part in _chunk(text):
        try:
            bot.reply_to(message, part, parse_mode="Markdown")
        except ApiTelegramException as exc:
            logger.warning("Markdown reply failed (%s) - retrying as plain text", exc)
            try:
                bot.reply_to(message, part)
            except Exception as inner:  # noqa: BLE001
                logger.error("Failed to deliver reply: %s", inner)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to deliver reply: %s", exc)


def _history_for(chat_id: int) -> List[Dict[str, str]]:
    return list(_conversations[chat_id])


def _remember(chat_id: int, role: str, content: str) -> None:
    _conversations[chat_id].append({"role": role, "content": content})


def _archive(message: Any, prompt: str, response: str, provider: Optional[str]) -> None:
    """Fire-and-forget archive of the exchange to the Telegram channel DB."""
    if not LOG_TO_CHANNEL:
        return

    def _worker() -> None:
        try:
            log_interaction(
                user_id=getattr(message.from_user, "id", None),
                username=getattr(message.from_user, "username", None),
                prompt=prompt,
                response=response,
                source="telegram",
                provider=provider,
                extra={"chat_id": message.chat.id},
            )
        except Exception as exc:  # noqa: BLE001 - logging must never break the bot
            logger.error("Channel archive failed: %s", exc)

    threading.Thread(target=_worker, daemon=True, name="tg-db-archive").start()


# --------------------------------------------------------------------------- #
# Bot construction & handlers                                                  #
# --------------------------------------------------------------------------- #


def create_bot() -> Optional["telebot.TeleBot"]:
    """Instantiate the ``TeleBot`` and register all handlers."""
    global bot

    if not TELEBOT_AVAILABLE:
        logger.error("Cannot create bot: pyTelegramBotAPI is not installed")
        return None
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("Cannot create bot: TELEGRAM_BOT_TOKEN is not set")
        return None

    bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, parse_mode=None, threaded=True)

    @bot.message_handler(commands=["start"])
    def handle_start(message: Any) -> None:
        _conversations.pop(message.chat.id, None)
        _safe_reply(message, WELCOME_TEXT)

    @bot.message_handler(commands=["help"])
    def handle_help(message: Any) -> None:
        _safe_reply(message, HELP_TEXT)

    @bot.message_handler(commands=["reset"])
    def handle_reset(message: Any) -> None:
        _conversations.pop(message.chat.id, None)
        _safe_reply(message, "🧹 Conversation memory cleared.")

    @bot.message_handler(commands=["status"])
    def handle_status(message: Any) -> None:
        from ai_router import available_providers  # local import to avoid cycles

        lines = ["📊 *Neuro-System status*", ""]
        for provider in available_providers():
            mark = "✅" if provider["configured"] else "⛔"
            lines.append(f"{mark} `{provider['name']}` - {provider['model']}")
        lines += [
            "",
            f"Messages handled: {_stats['messages_handled']}",
            f"Errors: {_stats['errors']}",
            f"Last provider used: {_stats['last_provider'] or 'n/a'}",
        ]
        _safe_reply(message, "\n".join(lines))

    @bot.message_handler(content_types=["text"])
    def handle_text(message: Any) -> None:
        """Route a plain user message through the Gemma 3 fallback chain."""
        chat_id = message.chat.id
        prompt = (message.text or "").strip()
        if not prompt:
            return

        try:
            bot.send_chat_action(chat_id, "typing")
        except Exception:  # noqa: BLE001 - purely cosmetic
            pass

        try:
            result = smart_gemma_router_verbose(
                prompt,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                history=_history_for(chat_id),
                raise_on_failure=False,
            )
            answer, provider = result.text, result.provider
            _stats["last_provider"] = provider
            _stats["messages_handled"] += 1
        except Exception as exc:  # noqa: BLE001
            _stats["errors"] += 1
            logger.exception("Failed to generate a reply: %s", exc)
            _safe_reply(message, "⚠️ Something went wrong while thinking. Please try again.")
            return

        _remember(chat_id, "user", prompt)
        _remember(chat_id, "assistant", answer)
        _safe_reply(message, answer)
        _archive(message, prompt, answer, provider)

    @bot.message_handler(
        content_types=["photo", "document", "audio", "voice", "video", "sticker"]
    )
    def handle_unsupported(message: Any) -> None:
        _safe_reply(message, "📎 I can only process text messages right now.")

    logger.info("Telegram bot handlers registered")
    return bot


# --------------------------------------------------------------------------- #
# Polling loop / thread management                                             #
# --------------------------------------------------------------------------- #


def _polling_loop() -> None:
    """Resilient infinite polling loop - restarts on transient failures."""
    global bot

    if bot is None and create_bot() is None:
        _stats["running"] = False
        return

    _stats["running"] = True
    _stats["started_at"] = time.time()
    logger.info("Telegram polling loop started")

    # Drop any webhook so long polling is allowed.
    try:
        bot.remove_webhook()
    except Exception as exc:  # noqa: BLE001
        logger.debug("remove_webhook failed (safe to ignore): %s", exc)

    while not _stop_event.is_set():
        try:
            bot.infinity_polling(
                timeout=POLLING_TIMEOUT,
                long_polling_timeout=POLLING_TIMEOUT,
                skip_pending=True,
                logger_level=logging.WARNING,
            )
        except Exception as exc:  # noqa: BLE001 - keep the bot alive no matter what
            _stats["errors"] += 1
            logger.error("Polling crashed: %s - restarting in %ss", exc, POLLING_RESTART_DELAY)
            if _stop_event.wait(POLLING_RESTART_DELAY):
                break
        else:
            # infinity_polling returned cleanly -> stop requested.
            break

    _stats["running"] = False
    logger.info("Telegram polling loop stopped")


def start_bot_thread() -> Optional[threading.Thread]:
    """Start Telegram polling in a daemon thread. Safe to call more than once."""
    global _bot_thread

    with _lock:
        if not ENABLE_TELEGRAM_BOT:
            logger.info("Telegram bot disabled via ENABLE_TELEGRAM_BOT=false")
            return None
        if not TELEGRAM_BOT_TOKEN:
            logger.warning("Telegram bot not started: TELEGRAM_BOT_TOKEN is missing")
            return None
        if not TELEBOT_AVAILABLE:
            logger.warning("Telegram bot not started: pyTelegramBotAPI is missing")
            return None
        if _bot_thread and _bot_thread.is_alive():
            logger.info("Telegram bot thread is already running")
            return _bot_thread

        _stop_event.clear()
        _bot_thread = threading.Thread(
            target=_polling_loop, name="telegram-bot-polling", daemon=True
        )
        _bot_thread.start()
        logger.info("Telegram bot thread launched")
        return _bot_thread


def stop_bot() -> None:
    """Signal the polling loop to stop and wait briefly for the thread to exit."""
    global _bot_thread

    _stop_event.set()
    if bot is not None:
        try:
            bot.stop_polling()
        except Exception as exc:  # noqa: BLE001
            logger.debug("stop_polling raised: %s", exc)

    if _bot_thread and _bot_thread.is_alive():
        _bot_thread.join(timeout=5)
    _bot_thread = None
    _stats["running"] = False
    logger.info("Telegram bot stopped")


def get_bot_status() -> Dict[str, Any]:
    """Return a snapshot of bot health for the ``/health`` endpoint."""
    uptime = int(time.time() - _stats["started_at"]) if _stats["started_at"] else 0
    return {
        "enabled": ENABLE_TELEGRAM_BOT,
        "telebot_installed": TELEBOT_AVAILABLE,
        "token_configured": bool(TELEGRAM_BOT_TOKEN),
        "running": bool(_bot_thread and _bot_thread.is_alive() and _stats["running"]),
        "uptime_seconds": uptime,
        "messages_handled": _stats["messages_handled"],
        "errors": _stats["errors"],
        "last_provider": _stats["last_provider"],
        "active_conversations": len(_conversations),
    }


if __name__ == "__main__":  # pragma: no cover - run the bot standalone
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    create_bot()
    _polling_loop()
