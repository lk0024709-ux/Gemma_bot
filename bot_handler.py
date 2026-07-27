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

import base64
import logging
import os
import threading
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

from dotenv import load_dotenv

from ai_router import (
    CORE_MODE,
    DEFAULT_MODE,
    DEFAULT_SYSTEM_PROMPT,
    MODEL_MAP,
    VISION_MODE,
    capability_router,
    normalise_mode,
)
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

# --------------------------------------------------------------------------- #
# Force-subscribe / channel membership gate                                    #
# --------------------------------------------------------------------------- #

# Channel users must join before they can talk to the AI (@handle or -100... id).
REQUIRED_CHANNEL_ID = os.getenv("REQUIRED_CHANNEL_ID", "").strip()
CHANNEL_INVITE_LINK = os.getenv("CHANNEL_INVITE_LINK", "").strip()

# Membership results are cached briefly so we don't hit the Telegram API on
# every single message (keeps replies fast and avoids rate limits).
MEMBERSHIP_CACHE_TTL = int(os.getenv("MEMBERSHIP_CACHE_TTL", "300"))  # seconds
# Negative results expire quickly so a user who just joined isn't stuck waiting.
MEMBERSHIP_NEGATIVE_TTL = int(os.getenv("MEMBERSHIP_NEGATIVE_TTL", "20"))

# If the membership check itself fails (bot not admin, channel not found,
# Telegram outage) should we let the user through? Default: NO (fail closed),
# because failing open silently disables the whole gate.
FORCE_SUB_FAIL_OPEN = os.getenv("FORCE_SUB_FAIL_OPEN", "false").lower() in ("1", "true", "yes")

# Statuses that count as "in the channel".
MEMBER_STATUSES = frozenset({"member", "administrator", "creator"})

ACCESS_DENIED_TEXT = "🔒 Access Denied! Please join our channel to use this AI."

_membership_cache: Dict[int, tuple] = {}  # user_id -> (is_member, expires_at)
_membership_lock = threading.Lock()

WELCOME_TEXT = (
    "👋 *Gemma 3 Neuro-System online*\n\n"
    "Send me any message and I'll answer using the Gemma 3 model family "
    "with automatic multi-provider fallback.\n\n"
    "*Modes*\n"
    "/core - 🌐 Gemma 3 balanced engine (default)\n"
    "/flash - ⚡ fastest replies\n"
    "/reasoning - 🧩 step-by-step logic\n"
    "/pro - 🎓 heavy coding & expert tasks\n"
    "📷 Send a photo for Gemma 3 vision.\n\n"
    "*Commands*\n"
    "/mode - show the active engine\n"
    "/reset - clear our conversation memory\n"
    "/status - show provider & bot status"
)

HELP_TEXT = (
    "🧠 *How to use*\n\n"
    "Just type a question. I keep a short rolling memory of our chat so "
    "follow-ups work naturally.\n\n"
    "Switch engines any time with /core, /flash, /reasoning or /pro, and send a "
    "photo to use Gemma 3 vision. /mode shows what's active.\n\n"
    "Use /reset to wipe that memory, /status to see which AI provider is live."
)

# --------------------------------------------------------------------------- #
# State                                                                        #
# --------------------------------------------------------------------------- #

bot: Optional["telebot.TeleBot"] = None
# Fallback instance used by the REST API when polling is disabled.
_api_bot: Optional["telebot.TeleBot"] = None
_bot_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_lock = threading.Lock()

# chat_id -> rolling deque of {"role", "content"}
_conversations: Dict[int, Deque[Dict[str, str]]] = defaultdict(
    lambda: deque(maxlen=max(HISTORY_TURNS, 2))
)

# chat_id -> selected capability. Defaults to `core` (Gemma 3 balanced engine).
user_task_mode: Dict[int, str] = defaultdict(lambda: DEFAULT_MODE)

# Largest Telegram photo we will download and base64 into a Gemma 3 request.
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(4 * 1024 * 1024)))  # 4 MB

_stats: Dict[str, Any] = {
    "started_at": None,
    "messages_handled": 0,
    "images_handled": 0,
    "errors": 0,
    "blocked": 0,  # requests rejected by the force-subscribe gate
    "last_provider": None,
    "running": False,
}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _guess_mime(path: str) -> str:
    """Map a Telegram file path extension to an image MIME type."""
    ext = (path.rsplit(".", 1)[-1] if "." in path else "").lower()
    return {
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
        "heic": "image/heic",
        "heif": "image/heif",
    }.get(ext, "image/jpeg")


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
# Force-subscribe helpers                                                      #
# --------------------------------------------------------------------------- #


def force_sub_enabled() -> bool:
    """The gate is only active once a required channel is configured."""
    return bool(REQUIRED_CHANNEL_ID)


def _cache_get(user_id: int) -> Optional[bool]:
    with _membership_lock:
        entry = _membership_cache.get(user_id)
        if not entry:
            return None
        is_member, expires_at = entry
        if time.time() >= expires_at:
            _membership_cache.pop(user_id, None)
            return None
        return is_member


def _cache_set(user_id: int, is_member: bool) -> None:
    ttl = MEMBERSHIP_CACHE_TTL if is_member else MEMBERSHIP_NEGATIVE_TTL
    with _membership_lock:
        # Cheap eviction so the dict can't grow without bound.
        if len(_membership_cache) > 10_000:
            now = time.time()
            for uid, (_, exp) in list(_membership_cache.items()):
                if now >= exp:
                    _membership_cache.pop(uid, None)
        _membership_cache[user_id] = (is_member, time.time() + ttl)


def invalidate_membership(user_id: int) -> None:
    """Drop a cached verdict (e.g. right after the user taps 'I've joined')."""
    with _membership_lock:
        _membership_cache.pop(user_id, None)


def is_user_member(
    bot_instance: Any,
    user_id: int,
    channel_id: Optional[str] = None,
    use_cache: bool = True,
) -> bool:
    """Return ``True`` if ``user_id`` belongs to ``channel_id``.

    Uses ``bot.get_chat_member()`` and accepts the statuses ``member``,
    ``administrator`` and ``creator``. ``left`` and ``kicked`` are rejected;
    ``restricted`` is accepted only when the user is still subscribed
    (``is_member`` is true), which is how Telegram represents a muted member.

    Any failure (bot is not an admin of the channel, wrong channel id, network
    error) is swallowed and resolved via :data:`FORCE_SUB_FAIL_OPEN` so a
    misconfiguration can never crash a request handler.
    """
    channel = (channel_id or REQUIRED_CHANNEL_ID or "").strip()

    # No channel configured -> the gate is disabled, everybody is allowed.
    if not channel:
        return True
    if bot_instance is None:
        logger.warning("Membership check skipped: no bot instance available")
        return FORCE_SUB_FAIL_OPEN
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        logger.warning("Membership check got an invalid user_id: %r", user_id)
        return False

    if use_cache:
        cached = _cache_get(user_id)
        if cached is not None:
            return cached

    try:
        member = bot_instance.get_chat_member(channel, user_id)
        status = getattr(member, "status", "") or ""
        is_member = status in MEMBER_STATUSES
        # A "restricted" user is still in the channel unless is_member is False.
        if not is_member and status == "restricted":
            is_member = bool(getattr(member, "is_member", False))
    except ApiTelegramException as exc:
        description = str(getattr(exc, "description", exc) or exc).lower()
        # "user not found" / "participant not found" = a definitive "not a member",
        # not an infrastructure failure, so don't fail-open on it.
        if "not found" in description:
            logger.debug("User %s is not a participant of %s", user_id, channel)
            _cache_set(user_id, False)
            return False
        logger.error(
            "Membership check failed for user %s in %s: %s "
            "(is the bot an administrator of the channel?)",
            user_id, channel, exc,
        )
        return FORCE_SUB_FAIL_OPEN
    except Exception as exc:  # noqa: BLE001 - never propagate into a handler
        logger.error("Unexpected membership check error for user %s: %s", user_id, exc)
        return FORCE_SUB_FAIL_OPEN

    _cache_set(user_id, is_member)
    logger.debug("User %s membership in %s: %s", user_id, channel, is_member)
    return is_member


def build_join_markup() -> Optional[Any]:
    """Inline keyboard with a 'Join channel' button and a re-check button."""
    if not TELEBOT_AVAILABLE:
        return None

    link = CHANNEL_INVITE_LINK
    if not link and REQUIRED_CHANNEL_ID.startswith("@"):
        link = f"https://t.me/{REQUIRED_CHANNEL_ID.lstrip('@')}"

    markup = telebot.types.InlineKeyboardMarkup()
    if link:
        markup.add(telebot.types.InlineKeyboardButton("📢 Join Channel", url=link))
    markup.add(telebot.types.InlineKeyboardButton("✅ I've Joined", callback_data="check_membership"))
    return markup


def _send_access_denied(message: Any) -> None:
    """Send the force-subscribe prompt with the join button."""
    if bot is None:
        return
    try:
        bot.reply_to(message, ACCESS_DENIED_TEXT, reply_markup=build_join_markup())
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not deliver the access-denied prompt: %s", exc)


def _guard(message: Any) -> bool:
    """Return ``True`` when the sender may use the AI, else prompt them to join."""
    if not force_sub_enabled():
        return True

    user = getattr(message, "from_user", None)
    user_id = getattr(user, "id", None)
    if user_id is None:
        return FORCE_SUB_FAIL_OPEN

    if is_user_member(bot, user_id, REQUIRED_CHANNEL_ID):
        return True

    _stats["blocked"] += 1
    _send_access_denied(message)
    return False


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
        # /start always answers - it's how a blocked user learns what to do.
        _conversations.pop(message.chat.id, None)
        if not _guard(message):
            return
        _safe_reply(message, WELCOME_TEXT)

    @bot.message_handler(commands=["help"])
    def handle_help(message: Any) -> None:
        if not _guard(message):
            return
        _safe_reply(message, HELP_TEXT)

    @bot.message_handler(commands=["reset"])
    def handle_reset(message: Any) -> None:
        if not _guard(message):
            return
        _conversations.pop(message.chat.id, None)
        user_task_mode[message.chat.id] = DEFAULT_MODE
        _safe_reply(message, "🧹 Conversation memory cleared. Mode reset to 🌐 Core.")

    # ---------------------- capability mode switches ---------------------- #

    @bot.message_handler(commands=["core"])
    def handle_core(message: Any) -> None:
        if not _guard(message):
            return
        user_task_mode[message.chat.id] = CORE_MODE
        _safe_reply(message, "🌐 Core Mode activated! (Gemma 3 Balanced Engine)")

    @bot.message_handler(commands=["flash"])
    def handle_flash(message: Any) -> None:
        if not _guard(message):
            return
        user_task_mode[message.chat.id] = "flash"
        _safe_reply(message, "⚡ Flash Mode activated! (Groq Llama 8B — instant replies)")

    @bot.message_handler(commands=["reasoning"])
    def handle_reasoning(message: Any) -> None:
        if not _guard(message):
            return
        user_task_mode[message.chat.id] = "reasoning"
        _safe_reply(message, "🧩 Reasoning Mode activated! (DeepSeek-R1 — step-by-step logic)")

    @bot.message_handler(commands=["pro"])
    def handle_pro(message: Any) -> None:
        if not _guard(message):
            return
        user_task_mode[message.chat.id] = "pro"
        _safe_reply(message, "🎓 Pro Mode activated! (Llama 70B — heavy coding & expert tasks)")

    @bot.message_handler(commands=["mode"])
    def handle_mode(message: Any) -> None:
        """Show the active capability and everything available."""
        if not _guard(message):
            return
        current = user_task_mode[message.chat.id]
        lines = [f"🎛 *Current mode:* {MODEL_MAP[current]['label']}", ""]
        for name, spec in MODEL_MAP.items():
            if name == VISION_MODE:
                continue  # automatic, not user-selectable
            mark = "▶️" if name == current else "  "
            lines.append(f"{mark} /{name} — {spec['description']}")
        lines += ["", "📷 Send a photo and Gemma 3 vision handles it automatically."]
        _safe_reply(message, "\n".join(lines))

    @bot.message_handler(commands=["status"])
    def handle_status(message: Any) -> None:
        from ai_router import available_modes  # local import to avoid cycles

        current = user_task_mode[message.chat.id]
        lines = ["📊 *Neuro-System status*", "", "*Capabilities*"]
        for spec in available_modes():
            mark = "✅" if spec["configured"] else "⛔"
            arrow = " ◀️" if spec["mode"] == current else ""
            lines.append(f"{mark} `{spec['mode']}` — {spec['model']}{arrow}")
        lines += [
            "",
            f"Force-subscribe: {'on - ' + REQUIRED_CHANNEL_ID if force_sub_enabled() else 'off'}",
            f"Messages handled: {_stats['messages_handled']}",
            f"Images handled: {_stats['images_handled']}",
            f"Blocked (not a member): {_stats['blocked']}",
            f"Errors: {_stats['errors']}",
            f"Last engine used: {_stats['last_provider'] or 'n/a'}",
        ]
        _safe_reply(message, "\n".join(lines))

    @bot.callback_query_handler(func=lambda call: call.data == "check_membership")
    def handle_check_membership(call: Any) -> None:
        """Re-verify membership when the user taps '✅ I've Joined'."""
        user_id = call.from_user.id
        invalidate_membership(user_id)  # force a fresh lookup

        if is_user_member(bot, user_id, REQUIRED_CHANNEL_ID, use_cache=False):
            try:
                bot.answer_callback_query(call.id, "✅ Verified. You're in!")
                bot.edit_message_text(
                    "✅ *Access granted!* Thanks for joining.\n\nSend me a message to get started.",
                    chat_id=call.message.chat.id,
                    message_id=call.message.message_id,
                    parse_mode="Markdown",
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Could not confirm membership to the user: %s", exc)
        else:
            try:
                bot.answer_callback_query(
                    call.id, "❌ You're still not in the channel. Please join first.", show_alert=True
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Could not answer the callback query: %s", exc)

    @bot.message_handler(content_types=["text"])
    def handle_text(message: Any) -> None:
        """Route a plain user message through the Gemma 3 fallback chain."""
        chat_id = message.chat.id
        prompt = (message.text or "").strip()
        if not prompt:
            return

        # Force-subscribe gate: never reach the AI router for non-members.
        if not _guard(message):
            return

        try:
            bot.send_chat_action(chat_id, "typing")
        except Exception:  # noqa: BLE001 - purely cosmetic
            pass

        try:
            result = capability_router(
                prompt,
                mode=user_task_mode[chat_id],
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                history=_history_for(chat_id),
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

    @bot.message_handler(content_types=["photo"])
    def handle_photo(message: Any) -> None:
        """Send a photo to the Gemma 3 vision engine (Google AI Studio)."""
        chat_id = message.chat.id
        if not _guard(message):
            return

        try:
            bot.send_chat_action(chat_id, "typing")
        except Exception:  # noqa: BLE001
            pass

        # message.photo is ordered smallest -> largest; take the biggest that
        # still fits our size budget so we don't blow up the request payload.
        try:
            candidates = [p for p in message.photo if (p.file_size or 0) <= MAX_IMAGE_BYTES]
            photo = candidates[-1] if candidates else message.photo[0]
            file_info = bot.get_file(photo.file_id)
            raw = bot.download_file(file_info.file_path)
        except Exception as exc:  # noqa: BLE001
            _stats["errors"] += 1
            logger.error("Could not download the photo: %s", exc)
            _safe_reply(message, "⚠️ I couldn't download that image. Please try again.")
            return

        if len(raw) > MAX_IMAGE_BYTES:
            _safe_reply(
                message,
                f"🖼 That image is too large (max {MAX_IMAGE_BYTES // (1024 * 1024)} MB). "
                "Please send a smaller one.",
            )
            return

        image_b64 = base64.b64encode(raw).decode("ascii")
        mime = _guess_mime(getattr(file_info, "file_path", "") or "")
        prompt = (message.caption or "").strip() or "Describe this image in detail."

        try:
            result = capability_router(
                prompt,
                mode=VISION_MODE,
                images=[{"data": image_b64, "mime_type": mime}],
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                history=_history_for(chat_id),
            )
            answer, provider = result.text, result.provider
            _stats["last_provider"] = provider
            _stats["images_handled"] += 1
        except Exception as exc:  # noqa: BLE001
            _stats["errors"] += 1
            logger.exception("Vision request failed: %s", exc)
            _safe_reply(message, "⚠️ Something went wrong analysing that image.")
            return

        # Keep the transcript coherent without storing the raw image bytes.
        _remember(chat_id, "user", f"[image] {prompt}")
        _remember(chat_id, "assistant", answer)
        _safe_reply(message, answer)
        _archive(message, f"[image] {prompt}", answer, provider)

    @bot.message_handler(
        content_types=["document", "audio", "voice", "video", "sticker"]
    )
    def handle_unsupported(message: Any) -> None:
        if not _guard(message):
            return
        _safe_reply(
            message,
            "📎 I can handle text and photos. Send an image and I'll analyse it with "
            "Gemma 3 vision.",
        )

    # Populate the in-app command menu (best effort - never fatal).
    try:
        bot.set_my_commands([
            telebot.types.BotCommand("core", "🌐 Gemma 3 balanced engine (default)"),
            telebot.types.BotCommand("flash", "⚡ Fastest replies"),
            telebot.types.BotCommand("reasoning", "🧩 Step-by-step logic"),
            telebot.types.BotCommand("pro", "🎓 Heavy coding & expert tasks"),
            telebot.types.BotCommand("mode", "Show the active engine"),
            telebot.types.BotCommand("reset", "Clear conversation memory"),
            telebot.types.BotCommand("status", "Provider & bot status"),
            telebot.types.BotCommand("help", "Usage help"),
        ])
    except Exception as exc:  # noqa: BLE001
        logger.debug("set_my_commands failed (safe to ignore): %s", exc)

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


def get_bot() -> Optional[Any]:
    """Return a ``TeleBot`` usable for API-side calls (e.g. membership checks).

    If polling is disabled or hasn't started yet, a lightweight instance is
    created on demand so the Mini App gate still works in an API-only
    deployment. Never raises.
    """
    global _api_bot

    if bot is not None:
        return bot
    if not TELEBOT_AVAILABLE or not TELEGRAM_BOT_TOKEN:
        return None

    with _lock:
        if _api_bot is None:
            try:
                _api_bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, parse_mode=None, threaded=False)
                logger.info("Created a standalone TeleBot instance for API-side checks")
            except Exception as exc:  # noqa: BLE001
                logger.error("Could not create the API-side TeleBot: %s", exc)
                return None
    return _api_bot


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
        "images_handled": _stats["images_handled"],
        "blocked": _stats["blocked"],
        "errors": _stats["errors"],
        "last_provider": _stats["last_provider"],
        "active_conversations": len(_conversations),
        "force_subscribe": {
            "enabled": force_sub_enabled(),
            "channel": REQUIRED_CHANNEL_ID or None,
            "invite_link": CHANNEL_INVITE_LINK or None,
            "fail_open": FORCE_SUB_FAIL_OPEN,
        },
    }


if __name__ == "__main__":  # pragma: no cover - run the bot standalone
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    create_bot()
    _polling_loop()
