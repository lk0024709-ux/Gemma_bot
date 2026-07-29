"""
bot_handler.py - Telegram Bot Command Routing & State Management for IRA

Uses python-telegram-bot v20+ (async-native).

Commands:
  /Start    - Welcome message with command list
  /Flash    - Groq mode (llama3-8b-8192)
  /Thinking - DeepSeek mode (deepseek-r1)
  /Pro      - Scout mode (llama-4-scout)
  /Expert   - Maverick mode (llama-4-maverick)
  /Core     - Gemma mode (gemma-3) [default]
  /Image    - Flux mode (FLUX.1-schnell)
  /Custom   - Set custom system prompt
  /IRA      - Activate IRA identity

State: user_states[chat_id] = {"mode": "Gemma", "system_prompt": ""}

Logic:
  - If command has text after it (e.g., /Flash What is AI?),
    generate response immediately using that mode.
  - If command is alone (e.g., /Flash), switch active mode.
"""

import logging
from typing import Dict, Optional

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from ai_router import (
    route_text,
    route_image,
    MODE_GROQ,
    MODE_DEEPSEEK,
    MODE_SCOUT,
    MODE_MAVERICK,
    MODE_GEMMA,
    MODE_FLUX,
    DEFAULT_MODE,
    IRA_SYSTEM_PROMPT,
    IRAAllProvidersFailed,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State Management
# ---------------------------------------------------------------------------

user_states: Dict[int, dict] = {}


def _get_state(chat_id: int) -> dict:
    """Get or initialize user state for a chat."""
    if chat_id not in user_states:
        user_states[chat_id] = {"mode": DEFAULT_MODE, "system_prompt": ""}
    return user_states[chat_id]


def _set_mode(chat_id: int, mode: str) -> None:
    """Set the active mode for a chat."""
    state = _get_state(chat_id)
    state["mode"] = mode


def _set_system_prompt(chat_id: int, prompt: str) -> None:
    """Set the custom system prompt for a chat."""
    state = _get_state(chat_id)
    state["system_prompt"] = prompt


# ---------------------------------------------------------------------------
# Welcome / Help
# ---------------------------------------------------------------------------

WELCOME_TEXT = """🤖 *Welcome to IRA — Your Multi-Model AI Assistant!*

Created by *Aditya Upadhyay*

Here are your available commands:

⚡ /Flash — Ultra-fast mode (Groq Llama 3)
🧠 /Thinking — Deep reasoning mode (DeepSeek R1)
💼 /Pro — Professional scout mode (Llama 4 Scout)
🎯 /Expert — Expert maverick mode (Llama 4 Maverick)
💎 /Core — Balanced default mode (Gemma 3)
🎨 /Image — Image generation mode (FLUX.1-schnell)

📝 /Custom <prompt> — Set a custom response style
🤖 /IRA — Activate IRA's full identity

*How to use:*
• Send a command alone to switch modes (e.g., `/Flash`)
• Add text after a command for instant response (e.g., `/Flash What is AI?`)
• In any mode, just send a message and I'll respond!

Currently active: *{mode}*
"""


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command - send welcome message."""
    chat_id = update.effective_chat.id
    state = _get_state(chat_id)
    text = WELCOME_TEXT.format(mode=state["mode"])
    try:
        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception:
        logger.exception("Failed to send welcome message")
        await update.message.reply_text("Welcome to IRA! Send any message to get started.")


# ---------------------------------------------------------------------------
# Mode Switch Commands
# ---------------------------------------------------------------------------

async def _handle_mode_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    mode: str,
    mode_label: str,
    mode_emoji: str,
) -> None:
    """
    Generic handler for mode-switching commands.

    If text follows the command, generate response in that mode immediately.
    If command alone, switch the user's active mode.
    """
    chat_id = update.effective_chat.id
    message_text = update.message.text or ""

    # Extract text after the command (e.g., "/Flash What is AI?" -> "What is AI?")
    parts = message_text.split(maxsplit=1)
    inline_prompt = parts[1].strip() if len(parts) > 1 else ""

    if inline_prompt:
        # Generate response immediately using this mode
        await _generate_and_reply(update, inline_prompt, mode, chat_id)
    else:
        # Switch mode
        _set_mode(chat_id, mode)
        try:
            await update.message.reply_text(
                f"{mode_emoji} Switched to {mode_label} mode."
            )
        except Exception:
            logger.exception("Failed to send mode switch confirmation")


async def cmd_flash(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /Flash - Groq mode."""
    await _handle_mode_command(update, context, MODE_GROQ, "Flash (Groq)", "⚡")


async def cmd_thinking(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /Thinking - DeepSeek mode."""
    await _handle_mode_command(update, context, MODE_DEEPSEEK, "Thinking (DeepSeek)", "🧠")


async def cmd_pro(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /Pro - Scout mode."""
    await _handle_mode_command(update, context, MODE_SCOUT, "Pro (Scout)", "💼")


async def cmd_expert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /Expert - Maverick mode."""
    await _handle_mode_command(update, context, MODE_MAVERICK, "Expert (Maverick)", "🎯")


async def cmd_core(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /Core - Gemma mode (default)."""
    await _handle_mode_command(update, context, MODE_GEMMA, "Core (Gemma)", "💎")


async def cmd_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /Image - Flux image generation mode."""
    chat_id = update.effective_chat.id
    message_text = update.message.text or ""

    parts = message_text.split(maxsplit=1)
    inline_prompt = parts[1].strip() if len(parts) > 1 else ""

    if inline_prompt:
        # Generate image immediately
        await _generate_image_and_reply(update, inline_prompt, chat_id)
    else:
        # Switch to image mode
        _set_mode(chat_id, MODE_FLUX)
        try:
            await update.message.reply_text("🎨 Switched to Image mode.")
        except Exception:
            logger.exception("Failed to send mode switch confirmation")


# ---------------------------------------------------------------------------
# Custom Prompt & IRA Identity Commands
# ---------------------------------------------------------------------------

async def cmd_custom(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /Custom <prompt> - Update the system prompt."""
    chat_id = update.effective_chat.id
    message_text = update.message.text or ""

    parts = message_text.split(maxsplit=1)
    custom_prompt = parts[1].strip() if len(parts) > 1 else ""

    if custom_prompt:
        _set_system_prompt(chat_id, custom_prompt)
        try:
            await update.message.reply_text("✅ Custom response style updated.")
        except Exception:
            logger.exception("Failed to confirm custom prompt update")
    else:
        try:
            await update.message.reply_text(
                "📝 Usage: /Custom <your preferred response style>\n"
                "Example: /Custom Reply in a pirate accent"
            )
        except Exception:
            logger.exception("Failed to send custom prompt usage")


async def cmd_ira(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /IRA - Activate IRA identity."""
    chat_id = update.effective_chat.id
    _set_system_prompt(chat_id, IRA_SYSTEM_PROMPT)
    try:
        await update.message.reply_text(
            "🤖 IRA Identity Activated.\n\n"
            "I am IRA, an advanced multi-model AI assistant created by Aditya Upadhyay. "
            "All my systems are fully operational and at your service."
        )
    except Exception:
        logger.exception("Failed to send IRA identity confirmation")


# ---------------------------------------------------------------------------
# Message Handler (text messages, non-command)
# ---------------------------------------------------------------------------

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle all non-command text messages.

    Routes based on current mode:
      - Flux mode -> Image Generator
      - Any other mode -> Text Generator (with user's system_prompt)
    """
    chat_id = update.effective_chat.id
    state = _get_state(chat_id)
    user_text = update.message.text or ""

    if not user_text.strip():
        return

    mode = state["mode"]
    system_prompt = state["system_prompt"]

    if mode == MODE_FLUX:
        await _generate_image_and_reply(update, user_text, chat_id)
    else:
        await _generate_and_reply(update, user_text, mode, chat_id, system_prompt)


# ---------------------------------------------------------------------------
# Internal: Generate & Reply Helpers
# ---------------------------------------------------------------------------

async def _generate_and_reply(
    update: Update,
    prompt: str,
    mode: str,
    chat_id: int,
    system_prompt: str = "",
) -> None:
    """
    Send a 'Thinking...' placeholder, generate AI response, then edit or replace it.
    """
    status_msg = None
    try:
        # Send transient status message
        status_msg = await update.message.reply_text("💬 Thinking...")
        await update.effective_chat.send_action(ChatAction.TYPING)
    except Exception:
        logger.warning("Could not send status message, continuing without it")
        status_msg = None

    try:
        reply = route_text(prompt, mode, system_prompt=system_prompt)

        # Edit the status message with the actual reply
        if status_msg:
            try:
                await status_msg.edit_text(reply)
            except Exception:
                # If edit fails (e.g., message too long), delete and send new
                try:
                    await status_msg.delete()
                except Exception:
                    pass
                await update.message.reply_text(reply)
        else:
            await update.message.reply_text(reply)

    except IRAAllProvidersFailed as exc:
        error_text = "⚠️ Sorry, the AI service is temporarily unavailable. Please try again shortly."
        logger.error("All providers failed for chat %d: %s", chat_id, exc)

        if status_msg:
            try:
                await status_msg.edit_text(error_text)
            except Exception:
                try:
                    await status_msg.delete()
                except Exception:
                    pass
                await update.message.reply_text(error_text)
        else:
            await update.message.reply_text(error_text)

    except Exception as exc:
        error_text = "❌ An unexpected error occurred. Please try again."
        logger.exception("Unexpected error generating response for chat %d", chat_id)

        if status_msg:
            try:
                await status_msg.edit_text(error_text)
            except Exception:
                try:
                    await status_msg.delete()
                except Exception:
                    pass
                await update.message.reply_text(error_text)
        else:
            await update.message.reply_text(error_text)


async def _generate_image_and_reply(
    update: Update,
    prompt: str,
    chat_id: int,
) -> None:
    """
    Send a 'Generating...' placeholder, generate image, then send photo and delete placeholder.
    """
    status_msg = None
    try:
        status_msg = await update.message.reply_text("🎨 Generating...")
        await update.effective_chat.send_action(ChatAction.UPLOAD_PHOTO)
    except Exception:
        logger.warning("Could not send status message for image generation")
        status_msg = None

    try:
        image_bytes = route_image(prompt)

        # Send the photo
        import io
        photo = io.BytesIO(image_bytes)
        photo.name = "ira_generated_image.png"

        await update.message.reply_photo(
            photo=photo,
            caption=f"✨ Generated: \"{prompt}\"",
        )

        # Delete the status message
        if status_msg:
            try:
                await status_msg.delete()
            except Exception:
                pass

    except IRAAllProvidersFailed as exc:
        error_text = "⚠️ Image generation service is temporarily unavailable. Please try again shortly."
        logger.error("All image providers failed for chat %d: %s", chat_id, exc)

        if status_msg:
            try:
                await status_msg.edit_text(error_text)
            except Exception:
                try:
                    await status_msg.delete()
                except Exception:
                    pass
                await update.message.reply_text(error_text)
        else:
            await update.message.reply_text(error_text)

    except Exception as exc:
        error_text = "❌ Image generation failed. Please try a different prompt."
        logger.exception("Unexpected error generating image for chat %d", chat_id)

        if status_msg:
            try:
                await status_msg.edit_text(error_text)
            except Exception:
                try:
                    await status_msg.delete()
                except Exception:
                    pass
                await update.message.reply_text(error_text)
        else:
            await update.message.reply_text(error_text)


# ---------------------------------------------------------------------------
# Application Builder - Called by main.py
# ---------------------------------------------------------------------------

def register_handlers(application: Application) -> None:
    """
    Register all command and message handlers on the Application instance.
    Called from main.py during bot initialization.
    """
    # Command handlers
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("flash", cmd_flash))
    application.add_handler(CommandHandler("thinking", cmd_thinking))
    application.add_handler(CommandHandler("pro", cmd_pro))
    application.add_handler(CommandHandler("expert", cmd_expert))
    application.add_handler(CommandHandler("core", cmd_core))
    application.add_handler(CommandHandler("image", cmd_image))
    application.add_handler(CommandHandler("custom", cmd_custom))
    application.add_handler(CommandHandler("ira", cmd_ira))

    # Text message handler (non-command messages)
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message)
    )

    logger.info("All IRA bot handlers registered.")
