import os
import base64
import asyncio
import logging
import re
import telebot
from typing import Dict

# Import router engines and database
from ai_router import smart_gemma_router, generate_image_router

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Bot Token
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN.startswith("<YOUR_"):
    # Fallback/placeholder to avoid direct crash during static validation imports
    TELEGRAM_BOT_TOKEN = "754321098:AAG_mock_token_for_validation_purposes"

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# 1. State Management
# In-memory dictionary tracking active modes for each chat_id (default "normal")
user_modes: Dict[int, str] = {}


# Helper to run async coroutine from sync telebot handler
def run_async_coro(coro_or_result):
    """Accept either a coroutine or a normal synchronous result.
    If coro_or_result is a coroutine, run it and return its result. Otherwise
    return the provided result directly. This avoids errors when callers
    pass sync functions' return values by accident.
    """
    try:
        if asyncio.iscoroutine(coro_or_result):
            return asyncio.run(coro_or_result)
        return coro_or_result
    except RuntimeError:
        # If there's already an active event loop in the current thread
        loop = asyncio.get_event_loop()
        if asyncio.iscoroutine(coro_or_result):
            return loop.run_until_complete(coro_or_result)
        return coro_or_result


# Helper to query gatekeeper safely without circular imports
async def check_membership(user_id: int) -> bool:
    try:
        from main import is_user_member
        return await is_user_member(user_id)
    except Exception as e:
        logger.warning(f"Error checking membership gatekeeper: {e}. Defaulting to True.")
        return True


# ---------------------------------------------------------------------------
# Command Handlers (Modes)
# ---------------------------------------------------------------------------

@bot.message_handler(commands=['flash'])
def set_flash_mode(message):
    chat_id = message.chat.id
    user_modes[chat_id] = "flash"
    bot.reply_to(message, "⚡ Flash Mode Activated! Responses will be lightning fast.")


@bot.message_handler(commands=['reasoning'])
def set_reasoning_mode(message):
    chat_id = message.chat.id
    user_modes[chat_id] = "reasoning"
    bot.reply_to(message, "🧠 Reasoning Mode Activated! I’ll analyze requests step-by-step before concluding.")


@bot.message_handler(commands=['pro'])
def set_pro_mode(message):
    chat_id = message.chat.id
    user_modes[chat_id] = "pro"
    bot.reply_to(message, "💼 Pro Mode Activated! Expect senior-level, professional, highly accurate responses.")


@bot.message_handler(commands=['normal'])
def set_normal_mode(message):
    chat_id = message.chat.id
    user_modes[chat_id] = "normal"
    bot.reply_to(message, "✅ Normal Mode Activated! Back to balanced responses.")


# ---------------------------------------------------------------------------
# Command Handlers (Image Generation via Hugging Face Flux models)
# ---------------------------------------------------------------------------

# This handler covers /draw, /imagine and /image commands
@bot.message_handler(commands=['draw', 'imagine', 'image'])
def handle_draw_command(message):
    chat_id = message.chat.id
    user_id = message.from_user.id

    # Validate gatekeeper membership
    if not run_async_coro(check_membership(user_id)):
        bot.reply_to(message, "🚫 Access Denied: You must join our channel to use this bot.")
        return

    # Extract user prompt from slash command
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        bot.reply_to(message, "🎨 Please specify a prompt. Example: `/draw a futuristic spaceship`", parse_mode="Markdown")
        return

    prompt = parts[1].strip()
    _handle_image_generation_flow(message, prompt)


# Handler to detect inline or text-triggered @image <prompt>
@bot.message_handler(func=lambda m: isinstance(m.text, str) and re.search(r"(^|\s)@image\s+(.+)", m.text, flags=re.IGNORECASE), content_types=['text'])
def handle_atimage_trigger(message):
    chat_id = message.chat.id
    user_id = message.from_user.id

    # Validate gatekeeper membership
    if not run_async_coro(check_membership(user_id)):
        bot.reply_to(message, "🚫 Access Denied: You must join our channel to use this bot.")
        return

    # Extract prompt after @image (first occurrence)
    m = re.search(r"@image\s+(.+)", message.text, flags=re.IGNORECASE)
    if not m:
        bot.reply_to(message, "🎨 Please provide a prompt after @image. Example: `@image a serene beach at sunset`")
        return

    prompt = m.group(1).strip()
    _handle_image_generation_flow(message, prompt)


def _handle_image_generation_flow(message, prompt: str):
    """Shared flow for image generation: sends progress, calls generator,
    handles exceptions and responds with photo bytes or error message.
    """
    chat_id = message.chat.id

    # Send immediate progress message
    status_msg = bot.reply_to(message, "🎨 Generating image with FLUX.1-schnell, please wait...")

    try:
        bot.send_chat_action(chat_id, "upload_photo")

        # generate_image_router is synchronous and returns bytes; our helper
        # run_async_coro will simply return non-coroutine results.
        img_bytes = run_async_coro(generate_image_router(prompt))

        if not img_bytes:
            raise RuntimeError("Image generation returned no data.")

        # Send photo back to the user
        bot.send_photo(
            chat_id,
            img_bytes,
            reply_to_message_id=message.message_id,
            caption=f"✨ Generated Masterpiece:\n\"{prompt}\""
        )

        # Delete helper status message
        try:
            bot.delete_message(chat_id, status_msg.message_id)
        except Exception:
            pass

    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        # Try to edit the status message if possible, otherwise reply
        try:
            bot.edit_message_text(f"❌ Image generation failed: {str(e)}", chat_id, status_msg.message_id)
        except Exception:
            try:
                bot.reply_to(message, f"❌ Image generation failed: {str(e)}")
            except Exception:
                logger.exception("Failed to notify user of image generation failure")


# ---------------------------------------------------------------------------
# Vision Handler (Image analysis via Google multimodal Gemma 3 endpoint)
# ---------------------------------------------------------------------------

@bot.message_handler(content_types=['photo'])
def handle_photo_message(message):
    chat_id = message.chat.id
    user_id = message.from_user.id

    # Validate gatekeeper membership
    if not run_async_coro(check_membership(user_id)):
        bot.reply_to(message, "🚫 Access Denied: You must join our channel to use this bot.")
        return

    bot.send_chat_action(chat_id, 'typing')

    try:
        # Retrieve highest resolution photo
        photo = message.photo[-1]
        file_info = bot.get_file(photo.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        # Convert to Base64 string
        image_base64 = base64.b64encode(downloaded_file).decode('utf-8')

        # Retrieve optional user caption or use default
        caption = message.caption or "Analyze this image"

        # Determine current chat mode
        mode = user_modes.get(chat_id, "normal")

        # Execute routing via primary multimodal engine (Google Gemma 3)
        reply = run_async_coro(smart_gemma_router(
            caption,
            mode=mode,
            image_base64=image_base64,
            chat_id=chat_id,
            user_id=user_id
        ))

        bot.reply_to(message, reply)
    except Exception as e:
        logger.error(f"Error handling photo message: {e}")
        bot.reply_to(message, "⚠️ Failed to analyze image. Please verify your config and try again.")


# ---------------------------------------------------------------------------
# Text Handler (Standard / Multi-tier fallback execution)
# ---------------------------------------------------------------------------

@bot.message_handler(func=lambda msg: True, content_types=['text'])
def handle_text_message(message):
    # Ensure standard text messages starting with "/" are treated as commands or ignored
    if message.text.startswith('/'):
        return

    chat_id = message.chat.id
    user_id = message.from_user.id

    # Validate gatekeeper membership
    if not run_async_coro(check_membership(user_id)):
        bot.reply_to(message, "🚫 Access Denied: You must join our channel to use this bot.")
        return

    bot.send_chat_action(chat_id, 'typing')

    # Fetch active chat execution mode
    mode = user_modes.get(chat_id, "normal")

    try:
        reply = run_async_coro(smart_gemma_router(
            message.text,
            mode=mode,
            chat_id=chat_id,
            user_id=user_id
        ))
        bot.reply_to(message, reply)
    except Exception as e:
        logger.error(f"Error handling text message: {e}")
        bot.reply_to(message, "⚠️ Failed to process your request. Please try again.")


def start_bot_polling():
    logger.info("Initializing Telegram Bot polling...")
    bot.infinity_polling(skip_pending=True)


if __name__ == "__main__":
    start_bot_polling()
