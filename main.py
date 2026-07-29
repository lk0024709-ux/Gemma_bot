"""
main.py - FastAPI + Telegram bot merged entrypoint (refactored)
- /chat POST now accepts optional "model" key in the JSON payload.
- Interactive model selection implemented via InlineKeyboardMarkup + CallbackQueryHandler
  (per-chat active model stored in CHAT_ACTIVE_MODEL).
- Uses ai_router.generate_ai_response(...) to ensure watermarking and routing/fallbacks.
"""
import os
import threading
import logging
from typing import Callable, Dict, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Telegram imports (v13 style)
from telegram import Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, CallbackQueryHandler

# Import AI routing function
from ai_router import generate_ai_response, get_current_model, is_secret_request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. FASTAPI & WEB APP SETUP ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://darling-sable-1c27ff.netlify.app", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageData(BaseModel):
    message: str
    user_id: str = "web_user"
    model: Optional[str] = None  # Optional model key for web UI switching

@app.post("/chat")
async def handle_web_chat(data: MessageData):
    user_text = data.message

    if is_secret_request(user_text):
         return {"reply": "I'm sorry, but I can't share secrets, API keys, or private configuration."}

    # Use explicit model from request if present, otherwise per-server current or default
    model_name = data.model or get_current_model()
    try:
        ai_reply = generate_ai_response(user_text, model_name)
    except Exception:
        logger.exception("Web /chat failed while contacting AI service")
        ai_reply = "Sorry, I had trouble contacting the AI service. Please try again later."

    return {"reply": ai_reply}

@app.get("/")
def home():
    return {"status": "FastAPI and Bot are both running smoothly!"}


# --- 2. Telegram bot + interactive model selection ---
COMMANDS: Dict[str, Dict] = {}
CHAT_ACTIVE_MODEL: Dict[int, str] = {}  # chat_id -> model string

# Model choices for interactive selection
MODEL_CHOICES = [
    ("✅ Gemma 3", "gemma-3-27b-it"),
    ("DeepSeek R1", "deepseek-r1"),
    ("Llama 4 Maverick", "llama-4-maverick"),
    ("Llama 3", "llama-3.1-8b-instant"),
    ("Flux (HF)", "black-forest-labs/FLUX.1-schnell"),
]

def register_command(dispatcher, name: str, handler: Callable, description: str):
    COMMANDS[name] = {"handler": handler, "description": description}
    dispatcher.add_handler(CommandHandler(name, handler))

def _build_models_keyboard(current_model: Optional[str] = None):
    keyboard = []
    for label, model in MODEL_CHOICES:
        if model == current_model:
            kb_label = f"{label} ✅"
        else:
            kb_label = label
        keyboard.append([InlineKeyboardButton(kb_label, callback_data=f"SET_MODEL:{model}")])
    return InlineKeyboardMarkup(keyboard)

def start(update: Update, context: CallbackContext):
    model_name = CHAT_ACTIVE_MODEL.get(update.effective_chat.id, get_current_model())
    welcome = "Hello! I am your friendly AI assistant. How can I help you today?\n\n"
    welcome += f"(Active model: {model_name})\n\n"
    welcome += "Use /models to change the active model for this chat.\n\n"
    welcome += "Here are the available commands:\n"
    for cmd, info in sorted(COMMANDS.items()):
        welcome += f"/{cmd} — {info['description']}\n"
    update.message.reply_text(welcome)

def help_command(update: Update, context: CallbackContext):
    help_text = "I can help with queries, AI conversation, and settings. Use /start to see available commands."
    update.message.reply_text(help_text)

def model_command(update: Update, context: CallbackContext):
    model_name = CHAT_ACTIVE_MODEL.get(update.effective_chat.id, get_current_model())
    update.message.reply_text(f"The bot is currently using model: {model_name}")

def models_command(update: Update, context: CallbackContext):
    """Send inline keyboard allowing users to switch models per chat."""
    chat_id = update.effective_chat.id
    current = CHAT_ACTIVE_MODEL.get(chat_id, get_current_model())
    keyboard = _build_models_keyboard(current_model=current)
    update.message.reply_text("Choose an active model for this chat:", reply_markup=keyboard)

def model_button_callback(update: Update, context: CallbackContext):
    """CallbackQuery handler to process inline model selection."""
    query = update.callback_query
    if not query or not query.data:
        return
    data = query.data
    if data.startswith("SET_MODEL:"):
        _, model_value = data.split(":", 1)
        chat_id = update.effective_chat.id
        CHAT_ACTIVE_MODEL[chat_id] = model_value
        try:
            query.answer(text=f"Active model set to {model_value}")
            # Edit original message to reflect selection
            query.edit_message_text(f"Active model for this chat is now: {model_value}")
        except Exception:
            logger.exception("Failed to answer or edit model selection message")

def echo_or_ai(update: Update, context: CallbackContext):
    user_text = update.message.text or ""

    if is_secret_request(user_text):
        update.message.reply_text("I'm sorry, but I can't share secrets, API keys, or private configuration.")
        return

    model_name = CHAT_ACTIVE_MODEL.get(update.effective_chat.id, get_current_model())
    try:
        ai_reply = generate_ai_response(user_text, model_name)
    except Exception:
        logger.exception("Error while generating AI reply for Telegram message")
        ai_reply = "Sorry, I had trouble contacting the AI service. Please try again later."

    update.message.reply_text(ai_reply)


# --- 3. BACKGROUND THREADING LOGIC ---
def run_telegram_bot():
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN")
    if not token:
        logger.error("Telegram bot token not found. Please set TELEGRAM_BOT_TOKEN or TELEGRAM_TOKEN environment variable.")
        return

    updater = Updater(token=token, use_context=True)
    dispatcher = updater.dispatcher

    register_command(dispatcher, "start", start, "Show welcome message and available commands")
    register_command(dispatcher, "help", help_command, "Get help and usage information")
    register_command(dispatcher, "model", model_command, "Show the currently active AI model")
    register_command(dispatcher, "models", models_command, "Interactively choose active model for this chat")

    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo_or_ai))
    dispatcher.add_handler(CallbackQueryHandler(model_button_callback))

    try:
        bot = updater.bot
        bot_commands = [BotCommand(cmd, data["description"]) for cmd, data in COMMANDS.items()]
        bot.set_my_commands(bot_commands)
    except Exception:
        logger.exception("Failed to set bot commands to Telegram API")

    logger.info("Starting Telegram Bot polling...")
    updater.start_polling()


# Launch bot on FastAPI startup in background thread
@app.on_event("startup")
def start_bot_in_background():
    logger.info("FastAPI starting... Launching Telegram Bot in a background thread.")
    threading.Thread(target=run_telegram_bot, daemon=True).start()
