"""
main.py - FastAPI + Telegram bot merged entrypoint

Features:
- FastAPI web endpoints (for Netlify web app) with CORS.
- /chat POST endpoint that reuses the same AI router and security checks.
- Background Telegram bot running in a daemon thread (python-telegram-bot v13 style).
Environment variables required:
  - TELEGRAM_BOT_TOKEN or TELEGRAM_TOKEN
  - OPENAI_API_KEY
  - CURRENT_MODEL (optional; default used if missing)
Run:
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import threading
import logging
from typing import Callable, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Telegram imports (v13 style). If you use v20 async, see notes below.
from telegram import Update, BotCommand
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

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

@app.post("/chat")
async def handle_web_chat(data: MessageData):
    user_text = data.message

    # Web app ke liye bhi same security aur model rules
    if is_secret_request(user_text):
         return {"reply": "I'm sorry, but I can't share secrets, API keys, or private configuration."}

    model_name = get_current_model()
    try:
        ai_reply = generate_ai_response(user_text, model_name)
    except Exception:
        ai_reply = "Sorry, I had trouble contacting the AI service. Please try again later."

    return {"reply": ai_reply}

@app.get("/")
def home():
    return {"status": "FastAPI and Bot are both running smoothly!"}


# --- 2. TELEGRAM BOT LOGIC (Copilot logic merged) ---
COMMANDS: Dict[str, Dict] = {}

def register_command(dispatcher, name: str, handler: Callable, description: str):
    COMMANDS[name] = {"handler": handler, "description": description}
    dispatcher.add_handler(CommandHandler(name, handler))

def start(update: Update, context: CallbackContext):
    model_name = get_current_model()
    welcome = "Hello! I am your friendly AI assistant. How can I help you today?\n\n"
    welcome += f"(Active model: {model_name})\n\n"
    welcome += "Here are the available commands:\n"
    for cmd, info in sorted(COMMANDS.items()):
        welcome += f"/{cmd} — {info['description']}\n"
    update.message.reply_text(welcome)

def help_command(update: Update, context: CallbackContext):
    help_text = "I can help with queries, AI conversation, and settings. Use /start to see available commands."
    update.message.reply_text(help_text)

def model_command(update: Update, context: CallbackContext):
    model_name = get_current_model()
    update.message.reply_text(f"The bot is currently using model: {model_name}")

def echo_or_ai(update: Update, context: CallbackContext):
    user_text = update.message.text or ""

    if is_secret_request(user_text):
        update.message.reply_text("I'm sorry, but I can't share secrets, API keys, or private configuration.")
        return

    model_name = get_current_model()
    try:
        ai_reply = generate_ai_response(user_text, model_name)
    except Exception:
        ai_reply = "Sorry, I had trouble contacting the AI service. Please try again later."

    update.message.reply_text(ai_reply)


# --- 3. BACKGROUND THREADING LOGIC ---
def run_telegram_bot():
    # Robust token fetching: check common env var names.
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN")
    if not token:
        logger.error("Telegram bot token not found. Please set TELEGRAM_BOT_TOKEN or TELEGRAM_TOKEN environment variable.")
        return

    updater = Updater(token=token, use_context=True)
    dispatcher = updater.dispatcher

    register_command(dispatcher, "start", start, "Show welcome message and available commands")
    register_command(dispatcher, "help", help_command, "Get help and usage information")
    register_command(dispatcher, "model", model_command, "Show the currently active AI model")

    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo_or_ai))

    try:
        bot = updater.bot
        bot_commands = [BotCommand(cmd, data["description"]) for cmd, data in COMMANDS.items()]
        bot.set_my_commands(bot_commands)
    except Exception:
        logger.exception("Failed to set bot commands to Telegram API")

    logger.info("Starting Telegram Bot polling...")
    updater.start_polling()
    # Removed updater.idle() to avoid signal registration in a non-main thread (ValueError).
    # The polling loop runs fine in a background daemon thread without idle().


# Launch bot on FastAPI startup in background thread
@app.on_event("startup")
def start_bot_in_background():
    logger.info("FastAPI starting... Launching Telegram Bot in a background thread.")
    threading.Thread(target=run_telegram_bot, daemon=True).start()
