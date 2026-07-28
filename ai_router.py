import os
import logging
from typing import List, Optional, Dict
import httpx
import dotenv

# Load environment variables from .env if present
dotenv.load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token/Key Rotation Utility
# ---------------------------------------------------------------------------
class TokenRotator:
    """
    Manages round-robin rotation across multiple environment variable key options.
    If multiple variables are populated, it rotates through them sequentially.
    """
    def __init__(self, env_vars: List[str], fallback_var: Optional[str] = None):
        self.env_vars = env_vars
        self.fallback_var = fallback_var
        self._counter = 0

    def get_token(self) -> Optional[str]:
        valid_tokens = []
        for var in self.env_vars:
            val = os.getenv(var)
            if val and val.strip() and not val.startswith("<YOUR_"):
                valid_tokens.append(val.strip())
        
        # If no explicit list, check the fallback (e.g. standard bare name)
        if not valid_tokens and self.fallback_var:
            val = os.getenv(self.fallback_var)
            if val and val.strip() and not val.startswith("<YOUR_"):
                valid_tokens.append(val.strip())
                
        if not valid_tokens:
            return None
            
        token = valid_tokens[self._counter % len(valid_tokens)]
        self._counter = (self._counter + 1) % len(valid_tokens)
        return token

# Initialize rotators based on the strict multi-tier mapping requirements
google_keys = TokenRotator(["GOOGLE_API_KEY_1", "GOOGLE_API_KEY_2", "GOOGLE_API_KEY_3"], "GOOGLE_API_KEY")
github_tokens = TokenRotator(["GITHUB_TOKEN_1", "GITHUB_TOKEN_2", "GITHUB_TOKEN_3"], "GITHUB_TOKEN")
groq_keys = TokenRotator(["GROQ_API_KEY_1"], "GROQ_API_KEY")
hf_tokens = TokenRotator(["HF_TOKEN_1", "HF_TOKEN_2"], "HF_TOKEN")


# ---------------------------------------------------------------------------
# In-Memory Database / Channel Memory Integration
# ---------------------------------------------------------------------------
class MemoryDB:
    """
    In-memory channel memory database to track short-term session context
    for robust, non-blocking conversational memory state.
    """
    def __init__(self, max_history: int = 10):
        self.history: Dict[int, List[Dict[str, str]]] = {}
        self.max_history = max_history

    def add_message(self, chat_id: int, role: str, content: str):
        if chat_id not in self.history:
            self.history[chat_id] = []
        self.history[chat_id].append({"role": role, "content": content})
        # Keep within limits
        if len(self.history[chat_id]) > self.max_history:
            self.history[chat_id].pop(0)

    def get_formatted_context(self, chat_id: int) -> str:
        messages = self.history.get(chat_id, [])
        if not messages:
            return ""
        formatted = []
        for msg in messages:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role_label}: {msg['content']}")
        return "\n".join(formatted)

memory_db = MemoryDB()


# ---------------------------------------------------------------------------
# API Call Providers
# ---------------------------------------------------------------------------

async def call_google_text_api(prompt: str, model: str = "gemma-3-27b-it") -> str:
    """
    Call Google AI Studio (gemma-3-27b-it) using round-robin rotated keys.
    """
    key = google_keys.get_token()
    if not key:
        raise ValueError("Google AI Studio API Key is not configured.")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Invalid response structure from Google AI Studio: {data}") from e


async def call_google_vision_api(prompt: str, image_base64: str, model: str = "gemma-3-27b-it") -> str:
    """
    Call Google AI Studio's multimodal gemma-3 endpoint using inline image data.
    """
    key = google_keys.get_token()
    if not key:
        raise ValueError("Google AI Studio API Key is not configured for vision.")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_base64
                        }
                    }
                ]
            }
        ]
    }
    
    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Invalid response structure from Google AI Studio (Vision): {data}") from e


async def call_github_models_api(prompt: str, model: str) -> str:
    """
    Call GitHub Models endpoint (deepseek-r1 / llama-4) using round-robin rotated tokens.
    """
    token = github_tokens.get_token()
    if not token:
        raise ValueError("GitHub Token is not configured.")
    
    url = "https://models.inference.ai.azure.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "model": model
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Invalid response structure from GitHub Models: {data}") from e


async def call_groq_api(prompt: str, model: str = "llama-3.1-8b-instant") -> str:
    """
    Call Groq API (llama-3.1-8b-instant) using GROQ_API_KEY_1 key.
    """
    key = groq_keys.get_token()
    if not key:
        raise ValueError("Groq API Key is not configured.")
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "model": model
    }
    
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Invalid response structure from Groq: {data}") from e


# ---------------------------------------------------------------------------
# Hugging Face Flux Image Generation Router
# ---------------------------------------------------------------------------
async def generate_image_router(prompt: str) -> bytes:
    """
    Generate an image using Hugging Face's Inference API, routing across FLUX.1 models
    and rotating through HF_TOKENs pool.
    """
    token = hf_tokens.get_token()
    if not token:
        raise ValueError("Hugging Face token is not configured.")
    
    models = [
        "black-forest-labs/FLUX.1-schnell",
        "black-forest-labs/FLUX.1-dev"
    ]
    
    last_err = None
    for model in models:
        url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": prompt
        }
        
        logger.info(f"Attempting image generation using model: {model}...")
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                # If we get valid binary image response, return it directly
                return response.content
        except Exception as e:
            logger.warning(f"Hugging Face image model {model} failed with error: {e}. Trying next model/token...")
            last_err = e
            # Rotate token to the next available token before trying next model
            token = hf_tokens.get_token() or token
            continue
            
    if last_err:
        raise last_err
    raise ValueError("Failed to generate image via Hugging Face. Ensure your HF tokens and model names are correct.")


# ---------------------------------------------------------------------------
# Smart Gemma Router (Text + Vision Fallback Routing)
# ---------------------------------------------------------------------------
async def smart_gemma_router(
    prompt: str,
    *,
    mode: str = "normal",
    image_base64: Optional[str] = None,
    chat_id: Optional[int] = None,
    user_id: Optional[int] = None
) -> str:
    """
    Directs incoming messages to the proper endpoint following strict multi-tier fallback architecture.
    
    Fallback path: Google AI Studio (gemma-3-27b-it) -> GitHub Models (deepseek-r1 / llama-4) -> Groq (llama-3.1-8b-instant).
    """
    normalized_mode = (mode or "normal").lower()
    
    # 1. Apply System Meta-Prompt Injection
    meta_prompt = ""
    if normalized_mode == "reasoning":
        meta_prompt = "[System: You are an advanced reasoning engine. Analyze the following request step-by-step before concluding.]"
    elif normalized_mode == "pro":
        meta_prompt = "[System: Act as a senior technical expert and consultant. Provide exhaustive, highly accurate, and professional output without fluff.]"

    # Compile the message prompt
    full_prompt = prompt
    if meta_prompt:
        full_prompt = f"{meta_prompt}\n{prompt}"

    # Get history if session tracks history and chat_id is available
    history_context = ""
    if chat_id is not None:
        history_context = memory_db.get_formatted_context(chat_id)
        
    # Weave history context if present
    if history_context:
        full_prompt = f"System: Conversational memory context of previous turns:\n{history_context}\n\nUser: {full_prompt}"

    # Record User Prompt to Memory Database
    if chat_id is not None:
        memory_db.add_message(chat_id, "user", prompt)

    # 2. Multimodal / Vision Handling (Exclusive to Google Gemma 3 endpoint)
    if image_base64 is not None:
        try:
            logger.info("Executing Google AI Studio Vision request...")
            reply = await call_google_vision_api(full_prompt, image_base64, model="gemma-3-27b-it")
            if chat_id is not None:
                memory_db.add_message(chat_id, "assistant", reply)
            return reply
        except Exception as e:
            logger.error(f"Google AI Studio Vision endpoint failed: {e}")
            return "👀 Vision mode is currently only available on the primary Google endpoint. Please try again."

    # 3. Flash Mode handling: immediately route to Groq (llama-3.1-8b-instant) for low-latency
    if normalized_mode == "flash":
        logger.info("Flash mode enabled. Immediately routing request to Groq...")
        try:
            reply = await call_groq_api(full_prompt, model="llama-3.1-8b-instant")
            if chat_id is not None:
                memory_db.add_message(chat_id, "assistant", reply)
            return reply
        except Exception as e:
            logger.warning(f"Groq API call (llama-3.1-8b-instant) failed in flash mode: {e}. Trying versatile model...")
            try:
                reply = await call_groq_api(full_prompt, model="llama-3.3-70b-versatile")
                if chat_id is not None:
                    memory_db.add_message(chat_id, "assistant", reply)
                return reply
            except Exception as e2:
                logger.error(f"All Groq endpoints failed in flash mode: {e2}")
                return "⚡ Flash Mode error: Low-latency Groq endpoints are currently unavailable. Please try again."

    # 4. Standard Flow: Google AI Studio -> GitHub Models -> Groq
    
    # TIER 1: Google AI Studio (gemma-3-27b-it)
    try:
        logger.info("Routing request to Google AI Studio Tier...")
        reply = await call_google_text_api(full_prompt, model="gemma-3-27b-it")
        if chat_id is not None:
            memory_db.add_message(chat_id, "assistant", reply)
        return reply
    except Exception as google_err:
        logger.warning(f"Google AI Studio (gemma-3-27b-it) failed: {google_err}. Transitioning to GitHub Models Fallback Tier...")

    # TIER 2: GitHub Models (deepseek-r1 -> llama-4-maverick -> llama-4-scout)
    github_models = ["deepseek-r1", "llama-4-maverick", "llama-4-scout"]
    for model in github_models:
        try:
            logger.info(f"Routing request to GitHub Models Tier (Model: {model})...")
            reply = await call_github_models_api(full_prompt, model=model)
            if chat_id is not None:
                memory_db.add_message(chat_id, "assistant", reply)
            return reply
        except Exception as github_err:
            logger.warning(f"GitHub Model {model} failed: {github_err}. Trying alternative/next GitHub Model/Token...")
            continue

    # TIER 3: Groq fallback (llama-3.1-8b-instant / llama-3.3-70b-versatile)
    try:
        logger.info("Routing request to Groq Fallback Tier...")
        reply = await call_groq_api(full_prompt, model="llama-3.1-8b-instant")
        if chat_id is not None:
            memory_db.add_message(chat_id, "assistant", reply)
        return reply
    except Exception as groq_err1:
        logger.warning(f"Groq primary model failed: {groq_err1}. Attempting versatile model...")
        try:
            reply = await call_groq_api(full_prompt, model="llama-3.3-70b-versatile")
            if chat_id is not None:
                memory_db.add_message(chat_id, "assistant", reply)
            return reply
        except Exception as groq_err2:
            logger.error(f"All backends (Google AI Studio, GitHub Models, Groq) have failed. Final error: {groq_err2}")
            return "❌ I'm sorry, our AI service is currently overwhelmed by rate limits. Please try again shortly."
