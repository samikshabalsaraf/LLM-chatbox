"""
Chatbot Backend - FastAPI + Free LLM (Ollama or Hugging Face)

Uses Ollama by default (local, free, no API key).
Optional: Hugging Face Inference API (free tier) via USE_HF=true and HF_TOKEN.
"""

from collections import deque
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
import os

# Set USE_HF=true and HF_TOKEN=your_token to use Hugging Face instead of Ollama
USE_HUGGINGFACE = os.getenv("USE_HF", "false").lower() in ("true", "1", "yes")
HF_TOKEN = os.getenv("HF_TOKEN", "")
# Hugging Face model (free inference; smaller = faster)
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

# Ollama settings (used when USE_HF is false)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Local model name: run `ollama run mistral` (or llama3, phi) first
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# How many recent messages to keep for context (system + user/assistant pairs)
MAX_HISTORY_MESSAGES = 5

app = FastAPI(title="Free LLM Chatbot", version="1.0.0")

# Add CORS middleware to allow requests from browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load HTML template from templates/ (path relative to this file so it works from any cwd)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
template_env = Environment(loader=FileSystemLoader(os.path.join(_BASE_DIR, "templates")))


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """Incoming chat request from the frontend."""

    message: str = Field(..., min_length=1, description="User message")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    # Optional: pass existing history (e.g. from frontend); backend also keeps its own
    history: Optional[list[dict]] = None


class ChatResponse(BaseModel):
    """Response sent back to the client."""

    reply: str
    model_used: str
    history: list[dict]  # Updated conversation for UI


# ---------------------------------------------------------------------------
# In-memory conversation store (per session; keyed by session_id for simplicity)
# In production you might use Redis or DB. We use a simple dict + deque.
# ---------------------------------------------------------------------------
# Format: { "session_id": deque([ {"role":"user","content":"..."}, {"role":"assistant","content":"..."} ], maxlen=MAX_HISTORY_MESSAGES) }
conversation_store: dict[str, deque] = {}

# Default session when none provided
DEFAULT_SESSION = "default"


def get_history(session_id: str) -> deque:
    """Get or create conversation history for this session (last N messages)."""
    if session_id not in conversation_store:
        conversation_store[session_id] = deque(maxlen=MAX_HISTORY_MESSAGES)
    return conversation_store[session_id]


# ---------------------------------------------------------------------------
# Prompt templating: system + user message structure
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful, friendly assistant. Answer concisely and clearly. If you don't know something, say so."""


def build_messages(history: deque, new_user_message: str) -> list[dict]:
    """
    Build messages list for LLM: system prompt + last N turns + current user message.
    Same structure works for both Ollama /api/chat and Hugging Face chat API.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": new_user_message})
    return messages


# ---------------------------------------------------------------------------
# LLM calls: Ollama (local) vs Hugging Face (free API)
# ---------------------------------------------------------------------------
async def call_ollama(messages: list[dict], temperature: float) -> str:
    """
    Call local Ollama /api/chat. No API key needed.
    Ensure Ollama is running and you have pulled a model: ollama run mistral
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return (data.get("message", {}) or {}).get("content", "").strip()


async def call_huggingface(messages, temperature):

    if not HF_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="HF_TOKEN not found."
        )

    prompt = ""

    for m in messages:
        if m["role"] == "system":
            prompt += f"<s>[INST] {m['content']} [/INST]\n"
        elif m["role"] == "user":
            prompt += f"[INST] {m['content']} [/INST]\n"
        else:
            prompt += m["content"] + "\n"

    async with httpx.AsyncClient(timeout=120) as client:

        response = await client.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers={
                "Authorization": f"Bearer {HF_TOKEN}"
            },
            json={
                "inputs": prompt,
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": 300,
                    "return_full_text": False
                }
            }
        )

        response.raise_for_status()

        result = response.json()

        if isinstance(result, list):
            return result[0]["generated_text"].strip()

        if "generated_text" in result:
            return result["generated_text"].strip()

        raise HTTPException(
            status_code=500,
            detail=str(result)
        )


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    try:
        template = template_env.get_template("index.html")
        return template.render()
    except Exception as e:
        import traceback
        traceback.print_exc()
        return HTMLResponse(f"<pre>{traceback.format_exc()}</pre>", status_code=500)


@app.get("/health")
async def health():
    """Health check; optionally verify Ollama is reachable when not using HF."""
    if not USE_HUGGINGFACE:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
                if r.status_code != 200:
                    return {"status": "degraded", "ollama": "unreachable"}
                return {"status": "ok", "backend": "ollama", "model": OLLAMA_MODEL}
        except Exception as e:
            return {"status": "degraded", "ollama": str(e)}
    return {"status": "ok", "backend": "huggingface", "model": HF_MODEL}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, session_id: str = DEFAULT_SESSION):
    """
    Main chat endpoint.
    - Appends user message to history, builds prompt with last N messages.
    - Calls Ollama or Hugging Face, then appends assistant reply to history.
    - Returns the reply and updated history for the frontend.
    """
    user_message = request.message.strip()
    temperature = request.temperature
    history = get_history(session_id)

    # Optional: sync with frontend-provided history (e.g. after page refresh)
    if request.history:
        history.clear()
        for m in request.history[-MAX_HISTORY_MESSAGES:]:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                history.append({"role": m["role"], "content": m["content"]})

    # Build messages from history + current user message (same format for both backends)
    messages = build_messages(history, user_message)
  try:
    if USE_HUGGINGFACE:
        reply = await call_huggingface(messages, temperature)
        model_used = HF_MODEL
    else:
        reply = await call_ollama(messages, temperature)
        model_used = OLLAMA_MODEL

except Exception as e:
    import traceback
    traceback.print_exc()

    return JSONResponse(
        status_code=500,
        content={
            "error": str(e),
            "traceback": traceback.format_exc()
        }
    )

except Exception as e:
    import traceback
    traceback.print_exc()
    raise HTTPException(status_code=500, detail=str(e))

    # Append to conversation memory (last 5 message pairs)
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": reply})

    return ChatResponse(
        reply=reply,
        model_used=model_used,
        history=[{"role": m["role"], "content": m["content"]} for m in history],
    )


# Optional: mount static files if you add CSS/JS files later
# app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
