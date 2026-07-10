# Free LLM Chatbot

A simple, modular chatbot using **Python + FastAPI** and a **free LLM** (Ollama locally or Hugging Face). No OpenAI, ChatGPT, Gemini, or paid APIs.

## Features

- **Prompt templating**: System message + user/assistant conversation structure
- **Conversation memory**: Keeps the last 5 messages for context
- **Temperature control**: Adjustable via the UI (0–2)
- **Simple frontend**: HTML + JavaScript; send messages and see replies in one place

## Project structure

```
.
├── app.py              # FastAPI backend + LLM calls (Ollama / Hugging Face)
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Chat UI
└── README.md           # This file
```

## Setup

### 1. Create a virtual environment (recommended)

```bash
cd "LLM-Prompt Project"
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Choose your LLM backend

#### Option A: Ollama (recommended — free, offline, no API key)

1. Install [Ollama](https://ollama.com) and start it.
2. Pull a model (run once):

   ```bash
   ollama run mistral
   # or: ollama run llama3
   # or: ollama run phi
   ```

3. Start the app (no env vars needed):

   ```bash
   python app.py
   # or: uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

4. Open **http://localhost:8000** in your browser.

Optional env vars for Ollama:

- `OLLAMA_BASE_URL` — default `http://localhost:11434`
- `OLLAMA_MODEL` — default `mistral` (use the same name as in `ollama run <model>`)

#### Option B: Hugging Face (free inference tier, needs token)

1. Get a free token: [Hugging Face → Settings → Access Tokens](https://huggingface.co/settings/tokens).
2. Run with:

   ```bash
   export USE_HF=true
   export HF_TOKEN=your_token_here
   python app.py
   ```

3. Open **http://localhost:8000**.

Optional: `HF_MODEL` (default: `mistralai/Mistral-7B-Instruct-v0.2`).

### 3. Run the app

From the project root:

```bash
python app.py
```

Then open **http://localhost:8000**. You should see the chat UI; type a message and click Send (or press Enter). The backend uses the last 5 messages as context and returns the model’s reply.

## API

- **GET /** — Serves the chat UI.
- **GET /health** — Health check; reports backend (ollama/huggingface) and model.
- **POST /api/chat** — Send a message and get a reply.

  Body (JSON):

  - `message` (string, required)
  - `temperature` (number, 0–2, default 0.7)
  - `history` (optional array of `{ role, content }` to sync history)

  Response: `{ "reply", "model_used", "history" }`.

## Notes

- **Ollama**: Fully free and offline; no API keys. Best for local use.
- **Hugging Face**: Free tier; requires `HF_TOKEN`. Good if you don’t want to run a model locally.
- Conversation memory is in-memory per session; restarting the server clears it.
