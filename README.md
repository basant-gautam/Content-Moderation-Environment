# Stateful Hybrid AI Content Moderation Environment

## What this project does

This project is a full moderation simulation stack:

- Frontend UI for manual testing
- FastAPI backend for moderation endpoints
- Hybrid moderation engine:
  - Groq LLM (primary)
  - Rule-based fallback (if Groq key is missing or API call fails)
- OpenEnv-style environment for step-based evaluation
- Deterministic grader for reproducible scoring

It is designed for hackathons and real moderation prototypes.

## End-to-end flow

User Input -> Frontend/API -> Hybrid Moderation -> Label + Action -> Scoring/Decision

## Project components

- app.py
  - FastAPI server
  - Endpoints: /, /health, /moderate, /reset, /step, /state, /docs
- moderation_logic.py
  - Groq client via OpenAI-compatible SDK
  - strict JSON moderation output parsing
  - deterministic fallback rules
- environment.py
  - reset(), step(action), state()
  - state fields include user_risk_score and content_status
  - multi-step hard examples support
- dataset.py
  - diverse examples with severity and task levels
- grader.py
  - deterministic reward and score logic
  - normalized score in [0, 1]
- frontend/index.html
  - browser UI for testing moderation calls
- run_app.py
  - starts backend and opens frontend automatically

## Requirements

- Windows (recommended)
- Python 3.11+ installed
- internet connection for Groq LLM mode

## 1) Install dependencies

Run from the project root:

```powershell
python -m pip install -r requirements.txt
```

## 2) Configure Groq API key

Temporary key for current terminal session:

```powershell
$env:GROQ_API_KEY="YOUR_GROQ_KEY"
```

Persist key for future terminals:

```powershell
setx GROQ_API_KEY "YOUR_GROQ_KEY"
```

Optional model override:

```powershell
$env:GROQ_MODEL="llama-3.1-8b-instant"
```

Notes:

- Default model preference includes llama3-8b-8192 for compatibility with earlier setup.
- If a model is unavailable/decommissioned, moderation_logic.py automatically tries fallback models.

## 3) Run backend + frontend automatically

Single command:

```powershell
python run_app.py
```

What it does:

- starts FastAPI backend at http://127.0.0.1:8000
- waits for /health to become ready
- opens frontend/index.html in your default browser

Stop server:

- press Ctrl+C in the same terminal

## 4) API endpoints

### GET /

Returns a basic service message and docs path.

### GET /health

Returns service health and dataset size.

### POST /moderate

Request body:

```json
{
  "text": "your comment",
  "metadata": {}
}
```

Response body:

```json
{
  "label": "safe|spam|hate|violence",
  "action": "allow|delete|flag|escalate"
}
```

### POST /reset

- resets the stateful moderation environment
- returns first observation and done flag

### POST /step

Request body:

```json
{
  "label": "safe|spam|hate|violence",
  "action": "allow|delete|flag|escalate"
}
```

Returns next observation, reward, done, and grading info.

### GET /state

Returns current environment state, aggregate reward, and dynamic fields such as user_risk_score/content_status.

### Interactive docs

- http://127.0.0.1:8000/docs

## 5) Frontend usage

If launched via run_app.py, frontend opens automatically.

Manual usage:

- open frontend/index.html in browser
- enter text
- click Moderate
- label and action are shown
- optional: click Load Demo only if a /demo endpoint is added (the current backend does not implement /demo)

## 6) Evaluation mode

Run OpenEnv-style inference client:

```powershell
python inference.py
```

Environment variables used by inference.py:

- API_BASE_URL (default: https://basant-levi-ai-content-moderation-openenv.hf.space)
- MODEL_NAME (default: llama-3.1-8b-instant)
- HF_TOKEN or OPENAI_API_KEY (required)
- OPENAI_BASE_URL (optional, default: https://api.groq.com/openai/v1)

## 7) Docker mode

Build:

```powershell
docker build -t ai-moderation-openenv .
```

Run:

```powershell
docker run -p 8000:8000 ai-moderation-openenv
```

## Troubleshooting

### Backend starts but frontend does not open

- manually open frontend/index.html
- or open API docs at http://127.0.0.1:8000/docs

### Moderation seems rule-based only

- verify key exists in terminal:

```powershell
echo $env:GROQ_API_KEY
```

- ensure internet access
- check model availability on Groq
- fallback to rules is expected when Groq call fails

### Port already in use

- stop existing process on port 8000
- restart with python run_app.py

### PowerShell profile parse errors appear before run

- this can happen if your local PowerShell profile has syntax issues
- project can still run; fix profile separately if needed

## Security best practice

- do not hardcode API keys into source files
- keep keys in environment variables
- rotate keys immediately if exposed publicly
