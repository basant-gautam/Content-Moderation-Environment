from __future__ import annotations
from typing import Any, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Internal imports from your files
from dataset import load_dataset
from environment import ContentModerationEnv
from moderation_logic import moderate_text

app = FastAPI(
    title="AI Content Moderation Environment",
    description="OpenEnv compliant moderation API.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the OpenEnv Environment
env = ContentModerationEnv()

# --- Request/Response Models ---
class ActionRequest(BaseModel):
    label: str
    action: str = "allow"

class ModerateRequest(BaseModel):
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

# --- Endpoints ---

@app.get("/")
def root():
    return {"message": "OpenEnv Moderation Server is Running", "docs": "/docs"}

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "dataset_size": len(load_dataset())}

@app.post("/reset")
def reset() -> Dict[str, Any]:
    """MANDATORY: Resets the environment and returns the first observation."""
    observation = env.reset()
    return {
        "observation": {
            "text": str(observation.get("text", "")),
            "metadata": dict(observation.get("metadata", {})),
        },
        "done": bool(env.done),
        "info": {},
    }

@app.post("/step")
def step(action: ActionRequest) -> Dict[str, Any]:
    """MANDATORY: Takes an action and returns next observation and done status."""
    try:
        action_dict = {"label": action.label, "action": action.action}
        result = env.step(action_dict)
        observation = result.get("observation") or {"text": "", "metadata": {}}
        return {
            "observation": {
                "text": str(observation.get("text", "")),
                "metadata": dict(observation.get("metadata", {})),
            },
            "done": bool(result.get("done", False)),
            "info": dict(result.get("info", {})),
        }
    except Exception as exc:
        return {
            "observation": {"text": "", "metadata": {}},
            "done": True,
            "info": {"error": "step_failed", "message": str(exc)},
        }

@app.get("/state")
def get_state():
    """MANDATORY: Returns the current state of the environment."""
    return env.state()

# Optional: Keep your old moderate endpoint for backward compatibility
@app.post("/moderate")
def moderate(request: ModerateRequest):
    result = moderate_text(request.text, request.metadata)
    return result