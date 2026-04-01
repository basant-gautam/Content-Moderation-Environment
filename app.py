from __future__ import annotations
import random
from typing import Any, Dict, Literal, Optional
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
    action: str = "flag"

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
def reset():
    """MANDATORY: Resets the environment and returns the first observation."""
    observation = env.reset()
    return {"observation": observation, "done": env.done}

@app.post("/step")
def step(action: ActionRequest):
    """MANDATORY: Takes an action and returns next observation, reward, and done status."""
    # Converting request to dict for environment compatibility
    action_dict = {"label": action.label, "action": action.action}
    result = env.step(action_dict)
    return result

@app.get("/state")
def get_state():
    """MANDATORY: Returns the current state of the environment."""
    return env.state()

# Optional: Keep your old moderate endpoint for backward compatibility
@app.post("/moderate")
def moderate(request: ModerateRequest):
    result = moderate_text(request.text, request.metadata)
    return result