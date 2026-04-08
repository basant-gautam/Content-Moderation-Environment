from __future__ import annotations
from typing import Any, Dict
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Internal imports from your files
from server.dataset import load_dataset
from server.environment import ContentModerationEnv
from server.moderation_logic import moderate_text

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

@app.get("/tasks")
def get_tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {"id": "easy", "name": "Easy Moderation", "grader": {"grader_available": True, "score_field": "info.task_score", "score_range": {"min": 0.01, "max": 0.99}}},
            {"id": "medium", "name": "Medium Moderation", "grader": {"grader_available": True, "score_field": "info.task_score", "score_range": {"min": 0.01, "max": 0.99}}},
            {"id": "hard", "name": "Hard Moderation", "grader": {"grader_available": True, "score_field": "info.task_score", "score_range": {"min": 0.01, "max": 0.99}}},
        ],
        "easy": {"grader_available": True},
        "medium": {"grader_available": True},
        "hard": {"grader_available": True},
    }

@app.post("/reset")
async def reset(request: Request) -> Dict[str, Any]:
    """MANDATORY: Resets the environment dynamically based on requested task."""
    global env
    try:
        body = await request.json()
        task_name = body.get("task", body.get("task_id", body.get("task_name", "")))
        
        # Smart matching for task name
        if "easy" in task_name: task_name = "easy"
        elif "medium" in task_name: task_name = "medium"
        elif "hard" in task_name: task_name = "hard"
        
        if task_name in ["easy", "medium", "hard"]:
            env = ContentModerationEnv(examples=load_dataset(task_name))
        else:
            env = ContentModerationEnv(examples=load_dataset())
    except Exception:
        env = ContentModerationEnv(examples=load_dataset())
        
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
        info = dict(result.get("info", {}))
        
        # The ultimate safeguard against 0.0 or 1.0
        raw_score = float(info.get("score", 0.01))
        safe_score = min(0.99, max(0.01, raw_score))
        
        info["task_score"] = safe_score
        info["score"] = safe_score
        
        return {
            "observation": {
                "text": str(observation.get("text", "")),
                "metadata": dict(observation.get("metadata", {})),
            },
            "reward": float(info.get("reward", 0.0)),
            "score": safe_score,
            "done": bool(result.get("done", False)),
            "info": info,
        }
    except Exception as exc:
        return {
            "observation": {"text": "", "metadata": {}},
            "reward": 0.0,
            "score": 0.01,
            "done": True,
            "info": {"error": "step_failed", "message": str(exc), "task_score": 0.01},
        }

@app.get("/state")
def get_state():
    return env.state()

@app.post("/moderate")
def moderate(request: ModerateRequest):
    return moderate_text(request.text, request.metadata)

def main():
    return app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)