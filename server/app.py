from __future__ import annotations
import math
from typing import Any, Dict
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from server.dataset import load_dataset
from server.environment import ContentModerationEnv
from server.moderation_logic import moderate_text

app = FastAPI(title="AI Content Moderation Environment")

TASK_SCORE_MIN = 0.0
TASK_SCORE_MAX = 1.0

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

env = ContentModerationEnv()
current_task_id = "all"

class ActionRequest(BaseModel):
    label: str
    action: str = "allow"

class ModerateRequest(BaseModel):
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

def _bounded_score(value: Any, default: float = 0.01) -> float:
    try: numeric = float(value)
    except (TypeError, ValueError): numeric = default
    if not math.isfinite(numeric): numeric = default
    return round(min(0.99, max(0.01, numeric)), 4)

def _normalize_task_name(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if "easy" in text: return "easy"
    if "medium" in text: return "medium"
    if "hard" in text: return "hard"
    return ""

def _extract_task_name(body: Dict[str, Any], request: Request) -> str:
    candidates = [body.get("task"), body.get("task_id"), request.query_params.get("task")]
    for candidate in candidates:
        normalized = _normalize_task_name(candidate)
        if normalized: return normalized
    return ""

def _extract_action_payload(payload: Dict[str, Any]) -> Dict[str, str]:
    return {
        "label": str(payload.get("label", "safe")),
        "action": str(payload.get("action", payload.get("moderation_action", "allow"))),
    }

def _task_entry(task_id: str, name: str, description: str) -> Dict[str, Any]:
    return {
        "id": task_id, "name": name, "difficulty": task_id, "description": description,
        "grader_available": True, "score_field": "info.task_score",
        "score_range": {"min": TASK_SCORE_MIN, "max": TASK_SCORE_MAX},
        "grader": {"grader_available": True, "type": "scalar", "score_field": "info.task_score", "score_range": {"min": TASK_SCORE_MIN, "max": TASK_SCORE_MAX}}
    }

@app.get("/")
def root(): return {"message": "Running", "docs": "/docs"}

@app.get("/health")
def health() -> Dict[str, Any]: return {"status": "ok"}

@app.get("/tasks")
def get_tasks() -> Dict[str, Any]:
    tasks = [_task_entry("easy", "Easy", ""), _task_entry("medium", "Medium", ""), _task_entry("hard", "Hard", "")]
    return {"tasks": tasks, "task_count": 3}

@app.post("/reset")
async def reset(request: Request) -> Dict[str, Any]:
    global env, current_task_id
    try: body = await request.json()
    except Exception: body = {}

    task_name = _extract_task_name(body, request)
    if task_name in ("easy", "medium", "hard"):
        env = ContentModerationEnv(examples=load_dataset(task_name))
        current_task_id = task_name
    else:
        env = ContentModerationEnv(examples=load_dataset())
        current_task_id = "all"
        
    observation = env.reset()
    return {
        "observation": {"text": str(observation.get("text", "")), "metadata": dict(observation.get("metadata", {}))},
        "done": bool(env.done),
        "info": {"task_id": current_task_id if current_task_id != "all" else "easy", "task_score": 0.01, "score": 0.01},
    }

@app.post("/step")
async def step(request: Request) -> Dict[str, Any]:
    try:
        payload = await request.json()
        action_dict = _extract_action_payload(payload)
        result = env.step(action_dict)
        observation = result.get("observation") or {"text": "", "metadata": {}}
        info = dict(result.get("info", {}))
        
        info["task_id"] = current_task_id if current_task_id != "all" else "easy"

        safe_score = _bounded_score(info.get("task_score", info.get("score", 0.01)))
        info["task_score"] = safe_score
        info["score"] = safe_score
        info["normalized_score"] = safe_score
        
        return {
            "observation": {"text": str(observation.get("text", "")), "metadata": dict(observation.get("metadata", {}))},
            "reward": float(info.get("reward", 0.0)),
            "score": safe_score,
            "done": bool(result.get("done", False)),
            "info": info,
        }
    except Exception as exc:
        return {"observation": {"text": "", "metadata": {}}, "reward": 0.0, "score": 0.01, "done": True, "info": {"task_score": 0.01}}

@app.get("/state")
def get_state(): return env.state()

@app.post("/moderate")
def moderate(request: ModerateRequest): return moderate_text(request.text, request.metadata)

def main(): return app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)