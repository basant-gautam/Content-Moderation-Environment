from __future__ import annotations
import math
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
current_task_id = "all"

# --- Request/Response Models ---
class ActionRequest(BaseModel):
    label: str
    action: str = "allow"

class ModerateRequest(BaseModel):
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


def _bounded_score(value: Any, default: float = 0.01) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    if not math.isfinite(numeric):
        numeric = default
    return round(min(0.99, max(0.01, numeric)), 4)


def _normalize_task_name(raw: Any) -> str:
    if isinstance(raw, dict):
        for key in ("id", "name", "task_id", "task_name", "task"):
            if key in raw:
                return _normalize_task_name(raw.get(key))
        return ""
    if isinstance(raw, list):
        for item in raw:
            normalized = _normalize_task_name(item)
            if normalized:
                return normalized
        return ""
    text = str(raw or "").strip().lower()
    if "easy" in text:
        return "easy"
    if "medium" in text:
        return "medium"
    if "hard" in text:
        return "hard"
    return ""


def _extract_task_name(body: Dict[str, Any], request: Request) -> str:
    candidates = [
        body.get("task"),
        body.get("task_id"),
        body.get("task_name"),
        body.get("taskId"),
        body.get("taskName"),
        body.get("id"),
        body.get("name"),
        body.get("config"),
        request.query_params.get("task"),
        request.query_params.get("task_id"),
        request.query_params.get("task_name"),
    ]
    for candidate in candidates:
        normalized = _normalize_task_name(candidate)
        if normalized:
            return normalized
    return ""


def _extract_action_payload(payload: Dict[str, Any]) -> Dict[str, str]:
    if "label" in payload and "action" in payload:
        return {
            "label": str(payload.get("label", "safe")),
            "action": str(payload.get("action", "allow")),
        }

    nested = payload.get("action")
    if isinstance(nested, dict):
        return {
            "label": str(nested.get("label", payload.get("label", "safe"))),
            "action": str(nested.get("action", payload.get("moderation_action", "allow"))),
        }

    return {
        "label": str(payload.get("label", "safe")),
        "action": str(payload.get("moderation_action", "allow")),
    }

# --- Endpoints ---
@app.get("/")
def root():
    return {"message": "OpenEnv Moderation Server is Running", "docs": "/docs"}

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "dataset_size": len(load_dataset())}

@app.get("/tasks")
def get_tasks() -> Dict[str, Any]:
    tasks = [
        {
            "id": "easy",
            "name": "Easy Moderation",
            "grader_available": True,
            "score_field": "info.task_score",
            "score_range": {"min": 0.01, "max": 0.99},
            "grader": {
                "grader_available": True,
                "type": "scalar",
                "score_field": "info.task_score",
                "score_range": {"min": 0.01, "max": 0.99},
            },
        },
        {
            "id": "medium",
            "name": "Medium Moderation",
            "grader_available": True,
            "score_field": "info.task_score",
            "score_range": {"min": 0.01, "max": 0.99},
            "grader": {
                "grader_available": True,
                "type": "scalar",
                "score_field": "info.task_score",
                "score_range": {"min": 0.01, "max": 0.99},
            },
        },
        {
            "id": "hard",
            "name": "Hard Moderation",
            "grader_available": True,
            "score_field": "info.task_score",
            "score_range": {"min": 0.01, "max": 0.99},
            "grader": {
                "grader_available": True,
                "type": "scalar",
                "score_field": "info.task_score",
                "score_range": {"min": 0.01, "max": 0.99},
            },
        },
    ]
    return {
        "tasks": tasks,
        "easy": {"grader_available": True, "score_field": "info.task_score"},
        "medium": {"grader_available": True, "score_field": "info.task_score"},
        "hard": {"grader_available": True, "score_field": "info.task_score"},
        "task_count": 3,
    }

@app.post("/reset")
async def reset(request: Request) -> Dict[str, Any]:
    """MANDATORY: Resets the environment dynamically based on requested task."""
    global env, current_task_id
    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}

    try:
        task_name = _extract_task_name(body, request)
        if task_name in ("easy", "medium", "hard"):
            env = ContentModerationEnv(examples=load_dataset(task_name))
            current_task_id = task_name
        else:
            env = ContentModerationEnv(examples=load_dataset())
            current_task_id = "all"
    except Exception:
        env = ContentModerationEnv(examples=load_dataset())
        current_task_id = "all"
        
    observation = env.reset()
    task_id = str(observation.get("metadata", {}).get("level", current_task_id))
    if task_id not in ("easy", "medium", "hard"):
        task_id = current_task_id if current_task_id in ("easy", "medium", "hard") else "easy"

    return {
        "observation": {
            "text": str(observation.get("text", "")),
            "metadata": dict(observation.get("metadata", {})),
        },
        "done": bool(env.done),
        "info": {"task_id": task_id, "task_score": 0.01, "score": 0.01},
    }

@app.post("/step")
async def step(request: Request) -> Dict[str, Any]:
    """MANDATORY: Takes an action and returns next observation and done status."""
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}

    try:
        action_dict = _extract_action_payload(payload)
        result = env.step(action_dict)
        observation = result.get("observation") or {"text": "", "metadata": {}}
        info = dict(result.get("info", {}))
        if not info.get("task_id"):
            if env.history:
                info["task_id"] = str(env.history[-1].get("task_id", current_task_id))
            else:
                info["task_id"] = current_task_id if current_task_id in ("easy", "medium", "hard") else "easy"

        safe_score = _bounded_score(info.get("task_score", info.get("score", 0.01)))
        info["task_score"] = safe_score
        info["score"] = safe_score
        info["normalized_score"] = safe_score
        
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
            "info": {"error": "step_failed", "message": str(exc), "task_score": 0.01, "score": 0.01, "task_id": current_task_id if current_task_id in ("easy", "medium", "hard") else "easy"},
        }

@app.get("/state")
def get_state():
    return env.state()

@app.post("/moderate")
def moderate(request: ModerateRequest):
    return moderate_text(request.text, request.metadata)


@app.get("/demo")
def demo() -> Dict[str, Any]:
    sample_text = "Win a free iPhone today. Click here now and claim your prize!"
    prediction = moderate_text(sample_text, {"channel": "comment", "reports": 1, "contains_url": False})
    return {
        "sample": {"text": sample_text},
        "prediction": prediction,
        "tasks": ["easy", "medium", "hard"],
    }

def main():
    return app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
