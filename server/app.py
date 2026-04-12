from __future__ import annotations
import math
from typing import Any, Dict
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from server.dataset import load_dataset
from server.environment import ContentModerationEnv
from server.moderation_logic import moderate_text

app = FastAPI(title="AI Content Moderation Environment", version="1.0.0")

# STRICT OPEN-INTERVAL SCORE RANGE FOR TASK VALIDATION
TASK_SCORE_MIN = 0.01
TASK_SCORE_MAX = 0.99

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
    if isinstance(raw, dict):
        for key in ("id", "name", "task_id", "task_name", "task", "taskName"):
            if key in raw:
                normalized = _normalize_task_name(raw.get(key))
                if normalized:
                    return normalized
        return ""
    if isinstance(raw, list):
        for item in raw:
            normalized = _normalize_task_name(item)
            if normalized:
                return normalized
        return ""
    text = str(raw or "").strip().lower()
    if "easy" in text: return "easy"
    if "medium" in text: return "medium"
    if "hard" in text: return "hard"
    return ""

def _extract_task_name(body: Dict[str, Any], request: Request) -> str:
    candidates = [
        body.get("task"),
        body.get("task_id"),
        body.get("task_name"),
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
        if normalized: return normalized
    return ""

def _extract_action_payload(payload: Dict[str, Any]) -> Dict[str, str]:
    return {"label": str(payload.get("label", "safe")), "action": str(payload.get("action", payload.get("moderation_action", "allow")))}

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
def health() -> Dict[str, Any]: return {"status": "ok", "dataset_size": len(load_dataset())}

@app.get("/tasks")
def get_tasks() -> Dict[str, Any]:
    tasks = [
        _task_entry("easy", "Easy Moderation", "Spam-focused moderation samples."),
        _task_entry("medium", "Medium Moderation", "Hate and abusive content detection."),
        _task_entry("hard", "Hard Moderation", "Context-aware and multi-step moderation edge cases."),
    ]
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
    task_id = str(observation.get("metadata", {}).get("level", current_task_id))
    if task_id not in ("easy", "medium", "hard"):
        task_id = current_task_id if current_task_id in ("easy", "medium", "hard") else "easy"
    return {
        "observation": {"text": str(observation.get("text", "")), "metadata": dict(observation.get("metadata", {}))},
        "done": bool(env.done),
        "info": {"task_id": task_id, "task_score": 0.01, "score": 0.01},
    }

@app.post("/step")
async def step(request: Request) -> Dict[str, Any]:
    try:
        payload = await request.json()
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
        safe_reward = _bounded_score(info.get("reward", 0.01))
        safe_avg = _bounded_score(info.get("episode_average_score", safe_score))

        info["task_score"] = safe_avg
        info["score"] = safe_score
        info["normalized_score"] = safe_score
        info["reward"] = safe_reward
        info["episode_average_score"] = safe_avg
        info["episode_average_reward"] = safe_reward
        
        return {
            "observation": {"text": str(observation.get("text", "")), "metadata": dict(observation.get("metadata", {}))},
            "reward": safe_reward,
            "score": safe_score,
            "done": bool(result.get("done", False)),
            "info": info,
        }
    except Exception as exc:
        safe_err = 0.01
        return {"observation": {"text": "", "metadata": {}}, "reward": safe_err, "score": safe_err, "done": True, "info": {"task_score": safe_err, "score": safe_err}}

@app.get("/state")
def get_state(): return env.state()

@app.post("/moderate")
def moderate(request: ModerateRequest): return moderate_text(request.text, request.metadata)

def main(): return app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)