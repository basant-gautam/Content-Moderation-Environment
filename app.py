from __future__ import annotations

import random
from typing import Any, Dict, Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dataset import load_dataset
from moderation_logic import moderate_text

app = FastAPI(
    title="AI Content Moderation Environment",
    description="Lightweight moderation API backed by a deterministic heuristic policy.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModerateRequest(BaseModel):
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModerateResponse(BaseModel):
    label: Literal["safe", "spam", "hate", "violence"]
    action: Literal["allow", "delete", "flag", "escalate"]


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "dataset_size": len(load_dataset())}


@app.post("/moderate", response_model=ModerateResponse)
def moderate(request: ModerateRequest) -> ModerateResponse:
    result = moderate_text(request.text, request.metadata)
    return ModerateResponse(**result)


@app.get("/demo")
def demo() -> Dict[str, Any]:
    example = random.choice(load_dataset())
    result = moderate_text(example["text"], example.get("metadata", {}))
    return {
        "input": {
            "id": example["id"],
            "text": example["text"],
            "metadata": example.get("metadata", {}),
        },
        "output": result,
    }
