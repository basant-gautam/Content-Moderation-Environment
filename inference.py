from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from environment import ContentModerationEnv
from grader import average_score
from moderation_logic import moderate_text

SYSTEM_PROMPT = """You are a content moderation agent.
Return only a JSON object with this exact schema:
{"label":"safe|spam|hate|violence","action":"allow|delete|flag|escalate"}
Choose a single best label and action."""

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _parse_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}


def ask_llm(client: Any, model: str, observation: Dict[str, Any]) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(observation, ensure_ascii=True),
            },
        ],
    )
    content = response.choices[0].message.content or "{}"
    return _parse_json_object(content)


def run_episode(model: str, local_only: bool, max_steps: int | None) -> Dict[str, Any]:
    env = ContentModerationEnv(max_steps=max_steps)
    env.reset()

    groq_api_key = os.getenv("GROQ_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    provider = "local_heuristic"
    use_llm = not local_only and bool(groq_api_key or openai_api_key)
    client = None
    if use_llm:
        try:
            from openai import OpenAI
        except ImportError:
            use_llm = False
        else:
            if groq_api_key:
                provider = "groq"
                model = os.getenv("GROQ_MODEL", model or "llama-3.1-8b-instant")
                client = OpenAI(base_url=GROQ_BASE_URL, api_key=groq_api_key)
            elif openai_api_key:
                provider = "openai"
                client = OpenAI(api_key=openai_api_key)
            else:
                use_llm = False

    results: List[Dict[str, Any]] = []

    while True:
        current_state = env.state()
        observation = current_state["observation"]
        if observation is None:
            break

        if use_llm and client is not None:
            try:
                prediction = ask_llm(client, model, observation)
            except Exception:
                provider = "local_heuristic"
                prediction = moderate_text(observation["text"], observation["metadata"])
        else:
            prediction = moderate_text(observation["text"], observation["metadata"])

        transition = env.step(prediction)
        results.append(transition["info"])

        if transition["done"]:
            break

    level_scores: Dict[str, List[float]] = {"easy": [], "medium": [], "hard": []}
    for result in results:
        level_scores[result["level"]].append(result["score"])

    summary = {
        "mode": provider if use_llm else "local_heuristic",
        "model": model if use_llm else "rule_based",
        "evaluated_examples": len(results),
        "final_score": average_score(result["score"] for result in results),
        "level_scores": {
            level: average_score(scores) for level, scores in level_scores.items() if scores
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run moderation evaluation across the dataset.")
    parser.add_argument(
        "--model",
        default=os.getenv("GROQ_MODEL", os.getenv("OPENAI_MODEL", "llama-3.1-8b-instant")),
        help="Model name used by LLM provider (Groq preferred).",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Skip the OpenAI API and use the local heuristic baseline.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Limit the number of evaluation steps.",
    )
    args = parser.parse_args()

    summary = run_episode(args.model, args.local_only, args.max_steps)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
