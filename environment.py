from __future__ import annotations

from typing import Any, Dict, List, Optional

from dataset import load_dataset
from grader import average_score, grade_prediction


class ContentModerationEnv:
    """Minimal OpenEnv-style environment for moderation episodes."""

    def __init__(
        self,
        examples: Optional[List[Dict[str, Any]]] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        self.examples = examples or load_dataset()
        if max_steps is None:
            self.max_steps = len(self.examples)
        else:
            self.max_steps = max(0, min(max_steps, len(self.examples)))
        self.current_index = 0
        self.example_step = 1
        self.total_reward = 0.0
        self.done = False
        self.user_risk_score = 0.0
        self.content_status = "open"
        self.history: List[Dict[str, Any]] = []

    def _observation_for_index(self, index: int) -> Optional[Dict[str, Any]]:
        if index >= self.max_steps:
            return None
        example = self.examples[index]
        metadata = dict(example["metadata"])
        metadata["severity"] = example.get("severity", "low")
        metadata["level"] = example.get("level", "easy")
        metadata["review_step"] = self.example_step
        metadata["required_steps"] = max(1, int(example.get("required_steps", 1)))
        return {"text": example["text"], "metadata": metadata}

    def _required_steps(self, example: Dict[str, Any]) -> int:
        return max(1, int(example.get("required_steps", 1)))

    def _update_dynamic_state(self, predicted_action: str, expected_label: str, severity: str) -> None:
        if predicted_action in ("flag", "escalate"):
            self.content_status = "flagged"
        elif predicted_action in ("allow", "delete"):
            self.content_status = "resolved"

        risk_delta = 0.0
        if predicted_action == "allow" and expected_label != "safe":
            risk_delta += 0.4 if severity == "high" else 0.25
        elif predicted_action in ("flag", "escalate"):
            risk_delta += 0.05
        elif predicted_action == "delete" and expected_label == "safe":
            risk_delta += 0.2
        else:
            risk_delta -= 0.05

        self.user_risk_score = round(min(1.0, max(0.0, self.user_risk_score + risk_delta)), 4)

    def reset(self) -> Dict[str, Any]:
        self.current_index = 0
        self.example_step = 1
        self.total_reward = 0.0
        self.done = self.max_steps == 0
        self.user_risk_score = 0.0
        self.content_status = "open"
        self.history = []
        return self._observation_for_index(self.current_index) or {"text": "", "metadata": {}}

    def state(self) -> Dict[str, Any]:
        return {
            "step": self.current_index,
            "max_steps": self.max_steps,
            "done": self.done,
            "total_reward": round(self.total_reward, 4),
            "average_reward": average_score(item["reward"] for item in self.history),
            "user_risk_score": self.user_risk_score,
            "content_status": self.content_status,
            "example_step": self.example_step,
            "observation": None if self.done else self._observation_for_index(self.current_index),
        }

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if self.done or self.current_index >= self.max_steps:
            self.done = True
            return {
                "observation": None,
                "reward": 0.0,
                "done": True,
                "info": {"error": "episode_complete"},
            }

        example = self.examples[self.current_index]
        grading = grade_prediction(action, example)
        reward = grading["raw_reward"]
        score = grading["score"]
        required_steps = self._required_steps(example)

        predicted_action = ""
        if isinstance(action, dict):
            predicted_action = str(action.get("action", ""))
        self._update_dynamic_state(
            predicted_action=predicted_action,
            expected_label=example["expected"]["label"],
            severity=example.get("severity", "low"),
        )

        self.total_reward += reward
        self.history.append(
            {
                "example_id": example["id"],
                "level": example["level"],
                "severity": example.get("severity", "low"),
                "example_step": self.example_step,
                "required_steps": required_steps,
                "prediction": action,
                "expected": example["expected"],
                "reward": reward,
                "score": score,
                "grading": grading,
            }
        )

        is_last_step_for_example = self.example_step >= required_steps
        if is_last_step_for_example:
            self.current_index += 1
            self.example_step = 1
            self.content_status = "open" if self.current_index < self.max_steps else "resolved"
        else:
            self.example_step += 1
            if self.content_status == "open":
                self.content_status = "flagged"

        self.done = self.current_index >= self.max_steps
        next_observation = None if self.done else self._observation_for_index(self.current_index)

        return {
            "observation": next_observation,
            "reward": reward,
            "done": self.done,
            "info": {
                "example_id": example["id"],
                "level": example["level"],
                "severity": example.get("severity", "low"),
                "example_step": self.history[-1]["example_step"],
                "required_steps": required_steps,
                "prediction": action,
                "expected": example["expected"],
                "reward": reward,
                "score": score,
                "normalized_score": score,
                "breakdown": grading["breakdown"],
                "valid": grading["valid"],
                "errors": grading["errors"],
                "content_status": self.content_status,
                "user_risk_score": self.user_risk_score,
                "episode_average_reward": average_score(item["reward"] for item in self.history),
                "episode_average_score": average_score(item["score"] for item in self.history),
            },
        }


OpenEnvModerationEnv = ContentModerationEnv
