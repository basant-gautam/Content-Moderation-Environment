from __future__ import annotations

from typing import Any, Dict, Iterable, List

from dataset import VALID_ACTIONS, VALID_LABELS

INVALID_OUTPUT_PENALTY = 0.6
WRONG_ACTION_PENALTY = 0.25
UNSAFE_ALLOW_PENALTY = 0.6
OVERMODERATION_PENALTY = 0.25

SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3}


def validate_prediction(prediction: Any) -> List[str]:
    errors: List[str] = []

    if not isinstance(prediction, dict):
        return ["prediction must be a JSON object"]

    if "label" not in prediction:
        errors.append("missing label")
    if "action" not in prediction:
        errors.append("missing action")

    label = prediction.get("label")
    action = prediction.get("action")

    if label is not None and label not in VALID_LABELS:
        errors.append(f"invalid label: {label}")
    if action is not None and action not in VALID_ACTIONS:
        errors.append(f"invalid action: {action}")

    return errors


def grade_prediction(prediction: Any, example: Dict[str, Any]) -> Dict[str, Any]:
    errors = validate_prediction(prediction)
    expected = example["expected"]

    if errors:
        raw_reward = -INVALID_OUTPUT_PENALTY
        return {
            "raw_reward": raw_reward,
            "score": 0.0,
            "valid": False,
            "errors": errors,
            "breakdown": {
                "label_reward": 0.0,
                "action_reward": 0.0,
                "severity_reward": 0.0,
                "penalty": raw_reward,
            },
            "expected": expected,
        }

    predicted_label = prediction["label"]
    predicted_action = prediction["action"]

    expected_severity = example.get("severity", "low")

    label_reward = 0.35 if predicted_label == expected["label"] else 0.0
    action_reward = 0.25 if predicted_action == expected["action"] else 0.0
    severity_handling_reward = 0.2 if predicted_action in example["severity_actions"] else 0.0

    predicted_severity = "low"
    if predicted_action in ("delete", "escalate") or predicted_label in ("hate", "violence"):
        predicted_severity = "high"
    elif predicted_action == "flag" or predicted_label == "spam":
        predicted_severity = "medium"

    severity_diff = abs(SEVERITY_ORDER.get(predicted_severity, 1) - SEVERITY_ORDER.get(expected_severity, 1))
    if severity_diff == 0:
        severity_bonus = 0.2
    elif severity_diff == 1:
        severity_bonus = 0.1
    else:
        severity_bonus = 0.0

    penalty = 0.0
    if predicted_action not in example["severity_actions"]:
        penalty -= WRONG_ACTION_PENALTY

    if expected["label"] != "safe" and predicted_action == "allow":
        penalty -= UNSAFE_ALLOW_PENALTY

    if expected["label"] == "safe" and predicted_action != "allow":
        penalty -= OVERMODERATION_PENALTY

    raw_reward = round(
        label_reward + action_reward + severity_handling_reward + severity_bonus + penalty,
        4,
    )
    score = round(min(1.0, max(0.0, raw_reward)), 4)

    return {
        "raw_reward": raw_reward,
        "score": score,
        "valid": True,
        "errors": [],
        "breakdown": {
            "label_reward": label_reward,
            "action_reward": action_reward,
            "severity_handling_reward": severity_handling_reward,
            "severity_bonus": severity_bonus,
            "penalty": penalty,
        },
        "expected": expected,
        "expected_severity": expected_severity,
        "predicted_severity": predicted_severity,
    }


def average_score(scores: Iterable[float]) -> float:
    scores = list(scores)
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 4)
