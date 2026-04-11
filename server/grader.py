from __future__ import annotations
from typing import Any, Dict, Iterable, List
from server.dataset import VALID_ACTIONS, VALID_LABELS

INVALID_OUTPUT_PENALTY = 0.6
WRONG_ACTION_PENALTY = 0.25
UNSAFE_ALLOW_PENALTY = 0.6
OVERMODERATION_PENALTY = 0.25

SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3}

def validate_prediction(prediction: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(prediction, dict): return ["prediction must be a JSON object"]
    if "label" not in prediction: errors.append("missing label")
    if "action" not in prediction: errors.append("missing action")
    label = prediction.get("label")
    action = prediction.get("action")
    if label is not None and label not in VALID_LABELS: errors.append(f"invalid label: {label}")
    if action is not None and action not in VALID_ACTIONS: errors.append(f"invalid action: {action}")
    return errors

def grade_prediction(prediction: Any, example: Dict[str, Any]) -> Dict[str, Any]:
    errors = validate_prediction(prediction)
    expected = example["expected"]

    if errors:
        return {
            "raw_reward": -INVALID_OUTPUT_PENALTY,
            "score": 0.01,
            "valid": False,
            "errors": errors,
            "breakdown": {},
            "expected": expected,
            "expected_severity": example.get("severity", "low"),
        }

    expected_severity = example.get("severity", "low")
    predicted_action = str(prediction.get("action", ""))
    
    label_reward = 0.5 if prediction.get("label") == expected["label"] else 0.0
    action_reward = 0.5 if predicted_action == expected["action"] else 0.0

    severity_handling_reward = 0.0
    if expected["label"] != "safe":
        if expected_severity == "high" and predicted_action == "escalate": severity_handling_reward = 0.2
        elif expected_severity == "medium" and predicted_action in ("delete", "flag"): severity_handling_reward = 0.1
        elif expected_severity == "low" and predicted_action == "flag": severity_handling_reward = 0.05

    predicted_severity = "low"
    if predicted_action == "escalate": predicted_severity = "high"
    elif predicted_action in ("delete", "flag"): predicted_severity = "medium"

    severity_diff = abs(SEVERITY_ORDER.get(predicted_severity, 1) - SEVERITY_ORDER.get(expected_severity, 1))
    if severity_diff == 0: severity_bonus = 0.2
    elif severity_diff == 1: severity_bonus = 0.1
    else: severity_bonus = 0.0

    penalty = 0.0
    if predicted_action not in example.get("severity_actions", []): penalty -= WRONG_ACTION_PENALTY
    if expected["label"] != "safe" and predicted_action == "allow": penalty -= UNSAFE_ALLOW_PENALTY
    if expected["label"] == "safe" and predicted_action != "allow": penalty -= OVERMODERATION_PENALTY

    raw_reward = round(label_reward + action_reward + severity_handling_reward + severity_bonus + penalty, 4)
    score = round(min(0.99, max(0.01, float(raw_reward))), 4)

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
    }

def average_score(scores: Iterable[float]) -> float:
    scores_list = list(scores)
    # THE FINAL FIX: Agar validator khali test run kare, toh 0.0 nahi 0.01 bhejenge
    if not scores_list:
        return 0.01
    avg = sum(scores_list) / len(scores_list)
    return round(min(0.99, max(0.01, float(avg))), 4)