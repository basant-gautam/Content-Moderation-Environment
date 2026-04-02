from __future__ import annotations

import json
import os
import re
from typing import Any, Dict

from server.dataset import VALID_ACTIONS, VALID_LABELS

SAFE_CONTEXT_MARKERS = (
    "is hateful and should be reported",
    "condemns people who say",
    "this comment is spam",
    "quoted",
    "example of hate",
    "report this",
)

SAFE_METAPHOR_MARKERS = (
    "killed them in the finals",
    "kill for some coffee",
)

SPAM_KEYWORDS = (
    "win a free",
    "click here",
    "claim your prize",
    "guaranteed crypto profit",
    "double your money",
    "dm me",
    "followers fast",
    "cheap promo",
    "account is locked",
    "verify now",
    "sign up now",
    "get rich fast",
)

INDIVIDUAL_ABUSE_MARKERS = (
    "useless moron",
    "shut up forever",
    "idiot",
    "trash",
)

GROUP_ATTACK_PATTERNS = (
    r"\bthose immigrants\b.*\b(disgusting|leave)\b",
    r"\bwomen are\b.*\bstupid\b",
    r"\bthose refugees are animals\b",
    r"\bgo back to your country\b",
)

VIOLENCE_PATTERNS = (
    r"\bi will\b.*\b(kill|shoot|stab|burn|attack)\b",
    r"\blet'?s\b.*\b(kill|shoot|stab|burn|attack)\b",
    r"\bbring a knife\b",
    r"\bburn their store down\b",
    r"\bbreak (his|her|your) jaw\b",
    r"\bmake you pay\b",
)

SYSTEM_PROMPT = (
    "You are a precise content moderation assistant. Return only JSON with keys: "
    "label and action. Label must be one of safe, spam, hate, violence. Action must be one "
    "of allow, delete, flag, escalate."
)

GROQ_MODEL_CANDIDATES = (
    os.getenv("GROQ_MODEL", "llama3-8b-8192"),
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _parse_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def _sanitize_prediction(prediction: Dict[str, Any]) -> Dict[str, str] | None:
    label = prediction.get("label")
    action = prediction.get("action")
    if label in VALID_LABELS and action in VALID_ACTIONS:
        return {"label": label, "action": action}
    return None


def _moderate_with_groq(text: str, metadata: Dict[str, Any]) -> Dict[str, str] | None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
        )
        payload = json.dumps({"text": text, "metadata": metadata}, ensure_ascii=True)

        for model in GROQ_MODEL_CANDIDATES:
            try:
                response = client.chat.completions.create(
                    model=model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": payload},
                    ],
                )
                content = response.choices[0].message.content or "{}"
                parsed = _sanitize_prediction(_parse_json_object(content))
                if parsed is not None:
                    return parsed
            except Exception:
                continue

        return None
    except Exception:
        return None


def _looks_like_safe_context(text: str) -> bool:
    normalized = _normalize(text)
    return any(marker in normalized for marker in SAFE_CONTEXT_MARKERS + SAFE_METAPHOR_MARKERS)


def _spam_score(text: str, metadata: Dict[str, Any]) -> int:
    normalized = _normalize(text)
    score = sum(1 for keyword in SPAM_KEYWORDS if keyword in normalized)
    score += len(re.findall(r"(https?://\S+|www\.\S+)", normalized))
    if metadata.get("contains_url"):
        score += 1
    if re.search(r"\bearn\s*\$?\d+", normalized):
        score += 2
    return score


def _is_hate(text: str) -> bool:
    normalized = _normalize(text)
    if any(re.search(pattern, normalized) for pattern in GROUP_ATTACK_PATTERNS):
        return True
    return any(marker in normalized for marker in INDIVIDUAL_ABUSE_MARKERS)


def _violence_score(text: str) -> int:
    normalized = _normalize(text)
    score = 0
    if any(re.search(pattern, normalized) for pattern in VIOLENCE_PATTERNS):
        score += 3
    if any(token in normalized for token in ("knife", "gun", "bomb", "tonight", "tomorrow")):
        score += 1
    return score


def moderate_text(text: str, metadata: Dict[str, Any] | None = None) -> Dict[str, str]:
    llm_result = _moderate_with_groq(text, metadata or {})
    if llm_result is not None:
        return llm_result

    metadata = metadata or {}
    normalized = _normalize(text)

    if _looks_like_safe_context(text):
        return {"label": "safe", "action": "allow"}

    violence_score = _violence_score(text)
    if violence_score >= 3:
        action = "escalate" if violence_score >= 4 else "delete"
        return {"label": "violence", "action": action}

    if _is_hate(text):
        if any(re.search(pattern, normalized) for pattern in GROUP_ATTACK_PATTERNS):
            return {"label": "hate", "action": "delete"}
        return {"label": "hate", "action": "flag"}

    spam_score = _spam_score(text, metadata)
    if spam_score >= 2:
        action = "delete" if spam_score >= 3 else "flag"
        return {"label": "spam", "action": action}

    return {"label": "safe", "action": "allow"}
