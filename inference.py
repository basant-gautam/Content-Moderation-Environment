import os
import time
try:
    import requests
except ImportError:
    requests = None
from openai import OpenAI

# 🚨 BACKEND SERVER LOCAL URL 🚨
ENV_URL = os.environ.get("ENV_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 15

# 🚨 THE FIX: Bypass proxy for local server requests 🚨
# This prevents the platform's LLM proxy from intercepting our local backend calls
LOCAL_PROXIES = {"http": None, "https": None}

LABEL_TO_ACTION = {"safe": "allow", "spam": "delete", "hate": "flag", "violence": "escalate"}

def _ensure_requests_available():
    if requests is None:
        raise RuntimeError("requests dependency is required to run inference.py")

def _wait_for_env(env_url, timeout_seconds=60):
    _ensure_requests_available()
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            # Proxies bypassed here
            response = requests.get(
                f"{env_url}/health", 
                timeout=REQUEST_TIMEOUT_SECONDS,
                proxies=LOCAL_PROXIES
            )
            if response.ok:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(1.0)
    raise RuntimeError(f"environment backend not reachable at {env_url}: {last_error}")

def _probe_llm_proxy(client, model_name):
    try:
        client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Reply with only: safe"}],
            temperature=0,
            max_tokens=2,
        )
    except Exception as e:
        print(f"Proxy probe skipped/failed: {str(e)}", flush=True)

def evaluate_task(client, task_name, model_name):
    print(f"[START] task={task_name} env=content-moderation-v1 model={model_name}", flush=True)
    rewards = []
    steps_taken = 0
    success = False
    
    try:
        _ensure_requests_available()
        
        # Proxies bypassed here for /reset
        reset_resp = requests.post(
            f"{ENV_URL}/reset",
            json={"task": task_name},
            timeout=REQUEST_TIMEOUT_SECONDS,
            proxies=LOCAL_PROXIES
        )
        reset_resp.raise_for_status()
        data = reset_resp.json() if reset_resp.status_code == 200 else {}
        observation = data.get("observation")
        done = data.get("done", False)

        for step in range(1, 43):
            if done or not observation: break
            text_to_moderate = observation.get("text", "")

            # LLM Proxy IS used here natively via the OpenAI client
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a content moderator. Classify text as: safe, spam, hate, or violence. Reply with ONLY the label."},
                    {"role": "user", "content": text_to_moderate}
                ],
                temperature=0.1
            )
            action_label = "".join(filter(str.isalpha, completion.choices[0].message.content.strip().lower()))
            action = LABEL_TO_ACTION.get(action_label, "flag")

            # Proxies bypassed here for /step
            step_resp = requests.post(
                f"{ENV_URL}/step",
                json={"label": action_label, "action": action},
                timeout=REQUEST_TIMEOUT_SECONDS,
                proxies=LOCAL_PROXIES
            )
            step_resp.raise_for_status()
            step_resp = step_resp.json()
            
            reward = step_resp.get("reward", step_resp.get("info", {}).get("reward", 0.01))
            done = step_resp.get("done", False)
            observation = step_resp.get("observation")
            
            print(f"[STEP] step={step} action={action_label} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            rewards.append(reward)
            steps_taken = step

        avg_reward = sum(rewards) / len(rewards) if rewards else 0.01
        success = avg_reward > 0.1
    except Exception as e:
        print(f"Error Details: {str(e)}", flush=True)
    finally:
        rewards_str = ",".join([f"{r:.2f}" for r in rewards]) if rewards else "0.01"
        print(f"[END] success={str(success).lower()} steps={steps_taken} rewards={rewards_str}", flush=True)

def main():
    model_name = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
    
    # Official client uses the environment proxy correctly
    client = OpenAI(base_url=os.environ["API_BASE_URL"], api_key=os.environ["API_KEY"])
    
    _wait_for_env(ENV_URL)
    _probe_llm_proxy(client, model_name)
    
    for task in ["easy", "medium", "hard"]:
        evaluate_task(client, task, model_name)

if __name__ == "__main__":
    main()