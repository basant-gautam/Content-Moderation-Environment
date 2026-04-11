import os
try:
    import requests
except ImportError:
    requests = None
from openai import OpenAI

# 1. TUMHARA BACKEND URL (Isko hardcoded hi rakhna hai, os.getenv mat lagana yahan)
ENV_URL = "https://basant-levi-ai-content-moderation-openenv.hf.space"

# 2. SCALER KA PROXY (Inko OpenAI client ke andar daalna hai)
LLM_PROXY_URL = os.getenv("API_BASE_URL")
LLM_API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

LABEL_TO_ACTION = {"safe": "allow", "spam": "delete", "hate": "flag", "violence": "escalate"}

def evaluate_task(client, task_name):
    print(f"[START] task={task_name} env=content-moderation-v1 model={MODEL_NAME}", flush=True)
    rewards = []
    steps_taken = 0
    success = False
    
    try:
        reset_resp = requests.post(f"{ENV_URL}/reset", json={"task": task_name})
        data = reset_resp.json() if reset_resp.status_code == 200 else {}
        observation = data.get("observation")
        done = data.get("done", False)

        for step in range(1, 43):
            if done or not observation: break
            text_to_moderate = observation.get("text", "")

            # YAHAN SCALER KE PROXY PAR CALL JAYEGI
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": f"Classify this text as one of [safe, spam, hate, violence]. Reply with only the label: {text_to_moderate}"}],
                temperature=0.1
            )
            action_label = "".join(filter(str.isalpha, completion.choices[0].message.content.strip().lower()))
            action = LABEL_TO_ACTION.get(action_label, "flag")

            step_resp = requests.post(f"{ENV_URL}/step", json={"label": action_label, "action": action}).json()
            reward = step_resp.get("reward", step_resp.get("info", {}).get("reward", 0.0))
            done = step_resp.get("done", False)
            observation = step_resp.get("observation")
            
            print(f"[STEP] step={step} action={action_label} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            rewards.append(reward)
            steps_taken = step

        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        success = avg_reward > 0.1
    except Exception as e:
        print(f"Error Details: {str(e)}")
    finally:
        rewards_str = ",".join([f"{r:.2f}" for r in rewards]) if rewards else ""
        print(f"[END] success={str(success).lower()} steps={steps_taken} rewards={rewards_str}", flush=True)

def main():
    if not LLM_PROXY_URL or not LLM_API_KEY: return
    # STRICTLY USING SCALER'S PROXY URL HERE
    client = OpenAI(base_url=LLM_PROXY_URL, api_key=LLM_API_KEY)
    
    # MAGIC LOOP FOR ALL 3 TASKS
    for task in ["easy", "medium", "hard"]:
        evaluate_task(client, task)

if __name__ == "__main__":
    main()