import os
try:
    import requests
except ImportError:
    requests = None
from openai import OpenAI

# 🚨 BACKEND SERVER LOCALHOST PE HI RAHEGA 🚨
ENV_URL = "http://127.0.0.1:8000"

LABEL_TO_ACTION = {"safe": "allow", "spam": "delete", "hate": "flag", "violence": "escalate"}

def evaluate_task(client, task_name, model_name):
    print(f"[START] task={task_name} env=content-moderation-v1 model={model_name}", flush=True)
    rewards = []
    steps_taken = 0
    success = False
    
    try:
        # Clean POST request for reset
        reset_resp = requests.post(f"{ENV_URL}/reset", json={"task": task_name})
        data = reset_resp.json() if reset_resp.status_code == 200 else {}
        observation = data.get("observation")
        done = data.get("done", False)

        for step in range(1, 43):
            if done or not observation: break
            text_to_moderate = observation.get("text", "")

            # 🚨 UPDATED: System Prompt added as per maintainer's suggestion 🚨
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a content moderator. Classify text as: safe, spam, hate, or violence. Reply with ONLY the label."},
                    {"role": "user", "content": text_to_moderate}
                ],
                temperature=0.1
            )
            
            # Clean string extraction
            response_text = completion.choices[0].message.content.strip().lower()
            action_label = "".join(filter(str.isalpha, response_text))
            action = LABEL_TO_ACTION.get(action_label, "flag")

            step_resp = requests.post(f"{ENV_URL}/step", json={"label": action_label, "action": action}).json()
            
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
    client = OpenAI(base_url=os.environ["API_BASE_URL"], api_key=os.environ["API_KEY"])
    
    for task in ["easy", "medium", "hard"]:
        evaluate_task(client, task, model_name)

if __name__ == "__main__":
    main()