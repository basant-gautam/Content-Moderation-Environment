import os
import requests
from openai import OpenAI

# SCALER RUNS SERVER LOCALLY. DO NOT USE HF SPACE URL HERE!
ENV_URL = "http://127.0.0.1:8000"

def evaluate_task(client, task_name, model_name):
    print(f"[START] task={task_name} env=content-moderation-v1 model={model_name}", flush=True)
    rewards = []
    steps_taken = 0
    success = False
    
    try:
        reset_resp = requests.post(f"{ENV_URL}/reset", json={"task": task_name})
        if reset_resp.status_code != 200:
            raise Exception("Reset Failed")
            
        data = reset_resp.json()
        observation = data.get("observation")
        done = data.get("done", False)

        for step in range(1, 43):
            if done or not observation: break
            text_to_moderate = observation.get("text", "")

            # AI decision making logic via SCALER PROXY
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": f"Classify this text as safe, spam, hate, or violence. Reply with only the label: {text_to_moderate}"}],
                temperature=0.1
            )
            action_label = "".join(filter(str.isalpha, completion.choices[0].message.content.strip().lower()))
            action = {"safe": "allow", "spam": "delete", "hate": "flag", "violence": "escalate"}.get(action_label, "flag")

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
    # STRICTLY USE os.environ AS SCALER EXPLICITLY EXPECTS
    try:
        api_base_url = os.environ["API_BASE_URL"]
        api_key = os.environ["API_KEY"]
    except KeyError:
        print("Missing API_BASE_URL or API_KEY in environment variables", flush=True)
        return
        
    model_name = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
    
    client = OpenAI(base_url=api_base_url, api_key=api_key)
    
    # RUN ALL 3 TASKS
    for task in ["easy", "medium", "hard"]:
        evaluate_task(client, task, model_name)

if __name__ == "__main__":
    main()