import os
try:
    import requests
except ImportError:
    requests = None
from openai import OpenAI

# --- THE MOST IMPORTANT FIX: CORRECT VARIABLES ---
# Scaler injects API_BASE_URL for the LLM Proxy, NOT for the environment server!
LLM_PROXY_URL = os.getenv("API_BASE_URL") 
LLM_API_KEY = os.getenv("API_KEY")

# Tumhara backend URL hardcoded hona chahiye, Scaler ise inject nahi karta.
ENV_URL = "https://basant-levi-ai-content-moderation-openenv.hf.space"
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

def evaluate_task(client, task_name):
    # Validator ab yahan se padhega ki kaunsa exact task chal raha hai
    print(f"[START] task={task_name} env=content-moderation-v1 model={MODEL_NAME}", flush=True)
    
    rewards = []
    steps_taken = 0
    success = False
    
    try:
        reset_resp = requests.post(f"{ENV_URL}/reset", json={"task": task_name})
        if reset_resp.status_code != 200:
            raise Exception(f"Reset Failed")
            
        data = reset_resp.json()
        observation = data.get("observation")
        done = data.get("done", False)

        for step in range(1, 43):
            if done or not observation:
                break
                
            text_to_moderate = observation.get("text", "")

            # YAHAN SCALER KE PROXY PAR CALL JAYEGI
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": f"Classify this text as [safe, spam, hate, violence]. Reply with ONLY the word: {text_to_moderate}"}],
                temperature=0.1
            )
            action_label = completion.choices[0].message.content.strip().lower()
            action_label = "".join(filter(str.isalpha, action_label))
            
            # Action Mapping logic
            if action_label == "safe": action = "allow"
            elif action_label == "spam": action = "delete"
            elif action_label == "hate": action = "flag"
            elif action_label == "violence": action = "escalate"
            else: action = "flag"

            step_resp = requests.post(f"{ENV_URL}/step", json={"label": action_label, "action": action}).json()
            
            reward = step_resp.get("reward", step_resp.get("info", {}).get("reward", 0.0))
            done = step_resp.get("done", False)
            observation = step_resp.get("observation")
            
            print(f"[STEP] step={step} action={action_label} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            
            rewards.append(reward)
            steps_taken = step

        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        success = avg_reward >= 0.1

    except Exception as e:
        print(f"Error Details: {str(e)}")
    finally:
        rewards_str = ",".join([f"{r:.2f}" for r in rewards]) if rewards else ""
        print(f"[END] success={str(success).lower()} steps={steps_taken} rewards={rewards_str}", flush=True)

def main():
    if not LLM_PROXY_URL or not LLM_API_KEY:
        print("[END] success=false steps=0 rewards= error=Missing Scaler Variables", flush=True)
        return

    # SCALER KA URL AUR KEY STRICTLY OPENAI CLIENT KO PASS KIYA HAI
    client = OpenAI(base_url=LLM_PROXY_URL, api_key=LLM_API_KEY)
    
    # MAGIC FIX: Teeno tasks sequentially call honge!
    for task in ["easy", "medium", "hard"]:
        evaluate_task(client, task)

if __name__ == "__main__":
    main()