import os
import requests
from openai import OpenAI
from typing import List, Optional

# --- CONFIGURATION (Environment Variables) ---
# Pass API settings using environment variables.
API_BASE_URL = os.getenv("API_BASE_URL", "https://basant-levi-ai-content-moderation-openenv.hf.space").rstrip('/')
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
# Use HF_TOKEN or OPENAI_API_KEY as the API key source.
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

def main():
    # MANDATORY START LINE
    print(f"[START] task=moderation-task env=content-moderation-v1 model={MODEL_NAME}", flush=True)
    
    if not API_KEY:
        print("[END] success=false steps=0 rewards= error=Missing API Key in environment variables", flush=True)
        return

    # OpenAI-compatible client configured by environment variables.
    # If OPENAI_BASE_URL is not set, the default Groq-compatible base URL is used.
    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
        api_key=API_KEY
    )
    
    rewards = []
    steps_taken = 0
    success = False
    
    try:
        # Step 1: Reset Environment
        reset_resp = requests.post(f"{API_BASE_URL}/reset")
        if reset_resp.status_code != 200:
            raise Exception(f"Reset Failed: {reset_resp.status_code}")
            
        data = reset_resp.json()
        observation = data.get("observation")
        done = data.get("done", False)

        # Step 2: Moderation Loop (42 examples)
        for step in range(1, 43):
            if done or not observation:
                break
                
            text_to_moderate = observation.get("text", "")

            # AI decision making logic
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{
                    "role": "user", 
                    "content": f"Classify this text as [safe, spam, hate, violence]. Reply with ONLY the word: {text_to_moderate}"
                }],
                temperature=0.1
            )
            action_label = completion.choices[0].message.content.strip().lower()
            # Cleaning the label
            action_label = "".join(filter(str.isalpha, action_label))

            # Step 3: Update Environment
            step_resp = requests.post(
                f"{API_BASE_URL}/step", 
                json={"label": action_label, "action": "flag"}
            ).json()
            
            reward = step_resp.get("reward", 0.0)
            done = step_resp.get("done", False)
            observation = step_resp.get("observation")
            
            # MANDATORY STEP LINE
            print(f"[STEP] step={step} action={action_label} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            
            rewards.append(reward)
            steps_taken = step

        # Final Evaluation
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        success = avg_reward > 0.3

    except Exception as e:
        print(f"Error Details: {str(e)}")
    finally:
        # MANDATORY END LINE
        rewards_str = ",".join([f"{r:.2f}" for r in rewards]) if rewards else ""
        print(f"[END] success={str(success).lower()} steps={steps_taken} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    main()