import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import FairnessEnv
import pandas as pd

def update_model():
    # Load the human edits collected by feedback.py
    feedback_path = "database/live_experience.csv"
    
    if not os.path.exists(feedback_path):
        print(f"❌ No feedback data found at {feedback_path}")
        print("Run test_feedback.py first to log human corrections.")
        return
    
    live_df = pd.read_csv(feedback_path)
    
    print(f"📊 Loaded {len(live_df)} feedback entries across {live_df['group_id'].nunique()} groups")
    print(f"   Budget range: ${live_df['max_budget'].min():.0f} - ${live_df['max_budget'].max():.0f}")
    print(f"   Human allocations: ${live_df['amount_allocated'].min():.2f} - ${live_df['amount_allocated'].max():.2f}")
    
    # Create training environment from feedback
    env = DummyVecEnv([lambda: FairnessEnv(live_df)])
    
    # Load the current model
    model_path = "models/fairness_rl_model"
    if os.path.exists(f"{model_path}.zip"):
        model = PPO.load(model_path, env=env)
        print(f"✓ Loaded existing model from {model_path}")
    else:
        print(f"⚠️  No existing model found. Creating new model.")
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0002, ent_coef=0.02)
    
    # Fine-tune with human feedback (more timesteps for noticeable change)
    timesteps = max(20000, len(live_df) * 1000)  # At least 20k, or 1k per feedback entry
    print(f"🔄 Fine-tuning for {timesteps:,} timesteps with human corrections...")
    model.learn(total_timesteps=timesteps)
    
    # Save the updated version
    model.save(model_path)
    print(f"✅ Model updated and saved to {model_path}.zip")
    print(f"   Restart your Python session before running predict.py to see changes.")

if __name__ == "__main__":
    update_model()
