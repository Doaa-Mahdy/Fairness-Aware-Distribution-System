import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import FairnessEnv

# 1. Load Data (no scaling to keep budget/constraints consistent with inference)
df = pd.read_csv('charity_recipients_dataset.csv')

# 2. Initialize Env
env = DummyVecEnv([lambda: FairnessEnv(df)])

# 3. Train with PPO
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=0.0002,
    ent_coef=0.02, # exploration
    batch_size=64,
    n_steps=1024,
    clip_range=0.2,
    vf_coef=0.5
)

print("Starting training with budget-aware reward...")
model.learn(total_timesteps=100000)

# 4. Save
os.makedirs("models", exist_ok=True)
model.save("models/fairness_rl_model")
print("Model saved to models/fairness_rl_model")