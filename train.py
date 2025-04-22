import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from env.speedbreaker_env import CarSpeedbreakerEnv

# Create environment
env = CarSpeedbreakerEnv(render=True)
check_env(env)

# Create logs directory
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

# Setup checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=log_dir,
    name_prefix="speedbreaker_model"
)

# Create and train model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01
)

model.learn(
    total_timesteps=500000,
    callback=checkpoint_callback
)

# Save final model
model.save("speedbreaker_agent_final")
env.close()
