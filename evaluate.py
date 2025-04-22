import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import time

# Load environment and model
env = CarSpeedbreakerEnv(render=True)
model = PPO.load("speedbreaker_agent_final")

# For storing data
positions = []
accelerations = []
speedbreaker_detections = []
actual_speedbreakers = []

# Run evaluation
obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # Store data for analysis
    positions.append(info['position'][0])  # x-position
    accelerations.append(obs[2])  # z-acceleration

    # Detect speedbreaker from acceleration variance
    accel_z_values = [a[2] for a in env.accel_history]
    accel_z_variance = np.var(accel_z_values)
    speedbreaker_detected = accel_z_variance > 1.0
    speedbreaker_detections.append(speedbreaker_detected)

    # Record actual speedbreaker positions
    actual_speedbreakers.append(env._is_on_speedbreaker())

    done = terminated or truncated
    time.sleep(0.01)  # Slow down visualization

env.close()

# Plot results
plt.figure(figsize=(15, 10))

# Plot 1: Position vs Acceleration
plt.subplot(3, 1, 1)
plt.plot(positions, accelerations)
plt.title('Vertical Acceleration vs Position')
plt.xlabel('Position (m)')
plt.ylabel('Acceleration (m/s²)')

# Plot 2: Speedbreaker Detections
plt.subplot(3, 1, 2)
plt.plot(positions, speedbreaker_detections, label='Detected')
plt.plot(positions, actual_speedbreakers, label='Actual')
plt.title('Speedbreaker Detection')
plt.xlabel('Position (m)')
plt.ylabel('Detection')
plt.legend()

# Plot 3: Acceleration Variance
accel_variance = []
for i in range(len(positions)):
    if i < 10:
        accel_variance.append(0)
    else:
        accel_variance.append(np.var(accelerations[i - 10:i]))

plt.subplot(3, 1, 3)
plt.plot(positions, accel_variance)
plt.title('Acceleration Variance (Detection Feature)')
plt.xlabel('Position (m)')
plt.ylabel('Variance')

plt.tight_layout()
plt.savefig('speedbreaker_analysis.png')
plt.show()
