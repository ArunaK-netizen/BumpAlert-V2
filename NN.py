import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from env.speedbreaker_env import CarSpeedbreakerEnv

# Collect data from simulation
def collect_training_data(env, num_episodes=10):
    X = []
    y = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False

        # For storing sequence data
        sequence = []

        while not done:
            action = env.action_space.sample()  # Random actions to explore
            obs, reward, terminated, truncated, info = env.step(action)

            # Extract accelerometer data
            accel = obs[:3]
            sequence.append(accel)

            # If we have enough data for a sequence
            if len(sequence) >= 20:
                X.append(np.array(sequence[-20:]))
                y.append(1 if info['on_speedbreaker'] else 0)

            done = terminated or truncated

    return np.array(X), np.array(y)


# Create and train LSTM model
def train_speedbreaker_detector(X, y):
    model = Sequential([
        LSTM(64, input_shape=(20, 3), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X, y,
        epochs=20,
        batch_size=32,
        validation_split=0.2
    )

    return model


# Main execution
env = CarSpeedbreakerEnv(render=False)  # Use render=False for faster data collection

# Collect data
print("Collecting training data...")
X, y = collect_training_data(env, num_episodes=50)
print(f"Collected {len(X)} samples, with {sum(y)} positive examples")

# Train model
print("Training speedbreaker detector...")
detector_model = train_speedbreaker_detector(X, y)

# Save model
detector_model.save("speedbreaker_detector.h5")
print("Model saved as 'speedbreaker_detector.h5'")

env.close()
