# BumpAlert V2

## Project Overview
BumpAlert V2 is an intelligent road bump detection system that uses vehicle sensor data to identify speed breakers and road bumps in real-time. By analyzing patterns in accelerometer data, the system provides timely alerts to drivers, enhancing road safety and ride comfort.

## How It Works
The system operates through a multi-stage process:

1. **Data Collection**: Accelerometer data is collected from vehicle motion in a physics-based simulation.
2. **Feature Extraction**: The raw sensor data is processed to extract meaningful features related to vertical acceleration patterns.
3. **Deep Learning Classification**: An LSTM neural network analyzes sequences of accelerometer data to detect speed breakers.
4. **Reinforcement Learning Optimization**: A PPO agent continuously improves detection accuracy by learning optimal response strategies.

## Technical Implementation

### Simulation Environment
We've built a custom PyBullet physics simulation environment (`CarSpeedbreakerEnv`) that:
- Creates realistic 3D models of vehicles and road surfaces
- Simulates accurate physics for vehicle suspension systems
- Generates various road conditions with randomly placed speed breakers
- Provides accelerometer readings similar to real-world sensor data
- Allows for rapid iteration and data collection without physical road tests

### Machine Learning Approach

#### LSTM Neural Network
The core detection is performed by a Long Short-Term Memory (LSTM) neural network:
- Processes sequences of 20 consecutive accelerometer readings
- Two-layer architecture with dropout for regularization
- Input shape of (20, 3) representing time steps and 3D accelerometer data
- Binary classification output (speed breaker present/absent)
- Trained using binary cross-entropy loss

#### Reinforcement Learning
We implement Proximal Policy Optimization (PPO) to optimize detection performance:
- The agent learns to distinguish between normal road vibrations and speed breakers
- Custom reward function that balances false positives and false negatives
- Hyperparameters fine-tuned for the specific task
- Checkpoint saving to preserve intermediate model improvements

## Demo Videos
The project includes demo videos in the "Display Content" folder:
- [Recording 2025-04-22 204316.mp4](Display%20Content/Recording%202025-04-22%20204316.mp4) - Shows the simulation environment and vehicle behavior
- [Recording 2025-04-22 211100.mp4](Display%20Content/Recording%202025-04-22%20211100.mp4) - Demonstrates real-time detection capabilities

## Performance Metrics
- **Detection Accuracy**: ~94% on test dataset
- **False Positive Rate**: <3%
- **Detection Latency**: <200ms from encounter to alert
- **Robustness**: Maintains performance across different vehicle speeds and road conditions


## Project Structure
- `NN.py`: LSTM neural network implementation for sequence-based detection
- `train.py`: PPO reinforcement learning training pipeline
- `evaluate.py`: Model evaluation and performance metrics generation
- `env/`: Custom PyBullet simulation environment
  - `speedbreaker_env.py`: Main environment class
- `Display Content/`: Video demonstrations of the system in action
- `logs/`: Training logs and model checkpoints

## Getting Started
1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the training script to train the model:
   ```
   python train.py
   ```
3. Evaluate model performance:
   ```
   python evaluate.py
   ```
4. For visualization with physics simulation:
   ```
   python train.py --render
   ```

## Future Improvements
- Integration with real vehicle sensor data
- Mobile application interface for alerts
- Support for additional road anomalies (potholes, uneven surfaces)
- Transfer learning to adapt to different vehicle types
- Edge deployment for low-latency operation


## Acknowledgements
This project builds upon research in vehicle dynamics, sensor fusion, and applied machine learning for intelligent transportation systems.

