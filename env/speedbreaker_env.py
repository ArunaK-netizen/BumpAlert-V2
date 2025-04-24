import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os
import time
import math
from collections import deque


class CarSpeedbreakerEnv(gym.Env):
    def __init__(self, render=False):
        super(CarSpeedbreakerEnv, self).__init__()
        self.render_mode = render
        if render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.time_step = 1.0 / 240.0  # Higher precision for better physics
        p.setTimeStep(self.time_step)
        self.max_steps = 1000

        # Observation space: accelerometer (x,y,z), gyroscope (x,y,z), position (x,y,z), velocity (x,y,z)
        self.observation_space = spaces.Box(
            low=np.array([-150, -150, -150, -5, -5, -5, -20, -20, -20, -10, -10, -10]),
            high=np.array([150, 150, 150, 5, 5, 5, 20, 20, 20, 10, 10, 10]),
            dtype=np.float32
        )

        # Action space: [steering, throttle]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # For storing accelerometer history
        self.accel_history = deque(maxlen=10)

        # Store speedbreaker objects for visualization
        self.speedbreaker_objects = []

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        # Load ground plane
        self.plane = p.loadURDF("plane.urdf")

        # Create a straight road
        road_length = 50
        road_width = 5

        # Create road texture
        road_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[road_length / 2, road_width / 2, 0.01],
            rgbaColor=[0.4, 0.4, 0.4, 1]
        )
        road_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[road_length / 2, road_width / 2, 0.01]
        )
        self.road = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=road_collision,
            baseVisualShapeIndex=road_visual,
            basePosition=[road_length / 2, 0, 0]
        )

        # Add road boundaries (curbs)
        curb_height = 0.1
        curb_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[road_length / 2, 0.2, curb_height / 2],
            rgbaColor=[0.9, 0.9, 0.9, 1]
        )
        curb_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[road_length / 2, 0.2, curb_height / 2]
        )

        # Left curb
        self.left_curb = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=curb_collision,
            baseVisualShapeIndex=curb_visual,
            basePosition=[road_length / 2, road_width / 2 + 0.1, curb_height / 2]
        )

        # Right curb
        self.right_curb = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=curb_collision,
            baseVisualShapeIndex=curb_visual,
            basePosition=[road_length / 2, -road_width / 2 - 0.1, curb_height / 2]
        )

        # Load car - using racecar model from pybullet_data
        # Load car with scaling
        scale_factor = 2.0  # Double the size
        self.car = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "racecar/racecar.urdf"),
                              [0, 0, 0.3],
                              p.getQuaternionFromEuler([0, 0, math.pi]),
                              globalScaling=scale_factor)

        # Let physics stabilize
        for _ in range(10):
            p.stepSimulation()

        # Create cylindrical speedbreakers at more frequent positions
        self.speedbreakers = []
        self.speedbreaker_objects = []

        # Add more speedbreakers at varying intervals
        speedbreaker_positions = [5, 8, 12, 16, 21, 25, 30, 34, 38, 42, 46]

        for pos in speedbreaker_positions:
            # Alternate speedbreakers between center, left, and right positions
            y_offset = 0
            if pos % 3 == 1:
                y_offset = 1  # Right side of road
            elif pos % 3 == 2:
                y_offset = -1  # Left side of road

            self._create_cylindrical_speedbreaker(pos, y_offset, 0.05)

        # Setup car wheels and steering
        self.wheels = [2, 3, 5, 7]  # Wheel indices in the URDF
        self.steering = [4, 6]  # Steering joint indices

        # Reset history
        self.accel_history.clear()
        for _ in range(10):
            self.accel_history.append(np.zeros(3))

        self.prev_pos = np.zeros(3)
        self.prev_vel = np.zeros(3)
        self.step_count = 0

        # Initial observation
        obs = self._get_obs()
        return obs, {}

    def _update_follow_camera(self):
        """Update camera to follow the car from behind"""
        # Get car position and orientation
        car_pos, car_orn = p.getBasePositionAndOrientation(self.car)

        # Calculate yaw angle from quaternion
        yaw = p.getEulerFromQuaternion(car_orn)[-1]

        # Set camera parameters
        distance = 7.0  # Distance behind the car
        height = 3.0  # Height above the car
        pitch = -20  # Looking down slightly

        # Update debug visualizer camera
        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=math.degrees(yaw),  # Convert to degrees
            cameraPitch=pitch,
            cameraTargetPosition=car_pos
        )

    def _create_cylindrical_speedbreaker(self, x, y, height):
        """Create a more realistic speedbreaker (speed bump) at the specified position"""
        # Improved speedbreaker dimensions
        width = 0.4  # Width of the speedbreaker (in driving direction)
        length = 3.5  # Length of the speedbreaker (across the road)
        radius = height * 1.5  # Increased radius for smoother transition

        # Create a visual mesh for a cylindrical segment with smoother transitions
        num_segments = 16  # More segments for smoother curve
        vertices = []
        indices = []

        # Create a half-cylinder shape with tapered edges
        for i in range(num_segments + 1):
            angle = math.pi * i / num_segments
            # Bottom vertices
            vertices.append([-width / 2, -length / 2, 0])
            vertices.append([-width / 2, length / 2, 0])

            # Top vertices with cylindrical shape
            x_offset = width / 2 - radius * math.cos(angle)
            z_offset = radius * math.sin(angle)

            # Apply tapering at the edges for smoother transition
            edge_factor = 1.0
            if i <= 2 or i >= num_segments - 2:
                edge_factor = 0.7  # Reduce height at edges

            vertices.append([x_offset - width / 2, -length / 2, z_offset * edge_factor])
            vertices.append([x_offset - width / 2, length / 2, z_offset * edge_factor])

            # Add connection triangles
            if i < num_segments:
                base = i * 4
                # Connect vertices to form triangles
                indices.extend([base, base + 2, base + 4])  # Left side
                indices.extend([base + 2, base + 4, base + 6])
                indices.extend([base + 1, base + 5, base + 3])  # Right side
                indices.extend([base + 1, base + 7, base + 5])
                indices.extend([base + 2, base + 3, base + 6])  # Top
                indices.extend([base + 3, base + 7, base + 6])
                indices.extend([base, base + 1, base + 2])  # Bottom
                indices.extend([base + 1, base + 3, base + 2])

        # Create collision and visual shapes
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            vertices=vertices,
            indices=indices,
            meshScale=[1, 1, 1]
        )

        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            vertices=vertices,
            indices=indices,
            meshScale=[1, 1, 1],
            rgbaColor=[0.8, 0.2, 0.2, 1]
        )

        # Create the speedbreaker multibody
        speedbreaker = p.createMultiBody(
            baseMass=0,  # Static object
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[x, y, 0]
        )

        self.speedbreakers.append(speedbreaker)
        self.speedbreaker_objects.append((x, y, height))

        return speedbreaker

    def step(self, action):
        # Extract steering and throttle from action
        steering = float(action[0])  # Range: -1 to 1
        throttle = float(action[1])  # Range: -1 to 1

        # Apply steering
        max_steering_angle = 0.5
        for steer_joint in self.steering:
            p.setJointMotorControl2(self.car, steer_joint,
                                    p.POSITION_CONTROL,
                                    targetPosition=steering * max_steering_angle)

        # Apply throttle to wheels
        max_force = 20
        for wheel in self.wheels:
            p.setJointMotorControl2(self.car, wheel,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=-throttle * 30,
                                    force=max_force)

        # Step simulation multiple times for smoother physics
        for _ in range(10):
            p.stepSimulation()

        self.step_count += 1

        # Get observation
        obs = self._get_obs()

        # Calculate reward
        reward = self._calculate_reward(obs)

        # Check termination conditions
        pos, _ = p.getBasePositionAndOrientation(self.car)

        # Terminate if car reaches the end or goes too far off road
        road_width = 5.0
        off_road_distance = abs(pos[1]) - road_width / 2

        terminated = pos[0] > 45 or off_road_distance > 2.0
        truncated = self.step_count > self.max_steps

        info = {
            'position': pos,
            'acceleration': obs[:3],
            'on_speedbreaker': self._is_on_speedbreaker(),
            'off_road': off_road_distance > 0
        }

        self._update_follow_camera()

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # Get car position and orientation
        pos, orn = p.getBasePositionAndOrientation(self.car)
        pos = np.array(pos)

        # Get car velocity
        lin_vel, ang_vel = p.getBaseVelocity(self.car)
        lin_vel = np.array(lin_vel)
        ang_vel = np.array(ang_vel)

        # Calculate acceleration from velocity change
        if hasattr(self, 'prev_vel'):
            accel = (lin_vel - self.prev_vel) / self.time_step
        else:
            accel = np.zeros(3)

        # Add noise to simulate real sensors
        accel += np.random.normal(0, 0.1, 3)
        ang_vel += np.random.normal(0, 0.05, 3)

        # Store current velocity for next step
        self.prev_vel = lin_vel.copy()

        # Store acceleration in history
        self.accel_history.append(accel)

        # Clip acceleration to observation space bounds
        accel_clipped = np.clip(accel, -150, 150)
        ang_vel_clipped = np.clip(ang_vel, -5, 5)

        # Create observation vector
        obs = np.concatenate([
            accel_clipped,  # Accelerometer (x,y,z)
            ang_vel_clipped,  # Gyroscope (x,y,z)
            pos,  # Position (x,y,z)
            lin_vel  # Velocity (x,y,z)
        ]).astype(np.float32)

        return obs

    def _is_on_speedbreaker(self):
        pos, _ = p.getBasePositionAndOrientation(self.car)

        # Check if car is on any speedbreaker
        for sb_pos, _, _ in self.speedbreaker_objects:
            if abs(pos[0] - sb_pos) < 0.5:  # Narrower detection range
                return True
        return False

    def _calculate_reward(self, obs):
        pos, _ = p.getBasePositionAndOrientation(self.car)
        accel = obs[:3]

        # Base reward for forward progress
        progress_reward = (pos[0] - self.prev_pos[0]) * 10
        self.prev_pos = np.array(pos)

        # Initialize total reward
        reward = progress_reward

        # Detect speedbreaker by vertical acceleration
        is_on_speedbreaker = self._is_on_speedbreaker()

        # Calculate vertical acceleration variance over recent history
        accel_z_values = [a[2] for a in self.accel_history]
        accel_z_variance = np.var(accel_z_values)

        # Reward for correctly identifying speedbreakers
        if is_on_speedbreaker:
            # If we detect high vertical acceleration on a speedbreaker, that's good!
            if accel_z_variance > 1.0:
                reward += 5.0
        else:
            # Penalize false positives - high acceleration when not on speedbreaker
            if accel_z_variance > 1.0:
                reward -= 2.0

        # Stronger penalty for going off track
        road_width = 5.0
        distance_from_center = abs(pos[1])

        if distance_from_center > road_width / 2:
            # Calculate how far off the road the car is
            off_road_distance = distance_from_center - road_width / 2
            # Apply progressive penalty (gets worse the further off-road)
            off_road_penalty = -10.0 - (off_road_distance * 5.0)
            reward += off_road_penalty

            # Early termination if car goes too far off road
            if off_road_distance > 2.0:
                self.off_road_termination = True
        else:
            # Small reward for staying centered on the road
            centering_reward = 1.0 * (1.0 - (distance_from_center / (road_width / 2)))
            reward += centering_reward

        return reward

    def close(self):
        p.disconnect()