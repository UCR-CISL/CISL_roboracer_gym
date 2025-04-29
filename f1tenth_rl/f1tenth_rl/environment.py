#!/usr/bin/env python3

import numpy as np
import math
import time
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
from transforms3d.euler import euler_from_quaternion, quaternion_from_euler

class F1TenthEnv:
    """
    Environment class that handles interaction with the F1TENTH simulator
    """
    def __init__(self, node=None, config=None):
        """
        Initialize the environment.
        
        Args:
            node: ROS node that contains publishers/subscribers
            config: Configuration dictionary with environment parameters
        """
        # Use provided config or default values
        self.config = config or {}
        
        # Track dimensions (approximate values for typical F1TENTH tracks)
        self.track_width = self.config.get('track_width', 2.0)  # meters
        
        # Collision detection
        self.collision_threshold = self.config.get('collision_threshold', 0.3)  # meters
        
        # Episode tracking
        self.episode_step = 0
        self.max_episode_steps = self.config.get('max_steps_per_episode', 1000)
        
        # Reward weights
        self.collision_penalty = self.config.get('collision_penalty', -100.0)
        self.speed_reward_weight = self.config.get('speed_reward_weight', 1.0)
        self.steering_penalty_weight = self.config.get('steering_penalty_weight', -0.5)
        self.progress_reward_weight = self.config.get('progress_reward_weight', 2.0)
        self.centerline_reward_weight = self.config.get('centerline_reward_weight', 1.0)
        
        # Target speed for reward calculation
        self.target_speed = self.config.get('target_speed', 2.0)  # m/s
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_steps_list = []
        self.best_reward = -float('inf')
        self.prev_speed = 0.0
        self.prev_pose = None
        self.current_speed = 0.0
        
        # Track progress tracking
        self.start_position = None
        self.lap_progress = 0.0
        self.lap_count = 0
        self.traveled_distance = 0.0
        self.lap_time = 0.0
        self.lap_start_time = None
        
        # Store node reference for reset functionality
        self.node = node
        
        # Configure starting position
        start_x = self.config.get('start_x', 0.0)
        start_y = self.config.get('start_y', 0.0)
        start_yaw = self.config.get('start_yaw', 0.0)
        self.start_position = [start_x, start_y, start_yaw]
        
        # Create reset publisher if node is provided
        if self.node is not None:
            self.reset_pub = self.node.create_publisher(
                PoseWithCovarianceStamped,
                '/initialpose',
                10
            )

    def reset_car_position(self):
        """Reset the car to its starting position"""
        if not hasattr(self, 'reset_pub'):
            return False
            
        # Create the pose message
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.frame_id = 'map'
        pose_msg.header.stamp.sec = int(time.time())
        
        # Set position (x, y, z)
        pose_msg.pose.pose.position.x = self.start_position[0]
        pose_msg.pose.pose.position.y = self.start_position[1]
        pose_msg.pose.pose.position.z = 0.0
        
        # Set orientation (as quaternion from yaw)
        q = quaternion_from_euler(0, 0, self.start_position[2])
        pose_msg.pose.pose.orientation.x = q[0]
        pose_msg.pose.pose.orientation.y = q[1]
        pose_msg.pose.pose.orientation.z = q[2]
        pose_msg.pose.pose.orientation.w = q[3]
        
        # Publish the pose
        self.reset_pub.publish(pose_msg)
        
        # Sleep a bit to let the simulator respond
        time.sleep(0.2)
        
        return True
    
    def reset(self):
        """Reset environment state at the beginning of an episode"""
        self.prev_speed = 0.0
        self.prev_pose = None
        self.current_speed = 0.0
        self.lap_progress = 0.0
        self.lap_count = 0
        self.episode_step = 0
        self.traveled_distance = 0.0
        self.lap_time = 0.0
        self.lap_start_time = time.time()
        
        # Reset the car position if we have a node
        if hasattr(self, 'reset_pub'):
            self.reset_car_position()
        
        return None  # In ROS we get states from callbacks, not directly
    
    def step(self, action, scan, odom):
        """
        Execute a step in the environment
        
        Args:
            action: [steering_angle, velocity]
            scan: LaserScan message
            odom: Odometry message
            
        Returns:
            next_state: current laser scan ranges
            reward: calculated reward
            done: whether episode is done
            info: additional information
        """
        # Extract current speed and position
        current_speed = self._get_speed_from_odom(odom)
        current_pose = odom.pose.pose
        
        # Store current speed
        self.current_speed = current_speed
        
        # Initialize starting position if not set
        if self.start_position is None and current_pose is not None:
            self.start_position = [
                current_pose.position.x,
                current_pose.position.y,
                self._get_yaw_from_pose(current_pose)
            ]
            self.lap_start_time = time.time()
        
        # Check for collision
        collision = self._check_collision(scan)
        
        # Calculate reward components
        speed_reward = self._speed_reward(current_speed)
        steering_penalty = self._steering_penalty(action[0])
        progress_reward = self._progress_reward(current_pose)
        centerline_reward = self._centerline_reward(scan)
        collision_penalty = self.collision_penalty if collision else 0.0
        
        # Combine reward components
        reward = (
            speed_reward * self.speed_reward_weight + 
            steering_penalty * self.steering_penalty_weight + 
            progress_reward * self.progress_reward_weight +
            centerline_reward * self.centerline_reward_weight +
            collision_penalty
        )
        
        # Check if episode is done
        self.episode_step += 1
        done = collision or self.episode_step >= self.max_episode_steps
        
        # Check for lap completion (simplified - would need track map for accurate implementation)
        if self.prev_pose is not None and self.start_position is not None:
            # Calculate distance to starting position
            dx = current_pose.position.x - self.start_position[0]
            dy = current_pose.position.y - self.start_position[1]
            dist_to_start = math.sqrt(dx*dx + dy*dy)
            
            # If close to start and we've traveled some distance, count as lap completion
            if dist_to_start < 1.0 and self.traveled_distance > 10.0:
                self.lap_count += 1
                new_lap_time = time.time() - self.lap_start_time
                
                # If this is the second lap or later (avoid counting initial position as a lap)
                if self.lap_count > 1:
                    # Add bonus for completing a lap
                    reward += 100.0
                    
                    # Store lap time and reset
                    self.lap_time = new_lap_time
                    self.lap_start_time = time.time()
                    
                    # Log lap completion if we have a node
                    if self.node:
                        self.node.get_logger().info(f"Lap {self.lap_count} completed in {self.lap_time:.2f} seconds!")
        
        # Extract state from laser scan
        ranges = np.array(scan.ranges)
        ranges[np.isinf(ranges)] = 10.0  # Replace inf with large value
        state = ranges / 10.0  # Normalize to [0, 1]
        
        # Additional info
        info = {
            'speed': current_speed,
            'collision': collision,
            'lap_progress': self.lap_progress,
            'lap_count': self.lap_count,
            'traveled_distance': self.traveled_distance,
            'lap_time': self.lap_time,
            'reward': {
                'speed': speed_reward * self.speed_reward_weight,
                'steering': steering_penalty * self.steering_penalty_weight,
                'progress': progress_reward * self.progress_reward_weight,
                'centerline': centerline_reward * self.centerline_reward_weight,
                'collision': collision_penalty,
                'total': reward
            }
        }
        
        # Update previous values
        self.prev_speed = current_speed
        self.prev_pose = current_pose
        
        return state, reward, done, info
    
    def _get_speed_from_odom(self, odom):
        """Extract linear speed from odometry message"""
        if odom is None:
            return 0.0
            
        vx = odom.twist.twist.linear.x
        vy = odom.twist.twist.linear.y
        return math.sqrt(vx**2 + vy**2)
    
    def _get_yaw_from_pose(self, pose):
        """Extract yaw angle from pose orientation quaternion"""
        if pose is None:
            return 0.0
            
        orientation_q = pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        return yaw
    
    def _check_collision(self, scan):
        """Check if any laser scan ray indicates a collision"""
        if scan is None:
            return False
            
        # Check if any laser reading is below threshold
        ranges = np.array(scan.ranges)
        min_dist = np.min(ranges[np.isfinite(ranges)]) if np.any(np.isfinite(ranges)) else float('inf')
        return min_dist < self.collision_threshold
    
    def _speed_reward(self, speed):
        """Reward for maintaining speed"""
        # Gaussian reward - highest at target speed, falls off as we deviate
        return math.exp(-0.5 * ((speed - self.target_speed) / 1.0)**2)
    
    def _steering_penalty(self, steering_angle):
        """Penalty for excessive steering"""
        # Discourage sharp turns
        return -abs(steering_angle)
    
    def _progress_reward(self, current_pose):
        """Reward for making progress around the track"""
        if self.prev_pose is None:
            return 0.0
            
        # Calculate distance traveled
        dx = current_pose.position.x - self.prev_pose.position.x
        dy = current_pose.position.y - self.prev_pose.position.y
        distance = math.sqrt(dx**2 + dy**2)
        
        # Update total traveled distance
        self.traveled_distance += distance
        
        # Progress reward proportional to distance
        return distance
    
    def _centerline_reward(self, scan):
        """Reward for following the centerline of the track"""
        if scan is None:
            return 0.0
            
        # Get left and right distances
        ranges = np.array(scan.ranges)
        n_ranges = len(ranges)
        
        # Get left and right scan indices (90 degrees left and right)
        left_idx = n_ranges // 4  # 90 degrees left
        right_idx = 3 * n_ranges // 4  # 90 degrees right
        
        # Get ranges around these points (average of 5 readings)
        left_ranges = ranges[max(0, left_idx-2):min(n_ranges, left_idx+3)]
        right_ranges = ranges[max(0, right_idx-2):min(n_ranges, right_idx+3)]
        
        # Remove inf values
        left_ranges = left_ranges[np.isfinite(left_ranges)]
        right_ranges = right_ranges[np.isfinite(right_ranges)]
        
        # If no valid readings, return neutral reward
        if len(left_ranges) == 0 or len(right_ranges) == 0:
            return 0.0
            
        # Calculate average distances
        left_dist = np.mean(left_ranges)
        right_dist = np.mean(right_ranges)
        
        # Reward is inversely proportional to the difference between left and right distances
        # The closer the car is to the centerline, the more equal these distances should be
        diff = abs(left_dist - right_dist)
        
        # Normalize difference to track width and compute reward
        normalized_diff = min(diff / self.track_width, 1.0)
        
        # Higher reward for being closer to centerline
        return math.exp(-5.0 * normalized_diff)
        
    def get_stats(self):
        """Get environment statistics"""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps_list,
            'best_reward': self.best_reward,
            'lap_count': self.lap_count,
            'best_lap_time': self.lap_time if self.lap_count > 0 else None
        }