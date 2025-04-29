#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np
import torch
import os
import yaml
from datetime import datetime
import time
import threading
import matplotlib.pyplot as plt

from f1tenth_rl.environment import F1TenthEnv
from f1tenth_rl.agents.dqn_agent import DQNAgent
# Optionally support PPO when implemented
# from f1tenth_rl.agents.ppo_agent import PPOAgent

class RLAgentNode(Node):
    def __init__(self):
        super().__init__('rl_agent_node')
        
        # Declare parameters
        self.declare_parameter('training_mode', True)
        self.declare_parameter('model_type', 'dqn')
        self.declare_parameter('model_path', '')
        self.declare_parameter('save_path', 'models/')
        self.declare_parameter('max_steps_per_episode', 1000)
        self.declare_parameter('save_interval', 10)
        self.declare_parameter('track_width', 2.0)
        
        # Environment parameters
        self.declare_parameter('collision_threshold', 0.3)
        self.declare_parameter('target_speed', 2.0)
        
        # Agent parameters
        self.declare_parameter('hidden_dim', 256)
        self.declare_parameter('learning_rate', 0.0001)
        self.declare_parameter('batch_size', 64)
        self.declare_parameter('gamma', 0.98)
        self.declare_parameter('epsilon_start', 1.0)
        self.declare_parameter('epsilon_end', 0.05)
        self.declare_parameter('epsilon_decay', 0.99)
        
        # Reward parameters
        self.declare_parameter('collision_penalty', -500.0)
        self.declare_parameter('speed_reward_weight', 0.3)
        self.declare_parameter('steering_penalty_weight', -10.0)
        self.declare_parameter('progress_reward_weight', 5.0)
        self.declare_parameter('centerline_reward_weight', 25.0)
        
        # Car starting position
        self.declare_parameter('start_x', 0.0)
        self.declare_parameter('start_y', 0.0)
        self.declare_parameter('start_yaw', 0.0)
        
        # Get parameters
        self.training_mode = self.get_parameter('training_mode').value
        self.model_type = self.get_parameter('model_type').value
        self.model_path = self.get_parameter('model_path').value
        self.save_path = self.get_parameter('save_path').value
        self.save_interval = self.get_parameter('save_interval').value
        
        # Create directories if they don't exist
        os.makedirs(self.save_path, exist_ok=True)
        
        # Set up configuration for environment
        self.env_config = {
            'max_steps_per_episode': self.get_parameter('max_steps_per_episode').value,
            'track_width': self.get_parameter('track_width').value,
            'collision_threshold': self.get_parameter('collision_threshold').value,
            'target_speed': self.get_parameter('target_speed').value,
            'collision_penalty': self.get_parameter('collision_penalty').value,
            'speed_reward_weight': self.get_parameter('speed_reward_weight').value,
            'steering_penalty_weight': self.get_parameter('steering_penalty_weight').value,
            'progress_reward_weight': self.get_parameter('progress_reward_weight').value,
            'centerline_reward_weight': self.get_parameter('centerline_reward_weight').value,
            'start_x': self.get_parameter('start_x').value,
            'start_y': self.get_parameter('start_y').value,
            'start_yaw': self.get_parameter('start_yaw').value
        }
        
        # Publishers and subscribers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        
        # Store the latest observations
        self.latest_scan = None
        self.latest_odom = None
        
        # Initialize environment with reference to this node
        self.env = F1TenthEnv(node=self, config=self.env_config)
        
        # Initialize RL agent
        self.initialize_agent()
        
        # Training data
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode = 0
        self.current_episode_steps = 0
        self.current_episode_reward = 0.0
        self.best_episode_reward = -float('inf')
        self.total_steps = 0
        
        # Training loop timer (runs at 10Hz)
        self.timer = self.create_timer(0.1, self.training_loop)
        
        # Visualization timer (update plots every 30 seconds if in training mode)
        if self.training_mode:
            self.plot_timer = self.create_timer(30.0, self.update_plots)
        
        self.get_logger().info(f"RL Agent Node initialized in {'training' if self.training_mode else 'evaluation'} mode")
        self.get_logger().info(f"Using {self.model_type} agent")
        if self.model_path:
            self.get_logger().info(f"Loaded model from {self.model_path}")
        else:
            self.get_logger().info("Training from scratch")
            
    def initialize_agent(self):
        """Initialize the RL agent based on specified model type"""
        # Get laser scan dimensions
        state_dim = 1080  # Default, will be updated after first scan
        
        # Define discrete action space
        self.actions = []
        for steering in np.linspace(-0.4, 0.4, 5):  # 5 steering angles from -0.4 to 0.4 rad
            for velocity in [1.0, 2.0, 3.0]:  # 3 velocity options
                self.actions.append([steering, velocity])
        
        if self.model_type.lower() == 'dqn':
            # Setup hyperparameters for DQN
            self.agent = DQNAgent(
                state_dim=state_dim,
                action_space=self.actions,
                hidden_dim=self.get_parameter('hidden_dim').value,
                learning_rate=self.get_parameter('learning_rate').value,
                gamma=self.get_parameter('gamma').value,
                epsilon_start=self.get_parameter('epsilon_start').value,
                epsilon_end=self.get_parameter('epsilon_end').value,
                epsilon_decay=self.get_parameter('epsilon_decay').value,
                batch_size=self.get_parameter('batch_size').value
            )
            
            self.get_logger().info(f"Initialized DQN agent with {len(self.actions)} discrete actions")
            
        # elif self.model_type.lower() == 'ppo':
            # Future support for PPO implementation
        else:
            self.get_logger().error(f"Unsupported model type: {self.model_type}")
            return
        
        # Load model if specified
        if self.model_path:
            try:
                self.agent.load(self.model_path)
                self.get_logger().info(f"Successfully loaded model from {self.model_path}")
                
                # Set agent to evaluation mode if not in training
                if not self.training_mode:
                    self.agent.eval()
            except Exception as e:
                self.get_logger().error(f"Failed to load model: {str(e)}")
                
    def scan_callback(self, msg):
        """Store the latest laser scan data"""
        self.latest_scan = msg
        
        # Update state dimension if needed
        if hasattr(self, 'agent') and len(msg.ranges) != self.agent.state_dim:
            self.get_logger().info(f"Updating state dimension to {len(msg.ranges)}")
            self.agent.state_dim = len(msg.ranges)
    
    def odom_callback(self, msg):
        """Store the latest odometry data"""
        self.latest_odom = msg
        
    def publish_drive_command(self, steering, velocity):
        """Publish drive command to the F1TENTH car"""
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = float(steering)
        msg.drive.speed = float(velocity)
        self.drive_pub.publish(msg)
        
    def get_state_from_scan(self, scan):
        """Convert laser scan to normalized state vector"""
        if scan is None:
            return None
            
        # Extract ranges and handle inf values
        ranges = np.array(scan.ranges)
        ranges[np.isinf(ranges)] = 10.0  # Replace inf with a large value
        
        # Normalize ranges to [0, 1]
        normalized_ranges = ranges / 10.0
        
        return normalized_ranges
    
    def training_loop(self):
        """Main RL training/inference loop"""
        # Wait for sensor data
        if self.latest_scan is None or self.latest_odom is None:
            return
            
        # Get current state
        current_state = self.get_state_from_scan(self.latest_scan)
        if current_state is None:
            return
            
        # Select action
        if self.training_mode:
            # In training mode, select action with exploration
            action_values, action_idx = self.agent.select_action(current_state)
        else:
            # In evaluation mode, select action deterministically
            action_values, action_idx = self.agent.select_action(current_state, deterministic=True)
            
        # Extract steering angle and velocity from action
        steering, velocity = action_values
        
        # Execute action
        self.publish_drive_command(steering, velocity)
        
        # Step environment (will be processed in next callback)
        if self.training_mode and hasattr(self, 'prev_state'):
            # Step the environment
            next_state, reward, done, info = self.env.step(
                action=[steering, velocity],
                scan=self.latest_scan,
                odom=self.latest_odom
            )
            
            # Update agent
            metrics = self.agent.update(
                self.prev_state,
                action_idx,
                reward,
                next_state,
                done
            )
            
            # Track episode progress
            self.current_episode_reward += reward
            self.current_episode_steps += 1
            self.total_steps += 1
            
            # Handle episode end
            if done or self.current_episode_steps >= self.env_config['max_steps_per_episode']:
                # Log episode stats
                self.get_logger().info(
                    f"Episode {self.current_episode+1} completed: "
                    f"Steps={self.current_episode_steps}, "
                    f"Reward={self.current_episode_reward:.2f}, "
                    f"Epsilon={self.agent.epsilon:.4f}"
                )
                
                # Store episode data
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_steps)
                
                # Check for best episode
                if self.current_episode_reward > self.best_episode_reward:
                    self.best_episode_reward = self.current_episode_reward
                    # Save best model
                    best_model_path = os.path.join(self.save_path, f"{self.model_type}_best.pt")
                    self.agent.save(best_model_path)
                    self.get_logger().info(f"New best episode! Saved model to {best_model_path}")
                
                # Save model periodically
                if (self.current_episode + 1) % self.save_interval == 0:
                    save_path = os.path.join(
                        self.save_path, 
                        f"{self.model_type}_episode_{self.current_episode+1}.pt"
                    )
                    self.agent.save(save_path)
                    self.get_logger().info(f"Saved model checkpoint to {save_path}")
                
                # Reset for next episode
                self.env.reset()
                self.current_episode += 1
                self.current_episode_steps = 0
                self.current_episode_reward = 0.0
                self.prev_state = None
                return
            
        # Store current state for next update
        self.prev_state = current_state
    
    def update_plots(self):
        """Update and save training plots"""
        if not self.training_mode or len(self.episode_rewards) < 2:
            return
            
        try:
            # Create plots directory
            plots_dir = os.path.join(self.save_path, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot episode rewards
            plt.figure(figsize=(10, 6))
            plt.plot(self.episode_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title(f'{self.model_type.upper()} Training Progress')
            plt.savefig(os.path.join(plots_dir, 'episode_rewards.png'))
            plt.close()
            
            # Plot moving average of rewards
            if len(self.episode_rewards) >= 10:
                window_size = min(10, len(self.episode_rewards))
                moving_avg = np.convolve(
                    self.episode_rewards, 
                    np.ones(window_size)/window_size, 
                    mode='valid'
                )
                
                plt.figure(figsize=(10, 6))
                plt.plot(moving_avg)
                plt.xlabel('Episode')
                plt.ylabel(f'Average Reward (window={window_size})')
                plt.title(f'{self.model_type.upper()} Moving Average Reward')
                plt.savefig(os.path.join(plots_dir, 'moving_avg_reward.png'))
                plt.close()
                
            # Save training stats
            np.savetxt(
                os.path.join(plots_dir, 'episode_rewards.txt'),
                np.array(self.episode_rewards)
            )
            np.savetxt(
                os.path.join(plots_dir, 'episode_lengths.txt'),
                np.array(self.episode_lengths)
            )
            
            self.get_logger().info(f"Updated training plots in {plots_dir}")
        except Exception as e:
            self.get_logger().error(f"Error updating plots: {str(e)}")
    
def main(args=None):
    rclpy.init(args=args)
    
    try:
        # Create and run the RL agent node
        rl_agent_node = RLAgentNode()
        
        # Spin the node to process callbacks
        rclpy.spin(rl_agent_node)
    except KeyboardInterrupt:
        print('Keyboard interrupt detected, shutting down...')
    except Exception as e:
        print(f'Unexpected error: {str(e)}')
    finally:
        # Clean up and shutdown
        if 'rl_agent_node' in locals():
            rl_agent_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
    