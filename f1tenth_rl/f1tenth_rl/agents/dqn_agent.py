#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
from .base_agent import BaseAgent

class DQNetwork(nn.Module):
    """
    Deep Q-Network for F1TENTH reinforcement learning
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent(BaseAgent):
    """
    DQN Agent implementation for the F1TENTH simulator
    """
    def __init__(self, state_dim, action_space, hidden_dim=256, learning_rate=3e-4, 
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update_freq=10,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the DQN agent.
        
        Args:
            state_dim: Dimensionality of the state space
            action_space: List of possible actions or action space dimensions
            hidden_dim: Size of hidden layers in the network
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Rate at which exploration decays
            buffer_size: Size of the replay buffer
            batch_size: Number of samples to use for each update
            target_update_freq: How often to update the target network
            device: Device to run the neural networks on ('cpu' or 'cuda')
        """
        super().__init__(state_dim, action_space, device)
        
        if isinstance(action_space, list):
            self.action_dim = len(action_space)
            self.actions = action_space
            self.discrete = True
        else:
            # For future support of continuous actions if needed
            self.action_dim = action_space
            self.actions = None
            self.discrete = False
        
        # Create Q-networks (online and target)
        self.q_network = DQNetwork(state_dim, self.action_dim, hidden_dim).to(device)
        self.target_network = DQNetwork(state_dim, self.action_dim, hidden_dim).to(device)
        
        # Copy parameters from online to target network
        self.update_target_network()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        # Training metrics
        self.loss_history = []
    
    def update_target_network(self):
        """Copy parameters from online to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def select_action(self, state, deterministic=False):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            deterministic: Whether to select actions deterministically
            
        Returns:
            Selected action values
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
        # Set networks to evaluation mode for inference
        self.q_network.eval()
            
        # Epsilon-greedy exploration in training mode
        if not deterministic and self.train_mode and random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                action_idx = torch.argmax(q_values, dim=1).item()
        
        # Set networks back to training mode if in training
        if self.train_mode:
            self.q_network.train()
            
        if self.discrete:
            return self.actions[action_idx], action_idx
        else:
            return action_idx
    
    def update(self, state, action, reward, next_state, done):
        """
        Update the agent using experience replay
        
        Args:
            state: Current state
            action: Action taken (index for discrete actions)
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            
        Returns:
            Dictionary with loss information
        """
        # Store transition in replay buffer
        self.replay_buffer.append((state, action, reward, next_state, done))
        
        # Only update if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return {'loss': None}
            
        # Sample mini-batch from replay buffer
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        # Get current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Update online network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_network()
            self.update_counter = 0
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Store loss
        self.loss_history.append(loss.item())
        
        return {'loss': loss.item(), 'epsilon': self.epsilon}
    
    def save(self, path):
        """Save model to file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'hyperparams': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'target_update_freq': self.target_update_freq,
                'batch_size': self.batch_size
            }
        }, path)
    
    def load(self, path):
        """Load model from file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
            
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        
        # Optionally load hyperparameters if they exist in the checkpoint
        if 'hyperparams' in checkpoint:
            hyperparams = checkpoint['hyperparams']
            self.gamma = hyperparams.get('gamma', self.gamma)
            self.epsilon_min = hyperparams.get('epsilon_min', self.epsilon_min)
            self.epsilon_decay = hyperparams.get('epsilon_decay', self.epsilon_decay)
            self.target_update_freq = hyperparams.get('target_update_freq', self.target_update_freq)
            self.batch_size = hyperparams.get('batch_size', self.batch_size)