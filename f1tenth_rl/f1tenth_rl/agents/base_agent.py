#!/usr/bin/env python3

import torch
import numpy as np
import os
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract base class for all reinforcement learning agents.
    This provides a common interface for different RL algorithms.
    """
    def __init__(self, state_dim, action_spec, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the base agent.
        
        Args:
            state_dim: Dimensionality of the state space
            action_spec: Specification of the action space (can be discrete or continuous)
            device: Device to run the neural networks on ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.action_spec = action_spec
        self.device = device
        self.train_mode = True
        
    @abstractmethod
    def select_action(self, state, deterministic=False):
        """
        Select an action based on the current state.
        
        Args:
            state: Current state
            deterministic: Whether to select actions deterministically (for evaluation)
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        """
        Update the agent's policy based on experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            
        Returns:
            Dictionary of update metrics (e.g., loss values)
        """
        pass
    
    @abstractmethod
    def save(self, path):
        """
        Save the agent's model to a file.
        
        Args:
            path: Path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path):
        """
        Load the agent's model from a file.
        
        Args:
            path: Path to load the model from
        """
        pass
    
    def train(self):
        """Set the agent to training mode."""
        self.train_mode = True
    
    def eval(self):
        """Set the agent to evaluation mode."""
        self.train_mode = False
        
    def get_model_info(self):
        """Get information about the agent's model(s)."""
        return {
            'type': self.__class__.__name__,
            'state_dim': self.state_dim,
            'action_spec': self.action_spec
        }