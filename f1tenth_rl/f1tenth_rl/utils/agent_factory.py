#!/usr/bin/env python3
"""
Agent factory utility for creating RL agents based on specified type
"""

import os
import torch
import numpy as np

def create_agent(model_type, state_dim, **kwargs):
    """
    Create an RL agent based on specified type
    
    Args:
        model_type: Type of RL algorithm ('dqn', 'ppo', etc.)
        state_dim: Dimensionality of the state space
        **kwargs: Additional parameters for the agent
        
    Returns:
        Instantiated agent object
    """
    if model_type.lower() == 'dqn':
        from f1tenth_rl.agents.dqn_agent import DQNAgent
        
        # Create discrete action space if not provided
        if 'action_space' not in kwargs:
            actions = []
            for steering in np.linspace(-0.4, 0.4, 5):  # 5 steering angles
                for velocity in [1.0, 2.0, 3.0]:  # 3 velocity options
                    actions.append([steering, velocity])
            kwargs['action_space'] = actions
        
        # Create agent with default or provided parameters
        return DQNAgent(
            state_dim=state_dim,
            hidden_dim=kwargs.get('hidden_dim', 256),
            learning_rate=kwargs.get('learning_rate', 3e-4),
            gamma=kwargs.get('gamma', 0.99),
            epsilon_start=kwargs.get('epsilon_start', 1.0),
            epsilon_end=kwargs.get('epsilon_end', 0.05),
            epsilon_decay=kwargs.get('epsilon_decay', 0.995),
            batch_size=kwargs.get('batch_size', 64),
            buffer_size=kwargs.get('buffer_size', 10000),
            target_update_freq=kwargs.get('target_update_freq', 10),
            device=kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
            **kwargs
        )
    
    # Add support for other agent types here
    # elif model_type.lower() == 'ppo':
    #     from f1tenth_rl.agents.ppo_agent import PPOAgent
    #     return PPOAgent(...)
    
    else:
        raise ValueError(f"Unsupported agent type: {model_type}")

def load_agent(model_path, model_type='dqn'):
    """
    Load a pre-trained agent from disk
    
    Args:
        model_path: Path to the saved model file
        model_type: Type of RL algorithm
        
    Returns:
        Loaded agent object
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # For DQN agent
    if model_type.lower() == 'dqn':
        from f1tenth_rl.agents.dqn_agent import DQNAgent
        
        # Create a temporary agent with dummy values
        # (real values will be loaded from checkpoint)
        agent = DQNAgent(state_dim=1080, action_space=[[0, 0]])
        
        # Load model parameters
        agent.load(model_path)
        
        return agent
    
    # Add support for other agent types here
    
    else:
        raise ValueError(f"Unsupported agent type: {model_type}")