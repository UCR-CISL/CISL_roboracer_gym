#!/usr/bin/env python3
"""
CLI utilities for F1TENTH RL package
Provides wrapper functions for training, evaluation, and deployment
"""

import subprocess
import os
import sys
import argparse
import time
from datetime import datetime

def run_simulator():
    """Start the F1TENTH simulator"""
    try:
        print("Starting F1TENTH simulator...")
        # Use subprocess.Popen to start in a new process so it doesn't block
        process = subprocess.Popen(
            "ros2 launch f1tenth_gym_ros gym_bridge_launch.py",
            shell=True
        )
        
        # Wait a bit for the simulator to initialize
        time.sleep(5)
        
        return process
    except Exception as e:
        print(f"Error starting simulator: {str(e)}")
        return None

def run_agent(training_mode=True, model_type='dqn', model_path='', save_path='models/'):
    """Start the RL agent with specified parameters"""
    try:
        # Build command with parameters
        cmd = [
            "ros2 launch f1tenth_rl rl_agent_launch.py",
            f"training_mode:={str(training_mode).lower()}",
            f"model_type:={model_type}"
        ]
        
        # Only add model_path if provided
        if model_path:
            cmd.append(f"model_path:={model_path}")
            
        # Add save_path
        cmd.append(f"save_path:={save_path}")
        
        # Join and execute
        cmd_str = " ".join(cmd)
        print(f"Running agent with command: {cmd_str}")
        
        # Start agent process
        process = subprocess.Popen(cmd_str, shell=True)
        
        return process
    except Exception as e:
        print(f"Error starting agent: {str(e)}")
        return None

def train(args):
    """Train a new RL model or continue training an existing one"""
    # Create a timestamp-based directory for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_path, f"{args.model_type}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting training with {args.model_type} agent...")
    print(f"Models will be saved to: {save_dir}")
    
    # Start simulator
    sim_process = run_simulator()
    if not sim_process:
        print("Failed to start simulator. Exiting.")
        return
    
    try:
        # Start agent
        agent_process = run_agent(
            training_mode=True,
            model_type=args.model_type,
            model_path=args.model_path,
            save_path=save_dir
        )
        
        if not agent_process:
            print("Failed to start agent. Exiting.")
            sim_process.terminate()
            return
        
        # Main loop to keep process running and handle keyboard interrupt
        print("\nTraining started. Press Ctrl+C to stop...\n")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Clean up
        print("Shutting down...")
        if 'agent_process' in locals():
            agent_process.terminate()
        if 'sim_process' in locals():
            sim_process.terminate()
        print("Training session ended.")

def evaluate(args):
    """Evaluate a trained model"""
    if not args.model_path:
        print("Error: Model path is required for evaluation.")
        return
        
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    print(f"Starting evaluation of model: {args.model_path}")
    
    # Start simulator
    sim_process = run_simulator()
    if not sim_process:
        print("Failed to start simulator. Exiting.")
        return
    
    try:
        # Start agent in evaluation mode
        agent_process = run_agent(
            training_mode=False,
            model_type=args.model_type,
            model_path=args.model_path
        )
        
        if not agent_process:
            print("Failed to start agent. Exiting.")
            sim_process.terminate()
            return
        
        # Main loop to keep process running and handle keyboard interrupt
        print("\nEvaluation started. Press Ctrl+C to stop...\n")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    finally:
        # Clean up
        print("Shutting down...")
        if 'agent_process' in locals():
            agent_process.terminate()
        if 'sim_process' in locals():
            sim_process.terminate()
        print("Evaluation session ended.")

def main():
    """Main entry point for CLI utility"""
    parser = argparse.ArgumentParser(description='F1TENTH RL CLI Tools')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--model-type', type=str, default='dqn',
                             choices=['dqn'], help='Type of RL algorithm')
    train_parser.add_argument('--model-path', type=str, default='',
                            help='Path to existing model to continue training (optional)')
    train_parser.add_argument('--save-path', type=str, default='models',
                            help='Directory to save trained models')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a model')
    eval_parser.add_argument('--model-path', type=str, required=True,
                           help='Path to trained model')
    eval_parser.add_argument('--model-type', type=str, default='dqn',
                           choices=['dqn'], help='Type of RL algorithm')
    
    args = parser.parse_args()
    
    # Execute appropriate command
    if args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()