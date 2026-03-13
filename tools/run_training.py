#!/usr/bin/env python3
"""
Simple runner script for training with YAML configuration.
This script demonstrates how to run training with the YAML config file.
"""

import os
import sys
import yaml

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_and_modify_config(config_path='configs/default.yaml', overrides=None):
    """
    Load configuration from YAML file and optionally override specific values.
    
    Args:
        config_path: Path to config file
        overrides: Dictionary of values to override in the config
        
    Returns:
        dict: Modified configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides if provided
    if overrides:
        config.update(overrides)
    
    return config

def main():
    """
    Example of how to run training with custom configuration overrides.
    """
    # Example 1: Use default configuration
    print("Running with default configuration...")
    os.system("python tools/train.py")
    
    # Example 2: Run with custom overrides (uncomment to use)
    # overrides = {
    #     'num_epochs': 100,
    #     'batch_size': 4,
    #     'learning_rate': 0.0002,
    #     'use_wandb': False
    # }
    # config = load_and_modify_config(overrides=overrides)
    # 
    # # Save modified config to a temporary file
    # temp_config_path = 'configs/temp_config.yaml'
    # with open(temp_config_path, 'w') as f:
    #     yaml.dump(config, f, default_flow_style=False)
    # 
    # print("Running with custom configuration...")
    # print(f"Configuration: {config}")
    # 
    # # You would need to modify train.py to accept a config path parameter
    # # or modify the load_config function to use the temp config

if __name__ == "__main__":
    main() 