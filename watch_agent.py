"""
Watch a trained DQN agent play Atari Breakout
Uses standard gym (not envpool) for visualization support
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import gymnasium as gym
import ale_py  # Required for ALE namespace registration
from collections import deque
from dqn import DQN

# Register ALE environments
gym.register_envs(ale_py)

# Standalone boilerplate before relative imports
from pathlib import Path
if __package__ is None:
    DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(DIR.parent))
    __package__ = DIR.name

RUNS_DIR = "runs"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class AgentViewer:
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameters = yaml.safe_load(file)
            hyperparameters = all_hyperparameters[hyperparameter_set]
            
        self.hyperparameter_set = hyperparameter_set
        self.env_id = hyperparameters['env_id']
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']
        
        # Path to saved model
        self.config_dir = os.path.join(RUNS_DIR, self.hyperparameter_set)
        self.MODEL_FILE = os.path.join(self.config_dir, f'{self.hyperparameter_set}.pt')
        
    def watch(self, num_episodes=5, render_mode='human', frame_stack=4):
        """
        Watch the trained agent play
        
        Args:
            num_episodes: Number of episodes to watch
            render_mode: 'human' for window display, 'rgb_array' for recording
            frame_stack: Number of frames to stack (default 4, matching envpool)
        """
        # Check if model exists
        if not os.path.exists(self.MODEL_FILE):
            raise FileNotFoundError(f"Model file not found: {self.MODEL_FILE}")
        
        print(f"Loading model from: {self.MODEL_FILE}")
        
        # Create environment with rendering
        # Gymnasium uses ALE/ prefix for Atari environments
        env_name = self.env_id
        if not env_name.startswith('ALE/'):
            # Convert Breakout-v5 to ALE/Breakout-v5
            env_name = f'ALE/{env_name}'
        
        # Wrap with frame stacking to match training setup (envpool uses 4 frames by default)
        env = gym.make(env_name, render_mode=render_mode, frameskip=1)
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.FrameStackObservation(env, frame_stack)
        
        # Get observation and action space info
        num_actions = env.action_space.n
        obs_shape = env.observation_space.shape
        
        print(f"Environment: {self.env_id}")
        print(f"Observation shape: {obs_shape} (after preprocessing)")
        print(f"Number of actions: {num_actions}")
        print(f"Frame stacking: {frame_stack} frames")
        
        # Create and load policy network
        policy_dqn = DQN(obs_shape, num_actions, self.enable_dueling_dqn).to(device)
        policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device, weights_only=False))
        policy_dqn.eval()
        
        print(f"\nWatching agent play for {num_episodes} episodes...")
        print("Press Ctrl+C to stop early\n")
        
        total_rewards = []
        
        try:
            for episode in range(num_episodes):
                state, _ = env.reset()
                state = np.array(state)
                
                episode_reward = 0.0
                episode_steps = 0
                terminated = False
                truncated = False
                
                while not (terminated or truncated):
                    # Select action greedily (no exploration)
                    with torch.no_grad():
                        # FrameStackObservation returns shape (stack, H, W) which is already correct
                        # Convert LazyFrames to numpy array if needed
                        state_array = np.array(state)
                        state_tensor = torch.from_numpy(state_array).float().unsqueeze(0).to(device)
                        action = policy_dqn(state_tensor).squeeze().argmax().item()
                    
                    # Take action in environment
                    next_state, reward, terminated, truncated, info = env.step(action)
                    next_state = np.array(next_state)
                    
                    episode_reward += reward
                    episode_steps += 1
                    state = next_state
                    
                    # Print progress every 50 steps
                    if episode_steps % 50 == 0:
                        print(f"  Step {episode_steps}: reward={episode_reward:.1f}")
                
                total_rewards.append(episode_reward)
                print(f"Episode {episode + 1}/{num_episodes}: Reward = {episode_reward:.2f}, Steps = {episode_steps}")
        
        except KeyboardInterrupt:
            print("\nStopped by user")
        
        finally:
            env.close()
            
        if total_rewards:
            print(f"\nAverage reward over {len(total_rewards)} episodes: {np.mean(total_rewards):.2f}")
            print(f"Min: {np.min(total_rewards):.2f}, Max: {np.max(total_rewards):.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Watch a trained DQN agent play Atari')
    parser.add_argument('hyperparameters', help='Hyperparameter set name (e.g., breakout1)')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to watch (default: 5)')
    parser.add_argument('--render-mode', type=str, default='rgb_array', 
                        choices=['human', 'rgb_array'], 
                        help='Rendering mode: "human" for window (requires X11), "rgb_array" for headless (default: rgb_array)')
    args = parser.parse_args()
    
    viewer = AgentViewer(hyperparameter_set=args.hyperparameters)
    viewer.watch(num_episodes=args.episodes, render_mode=args.render_mode)
