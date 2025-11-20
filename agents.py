import csv
import gymnasium as gym 
import ale_py  # Required for ALE namespace registration
import matplotlib
import torch
from torch import nn
from dqn import DQN
import itertools
import yaml
import random
import os
import numpy as np
import matplotlib
import argparse
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.buffers import ReplayBuffer


# for printing date and time
DATE_FORMAT = "%m-%d %H-%M-%S"

# directory to saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg' : used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class Agent:
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameters_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameters_sets[hyperparameter_set]
            
        self.hyperparameter_set = hyperparameter_set    
            
        self.env_id             = hyperparameters['env_id']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size    = hyperparameters['mini_batch_size']
        self.epsilon_init       = hyperparameters['epsilon_init']
        self.epsilon_decay      = hyperparameters['epsilon_decay']
        self.epsilon_min        = hyperparameters['epsilon_min']
        self.learning_rate_a    = hyperparameters['learning_rate_a']
        self.discount_factor_g  = hyperparameters['discount_factor_g']
        self.stop_on_reward     = hyperparameters['stop_on_reward']
        self.network_sync_rate  = hyperparameters['network_sync_rate']
        self.env_make_params    = hyperparameters.get('env_make_params', {})
        self.enable_double_dqn  = hyperparameters['enable_double_dqn']
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']

        self.loss_fn = nn.HuberLoss()   # NN Loss function. Huber loss is more robust to outliers than MSE 
        self.optimizer = None         # NN optimizer. Initialized later
        
        #path to run info
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')
        self.METRICS_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}_metrics.csv')
        self._metrics_header_written = False
    
    def run(self, is_training=True, render=False):
        if is_training:
            start_time=datetime.now()
            last_graph_update_time = start_time
            
            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')
            self._init_metrics_file()

        env = gym.make('ALE/Breakout-v5', render_mode='human' if render else None)
        env = CropObservation(env, top=34, bottom=0)  # remove scoreboard rows
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = MaxAndSkipEnv(env, skip=4)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        
        num_actions = env.action_space.n
        obs_shape = env.observation_space.shape
        
        reward_per_episode = []
        epsilon_by_episode = []
        mean_reward_history = []
        steps_history = []
        
        policy_dqn = DQN(obs_shape, num_actions, self.enable_dueling_dqn).to(device)
        
        
        if is_training:
            # initialize replay memory
            memory = ReplayBuffer(
                buffer_size=self.replay_memory_size,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device,
                optimize_memory_usage=True,
                handle_timeout_termination=False
            )

            
            # initialize epsilon
            epsilon = self.epsilon_init
            total_steps = 0
            last_loss = None

            # create target network and make it identical to the policy network
            target_dqn = DQN(obs_shape, num_actions, self.enable_dueling_dqn).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            
            # Policy network optimizer. "Adam" optimizer can be swapped to something else
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
            
            # track number of steps taken. used for syncing policy => target network
            step_count = 0
            
            # track best reward
            best_reward = -999999
        
        else:
            # load learned policy
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            
            # switch model to evaluation mode
            policy_dqn.eval()
            

        for episode in itertools.count():
            state, info = env.reset()
            state = np.asarray(state, dtype=np.uint8)
            terminated = False
            episode_reward = 0.0
            episode_steps = 0
            
            if is_training:
                    current_life = info.get('lives', 0) if 'lives' in info else 0
            
            while (not terminated and episode_reward < self.stop_on_reward):
                # select action based on epsilon greedy
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        state_tensor = torch.from_numpy(state).float().to(device)
                        action = policy_dqn(state_tensor.unsqueeze(dim=0)).squeeze().argmax()



                
                # Processing:
                action_value = int(action.item()) if isinstance(action, torch.Tensor) else int(action)
                new_state, reward, terminated, truncated, info = env.step(action_value)
                new_state = np.asarray(new_state, dtype=np.uint8)
                
                episode_reward += reward
                episode_steps += 1
                
                
                if is_training:
                    total_steps += 1
                    # handle life loss for breakout
                    done = terminated or truncated
                    if 'lives' in info:
                        if info['lives'] < current_life:
                            done = True
                        current_life = info['lives']
                            
                    memory.add(
                        obs=state,
                        next_obs=new_state,
                        action=np.array(action_value, dtype=np.int64),
                        reward=np.array(reward, dtype=np.float32),
                        done=np.array(done, dtype=bool),
                        infos=[info]
                    )
                    
                    step_count+=1
    

                state = new_state
                
            reward_per_episode.append(episode_reward)
            
            if is_training:
                epsilon_by_episode.append(epsilon)
                steps_history.append(total_steps)
                mean_reward = np.mean(reward_per_episode[-100:])
                mean_reward_history.append(mean_reward)
                self._append_metrics_row(
                    episode=episode,
                    total_steps=total_steps,
                    episode_reward=episode_reward,
                    rolling_reward=mean_reward,
                    epsilon=epsilon,
                    loss=last_loss
                )
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message+ '\n')
                
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(
                        steps_history=steps_history,
                        rewards_per_episode=reward_per_episode,
                        mean_rewards=mean_reward_history,
                        epsilon_history=epsilon_by_episode
                    )
                    last_graph_update_time = current_time
                
                if memory.pos >= self.mini_batch_size:
                    #Sample from memory
                    data = memory.sample(self.mini_batch_size)
                    last_loss = self.optimize(data, policy_dqn, target_dqn)
                    
                    #decay epsilon
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    
                    #copy policy network to target network after a certain number of step
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count=0
                    
    def save_graph(self, steps_history, rewards_per_episode, mean_rewards, epsilon_history):
        if not steps_history:
            return

        fig = plt.figure(1, figsize=(10, 6))

        plt.subplot(211)
        plt.xlabel('Environment Steps')
        plt.ylabel('Mean Reward (100 eps)')
        plt.plot(steps_history, mean_rewards, label='Rolling avg reward')
        plt.plot(steps_history, rewards_per_episode[-len(steps_history):], alpha=0.3, label='Episode reward')
        plt.legend(loc='lower right')

        plt.subplot(212)
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon')
        plt.plot(range(len(epsilon_history)), epsilon_history)

        plt.subplots_adjust(hspace=0.6)

        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    def _init_metrics_file(self):
        with open(self.METRICS_FILE, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["episode", "total_steps", "reward", "rolling_reward_100", "epsilon", "loss"])
        self._metrics_header_written = True

    def _append_metrics_row(self, episode, total_steps, episode_reward, rolling_reward, epsilon, loss):
        if not self._metrics_header_written:
            self._init_metrics_file()
        loss_value = float(loss) if loss is not None else float('nan')
        with open(self.METRICS_FILE, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([episode, total_steps, episode_reward, rolling_reward, epsilon, loss_value])

                    
    def optimize(self, data, policy_dqn, target_dqn):
        
        
        states = data.observations.to(device)
        actions = data.actions.long().to(device)
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        new_states = data.next_observations.to(device)
        rewards = data.rewards.to(device)
        terminations = data.dones.to(device)
        
        with torch.no_grad():
            if self.enable_double_dqn:
                # Double DQN
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (1-terminations) * self.discount_factor_g * \
                    target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
            
        
            
        current_q = policy_dqn(states).gather(1, actions).squeeze(1)
        
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients(backpropagation)
        self.optimizer.step()       # Update network parameters i.e. weights and biases
        return loss.item()
            
class CropObservation(gym.ObservationWrapper):
    def __init__(self, env, top=34, bottom=0):
        super().__init__(env)
        h, w = env.observation_space.shape[:2]
        self.top = top
        self.bottom = bottom
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(h - top - bottom, w, env.observation_space.shape[2]),
            dtype=np.uint8,
        )

    def observation(self, obs):
        return obs[self.top : obs.shape[0] - self.bottom]
            
if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)