# DQN-Breakout
RL agent using DQN to solve Breakout a ATARI game


## Conda env setup

need to be run in *WSL* since envpool does not work on Windows
-- install miniconda:
'''
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh
source ~/.bashrc
conda --version
'''

-- Create conda env
'''
conda create -n envpool python3.10 -y
conda activate envpool
'''

-- install pytorch
'''
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
'''

-- install requirements
'''
pip install -r requirements
'''

## Implementation Log

**2025-11-21**: Enhanced DQN implementation with envpool integration for faster training. Fixed compatibility issues between old gym (envpool) and new gymnasium (stable-baselines3), implemented GPU-accelerated TorchAtariReplayBuffer, and added optimized hyperparameter configurations (breakout_fast, breakout_ultrafast) with organized output structure in config-specific subdirectories.


**2025-11-22**: implemented a max step param for the overall training
Ajusting graphing,
adding seperate file for view the agents in agent. Add to be implemented in seperate file since envpool doesn't allow render. And I did't want for both gym environment to be running in same file.



