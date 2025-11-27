# DQN-Breakout

A Deep Q-Network (DQN) reinforcement learning agent for solving Atari Breakout.

## Environment Setup

### Conda Environment

Note: Some features require WSL on Windows as `envpool` does not work natively on Windows.

**Install Miniconda:**

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh
source ~/.bashrc
conda --version
```

**Create Conda Environment:**

```bash
conda create -n envpool python=3.10 -y
conda activate envpool
```

**Install PyTorch:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Install Requirements:**

```bash
pip install -r requirements.txt
```

## Features

- **Double DQN**: Reduces overestimation bias
- **Dueling DQN**: Separates value and advantage estimation
- **Experience Replay**: Stabilizes training with replay buffer
- **Target Network**: Periodic synchronization for stable learning
- **Background Replacement**: Test robustness with noise and natural video backgrounds
- **Configurable Hyperparameters**: YAML-based configuration system

## Usage

Run training with different configurations:

```bash
python agents2.0.py breakout1 --train
python agents2.0.py breakout_noise  --train  # With Gaussian noise background
python agents2.0.py breakout_video  --train # With natural video background
```

Watch trained agent:

```bash
python watch_agent.py --model runs/breakout1/breakout1.pt
```

## Implementation Log

**2025-11-21**: Enhanced DQN implementation with `envpool` integration for faster training. Fixed compatibility issues between old gym (`envpool`) and new gymnasium (`stable-baselines3`), implemented GPU-accelerated `TorchAtariReplayBuffer`, and added optimized hyperparameter configurations (`breakout_fast`, `breakout_ultrafast`) with organized output structure in config-specific subdirectories.

**2025-11-22**: Implemented max step parameter for overall training control. Adjusted graphing functionality and added separate file for viewing agents in action. Separated from `envpool` environment to avoid render mode limitations.

**2025-11-23**: Discontinued `envpool` environment due to compatibility issues and limitations (no render mode). Implemented `agents1.3.py` that recreates parameters used in original DQN training. Identified epsilon decay issues requiring code review for `breakout3.1` configuration.

**2025-11-25**: Created `agents1.4.py` to fix critical training issues found in version 1.3:

- Fixed training loop placement (was running once per episode, now runs every `update_freq` steps)
- Added reward clipping (-1, 1) to stabilize training
- Added gradient clipping (norm 10.0) to prevent exploding gradients
- Verified fixes align with standard DQN implementations

**2025-11-26**: Implemented `natural_wrapper` for background replacement to test DQN robustness with Gaussian noise and natural videos. Updated `agents2.0.py` to support these new configurations for evaluating agent performance under visual perturbations.
