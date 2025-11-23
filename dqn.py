import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, obs_shape, action_dim, enable_dueling_dqn=True):
        super(DQN, self).__init__()
        
        self.enable_dueling_dqn = enable_dueling_dqn
        
        # convolutional layers (DeepMind architecture)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        # fully connected layers (7x7 after 3 conv layers)
        self.flatten_size = 64*7*7  # 3136
        
        if self.enable_dueling_dqn:
            # value stream
            self.fc_value = nn.Linear(self.flatten_size, 512)
            self.value = nn.Linear(512, 1)
            
            # advantage stream
            self.fc_advantage = nn.Linear(self.flatten_size, 512)
            self.advantage = nn.Linear(512, action_dim)
            
        else:
            # standard DQN
            self.fc1 = nn.Linear(self.flatten_size, 512)
            self.fc2 = nn.Linear(512, action_dim)
    
    def forward(self, x):
        # normalize pixel values from [0, 255] to [0, 1]
        x=x/255.0
        
        # convolutional feature extraction (3 layers as per DeepMind)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # flatten: (batch, 64, 7, 7) -> (batch, 64*7*7)
        x = x.view(x.size(0), -1) # flatten
        
        if self.enable_dueling_dqn:
            # value calc
            v = F.relu(self.fc_value(x))
            V = self.value(v)
            
            # advantage calc
            a = F.relu(self.fc_advantage(x))
            A = self.advantage(a)
            
            # calculate Q 
            Q = V + A - torch.mean(A, dim=1, keepdim=True)
        else:
            x = F.relu(self.fc1(x))
            Q = self.fc2(x)

        return Q
    
if __name__ == "__main__":
    state_dim = 12
    action_dim = 2
    net = DQN(state_dim, action_dim)
    state = torch.randn((10, state_dim))
    output = net(state)
    print(output) 
