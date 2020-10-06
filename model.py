import torch
import torch.nn as nn
import torch.nn.functional as F

class Navigation_Net(nn.Module):
    
    def __init__(self, state_size, action_size, seed):
        super (Navigation_Net, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, state_size * 2)
        self.fc2 = nn.Linear(state_size * 2, state_size // 2)
        self.fc3 = nn.Linear(state_size // 2, action_size)
        
    def forward(self, x):
        fc1_out = F.relu(self.fc1(x))
        fc2_out = F.relu(self.fc2(fc1_out))
        
        return self.fc3(fc2_out)
        
        