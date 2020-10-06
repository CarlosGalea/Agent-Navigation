import numpy as np
import random
from collections import namedtuple, deque

from model import Navigation_Net

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)        #Replay Buffer
BATCH_SIZE = 32               #Minibatch Size
GAMMA = 0.99                  #Discount Factor
TAU = 1e-3                    #Soft update of target parameters
LR = 5e-4                     #Learning Rate
UPDATE_EVERY = 4              #No. of actions between each pair of successive updates

#Check wheter user is using GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    
    def __init__(self, state_size, action_size, seed):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        #Loading Model
        self.net_local = Navigation_Net(state_size, action_size, seed).to(device)
        self.net_target = Navigation_Net(state_size, action_size, seed).to(device)
                
        #Optimizer
        self.optimizer = optim.Adam(self.net_local.parameters(), lr=LR)
        
        #Setting Replay Memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        #Setting Time Step
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        
        #Save learning experience in memory
        self.memory.add(state, action, reward, next_state, done)
        
        #Increment time step and check if enough number of actions were taken
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        
        if self.t_step == 0:
            #If enough samples in memory are available, get random memory and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
                
    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.net_local.eval()
        
        with torch.no_grad():
            action_value = self.net_local(state)
            
        self.net_local.train()
        
        #Return epsilon-greedy action for given state 
        if random.random() > eps:
            return np.argmax(action_value.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, done = experiences
        
        #Get max predicted Q values (for next states) from target model
        Q_targets_next = self.net_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        #Compute Q targets for current state
        Q_targets = rewards + (gamma * Q_targets_next * (1-done))
        
        #Get expected Q values from local model
        Q_values = self.net_local(states).gather(1, actions)
        
        #Calculate Loss
        loss = F.mse_loss(Q_values, Q_targets)
        
        #Optimize Loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        #Updating target network
        self.soft_update(self.net_local, self.net_target, TAU)
        
    def soft_update(self, local_model, target_model, tau):
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
        
        
        