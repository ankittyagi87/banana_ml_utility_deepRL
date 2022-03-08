import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, prioritized_replay=False):
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
        self.prioritized_replay = prioritized_replay
        
        if self.prioritized_replay:
            self.priorities = np.ones((buffer_size,))
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        if self.prioritized_replay:
            probs = np.power(self.priorities[:len(self.memory)], 0.6)
            probs /= probs.sum()
            indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
            experiences = [self.memory[idx] for idx in indices]
        else: 
            experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        if self.prioritized_replay:
            weights = np.power(len(self.memory) * probs[indices], -0.4)
            weights /= weights.max()
            weights = torch.from_numpy(weights).float().to(device)
            return (states, actions, rewards, next_states, dones, indices, weights)
        else:
            return (states, actions, rewards, next_states, dones)
        
    def update_priorities(self, indices, priorities):
        for idx in indices:
            self.priorities[idx] = priorities

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)