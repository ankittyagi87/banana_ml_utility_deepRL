import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNetwork_dueling(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, dueling_unit=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork_dueling, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        self.fc_value = nn.Linear(fc2_units, dueling_unit)
        self.fc_adv = nn.Linear(fc2_units, dueling_unit)

        self.value = nn.Linear(dueling_unit, 1)
        self.adv = nn.Linear(dueling_unit, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = F.relu(self.fc_value(x))
        adv = F.relu(self.fc_adv(x))

        value = self.value(value)
        adv = self.adv(adv)
        advAverage = torch.mean(adv, dim=1, keepdim=True)
        
        return value + adv - advAverage