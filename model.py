import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=32, fc2_units=32):
        """ Model initialization 

        Parameters
        ----------
            state_size (discrete or numpy) - shape of environment state
            action_size (discrete or numpy) - shape of action applied to environment  
            seed (int) - seed for random numbers

        """

        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units, bias = False)
        # self.fc2 = nn.Linear(fc1_units, fc2_units, bias = False)
        self.fc3 = nn.Linear(fc2_units, action_size, bias = False)

    def forward(self, state):
        """ Forward pass (normalize here to prevent floats being stored in memory """

        x = self.fc1(state)
        # x = self.fc2(x)
        return self.fc3(x)

class QNetworkDuel(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=32,fc2_units=32):
        """ Model initialization 

        Parameters
        ----------
            state_size (discrete or numpy) - shape of environment state
            action_size (discrete or numpy) - shape of action applied to environment  
            seed (int) - seed for random numbers

        """

        super(QNetworkDuel, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, fc1_units)

        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc1_units, fc2_units)

        self.fc4 = nn.Linear(fc2_units, action_size)
        self.value = nn.Linear(fc2_units,1)

    def forward(self, state):
        """ Forward pass (normalize here to prevent floats being stored in memory """
        x = F.relu(self.fc1(state))

        # value = F.relu(self.fc2(x))
        # advantage = F.relu(self.fc3(x))

        value = self.value(x)
        advantage = self.fc4(x)

        mean = torch.mean(advantage, 1).unsqueeze(1)
        aggregate = torch.add(value, torch.add(advantage,torch.neg(mean)))

        return aggregate
