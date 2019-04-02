import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        """ Model initialization 

        Parameters
        ----------
            state_size (discrete or numpy) - shape of environment state
            action_size (discrete or numpy) - shape of action applied to environment  
            seed (int) - seed for random numbers

        """

        super(ConvQNetwork, self).__init__()
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)

        self.conv1 = nn.Conv2d(1, 32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)

        self.fc1 = nn.Linear(7*7*64, 256)
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, state):
        """ Forward pass (normalize here to prevent floats being stored in memory """
        state = state/255.0
        state = state.unsqueeze(1)

        x =  F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
    
        return x

class ConvQNetworkDuel(nn.Module):
    def __init__(self, state_size, action_size, seed):
        """ Model initialization 

        Parameters
        ----------
            state_size (discrete or numpy) - shape of environment state
            action_size (discrete or numpy) - shape of action applied to environment  
            seed (int) - seed for random numbers

        """

        super(ConvQNetworkDuel, self).__init__()
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)

        self.conv1 = nn.Conv2d(1, 32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)

        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, action_size)

        self.fc3 = nn.Linear(7*7*64, 512)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, state):
        """ Forward pass (normalize here to prevent floats being stored in memory """
        state = state/255.0
        state = state.unsqueeze(1)

        x =  F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        
        advantage = F.relu(self.fc1(x))
        value = F.relu(self.fc3(x))

        advantage = self.fc2(advantage)
        value = self.fc4(value)

        mean = torch.mean(advantage, 1).unsqueeze(1)
        aggregate = torch.add(value, torch.add(advantage,torch.neg(mean)))
        return aggregate