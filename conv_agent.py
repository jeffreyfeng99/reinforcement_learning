import os
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from conv_model import ConvQNetwork, ConvQNetworkDuel
from memory import ReplayBuffer, PriorityReplayBuffer
from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ConvAgent():
    """ Agent containing neural network and memory cache

    Attributes
    ----------
        BUFFER_SIZE (int) - total sample capacity of memory cache
        BATCH_SIZE (int) - number of samples to retrieve from memory
        GAMMA (float) - discount factor when calculating target Q values
        TAU (float) - interpolation factor for updating target network
        LR (float) - learning rate
        UPDATE_EVERY (int) - number of iterations between model updates
        CONV (bool) - flag for using convolutional network (if state is an image)
        DUEL (bool) - flag for using dueling dqn
        DOUBLE (bool) - flag for using double dqn
        PRIORITY (bool) - flag for using prioritized experience replay
        NAME (string) - name of current session
        seed (int) - seed for random numbers
        state_size (discrete or numpy) - shape of environment state
        action_size (discrete or numpy) - shape of action applied to environment  
        
    """

    def __init__(self, state_size, action_size, options):
        """ Initialize attributes, memory, writers, and model

        Parameters
        ----------
            state_size (discrete or numpy) - shape of environment state
            action_size (discrete or numpy) - shape of action applied to environment 
            options (Options object) - parameters initialized in run.py

        """

        # Initialize attributes
        self.BUFFER_SIZE = int(options.buffer_size)
        self.BATCH_SIZE = int(options.batch_size)         
        self.GAMMA = float(options.gamma)
        self.TAU = float(options.tau)
        self.LR = float(options.lr)
        self.UPDATE_EVERY = int(options.update_every)   
        self.CONV = bool(options.conv)
        self.DUEL = bool(options.duel)
        self.DOUBLE = bool(options.double)
        self.PRIORITY = bool(options.priority)
        self.NAME = str(options.name)
        self.seed = int(options.seed)
        self.state_size = state_size
        self.action_size = action_size

        # Seed random numbers
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Set up tensorboard writer if training
        if bool(options.test) == False:
            self.writer = SummaryWriter(os.path.join('./tensorboard/', self.NAME, 'loss'))

        # Initialize Q-Network
        if self.DUEL == False:
            self.qnetwork_local = ConvQNetwork(state_size, action_size, self.seed).to(device)
            self.qnetwork_target = ConvQNetwork(state_size, action_size, self.seed).to(device)
        else:
            self.qnetwork_local = ConvQNetworkDuel(state_size, action_size, self.seed).to(device)
            self.qnetwork_target = ConvQNetworkDuel(state_size, action_size, self.seed).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)

        # Initialize replay memory
        if self.PRIORITY == False:
            self.memory = ReplayBuffer(action_size, self.BUFFER_SIZE, self.BATCH_SIZE, self.seed, self.CONV, device)
        else:
            self.memory = PriorityReplayBuffer(action_size, self.BUFFER_SIZE, self.BATCH_SIZE, self.seed, self.CONV, device)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.iter = 0
        self.state_list = deque(maxlen = 4)
    
    def step(self, state, action, reward, next_state, done):
        """ Perform an agent step - store recent information to memory and update the model
            
        Parameters
        ----------
            state (discrete or numpy) - state given by the environment
            action (discrete) - action applied to the environment
            reward (float) - reward from environment for taking the action at the current state
            next_state (discrete or numpy) - state given by the environment as a result of the action
            done (bool) - whether the environment session has completed
        """

        # if len(self.state_list) == 4:
        #     new_state = np.array(self.state_list)
        #     temp_state = self.state_list
        #     temp_state.append(next_state)
        self.memory.store(state, action, reward, next_state, done) # Store information to memory

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY # Update current step count
        if self.t_step == 0: # if t_step falls on an UPDATE_EVERY iteration
            if len(self.memory) > self.BATCH_SIZE: # if memory has enough information
                experiences = self.memory.sample() # extract a random sample from memory
                self.learn(experiences, self.GAMMA) # update the model based on experiences from memory
               
    def act(self, state, eps=0.):
        """ Recieve an action from the current model

        Parameters
        ----------
            state (discrete or numpy) - current state given by the environment
            eps (float) - factor for epsilon-greedy action selection

        Returns
        -------
            action (discrete) - action to apply to the environment
        """

        # self.state_list.append(state)
        # if len(self.state_list) == 4:
        #     new_state = np.array(self.state_list)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device) # process state as a tensor

        self.qnetwork_local.eval() # turn on eval mode 
        with torch.no_grad():
            action_values = self.qnetwork_local(state) # get the action given the current state
        self.qnetwork_local.train() # turn back to train mode

        # Epsilon-greedy action selection (if eps is higher, there is less exploration)
        if np.random.rand(1) < eps:
            return np.random.randint(0,self.action_size) # return random action (exploration)
        else:
            return np.argmax(action_values.cpu().data.numpy()) # return best decision (exploitation)

       
    def learn(self, experiences, gamma):
        """ Update the model based on a batch of experiences from memory

        Paramters
        ---------
            experiences (list<sample from memory>) - list of items extracted from memory replay
            gamma (float) - discount factor
            
        """

        # Split experiences into relevant vectors
        if self.PRIORITY == False:
            states, actions, rewards, next_states, dones = experiences
        else:
            states, actions, rewards, next_states, dones, b_idx, b_ISWeights = experiences

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1,actions)

        if self.DOUBLE == False:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1) # Get max predicted Q values (for next states) from target model (BATCH_SIZE x 1)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)) # Compute Q targets for current states (BATCH_SIZE x 1)
        else:
            Q_next_state =  self.qnetwork_local(next_states).detach() # Get max predicted Q values (for next states) from local model (BATCH_SIZE x action_size) 
            Q_targets_next = self.qnetwork_target(next_states).detach() # Get max predicted Q values (for next states) from target model (BATCH_SIZE x action_size) 
            actions_next = Q_next_state.argmax(1).unsqueeze(1) # Choose best actions given by the local model (BATCH_SIZE x 1)
            Q_targets = Q_targets_next.gather(1, actions_next) # Select the Q values from the target model at the indices given by the local model (BATCH_SIZE x 1)
            Q_targets = rewards + (gamma * Q_targets) # Compute Q targets (BATCH_SIZE x 1)

            # if the experience results in a finished environment session, set Q_target to reward
            for i in range(len(dones)):
                if dones[i] == 1:
                    Q_targets[i] == rewards[i]

        # Calculate losses
        if self.PRIORITY == False:
            loss = F.smooth_l1_loss(Q_expected, Q_targets) 
        else:
            abs_error = torch.abs(Q_targets - Q_expected).detach().cpu().numpy() # absolute difference (used to update prioritized experience replay)
            loss = torch.mul(b_ISWeights, F.smooth_l1_loss(Q_expected, Q_targets)).mean() # loss calculater (multiplied by weights given by experience replay)
            self.memory.batch_update(b_idx, abs_error) # update experience replay

        # Backpropogation and descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.TAU) # Update target model with local model
        self.writer.add_scalar('Loss/' + self.NAME + "_loss", loss, self.iter) # Update summary writer
        self.iter += 1  


    def soft_update(self, local_model, target_model, tau):
        """ θ_target = τ*θ_local + (1 - τ)*θ_target

        Parameters
        ----------
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation factor
        """

        # Copy weights
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()): 
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def fillEmptyMemory(self, env):
        """ If using prioritized experience replay, fill memory with random experiences

        Parameters
        -----------
            env (environment) - enviroment that takes in actions and produces states and results
        """

        for i in range(self.BUFFER_SIZE):
            if i == 0:
                state = env.reset()
                state = preprocess(state)
            
            action = random.choice(np.arange(self.action_size)) # Choose random action
            
            next_state, reward, done, _ = env.step(action) # Get next_state, reward, and done for that action at the given state
            next_state = preprocess(next_state) # preprocess state

            if done:
                self.memory.store(state, action, reward, next_state, done) # store to memory
                next_state = np.zeros(state.shape) # reset next_state
                state = env.reset() # reset environment
                state = preprocess(state) # preprocess state
            else:
                self.memory.store(state, action, reward, next_state, done) # store to memory
                state = next_state # update state with next_state


    def learningRate(self):
        """ If using an optimizer scheduler, take a step """
        self.scheduler.step()


def downsample(img):
    img = Image.fromarray(img)
    img = img.crop((0,25,160,210))
    img = img.resize((84,84))
    img = np.asarray(img)
    return img


def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)


def preprocess(img):
    img = to_grayscale(downsample(img))
    return img