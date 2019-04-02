import torch
import random
from collections import namedtuple, deque
import numpy as np

class SumTree(object):
    """ Data structure for maintaining prioritized experience replay
    
    Attributes
    ----------
        capacity (int) - capacity of data: number of leaf nodes of the tree
        tree (numpy) - array representing complete binary sumtree of priorities
        data (numpy) - contains the experiences (so the size of data is capacity)
        data_pointer (int) - current index of data array
    """
    def __init__(self, capacity):
        self.capacity = capacity 
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
    
    def add(self, priority, data):
        """ Add data to the structure

        Parameters
        ----------
            priority (float) - priority value of the new data
            data (tuple) - experience oject (generally state, action, next_state, reward, done)
        """

        self.data[self.data_pointer] = data # add new data to data array at data_pointer 

        tree_index = self.data_pointer + self.capacity - 1 # equivalent index of data_pointer on the leafs of sum_tree
        self.update (tree_index, priority) # update priorities of the tree

        self.data_pointer += 1 # increase data_pointer index
        
        if self.data_pointer >= self.capacity: # wrap around to overrite if at capacity
            self.data_pointer = 0
            
    def update(self, tree_index, priority):
        """ Update priorities on the tree

        Parameters
        ----------
            tree_index (int) - index representing leaf on the sum_tree
            priority (float) - priority score of the experience
        """

        change = priority - self.tree[tree_index] # difference between new priority and previous priority
        self.tree[tree_index] = priority # set the new priority

        while tree_index != 0: # sequentially update parents by adding the difference
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
    
    def get_leaf(self, v):
        """ Obtain leaf given specified priority value

        Parameters
        ----------
            v (float) - priority value

        Returns
        -------
            leaf_index (int) - index of returned value 
            self.tree[leaf_index] (float) - priority at the index
            self.data[data_index] (int) - experience at the index 
        """

        parent_index = 0
        
        while True: 
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            
            else: # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
            
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    @property
    def total_priority(self):
        return self.tree[0] # Returns the root node


class PriorityReplayBuffer(object):  
    """ Priority replay buffer leveraging sumtree data structure
    
    Attributes
    ----------
        tree (subtree) - data structure to manage priorities
        buffer_size (int) - capacity of experiences
        device (torch.device) - device to use for computations (ie cuda,cpu)
        conv (bool) - flag for usage of convolutional networks (ie image states)
        action_size (discrete or numpy) - shape of action applied to environment
        batch_size (int) - size of batch to extract during learning
        experience (tuple) - tuple format for experience
        PER_e (float) - Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
        PER_a (float) - Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
        PER_b (float) - importance-sampling, from initial value increasing to 1
        PER_b_increment_per_sampling (float) - change in PER_b each sample
        absolute_error_upper (float) = 1. - clipped absolute error

    """

    def __init__(self, action_size, buffer_size, batch_size, seed, conv, device):
        self.tree = SumTree(buffer_size)
        self.buffer_size = buffer_size
        self.device = device
        self.conv = conv
        self.action_size = action_size
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        np.random.seed(seed)

        self.PER_e = 0.01
        self.PER_a = 0.6
        self.PER_b = 0.4
        self.PER_b_increment_per_sampling = 0.001
        self.absolute_error_upper = 1. 
        

    def store(self, state, action, reward, next_state, done):
        """ Each new experience have a score of max_prority (so it is chosed by the agent to improve)

        Parameters
        ----------
            state (discrete or numpy) - state given by the environment
            action (discrete) - action applied to the environment
            reward (float) - reward from environment for taking the action at the current state
            next_state (discrete or numpy) - state given by the environment as a result of the action
            done (bool) - whether the environment session has completed
        """
        max_priority = np.max(self.tree.tree[-self.tree.capacity:]) # max value amogst the leaves

        e = self.experience(state, action, reward, next_state, done) # create experience structure

        if max_priority == 0: # avoid having 0 priorities
            max_priority = self.absolute_error_upper 
        
        self.tree.add(max_priority, e)   # add priority and experience to the sumtree

        

    def sample(self):
        """ Extract a sample with priority considerations

        Returns
        -------
            batch_size number of experiences

        Notes
        -----
            First, to sample a minibatch of k size, the range [0, priority_total] is divided into k ranges.
            Then a value is uniformly sampled from each range
            We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
            Then, we calculate IS weights for each minibatch element

        """

        # Initialize empty experiece, id, and weights lists
        experiences = []
        b_idx, b_ISWeights = np.empty((self.batch_size,), dtype=np.int32), np.empty((self.batch_size, 1), dtype=np.float32)
        
        priority_segment = self.tree.total_priority / self.batch_size # Calculate the priority segment
    
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling]) # Increase the PER_b each time we sample a new minibatch
        
        # Calculating the max_weight (used to normalize IS weights)
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority # minimim priority divided by total priority 
        max_weight = (p_min * self.batch_size) ** (-self.PER_b) # max_weight (notice -self.PER_b)
        
        for i in range(self.batch_size): # collect experiences
            a, b = priority_segment * i, priority_segment * (i + 1) # bounds of priority segment
            value = np.random.uniform(a, b) # random selection within segment
            
            index, priority, data = self.tree.get_leaf(value) # find index, priority, and experience given the value
            
            sampling_probabilities = priority / self.tree.total_priority # #P(i) = priority/sum(priority)
            
            b_ISWeights[i, 0] = np.power(self.batch_size * sampling_probabilities, -self.PER_b)/ max_weight #  IS = (1/N * 1/P(i))**b /max _weight == (N*P(i))**-b/max_wieght
                                   
            b_idx[i]= index # add index
            
            experiences.append(data) # add experience to batch

        # Process experiences into vectors
        if self.conv == True:
            states = torch.from_numpy(np.vstack([np.expand_dims(e.state,0) for e in experiences if e is not None])).float().to(self.device)
            next_states = torch.from_numpy(np.vstack([np.expand_dims(e.next_state,0) for e in experiences if e is not None])).float().to(self.device)
        else:
            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)    
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        b_ISWeights = torch.from_numpy(b_ISWeights).to(self.device)
        return (states, actions, rewards, next_states, dones, b_idx, b_ISWeights)
    

    def batch_update(self, tree_idx, abs_errors):
        """ After updating the model with a batch of experiences, update the priorities of those experiences using the absolute errors

        Parameters
        ----------
            tree_idx (numpy) - array of indices of experiences 
            abs_errors (numpy) - array of errors return from the result of a training update

        """

        abs_errors += self.PER_e  # avoid errors of 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper) # maximum of 1
        ps = np.power(clipped_errors, self.PER_a) # factor that determines random selection vs selecting max priority

        for ti, p in zip(tree_idx, ps): # update tree
            self.tree.update(ti, p)

    def __len__(self):
        return self.buffer_size

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed, conv, device):
        """ Replay memory leveraging queue data structure
    
        Attributes
        ----------
            device (torch.device) - device to use for computations (ie cuda,cpu)
            conv (bool) - flag for usage of convolutional networks (ie image states)
            action_size (discrete or numpy) - shape of action applied to environment
            batch_size (int) - size of batch to extract during learning
            experience (tuple) - tuple format for experience
            memory (deque) - queue with a max capacity of buffer_size

        """
        self.conv = conv
        self.device = device
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)
    
    def store(self, state, action, reward, next_state, done):
        """ Store experience in memory
            
            Parameters
            ----------
                state (discrete or numpy) - state given by the environment
                action (discrete) - action applied to the environment
                reward (float) - reward from environment for taking the action at the current state
                next_state (discrete or numpy) - state given by the environment as a result of the action
                done (bool) - whether the environment session has completed
        """
        e = self.experience(state, action, reward, next_state, done)
        
        self.memory.append(e)
    
    def sample(self):
        """ Randomly sample a batch of experiences from memory """

        experiences = random.sample(self.memory, k=self.batch_size)

        # Sample batch from latest experiences
        # experiences = []
        # i = -1
        # while i>-1*self.batch_size-1:
        #     experiences.append(self.memory[i])
        #     i -= 1

        # Process experiences into vectors
        if self.conv == True:
            states = torch.from_numpy(np.vstack([np.expand_dims(e.state,0) for e in experiences if e is not None])).float().to(self.device)
            next_states = torch.from_numpy(np.vstack([np.expand_dims(e.next_state,0) for e in experiences if e is not None])).float().to(self.device)
        else:
            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)