import random
from collections import deque
import argparse
import os

import torch
import gym
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter

from agent import Agent
from conv_agent import ConvAgent, preprocess
from options import Options
from custom_env import tictactoe

def train_dqn(options, agent, env):
    """ Script to train the reinforcement learning model

    Parameters
    ----------
        options: Options objects containing the attributes and hyperparameters of the training session
        agent: Agent object containing the model and memory
        env: Environment object that based on OpenAI's gym environments

    """

    writer = SummaryWriter(os.path.join('./tensorboard/', options.name, 'reward')) # Create a tensorboard writer for rewards

    scores_window = deque(maxlen=100) # Keeps track of 100 most recent awards
    eps = options.eps_start # initialize starting epsilon (for epsilon-greedy selection of action)

    for i_episode in trange(1, options.episodes+1): # training loop
        state = env.reset() # reset the environment
        score = 0 # initialize score for the current episode

        if options.conv == True: # preprocess if state is an image
            state = preprocess(state)

        for t in range(options.steps): # number of steps per episode
            action = agent.act(state, eps) # get an action from the agent based on the current state
            next_state, reward, done, _ = env.step(action) # take a step in the environment based on the action
            score += reward # score is an accumulation of the rewards for each step

            if options.conv == True: # preprocess if state is an image
                next_state = preprocess(next_state)

            agent.step(state, action, reward, next_state, done) # send current snapshot of values to the agent

            if done: # if episode is finished 
                eps = (options.eps_start - options.eps_end)*(max((10000-i_episode)/float(10000),0)) + options.eps_end#max(options.eps_end, eps - options.eps_decay) # eps = max(options.eps_end, eps - options.eps_decay)
                # agent.learningRate()
                break 
            else:
                state = next_state
            # env.render() 

        writer.add_scalar('Reward/' + options.name, score, i_episode) # Add the total score for the previous episode to the graph
        scores_window.append(score) # Add the total score for the previous episode to the score window     

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="") # Average of 100 most recent scores

        if i_episode % 100 == 0: 
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window))) # Average of 100 most recent scores
        
        if i_episode % 1000 == 0: # Save the model state
            torch.save(agent.qnetwork_local.state_dict(), os.path.join('./checkpoints',options.name, str(i_episode) + '.pth'))
            torch.save(agent.optimizer.state_dict(), os.path.join('./checkpoints',options.name, str(i_episode) + '_optimizer.pth'))
        
        if i_episode == options.episodes or  np.mean(scores_window) >=options.success_score: # Save on success or on finish
            print('\nSession finished in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), os.path.join('./checkpoints',options.name, 'last.pth'))
            torch.save(agent.optimizer.state_dict(), os.path.join('./checkpoints',options.name, 'last_optimizer.pth'))
            break
    return 


def test_dqn(options, agent, env):
    """ Script to test the model

    Parameters
    ----------
        options: Options objects containing the attributes and hyperparameters of the training session
        agent: Agent object containing the model and memory
        env: Environment object that based on OpenAI's gym environments

    TODO
    ----
        Implement a validation session for tuning of hyperparameters
    """

    agent.qnetwork_local.load_state_dict(torch.load(os.path.join('./checkpoints',options.name, 'last.pth'))) # load the desired weights

    env.render()
    for i in range(10): # Number of episodes
        state = env.reset()
        for j in range(200): # Number of steps per episode
            if options.conv == True: # Preprocess if state is an image
                state = preprocess(state)
            action = agent.act(state) # get an action from the agent based onthe current state
            env.render()
            state, reward, done, _ = env.step(action) # take a step in the environment based on the action
            print(state, reward, done, _ )
            if done:
                break 
    env.close()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help="name of file")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--conv', action='store_true')
    parser.add_argument('--duel', action ='store_true')
    parser.add_argument('--double', action = 'store_true')
    parser.add_argument('--priority', action = 'store_true')
    args = parser.parse_args()

    # Options object to store agent parameters and model hyperparameters
    options = Options (
        name = args.name,
        buffer_size = int(1e5), 
        batch_size = 32,        
        gamma = 0.99,            
        tau = 1e-3,              
        lr = 0.0001,               
        update_every = 4,     
        conv = args.conv,
        duel = args.duel,
        double = args.double,
        priority = args.priority,
        test = args.test,
        env = 'TicTacToe',
        seed = 3,
        episodes = 200000,
        steps = 20,
        eps_start = 0.6,
        eps_end = 0.1, 
        eps_decay = (0.6-0.1)/10000,
        success_score = 65.0
        )

    # Document options
    print(options)
    if options.test == False: 
        options.save_as_txt()

    # Initialize the environment, state_shape, and action_size
    env = tictactoe.TicTacToeEnv()
    state_shape = env.observation_space#.shape[0] 
    action_size =  env.action_space#.n 
    env.seed(options.seed)
    state = env.reset()

    # Create different agents depending on the format of the state (discrete list vs image)
    if options.conv == False:
        agent = Agent(state_shape, action_size, options)
    else: 
        agent = ConvAgent(state_shape, action_size, options)

    # For priority experience replay, memory must be randomly initialized
    if options.test == False and options.priority == True:
        agent.fillEmptyMemory(env)

    # Run desired session
    if (options.test == False):
    	train_dqn(options, agent, env)
    else:
    	test_dqn(options, agent, env)


