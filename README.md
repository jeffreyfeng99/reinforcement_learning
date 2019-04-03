## Experimentation with reinforcement learning 

Design for a generalized framework with options for double dqn, duel dqn, and prioritized experience replays.

### Scripts
* run.py - main facilitation of training and testing (hyperparameters are set here)
* options.py - object for storage of parameters (parameters and model requirements vary between environments)
* memory.py - structures for storing experiences in a memory replay (both standard queue and prioritized experience replay)
* Agents - both are nearly identical - separated for more efficient testing and cleaner individual codes 
  * agent.py - agent that contains the network, and generates actions based on states
  * conv_agent.py - agent that supports convolutional networks (ie image states)
 
