# DRL Agent Navigation

## Agent Implementation

### Deep Q-Networks

This project makes use of *Value Based* method called a [Deep Q-Network](https://towardsdatascience.com/dqn-part-1-vanilla-deep-q-networks-6eb4a00febfb#:~:text=Deep%20Q%20learning%2C%20as%20published,of%20low%2Ddimensional%20feature%20vectors.)

Two different approaches of Deep Q-Learning were combined for this project:
- A Reinforcement Learning method called [Q Learning](https://en.wikipedia.org/wiki/Q-learning)
- A Deep Neural Network to learn from backpropagation and containing fully connected layers.

This implementation also includes 2 major training improvements to:
- Experience Replay 
- Fixed Q Targets

#### Experience Replay
> When the agent interacts with the environment, the sequence of experience tuples can be highly correlated. 
The naive Q-learning algorithm that learns from each of these experience tuples in sequential order runs the risk of getting 
swayed by the effects of this correlation. By instead keeping track of a replay buffer and using experience replay to sample from the buffer at random, 
we can prevent action values from oscillating or diverging catastrophically.

#### Fixed Q Targets
> We saw in the Deep Q Learning article that, when we want to calculate the TD error (aka the loss), we calculate the difference between the TD target (Q_target) 
and the current Q value (estimation of Q).
But we don’t have any idea of the real TD target. We need to estimate it. Using the Bellman equation, 
we saw that the TD target is just the reward of taking that action at that state plus the discounted 
highest Q value for the next state.
However, the problem is that we using the same parameters (weights) for estimating the target and 
the Q value. As a consequence, there is a big correlation between the TD target and the parameters 
(w) we are changing.
Therefore, it means that at every step of training, our Q values shift but also the target value shifts. 
So, we’re getting closer to our target but the target is also moving. It’s like chasing a moving target! 
This lead to a big oscillation in training.

### Code Implementation

The code consists of: 

- model.py: This python file includes the code for the Deep Neural Network model used by the agent to learn through forward and backpropogation methods.
This neural network model consists of 3 fully connected layers, along with 2 ReLU activation functions on the first 2 layers. The model accepts the environment's 
state dimensions as input and its output size depends on the agent's action size.
- agent.py: The python file defines both the DQN Agent and DQN Replay Buffer.
  - The DQN agent class provides several methods :
    - constructor : 
      - Initialize the memory buffer (*Replay Buffer*)
      - Initialize 2 instance of the Neural Network : the *target* network and the *local* network
    - step() : 
      - Allows to store a step taken by the agent (state, action, reward, next_state, done) in the Replay Buffer/Memory
      - Every 4 steps (and if their are enough samples available in the Replay Buffer), update the *target* network weights with the current weight values from the *local* network (That's part of the Fixed Q Targets technique)
    - act() which returns actions for the given state as per current policy (Note : The action selection use an Epsilon-greedy selection so that to balance between *exploration* and *exploitation* for the Q Learning)
    - learn() which update the Neural Network value parameters using given batch of experiences from the Replay Buffer. 
    - soft_update() is called by learn() to softly updates the value from the *target* Neural Network from the *local* network weights (That's part of the Fixed Q Targets technique)
  - The ReplayBuffer class implements a fixed-size buffer to store experience tuples  (state, action, reward, next_state, done) 
    - add() allows to add an experience step to the memory
    - sample() allows to randomly sample a batch of experience steps for the learning  
- DRL_BananaAgent.ipynb: This Jupyter Lab Notebook imports the necessary packages, examine that environment's state and action spaces, train the agent and plot 
the obtained results.

### DQN Parameters

Parameters defined in the 'agent.py' file:

![Parameters](images/parameters)

Neural Network Architecture: 

![Model Architecture](images/model_architecture)
