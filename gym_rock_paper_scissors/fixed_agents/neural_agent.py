from random import seed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from itertools import count
from collections import namedtuple


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class NeuralAgent():
    '''
    Representes a fixed agent which uses a mixed strategy.
    A mixed strategy is represented as a support vector (probability) distribution over
    the set of all possible actions.
    '''

    def __init__(self, support_vector = [0.5,0.5,0], name = 'Amelia', seed = None):
        '''
        Checks that the support vector is a valid probability distribution
        :param support_vector: support vector for all three possible pure strategies [ROCK, PAPER, SCISSORS]
        :throws ValueError: If support vector is not a valid probability distribution
        '''
        if any(map(lambda support: support < 0, support_vector)):
            raise ValueError('Every support in the support vector should be a positive number. Given supports: {}'.format(support_vector))
        if sum(support_vector) != 1.0:
            raise ValueError('The sum of all supports in the support_vector should sum up to 1. Given supports: {}'.format(support_vector))
        self.support_vector = support_vector
        self.name = name
        self.model = Policy()
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-2)
        self.eps = np.finfo(np.float32).eps.item()
        if seed:
            torch.manual_seed(seed)

        

    def take_action(self, state):
        '''
        Samples an action based on the probabilities presented by the agent's support vector
        :param state: Ignored for fixed agents
        '''
        if type(state) == list:
            state = np.array(state).flatten()
        state = torch.from_numpy(state).float()
        probs, state_value = self.model(state)

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)

        # and sample an action using the distribution
        action = m.sample()

        # save to action buffer
        self.model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        # the action to take (left or right)
        
        return action.item()

    def handle_experience(self, reward, *args):
        self.model.rewards.append(reward)
    
    def learn_strategy(self, verbose = False):
        R = 0
        gamma = 1.0
        saved_actions = self.model.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.model.rewards[::-1]:
            # calculate the discounted value
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss 
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

            # Now let the loss be the true return and see how it does
            # value_losses.append(torch.tensor([R]))

        # if verbose:
            # print(self.name)
            # print(value_losses)
            # print(R)
            # print(self.model.rewards)
        # reset gradients
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # perform backprop
        loss.backward()
        self.optimizer.step()

        # reset rewards and action buffer
        del self.model.rewards[:]
        del self.model.saved_actions[:]
    
    def peek_model(self, state):
        if type(state) == list:
            state = np.array(state).flatten()
        state = torch.from_numpy(state).float()
        probs, state_value = self.model(state)
        return probs, state_value

    def clone(self, training):
        return NeuralAgent(support_vector=self.support_vector, name=self.name)




class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, state_size = 10*3, action_size = 3):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(state_size, 128)
        
        # actor's layer
        self.action_head = nn.Linear(128, action_size)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return action_prob, state_values



