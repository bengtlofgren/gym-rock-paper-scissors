import numpy as np


class CounterAgent():
    '''
    Representes a fixed agent which uses a mixed strategy.
    A mixed strategy is represented as a support vector (probability) distribution over
    the set of all possible actions.
    '''

    def __init__(self, name):
        '''
        Checks that the support vector is a valid probability distribution
        :param support_vector: support vector for all three possible pure strategies [ROCK, PAPER, SCISSORS]
        :throws ValueError: If support vector is not a valid probability distribution
        '''
        self.name = name
        self.history = 0

    def take_action(self, state):
        '''
        Samples an action based on the probabilities presented by the agent's support vector
        :param state: Ignored for fixed agents
        '''
        return (self.history + 1) % 3

    def handle_experience(self, state, env):
        decoded_observations = env.decode_state(state)
        i = 0
        prev_op_action = decoded_observations[i]
        while prev_op_action is None:
            prev_op_action = decoded_observations[i+1]
            i+=1
        self.history = prev_op_action[1].value

    def clone(self, training):
        return CounterAgent(self.name)