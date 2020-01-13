import numpy as np
import random
import matplotlib.pyplot as plt

import argparse
import os
import sys
import string

from time import time
from datetime import datetime

EPSILON = 0.1
MAX_NUM_TASKS = 2000

NOW = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SEED = 197710

# #############################################################################
#
# Parser
#
# #############################################################################


def get_arguments():
    def _str_to_bool(s):
        """Convert string to boolean (in argparse context)"""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='Breaking ciphered texts with HMM')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Seed for the random number generator.')
    parser.add_argument('-k', '--actions', type=int, default=10,
                        help='Number of arms on the bandit. Default: k=10.')
    parser.add_argument('-n', '--trials', type=int, default=MAX_NUM_TASKS,
                        help='Number of time steps or number of times one of the arms '
                        'of the bandit will be pulled. Default: ' + str(MAX_NUM_TASKS))

    return parser.parse_args()

# #############################################################################
#
# Plotting
#
# #############################################################################

def plot_reward_distr(data):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4), sharey=True)

    ax.set_ylabel('Reward distribution')
    ax.set_title('The {}-armed testbed'.format(len(data)))

    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(data) + 1))
    ax.set_xlim(0.25, len(data) + 0.75)
    ax.set_xlabel('Action')
    
    violin = ax.violinplot(data,
                           showmeans=True,
                           widths=0.25,
                           showmedians=False,
                           showextrema=False)
    
    plt.axhline(y=0, color='r', linestyle='--')
    plt.subplots_adjust(bottom=0.15, wspace=0.05)
    plt.show()

# #############################################################################
#
# Main
#
# #############################################################################

def softmax(x):

    # subtract max for numerical stability
    # (does not change result because of identity softmax(x) = softmax(x + c))
    z = x - max(x)

    return np.exp(z) / np.sum(np.exp(z)) 


def random_argmax(vector):
    '''Helper function to select argmax at random... not just first one.'''

    index = np.random.choice(np.where(vector == vector.max())[0])
    
    return index
    

class Bandit():

    def __init__(self, k=10, n=2000, seed=SEED):
        
        np.random.seed(seed)

        self.k = k
        self.n = n
        
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        
        self.regret = 0
        self.reward = 0

        print('Initializing {}-armed bandit...\n\nThe true values q_*(a) for each '
            'action a=0, 1,..., {} were selected according to a normal '
            'distribution with mean zero and unit variance and then the '
            'actual rewards were selected according to a mean q_*(a) unit '
            'variance normal distribution.'.format(k, k-1))

        # defines the true value q_star for each action a=0, 1, ..., k
        self.q_star = np.random.randn(k)

        # defines the rewards distributions for each action a=0, 1, ..., k
        # according to normal densities mean q_star(a) and variance 1
        #Q = np.random.normal(q_star, 1)

        data = [sorted(np.random.normal(action, 1, 10000)) for action in self.q_star]
        plot_reward_distr(data)

    def get_reward(self, action):
        '''Action produces a reward from a normal distribution with mean
        q_*(action) and variance 1'''
        R = np.random.normal(self.q_star(action), 1)
        self.N[action] += 1
        self.Q[action] = self.Q[action] + (R - self.Q[action]) / self.N[action]

        return R

    def get_regret(self, action):
        self.regret = self.regret + (max(q_star) - q_star[action])
        
        return self.regret

    def boltzmann(self):
        pass

    def ucb(self):
        pass

    def thompson(self, observation):
        """Picks action according to Thompson sampling with Beta posterior for action selection."""
        sampled_means = self._get_posterior_sample()
        action = random_argmax(sampled_means)
        return action

    def run_onetrial(self):
        pass

    def run(self):
        pass

    def _get_posterior_sample(self):
        self.prior_success = np.array([a0 for arm in range(self.k)])
        self.prior_failure = np.array([b0 for arm in range(self.k)])

        return np.random.beta(self.prior_success, self.prior_failure)
 

def main():

    # parses command line arguments
    args = get_arguments()
    env = Bandit(10)



if __name__ == '__main__':
    main()
